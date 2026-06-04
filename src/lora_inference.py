"""
Local LoRA adapter inference — same role as py2many/py2cpp in the transpile stack.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTERS_DIR = PROJECT_ROOT / "trained_adapeters"

_MODEL_CACHE: dict[str, Any] = {}


def clear_model_cache() -> None:
    """Free GPU/MPS/CPU memory before loading another adapter (7B + 1.5B + 0.5B do not fit together)."""
    import gc

    import torch

    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def build_translation_prompt(python_source: str, *, harness_style: bool = False) -> str:
    """Build the user prompt for C++ generation.

    Default (harness_style=False): same as training_data / SFT.
    harness_style=True: stricter rules for pipeline compile/test harness (use for base vs LoRA eval).
    """
    if harness_style:
        return (
            "### Instruction:\n"
            "Translate this Python code to C++.\n\n"
            "Rules:\n"
            "- Output ONLY the function(s) required (no main, no tests, no cout/printf).\n"
            "- Use std::vector<...> for Python list parameters.\n"
            "- Do not write #include lines.\n"
            "- Keep the same function name and parameter count as Python.\n\n"
            "### Python:\n"
            f"{python_source.strip()}\n\n"
            "### C++:\n"
        )
    return (
        "### Instruction:\n"
        "Translate this Python code to C++.\n\n"
        "### Python:\n"
        f"{python_source.strip()}\n\n"
        "### C++:\n"
    )


def _strip_markdown_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = [line for line in value.splitlines() if not line.strip().startswith("```")]
        return "\n".join(lines).strip()
    return value


_EXPLANATION_HEADING = re.compile(r"(?im)^#{1,3}\s*Explanation\b")
_VARIANT_SEPARATOR = re.compile(r"\n={3,}\s*\n")


def _looks_like_cpp(fragment: str) -> bool:
    return bool(
        re.search(
            r"\b(int|void|double|float|bool|char|unsigned|long|auto|std::)\b|[{;]",
            fragment,
        )
    )


def _take_first_cpp_variant(text: str) -> str:
    parts = _VARIANT_SEPARATOR.split(text)
    if len(parts) <= 1:
        return text
    for part in parts:
        candidate = part.strip()
        if candidate and _looks_like_cpp(candidate):
            return candidate
    return parts[0].strip()


def _drop_trailing_non_cpp(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if lines and stripped.startswith("def "):
            break
        if lines and _EXPLANATION_HEADING.match(stripped):
            break
        if lines and stripped.startswith("### ") and not stripped.lower().startswith("### c++"):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _strip_include_lines(text: str) -> str:
    kept = [line for line in text.splitlines() if not line.strip().startswith("#include")]
    return "\n".join(kept).strip()


def extract_cpp_from_model_output(raw_text: str) -> str:
    """Keep the first plausible C++ snippet; drop chat, alternates, and trailing Python."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:cpp|c\+\+)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        text = _strip_markdown_fence(text)

    expl = _EXPLANATION_HEADING.search(text)
    if expl:
        text = text[: expl.start()]

    text = _take_first_cpp_variant(text)
    text = _drop_trailing_non_cpp(text)
    text = _strip_include_lines(text)
    return text.strip()


def _read_base_model_name(model_dir: Path) -> str | None:
    config_path = model_dir / "adapter_config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_name = str(config.get("base_model_name_or_path", "")).strip()
    return base_name or None


def discover_lora_backends(adapters_dir: Path | None = None) -> dict[str, Path]:
    root = (adapters_dir or DEFAULT_ADAPTERS_DIR).resolve()
    if not root.is_dir():
        return {}
    backends: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "adapter_config.json").is_file():
            continue
        if not (child / "adapter_model.safetensors").is_file() and not (child / "adapter_model.bin").is_file():
            continue
        backends[f"lora_{child.name}"] = child
    return backends


def discover_base_backends(adapters_dir: Path | None = None) -> dict[str, str]:
    """Map base_<slug> -> HuggingFace model id (uses adapter_config.json only; no LoRA weights)."""
    root = (adapters_dir or DEFAULT_ADAPTERS_DIR).resolve()
    if not root.is_dir():
        return {}
    backends: dict[str, str] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        base_name = _read_base_model_name(child)
        if base_name:
            backends[f"base_{child.name}"] = base_name
    return backends


def adapter_path_for_backend(backend: str, adapters_dir: Path | None = None) -> Path | None:
    if not backend.startswith("lora_"):
        return None
    return discover_lora_backends(adapters_dir).get(backend)


def _model_dtype_and_device() -> tuple[Any, bool, bool]:
    import torch

    use_cuda = torch.cuda.is_available()
    use_mps = not use_cuda and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    if use_cuda and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif use_cuda or use_mps:
        dtype = torch.float16
    else:
        dtype = torch.float32
    return dtype, use_cuda, use_mps


def _load_base_model(hf_model_id: str) -> tuple[Any, Any]:
    key = f"base:{hf_model_id}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype, use_cuda, use_mps = _model_dtype_and_device()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": dtype}
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(hf_model_id, **model_kwargs)
    if use_mps:
        model = model.to("mps")
    model.eval()

    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _load_model(adapter_dir: Path) -> tuple[Any, Any]:
    key = str(adapter_dir.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_name = _read_base_model_name(adapter_dir)
    if not base_name:
        raise ValueError(f"Missing base_model_name_or_path in {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype, use_cuda, use_mps = _model_dtype_and_device()

    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": dtype}
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    base = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if use_mps:
        model = model.to("mps")
    model.eval()

    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _generate_from_model(
    python_source: str,
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int,
    harness_style: bool = False,
) -> tuple[str, dict[str, Any]]:
    import torch

    prompt = build_translation_prompt(python_source, harness_style=harness_style)
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        device = model.device
    else:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    cpp_text = extract_cpp_from_model_output(raw)
    usage = {
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "completion_tokens": int(new_tokens.shape[0]),
        "total_tokens": int(inputs["input_ids"].shape[1] + new_tokens.shape[0]),
        "latency_ms": latency_ms,
    }
    return cpp_text, usage


def generate_cpp_base(
    python_source: str,
    hf_model_id: str,
    *,
    max_new_tokens: int = 2048,
    harness_style: bool = False,
) -> tuple[str, dict[str, Any]]:
    model, tokenizer = _load_base_model(hf_model_id)
    return _generate_from_model(
        python_source, model, tokenizer, max_new_tokens=max_new_tokens, harness_style=harness_style
    )


def generate_cpp(
    python_source: str,
    adapter_dir: Path,
    *,
    max_new_tokens: int = 2048,
    harness_style: bool = False,
) -> tuple[str, dict[str, Any]]:
    model, tokenizer = _load_model(adapter_dir)
    return _generate_from_model(
        python_source, model, tokenizer, max_new_tokens=max_new_tokens, harness_style=harness_style
    )

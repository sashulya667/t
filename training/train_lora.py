"""
LoRA fine-tune Qwen2.5-Coder on Python -> C++ JSONL data.

Examples:
  pip install -r requirements-training.txt

  # Mac / local smoke
  python -m training.train_lora --config training/configs/small.yaml

  # Vast.ai CUDA — one model
  python -m training.train_lora --config training/configs/vast/coder-0.5b.yaml

  # Vast — all three coders (0.5B, 1.5B, 7B)
  bash training/run_vast_all.sh
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

from training.dataset import build_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Passed through to SFTConfig only when supported by the installed trl version.
OPTIONAL_SFT_KEYS = (
    "dataloader_num_workers",
    "optim",
    "packing",
    "eval_packing",
    "pad_to_multiple_of",
)


def _optional_sft_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    valid = set(inspect.signature(SFTConfig.__init__).parameters) - {"self"}
    out = {k: cfg[k] for k in OPTIONAL_SFT_KEYS if k in cfg and k in valid}
    skipped = [k for k in OPTIONAL_SFT_KEYS if k in cfg and k not in valid]
    if skipped:
        print(f"[train_lora] Ignoring unsupported SFTConfig keys: {', '.join(skipped)}")
    return out


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_mixed_precision(cfg: dict[str, Any]) -> tuple[bool, bool]:
    if cfg.get("bf16") is True:
        return True, False
    if cfg.get("fp16") is True:
        return False, True
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return True, False
        return False, True
    return False, False


def resolve_attn_implementation(cfg: dict[str, Any]) -> str | None:
    requested = cfg.get("attn_implementation")
    if not requested:
        return None
    if requested != "flash_attention_2":
        return str(requested)
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("[train_lora] flash-attn not installed; using sdpa attention.")
        return "sdpa"
    return "flash_attention_2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-Coder")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "training/configs/small.yaml",
        help="YAML profile (e.g. training/configs/vast/coder-0.5b.yaml)",
    )
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--val-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Debug cap on steps")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    cfg = load_config(_resolve_path(cli.config))

    train_file = _resolve_path(cli.train_file or cfg["train_file"])
    val_file = _resolve_path(cli.val_file or cfg["val_file"])
    output_dir = _resolve_path(cli.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model_name"]
    use_bf16, use_fp16 = pick_mixed_precision(cfg)

    print(f"Config: {cli.config}")
    print(f"Model: {model_name}")
    print(f"Train: {train_file}")
    print(f"Val:   {val_file}")
    print(f"Out:   {output_dir}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"Device: cuda — {props.name} ({props.total_memory / 2**30:.1f} GiB)")
    elif torch.backends.mps.is_available():
        print("Device: mps (Apple GPU)")
    else:
        print("Device: cpu")

    train_ds = build_dataset(train_file)
    eval_ds = build_dataset(val_file)
    print(f"Examples — train: {len(train_ds)}, val: {len(eval_ds)}")

    model_init_kwargs: dict[str, Any] = {"trust_remote_code": True}
    attn = resolve_attn_implementation(cfg)
    if attn:
        model_init_kwargs["attn_implementation"] = attn

    if cfg.get("load_in_4bit"):
        model_init_kwargs["load_in_4bit"] = True
        model_init_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_init_kwargs["dtype"] = torch.bfloat16 if use_bf16 else torch.float16
        model_init_kwargs["device_map"] = "auto"

    peft_config = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )

    max_steps = cli.max_steps if cli.max_steps is not None else -1

    warmup_kwargs: dict[str, Any] = {}
    if "warmup_steps" in cfg:
        warmup_kwargs["warmup_steps"] = int(cfg["warmup_steps"])
    else:
        warmup_kwargs["warmup_ratio"] = float(cfg.get("warmup_ratio", 0.03))

    eval_strategy = str(cfg.get("eval_strategy", "steps"))
    save_strategy = str(cfg.get("save_strategy", eval_strategy))

    eval_save_kwargs: dict[str, Any] = {
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
    }
    if eval_strategy == "steps":
        eval_save_kwargs["eval_steps"] = int(cfg["eval_steps"])
    if save_strategy == "steps":
        eval_save_kwargs["save_steps"] = int(cfg["save_steps"])

    use_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
    pin_memory = cfg.get("dataloader_pin_memory", not use_mps)

    optional_sft = _optional_sft_kwargs(cfg)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        **warmup_kwargs,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        logging_steps=int(cfg["logging_steps"]),
        **eval_save_kwargs,
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        dataloader_pin_memory=bool(pin_memory),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
        max_steps=max_steps,
        max_length=int(cfg["max_seq_length"]),
        completion_only_loss=True,
        gradient_checkpointing=bool(cfg.get("use_gradient_checkpointing", True)),
        loss_type=str(cfg.get("loss_type", "nll")),
        model_init_kwargs=model_init_kwargs,
        **optional_sft,
    )

    trainer = SFTTrainer(
        model=model_name,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    trainer.train()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(str(final_dir))
    print(f"Done. Adapter saved to {final_dir}")


if __name__ == "__main__":
    main()

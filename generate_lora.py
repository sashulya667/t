#!/usr/bin/env python3
"""
Phase 1 only: LoRA inference on a GPU server. Writes generation/ + candidates/ for local eval.

Works from the full repo or from lora_server_bundle/ (see training/pack_lora_bundle.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        _ = kwargs
        return iterable


def _bundle_roots() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    if (here / "src" / "lora_inference.py").is_file():
        return here, here / "src"
    repo = here.parent
    return repo, repo / "src"


REPO_ROOT, SRC_DIR = _bundle_roots()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lora_inference import clear_model_cache, discover_lora_backends, generate_cpp  # noqa: E402


def _iso_now() -> str:
    return datetime.now().isoformat()


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _relativize(cpp_path: Path, run_dir: Path) -> str:
    try:
        return str(cpp_path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(cpp_path)


def _backend_folder_name(backend: str) -> str:
    return backend.removeprefix("lora_")


def _zip_directory(source_dir: Path, *, exclude_suffixes: tuple[str, ...] = (".zip",)) -> Path:
    """Create <parent>/<name>.zip containing all files under source_dir."""
    source_dir = source_dir.resolve()
    zip_path = source_dir.parent / f"{source_dir.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if any(str(file_path).endswith(suffix) for suffix in exclude_suffixes):
                continue
            arcname = file_path.relative_to(source_dir.parent)
            zf.write(file_path, arcname)
    return zip_path


def _copy_eval_metadata(variants: Path | None, run_dir: Path) -> None:
    if variants is None:
        return
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(variants.resolve(), data_dir / "metadata.jsonl")


def _run_generation(
    *,
    out_dir: Path,
    backends: list[str],
    discovered: dict[str, Path],
    tests_by_item: dict[str, dict[str, Any]],
    item_ids: list[str],
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = out_dir / "candidates"
    generation_dir = out_dir / "generation"
    generation_dir.mkdir(parents=True, exist_ok=True)
    records_path = generation_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")

    tasks: list[tuple[str, str]] = []
    for backend in backends:
        for item_id in item_ids:
            tasks.append((backend, item_id))

    last_backend: str | None = None
    for backend, item_id in tqdm(tasks, total=len(tasks), desc=f"lora-generate:{out_dir.name}", unit="item"):
        if backend != last_backend:
            clear_model_cache()
            last_backend = backend
            print(f"[lora] backend: {backend} -> {out_dir}", flush=True)

        item_row = tests_by_item[item_id]
        source_text = str(
            item_row.get("python_source_untyped") or item_row.get("python_source_original") or ""
        ).strip()
        adapter_dir = discovered[backend]
        variant = "finetuned"
        gen_record: dict[str, Any] = {
            "item_id": item_id,
            "backend": backend,
            "strategy": variant,
            "source_path": "",
            "cpp_path": "",
            "generation_ok": False,
            "failure_reason": "",
            "cxx_standard": "c++17",
            "include_dirs": [],
            "llm_usage": {
                "calls": [],
                "llm_calls_total": 0,
                "llm_prompt_tokens_total": 0,
                "llm_completion_tokens_total": 0,
                "llm_total_tokens_total": 0,
                "llm_latency_ms_total": 0,
            },
            "started_at": _iso_now(),
            "finished_at": "",
            "duration_ms": 0,
        }
        t0 = time.perf_counter()
        if not source_text:
            gen_record["failure_reason"] = "missing_source"
        else:
            try:
                cpp_text, usage = generate_cpp(source_text, adapter_dir)
                if cpp_text.strip():
                    cpp_target_dir = candidates_dir / backend / variant
                    cpp_target_dir.mkdir(parents=True, exist_ok=True)
                    cpp_target = cpp_target_dir / f"{item_id}.cpp"
                    cpp_target.write_text(cpp_text, encoding="utf-8")
                    gen_record["generation_ok"] = True
                    gen_record["cpp_path"] = _relativize(cpp_target, out_dir)
                    gen_record["llm_usage"] = {
                        "calls": [{"source": "local_lora", **usage}],
                        "llm_calls_total": 1,
                        "llm_prompt_tokens_total": int(usage.get("prompt_tokens", 0) or 0),
                        "llm_completion_tokens_total": int(usage.get("completion_tokens", 0) or 0),
                        "llm_total_tokens_total": int(usage.get("total_tokens", 0) or 0),
                        "llm_latency_ms_total": int(usage.get("latency_ms", 0) or 0),
                    }
                else:
                    gen_record["failure_reason"] = "empty_cpp_output"
            except Exception as exc:  # noqa: BLE001
                gen_record["failure_reason"] = f"lora_exception:{exc}"

        gen_record["finished_at"] = _iso_now()
        gen_record["duration_ms"] = int((time.perf_counter() - t0) * 1000)
        _write_jsonl(records_path, gen_record)

    return len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA generate-only (GPU server)")
    parser.add_argument("--tests", type=Path, required=True, help="dataset_with_tests.jsonl")
    parser.add_argument(
        "--variants",
        type=Path,
        default=None,
        help="Optional metadata.jsonl — copied into run data/ for Mac eval_local",
    )
    parser.add_argument(
        "--adapters-dir",
        type=Path,
        default=None,
        help="Adapters root (default: ./adapters in bundle, else repo trained_adapeters/)",
    )
    parser.add_argument("--adapter", action="append", default=[], help="lora_<folder> backend(s)")
    parser.add_argument("--out", type=Path, required=True, help="Run output dir (copy back to Mac)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--per-adapter",
        action="store_true",
        help="Write each adapter under --out/<adapter_name>/ (separate records + zip each)",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not create .zip archives after generation",
    )
    args = parser.parse_args()

    adapters_dir = args.adapters_dir
    if adapters_dir is None:
        bundle_adapters = REPO_ROOT / "adapters"
        adapters_dir = bundle_adapters if bundle_adapters.is_dir() else REPO_ROOT / "trained_adapeters"

    discovered = discover_lora_backends(adapters_dir)
    if not discovered:
        raise SystemExit(f"No adapters under {adapters_dir}")

    backends = list(args.adapter) if args.adapter else sorted(discovered.keys())
    for name in backends:
        if name not in discovered:
            raise SystemExit(f"Unknown {name}. Available: {', '.join(sorted(discovered))}")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tests_by_item = {str(r.get("item_id", "")): r for r in _load_jsonl(args.tests.resolve())}
    item_ids = sorted(i for i in tests_by_item if i)
    if args.limit is not None:
        item_ids = item_ids[: args.limit]

    zip_paths: list[Path] = []
    run_dirs: list[Path] = []

    if args.per_adapter and len(backends) > 1:
        total_records = 0
        for backend in backends:
            sub_out = out_dir / _backend_folder_name(backend)
            run_dirs.append(sub_out)
            _copy_eval_metadata(args.variants, sub_out)
            total_records += _run_generation(
                out_dir=sub_out,
                backends=[backend],
                discovered=discovered,
                tests_by_item=tests_by_item,
                item_ids=item_ids,
            )
            if not args.no_zip:
                zip_paths.append(_zip_directory(sub_out))
        if not args.no_zip and len(backends) > 1:
            zip_paths.append(_zip_directory(out_dir))
    else:
        run_dirs.append(out_dir)
        _copy_eval_metadata(args.variants, out_dir)
        total_records = _run_generation(
            out_dir=out_dir,
            backends=backends,
            discovered=discovered,
            tests_by_item=tests_by_item,
            item_ids=item_ids,
        )
        if not args.no_zip:
            zip_paths.append(_zip_directory(out_dir))

    for run_root in run_dirs:
        manifest = {
            "phase": "generate_only",
            "layout": "per_adapter" if args.per_adapter and len(backends) > 1 else "combined",
            "tests": str(args.tests.resolve()),
            "variants": str(args.variants.resolve()) if args.variants else None,
            "adapters_dir": str(adapters_dir.resolve()),
            "backends": backends if run_root == out_dir else [_backend_folder_name(b) for b in backends if run_root.name == _backend_folder_name(b)],
            "item_count": len(item_ids),
            "run_dir": str(run_root),
        }
        if args.per_adapter and len(backends) > 1 and run_root != out_dir:
            manifest["backends"] = [f"lora_{run_root.name}"]
        (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    parent_manifest = {
        "phase": "generate_only",
        "layout": "per_adapter" if args.per_adapter and len(backends) > 1 else "combined",
        "tests": str(args.tests.resolve()),
        "variants": str(args.variants.resolve()) if args.variants else None,
        "backends": backends,
        "item_count": len(item_ids),
        "run_dirs": [str(p) for p in run_dirs],
        "zip_archives": [str(p) for p in zip_paths],
    }
    (out_dir / "manifest.json").write_text(json.dumps(parent_manifest, indent=2), encoding="utf-8")

    print(f"Done. Output: {out_dir}")
    if args.per_adapter and len(backends) > 1:
        print("  Layout: one folder per adapter under --out/")
        for sub in run_dirs:
            print(f"    {sub.name}/  (generation/ + candidates/)")
    else:
        print("  Layout: combined (candidates/<backend>/finetuned/*.cpp)")
    if zip_paths:
        print("  Zip archives (copy these to your Mac):")
        for zp in zip_paths:
            size_mb = zp.stat().st_size / (1024 * 1024)
            print(f"    {zp}  ({size_mb:.1f} MB)")
    else:
        print("  Copy the folder(s) above to your Mac for eval_local.")


if __name__ == "__main__":
    main()

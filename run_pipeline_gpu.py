#!/usr/bin/env python3
"""
GPU server workflow: train LoRA on pipeline train split → zip adapter → infer → zip run.

Run from lora_server_bundle/ after upload:

  python run_pipeline_gpu.py --config configs/pipeline-1.5b.yaml
  python run_pipeline_gpu.py --config configs/pipeline-1.5b.yaml --infer-on full
  python run_pipeline_gpu.py --config configs/pipeline-1.5b.yaml --skip-train
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


def _bundle_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "data" / "splits" / "train.jsonl").is_file():
        return here
    repo = here.parent
    if (repo / "lora_server_bundle" / "data" / "splits" / "train.jsonl").is_file():
        return repo / "lora_server_bundle"
    return here


def _zip_directory(source_dir: Path, zip_path: Path | None = None) -> Path:
    source_dir = source_dir.resolve()
    out = zip_path or (source_dir.parent / f"{source_dir.name}.zip")
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix == ".zip":
                continue
            arcname = file_path.relative_to(source_dir.parent)
            zf.write(file_path, arcname)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_bundle_path(bundle: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else bundle / path


def _run_train(bundle: Path, config_path: Path) -> Path:
    cfg = _load_yaml(config_path)
    if not (bundle / "training" / "train_lora.py").is_file():
        raise SystemExit(f"Missing training package under {bundle / 'training'}")

    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(bundle)
    cmd = [
        sys.executable,
        "-m",
        "training.train_lora",
        "--config",
        str(config_path.resolve()),
    ]
    print("[pipeline] Training adapter...", flush=True)
    subprocess.run(cmd, cwd=bundle, env=env, check=True)

    output_dir = _resolve_bundle_path(bundle, cfg["output_dir"])
    final_dir = output_dir / "final"
    if not final_dir.is_dir():
        raise SystemExit(f"Training finished but adapter missing: {final_dir}")

    adapter_slug = str(cfg.get("adapter_slug", "adapter"))
    adapters_dir = bundle / "adapters" / adapter_slug
    if adapters_dir.exists():
        shutil.rmtree(adapters_dir)
    shutil.copytree(final_dir, adapters_dir)
    print(f"[pipeline] Adapter installed at {adapters_dir}", flush=True)
    return adapters_dir


def _zip_adapter(bundle: Path, config_path: Path, adapter_dir: Path) -> Path:
    cfg = _load_yaml(config_path)
    slug = str(cfg.get("adapter_slug", adapter_dir.name))
    zip_path = bundle / "downloads" / f"adapter_{slug}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    created = _zip_directory(adapter_dir, zip_path)
    print(f"[pipeline] Adapter zip: {created} ({created.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return created


def _run_inference(
    bundle: Path,
    config_path: Path,
    *,
    infer_on: str,
    harness_prompt: bool,
    max_new_tokens: int | None,
) -> Path:
    cfg = _load_yaml(config_path)
    slug = str(cfg.get("adapter_slug", "adapter"))
    backend = f"lora_{slug}"

    if infer_on == "test":
        tests = bundle / "data/splits/test_dataset_with_tests.jsonl"
        variants = bundle / "data/splits/test_metadata.jsonl"
        out_dir = bundle / "runs" / f"infer_test_{slug}"
    elif infer_on == "full":
        tests = bundle / "data/dataset_with_tests.jsonl"
        variants = bundle / "data/metadata.jsonl"
        out_dir = bundle / "runs" / f"infer_full_{slug}"
    else:
        raise ValueError(f"Unknown infer_on={infer_on!r}")

    if not tests.is_file():
        raise SystemExit(f"Missing tests file: {tests}")

    gen_script = bundle / "generate_lora.py"
    cmd = [
        sys.executable,
        str(gen_script),
        "--tests",
        str(tests),
        "--adapters-dir",
        str(bundle / "adapters"),
        "--adapter",
        backend,
        "--out",
        str(out_dir),
    ]
    if variants.is_file():
        cmd.extend(["--variants", str(variants)])
    if harness_prompt:
        cmd.append("--harness-prompt")
    if max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])

    print(f"[pipeline] Inference on {infer_on} ({tests.name})...", flush=True)
    subprocess.run(cmd, cwd=bundle, check=True)
    return out_dir


def _zip_inference(bundle: Path, run_dir: Path, config_path: Path, infer_on: str) -> Path:
    cfg = _load_yaml(config_path)
    slug = str(cfg.get("adapter_slug", "adapter"))
    zip_path = bundle / "downloads" / f"inference_{infer_on}_{slug}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    created = _zip_directory(run_dir, zip_path)
    print(f"[pipeline] Inference zip: {created} ({created.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Train adapter on pipeline split, infer, zip downloads")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Bundle config, e.g. configs/pipeline-1.5b.yaml",
    )
    parser.add_argument(
        "--infer-on",
        choices=("test", "full"),
        default="test",
        help="Run inference on held-out test split (default) or full benchmark",
    )
    parser.add_argument("--skip-train", action="store_true", help="Use existing adapters/<slug>/")
    parser.add_argument("--skip-infer", action="store_true", help="Train + zip adapter only")
    parser.add_argument("--harness-prompt", action="store_true", help="Stricter inference prompt for eval harness")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    bundle = _bundle_root()
    config_path = args.config if args.config.is_absolute() else bundle / args.config
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    manifest_path = bundle / "data/splits/split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"[pipeline] Split: train={manifest.get('train_items')} "
            f"val={manifest.get('val_items')} test={manifest.get('test_items')}",
            flush=True,
        )

    cfg = _load_yaml(config_path)
    adapter_slug = str(cfg.get("adapter_slug", "adapter"))
    adapter_dir = bundle / "adapters" / adapter_slug

    if args.skip_train:
        if not adapter_dir.is_dir():
            raise SystemExit(f"--skip-train but missing {adapter_dir}")
        print(f"[pipeline] Using existing adapter: {adapter_dir}", flush=True)
    else:
        adapter_dir = _run_train(bundle, config_path)

    _zip_adapter(bundle, config_path, adapter_dir)

    if not args.skip_infer:
        run_dir = _run_inference(
            bundle,
            config_path,
            infer_on=args.infer_on,
            harness_prompt=bool(args.harness_prompt),
            max_new_tokens=args.max_new_tokens,
        )
        _zip_inference(bundle, run_dir, config_path, args.infer_on)

    print(f"[pipeline] Done. Downloads under {bundle / 'downloads'}", flush=True)


if __name__ == "__main__":
    main()

# LoRA server bundle (inference only)

## 1. On the GPU server

```bash
cd lora_server_bundle
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements-lora-server.txt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LORA_EVAL_MAX_NEW_TOKENS=1024

# One adapter (recommended per job):
python generate_lora.py \
  --tests data/dataset_with_tests.jsonl \
  --variants data/metadata.jsonl \
  --adapters-dir adapters \
  --adapter lora_qwen2.5-coder-7b \
  --limit 50 \
  --out ../lora_run_7b

# All adapters — separate folders + one .zip per adapter (recommended):
python generate_lora.py \
  --tests data/dataset_with_tests.jsonl \
  --variants data/metadata.jsonl \
  --adapters-dir adapters \
  --limit 50 \
  --per-adapter \
  --out ../lora_run_all

# All adapters in one combined folder + lora_run_all.zip:
python generate_lora.py \
  --tests data/dataset_with_tests.jsonl \
  --variants data/metadata.jsonl \
  --adapters-dir adapters \
  --limit 50 \
  --out ../lora_run_combined
```

Zips are created automatically (use `--no-zip` to skip). Base Qwen weights download from Hugging Face on first use (~1G / ~3G / ~15G).

## 2. Copy zip(s) or folder(s) back to your Mac

```bash
# Per-adapter layout: grab each zip
scp user@SERVER:~/lora_run_all/qwen2.5-coder-7b.zip .
unzip qwen2.5-coder-7b.zip

# Or rsync the whole tree
rsync -avz user@SERVER:~/lora_run_all/ ./lora_run_all/
```

## 3. On your Mac (full repo): compile, tests, CodeBLEU

One adapter at a time (per-adapter layout):

```bash
cd /path/to/enriched-python-ast-transpiler
source .venv/bin/activate   # needs requirements-pipeline.txt (g++, codebleu)

python -m training.eval_local \
  --run-dir ./lora_run_all/qwen2.5-coder-7b \
  --tests results/pipeline_runs/20260530_165641/02_tests/dataset_with_tests.jsonl \
  --limit 50
```

Combined layout: use `--run-dir ./lora_run_combined` instead.

Results: `<run-dir>/evaluation/records.jsonl`, `<run-dir>/summary.json`

Optional speed benchmark: add `--benchmark`

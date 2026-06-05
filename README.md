# LoRA server bundle (pipeline train → infer)

Trains a LoRA adapter on the **pipeline benchmark train split** (~80% of ~568 items),
then runs inference on the **held-out test split** (or full dataset).

## 1. On Mac — pack (already done if you uploaded this folder)

```bash
python -m training.pack_lora_bundle \
  --tests results/pipeline_runs/20260530_165641/02_tests/dataset_with_tests.jsonl \
  --variants results/pipeline_runs/20260530_165641/04_variants/metadata.jsonl \
  --model 1.5b
```

## 2. On GPU server

```bash
cd lora_server_bundle
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements-lora-bundle.txt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LORA_EVAL_MAX_NEW_TOKENS=1024

# Train + zip adapter + infer on TEST split + zip results
python run_pipeline_gpu.py --config configs/pipeline-1.5b.yaml

# Infer on full 568-item benchmark instead of test-only
python run_pipeline_gpu.py --config configs/pipeline-1.5b.yaml --infer-on full

# Re-infer only (adapter already trained)
python run_pipeline_gpu.py --config configs/pipeline-1.5b.yaml --skip-train

# Stricter eval-style prompt at inference
python run_pipeline_gpu.py --config configs/pipeline-1.5b.yaml --harness-prompt
```

Configs: `configs/pipeline-0.5b.yaml`, `pipeline-1.5b.yaml`, `pipeline-7b.yaml`

## 3. Download zips

```bash
scp user@SERVER:~/lora_server_bundle/downloads/adapter_qwen2.5-coder-1_5b.zip .
scp user@SERVER:~/lora_server_bundle/downloads/inference_test_qwen2.5-coder-1_5b.zip .
```

## 4. On Mac — compile, tests, CodeBLEU

```bash
cd /path/to/enriched-python-ast-transpiler
unzip inference_test_qwen2.5-coder-1_5b.zip

python -m training.eval_local \
  --run-dir ./runs/infer_test_qwen2.5-coder-1_5b \
  --tests results/pipeline_runs/20260530_165641/02_tests/dataset_with_tests.jsonl \
  --variants results/pipeline_runs/20260530_165641/04_variants/metadata.jsonl
```

For test-only eval, filter is implicit (run dir only has test items).

## Data layout

| Path | Purpose |
|------|---------|
| `data/splits/train.jsonl` | SFT training (pipeline cpp_reference) |
| `data/splits/val.jsonl` | SFT validation |
| `data/splits/test_dataset_with_tests.jsonl` | Held-out inference/eval |
| `data/splits/split_manifest.json` | train/val/test item_ids |
| `data/dataset_with_tests.jsonl` | Full benchmark (optional full infer) |

## Inference-only (legacy)

```bash
python generate_lora.py \
  --tests data/splits/test_dataset_with_tests.jsonl \
  --variants data/splits/test_metadata.jsonl \
  --adapters-dir adapters \
  --adapter lora_qwen2.5-coder-1_5b \
  --out ../lora_run_test
```

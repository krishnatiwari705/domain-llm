# DomainLLM

DomainLLM is a reproducible, end-to-end Banking77 intent-classification project using
**Qwen/Qwen2.5-1.5B-Instruct** and parameter-efficient LoRA fine-tuning. It downloads
the public dataset, produces deterministic training records, evaluates the base model
and adapter with the same exact-match metric, and serves predictions through FastAPI.

## What is included

* **Automated data pipeline:** downloads `mteb/banking77`, converts numeric labels to
  canonical names, writes `train.jsonl`, `test.jsonl`, and `labels.json`.
* **Baseline and fine-tuned evaluation:** JSON artifacts preserve every prediction,
  raw model response, correct count, and accuracy. `compare-results` derives its change
  from these artifacts; this repository intentionally contains no invented scores.
* **LoRA SFT:** applies LoRA to Qwen attention projections while preserving base model
  weights. Training settings live in `configs/default.yaml`.
* **Production surface:** validated FastAPI request schema, health endpoint, Docker,
  unit tests, and GitHub Actions CI.

## Quick start

Python 3.10+ and sufficient RAM/VRAM for the 1.5B model are required. GPU is strongly
recommended for training.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
domain-llm prepare
# Measures the untouched model; it does not claim a result until you run it.
domain-llm evaluate-model --output artifacts/baseline.json
domain-llm train
domain-llm evaluate-model --adapter artifacts/lora --output artifacts/finetuned.json
domain-llm compare-results artifacts/baseline.json artifacts/finetuned.json
```

The generated evaluation artifact has `samples`, `correct`, `accuracy`, and individual
predictions. Use the saved artifacts—not a copied README number—for experiment reports.

## Configuration

`configs/default.yaml` is the single configuration source. Duplicate and edit it for
experiments, then pass `--config path/to/experiment.yaml` to `prepare`, `train`, or
`evaluate-model`. Important settings include model identifier, output locations, LoRA
rank/alpha/dropout, batch size, sequence length, seed, and epochs.

## API

Prepare data and train (or provide a previously trained adapter), then run:

```bash
ADAPTER_PATH=artifacts/lora LABELS_PATH=data/processed/labels.json \
uvicorn domain_llm.api:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/v1/predict -H 'content-type: application/json' \
  -d '{"text":"My transfer is still pending"}'
```

The `POST /v1/predict` response has canonical `intent` and model `raw_output` fields.
The service returns 503 if model loading/inference is unavailable rather than exposing
internal exception details. Inputs are limited to 1–2,000 characters.

## Docker

```bash
docker compose up --build
```

The compose file mounts local `data/` and `artifacts/` read-only. Model downloads occur
on first use unless a compatible Hugging Face cache is supplied to the container.

## Testing and CI

```bash
ruff check domain_llm tests
pytest -q
```

CI runs both commands on Python 3.11. Unit tests do not download the dataset or model;
the data/model workflow is deliberately explicit so CI remains fast and repeatable.

## Reproducibility notes

Evaluation uses greedy decoding (`do_sample=False`) and one shared label-aware prompt
for baseline and fine-tuned runs. The dataset snapshot supplied by Hugging Face can
change independently; archive generated JSONL and evaluation JSON files with any
reported result. Observe Banking77's dataset license and Qwen's model license before
deployment.

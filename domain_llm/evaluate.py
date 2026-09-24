"""Evaluation and artifact comparison without fabricated metrics."""
from __future__ import annotations
import json
from pathlib import Path
from .data import read_jsonl
from .inference import IntentPredictor

def evaluate(predictor: IntentPredictor, test_file: str, output_file: str) -> dict:
    rows = read_jsonl(test_file)
    predictions = []
    correct = 0
    for row in rows:
        prediction, raw = predictor.predict(row["text"])
        correct += prediction == row["label"]
        predictions.append({"text": row["text"], "expected": row["label"], "prediction": prediction, "raw": raw})
    result = {"samples": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else 0.0, "predictions": predictions}
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result

def compare(baseline_file: str, finetuned_file: str) -> dict:
    base, tuned = json.loads(Path(baseline_file).read_text()), json.loads(Path(finetuned_file).read_text())
    return {"baseline_accuracy": base["accuracy"], "finetuned_accuracy": tuned["accuracy"], "absolute_change": tuned["accuracy"] - base["accuracy"], "samples": tuned["samples"]}

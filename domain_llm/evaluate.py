"""Evaluation and artifact comparison."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .data import read_jsonl
from .inference import IntentPredictor


def _classification_metrics(expected: list[str], predicted: list[str]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    labels = sorted(set(expected))
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(expected, predicted, labels=labels, average="weighted", zero_division=0)
        ),
    }


def evaluate(
    predictor: IntentPredictor,
    test_file: str,
    output_file: str,
    batch_size: int = 4,
) -> dict:
    rows = read_jsonl(test_file)
    predictions = []
    expected_labels = []
    predicted_labels = []
    correct = 0
    total_latency = 0.0

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [row["text"] for row in batch]

        started = time.perf_counter()
        batch_predictions = predictor.predict_batch(texts)
        total_latency += time.perf_counter() - started

        for row, (prediction, raw) in zip(batch, batch_predictions):
            expected = row["label"]
            correct += prediction == expected
            expected_labels.append(expected)
            predicted_labels.append(prediction)
            predictions.append(
                {
                    "text": row["text"],
                    "expected": expected,
                    "prediction": prediction,
                    "raw": raw,
                }
            )

    metrics = (
        _classification_metrics(expected_labels, predicted_labels)
        if rows
        else {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    )
    parameters = predictor.parameter_counts()
    result = {
        "samples": len(rows),
        "correct": correct,
        **metrics,
        "total_inference_seconds": total_latency,
        "average_latency_seconds": total_latency / len(rows) if rows else 0.0,
        "parameters": parameters,
        "predictions": predictions,
    }
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def compare(baseline_file: str, finetuned_file: str) -> dict:
    base = json.loads(Path(baseline_file).read_text())
    tuned = json.loads(Path(finetuned_file).read_text())
    result = {
        "baseline_accuracy": base["accuracy"],
        "finetuned_accuracy": tuned["accuracy"],
        "absolute_change": tuned["accuracy"] - base["accuracy"],
        "samples": tuned["samples"],
    }
    for metric in ("macro_f1", "weighted_f1"):
        if metric in base and metric in tuned:
            result[f"baseline_{metric}"] = base[metric]
            result[f"finetuned_{metric}"] = tuned[metric]
            result[f"{metric}_absolute_change"] = tuned[metric] - base[metric]
    if "average_latency_seconds" in base and "average_latency_seconds" in tuned:
        result["baseline_average_latency_seconds"] = base["average_latency_seconds"]
        result["finetuned_average_latency_seconds"] = tuned["average_latency_seconds"]
    return result

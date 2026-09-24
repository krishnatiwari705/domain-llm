import json

from domain_llm.evaluate import _classification_metrics, compare


def test_classification_metrics_include_f1_scores():
    result = _classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert result["accuracy"] == 0.75
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert 0.0 <= result["weighted_f1"] <= 1.0


def test_compare_reports_measured_delta(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({
        "accuracy": 0.25,
        "macro_f1": 0.2,
        "weighted_f1": 0.25,
        "average_latency_seconds": 0.2,
        "samples": 4,
    }))
    b.write_text(json.dumps({
        "accuracy": 0.5,
        "macro_f1": 0.45,
        "weighted_f1": 0.5,
        "average_latency_seconds": 0.3,
        "samples": 4,
    }))
    result = compare(str(a), str(b))
    assert result["absolute_change"] == 0.25
    assert result["macro_f1_absolute_change"] == 0.25
    assert result["weighted_f1_absolute_change"] == 0.25
    assert result["baseline_average_latency_seconds"] == 0.2

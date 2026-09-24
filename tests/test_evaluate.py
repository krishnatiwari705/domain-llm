import json
from domain_llm.evaluate import compare

def test_compare_reports_measured_delta(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"accuracy": .25, "samples": 4}))
    b.write_text(json.dumps({"accuracy": .5, "samples": 4}))
    assert compare(str(a), str(b))["absolute_change"] == .25

"""Dataset download, validation, and training record creation."""
from __future__ import annotations
import json
from pathlib import Path
from .prompts import training_text

def prepare_dataset(dataset_name: str, output_dir: str, train_split="train", test_split="test") -> dict:
    """Download Banking77 and save JSONL records plus label metadata reproducibly."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name)
    labels = ds[train_split].features["label"].names
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    for source, name in ((train_split, "train"), (test_split, "test")):
        records = []
        for row in ds[source]:
            label = labels[row["label"]] if isinstance(row["label"], int) else row["label"]
            records.append({"text": row["text"], "label": label, "training_text": training_text(row["text"], label, labels)})
        with (target / f"{name}.jsonl").open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (target / "labels.json").write_text(json.dumps(labels, indent=2))
    return {"labels": len(labels), "train": len(ds[train_split]), "test": len(ds[test_split])}

def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

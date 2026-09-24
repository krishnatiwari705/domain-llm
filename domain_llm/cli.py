"""Command-line workflow entrypoints."""
from __future__ import annotations

import json

import typer

from .config import load_config
from .data import prepare_dataset
from .evaluate import compare, evaluate
from .inference import IntentPredictor
from .train import train_lora

app = typer.Typer(no_args_is_help=True)
def train(config: str = "configs/default.yaml"):
    train_lora(load_config(config))
@app.command()
def evaluate_model(config: str = "configs/default.yaml", adapter: str = "", output: str = "artifacts/evaluation.json"):
    c = load_config(config); p = IntentPredictor(c.model.name, f"{c.data.output_dir}/labels.json", adapter or None, c.model.device_map, c.model.max_new_tokens); typer.echo(json.dumps(evaluate(p, f"{c.data.output_dir}/test.jsonl", output)))
@app.command()
def compare_results(baseline: str, finetuned: str):
    typer.echo(json.dumps(compare(baseline, finetuned), indent=2))

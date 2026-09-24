"""Typed, file-based configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 32
    device_map: str = "auto"
@dataclass
class DataConfig:
    dataset: str = "PolyAI/banking77"
    train_split: str = "train"
    test_split: str = "test"
    output_dir: str = "data/processed"
@dataclass
class TrainingConfig:
    output_dir: str = "artifacts/lora"
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load YAML, rejecting unknown top-level sections to catch misspellings."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Configuration loading requires PyYAML; install project dependencies.") from exc
    raw: dict = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {"model", "data", "training"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )

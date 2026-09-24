import pytest
from domain_llm.config import Config, load_config

def test_default_config_values():
    config = Config()
    assert config.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.training.lora_r == 16

def test_config_dependency_error_is_clear(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "yaml", None)
    with pytest.raises(RuntimeError, match="PyYAML"):
        load_config()

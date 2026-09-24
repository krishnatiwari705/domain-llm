import pytest

from domain_llm.config import Config, load_config


def test_default_config_values():

def test_config_dependency_error_is_clear(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "yaml", None)
    with pytest.raises(RuntimeError, match="PyYAML"):
        load_config()

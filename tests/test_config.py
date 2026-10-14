# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import pytest
import yaml

import litgpt.config as config_module
from litgpt import Config
from litgpt.config import find_multiple


def test_config():
    config = Config()
    assert config.name == ""
    assert config.block_size == 4096

    config = Config(block_size=2048)
    assert config.block_size == 2048

    config = Config.from_name("delphi-2m")
    assert config.block_size == 48

    config = Config.from_name("delphi-2m", block_size=4096)
    assert config.block_size == 4096


def test_nonexisting_name():
    with pytest.raises(ValueError, match="'invalid-model-name' is not a supported config name"):
        Config.from_name("invalid-model-name")


@pytest.mark.parametrize("config", config_module.configs, ids=[c["name"] for c in config_module.configs])
def test_named_configs_are_valid(config):
    assert Config.from_name(config["name"]).name == config["name"]


def test_from_checkpoint(tmp_path):
    # 1. Neither `lit_config.py` nor matching config exists.
    with pytest.raises(FileNotFoundError, match="neither 'model_config.yaml' nor matching config exists"):
        Config.from_checkpoint(tmp_path / "non_existing_checkpoint")

    # 2. If `lit_config.py` doesn't exists, but there is a matching config in `litgpt/config.py`.
    config = Config.from_checkpoint(tmp_path / "delphi-2m")
    assert config.name == "delphi-2m"
    assert config.block_size == 48
    assert config.n_layer == 12

    # 3. If only `lit_config.py` exists.
    config_data = {"name": "delphi-2m", "block_size": 24, "n_layer": 2}
    with open(tmp_path / "model_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config_data, file)
    config = Config.from_checkpoint(tmp_path)
    assert config.name == "delphi-2m"
    assert config.block_size == 24
    assert config.n_layer == 2

    # 4. Both `lit_config.py` and a matching config exist, but `lit_config.py` supersedes matching config
    (tmp_path / "delphi-2m").mkdir()
    with open(tmp_path / "delphi-2m/model_config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config_data, file)
    config = Config.from_checkpoint(tmp_path / "delphi-2m")
    assert config.name == "delphi-2m"
    assert config.block_size == 24
    assert config.n_layer == 2


@pytest.mark.parametrize("head_size", [None, 128])
def test_head_size(head_size):
    config = Config(head_size)

    assert config.head_size == head_size or config.n_embd // config.n_head


def test_find_multiple():
    assert find_multiple(17, 5) == 20
    assert find_multiple(30, 7) == 35
    assert find_multiple(10, 2) == 10
    assert find_multiple(5, 10) == 10
    assert find_multiple(50254, 128) == 50304
    assert find_multiple(50254, 256) == 50432
    assert find_multiple(50254, 512) == 50688

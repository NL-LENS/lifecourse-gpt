# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""The shipped Delphi config must keep matching `litgpt.pretrain`'s signature."""

from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest

pytest.importorskip("duckdb")

import litgpt.pretrain as pretrain  # noqa: E402
from litgpt.config import Config  # noqa: E402
from litgpt.life_events.delphi.data import DelphiLifeEvents  # noqa: E402
from litgpt.model import model_class  # noqa: E402
from litgpt.utils import CLI  # noqa: E402

CONFIG = Path(__file__).parent.parent.parent / "config_hub" / "pretrain" / "delphi-2m.yaml"


def test_config_parses_and_selects_delphi(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(pretrain, "main", Mock(side_effect=lambda *a, **kw: captured.update(kw)))
    argv = ["pretrain", "--config", str(CONFIG), "--devices", "1", "--out_dir", str(tmp_path)]
    with mock.patch("sys.argv", argv):
        CLI(pretrain.setup)

    pretrain.main.assert_called_once()
    assert isinstance(captured["data"], DelphiLifeEvents)
    config = captured["config"]
    assert model_class(config).__name__ == "Delphi"
    # The vocabulary has to cover the dummy data exactly: padding entries are not free
    # under the competing-risks loss.
    assert config.vocab_size == config.padded_vocab_size
    assert config.rotary_percentage == 0.0


def test_config_matches_the_architecture_it_names():
    """The YAML spells the architecture out, only because `model_config` replaces a
    named config wholesale. The two must not drift apart."""
    import yaml

    from_yaml = yaml.safe_load(CONFIG.read_text())["model_config"]
    named = Config.from_name("delphi-2m")
    for field, value in from_yaml.items():
        if field in ("vocab_size", "padded_vocab_size"):
            continue  # deliberately the dummy data's vocabulary, not Delphi's 1270
        assert value == getattr(named, field), field


def test_prepare_data_only_builds_the_cache_without_training(tmp_path, dataset, monkeypatch):
    """`litgpt pretrain --prepare_data_only true` writes the dataset and stops."""
    monkeypatch.setattr(pretrain, "main", Mock(side_effect=AssertionError("training must not start")))
    event_paths, person_path = dataset
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    argv = [
        "pretrain",
        "delphi-2m",
        "--prepare_data_only",
        "true",
        "--out_dir",
        str(out_dir),
        "--devices",
        "1",
        "--data",
        "DelphiLifeEvents",
        f"--data.event_paths=[{','.join(event_paths)}]",
        "--data.person_path",
        str(person_path),
        "--data.cache_dir",
        str(cache_dir),
        "--data.val_split_fraction",
        "0.34",
    ]
    with mock.patch("sys.argv", argv):
        CLI(pretrain.setup)

    pretrain.main.assert_not_called()
    assert (cache_dir / "meta.json").is_file()
    assert (cache_dir / "vocab.json").is_file()
    # A preparation run has no business creating a run directory or a logger.
    assert not out_dir.exists()

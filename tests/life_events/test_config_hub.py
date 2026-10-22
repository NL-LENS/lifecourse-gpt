# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""The shipped Delphi config must keep matching `litgpt.pretrain`'s signature."""

import re
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest

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
    assert config.rotary_percentage == 0.0


def test_config_matches_the_architecture_it_names():
    """The YAML spells the architecture out, only because `model_config` replaces a
    named config wholesale. The two must not drift apart."""
    import yaml

    from_yaml = yaml.safe_load(CONFIG.read_text())["model_config"]
    named = Config.from_name("delphi-2m")
    for field, value in from_yaml.items():
        assert value == getattr(named, field), field
    # Read from the data cache instead, so it can never disagree with the dataset.
    assert not {"vocab_size", "padded_vocab_size", "target_vocab_size"} & from_yaml.keys()


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


def test_pretrain_sizes_the_vocabulary_from_the_cache(tmp_path, static_dataset, monkeypatch, capsys):
    """The model config's vocabulary is a placeholder: `main` builds the model from the cache's,
    and logs that config rather than the one it was given."""
    from dataclasses import asdict

    import lightning as L

    from litgpt.args import EvalArgs, TrainArgs

    class Built(Exception):
        pass

    def stop(**kwargs):
        raise Built(kwargs["state"]["model"].config)

    monkeypatch.setattr(pretrain, "fit", stop)
    event_paths, person_path = static_dataset
    data = DelphiLifeEvents(
        event_paths=event_paths,
        person_path=person_path,
        cache_dir=tmp_path / "cache",
        static_fields=["GENDER", "MIGRATION"],
        val_split_fraction=1 / 3,
        num_workers=0,
    )
    config = Config.from_name("delphi-2m", block_size=8, n_layer=1, n_head=1, n_embd=4, intermediate_size=8)
    with pytest.raises(Built) as built:
        pretrain.main(
            fabric=L.Fabric(accelerator="cpu", devices=1),
            devices=1,
            seed=0,
            initial_checkpoint_dir=None,
            resume=False,
            config=config,
            data=data,
            out_dir=tmp_path / "out",
            train=TrainArgs(global_batch_size=2, micro_batch_size=2, max_tokens=16, max_norm=1.0, compile=False),
            eval=EvalArgs(),
            optimizer="AdamW",
            hparams={"model_config": asdict(config)},
        )

    model_config = built.value.args[0]
    assert data.target_vocab_size < data.vocab_size < config.vocab_size
    assert model_config.vocab_size == model_config.padded_vocab_size == data.vocab_size
    assert model_config.target_vocab_size == data.target_vocab_size
    logged = re.findall(r"'(\w*vocab_size)': (\d+)", capsys.readouterr().out)
    assert dict(logged) == {
        "vocab_size": str(data.vocab_size),
        "padded_vocab_size": str(data.vocab_size),
        "target_vocab_size": str(data.target_vocab_size),
    }

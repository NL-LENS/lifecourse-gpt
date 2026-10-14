from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest

import litgpt.pretrain as pretrain
from litgpt.__main__ import main


def test_cli():
    out = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out), mock.patch("sys.argv", ["litgpt", "-h"]):
        main()
    out = out.getvalue()
    assert "usage: litgpt" in out
    assert "{pretrain,dummy_data,generate}" in out
    assert (
        """Available subcommands:
    pretrain            Pretrain a model."""
        in out
    )

    out = StringIO()
    with pytest.raises(SystemExit), redirect_stdout(out), mock.patch("sys.argv", ["litgpt", "pretrain", "-h"]):
        main()
    out = out.getvalue()
    assert (
        """--train.max_tokens MAX_TOKENS
                        Total number of tokens to train on (type: int | None,
                        default: 3000000000000)"""
        in out
    )


def test_pretrain_allows_max_steps(tmp_path):
    # Ensure --train.max_steps is accepted by the CLI for pretrain
    # and only emits a warning instead of raising a validation error.
    args = [
        "litgpt",
        "pretrain",
        "--config",
        str(Path(__file__).parent.parent / "config_hub" / "pretrain" / "delphi-2m.yaml"),
        "--train.max_steps=1",
        f"--out_dir={tmp_path}",
        "--logger_name=csv",
        "--devices=1",
    ]

    def validate_only(**kwargs):
        pretrain.validate_args(kwargs["train"], kwargs["eval"], kwargs["initial_checkpoint_dir"], kwargs["resume"])

    with (
        pytest.warns(UserWarning, match="max_steps"),
        mock.patch("sys.argv", args),
        mock.patch.object(pretrain, "main", side_effect=validate_only),
    ):
        main()

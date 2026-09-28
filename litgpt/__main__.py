# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import warnings

import torch
from jsonargparse import CLI, set_docstring_parse_options

from litgpt.generate import setup as generate_fn
from litgpt.life_events.dummy_data import generate as dummy_data_fn
from litgpt.parser_config import parser_commands
from litgpt.pretrain import setup as pretrain_fn

PARSER_DATA = {
    "pretrain": pretrain_fn,
    "dummy_data": dummy_data_fn,
    "generate": generate_fn,
}


def _check_commands():
    assert set(parser_commands()) == set(PARSER_DATA.keys()), (
        "PARSER_DATA has to be kept in sync with litgpt.parser_config.parser_commands()"
    )


def main() -> None:
    _check_commands()

    set_docstring_parse_options(attribute_docstrings=True)

    # PyTorch bug that raises a false-positive warning
    # More info: https://github.com/Lightning-AI/litgpt/issues/1561
    warning_message = r"The epoch parameter in `scheduler.step\(\)` was not necessary and is being deprecated.*"

    warnings.filterwarnings(
        action="ignore", message=warning_message, category=UserWarning, module=r".*torch\.optim\.lr_scheduler.*"
    )

    torch.set_float32_matmul_precision("high")
    CLI(PARSER_DATA)


if __name__ == "__main__":
    main()

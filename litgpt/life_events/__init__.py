# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Life-course event sequence modelling.

This module contains additional functionality for training GPT-style models on life-course event sequence data. The
module expects training and evaluation data in a preprocessed "silver" layer format
(see :mod:`data`).

The :mod:`data` module is responsible for abstract data loading and tokenization routines that are model-agnostic.
The different GPT architectures are defined in their own subpackages (e.g., :mod:`delphi`) and bring their own
requirements for input data, model components, loss functions, and sampling.

To add a new architecture, subclass `data.LifeEventsDataModule` and `litgpt.model.GPT`, and point the
model config's `model_class_name` at the latter.
"""

# NOTE: All life-event specific code should live in this folder so that pulling upstream changes becomes easy

# `litgpt.data` re-exports `DelphiLifeEvents` so that `--data DelphiLifeEvents` works on
# the CLI, and the data modules here subclass `litgpt.data.base.DataModule`. Whichever
# side is imported first, `litgpt.data` has to finish initializing before any module
# below it does, or the re-export finds a half-built module. Importing it here, before
# any submodule of this package can run, settles the order.
import litgpt.data  # noqa: E402, F401

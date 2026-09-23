# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Centralized package availability constants for optional dependencies."""

from lightning_utilities.core.imports import RequirementCache

# Logger-related constants
_SUPPORTED_LOGGERS: tuple[str, ...] = ("csv", "tensorboard", "wandb", "mlflow", "litlogger")

# Logger-related optional dependencies
_LITLOGGER_AVAILABLE = RequirementCache("litlogger>=0.1.7")

# PyTorch version-specific constants
_TORCH_EQUAL_2_7 = RequirementCache("torch>=2.7.0,<2.8")
_TORCH_EQUAL_2_8 = RequirementCache("torch>=2.8.0,<2.9")

# litgpt.life_events.data
_DUCKDB_AVAILABLE = RequirementCache("duckdb")
_PYARROW_AVAILABLE = RequirementCache("pyarrow")

# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Utility functions for training and inference."""

import inspect
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
import torch.utils._device
import yaml
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from typing_extensions import Self

from litgpt.constants import (
    _LITLOGGER_AVAILABLE,
    _SUPPORTED_LOGGERS,
)
from litgpt.types import LoggerChoice

if TYPE_CHECKING:
    from litgpt import Config


def init_out_dir(out_dir: Path) -> Path:
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    if not out_dir.is_absolute() and "LIGHTNING_ARTIFACTS_DIR" in os.environ:
        return Path(os.getenv("LIGHTNING_ARTIFACTS_DIR")) / out_dir
    return out_dir


def find_resume_path(resume: bool | Literal["auto"] | Path, out_dir: Path) -> Path | None:
    if not resume or isinstance(resume, Path):
        return resume

    resume_path = max(out_dir.rglob("step-*/*.pth"), key=(lambda p: int(p.parent.name.split("-")[1])), default=None)
    if resume == "auto":
        return resume_path
    if resume is True and resume_path is None:
        raise FileNotFoundError(
            f"You passed `--resume=True`, but no checkpoint file was found in `--out_dir={out_dir}`."
        )
    return resume_path


def num_parameters(module: nn.Module, requires_grad: bool | None = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state.shape)
            else:
                total += p.numel()
    return total


def reset_parameters(module: nn.Module) -> None:
    """Calls `reset_parameters` on the module and all its submodules."""
    for mod in module.modules():
        if callable(getattr(mod, "reset_parameters", None)):
            mod.reset_parameters()


def chunked_cross_entropy(
    logits: torch.Tensor | list[torch.Tensor],
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_index: int = -100,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude

    # lm_head was chunked (we are fine-tuning)
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

        # chunk cross entropy
        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        non_masked_elems = (targets != ignore_index).sum()
        # See [non_masked_elems div note]
        return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

    # lm_head wasn't chunked, chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    non_masked_elems = (targets != ignore_index).sum()
    # [non_masked_elems div note]:
    #   max(1, non_masked_elems) would be more ergonomic to avoid a division by zero. However that
    #   results in a python int which is then passed back to torch division. By using the
    #   `x.maximum(torch.ones_like(x))` pattern we avoid a cudaStreamSynchronize.
    return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))


def get_default_supported_precision(training: bool) -> str:
    """
    Return the default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: If True, returns '-mixed' version of the precision; if False, returns '-true' version.

    Returns:
        The default precision that is suitable for the task and is supported by the hardware.
    """
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed" if training else "bf16-true"
        else:
            return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


def compile_unsupported_reason() -> str | None:
    """Why `torch.compile` would fail on this machine, or `None` when it should work.

    Covers only what is knowable before the first compile: the interpreter, which torch
    checks itself, and the C++ compiler TorchInductor shells out to for its CPU backend.
    That compiler is MSVC on Windows, and `cl.exe` is on PATH only inside a developer
    command prompt or once the Build Tools are installed, which is what makes an
    unconditional `torch.compile` fail on an otherwise fine Windows machine.
    """
    if not torch._dynamo.is_inductor_supported():
        return f"TorchInductor does not support Python {sys.version_info.major}.{sys.version_info.minor}"
    if sys.platform == "win32" and shutil.which("cl") is None:
        return "TorchInductor compiles its CPU kernels with MSVC and `cl.exe` is not on PATH"
    return None


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> Self:
        return self


def CLI(*args: Any, **kwargs: Any) -> Any:
    from jsonargparse import CLI, set_docstring_parse_options

    set_docstring_parse_options(attribute_docstrings=True)

    return CLI(*args, **kwargs)


def capture_hparams() -> dict[str, Any]:
    """Captures the local variables ('hyperparameters') from where this function gets called."""
    caller_frame = inspect.currentframe().f_back
    locals_of_caller = caller_frame.f_locals
    hparams = {}
    for name, value in locals_of_caller.items():
        if value is None or isinstance(value, (int, float, str, bool, Path)):
            hparams[name] = value
        elif is_dataclass(value):
            hparams[name] = asdict(value)
        else:
            hparams[name] = str(value)
    return hparams


def save_config(config: "Config", checkpoint_dir: Path) -> None:
    config_dict = asdict(config)
    with open(checkpoint_dir / "model_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(config_dict, fp)


def parse_devices(devices: str | int) -> int:
    if devices in (-1, "auto"):
        return torch.cuda.device_count() or 1
    if isinstance(devices, int) and devices > 0:
        return devices
    raise ValueError(f"Devices must be 'auto' or a positive integer, got: {devices!r}")


def choose_logger(
    logger_name: LoggerChoice,
    out_dir: Path,
    name: str,
    log_interval: int = 1,
    log_args: dict | None = None,
    resume: bool | None = None,
    **kwargs: Any,
):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", flush_logs_every_n_steps=log_interval, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", **kwargs)
    if logger_name == "wandb":
        log_args = log_args or {}
        project = log_args.pop("project", name)
        run = log_args.pop("run", os.environ.get("WANDB_RUN_NAME"))
        group = log_args.pop("group", os.environ.get("WANDB_RUN_GROUP"))
        return WandbLogger(project=project, name=run, group=group, resume=resume, **kwargs)
    if logger_name == "mlflow":
        return MLFlowLogger(experiment_name=name, **kwargs)
    if logger_name == "litlogger":
        if not _LITLOGGER_AVAILABLE:
            raise ModuleNotFoundError(_LITLOGGER_AVAILABLE)
        from lightning.pytorch.loggers import LitLogger

        # Extract litlogger-specific args
        teamspace = log_args.pop("teamspace", None) if log_args else None
        metadata = log_args.pop("metadata", None) if log_args else None
        log_model = log_args.pop("log_model", False) if log_args else False
        save_logs = log_args.pop("save_logs", True) if log_args else True
        checkpoint_name = log_args.pop("checkpoint_name", None) if log_args else None

        return LitLogger(
            root_dir=(out_dir / "logs"),
            name=name,
            teamspace=teamspace,
            metadata=metadata,
            log_model=log_model,
            save_logs=save_logs,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
    raise ValueError(
        f"`--logger_name={logger_name}` is not a valid option. Choose from {', '.join(_SUPPORTED_LOGGERS)}."
    )


def instantiate_torch_optimizer(optimizer, model_parameters, **kwargs):
    # Special care taken where some optimizers do not have some parameters referenced in some of the code, for example "fused" in the pretrain.py script:
    #   bnb.optim.AdamW8bit
    #   grokadamw.GrokAdamW
    #   torch.optim.RMSprop

    if isinstance(optimizer, str):
        if "." in optimizer:
            class_module, class_name = optimizer.rsplit(".", 1)
        else:
            class_module, class_name = "torch.optim", optimizer

        module = __import__(class_module, fromlist=[class_name])
        optimizer_cls = getattr(module, class_name)

        valid_params = set(inspect.signature(optimizer_cls).parameters)
        kwargs = {key: value for key, value in dict(kwargs).items() if key in valid_params}
        optimizer = optimizer_cls(model_parameters, **kwargs)
    elif isinstance(optimizer, dict):
        optimizer = dict(optimizer)
        class_module, class_name = optimizer["class_path"].rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        optimizer_cls = getattr(module, class_name)

        valid_params = set(inspect.signature(optimizer_cls).parameters)
        kwargs = {key: value for key, value in dict(kwargs).items() if key in valid_params}

        optimizer["init_args"].update(kwargs)
        optimizer = instantiate_class(model_parameters, optimizer)
    else:
        raise ValueError(f'Unrecognized "optimizer" value: {optimizer}')

    return optimizer


def extend_checkpoint_dir(checkpoint_dir: Path) -> Path:
    new_checkpoint_dir = "checkpoints" / checkpoint_dir
    should_return_new_dir = (
        not checkpoint_dir.is_dir()
        and checkpoint_dir.parts[0] != "checkpoints"
        and not checkpoint_dir.is_absolute()
        and new_checkpoint_dir.exists()
    )
    return new_checkpoint_dir if should_return_new_dir else checkpoint_dir


def check_nvlink_connectivity(fabric=None):
    """Checks GPU connectivity for both NVIDIA and AMD GPUs.

    This function delegates to vendor-specific implementations based on
    the detected GPU vendor.
    """
    if fabric is not None:
        custom_print = fabric.print
    else:
        custom_print = print

    if os.getenv("RANK", "0") == "0":
        try:
            if torch.cuda.is_available():
                device_properties = torch.cuda.get_device_properties(0)
                gpu_name = device_properties.name.lower()
                if "nvidia" in gpu_name:
                    _check_nvidia_connectivity(custom_print)
                elif "advanced micro devices" in gpu_name or "amd" in gpu_name:
                    _check_amd_connectivity(custom_print)
                else:
                    custom_print(f"Unrecognized GPU vendor: {device_properties.name}")
            else:
                custom_print("No GPUs available")
        except Exception as e:
            custom_print(f"An error occurred while checking GPU connectivity: {e}")


def _check_nvidia_connectivity(custom_print):
    """Checks NVLink connectivity on NVIDIA GPUs."""
    result = subprocess.run(["nvidia-smi", "topo", "-m"], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        custom_print("Failed to run nvidia-smi")
        return

    lines = result.stdout.strip().split("\n")
    start_index = next((i for i, line in enumerate(lines) if "GPU0" in line), None)
    if start_index is None:
        custom_print("Failed to parse nvidia-smi output")
        return

    headers_line = lines[start_index]
    headers = headers_line.split()
    gpu_regex = re.compile(r"^GPU\d+$")
    gpu_count = len([header for header in headers if gpu_regex.match(header)])

    all_nvlink = True
    for line in lines[start_index + 1 : start_index + 1 + gpu_count]:
        columns = line.split()
        connections = columns[1 : 1 + gpu_count]
        if not all("NV" in conn for conn in connections if conn != "X"):
            all_nvlink = False
            break

    if all_nvlink:
        custom_print("All GPUs are fully connected via NVLink.")
    else:
        custom_print(
            "Warning: Not all GPUs are fully connected via NVLink. Some GPUs are connected via slower interfaces. "
            "It is recommended to switch to a different machine with faster GPU connections for optimal multi-GPU training performance."
        )


def _check_amd_connectivity(custom_print):
    """Checks XGMI connectivity on AMD GPUs."""
    result = subprocess.run(["rocm-smi", "--showtopotype"], stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        custom_print("Failed to run rocm-smi")
        return

    lines = result.stdout.strip().split("\n")
    gpu_header_index = next((i for i, line in enumerate(lines) if re.match(r"^\s*GPU0", line)), None)
    if gpu_header_index is None or gpu_header_index == 0:
        custom_print("Failed to parse rocm-smi output (no GPU headers found)")
        return

    header_line = lines[gpu_header_index - 1]
    headers = header_line.strip().split()
    gpu_regex = re.compile(r"^GPU\d+$")
    gpu_count = len([header for header in headers if gpu_regex.match(header)])

    gpu_lines = []
    for line in lines[gpu_header_index : gpu_header_index + gpu_count]:
        if re.match(r"^\s*GPU\d+", line):
            gpu_lines.append(line.strip())
    if len(gpu_lines) != gpu_count:
        custom_print("Mismatch in GPU count when parsing rocm-smi output")
        return

    all_xgmi = True
    for line in gpu_lines:
        columns = line.split()
        connections = columns[1 : 1 + gpu_count]
        for conn in connections:
            if conn not in ("XGMI", "0"):
                all_xgmi = False
                break
        if not all_xgmi:
            break

    if all_xgmi:
        custom_print("All GPUs are fully connected via XGMI.")
    else:
        custom_print(
            "Warning: Not all GPUs are fully connected via XGMI. Some GPUs are connected via slower interfaces. "
            "It is recommended to switch to a different machine with faster GPU connections for optimal multi-GPU training performance."
        )

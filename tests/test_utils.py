# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
import yaml
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.loggers import LitLogger, MLFlowLogger, WandbLogger
from lightning_utilities.core.imports import RequirementCache

from litgpt.args import TrainArgs
from litgpt.constants import _LITLOGGER_AVAILABLE
from litgpt.parser_config import save_hyperparameters
from litgpt.utils import (
    CLI,
    CycleIterator,
    capture_hparams,
    check_nvlink_connectivity,
    choose_logger,
    chunked_cross_entropy,
    extend_checkpoint_dir,
    find_resume_path,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
)

_TENSORBOARD_AVAILABLE = RequirementCache("tensorboard")
_WANDB_AVAILABLE = RequirementCache("wandb")
_MLFLOW_AVAILABLE = RequirementCache("mlflow")
_MLFLOW_SKINNY_AVAILABLE = RequirementCache("mlflow-skinny")


# match fails on windows. why did they have to use backslashes?


@pytest.mark.parametrize("B", (1, 2))
@pytest.mark.parametrize("ignore_index", (None, -1, -2, -100))
def test_chunked_cross_entropy(ignore_index, B):
    V = 50
    T = 25
    regular_logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))

    if ignore_index is not None:
        targets[:, [1, 4, 10, 19]] = ignore_index

    baseline_loss = F.cross_entropy(
        regular_logits.reshape(-1, regular_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=(ignore_index if ignore_index is not None else -100),
    )

    ignore_index = ignore_index if ignore_index is not None else -100
    regular_loss = chunked_cross_entropy(regular_logits, targets, chunk_size=0, ignore_index=ignore_index)
    assert torch.equal(baseline_loss, regular_loss)
    assert regular_loss.numel() == 1

    chunked_loss = chunked_cross_entropy(regular_logits, targets, chunk_size=10, ignore_index=ignore_index)
    torch.testing.assert_close(chunked_loss, regular_loss)
    torch.testing.assert_close(chunked_loss, baseline_loss)

    logit_chunk_size = 6
    assert T % logit_chunk_size != 0  # ensure leftover
    chunked_logits = list(regular_logits.split(logit_chunk_size, dim=1))
    chunked_loss = chunked_cross_entropy(chunked_logits, targets, chunk_size=0, ignore_index=ignore_index)
    torch.testing.assert_close(chunked_loss, regular_loss)
    torch.testing.assert_close(chunked_loss, baseline_loss)

    chunked_loss = chunked_cross_entropy(chunked_logits, targets, chunk_size=10, ignore_index=ignore_index)
    torch.testing.assert_close(chunked_loss, regular_loss)
    torch.testing.assert_close(chunked_loss, baseline_loss)


def test_num_parameters():
    model = torch.nn.Linear(2, 2)
    assert num_parameters(model) == 6
    assert num_parameters(model, requires_grad=True) == 6
    assert num_parameters(model, requires_grad=False) == 0

    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad = False
    assert num_parameters(model) == 6
    assert num_parameters(model, requires_grad=True) == 4
    assert num_parameters(model, requires_grad=False) == 2


def test_cycle_iterator():
    iterator = CycleIterator([])
    with pytest.raises(StopIteration):
        next(iterator)

    iterator = CycleIterator(range(3))
    assert iterator.epoch == 0
    assert next(iterator) == 0
    assert iterator.epoch == 0
    assert next(iterator) == 1
    assert iterator.epoch == 0
    assert next(iterator) == 2
    assert iterator.epoch == 0
    assert next(iterator) == 0
    assert iterator.epoch == 1


def test_parse_devices():
    with pytest.raises(ValueError, match="must be 'auto' or a positive integer"):
        assert parse_devices(0)
    with pytest.raises(ValueError, match="must be 'auto' or a positive integer"):
        assert parse_devices(-2)

    with mock.patch("litgpt.utils.torch.cuda.device_count", return_value=0):
        assert parse_devices("auto") == 1  # CPU
        assert parse_devices(10) == 10  # leave validation up to Fabric later on
    with mock.patch("litgpt.utils.torch.cuda.device_count", return_value=1):
        assert parse_devices("auto") == 1  # CUDA
    with mock.patch("litgpt.utils.torch.cuda.device_count", return_value=3):
        assert parse_devices("auto") == 3
        assert parse_devices(-1) == 3

    assert parse_devices(5) == 5


def test_capture_hparams():
    integer = 1
    string = "string"
    boolean = True
    none = None
    path = Path("/path")
    dataclass = TrainArgs()
    other = torch.nn.Linear(1, 1)
    hparams = capture_hparams()
    assert hparams == {
        "integer": integer,
        "string": string,
        "boolean": boolean,
        "none": none,
        "path": path,
        "dataclass": asdict(dataclass),
        "other": str(other),
    }


def _test_function(out_dir: Path, foo: bool = False, bar: int = 1):
    save_hyperparameters(_test_function, out_dir)


def test_save_hyperparameters(tmp_path):
    with mock.patch("sys.argv", ["any.py", str(tmp_path), "--foo", "True"]):
        CLI(_test_function)

    with open(tmp_path / "hyperparameters.yaml", encoding="utf-8") as file:
        hparams = yaml.full_load(file)

    assert hparams["out_dir"] == str(tmp_path)
    assert hparams["foo"] is True
    assert hparams["bar"] == 1


def _test_function2(out_dir: Path, foo: bool = False, bar: int = 1):
    assert False, "I only exist as a signature, but I should not run."


@pytest.mark.parametrize(
    "command",
    [
        "any.py",
        "litgpt pretrain",
        "litgpt generate",
    ],
)
def test_save_hyperparameters_known_commands(command, tmp_path):
    with mock.patch("sys.argv", [*command.split(" "), str(tmp_path), "--foo", "True"]):
        save_hyperparameters(_test_function2, tmp_path)

    with open(tmp_path / "hyperparameters.yaml", encoding="utf-8") as file:
        hparams = yaml.full_load(file)

    assert hparams["out_dir"] == str(tmp_path)
    assert hparams["foo"] is True
    assert hparams["bar"] == 1


def test_choose_logger(tmp_path):
    assert isinstance(choose_logger("csv", out_dir=tmp_path, name="csv"), CSVLogger)
    if _TENSORBOARD_AVAILABLE:
        assert isinstance(choose_logger("tensorboard", out_dir=tmp_path, name="tb"), TensorBoardLogger)
    if _WANDB_AVAILABLE:
        assert isinstance(choose_logger("wandb", out_dir=tmp_path, name="wandb"), WandbLogger)
    if _MLFLOW_AVAILABLE or _MLFLOW_SKINNY_AVAILABLE:
        assert isinstance(choose_logger("mlflow", out_dir=tmp_path, name="wandb"), MLFlowLogger)
    if _LITLOGGER_AVAILABLE:
        assert isinstance(choose_logger("litlogger", out_dir=tmp_path, name="litlogger"), LitLogger)
    with pytest.raises(ValueError, match="`--logger_name=foo` is not a valid option."):
        choose_logger("foo", out_dir=tmp_path, name="foo")


@pytest.mark.parametrize(
    "path_type, input_path, expected",
    [
        ("relative", "some/relative/path", "some/relative/path"),
        ("absolute", "/usr/absolute/path", "/usr/absolute/path"),
        ("env_relative", "some/relative/path", "prefix/some/relative/path"),
        ("env_absolute", "/usr/absolute/path", "/usr/absolute/path"),
    ],
)
def test_init_out_dir(path_type, input_path, expected):
    if path_type.startswith("env_"):
        with mock.patch.dict(os.environ, {"LIGHTNING_ARTIFACTS_DIR": "prefix"}):
            result = init_out_dir(input_path)
            assert result == Path(expected), f"Failed for {path_type} with input {input_path} (result {result})"
    else:
        result = init_out_dir(input_path)
        if "LIGHTNING_ARTIFACTS_DIR" not in os.environ:
            assert result == Path(expected), f"Failed for {path_type} with input {input_path} (result {result})"
        else:
            assert result == Path(os.getenv("LIGHTNING_ARTIFACTS_DIR")) / expected, (
                f"Failed for {path_type} with input {input_path} (result {result})"
            )


def test_find_resume_path(tmp_path):
    assert find_resume_path(resume=None, out_dir=Path("does/not/exist")) is None
    assert find_resume_path(resume=Path("does/not/exist"), out_dir=Path("does/not/matter")) == Path("does/not/exist")
    assert find_resume_path(resume=(tmp_path / "checkpoint.pt"), out_dir=Path("does/not/matter")) == (
        tmp_path / "checkpoint.pt"
    )

    # `resume='auto'` does not enforce the checkpoint to exist
    assert find_resume_path(resume="auto", out_dir=Path("does/not/exist")) is None

    # `resume=True` requires a checkpoint to exist
    with pytest.raises(FileNotFoundError, match="You passed `--resume=True`, but no checkpoint file was found"):
        find_resume_path(resume=True, out_dir=Path("does/not/exist"))
    with pytest.raises(FileNotFoundError, match="You passed `--resume=True`, but no checkpoint file was found"):
        find_resume_path(resume=True, out_dir=tmp_path)

    (tmp_path / "step-001").mkdir()
    (tmp_path / "step-001" / "lit_model.pth").touch()
    (tmp_path / "step-002").mkdir()
    (tmp_path / "step-002" / "lit_model.pth").touch()
    (tmp_path / "step-003").mkdir()
    (tmp_path / "step-003" / "lit_model.pth").touch()

    assert find_resume_path(resume=True, out_dir=tmp_path) == (tmp_path / "step-003" / "lit_model.pth")
    assert find_resume_path(resume="auto", out_dir=tmp_path) == (tmp_path / "step-003" / "lit_model.pth")


@pytest.fixture
def model_parameters():
    return [torch.nn.Parameter(torch.randn(2, 2))]


def test_instantiate_torch_optimizer_with_str(model_parameters):
    optimizer = instantiate_torch_optimizer("Adam", model_parameters, lr=0.01)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_instantiate_torch_optimizer_with_class(model_parameters):
    optimizer = instantiate_torch_optimizer(
        {"class_path": "torch.optim.Adam", "init_args": {"lr": 123}}, model_parameters, lr=0.02
    )
    assert isinstance(optimizer, torch.optim.Adam)
    # init args gets overridden
    assert optimizer.param_groups[0]["lr"] == 0.02


@pytest.mark.parametrize(
    "input_path, expected",
    [
        (Path("checkpoints/my_model"), Path("checkpoints/my_model")),
        (Path("checkpoints/my_model"), Path("./checkpoints/my_model")),
    ],
)
def test_extend_checkpoint_dir_is_prefixed(input_path, expected):
    original_dir = Path.cwd()  # Save the current directory
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        try:
            if not input_path.is_absolute():
                input_path = Path(tmp_dir) / input_path
            if not expected.is_absolute():
                expected = Path(tmp_dir) / expected
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.touch(exist_ok=True)
            assert extend_checkpoint_dir(input_path) == expected
        finally:
            os.chdir(original_dir)  # Reset the current directory


@pytest.mark.parametrize(
    "input_path, expected",
    [
        (Path("my_model"), Path("checkpoints/my_model")),
        (Path("my_model"), Path("./checkpoints/my_model")),
    ],
)
def test_extend_checkpoint_dir(input_path, expected):
    original_dir = Path.cwd()  # Save the current directory
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)

        try:
            if not input_path.is_absolute():
                input_path = Path(tmp_dir) / "checkpoints" / input_path
            if not expected.is_absolute():
                expected = Path(tmp_dir) / expected
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.touch(exist_ok=True)
            assert extend_checkpoint_dir(input_path) == expected
        finally:
            os.chdir(original_dir)  # Reset the current directory


@pytest.mark.parametrize(
    "input_path, expected",
    [
        (Path("my_model"), Path("my_model")),
        (Path("/my_model"), Path("/my_model")),
    ],
)
def test_extend_checkpoint_dir_dont_exist(input_path, expected):
    assert extend_checkpoint_dir(input_path) == expected


@pytest.fixture
def mock_cuda_is_available_true(monkeypatch):
    """Fixture to mock torch.cuda.is_available() to return True."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture
def mock_nvidia_device_properties(monkeypatch):
    """Fixture to mock torch.cuda.get_device_properties() for NVIDIA GPUs."""
    mock_device_properties = mock.MagicMock(name="GPU Device", spec=["name"])
    mock_device_properties.name = "NVIDIA RTX A6000"
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: mock_device_properties)


@pytest.fixture
def mock_amd_device_properties(monkeypatch):
    """Fixture to mock torch.cuda.get_device_properties() for AMD GPUs."""
    mock_device_properties = mock.MagicMock(name="GPU Device", spec=["name"])
    mock_device_properties.name = "AMD Instinct MI250X"
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: mock_device_properties)


@pytest.fixture
def all_nvlink_connected_output():
    return mock.MagicMock(
        stdout="""        GPU0	GPU1	GPU2	GPU3
GPU0	X	NV12	NV12	NV12
GPU1	NV12	X	NV12	NV12
GPU2	NV12	NV12	X	NV12
GPU3	NV12	NV12	NV12	X""",
        returncode=0,
    )


@mock.patch("subprocess.run")
def test_all_nvlink_connected(
    mock_run, all_nvlink_connected_output, mock_cuda_is_available_true, mock_nvidia_device_properties
):
    mock_run.return_value = all_nvlink_connected_output
    with mock.patch("builtins.print") as mock_print:
        check_nvlink_connectivity()
        mock_print.assert_any_call("All GPUs are fully connected via NVLink.")


@pytest.fixture
def nvlink_partially_connected_output():
    return mock.MagicMock(
        stdout="""        GPU0    GPU1    GPU2    GPU3    CPU Affinity
GPU0     X      NV1     SYS     SYS     0-7
GPU1    NV1      X      SYS     SYS     0-7
GPU2    SYS     SYS      X      NV1     8-15
GPU3    SYS     SYS     NV1      X      8-15

Legend:
  X   = Self
  NV1 = Connected via NVLink with 1 hop
  SYS = Connected via the PCIe or CPU subsystem""",
        returncode=0,
    )


@mock.patch("subprocess.run")
def test_nvlink_partially_connected_output(
    mock_run, nvlink_partially_connected_output, mock_cuda_is_available_true, mock_nvidia_device_properties
):
    mock_run.return_value = nvlink_partially_connected_output
    with mock.patch("builtins.print") as mock_print:
        check_nvlink_connectivity()
        mock_print.assert_any_call(
            "Warning: Not all GPUs are fully connected via NVLink. Some GPUs are connected via slower interfaces. "
            "It is recommended to switch to a different machine with faster GPU connections for optimal multi-GPU training performance."
        )


@pytest.fixture
def nvlink_not_connected_output():
    return mock.MagicMock(
        stdout="""        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      PHB     PHB     PHB     0-47    0               N/A
GPU1    PHB      X      PHB     PHB     0-47    0               N/A
GPU2    PHB     PHB      X      PHB     0-47    0               N/A
GPU3    PHB     PHB     PHB      X      0-47    0               N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks""",
        returncode=0,
    )


@mock.patch("subprocess.run")
def test_nvlink_not_connected_output(
    mock_run, nvlink_not_connected_output, mock_cuda_is_available_true, mock_nvidia_device_properties
):
    mock_run.return_value = nvlink_not_connected_output
    with mock.patch("builtins.print") as mock_print:
        check_nvlink_connectivity()
        mock_print.assert_any_call(
            "Warning: Not all GPUs are fully connected via NVLink. Some GPUs are connected via slower interfaces. "
            "It is recommended to switch to a different machine with faster GPU connections for optimal multi-GPU training performance."
        )


@pytest.fixture
def nvlink_all_gpu_connected_but_other_connected_output():
    return mock.MagicMock(
        stdout="""	GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	NIC0	NIC1	NIC2	NIC3	NIC4	NIC5	NIC6	NIC7	NIC8	NIC9	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	X 	NV12	NV12	NV12	NV12	NV12	NV12	NV12	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	0-63,128-191	0		N/A
GPU1	NV12	X 	NV12	NV12	NV12	NV12	NV12	NV12	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	0-63,128-191	0		N/A
GPU2	NV12	NV12	X 	NV12	NV12	NV12	NV12	NV12	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	0-63,128-191	0		N/A
GPU3	NV12	NV12	NV12	X 	NV12	NV12	NV12	NV12	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	0-63,128-191	0		N/A
GPU4	NV12	NV12	NV12	NV12	X 	NV12	NV12	NV12	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	PXB	PXB	64-127,192-254	1		N/A
GPU5	NV12	NV12	NV12	NV12	NV12	X 	NV12	NV12	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	PXB	PXB	64-127,192-254	1		N/A
GPU6	NV12	NV12	NV12	NV12	NV12	NV12	X 	NV12	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	64-127,192-254	1		N/A
GPU7	NV12	NV12	NV12	NV12	NV12	NV12	NV12	X 	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	64-127,192-254	1		N/A
NIC0	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	X 	PIX	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS
NIC1	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	PIX	X 	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS
NIC2	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	X 	PXB	SYS	SYS	SYS	SYS	SYS	SYS
NIC3	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	PXB	X 	SYS	SYS	SYS	SYS	SYS	SYS
NIC4	SYS	SYS	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	X 	PXB	SYS	SYS	SYS	SYS
NIC5	SYS	SYS	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	PXB	X 	SYS	SYS	SYS	SYS
NIC6	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	X 	PIX	SYS	SYS
NIC7	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	PIX	X 	SYS	SYS
NIC8	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	X 	PXB
NIC9	SYS	SYS	SYS	SYS	PXB	PXB	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	SYS	PXB	X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
  NIC6: mlx5_6
  NIC7: mlx5_7
  NIC8: mlx5_8
  NIC9: mlx5_9

""",
        returncode=0,
    )


@mock.patch("subprocess.run")
def test_nvlink_all_gpu_connected_but_other_connected_output(
    mock_run,
    nvlink_all_gpu_connected_but_other_connected_output,
    mock_cuda_is_available_true,
    mock_nvidia_device_properties,
):
    mock_run.return_value = nvlink_all_gpu_connected_but_other_connected_output
    with mock.patch("builtins.print") as mock_print:
        check_nvlink_connectivity()
    mock_print.assert_any_call("All GPUs are fully connected via NVLink.")


@pytest.fixture
def nvidia_smi_nvlink_output_dual_gpu_no_numa():
    return mock.MagicMock(
        stdout="""
        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV1     0-15    0               N/A
GPU1    NV1      X      0-15    0               N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
    """,
        returncode=0,
    )


@mock.patch("subprocess.run")
def test_check_nvlink_connectivity__returns_fully_connected_when_nvidia_all_nvlink_two_gpus(
    mock_run, nvidia_smi_nvlink_output_dual_gpu_no_numa, mock_cuda_is_available_true, mock_nvidia_device_properties
):
    mock_run.return_value = nvidia_smi_nvlink_output_dual_gpu_no_numa
    with mock.patch("builtins.print") as mock_print:
        check_nvlink_connectivity()
        mock_print.assert_any_call("All GPUs are fully connected via NVLink.")


@pytest.fixture
def rocm_smi_xgmi_output_multi_gpu():
    """
    rocm-smi --showtopotype on ROCm 6.0.3+
    """
    return mock.MagicMock(
        stdout="""
=============================== ROCm System Management Interface ============================
=============================== Link Type between two GPUs ===============================
       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7
GPU0   0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI
GPU1   XGMI         0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI
GPU2   XGMI         XGMI         0            XGMI         XGMI         XGMI         XGMI         XGMI
GPU3   XGMI         XGMI         XGMI         0            XGMI         XGMI         XGMI         XGMI
GPU4   XGMI         XGMI         XGMI         XGMI         0            XGMI         XGMI         XGMI
GPU5   XGMI         XGMI         XGMI         XGMI         XGMI         0            XGMI         XGMI
GPU6   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         0            XGMI
GPU7   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         0
================================== End of ROCm SMI Log ===================================
    """,
        returncode=0,
    )


@mock.patch("subprocess.run")
def test_check_nvlink_connectivity__returns_fully_connected_when_amd_all_xgmi_8_gpus(
    mock_run, rocm_smi_xgmi_output_multi_gpu, mock_cuda_is_available_true, mock_amd_device_properties
):
    mock_run.return_value = rocm_smi_xgmi_output_multi_gpu
    with mock.patch("builtins.print") as mock_print:
        check_nvlink_connectivity()
        mock_print.assert_any_call("All GPUs are fully connected via XGMI.")


@mock.patch("subprocess.run")
def test_check_nvlink_connectivity__returns_no_gpus_when_no_gpus(mock_run, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with mock.patch("builtins.print") as mock_print:
        check_nvlink_connectivity()
        mock_print.assert_any_call("No GPUs available")


@mock.patch("subprocess.run")
def test_check_nvlink_connectivity__returns_unrecognized_vendor_when_unrecognized_vendor(
    mock_run, monkeypatch, mock_cuda_is_available_true
):
    mock_device_properties = mock.MagicMock(name="GPU Device", spec=["name"])
    mock_device_properties.name = "GARAGE DIY HYPERSCALER GPU"
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: mock_device_properties)
    with mock.patch("builtins.print") as mock_print:
        check_nvlink_connectivity()
        mock_print.assert_any_call("Unrecognized GPU vendor: GARAGE DIY HYPERSCALER GPU")

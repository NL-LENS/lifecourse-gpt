# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import gc
import sys
from pathlib import Path

import pytest
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
if wd.is_dir():
    sys.path.append(str(wd))


@pytest.fixture(autouse=True)
def reclaim_cuda_memory():
    """Free GPU memory after a test that actually used it, to avoid accumulation across CUDA tests.

    Gated on ``memory_allocated`` (a cheap counter read) so the expensive ``gc.collect`` is skipped
    for the many CPU-only tests that never touch the GPU.
    """
    yield
    if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0:
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def restore_default_dtype():
    # just in case
    torch.set_default_dtype(torch.float32)


@pytest.fixture(autouse=True)
def destroy_process_group():
    yield

    import torch.distributed

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

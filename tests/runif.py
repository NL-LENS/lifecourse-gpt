# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import pytest
from lightning.fabric.utilities.testing import _runif_reasons


def RunIf(**kwargs):
    """Skip a test unless the environment meets `kwargs`, as `lightning.fabric.utilities.testing._RunIf` reads them,
    e.g. `min_cuda_gpus=1` or `standalone=True`."""
    reasons, marker_kwargs = _runif_reasons(**kwargs)
    return pytest.mark.skipif(condition=len(reasons) > 0, reason=f"Requires: [{' + '.join(reasons)}]", **marker_kwargs)

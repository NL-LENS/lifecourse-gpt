# Derivative files are under copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and
# Economic Innovations (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Delphi sequence model architecture: model, likelihood, data layout and sampler.

Adapted from https://github.com/gerstung-lab/Delphi under the MIT License.

"""

from litgpt.life_events.delphi.data import DelphiDataset, DelphiLifeEvents, delphi_collate_fn
from litgpt.life_events.delphi.generate import generate_trajectories
from litgpt.life_events.delphi.loss import chunked_cross_entropy_exp_waiting_time
from litgpt.life_events.delphi.mask import build_tie_mask, last_visible_index
from litgpt.life_events.delphi.model import AgeEncoding, Delphi

__all__ = [
    "AgeEncoding",
    "Delphi",
    "DelphiDataset",
    "DelphiLifeEvents",
    "build_tie_mask",
    "chunked_cross_entropy_exp_waiting_time",
    "delphi_collate_fn",
    "generate_trajectories",
    "last_visible_index",
]

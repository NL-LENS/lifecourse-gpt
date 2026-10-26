# Derivative files are under copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and
# Economic Innovations (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Delphi's tie mask: causal attention that also hides events sharing the target's age.

Kept apart from `model.py` because both the data pipeline (which builds the mask in its
collate function) and the loss (which measures waiting times from it) need it without
needing the model.
"""

import torch

__all__ = ["build_tie_mask", "last_visible_index"]


# Adapted from https://github.com/gerstung-lab/Delphi/blob/main/model.py#L228 under the MIT License
def build_tie_mask(
    age: torch.Tensor,
    targets_age: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal attention mask that also hides events tied to the prediction target.

    Events recorded at the same age are exchangeable within the sequence, so attending
    to one of them while predicting another leaks the target. Query `i` is therefore
    not allowed to attend to key `j` whenever `age[j] == targets_age[i]`. Since the
    mask depends on the targets, it is only meaningful during training; at generation
    time there is no target to leak.

    Args:
        age: Age of each token, shape `(B, T)`. Ages are compared for exact equality,
            so pass them in an integer unit (days) rather than as converted floats.
        targets_age: Age of each token's prediction target, shape `(B, T)`.
        pad_mask: Optional. `True` at real tokens, shape `(B, T)`.

    Returns:
        Boolean mask, shape `(B, 1, T, T)`, `True` where attention is allowed, ready to
        be passed to `Delphi.forward` as `attn_mask`. Rows that would otherwise be empty
        fall back to attending to themselves, which keeps the softmax well defined. Such
        rows occur for padded queries, and for a query whose target shares its own age.
    """
    if age.shape != targets_age.shape:
        raise ValueError(f"age and targets_age must have equal shape, but have {age.shape} != {targets_age.shape}")

    B, T = age.shape
    mask = torch.ones(T, T, dtype=torch.bool, device=age.device).tril().expand(B, 1, T, T).clone()
    if pad_mask is not None:
        mask &= pad_mask[:, None, None, :] & pad_mask[:, None, :, None]
    # A key at the age we are trying to predict would give the target away.
    mask &= age[:, None, None, :] != targets_age[:, None, :, None]
    eye = torch.eye(T, dtype=torch.bool, device=age.device)
    mask |= (~mask.any(-1, keepdim=True)) & eye
    return mask


# Adapted from https://github.com/gerstung-lab/Delphi/blob/main/model.py#L265 under the MIT License
def last_visible_index(attn_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last key each query is allowed to attend to.

    Companion to `build_tie_mask`. Once tied events are hidden from a query, the most
    recent information that query actually has is the last key still visible to it, not
    its own position. The waiting-time loss has to measure from there, otherwise the
    attention mask and the likelihood disagree about what the model knows.

    Args:
        attn_mask: Boolean mask, shape `(B, 1, T, T)`, `True` where attention is
            allowed, as returned by `build_tie_mask`.

    Returns:
        Indices into the key axis, shape `(B, T)`. Because `build_tie_mask` falls back
        to self-attention for rows that would otherwise be empty, every row has at least
        one visible key and the result is always a position the query can really see.
    """
    T = attn_mask.size(-1)
    positions = torch.arange(T, device=attn_mask.device, dtype=torch.float32).view(1, 1, 1, -1)
    return (attn_mask * positions).max(-1).indices.squeeze(2).squeeze(1)

# Partially generated and refined with Claude Code (Opus 5).
"""The competing-risks likelihood Delphi is trained under."""

import torch

from litgpt.life_events.delphi.mask import last_visible_index

__all__ = ["chunked_cross_entropy_exp_waiting_time"]


def chunked_cross_entropy_exp_waiting_time(
    logits: torch.Tensor | list[torch.Tensor],
    age: torch.Tensor,
    targets: torch.Tensor,
    targets_age: torch.Tensor,
    pad_mask: torch.Tensor,
    t_min: float = 0.0,
    chunk_size: int = 128,
    attn_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Negative log-likelihood of a continuous-time competing-risks process.

    Logits are read as log event rates: token `k` fires with rate `exp(logits[k])` per
    unit of age, so the time to the *next* event of any type is exponential with the
    total rate `sum_k exp(logits[k])`. The likelihood of observing event `j` after a
    wait of `dt` then factorizes exactly:

        log(rate_j) - total_rate * dt
            = [log(rate_j) - log(total_rate)]  -> `loss_ce`, which event
            + [log(total_rate) - total_rate * dt]  -> `loss_dt`, when

    The first bracket is the usual cross entropy, because softmax and the normalized
    rates are the same thing. The second pins down the absolute scale of the logits,
    which cross entropy alone leaves free. Add the two with equal weight to recover the
    joint likelihood; any other weighting is no longer a likelihood.

    Both terms share the `logsumexp` over the vocabulary and are averaged over the same
    positions, so they are computed in one pass here rather than by calling
    `chunked_cross_entropy` separately.

    Note that the sum runs over `config.padded_vocab_size`. Unused vocabulary entries
    still contribute `exp(logit)` to the total rate, so with a vocabulary much smaller
    than its padding the dead entries dominate at initialization. Set
    `padded_vocab_size` to the real vocabulary size, or mask the unused logits.

    Args:
        logits: Log rates, shape `(B, T, V)`, or the list of chunks produced by
            `Delphi.forward` with `lm_head_chunk_size > 0`.
        age: Age of each input token, shape `(B, T)`, in the same unit as `targets_age`.
        targets: Index of the next event, shape `(B, T)`. Values at positions excluded
            by `pad_mask` are never read and may be an ignore index.
        targets_age: Age at which the next event occurs, shape `(B, T)`.
        pad_mask: `True` at positions that contribute to the loss, shape `(B, T)`.
            Both terms use it, which is what keeps their sum a likelihood over one and
            the same set of positions.
        t_min: Floor on the expected waiting time, in the unit of `age`. Caps the
            implied rate at `1 / t_min`, so ties and same-day events cannot push it to
            infinity. `0.0` disables it.
        chunk_size: Number of flattened positions per chunk. Chunking lets the backward
            pass free each chunk's gradient buffer as it goes, which caps the memory
            spike from the `(B * T, V)` intermediates on a large vocabulary. `0`
            computes the loss in a single chunk.
        attn_mask: Optional. The tie mask from `build_tie_mask`, shape `(B, 1, T, T)`.
            Pass it whenever the model was run with one. Hiding tied events from the
            attention without it leaves the loss measuring the wait from a position the
            model could not see, which for a run of same-day events is a wait of zero
            floored to one day -- exactly the degenerate signal the tie mask exists to
            remove. See `last_visible_index`.

    Returns:
        `{"loss_ce": ..., "loss_dt": ...}`, each a scalar averaged over `pad_mask`.
    """
    # See `chunked_cross_entropy` for why chunking bounds the backward memory spike.
    if isinstance(logits, list):
        logits = torch.cat(logits, dim=1)

    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    pad_mask = pad_mask.reshape(-1)
    # The minimum difference in age is 1 because simultaneous events are masked out.
    waiting_time = torch.clamp(targets_age - age, min=1)
    if attn_mask is not None:
        # Measure from the last event the query can actually see. For a query whose own
        # target is tied to it, that is the last token before the tied run, and the
        # gathered wait spans the gap up to that run.
        waiting_time = torch.gather(waiting_time, -1, last_visible_index(attn_mask))
    # Promoted to float32 rather than to the logits' dtype: ages run into the tens of
    # thousands of days, which bf16 cannot resolve to better than a few hundred.
    waiting_time = waiting_time.reshape(-1).to(torch.float32)
    # Excluded positions may carry an ignore index, which `gather` cannot take.
    safe_targets = targets.clamp(min=0).unsqueeze(-1)

    if chunk_size <= 0:
        chunk_size = max(logits.size(0), 1)

    ce_chunks, dt_chunks = [], []
    for logit_chunk, target_chunk, wait_chunk, mask_chunk in zip(
        logits.split(chunk_size),
        safe_targets.split(chunk_size),
        waiting_time.split(chunk_size),
        pad_mask.split(chunk_size),
    ):
        log_total_rate = torch.logsumexp(logit_chunk, dim=-1)
        target_log_rate = logit_chunk.gather(-1, target_chunk).squeeze(-1)

        log_effective_rate = -torch.log(torch.exp(-log_total_rate) + t_min)
        # Kept in log space: `exp(log_rate) * waiting_time` overflows where the rate is
        # high, which is exactly where the loss is largest and matters most.
        rate_times_wait = torch.exp(log_effective_rate + torch.log(wait_chunk + t_min))

        # `where` rather than boolean indexing: it keeps shapes static and avoids the
        # device synchronization that a data-dependent index would force.
        zero = torch.zeros((), device=logits.device, dtype=log_total_rate.dtype)
        ce_chunks.append(torch.where(mask_chunk, log_total_rate - target_log_rate, zero))
        dt_chunks.append(torch.where(mask_chunk, rate_times_wait - log_effective_rate, zero))

    non_masked_elems = pad_mask.sum()
    # See [non_masked_elems div note] in `chunked_cross_entropy`.
    denominator = non_masked_elems.maximum(torch.ones_like(non_masked_elems))
    return {
        "loss_ce": torch.cat(ce_chunks).sum() / denominator,
        "loss_dt": torch.cat(dt_chunks).sum() / denominator,
    }

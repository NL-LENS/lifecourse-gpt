# Derivative files are under copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and
# Economic Innovations (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Life-course event trajectory sampling for Delphi.

A text sampler draws a token from a softmax and leaves time implicit, because the next
token simply follows the last one. Here the model's logits are log event rates, so a
step has to produce *both* the next event and the age at which it happens. See `loss.chunked_cross_entropy_exp_waiting_time` for the
likelihood these samples come from.
"""

from collections.abc import Sequence

import torch

from litgpt.life_events.delphi.data import NO_EVENT_ID, PAD_AGE, PAD_ID
from litgpt.life_events.delphi.model import Delphi

__all__ = ["generate_trajectories"]


# Adapted from https://github.com/gerstung-lab/Delphi/blob/main/model.py#L356 under the MIT license
def generate_trajectories(
    model: Delphi,
    idx: torch.Tensor,
    age: torch.Tensor,
    max_new_tokens: int = 100,
    *,
    max_age: float = 85 * 365.25,
    termination_tokens: Sequence[int] | None = None,
    ignore_tokens: Sequence[int] = (PAD_ID,),
    no_repeat: bool = False,
    repeatable_tokens: Sequence[int] = (NO_EVENT_ID,),
    t_min: float | None = None,
    max_wait: float = 80 * 365.25,
    pad_id: int = PAD_ID,
    pad_age: float = PAD_AGE,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continues each trajectory by sampling events and the ages at which they occur.

    Each step samples from the competing-risks process the model was fitted to. Writing
    `rate_k = exp(logit_k)` and `total = sum_k rate_k`, the joint density factorizes into
    a choice of event and a waiting time that are independent of each other:

        which -> Categorical(softmax(logits))
        when  -> Exponential(total)

    Sampling the two separately is exactly equivalent to drawing one exponential per
    event type and keeping the earliest, which is how Delphi writes it: the minimum of
    independent exponentials is `Exponential(total)`, and the index attaining it is
    distributed as `softmax(logits)`. The factored form is used here because it extends
    to `t_min`, which caps the rate at `1 / t_min` and which the per-type form cannot
    express (the cap applies to the total rate, not to any individual one).

    Past events are served from a KV cache, so each step only runs the newly sampled
    event through the model. That is exact: ages enter only through the input embedding,
    so the keys and values of an event never change once it has been appended, and the
    plain causal mask of the cache is the right one because tie masking is deliberately
    absent here. The tie mask hides a token's own target, and at sampling time there is
    no target to leak. Any cache already set on `model` is cleared on the way out.

    Prompts are right-padded, as `delphi_collate_fn` builds them, and every trajectory
    continues from its own last real event. Each row writes new events at its own cache
    position, overwriting its padding one slot at a time, so no attention mask is needed
    to hide the padding: it always sits after the row's real events, where the causal
    mask keeps it out of sight until it has been overwritten.

    Args:
        model: A model whose logits are log event rates, in events per unit of `age`.
        idx: Conditioning tokens, shape `(B, T)`, right-padded with `pad_id`. Every row
            needs at least one real token, and padding may only fill its right end.
        age: Age of each conditioning token, shape `(B, T)`, in the unit the model was
            trained on (days). Values at padded positions are ignored.
        max_new_tokens: Maximum number of events to append per trajectory.
        max_age: Stop a trajectory once its age passes this.
        termination_tokens: Event ids that end a trajectory, typically death. A
            trajectory that emits one is padded from then on.
        ignore_tokens: Event ids that can never be sampled, such as padding. They keep
            contributing to the total rate during training but must not be emitted.
        no_repeat: Forbid sampling an event already present in the trajectory. Right for
            a vocabulary of one-off diagnoses, wrong for recurring events such as
            starting a job, so it is off by default.
        repeatable_tokens: Event ids exempt from `no_repeat`. The default exempts the
            no-event token, as Delphi does: it marks the passing of time and recurs every
            few years in training, so blocking it after its first occurrence would force
            quiet stretches to be filled with real events. Add recurring events here to
            use `no_repeat` with a mixed vocabulary.
        t_min: Floor on the expected waiting time. `None` takes `model.config.t_min`,
            which keeps sampling consistent with the likelihood the model was fitted to.
        max_wait: Clamp on a single sampled waiting time, guarding against a
            near-zero rate producing an absurd age.
        pad_id: Padding token, both in the prompt and after the end of a trajectory.
        pad_age: Age written at positions after the end of a trajectory. The default
            matches the batches of `delphi_collate_fn`.
        generator: Optional RNG for reproducible sampling.

    Returns:
        `(idx, age)`, each shape `(B, T + max_new_tokens)` however early sampling stops,
        so that batches from separate calls line up. Rows are right-padded like the
        batches of `delphi_collate_fn`: positions past the end of a trajectory hold
        `pad_id` and age `pad_age`. Both are ordinary tensors, not inference tensors, so
        sampled cohorts can be scored under autograd or edited in place afterwards.
    """
    if idx.shape != age.shape:
        raise ValueError(f"idx and age must have equal shape, but have {idx.shape} != {age.shape}")
    real = idx != pad_id
    lengths = real.sum(-1)
    if idx.size(1) == 0 or bool((lengths == 0).any()):
        raise ValueError("Need at least one conditioning token in every row to continue from")
    if not torch.equal(real, torch.arange(idx.size(1), device=idx.device) < lengths[:, None]):
        raise ValueError(f"Prompts must be right-padded: `pad_id={pad_id}` may only fill the end of a row")

    t_min = model.config.t_min if t_min is None else t_min
    terminators = torch.tensor(list(termination_tokens or ()), dtype=idx.dtype, device=idx.device)
    ignored = torch.tensor(list(ignore_tokens or ()), dtype=torch.int64, device=idx.device)
    repeatable = torch.tensor(list(repeatable_tokens or ()), dtype=torch.int64, device=idx.device)

    was_training = model.training
    model.eval()

    # `inference_mode` rather than `no_grad`: it also skips version-counter and view
    # tracking. The catch is that tensors born inside
    # it are inference tensors, which cannot later be saved for backward or mutated in
    # place -- both of which a sampled cohort is likely to face, being data to analyse
    # rather than text to print. Hence a context manager and a clone on the way out,
    # instead of a decorator over the whole function.
    try:
        with torch.inference_mode():
            idx, age = _sample(
                model,
                idx,
                age,
                lengths,
                max_new_tokens,
                max_age,
                terminators,
                ignored,
                no_repeat,
                repeatable,
                t_min,
                max_wait,
                pad_id,
                pad_age,
                generator,
            )
    finally:
        model.clear_kv_cache()
        if was_training:
            model.train()
    # Cloned outside the context above, which is what clears the inference flag.
    return idx.clone(), age.clone()


def _sample(
    model: Delphi,
    idx: torch.Tensor,
    age: torch.Tensor,
    lengths: torch.Tensor,
    max_new_tokens: int,
    max_age: float,
    terminators: torch.Tensor,
    ignored: torch.Tensor,
    no_repeat: bool,
    repeatable: torch.Tensor,
    t_min: float,
    max_wait: float,
    pad_id: int,
    pad_age: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The sampling loop of `generate_trajectories`.

    Arguments are checked there. `lengths` counts the real tokens in each row of `idx`.
    """
    B, T = idx.shape
    # Capped at the model's context, which the loop below refuses to run past. Checked
    # there rather than here so that trajectories which terminate early are unaffected
    # by a generous `max_new_tokens`.
    cache_length = min(T + max_new_tokens, model.max_seq_length)
    dtype = next(model.parameters()).dtype
    model.set_kv_cache(batch_size=B, max_seq_length=cache_length, device=idx.device, dtype=dtype)

    # Sampled events are written into the padding of each row, at `pos`, the position
    # after its last real event. Ages past the end of a row are reset on the way out.
    out_idx = torch.cat([idx, idx.new_full((B, max_new_tokens), pad_id)], dim=1)
    out_age = torch.cat([age, age.new_zeros((B, max_new_tokens))], dim=1)
    rows = torch.arange(B, device=idx.device)
    pos = lengths.clone()
    last_age = age[rows, pos - 1]

    # A trajectory is done once it has terminated or run past `max_age`. It stops
    # advancing, but keeps being forwarded so the batch stays rectangular: its last
    # event is fed again, which rewrites the same keys and values into the same slot,
    # and whatever is sampled for it is discarded.
    done = torch.isin(idx, terminators).any(-1) | (last_age > max_age)

    for step in range(max_new_tokens):
        if bool(done.all()):
            break

        if step == 0:
            # Prefill with the whole prompt, padding included; it lies ahead of every
            # real event, so no real query can see it.
            logits = model(idx, age=age, input_pos=torch.arange(T, device=idx.device), input_pos_maxp1=T)
            logits = logits[rows, pos - 1]
        else:
            input_pos_maxp1 = int(pos.max())
            if input_pos_maxp1 > model.max_seq_length:
                raise ValueError(
                    f"Cannot forward sequence of length {input_pos_maxp1}, max seq length is only {model.max_seq_length}."
                )
            input_pos = (pos - 1)[:, None]
            logits = model(
                out_idx.gather(1, input_pos),
                age=out_age.gather(1, input_pos),
                input_pos=input_pos,
                input_pos_maxp1=input_pos_maxp1,
            )[:, -1]
        logits = logits.float()
        if ignored.numel():
            logits.index_fill_(-1, ignored, -torch.inf)
        if no_repeat:
            # Also hits the padding of each row, which only ever holds `pad_id`.
            seen = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, out_idx, True)
            if repeatable.numel():
                seen.index_fill_(-1, repeatable, False)
            logits.masked_fill_(seen, -torch.inf)

        # Every admissible event has been used up: nothing left to sample.
        done = done | torch.isinf(logits).all(-1)
        # Keep the softmax and the logsumexp well defined for rows we are discarding.
        logits = torch.where(done[:, None], torch.zeros_like(logits), logits)

        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1, generator=generator).squeeze(-1)

        log_total_rate = torch.logsumexp(logits, dim=-1)
        log_rate = -torch.log(torch.exp(-log_total_rate) + t_min)
        uniform = torch.rand(log_rate.shape, device=logits.device, generator=generator)
        # Inverse CDF of Exponential(rate), evaluated as a product to avoid forming the
        # rate itself, which overflows where the model predicts a very short wait.
        wait = torch.clamp(-torch.log(uniform) * torch.exp(-log_rate), min=0.0, max=max_wait)
        if not age.is_floating_point():
            # Rounded, not truncated. Ages are integer days, and truncation would shave
            # up to a day off every single step, so trajectories would drift steadily
            # too fast over the dozens of events in a lifetime.
            wait = wait.round()
        next_age = last_age + wait.to(age.dtype)

        active = ~done
        out_idx[rows[active], pos[active]] = next_token[active].to(out_idx.dtype)
        out_age[rows[active], pos[active]] = next_age[active]
        last_age = torch.where(active, next_age, last_age)
        pos = pos + active

        done = done | (active & torch.isin(next_token, terminators)) | (last_age > max_age)

    past_end = torch.arange(out_idx.size(1), device=idx.device) >= pos[:, None]
    return out_idx, out_age.masked_fill(past_end, pad_age)

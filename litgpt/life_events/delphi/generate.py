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

from litgpt.life_events.delphi.data import PAD_ID
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
    t_min: float | None = None,
    max_wait: float = 80 * 365.25,
    pad_id: int = PAD_ID,
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

    No KV cache: the full sequence is re-run each step. The cache is keyed by position
    and would have to be re-derived from ages anyway, and `GPT.forward` refuses a custom
    `attn_mask` alongside `input_pos`. Tie masking is deliberately absent here because it
    hides a token's own target, and at sampling time there is no target to leak.

    Args:
        model: A model whose logits are log event rates, in events per unit of `age`.
        idx: Conditioning tokens, shape `(B, T)`. Every row must be non-empty.
        age: Age of each conditioning token, shape `(B, T)`, in the unit the model was
            trained on (days).
        max_new_tokens: Maximum number of events to append per trajectory.
        max_age: Stop a trajectory once its age passes this.
        termination_tokens: Event ids that end a trajectory, typically death. A
            trajectory that emits one is padded from then on.
        ignore_tokens: Event ids that can never be sampled, such as padding. They keep
            contributing to the total rate during training but must not be emitted.
        no_repeat: Forbid sampling an event already present in the trajectory. Right for
            a vocabulary of one-off diagnoses, wrong for recurring events such as
            starting a job, so it is off by default.
        t_min: Floor on the expected waiting time. `None` takes `model.config.t_min`,
            which keeps sampling consistent with the likelihood the model was fitted to.
        max_wait: Clamp on a single sampled waiting time, guarding against a
            near-zero rate producing an absurd age.
        pad_id: Token written at positions after a trajectory has terminated.
        generator: Optional RNG for reproducible sampling.

    Returns:
        `(idx, age)`, each shape `(B, T + n)`, with `n <= max_new_tokens`. Positions
        after a trajectory terminates hold `pad_id`, and their age repeats the age at
        which it terminated. Both are ordinary tensors, not inference tensors, so
        sampled cohorts can be scored under autograd or edited in place afterwards.
    """
    if idx.shape != age.shape:
        raise ValueError(f"idx and age must have equal shape, but have {idx.shape} != {age.shape}")
    if idx.size(1) == 0:
        raise ValueError("Need at least one conditioning token to continue from")

    t_min = model.config.t_min if t_min is None else t_min
    terminators = torch.tensor(list(termination_tokens or ()), dtype=idx.dtype, device=idx.device)
    ignored = torch.tensor(list(ignore_tokens or ()), dtype=torch.int64, device=idx.device)

    was_training = model.training
    model.eval()

    # `inference_mode` rather than `no_grad`: it also skips version-counter and view
    # tracking. The catch is that tensors born inside
    # it are inference tensors, which cannot later be saved for backward or mutated in
    # place -- both of which a sampled cohort is likely to face, being data to analyse
    # rather than text to print. Hence a context manager and a clone on the way out,
    # instead of a decorator over the whole function.
    with torch.inference_mode():
        idx, age = _sample(
            model,
            idx,
            age,
            max_new_tokens,
            max_age,
            terminators,
            ignored,
            no_repeat,
            t_min,
            max_wait,
            pad_id,
            generator,
        )

    if was_training:
        model.train()
    # Cloned outside the context above, which is what clears the inference flag.
    return idx.clone(), age.clone()


def _sample(
    model: Delphi,
    idx: torch.Tensor,
    age: torch.Tensor,
    max_new_tokens: int,
    max_age: float,
    terminators: torch.Tensor,
    ignored: torch.Tensor,
    no_repeat: bool,
    t_min: float,
    max_wait: float,
    pad_id: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The sampling loop of `generate_trajectories`.

    Arguments are checked here.
    """
    # A trajectory is done once it has terminated or run past `max_age`; it keeps being
    # forwarded so the batch stays rectangular, but its samples are discarded.
    done = torch.isin(idx, terminators).any(-1) | (age[:, -1] > max_age)

    for _ in range(max_new_tokens):
        if bool(done.all()):
            break

        logits = model(idx, age=age)[:, -1, :].float()
        if ignored.numel():
            logits.index_fill_(-1, ignored, -torch.inf)
        if no_repeat:
            logits.scatter_(-1, idx, -torch.inf)

        # Every admissible event has been used up: nothing left to sample.
        done = done | torch.isinf(logits).all(-1)
        # Keep the softmax and the logsumexp well defined for rows we are discarding.
        logits = torch.where(done[:, None], torch.zeros_like(logits), logits)

        next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1, generator=generator)

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
        next_age = age[:, -1] + wait.to(age.dtype)

        idx = torch.cat([idx, torch.where(done[:, None], pad_id, next_token)], dim=1)
        age = torch.cat([age, torch.where(done, age[:, -1], next_age)[:, None]], dim=1)

        done = done | torch.isin(idx[:, -1], terminators) | (age[:, -1] > max_age)

    return idx, age

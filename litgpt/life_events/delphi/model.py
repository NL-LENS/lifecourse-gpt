# Partially generated and refined with Claude Code (Opus 5).
"""Delphi: a GPT over life-course events that conditions on the age of every event.

Adapted from https://github.com/gerstung-lab/Delphi/blob/main/model.py under the MIT
license. Relative to `litgpt.model.GPT` it adds two things to the input embedding -- a
sinusoidal encoding of each event's age and a dropout over whole event embeddings --
and trains under the competing-risks likelihood in `loss.py` instead of plain cross
entropy. Everything else (blocks, attention, head) is the stock GPT.
"""

import math
import warnings

import torch
import torch.nn as nn

from litgpt.config import Config
from litgpt.life_events.delphi.loss import chunked_cross_entropy_exp_waiting_time
from litgpt.model import GPT

__all__ = ["AgeEncoding", "Delphi"]


# Adapted from https://github.com/gerstung-lab/Delphi/blob/main/model.py#L119 under the MIT License
class AgeEncoding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(config.age_base) / config.n_embd))
        self.register_buffer("div_term", div_term)
        self.n_embd = config.n_embd
        self.linear = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode age using sine and cosine basis functions.

        Args:
            x: Age tensor (in days, must be integer).

        """
        B, T, _ = x.size()  # batch size, sequence length, 1

        x = x / 365.25 * self.div_term  # Convert from days into years

        y = torch.zeros(B, T, self.n_embd, device=x.device, dtype=x.dtype)
        y[..., 0::2] = torch.sin(x)
        y[..., 1::2] = torch.cos(x)

        return self.linear(y)  # batch size, sequence length, embed_dim


class Delphi(GPT):
    """Age-conditioned GPT over event tokens.

    Select it with `model_class_name="Delphi"` in the model config, as `delphi-2m` does.
    Pair it with `litgpt.life_events.delphi.data.DelphiLifeEvents`, whose batches carry
    exactly the keys `model_inputs` and `compute_loss` read.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if config.rotary_percentage > 0:
            warnings.warn(
                f"{type(self).__name__} is configured with rotary_percentage={config.rotary_percentage}. "
                "The age encoding is meant to be the only ordering signal; with RoPE on as well, the model "
                "can fit the index of an event in the sequence instead of its age."
            )
        self.transformer.wae = AgeEncoding(config)
        self.transformer.token_drop = nn.Dropout(config.token_dropout)

    def embed(self, idx: torch.Tensor, age: torch.Tensor | None = None) -> torch.Tensor:
        """Token embeddings plus the encoding of each token's age.

        Args:
            idx: Token indices, shape `(B, T)`.
            age: Optional. Age of each token in integer days, shape `(B, T)`. Without it
                the model sees no timing at all, which is only useful for debugging.
        """
        if age is not None and idx.shape != age.shape:
            raise ValueError(f"Token idx and age must have equal shape, but have {idx.shape} != {age.shape}")
        x = super().embed(idx)
        # Dropped before the age encoding is added, so a dropped event still leaves its
        # timing visible. A no-op unless `config.token_dropout` is set.
        x = self.transformer.token_drop(x)
        if age is not None:
            # We need shape (B, T, 1) here for broadcasting in the encoding
            x = x + self.transformer.wae(age.unsqueeze(-1))
        return x

    @staticmethod
    def model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"idx": batch["input_ids"], "age": batch["age"], "attn_mask": batch.get("attn_mask")}

    @staticmethod
    def compute_loss(
        logits: torch.Tensor | list[torch.Tensor], batch: dict[str, torch.Tensor], config: Config
    ) -> dict[str, torch.Tensor]:
        """`loss` is the sum of `loss_ce` and `loss_dt` with equal weight.

        That is not a tunable trade-off: the sum is exactly the negative log-likelihood of
        the competing-risks process, factored into which event occurs and when. Any other
        weighting stops being a likelihood. See `chunked_cross_entropy_exp_waiting_time`.
        """
        losses = chunked_cross_entropy_exp_waiting_time(
            logits,
            age=batch["age"],
            targets=batch["targets"],
            targets_age=batch["targets_age"],
            pad_mask=batch["pad_mask"],
            t_min=config.t_min,
            attn_mask=batch.get("attn_mask"),
            target_vocab_size=config.target_vocab_size,
        )
        return {"loss": losses["loss_ce"] + losses["loss_dt"], **losses}

    @staticmethod
    def example_batch(batch_size: int, seq_length: int, config: Config) -> dict[str, torch.Tensor]:
        # Ages feed the age encoding, the tie mask and both loss terms, so all three have
        # to be present for a FLOP count to reflect the real step. The values are
        # irrelevant on a meta device; only shapes and dtypes are.
        x = torch.randint(0, 1, (batch_size, seq_length))
        age = torch.zeros_like(x)
        return {
            "input_ids": x,
            "targets": x,
            "age": age,
            "targets_age": age,
            "pad_mask": torch.ones_like(x, dtype=torch.bool),
            "attn_mask": torch.ones(batch_size, 1, seq_length, seq_length, dtype=torch.bool),
        }

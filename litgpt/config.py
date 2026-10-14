# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from typing_extensions import Self


def find_multiple(n: int, k: int) -> int:
    """Utility function for finding the nearest value to n which is a multiple of k.

    NOTE: We define this function in this module rather than `litgpt.utils` so that users can import
    this file to do configuration manipulations in Python environments which do not include all the dependencies
    demanded by `litgpt.utils`.
    """
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class Config:
    name: str = ""
    # The `litgpt.model.GPT` subclass to build, see `litgpt.model.model_class`: a short
    # name such as "GPT" or "Delphi", or a dotted import path.
    model_class_name: str = "GPT"
    # General size parameters
    block_size: int = 4096
    n_layer: int = 16
    n_embd: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: int | None = None
    # Transformer block (structure, normalizations)
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    norm_qk: bool = False
    norm_qk_type: Literal["default", "olmo2"] = "default"
    post_attention_norm: bool = False
    post_mlp_norm: bool = False
    parallel_residual: bool = True
    shared_attention_norm: bool = False
    # Transformer block (self-attention)
    n_head: int = 32
    head_size: int | None = None
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: int | None = None
    attn_bias: bool = False
    attention_scores_scalar: int | None = None
    # If `sliding_window_size` is given, sliding window attention with this
    # size is used in layers where `sliding_window_indices` has a 1. The
    # default is all 1, so that sliding window attention is used in all
    # layers. If `len(sliding_window_indices) > n_layer`, we only use the
    # initial part.
    sliding_window_size: int | None = None
    sliding_window_indices: list[int] | None = None
    # if `attention_logit_softcapping` is used, cannot use optimized
    # `torch.nn.functional.scaled_dot_product_attention` (which implements
    # Flash attention), may result in higher memory and runtime footprint.
    attention_logit_softcapping: float | None = None
    # Rotary position embedding (RoPE)
    rope_base: int = 10000
    rotary_percentage: float = 0.25
    rope_condense_ratio: int = 1
    rope_adjustments: dict | None = None
    rope_interleave: bool = False
    # Transformer block (MLP)
    intermediate_size: int | None = None
    moe_intermediate_size: int | None = None
    bias: bool = True
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = "GptNeoxMLP"
    gelu_approximate: str = "none"
    n_expert: int = 0
    n_shared_expert: int | None = None
    n_expert_groups: int | None = None
    n_topk_groups: int | None = None
    n_topk_scores_per_group: int | None = None
    n_expert_per_token: int = 0
    first_k_dense_replace: int | None = None
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = False
    # GPT before/after blocks
    scale_embeddings: bool = False
    lm_head_bias: bool = False
    final_logit_softcapping: float | None = None
    norm_1: bool = True
    norm_2: bool = True
    latent_attention: dict | None = None
    # The base period of the RoPE embeddings for local attention.
    # If not provided, `rope_base` will be used for both local and global attention.
    rope_local_base_freq: float | None = None
    # If provided, must have `>= n_layer` entries, either 0 or 1. For 0,
    # `rope_base` is used, for 1 `rope_local_base_freq` is used. If
    # `len(rope_indices) > n_layer`, we only use the initial part.
    rope_indices: list[int] | None = None
    # Life-course (Delphi) models. Only read by `litgpt.life_events.delphi.model.Delphi`.
    # Age encoding: base of the geometric sequence of sinusoidal frequencies used
    # to encode a token's age. Analogous to `rope_base`, but the encoded quantity is
    # a continuous age in `age_unit` rather than an integer position.
    age_base: int = 10000
    # Unit the age is measured in before it is encoded: "days", "weeks", "months" or
    # "years". The fastest sinusoid has a period of 2*pi units and the slowest one of
    # 2*pi*age_base units, so the unit sets the shortest cycle the encoding can resolve:
    # about 6.3 years with "years", too slow for anything seasonal, and about 6.3 months
    # with "months". Ages always arrive in days; only the encoding's scale changes.
    age_unit: Literal["days", "weeks", "months", "years"] = "years"
    # Time-to-event loss: floor on the expected waiting time, in the same unit as `age`
    # (days). Caps the implied event rate at `1 / t_min`, so that tied or same-day events
    # cannot drive it to infinity. `0.0` disables it. See
    # `litgpt.life_events.delphi.loss.chunked_cross_entropy_exp_waiting_time`.
    t_min: float = 0.0
    # Dropout applied to the token embeddings only, before the age encoding is added.
    # Distinct from the residual dropout: it drops whole event embeddings while leaving
    # the age signal intact, so the model cannot lean on any single event.
    token_dropout: float = 0.0
    # Number of leading vocabulary entries that can be targets, i.e. possible outcomes of
    # the time-to-event process. Only their logits enter the total event rate, in the loss
    # and when sampling; the rest -- static person-level tokens, which the data module
    # places last, and any padding up to `padded_vocab_size` -- can be read but never occur.
    # `None` means `vocab_size`. `litgpt.pretrain` sets it from a life-events data module.
    target_vocab_size: int | None = None

    def __post_init__(self):
        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        if self.target_vocab_size is None:
            self.target_vocab_size = self.vocab_size
        elif not 0 < self.target_vocab_size <= self.vocab_size:
            raise ValueError(
                f"target_vocab_size must be in (0, vocab_size={self.vocab_size}], got {self.target_vocab_size}"
            )

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError(f"The config {self.name!r}, needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

        if self.sliding_window_size is not None:
            self.sliding_window_indices = check_indicator_and_length(
                self.sliding_window_indices,
                name="sliding_window_indices",
                required_length=self.n_layer,
            )

        if self.rope_local_base_freq is not None:
            self.rope_indices = check_indicator_and_length(
                self.rope_indices,
                name="rope_indices",
                required_length=self.n_layer,
            )

        if self.latent_attention is not None:
            self.q_lora_rank = self.latent_attention.get("q_lora_rank")
            self.kv_lora_rank = self.latent_attention.get("kv_lora_rank")
            self.qk_rope_head_dim = self.latent_attention.get("qk_rope_head_dim")
            self.qk_nope_head_dim = self.latent_attention.get("qk_nope_head_dim")
            self.v_head_dim = self.latent_attention.get("v_head_dim")
            assert (
                self.q_lora_rank
                and self.kv_lora_rank
                and self.qk_rope_head_dim
                and self.qk_nope_head_dim
                and self.v_head_dim
            ) is not None
            assert self.n_head == self.n_query_groups, "Latent attention does not support MQA/GQA"
            self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
            self.rope_n_elem = self.qk_rope_head_dim
        if self.first_k_dense_replace is not None:
            assert self.mlp_class_name == "LLaMAMoE"
        if self.n_expert_groups is not None:
            assert self.n_expert % self.n_expert_groups == 0 and self.n_expert_groups > 1
            assert self.n_topk_groups is not None
            experts_per_group = self.n_expert // self.n_expert_groups
            assert self.n_topk_scores_per_group is not None and self.n_topk_scores_per_group <= experts_per_group

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self | None:
        if name not in name_to_config:
            raise ValueError(f"{name!r} is not a supported config name, expected one of {list(name_to_config)}")

        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            file_kwargs = yaml.safe_load(fp)
            if file_kwargs is None:
                raise ValueError(f"{path} is empty which is likely unexpected.")
        file_kwargs.update(kwargs)
        return cls(**file_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Path, **kwargs: Any) -> Self:
        """Automatically load `model_config.yaml` and if it doesn't exist - a matching config from `litgpt/config.py`."""
        if (config_path := path / "model_config.yaml").is_file():
            return cls.from_file(config_path, **kwargs)
        if (model_name := path.name) in name_to_config:
            return cls.from_name(model_name, **kwargs)
        raise FileNotFoundError(f"For {str(path)!r} neither 'model_config.yaml' nor matching config exists.")

    @property
    def mlp_class(self) -> type:
        # `self.mlp_class_name` cannot be the type to keep the config serializable
        import litgpt.model

        return getattr(litgpt.model, self.mlp_class_name)

    @property
    def norm_class(self) -> type:
        # `self.norm_class_name` cannot be the type to keep the config serializable

        import torch  # Torch import is lazy to make config loading faster

        if self.norm_class_name == "RMSNorm":
            from litgpt.model import RMSNorm

            return RMSNorm

        return getattr(torch.nn, self.norm_class_name)


def check_indicator_and_length(
    params: list[int] | None,
    name: str,
    required_length: int,
    use_initial_part: bool = True,
    def_val: int = 1,
) -> list[int]:
    if params is None:
        return [def_val] * required_length
    if len(params) != required_length:
        if use_initial_part and len(params) > required_length:
            params = params[:required_length]
        else:
            raise ValueError(f"{name} = {params}, must have length {required_length}")
    if not set(params).issubset({0, 1}):
        raise ValueError(f"{name} = {params}, must only contain 0 and 1")
    return params


######################################
# Life-course event models (Delphi)
######################################
# Adapted from https://github.com/gerstung-lab/Delphi/blob/main/model.py under the MIT license
#
# `rotary_percentage=0.0` is important here because Delphi carries no positional
# embedding at all and lets the age encoding be the only ordering signal. Events arrive
# at irregular times, so a token's index in the sequence ("the third event") says much
# less than its age ("at 31.4 years"), and encoding both makes the model fit the index.
# Anything built on `AgeEncoding` should disable RoPE for the same reason. The model
# itself lives in `litgpt.life_events.delphi`.
#
# `vocab_size` and `padded_vocab_size` are placeholders: `litgpt.pretrain` replaces both
# with the vocabulary size of the life-events data module it trains on.
configs = [
    dict(
        name="delphi-2m",
        model_class_name="Delphi",
        block_size=48,
        vocab_size=1270,
        padded_vocab_size=1270,
        n_layer=12,
        n_head=12,
        n_embd=120,
        rotary_percentage=0.0,
        parallel_residual=False,
        bias=True,
        norm_class_name="LayerNorm",
        mlp_class_name="GptNeoxMLP",
        intermediate_size=480,
        age_unit="years",
        t_min=0.0,
        token_dropout=0.0,
    ),
]

name_to_config = {config["name"]: config for config in configs}

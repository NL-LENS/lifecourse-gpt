# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch

from litgpt import Config
from litgpt.model import MultiheadLatentAttention


@torch.inference_mode()
def test_multihead_latent_attention_kv_cache():
    """Test KV cache functionality"""
    config = Config(
        block_size=32,
        n_embd=64,
        n_head=4,
        n_query_groups=4,
        head_size=16,
        latent_attention={
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 16,
        },
    )

    mla = MultiheadLatentAttention(config, block_idx=0)

    # Build KV cache
    kv_cache = mla.build_kv_cache(batch_size=2, max_seq_length=32, device=torch.device("cpu"), dtype=torch.float32)

    # Check cache shapes
    assert kv_cache.k.shape == (2, config.n_head, 32, config.qk_head_dim)
    assert kv_cache.v.shape == (2, config.n_head, 32, config.v_head_dim)


@torch.inference_mode()
def test_multihead_latent_attention_with_mask():
    """Test attention with causal mask"""
    config = Config(
        n_embd=64,
        n_head=4,
        n_query_groups=4,
        head_size=16,
        latent_attention={
            "q_lora_rank": 32,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 16,
        },
    )

    mla = MultiheadLatentAttention(config, block_idx=0)

    batch_size, seq_len = 1, 8
    x = torch.randn(batch_size, seq_len, config.n_embd)
    cos = torch.randn(1, seq_len, config.qk_rope_head_dim)
    sin = torch.randn(1, seq_len, config.qk_rope_head_dim)

    # Create causal mask
    mask = torch.ones(seq_len, seq_len, dtype=x.dtype).triu(diagonal=1)
    mask.masked_fill_(mask.bool(), float("-inf"))
    mask = mask.view(1, 1, seq_len, seq_len)

    # Forward pass with mask
    output = mla(x, cos, sin, mask=mask)

    assert output.shape == (batch_size, seq_len, config.n_embd)


def sync_weights(litgpt_model, hf_model):
    """Copies weights from lit-gpt model to HF model."""
    print("Synchronizing weights...")
    with torch.no_grad():
        hf_model.q_a_proj.weight.copy_(litgpt_model.q_a_proj.weight)
        hf_model.q_a_layernorm.weight.copy_(litgpt_model.q_a_norm.weight)
        hf_model.q_b_proj.weight.copy_(litgpt_model.q_b_proj.weight)
        hf_model.kv_a_proj_with_mqa.weight.copy_(litgpt_model.kv_a_proj_with_mqa.weight)
        hf_model.kv_a_layernorm.weight.copy_(litgpt_model.kv_a_norm.weight)
        hf_model.kv_b_proj.weight.copy_(litgpt_model.kv_b_proj.weight)
        hf_model.o_proj.weight.copy_(litgpt_model.proj.weight)
    print("Synchronization complete.")

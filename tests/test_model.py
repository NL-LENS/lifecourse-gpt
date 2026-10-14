# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

from copy import deepcopy
from functools import partial
from unittest import mock

import pytest
import torch
from lightning import Fabric
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.fabric.utilities.init import _materialize_meta_tensors
from runif import RunIf
from torch._dynamo.backends import debugging
from torch.backends.cuda import (
    SDPAParams,
    SDPBackend,
    can_use_efficient_attention,
    can_use_flash_attention,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
)

import litgpt.config as config_module
from litgpt import GPT, Config
from litgpt.model import CausalSelfAttention, batched_index_copy_
from litgpt.model import qkv_reassemble as make_qkv_interleaved

# Upstream's `pythia-14m`: a small GPT-NeoX with partial RoPE, which `litgpt.config` no longer ships.
PYTHIA_14M = dict(name="pythia-14m", block_size=512, n_layer=6, n_embd=128, n_head=4, padding_multiple=128)


@RunIf(dynamo=True)
@torch.inference_mode()
def test_model_compile():
    model = GPT(Config(**{**PYTHIA_14M, "n_layer": 3}))
    x = torch.randint(model.config.vocab_size, size=(2, model.config.block_size), dtype=torch.int64)

    explanation = torch._dynamo.explain(model)(x)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    model = GPT(model.config)
    model.set_kv_cache(2)
    input_pos = torch.arange(model.config.block_size)
    explanation = torch._dynamo.explain(model)(x, input_pos)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


@torch.inference_mode()
@pytest.mark.parametrize(
    "max_seq_length", (25, pytest.param(23, marks=pytest.mark.xfail(raises=IndexError, strict=True)))
)
@pytest.mark.flaky(reruns=5)
def test_kv_cache(max_seq_length):
    config = Config(block_size=25, padded_vocab_size=5, n_layer=2, n_head=2, n_embd=8)
    model = GPT(config)
    idx = torch.randint(0, model.config.padded_vocab_size, (1, 5))
    max_new_tokens = 20
    model.max_seq_length = max_seq_length
    model.set_kv_cache(1)

    def generate(logits):
        logits = logits[:, -1:]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.argmax(probs).unsqueeze(0).unsqueeze(0)

    x_no_cache = idx
    x_cache = idx
    input_pos = torch.arange(0, 5)
    for _ in range(max_new_tokens):
        logits_no_cache = model(x_no_cache[:, -max_seq_length:])
        out_no_cache = generate(logits_no_cache)

        logits_cache = model(x_cache, input_pos)
        out_cache = generate(logits_cache)

        torch.testing.assert_close(out_no_cache, out_cache, rtol=0, atol=0)

        x_no_cache = torch.cat((x_no_cache, out_no_cache), dim=1)
        x_cache = out_cache
        input_pos = input_pos[-1:] + 1


@torch.inference_mode()
def test_model_kv_cache_amp():
    config = Config(**{**PYTHIA_14M, "n_layer": 2})
    model = GPT(config)
    encoded = torch.arange(45)
    model.set_kv_cache(batch_size=1)
    with torch.autocast("cpu", torch.bfloat16):
        output = model(encoded.unsqueeze(0), encoded)
    assert output.dtype is torch.bfloat16


def test_rope_cache_length():
    config = Config(**{**PYTHIA_14M, "n_layer": 2})
    model = GPT(config)
    model.max_seq_length = 128

    rope_len = model.rope_cache_length()
    assert rope_len == config.rope_n_elem

    # Verify it works with set_kv_cache
    model.set_kv_cache(batch_size=1)
    assert model.transformer.h[0].attn.kv_cache is not None


# https://github.com/pytorch/pytorch/blob/ad3572a5d/torch/testing/_internal/common_cuda.py#L31-L34
SUPPORTS_FLASH_ATTENTION = (
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0) and not _IS_WINDOWS
)


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("config", deepcopy(config_module.configs), ids=[c["name"] for c in config_module.configs])
@torch.inference_mode()
def test_sdpa_choice(config):
    torch.set_default_dtype(torch.float16)

    def assert_sdpa_backend(original_fn, q, k, v, mask):
        # SDPAParams gained an additional argument in PyTorch 2.5
        args = []
        if hasattr(SDPAParams, "enable_gqa"):
            args.append(False)
        params = SDPAParams(q, k, v, mask, 0.0, True, *args)
        if expected is SDPBackend.FLASH_ATTENTION:
            assert flash_sdp_enabled(), "flash_sdp_enabled() is False"
            if config.sliding_window_size is None:
                assert can_use_flash_attention(params, True), "can_use_flash_attention(params, True) is False"
        elif expected is SDPBackend.EFFICIENT_ATTENTION:
            assert mem_efficient_sdp_enabled(), "mem_efficient_sdp_enabled() is False"
            assert can_use_efficient_attention(params, True), "can_use_efficient_attention(params, True) is False"
        elif expected is SDPBackend.MATH:
            assert math_sdp_enabled(), "math_sdp_enabled() is False"
        else:
            raise NotImplementedError
        return original_fn(q, k, v, mask)

    config["n_layer"] = 1
    config = config_module.Config(**config)

    try:
        with torch.device("cuda"):
            model = GPT(config)
            x = torch.randint(0, 10, (2, 16), dtype=torch.int32)
    except torch.cuda.OutOfMemoryError:
        # best effort, if the GPU can load it
        pytest.xfail()

    for h in model.transformer.h:
        h.attn.scaled_dot_product_attention = partial(assert_sdpa_backend, h.attn.scaled_dot_product_attention)

    if SUPPORTS_FLASH_ATTENTION:
        expected = SDPBackend.FLASH_ATTENTION
        with torch.backends.cuda.sdp_kernel(enable_mem_efficient=False):
            model(x)

    expected = SDPBackend.EFFICIENT_ATTENTION if config.head_size % 8 == 0 else SDPBackend.MATH
    with torch.backends.cuda.sdp_kernel(enable_flash=False):
        model(x)


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("config", deepcopy(config_module.configs), ids=[c["name"] for c in config_module.configs])
@torch.inference_mode()
def test_sdpa_choice_kv_cache(config):
    torch.set_default_dtype(torch.float16)

    def assert_sdpa_backend(original_fn, q, k, v, mask):
        # SDPAParams gained an additional argument in PyTorch 2.5
        args = []
        if hasattr(SDPAParams, "enable_gqa"):
            args.append(False)
        params = SDPAParams(q, k, v, mask, 0.0, True, *args)
        if expected is SDPBackend.FLASH_ATTENTION:
            assert flash_sdp_enabled()
            assert can_use_flash_attention(params, True)
        elif expected is SDPBackend.EFFICIENT_ATTENTION:
            assert mem_efficient_sdp_enabled()
            assert can_use_efficient_attention(params, True)
        elif expected is SDPBackend.MATH:
            assert math_sdp_enabled()
        else:
            raise NotImplementedError
        return original_fn(q, k, v, mask)

    config["n_layer"] = 1
    config = config_module.Config(**config)

    try:
        with torch.device("cuda"):
            model = GPT(config)
            model.max_seq_length = 1
            model.set_kv_cache(2)
            x = torch.randint(0, 10, (2, 1), dtype=torch.int32)
            input_pos = torch.tensor([0], dtype=torch.long)
    except torch.cuda.OutOfMemoryError:
        # best effort, if the GPU can load it
        pytest.xfail()

    for h in model.transformer.h:
        h.attn.scaled_dot_product_attention = partial(assert_sdpa_backend, h.attn.scaled_dot_product_attention)

    if SUPPORTS_FLASH_ATTENTION:
        # flash attention does not support an attention mask
        expected = SDPBackend.MATH
        with torch.backends.cuda.sdp_kernel(enable_mem_efficient=False):
            model(x, input_pos)

    expected = (
        SDPBackend.EFFICIENT_ATTENTION if config.head_size % 8 == 0 and config.n_query_groups != 1 else SDPBackend.MATH
    )
    with torch.backends.cuda.sdp_kernel(enable_flash=False):
        model(x, input_pos)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_rope_init_under_fsdp():
    """Check that the rope cache is properly initialized"""
    fabric = Fabric(devices=2, strategy="fsdp", accelerator="cuda")
    fabric.launch()

    with fabric.init_module(empty_init=True):
        model = GPT(Config(**{**PYTHIA_14M, "n_layer": 1}))
    assert model.cos.device.type == "meta"
    assert model.sin.device.type == "meta"

    model = fabric.setup(model)
    assert model.cos.device.type == "cuda"
    assert model.sin.device.type == "cuda"
    cos, sin = model.rope_cache(device=fabric.device)
    torch.testing.assert_close(model.cos, cos)
    torch.testing.assert_close(model.sin, sin)


@RunIf(min_cuda_gpus=1)
def test_reset_parameters_device():
    with torch.device("meta"):
        model = GPT(Config(**{**PYTHIA_14M, "n_layer": 1}))
    _materialize_meta_tensors(model, torch.device("cuda"))
    model.reset_parameters()
    assert model.cos.device.type == "cuda"


def test_batched_index_copy_modes():
    # Mock the torch.backends.mps.is_available() function to simulate MPS availability
    with mock.patch("torch.backends.mps.is_available", return_value=True):
        # Mock the device type to simulate the "mps" device
        with mock.patch("torch.Tensor.device", new_callable=mock.PropertyMock) as mock_device:
            mock_device.return_value = torch.device("mps")

            # Test case when idx.dim() == 1
            t_original_1 = torch.randn(3, 5)
            dim_1 = 0
            idx_1 = torch.tensor([0, 2])
            val_1 = torch.randn(2, 5)

            t1_cpu = t_original_1.clone()
            t1_mps = t_original_1.clone()

            # Perform the index copy on CPU
            batched_index_copy_(t1_cpu, dim_1, idx_1, val_1)

            # Simulate the MPS index copy
            idx_1_mps = idx_1
            val_1_mps = val_1
            batched_index_copy_(t1_mps, dim_1, idx_1_mps, val_1_mps)
            assert torch.allclose(t1_cpu, t1_mps), "Mismatch with idx.dim() == 1 on mocked MPS"

            # Test case when idx.dim() == 2
            t_original_2 = torch.randn(2, 5, 4)
            dim_2 = 1
            idx_2 = torch.tensor([[0, 2], [1, 3]])
            val_2 = torch.randn(2, 2, 4)

            t2_cpu = t_original_2.clone()
            t2_mps = t_original_2.clone()

            # Perform the index copy on CPU
            batched_index_copy_(t2_cpu, dim_2, idx_2, val_2)

            # Simulate the MPS index copy
            idx_2_mps = idx_2
            val_2_mps = val_2
            batched_index_copy_(t2_mps, dim_2, idx_2_mps, val_2_mps)
            assert torch.allclose(t2_cpu, t2_mps), "Mismatch with idx.dim() == 2 on mocked MPS"

            # Additional test with negative dimension
            t_original_3 = torch.randn(2, 3, 4)
            dim_3 = -2
            idx_3 = torch.tensor([[0, 1], [1, 2]])
            val_3 = torch.randn(2, 2, 4)

            t3_cpu = t_original_3.clone()
            t3_mps = t_original_3.clone()

            # Perform the index copy on CPU
            batched_index_copy_(t3_cpu, dim_3, idx_3, val_3)

            # Simulate the MPS index copy
            idx_3_mps = idx_3
            val_3_mps = val_3
            batched_index_copy_(t3_mps, dim_3, idx_3_mps, val_3_mps)
            assert torch.allclose(t3_cpu, t3_mps), "Mismatch with negative dimension on mocked MPS"


def test_load_legacy_state_dict():
    """Check that a legacy state dict (with an interleaved placement in QKV matrix) can be loaded into a model with CausalSelfAttention layers."""
    config = Config(
        n_embd=32,
        n_head=4,
        head_size=8,
        n_query_groups=4,
        bias=True,
    )

    attention_1 = CausalSelfAttention(config=config, block_idx=0)

    # make weights to be as-like in a legacy checkpoint, with `attn.attn.weight` instead of `attn.qkv.weight`
    # and make them interleaved
    state_dict = deepcopy(attention_1.state_dict())
    state_dict["attn.weight"] = make_qkv_interleaved(state_dict.pop("qkv.weight"), config)
    state_dict["attn.bias"] = make_qkv_interleaved(state_dict.pop("qkv.bias"), config)

    attention_2 = CausalSelfAttention(config=config, block_idx=0)
    attention_2.load_state_dict(state_dict)


@pytest.mark.parametrize("n_query_groups", (1, 2, 4, 8))
@torch.inference_mode()
def test_kv_cache_buffer_shape(n_query_groups):
    batch_size = 3
    max_seq_length = 23
    config = Config(
        block_size=25,
        padded_vocab_size=5,
        n_layer=2,
        n_head=8,
        n_embd=16,
        n_query_groups=n_query_groups,
    )
    model = GPT(config)
    model.max_seq_length = max_seq_length
    model.set_kv_cache(batch_size)
    required_shape = (batch_size, n_query_groups, max_seq_length, config.head_size)
    for block in model.transformer.h:
        kv_cache = block.attn.kv_cache
        assert kv_cache is not None
        assert kv_cache.k.shape == required_shape
        assert kv_cache.v.shape == required_shape


@pytest.mark.parametrize(("rotary_percentage", "final_dim"), ((0.75, 3), (0.25, 2)))
@torch.inference_mode()
def test_rope_cos_sin_shapes_if_rope_n_elem_is_odd(rotary_percentage, final_dim):
    batch_size = 3
    config = Config(
        block_size=25,
        padded_vocab_size=5,
        n_layer=2,
        n_head=4,
        n_embd=16,
        rotary_percentage=rotary_percentage,
    )
    model = GPT(config)
    required_shape = (config.block_size, final_dim)
    assert model.cos.shape == required_shape
    assert model.sin.shape == required_shape


def test_forward_with_without_input_pos_maxp1():
    batch_size = 3
    config = Config(
        block_size=25,
        padded_vocab_size=5,
        n_layer=2,
        n_head=8,
        n_embd=16,
    )
    model = GPT(config)
    model.set_kv_cache(batch_size)
    idx = torch.randint(0, config.padded_vocab_size, (1, 10))
    input_pos = torch.arange(1, 11)
    input_pos_maxp1 = 11
    logits_with_maxp1 = model(idx, input_pos, input_pos_maxp1=input_pos_maxp1)
    logits_no_maxp1 = model(idx, input_pos)
    torch.testing.assert_close(logits_with_maxp1, logits_no_maxp1)


@torch.inference_mode()
def test_sliding_window_kv_cache_prefill_exceeds_window():
    """Test that prefilling with more tokens than sliding_window_size raises an error.

    When prefill length exceeds the sliding window size, the ring-buffer KV cache
    overwrites entries but the attention mask (a global causal matrix truncated by
    length) does not reflect the true absolute positions. This causes early query
    tokens to attend to overwritten future KV entries, silently violating causality.

    See: https://github.com/Lightning-AI/litgpt/issues/2182
    """
    from litgpt.model import KVCache

    sliding_window_size = 4
    batch_size = 1
    n_query_groups = 2
    head_size = 8

    k_shape = (batch_size, n_query_groups, sliding_window_size, head_size)
    v_shape = (batch_size, n_query_groups, sliding_window_size, head_size)

    cache = KVCache(
        k_shape,
        v_shape,
        is_sliding_window=True,
        sliding_window_size=sliding_window_size,
    )

    # Prefill with more tokens than the sliding window size
    prefill_len = sliding_window_size + 2  # 6 > 4
    input_pos = torch.arange(prefill_len)
    k = torch.randn(batch_size, n_query_groups, prefill_len, head_size)
    v = torch.randn(batch_size, n_query_groups, prefill_len, head_size)

    with pytest.raises(ValueError, match="Prefill length.*exceeds the sliding window size"):
        cache(input_pos, k, v)

    # Prefill within the window size should work fine
    safe_len = sliding_window_size
    input_pos_safe = torch.arange(safe_len)
    k_safe = torch.randn(batch_size, n_query_groups, safe_len, head_size)
    v_safe = torch.randn(batch_size, n_query_groups, safe_len, head_size)
    k_out, v_out = cache(input_pos_safe, k_safe, v_safe)
    assert k_out.shape == (batch_size, n_query_groups, safe_len, head_size)
    assert v_out.shape == (batch_size, n_query_groups, safe_len, head_size)

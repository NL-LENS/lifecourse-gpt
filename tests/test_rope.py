# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch

from litgpt.model import build_rope_cache


@torch.inference_mode()
def test_rope_cos_sin_shapes_if_rope_n_elem_is_odd():
    bs, seq_len, n_head, n_embed = 1, 6, 2, 8
    head_size = n_embed // n_head  # 4
    rotary_percentage = 0.75
    rope_n_elem = int(head_size * rotary_percentage)  # 3
    ours_cos, ours_sin = build_rope_cache(seq_len, rope_n_elem)
    required_shape = (seq_len, rope_n_elem)
    assert ours_cos.shape == required_shape
    assert ours_sin.shape == required_shape
    # Special case: If `rope_n_elem == 1`, the shape is extended. This is to
    # accommodate a current bug in Hugging Face, ensuring that other unit tests
    # pass.
    # https://github.com/huggingface/transformers/issues/35233
    rotary_percentage = 0.25
    rope_n_elem = int(head_size * rotary_percentage)  # 1
    ours_cos, ours_sin = build_rope_cache(seq_len, rope_n_elem)
    required_shape = (seq_len, rope_n_elem + 1)
    assert ours_cos.shape == required_shape
    assert ours_sin.shape == required_shape

# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Delphi's model, tie mask and likelihood, and the hooks it plugs into."""

import math

import pytest
import torch

from litgpt.config import Config
from litgpt.life_events.delphi.generate import generate_trajectories
from litgpt.life_events.delphi.loss import chunked_cross_entropy_exp_waiting_time
from litgpt.life_events.delphi.mask import build_tie_mask, last_visible_index
from litgpt.life_events.delphi.model import Delphi
from litgpt.model import GPT, model_class
from litgpt.pretrain import _unpack_batch, forward_loss


def _config(**kwargs):
    kwargs = (
        dict(
            model_class_name="Delphi",
            block_size=8,
            vocab_size=6,
            padded_vocab_size=6,
            n_layer=1,
            n_head=2,
            n_embd=8,
            rotary_percentage=0.0,
        )
        | kwargs
    )
    return Config(**kwargs)


def test_model_class_resolution():
    assert model_class(Config()) is GPT
    assert model_class(Config.from_name("delphi-2m")) is Delphi
    assert model_class(_config(model_class_name="litgpt.life_events.delphi.model.Delphi")) is Delphi
    with pytest.raises(ValueError, match="Unknown model_class_name"):
        model_class(_config(model_class_name="Nope"))
    with pytest.raises(TypeError, match="not a subclass"):
        model_class(_config(model_class_name="litgpt.config.Config"))


def test_plain_gpt_carries_no_delphi_modules():
    """Text checkpoints must keep loading: nothing Delphi-specific in a plain GPT."""
    keys = GPT(_config(model_class_name="GPT")).state_dict()
    assert not any(".wae." in key or "token_drop" in key for key in keys)
    delphi_keys = Delphi(_config()).state_dict()
    assert "transformer.wae.linear.weight" in delphi_keys
    assert set(delphi_keys) - set(keys) == {"transformer.wae.linear.weight", "transformer.wae.div_term"}


def test_plain_gpt_rejects_ages():
    model = GPT(_config(model_class_name="GPT"))
    idx = torch.zeros(1, 3, dtype=torch.int64)
    with pytest.raises(TypeError, match="age"):
        model(idx, age=idx)


def test_delphi_warns_about_rope():
    with pytest.warns(UserWarning, match="rotary_percentage"):
        Delphi(_config(rotary_percentage=0.25))


def test_delphi_forward_uses_age():
    torch.manual_seed(0)
    model = Delphi(_config())
    model.apply(model._init_weights)
    model.eval()
    idx = torch.tensor([[2, 3, 4]])
    young, old = torch.tensor([[0, 100, 200]]), torch.tensor([[0, 10000, 20000]])
    assert model(idx, age=young).shape == (1, 3, 6)
    assert not torch.allclose(model(idx, age=young), model(idx, age=old))
    with pytest.raises(ValueError, match="equal shape"):
        model(idx, age=young[:, :2])


def test_tie_mask_hides_keys_at_the_target_age():
    age = torch.tensor([[0, 10, 10, 20]])
    targets_age = torch.tensor([[10, 10, 20, 30]])
    mask = build_tie_mask(age, targets_age)[0, 0]
    # Query 0 predicts an event at age 10: it may only see itself (age 0).
    assert mask[0].tolist() == [True, False, False, False]
    # Query 1 sits at its own target's age, so it has to hide itself as well.
    assert mask[1].tolist() == [True, False, False, False]
    # Query 2 predicts age 20: both age-10 events are fair game now.
    assert mask[2].tolist() == [True, True, True, False]
    # The waiting time for query 1 is measured from the last event it can see.
    assert last_visible_index(build_tie_mask(age, targets_age)).tolist() == [[0, 0, 2, 3]]

    # A first query tied to its own target has nothing left to see, and falls back to
    # itself rather than attending to nothing.
    lone = build_tie_mask(torch.tensor([[10, 10]]), torch.tensor([[10, 20]]))[0, 0]
    assert lone[0].tolist() == [True, False]


def test_tie_mask_drops_padding():
    age = torch.tensor([[0, 5, -1]])
    targets_age = torch.tensor([[5, 9, -1]])
    mask = build_tie_mask(age, targets_age, pad_mask=torch.tensor([[True, True, False]]))[0, 0]
    assert not mask[:2, 2].any()
    assert mask[2].tolist() == [False, False, True]


def test_waiting_time_loss_matches_a_hand_computation():
    logits = torch.tensor([[[0.0, math.log(2.0)], [math.log(3.0), 0.0]]])
    age = torch.tensor([[0, 10]])
    targets = torch.tensor([[1, 0]])
    targets_age = torch.tensor([[10, 15]])
    pad_mask = torch.tensor([[True, True]])
    losses = chunked_cross_entropy_exp_waiting_time(logits, age, targets, targets_age, pad_mask)

    # Position 0: rates (1, 2), total 3, target 1 after 10 days.
    # Position 1: rates (3, 1), total 4, target 0 after 5 days.
    ce = [-math.log(2 / 3), -math.log(3 / 4)]
    dt = [3 * 10 - math.log(3), 4 * 5 - math.log(4)]
    assert losses["loss_ce"].item() == pytest.approx(sum(ce) / 2)
    assert losses["loss_dt"].item() == pytest.approx(sum(dt) / 2)

    # Chunking changes memory use, not the result.
    chunked = chunked_cross_entropy_exp_waiting_time(logits, age, targets, targets_age, pad_mask, chunk_size=1)
    assert all(torch.allclose(losses[k], chunked[k]) for k in losses)


def test_unpack_batch_only_truncates_dict_batches():
    b, t = 2, 5
    batch = {
        "input_ids": torch.zeros(b, t, dtype=torch.int32),
        "labels": torch.ones(b, t, dtype=torch.int32),
        "pad_mask": torch.ones(b, t, dtype=torch.bool),
        "note_x": torch.zeros(b, t, dtype=torch.float32),
        "attn_mask": torch.ones(b, 1, t, t, dtype=torch.bool),
    }
    unpacked = _unpack_batch(batch, max_seq_length=3)
    assert set(unpacked) == {"input_ids", "targets", "pad_mask", "note_x", "attn_mask"}
    assert unpacked["input_ids"].shape == (b, 3) and unpacked["input_ids"].dtype == torch.int64
    assert unpacked["targets"].dtype == torch.int64
    assert unpacked["pad_mask"].dtype == torch.bool and unpacked["note_x"].dtype == torch.float32
    assert unpacked["attn_mask"].shape == (b, 1, 3, 3)


def test_forward_loss_dispatches_on_the_model_class():
    torch.manual_seed(0)
    gpt = GPT(_config(model_class_name="GPT"))
    text = _unpack_batch(torch.randint(0, 6, (2, 9)), gpt.max_seq_length)
    assert set(forward_loss(gpt, text)) == {"loss"}

    delphi = Delphi(_config())
    example = Delphi.example_batch(2, 8, delphi.config)
    example["targets_age"] = example["age"] + 1
    assert set(forward_loss(delphi, example)) == {"loss", "loss_ce", "loss_dt"}


def test_example_batches_measure_on_the_meta_device():
    for config in (_config(model_class_name="GPT"), _config()):
        arch = model_class(config)
        with torch.device("meta"):
            model = arch(config)
            example = arch.example_batch(2, model.max_seq_length, config)
            loss = arch.compute_loss(model(**arch.model_inputs(example)), example, config)["loss"]
        assert loss.device.type == "meta"


def test_generate_trajectories():
    torch.manual_seed(0)
    model = Delphi(_config(block_size=16))
    model.apply(model._init_weights)
    idx, age = torch.tensor([[2, 3]]), torch.tensor([[0, 100]])
    # Nothing terminates a trajectory here, so every step must append a sampled event.
    new_idx, new_age = generate_trajectories(
        model, idx, age, max_new_tokens=4, max_age=float("inf"), generator=torch.Generator().manual_seed(0)
    )
    assert new_idx.shape == new_age.shape == (1, 6)
    assert torch.equal(new_idx[:, :2], idx)
    # Ages never go backwards, and the padding token is never sampled.
    assert (new_age.diff(dim=1) >= 0).all()
    assert not (new_idx[:, 2:] == 0).any()


def test_kv_cache_matches_full_forward():
    # Ages enter only through the input embedding, so an event's keys and values never
    # change once appended and the cache must reproduce a full re-run of the sequence.
    torch.manual_seed(0)
    model = Delphi(_config(block_size=16)).eval()
    model.apply(model._init_weights)
    idx = torch.randint(1, 6, (3, 12))
    age = torch.randint(0, 30_000, (3, 12)).sort(dim=1).values

    with torch.inference_mode():
        expected = model(idx, age=age)
        model.set_kv_cache(batch_size=3, max_seq_length=12)
        chunks = [model(idx[:, :5], age=age[:, :5], input_pos=torch.arange(5), input_pos_maxp1=5)]
        for pos in range(5, 12):
            chunks.append(
                model(
                    idx[:, pos : pos + 1],
                    age=age[:, pos : pos + 1],
                    input_pos=torch.tensor([pos]),
                    input_pos_maxp1=pos + 1,
                )
            )
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected)


def test_generate_trajectories_leaves_no_kv_cache_behind():
    model = Delphi(_config(block_size=16)).train()
    idx, age = torch.tensor([[2, 3]]), torch.tensor([[0, 100]])
    generate_trajectories(model, idx, age, max_new_tokens=3, max_age=float("inf"))
    assert model.mask_cache is None
    assert all(block.attn.kv_cache is None for block in model.transformer.h)
    assert model.training

    # Also when sampling fails partway, here by running past the model's context.
    with pytest.raises(ValueError, match="max seq length"):
        generate_trajectories(model, idx, age, max_new_tokens=20, max_age=float("inf"))
    assert model.mask_cache is None
    assert model.training

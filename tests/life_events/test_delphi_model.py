# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Delphi's model, tie mask and likelihood, and the hooks it plugs into."""

import math

import pytest
import torch

from litgpt.config import Config
from litgpt.life_events.delphi.data import NO_EVENT_ID, PAD_AGE, PAD_ID
from litgpt.life_events.delphi.generate import generate_trajectories
from litgpt.life_events.delphi.loss import chunked_cross_entropy_exp_waiting_time
from litgpt.life_events.delphi.mask import build_tie_mask, last_visible_index
from litgpt.life_events.delphi.model import AgeEncoding, Delphi
from litgpt.model import GPT, batched_index_select, model_class
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


def test_age_unit_rescales_the_age_encoding():
    """An age in months encodes like twelve times that age in years."""
    years, months = AgeEncoding(_config(age_unit="years")), AgeEncoding(_config(age_unit="months"))
    months.linear.load_state_dict(years.linear.state_dict())
    age = torch.tensor([[[0.0], [400.0], [9000.0]]])
    torch.testing.assert_close(months(age), years(age * 12))
    with pytest.raises(ValueError, match="age_unit"):
        AgeEncoding(_config(age_unit="decades"))


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


def test_batched_index_select_handles_empty_tensors():
    # Delphi has no rotary dims, so the RoPE cache it indexes by position is empty.
    t = torch.empty(16, 0)
    assert batched_index_select(t, 0, torch.tensor([[3], [5]])).shape == (2, 1, 0)
    t = torch.arange(32.0).view(16, 2)
    torch.testing.assert_close(batched_index_select(t, 0, torch.tensor([[3], [5]])), t[torch.tensor([[3], [5]])])


def test_generate_continues_each_row_from_its_last_real_event():
    torch.manual_seed(0)
    model = Delphi(_config(block_size=16))
    model.apply(model._init_weights)
    lengths = [2, 5, 3]
    idx = torch.tensor([[2, 3, 0, 0, 0], [4, 2, 5, 3, 2], [5, 5, 4, 0, 0]])
    age = torch.tensor([[10, 400, -1, -1, -1], [0, 50, 90, 300, 800], [5, 6, 700, -1, -1]])

    # Record the logits each step samples from: the prefill covers the whole prompt,
    # every later forward a single event per row.
    seen = []
    hook = model.register_forward_hook(lambda module, args, out: seen.append(out.detach().clone()))
    n = 4
    new_idx, new_age = generate_trajectories(model, idx, age, max_new_tokens=n, max_age=float("inf"))
    assert new_idx.shape == new_age.shape == (3, 5 + n)
    hook.remove()
    assert len(seen) == n

    for b, L in enumerate(lengths):
        assert torch.equal(new_idx[b, :L], idx[b, :L])
        assert not (new_idx[b, L : L + n] == 0).any()
        assert (new_idx[b, L + n :] == 0).all()
        assert (new_age[b, L - 1 : L + n].diff() >= 0).all()
        assert (new_age[b, L + n :] == PAD_AGE).all()
        # Each step saw exactly the logits of running this row alone and unpadded.
        with torch.inference_mode():
            expected = model(new_idx[b : b + 1, : L + n], age=new_age[b : b + 1, : L + n])[0, L - 1 : L + n - 1]
        actual = torch.stack([seen[0][b, L - 1]] + [out[b, 0] for out in seen[1:]])
        torch.testing.assert_close(actual, expected)


def test_generate_pads_rows_that_stop_early():
    model = Delphi(_config(block_size=16))
    idx = torch.tensor([[2, 3, 0], [4, 5, 2]])
    # The first row is already past `max_age`, so only the second one advances.
    age = torch.tensor([[10, 1_000, -1], [0, 50, 90]])
    new_idx, new_age = generate_trajectories(model, idx, age, max_new_tokens=3, max_age=500)
    assert new_idx.shape == new_age.shape == (2, 6)
    assert torch.equal(new_idx[0, :2], idx[0, :2])
    assert torch.equal(new_age[0, :2], age[0, :2])
    assert (new_idx[0, 2:] == 0).all()
    assert (new_age[0, 2:] == PAD_AGE).all()


def test_generate_output_has_a_fixed_width():
    model = Delphi(_config(block_size=16))
    idx, age = torch.tensor([[2, 3]]), torch.tensor([[0, 100]])
    # Stops before sampling anything, and still returns the full width.
    new_idx, new_age = generate_trajectories(model, idx, age, max_new_tokens=5, max_age=50)
    assert new_idx.shape == new_age.shape == (1, 7)
    assert (new_idx[:, 2:] == 0).all()
    assert (new_age[:, 2:] == PAD_AGE).all()


def test_generate_writes_a_custom_pad_age():
    model = Delphi(_config(block_size=16))
    idx, age = torch.tensor([[2, 3, 0]]), torch.tensor([[0, 100, -1]])
    _, new_age = generate_trajectories(model, idx, age, max_new_tokens=2, max_age=50, pad_age=-999)
    assert torch.equal(new_age, torch.tensor([[0, 100, -999, -999, -999]]))


@pytest.mark.parametrize(
    "idx",
    [
        pytest.param([[0, 2, 3], [2, 3, 4]], id="left-padded"),
        pytest.param([[2, 0, 3], [2, 3, 4]], id="padding-in-the-middle"),
        pytest.param([[0, 0, 0], [2, 3, 4]], id="empty-row"),
    ],
)
def test_generate_rejects_prompts_that_are_not_right_padded(idx):
    model = Delphi(_config(block_size=16))
    idx = torch.tensor(idx)
    with pytest.raises(ValueError, match="right-padded|at least one"):
        generate_trajectories(model, idx, torch.zeros_like(idx), max_new_tokens=2)


def _model_preferring(token: int) -> Delphi:
    """A model whose logits ignore the input and put nearly all mass on `token`."""
    model = Delphi(_config(block_size=16, lm_head_bias=True))
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.fill_(-10.0)
        model.lm_head.bias[token] = 10.0
    return model


def test_no_repeat_lets_no_event_tokens_recur():
    model = _model_preferring(NO_EVENT_ID)
    idx, age = torch.tensor([[2, NO_EVENT_ID, 3]]), torch.tensor([[0, 100, 200]])
    new_idx, _ = generate_trajectories(model, idx, age, max_new_tokens=4, max_age=float("inf"), no_repeat=True)
    assert (new_idx[0, 3:] == NO_EVENT_ID).all()


def test_no_repeat_blocks_everything_not_repeatable():
    model = _model_preferring(NO_EVENT_ID)
    idx, age = torch.tensor([[2, NO_EVENT_ID, 3]]), torch.tensor([[0, 100, 200]])
    new_idx, _ = generate_trajectories(
        model, idx, age, max_new_tokens=4, max_age=float("inf"), no_repeat=True, repeatable_tokens=()
    )
    # Only events 4 and 5 are left to sample, once each; then nothing admissible remains.
    assert sorted(new_idx[0, 3:5].tolist()) == [4, 5]
    assert (new_idx[0, 5:] == PAD_ID).all()


def test_target_vocab_size_defaults_to_the_vocabulary_and_is_bounded():
    assert _config().target_vocab_size == 6
    assert _config(target_vocab_size=4).target_vocab_size == 4
    with pytest.raises(ValueError, match="target_vocab_size must be in"):
        _config(target_vocab_size=7)


def test_logits_past_target_vocab_size_do_not_enter_the_loss():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4)
    age = torch.tensor([[0, 10, 20], [0, 5, 5]])
    targets = torch.tensor([[1, 2, 3], [0, 3, -100]])
    targets_age = torch.tensor([[10, 20, 30], [5, 5, -1]])
    pad_mask = targets != -100
    args = (age, targets, targets_age, pad_mask)
    expected = chunked_cross_entropy_exp_waiting_time(logits, *args)
    # Two more entries, huge, as statics or padding could be: sliced away, they change nothing.
    padded = torch.cat([logits, torch.full((2, 3, 2), 50.0)], dim=-1)
    actual = chunked_cross_entropy_exp_waiting_time(padded, *args, target_vocab_size=4)
    assert all(torch.allclose(expected[k], actual[k]) for k in expected)
    unmasked = chunked_cross_entropy_exp_waiting_time(padded, *args)
    assert unmasked["loss_ce"] > expected["loss_ce"] + 10


def test_compute_loss_reads_target_vocab_size_from_the_config():
    model = Delphi(_config(target_vocab_size=4, lm_head_bias=True))
    model.apply(model._init_weights)
    x = torch.tensor([[2, 3, 1, 2]])
    batch = {
        "input_ids": x,
        "labels": torch.tensor([[3, 1, 2, 3]]),
        "age": torch.tensor([[0, 10, 20, 30]]),
        "targets_age": torch.tensor([[10, 20, 30, 40]]),
        "pad_mask": torch.ones_like(x, dtype=torch.bool),
    }
    before = forward_loss(model, _unpack_batch(batch, model.max_seq_length))
    with torch.no_grad():
        model.lm_head.bias[4:] = 50.0
    after = forward_loss(model, _unpack_batch(batch, model.max_seq_length))
    assert all(torch.allclose(before[k], after[k]) for k in before)


def test_generate_never_samples_past_target_vocab_size():
    # Tokens 4 and 5 are statics: by far the likeliest logits, and in the prompt too.
    model = Delphi(_config(block_size=16, lm_head_bias=True, target_vocab_size=4))
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.fill_(0.0)
        model.lm_head.bias[4:] = 50.0
    idx, age = torch.tensor([[4, 5, 2]]), torch.tensor([[0, 0, 100]])
    # `ignore_tokens` and `repeatable_tokens` may name ids past `target_vocab_size` harmlessly.
    new_idx, _ = generate_trajectories(
        model,
        idx,
        age,
        max_new_tokens=8,
        max_age=float("inf"),
        no_repeat=True,
        ignore_tokens=(PAD_ID, 5),
        repeatable_tokens=(NO_EVENT_ID, 4),
    )
    sampled = new_idx[0, 3:]
    assert not (sampled >= 4).any()
    # Only 1 (repeatable) and 3 remain, and 2 is blocked by `no_repeat`.
    assert set(sampled[sampled != PAD_ID].tolist()) <= {NO_EVENT_ID, 3}

# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Delphi's batch layout: no-event tokens, windowing, the shift and the tie mask."""

import pickle
from functools import partial

import numpy as np
import pytest
import torch

pytest.importorskip("duckdb")

from litgpt.life_events.data import NoteField  # noqa: E402
from litgpt.life_events.delphi.data import (  # noqa: E402
    FIRST_EVENT_ID,
    NO_EVENT_ID,
    PAD_ID,
    DelphiLifeEvents,
    delphi_collate_fn,
)

from . import helpers  # noqa: E402

make_module = partial(helpers.make_module, cls=DelphiLifeEvents, no_event_mode="none")
prepared = partial(helpers.prepared, cls=DelphiLifeEvents, no_event_mode="none")


def test_reserves_pad_and_no_event(tmp_path, dataset):
    data = prepared(tmp_path, dataset)
    assert data.vocab.itos[:FIRST_EVENT_ID] == ["<pad>", "<no_event>"]
    assert data.vocab.stoi["<pad>"] == PAD_ID and data.vocab.stoi["<no_event>"] == NO_EVENT_ID
    assert data.vocab.first_event_id == FIRST_EVENT_ID


def test_batch_contract(tmp_path, dataset):
    data = prepared(tmp_path, dataset)
    batch = next(iter(data.train_dataloader()))
    assert set(batch) == {"input_ids", "labels", "age", "targets_age", "pad_mask", "attn_mask"}
    b, t = batch["input_ids"].shape
    assert t == data.max_seq_length  # pad_to_max_seq_length keeps `T` stable for torch.compile
    for key in ("labels", "age", "targets_age"):
        assert batch[key].shape == (b, t) and batch[key].dtype == torch.int64
    assert batch["pad_mask"].shape == (b, t) and batch["pad_mask"].dtype == torch.bool
    assert batch["attn_mask"].shape == (b, 1, t, t) and batch["attn_mask"].dtype == torch.bool

    # `chunked_cross_entropy_exp_waiting_time` gathers the waiting time at the last
    # *visible* key, which only measures the right interval because labels and ages are
    # an exact one-step shift of the inputs. Do not "optimize" this away.
    real = batch["pad_mask"][:, :-1] & batch["pad_mask"][:, 1:]
    assert torch.equal(batch["labels"][:, :-1][real], batch["input_ids"][:, 1:][real])
    assert torch.equal(batch["targets_age"][:, :-1][real], batch["age"][:, 1:][real])


def test_pad_mask_includes_no_event_but_not_padding(tmp_path, dataset):
    data = prepared(tmp_path, dataset, no_event_mode="regular", no_event_rate=5.0)
    batch = next(iter(data.train_dataloader()))
    assert (batch["labels"] == NO_EVENT_ID).any(), "fixture should produce no-event targets"
    assert batch["pad_mask"][batch["labels"] == NO_EVENT_ID].all()
    padding = batch["labels"] == data.ignore_index
    assert not batch["pad_mask"][padding].any()
    assert (batch["input_ids"][padding] == PAD_ID).all()


def test_no_event_mode_none_emits_no_no_event_tokens(tmp_path, dataset):
    data = prepared(tmp_path, dataset, no_event_mode="none")
    batch = next(iter(data.train_dataloader()))
    assert not (batch["input_ids"] == NO_EVENT_ID).any()
    assert not (batch["labels"] == NO_EVENT_ID).any()


def test_no_event_regular_grid_and_truncation(tmp_path, dataset):
    data = prepared(tmp_path, dataset, no_event_mode="regular", no_event_rate=5.0, max_seq_length=-1)
    store = data._store
    for i in range(len(data.train_dataset)):
        real_ages = store.sequence(int(data.train_dataset.indices[i]))["times"]
        end_of_observation = int(real_ages[-1])
        sample = data.train_dataset[i]
        tokens = torch.cat([sample["input_ids"], sample["labels"][-1:]])
        ages = torch.cat([sample["age"], sample["targets_age"][-1:]])
        inserted = ages[tokens == NO_EVENT_ID].numpy()
        assert inserted.size, "the 5-year grid should reach inside every fixture lifetime"
        # Delphi's grid, and nothing past the end of observation.
        grid = np.round(np.arange(0.0, 36525, 365.25 * 5.0) + 1.0).astype(np.int64)
        assert set(inserted.tolist()) <= set(grid.tolist())
        assert inserted.max() <= end_of_observation
        # A no-event token never shares a day with a real event.
        assert not set(inserted.tolist()) & set(real_ages.tolist())
        # Ages stay sorted after the merge.
        assert torch.equal(ages, torch.sort(ages).values)


def test_no_eos_token_is_emitted(tmp_path, dataset):
    """Deliberate deviation from LIFE_EVENT_DATA.md: the censored tail is dropped."""
    data = prepared(tmp_path, dataset, no_event_mode="none", max_seq_length=-1)
    sample = data.train_dataset[0]
    tokens = data._store.sequence(int(data.train_dataset.indices[0]))["tokens"]
    assert sample["labels"][-1].item() == int(tokens[-1])
    assert len(sample["input_ids"]) == len(tokens) - 1


def test_random_resamples_across_epochs_but_validation_does_not(tmp_path, dataset):
    data = prepared(tmp_path, dataset, no_event_mode="random", no_event_rate=5.0, num_workers=2)
    train = data.train_dataloader()
    first, second = [b["age"] for b in train], [b["age"] for b in train]
    assert any(not torch.equal(x, y) for x, y in zip(first, second))

    val = data.val_dataloader()
    assert all(torch.equal(x, y) for x, y in zip([b["age"] for b in val], [b["age"] for b in val]))


def test_identical_across_num_workers(tmp_path, dataset):
    """Pins the `info.seed - info.id` assumption in `DelphiDataset._rng`."""
    batches = []
    for workers in (0, 2):
        data = prepared(tmp_path / f"w{workers}", dataset, no_event_mode="random", num_workers=workers)
        batches.append([b["age"] for b in data.val_dataloader()])
    assert all(torch.equal(x, y) for x, y in zip(*batches))


def test_windowing(tmp_path, dataset):
    full = prepared(tmp_path / "full", dataset, no_event_mode="regular", max_seq_length=-1)
    lengths = [len(full.train_dataset[i]["input_ids"]) for i in range(len(full.train_dataset))]
    width = min(lengths) - 1
    assert width >= 2, "fixture must be long enough to actually window"

    left = prepared(tmp_path / "left", dataset, no_event_mode="regular", max_seq_length=width, select="left")
    right = prepared(tmp_path / "right", dataset, no_event_mode="regular", max_seq_length=width, select="right")
    # The whole merged age sequence, before any window is taken. `"regular"` rather than
    # `"random"` so the grid is identical across the three modules being compared.
    sample = full.train_dataset[0]
    merged = torch.cat([sample["age"], sample["targets_age"][-1:]])
    assert torch.equal(left.train_dataset[0]["age"], merged[:width])
    assert torch.equal(right.train_dataset[0]["age"], merged[-width - 1 : -1])

    rnd = prepared(tmp_path / "rnd", dataset, no_event_mode="regular", max_seq_length=width, select="random")
    window = rnd.train_dataset[0]["age"]
    assert len(window) == width
    assert torch.equal(window, torch.sort(window).values)


def test_collate_fn_is_picklable(tmp_path, dataset):
    data = prepared(tmp_path, dataset, notes_fields={"weight": NoteField("weight")})
    pickle.loads(pickle.dumps(data.train_dataloader().collate_fn))


def test_collate_rejects_oversized_sample():
    sample = {k: torch.zeros(5, dtype=torch.int64) for k in ("input_ids", "labels", "age", "targets_age")}
    with pytest.raises(ValueError, match="exceeds pad_to_length"):
        delphi_collate_fn([sample], pad_to_length=3)


def test_invalid_no_event_rate(tmp_path, dataset):
    with pytest.raises(ValueError, match="no_event_rate must be positive"):
        make_module(tmp_path, dataset, no_event_rate=0.0)


def test_note_features_travel_with_their_events(tmp_path, dataset):
    """Inserted no-event tokens take the fill; real events keep their value; padding the fill."""
    data = prepared(
        tmp_path,
        dataset,
        no_event_mode="regular",
        max_seq_length=-1,
        notes_fields={"weight": NoteField("weight", "DOUBLE", fill=-1.0), "a": NoteField("a", "INTEGER", fill=7)},
    )
    for i in range(len(data.train_dataset)):
        sample = data.train_dataset[i]
        stored = data._store.sequence(int(data.train_dataset.indices[i]))
        assert sample["note_weight"].dtype == torch.float32 and sample["note_a"].dtype == torch.int64
        assert sample["note_weight"].shape == sample["input_ids"].shape
        no_event = sample["input_ids"] == NO_EVENT_ID
        assert no_event.any()
        assert (sample["note_weight"][no_event] == -1.0).all() and (sample["note_a"][no_event] == 7).all()
        # The last real event is only ever a target, so the inputs hold all the others.
        assert sample["note_weight"][~no_event].tolist() == stored["note_weight"][:-1].tolist()
        assert sample["note_a"][~no_event].tolist() == stored["note_a"][:-1].tolist()
    # `weight` exists only in one fixture dataset and `a` only in the other, so each
    # feature holds both real values and fills.
    assert set(np.unique(data._store.features["note_weight"]).tolist()) == {2.0, -1.0}
    assert set(np.unique(data._store.features["note_a"]).tolist()) == {1, 7}

    data.connect(batch_size=2, max_seq_length=16)
    batch = next(iter(data.train_dataloader()))
    assert batch["note_weight"].shape == batch["input_ids"].shape
    padding = batch["input_ids"] == PAD_ID
    assert padding.any() and (batch["note_weight"][padding] == -1.0).all() and (batch["note_a"][padding] == 7).all()


def test_end_to_end_through_delphi_and_the_waiting_time_loss(tmp_path, dataset):
    """The batch layout is only correct if `Delphi` and the real loss accept it unchanged."""
    from litgpt.config import Config
    from litgpt.life_events.delphi.model import Delphi
    from litgpt.pretrain import _unpack_batch, forward_loss

    data = prepared(
        tmp_path, dataset, no_event_mode="regular", max_seq_length=8, notes_fields={"weight": NoteField("weight")}
    )
    batch = next(iter(data.train_dataloader()))

    config = Config(
        model_class_name="Delphi",
        block_size=8,
        vocab_size=data.vocab_size,
        padded_vocab_size=data.vocab_size,
        n_layer=1,
        n_head=2,
        n_embd=8,
        rotary_percentage=0.0,
        t_min=0.0,
    )
    model = Delphi(config)
    model.apply(model._init_weights)

    unpacked = _unpack_batch(batch, model.max_seq_length)
    assert unpacked["note_weight"].dtype == torch.float32
    losses = forward_loss(model, unpacked)
    assert set(losses) == {"loss", "loss_ce", "loss_dt"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert torch.allclose(losses["loss"], losses["loss_ce"] + losses["loss_dt"])

    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_pretrain_refuses_a_model_that_cannot_read_the_batches(tmp_path, dataset):
    """`--model_config.*` replaces `delphi-2m` wholesale, `model_class_name` included."""
    from litgpt.config import Config
    from litgpt.pretrain import setup

    data = make_module(tmp_path, dataset)
    with pytest.raises(ValueError, match="model_class_name='Delphi'"):
        setup("delphi-2m", model_config=Config(n_layer=1, n_head=1, n_embd=4), data=data, out_dir=tmp_path / "out")


def test_delphi_only_accepts_age(tmp_path, dataset):
    """The 100-year no-event horizon and `AgeEncoding` are both about age, not calendar time."""
    assert DelphiLifeEvents.time_origins == ("birth",)
    with pytest.raises(ValueError, match="reads time measured from"):
        make_module(tmp_path, dataset, time_origin="epoch")


def test_delphi_min_time_defaults_to_birth(tmp_path, dataset):
    assert make_module(tmp_path, dataset).min_time == 0

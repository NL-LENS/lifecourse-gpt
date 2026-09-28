# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
from collections import defaultdict
from datetime import date

import pytest
import torch
import yaml

from litgpt.config import Config
from litgpt.generate import setup as generate
from litgpt.life_events.delphi.data import DAYS_PER_YEAR, NO_EVENT_TOKEN, DelphiLifeEvents
from litgpt.life_events.delphi.model import Delphi
from litgpt.utils import save_config

from .helpers import BIRTHDATES, prepared, write_parquet, write_persons


def _checkpoint(tmp_path, dataset, config_kwargs=None, **data_kwargs):
    """What `litgpt pretrain` leaves behind, around an untrained model."""
    data = prepared(tmp_path, dataset, cls=DelphiLifeEvents, **data_kwargs)
    config = Config(
        name="tiny-delphi",
        model_class_name="Delphi",
        block_size=16,
        vocab_size=data.vocab_size,
        padded_vocab_size=data.vocab_size,
        n_layer=1,
        n_head=2,
        n_embd=8,
        rotary_percentage=0.0,
        **(config_kwargs or {}),
    )
    torch.manual_seed(0)
    model = Delphi(config)
    model.apply(model._init_weights)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    torch.save({"model": model.state_dict(), "iter_num": 0}, checkpoint_dir / "lit_model.pth")
    save_config(config, checkpoint_dir)
    event_paths, person_path = dataset
    init_args = dict(
        event_paths=event_paths,
        person_path=person_path,
        cache_dir=str(data.cache_dir),
        val_split_fraction=1 / 3,
        num_workers=0,
        **data_kwargs,
    )
    hparams = {"data": {"class_path": "litgpt.life_events.delphi.DelphiLifeEvents", "init_args": init_args}}
    (checkpoint_dir / "hyperparameters.yaml").write_text(yaml.dump(hparams))
    return checkpoint_dir, data


def _trajectories(path):
    """`{(ID, SAMPLE): [(days since birth, EVENT, GENERATED), ...]}`, in file order."""
    import pyarrow.parquet as pq

    trajectories = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        days = (row["TIME"] - date.fromisoformat(BIRTHDATES[row["ID"]])).days
        trajectories[row["ID"], row["SAMPLE"]].append((days, row["EVENT"], row["GENERATED"]))
    return trajectories


def test_generate_from_the_cache_with_a_cutoff(tmp_path, dataset):
    pytest.importorskip("pyarrow")
    checkpoint_dir, _ = _checkpoint(tmp_path, dataset)
    out_path = tmp_path / "out" / "trajectories.parquet"
    generate(
        checkpoint_dir,
        out_path=out_path,
        split="all",
        cutoff_age=22,
        num_samples=2,
        max_new_tokens=6,
        max_age=100,
        termination_events=["marriage"],
    )

    import pyarrow.parquet as pq

    assert pq.read_schema(out_path).names == ["ID", "TIME", "EVENT", "SAMPLE", "GENERATED"]
    trajectories = _trajectories(out_path)
    assert set(trajectories) == {(person, sample) for person in "abc" for sample in (0, 1)}

    cutoff = round(22 * DAYS_PER_YEAR)
    for rows in trajectories.values():
        prompt = [row for row in rows if not row[2]]
        sampled = [row for row in rows if row[2]]
        assert rows == prompt + sampled
        # Only what was observed by the cutoff, closed by a no-event token at the cutoff.
        assert all(days <= cutoff for days, _, _ in prompt)
        assert prompt[-1][:2] == (cutoff, NO_EVENT_TOKEN)

        assert 0 < len(sampled) <= 6
        ages = [days for days, _, _ in sampled]
        assert ages == sorted(ages) and ages[0] >= cutoff
        # Nothing follows a termination event.
        events = [event for _, event, _ in sampled]
        assert "marriage" not in events[:-1]


def test_generate_simulates_the_held_out_split_by_default(tmp_path, dataset):
    pytest.importorskip("pyarrow")
    checkpoint_dir, data = _checkpoint(tmp_path, dataset)
    out_path = tmp_path / "trajectories.parquet"
    generate(checkpoint_dir, out_path=out_path, max_new_tokens=4)
    val_ids = {str(data.val_dataset.store.ids[i]) for i in data.val_dataset.indices}
    assert {person for person, _ in _trajectories(out_path)} == val_ids


def test_generate_from_separate_prompt_datasets(tmp_path, dataset):
    pytest.importorskip("pyarrow")
    checkpoint_dir, _ = _checkpoint(tmp_path, dataset)
    events = write_parquet(
        tmp_path / "prompts.parquet",
        [("b", "1990-06-15", "birth"), ("b", "2010-01-01", "job"), ("b", "2012-01-01", "astronaut")],
        notes_sql="{'a': 1}",
    )
    persons = write_persons(tmp_path / "prompt_persons.parquet", {"b": BIRTHDATES["b"]})
    kwargs = dict(prompt_paths=[str(events)], prompt_person_path=persons, max_new_tokens=4)

    # The checkpoint's vocabulary has no astronauts.
    with pytest.raises(ValueError, match="absent from"):
        generate(checkpoint_dir, out_path=tmp_path / "rejected.parquet", **kwargs)

    out_path = tmp_path / "trajectories.parquet"
    generate(checkpoint_dir, out_path=out_path, on_unknown_event="drop", **kwargs)
    (rows,) = _trajectories(out_path).values()
    prompt = [event for _, event, generated in rows if not generated]
    assert [event for event in prompt if event != NO_EVENT_TOKEN] == ["birth", "job"]


def test_generate_needs_room_for_a_prompt(tmp_path, dataset):
    checkpoint_dir, _ = _checkpoint(tmp_path, dataset)
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(checkpoint_dir, out_path=tmp_path / "out.parquet", max_new_tokens=16)

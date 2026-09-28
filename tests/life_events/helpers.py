# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from litgpt.life_events.data import LifeEventsDataModule

BIRTHDATES = {"a": "2000-01-01", "b": "1990-06-15", "c": "1985-03-20"}


def write_parquet(path, rows, notes_sql):
    """rows: (id, iso date, event). `notes_sql` fixes this dataset's NOTES column."""
    import duckdb

    values = ", ".join(f"('{i}', DATE '{t}', '{e}', {notes_sql})" for i, t, e in rows)
    duckdb.connect().execute(
        f"COPY (SELECT * FROM (VALUES {values}) AS t(ID, TIME, EVENT, NOTES)) TO '{path}' (FORMAT PARQUET)"
    )
    return path


STATICS = {"GENDER": {"a": "F", "b": "M", "c": None}, "MIGRATION": {"a": "none", "b": "second", "c": "none"}}


def write_persons(path, birthdates=BIRTHDATES, statics=None):
    """`statics`: `{column: {id: value or None}}`, extra VARCHAR columns of the person table."""
    import duckdb

    statics = statics or {}

    def literal(value):
        return "CAST(NULL AS VARCHAR)" if value is None else f"'{value}'"

    values = ", ".join(
        "(" + ", ".join([f"'{i}'", f"DATE '{d}'", *(literal(statics[c][i]) for c in statics)]) + ")"
        for i, d in birthdates.items()
    )
    columns = ", ".join(["ID", "BIRTHDATE", *statics])
    duckdb.connect().execute(f"COPY (SELECT * FROM (VALUES {values}) AS t({columns})) TO '{path}' (FORMAT PARQUET)")
    return path


def write_int_id_parquet(path, rows, notes_sql="{'a': 1}", id_type="BIGINT"):
    import duckdb

    values = ", ".join(f"(CAST({i} AS {id_type}), DATE '{t}', '{e}', {notes_sql})" for i, t, e in rows)
    duckdb.connect().execute(
        f"COPY (SELECT * FROM (VALUES {values}) AS t(ID, TIME, EVENT, NOTES)) TO '{path}' (FORMAT PARQUET)"
    )
    return path


def write_int_persons(path, birthdates, id_type="BIGINT"):
    import duckdb

    values = ", ".join(f"(CAST({i} AS {id_type}), DATE '{d}')" for i, d in birthdates.items())
    duckdb.connect().execute(f"COPY (SELECT * FROM (VALUES {values}) AS t(ID, BIRTHDATE)) TO '{path}' (FORMAT PARQUET)")
    return path


class _SequenceDataset(Dataset):
    def __init__(self, store, indices):
        self.store, self.indices = store, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return {k: torch.from_numpy(np.array(v)) for k, v in self.store.sequence(int(self.indices[idx])).items()}


def _list_collate(samples):
    return samples


@dataclass
class PlainLifeEvents(LifeEventsDataModule):
    """The smallest possible architecture: raw sequences, one reserved token.

    Exercises the generic layer without any Delphi assumption, and doubles as a check
    that `LifeEventsDataModule` really is enough to build a new architecture on.
    """

    def make_dataset(self, store, indices, train):
        return _SequenceDataset(store, indices)

    def collate_fn(self):
        return _list_collate


def make_module(tmp_path, dataset, cls=PlainLifeEvents, **kwargs):
    event_paths, person_path = dataset
    kwargs.setdefault("val_split_fraction", 1 / 3)
    kwargs.setdefault("num_workers", 0)
    # `connect()` args are not dataclass fields, so they have to come out first.
    batch_size = kwargs.pop("batch_size", 2)
    max_seq_length = kwargs.pop("max_seq_length", 8)
    data = cls(event_paths=event_paths, person_path=person_path, cache_dir=tmp_path / "cache", **kwargs)
    data.connect(batch_size=batch_size, max_seq_length=max_seq_length)
    return data


def prepared(tmp_path, dataset, cls=PlainLifeEvents, **kwargs):
    data = make_module(tmp_path, dataset, cls=cls, **kwargs)
    data.prepare_data()
    data.setup()
    return data


def meta(data):
    return json.loads((data.cache_dir / "meta.json").read_text())

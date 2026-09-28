# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""The architecture-neutral half: silver-layer reading, the cache, the vocabulary."""

import json
import pickle

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from litgpt.life_events.data import EventVocab, NoteField  # noqa: E402
from litgpt.life_events.data.source import _INTEGER_ID_TYPES, _token_dtype  # noqa: E402

from .helpers import (  # noqa: E402
    PlainLifeEvents,
    make_module,
    meta,
    prepared,
    write_int_id_parquet,
    write_int_persons,
    write_parquet,
    write_persons,
)


def _sequence_of(data, person_id):
    index = list(np.asarray(data._store.ids)).index(person_id)
    return data._store.sequence(index)


def test_datamodule_smoke(tmp_path, dataset):
    data = make_module(tmp_path, dataset)
    # Has attributes from super class `LightningDataModule`, i.e. `super().__init__()` ran.
    assert data.prepare_data_per_node
    data.prepare_data()
    data.setup()
    for name in ("ids.npy", "offsets.npy", "times.npy", "tokens.npy", "vocab.json", "meta.json"):
        assert (data.cache_dir / name).is_file()
    assert len(data.train_dataset) == 2
    assert len(data.val_dataset) == 1
    # The subclass decides what a sample is; here, the raw sequence.
    assert set(next(iter(data.train_dataloader()))[0]) == {"tokens", "times"}


def test_heterogeneous_notes_schemas_are_merged(tmp_path, dataset):
    """The per-file projection means NOTES is never bound, whatever its struct."""
    data = prepared(tmp_path, dataset)
    # `move` comes only from the second dataset, `study` only from the first.
    assert {"move", "study", "marriage"} <= set(data.vocab.events)
    assert meta(data)["n_events"] == 11
    assert meta(data)["features"] == {}


def test_default_origin_is_age_in_whole_days_since_birth(tmp_path, dataset):
    data = prepared(tmp_path, dataset)
    assert (data._store.time_origin, data._store.time_unit) == ("birth", "day")
    assert meta(data)["time"] == {"origin": "birth", "unit": "day", "offset_reference": "1970-01-01"}
    times = np.asarray(data._store.times)
    assert times.dtype == np.int32
    # Individual "a": born 2000-01-01, first event the same day.
    assert times.min() == 0
    # 2018-09-01 minus 2000-01-01.
    assert (np.datetime64("2018-09-01") - np.datetime64("2000-01-01")).astype(int) in times.tolist()


def test_cooccurring_events_are_alphabetical(tmp_path, dataset):
    """Individual "a" has `job` and `move` on 2022-03-01, from two different datasets."""
    data = prepared(tmp_path, dataset)
    sequence = _sequence_of(data, "a")
    times = sequence["times"].tolist()
    tied = [data.vocab.itos[t] for t, time in zip(sequence["tokens"].tolist(), times) if times.count(time) > 1]
    assert tied == ["job", "move"]


def test_vocab_ids_and_roundtrip(tmp_path):
    vocab = EventVocab(["alpha", "beta"], specials=("<pad>", "<x>", "<y>"))
    assert vocab.pad_id == 0 and vocab.first_event_id == 3
    assert vocab.encode(["alpha", "beta"]) == [3, 4]
    assert vocab.vocab_size == 5
    vocab.save(tmp_path / "v.json")
    assert EventVocab.load(tmp_path / "v.json").itos == vocab.itos
    assert EventVocab.load(tmp_path / "v.json", ("<pad>", "<x>", "<y>")).itos == vocab.itos

    # Loading under a different layout of reserved ids would shift every event.
    with pytest.raises(ValueError, match="not compatible"):
        EventVocab.load(tmp_path / "v.json", ("<pad>",))
    (tmp_path / "legacy.json").write_text(json.dumps({"events": ["alpha"]}))
    with pytest.raises(ValueError, match="not compatible"):
        EventVocab.load(tmp_path / "legacy.json")

    with pytest.raises(ValueError, match="must not appear"):
        EventVocab(["<x>"], specials=("<pad>", "<x>"))
    with pytest.raises(ValueError, match="at least the padding"):
        EventVocab(["a"], specials=())


def test_vocab_path_override(tmp_path, dataset):
    EventVocab(["zebra", "birth", "job"]).save(tmp_path / "vocab.json")
    data = make_module(tmp_path, dataset, vocab_path=tmp_path / "vocab.json")
    with pytest.raises(ValueError, match="absent from"):
        data.prepare_data()

    data = make_module(tmp_path, dataset, vocab_path=tmp_path / "vocab.json", on_unknown_event="drop")
    data.cache_dir = tmp_path / "cache2"
    data.prepare_data()
    data.setup()
    # The override's own, non-alphabetical order is honoured.
    assert data.vocab.events == ["zebra", "birth", "job"]


def test_vocab_of_another_architecture_is_rejected(tmp_path, dataset):
    EventVocab(["birth"], specials=("<pad>", "<no_event>")).save(tmp_path / "vocab.json")
    data = make_module(tmp_path, dataset, vocab_path=tmp_path / "vocab.json")
    with pytest.raises(ValueError, match="not compatible"):
        data.prepare_data()


def test_missing_birthdate_and_negative_age_dropped(tmp_path, dataset):
    event_paths, _ = dataset
    # "c" has no birthdate; "b" gains an event dated before birth.
    person_path = write_persons(tmp_path / "partial.parquet", {"a": "2000-01-01", "b": "1990-06-15"})
    write_parquet(tmp_path / "events_c.parquet", [("b", "1980-01-01", "prenatal")], notes_sql="{'a': 1}")
    data = prepared(
        tmp_path, ([*event_paths, str(tmp_path / "events_c.parquet")], str(person_path)), val_split_fraction=0.5
    )
    stats = meta(data)["stats"]
    assert stats["n_no_birthdate"] == 3 and stats["n_persons_no_birthdate"] == 1
    assert stats["n_before_min_time"] == 1
    assert meta(data)["n_persons"] == 2
    assert "prenatal" not in data.vocab.events

    strict = make_module(
        tmp_path, ([*event_paths], str(person_path)), on_missing_birthdate="error", val_split_fraction=0.5
    )
    strict.cache_dir = tmp_path / "strict"
    with pytest.raises(ValueError, match="no usable BIRTHDATE"):
        strict.prepare_data()


def test_duplicate_person_rows_raise(tmp_path, dataset):
    event_paths, _ = dataset
    values = "('a', DATE '2000-01-01'), ('a', DATE '2001-01-01')"
    path = tmp_path / "dup.parquet"
    duckdb.connect().execute(f"COPY (SELECT * FROM (VALUES {values}) AS t(ID, BIRTHDATE)) TO '{path}' (FORMAT PARQUET)")
    data = make_module(tmp_path, (event_paths, str(path)))
    with pytest.raises(ValueError, match="appear more than once"):
        data.prepare_data()


def test_min_events_filter(tmp_path, dataset):
    data = prepared(tmp_path, dataset, min_events=4, val_split_fraction=0.5)
    # "c" has three events, "a" and "b" have four each.
    assert meta(data)["n_persons"] == 2
    assert meta(data)["stats"]["n_dropped_min_events"] == 1


def test_prepare_data_is_idempotent(tmp_path, dataset, monkeypatch):
    data = make_module(tmp_path, dataset)
    data.prepare_data()
    monkeypatch.setattr(
        "litgpt.life_events.data.source._connect_duckdb",
        lambda *a, **k: pytest.fail("prepare_data rebuilt an existing cache"),
    )
    data.prepare_data()


def test_val_split_is_by_individual(tmp_path, dataset):
    data = prepared(tmp_path, dataset)
    train_idx = set(data.train_dataset.indices.tolist())
    val_idx = set(data.val_dataset.indices.tolist())
    assert not train_idx & val_idx
    ids = np.asarray(data._store.ids)
    assert not {ids[i] for i in train_idx} & {ids[i] for i in val_idx}


def test_empty_val_split_raises(tmp_path, dataset):
    data = make_module(tmp_path, dataset, val_split_fraction=0.01)
    data.prepare_data()
    with pytest.raises(ValueError, match="holds out no individuals"):
        data.setup()


def test_dataloader_is_picklable(tmp_path, dataset):
    data = prepared(tmp_path, dataset)
    pickle.loads(pickle.dumps(data.train_dataloader().collate_fn))


def test_tokens_are_stored_narrow(tmp_path, dataset):
    data = prepared(tmp_path, dataset)
    assert np.load(data.cache_dir / "tokens.npy", mmap_mode="r").dtype == np.uint16
    assert np.load(data.cache_dir / "times.npy", mmap_mode="r").dtype == np.int32


def test_token_dtype_guard_widens_for_a_huge_vocabulary():
    assert _token_dtype(EventVocab([f"e{i}" for i in range(10)])) == np.uint16
    # uint16 wraps silently on overflow, which would remap event 65536 onto `<pad>`.
    assert _token_dtype(EventVocab([f"e{i}" for i in range(np.iinfo(np.uint16).max)])) == np.int32


def test_streaming_write_matches_single_shot(tmp_path, dataset, monkeypatch):
    """Many small batches must produce byte-identical output to one big fetch."""
    notes_fields = {"weight": NoteField("weight")}
    reference = prepared(tmp_path / "ref", dataset, notes_fields=notes_fields)
    monkeypatch.setattr("litgpt.life_events.data.source.STREAM_ROWS", 3)
    chunked = prepared(tmp_path / "chunked", dataset, notes_fields=notes_fields)
    for name in ("tokens.npy", "times.npy", "offsets.npy", "ids.npy", "note_weight.npy"):
        assert np.array_equal(
            np.load(reference.cache_dir / name, mmap_mode="r"), np.load(chunked.cache_dir / name, mmap_mode="r")
        ), name


def test_pyarrow_free_fallback_matches_streaming(tmp_path, dataset, monkeypatch):
    """The no-pyarrow path is slower, not different."""
    notes_fields = {"weight": NoteField("weight")}
    reference = prepared(tmp_path / "ref", dataset, notes_fields=notes_fields)
    monkeypatch.setattr("litgpt.life_events.data.source._PYARROW_AVAILABLE", False)
    fallback = prepared(tmp_path / "fallback", dataset, notes_fields=notes_fields)
    for name in ("tokens.npy", "times.npy", "note_weight.npy"):
        assert np.array_equal(
            np.load(reference.cache_dir / name, mmap_mode="r"), np.load(fallback.cache_dir / name, mmap_mode="r")
        ), name


def test_tie_break_stays_alphabetical_under_a_reordered_vocabulary(tmp_path, dataset):
    """Co-occurring events sort by alphabetical rank, not by token id.

    Individual "a" has `job` and `move` on the same day, from two different datasets.
    This vocabulary gives `move` the *lower* token, so sorting by token would reverse
    them; only the separately carried alphabetical rank gets it right.
    """
    EventVocab(["move", "job", "study", "marriage", "birth"]).save(tmp_path / "vocab.json")
    data = prepared(tmp_path, dataset, vocab_path=tmp_path / "vocab.json")
    assert data.vocab.stoi["move"] < data.vocab.stoi["job"]

    sequence = _sequence_of(data, "a")
    times = sequence["times"].tolist()
    tied = [data.vocab.itos[t] for t, time in zip(sequence["tokens"].tolist(), times) if times.count(time) > 1]
    assert tied == ["job", "move"]


def test_integer_ids_are_kept_as_integers(tmp_path, int_dataset):
    data = prepared(tmp_path, int_dataset)
    assert meta(data)["id_type"] == "BIGINT"
    ids = np.load(data.cache_dir / "ids.npy")
    assert ids.dtype == np.int64
    # Numeric order, not the lexicographic ['10', '100', '2'] a VARCHAR cast would give.
    assert ids.tolist() == [2, 10, 100]


def test_mixed_id_types_across_datasets_fall_back_to_varchar(tmp_path, int_dataset):
    """VARCHAR is the only type that cannot lose information when datasets disagree."""
    (int_events,), persons = int_dataset
    string_events = write_parquet(
        tmp_path / "string_events.parquet", [("10", "2011-07-07", "marriage")], notes_sql="{'b': 'x'}"
    )
    data = prepared(tmp_path, ([int_events, str(string_events)], persons))
    assert meta(data)["id_type"] == "VARCHAR"
    assert np.load(data.cache_dir / "ids.npy").dtype.kind == "U"
    # The string-keyed row still joined onto the integer-keyed individual.
    assert "marriage" in data.vocab.events


def test_person_table_type_also_decides(tmp_path, int_dataset):
    (int_events,), _ = int_dataset
    persons = write_persons(tmp_path / "str_persons.parquet", {"100": "2000-01-01", "10": "1990-06-15"})
    data = prepared(tmp_path, ([int_events], str(persons)), val_split_fraction=0.5)
    assert meta(data)["id_type"] == "VARCHAR"


def test_wide_integer_id_types_are_not_narrowed(tmp_path):
    """UBIGINT and HUGEINT can hold values BIGINT cannot; narrowing would merge people."""
    assert {"UBIGINT", "HUGEINT", "UHUGEINT"}.isdisjoint(_INTEGER_ID_TYPES)
    rows = [
        (1, "2000-01-01", "birth"),
        (1, "2010-01-01", "job"),
        (2, "1995-01-01", "birth"),
        (2, "2015-01-01", "job"),
    ]
    events = write_int_id_parquet(tmp_path / "ubig.parquet", rows, id_type="UBIGINT")
    persons = write_int_persons(tmp_path / "ubig_p.parquet", {1: "2000-01-01", 2: "1995-01-01"}, id_type="UBIGINT")
    data = prepared(tmp_path, ([str(events)], str(persons)), val_split_fraction=0.5)
    assert meta(data)["id_type"] == "VARCHAR"


# ---- NOTES fields ----------------------------------------------------------------------


def test_note_field_from_one_dataset_is_filled_in_the_other(tmp_path, dataset):
    """`weight` exists only in the second dataset's struct; the first gets `fill`."""
    data = prepared(tmp_path, dataset, notes_fields={"weight": NoteField("weight", "FLOAT", fill=-1.0)})
    assert (data.cache_dir / "note_weight.npy").is_file()
    assert meta(data)["features"] == {"weight": {"field": "weight", "dtype": "FLOAT", "fill": -1.0}}
    assert data._store.features["note_weight"].dtype == np.float32
    assert data._store.feature_fills == {"note_weight": -1.0}

    sequence = _sequence_of(data, "a")
    events = [data.vocab.itos[t] for t in sequence["tokens"].tolist()]
    # Only `move` comes from the dataset that carries `weight`.
    expected = [2.0 if event == "move" else -1.0 for event in events]
    assert sequence["note_weight"].tolist() == expected


def test_note_field_from_a_map_and_a_json_string(tmp_path):
    persons = str(write_persons(tmp_path / "persons.parquet", {"a": "2000-01-01", "b": "1990-01-01"}))
    as_map = write_parquet(
        tmp_path / "map.parquet",
        [("a", "2000-01-01", "birth"), ("b", "1990-01-01", "birth")],
        notes_sql="MAP {'income': '12'}",
    )
    as_json = write_parquet(
        tmp_path / "json.parquet",
        [("a", "2020-01-01", "job"), ("b", "2010-01-01", "job")],
        notes_sql="""'{"income": 34}'""",
    )
    data = prepared(
        tmp_path,
        ([str(as_map), str(as_json)], persons),
        val_split_fraction=0.5,
        notes_fields={"income": NoteField("income", "BIGINT")},
    )
    sequence = _sequence_of(data, "a")
    assert sequence["note_income"].dtype == np.int64
    assert sequence["note_income"].tolist() == [12, 34]


def test_note_field_that_does_not_cast_raises(tmp_path, dataset):
    data = make_module(tmp_path, dataset, notes_fields={"partner": NoteField("partner", "INTEGER")})
    with pytest.raises(duckdb.ConversionException):
        data.prepare_data()


def test_note_features_take_part_in_distinct(tmp_path):
    """Two otherwise identical events that disagree on an attribute are both kept."""
    persons = str(write_persons(tmp_path / "persons.parquet", {"a": "2000-01-01", "b": "1990-01-01"}))
    rows = [("a", "2000-01-01", "birth"), ("a", "2020-01-01", "job"), ("b", "1990-01-01", "birth")]
    first = write_parquet(tmp_path / "one.parquet", [*rows, ("b", "2010-01-01", "job")], notes_sql="{'x': 1}")
    second = write_parquet(tmp_path / "two.parquet", [*rows, ("b", "2010-01-01", "job")], notes_sql="{'x': 2}")
    kwargs = dict(val_split_fraction=0.5)
    plain = prepared(tmp_path / "plain", ([str(first), str(second)], persons), **kwargs)
    noted = prepared(
        tmp_path / "noted",
        ([str(first), str(second)], persons),
        notes_fields={"x": NoteField("x", "INTEGER")},
        **kwargs,
    )
    assert meta(plain)["n_events"] == 4
    assert meta(noted)["n_events"] == 8
    # Identical events stay in a deterministic order, the note value breaking the tie.
    assert _sequence_of(noted, "a")["note_x"].tolist() == [1, 2, 1, 2]


def test_note_field_validation():
    with pytest.raises(ValueError, match="dtype must be one of"):
        NoteField("x", "VARCHAR")
    with pytest.raises(ValueError, match="not representable"):
        NoteField("x", "TINYINT", fill=1000)
    with pytest.raises(ValueError, match="not representable"):
        NoteField("x", "INTEGER", fill=0.5)


def test_note_field_names_must_be_identifiers(tmp_path, dataset):
    data = make_module(tmp_path, dataset, notes_fields={"not a name": NoteField("x")})
    with pytest.raises(ValueError, match="valid identifier"):
        data.prepare_data()


def test_notes_fields_accepts_plain_dicts(tmp_path, dataset):
    data = make_module(tmp_path, dataset, notes_fields={"weight": {"field": "weight", "dtype": "FLOAT"}})
    assert data.notes_fields == {"weight": NoteField("weight", "FLOAT")}


def test_plain_module_uses_its_own_specials(tmp_path, dataset):
    data = prepared(tmp_path, dataset)
    assert PlainLifeEvents.specials == ("<pad>",)
    assert data.vocab.first_event_id == 1
    assert data.vocab_size == len(data.vocab.events) + 1


# ---- the time axis -------------------------------------------------------------------


def test_epoch_origin_needs_no_person_dataset(tmp_path, dataset):
    """A model on absolute time has no use for BIRTHDATE, so none is required."""
    event_paths, _ = dataset
    data = prepared(tmp_path, (event_paths, None), time_origin="epoch", val_split_fraction=1 / 3)
    assert data._store.time_origin == "epoch"
    assert meta(data)["sources"]["persons"] is None
    # Individuals come from the events themselves, and all three are kept.
    assert sorted(np.asarray(data._store.ids).tolist()) == ["a", "b", "c"]

    epoch = np.datetime64("1970-01-01")
    times = _sequence_of(data, "a")["times"].tolist()
    assert times[0] == (np.datetime64("2000-01-01") - epoch).astype(int)
    # Nothing is dropped for being "negative": before the reference date just means earlier.
    assert meta(data)["stats"]["n_before_min_time"] == 0
    assert meta(data)["filters"]["min_time"] is None


def test_origin_can_be_an_arbitrary_reference_date(tmp_path, dataset):
    event_paths, _ = dataset
    data = prepared(tmp_path, (event_paths, None), time_origin="2010-01-01", val_split_fraction=1 / 3)
    reference = np.datetime64("2010-01-01")
    times = _sequence_of(data, "a")["times"].tolist()
    assert times[0] == (np.datetime64("2000-01-01") - reference).astype(int) < 0
    assert times[-1] == (np.datetime64("2022-03-01") - reference).astype(int) > 0


def test_same_events_on_two_axes_differ_only_by_a_shift(tmp_path, dataset):
    """Age and absolute time are the same sequence measured from different places."""
    event_paths, person_path = dataset
    age = prepared(tmp_path / "age", (event_paths, person_path), val_split_fraction=1 / 3)
    absolute = prepared(tmp_path / "abs", (event_paths, None), time_origin="epoch", val_split_fraction=1 / 3)
    birth = (np.datetime64("2000-01-01") - np.datetime64("1970-01-01")).astype(int)
    assert _sequence_of(age, "a")["tokens"].tolist() == _sequence_of(absolute, "a")["tokens"].tolist()
    shifted = [t + birth for t in _sequence_of(age, "a")["times"].tolist()]
    assert shifted == _sequence_of(absolute, "a")["times"].tolist()


def _index_of(data, person_id):
    return list(np.asarray(data._store.ids)).index(person_id)


def test_both_axes_from_one_cache(tmp_path, dataset):
    """A model needing age *and* calendar time re-bases; nothing is stored twice."""
    aged = prepared(tmp_path / "age", dataset, val_split_fraction=1 / 3)
    absolute = prepared(tmp_path / "abs", (dataset[0], None), time_origin="epoch", val_split_fraction=1 / 3)

    store, i = aged._store, _index_of(aged, "a")
    ages = store.sequence(i)["times"]
    # "a" was born 2000-01-01, so that is where their age axis starts.
    assert store.time_offsets[i] == (np.datetime64("2000-01-01") - np.datetime64("1970-01-01")).astype(int)
    assert store.times_from(i, ages, "epoch").tolist() == _sequence_of(absolute, "a")["times"].tolist()
    # And onward to any other reference date, or back to where it started.
    from_2010 = store.times_from(i, ages, "2010-01-01")
    reference = (np.datetime64("2010-01-01") - np.datetime64("1970-01-01")).astype(int)
    assert from_2010.tolist() == [t - reference for t in store.times_from(i, ages, "epoch").tolist()]
    assert store.times_from(i, ages, "birth") is ages

    # Every individual has their own shift, which is the whole point of storing it per person.
    assert len({int(o) for o in np.asarray(store.time_offsets)}) == len(store)


def test_a_cache_without_birthdates_cannot_invent_age(tmp_path, dataset):
    absolute = prepared(tmp_path, (dataset[0], None), time_origin="epoch", val_split_fraction=1 / 3)
    store = absolute._store
    times = store.sequence(0)["times"]
    # A constant shift for everyone: the epoch is the same date for every individual.
    assert set(np.asarray(store.time_offsets).tolist()) == {0}
    with pytest.raises(ValueError, match="does not know any person's birthdate"):
        store.times_from(0, times, "birth")
    with pytest.raises(ValueError, match="must be 'birth'"):
        store.times_from(0, times, "the nineties")


def test_birth_origin_still_requires_a_person_dataset(tmp_path, dataset):
    event_paths, _ = dataset
    with pytest.raises(ValueError, match="person_path` is required"):
        make_module(tmp_path, (event_paths, None))


def test_unknown_origin_is_rejected(tmp_path, dataset):
    data = make_module(tmp_path, dataset, time_origin="yesterday")
    with pytest.raises(ValueError, match="time_origin must be"):
        data.prepare_data()


def test_min_time_is_opt_in_away_from_birth(tmp_path, dataset):
    """Every fixture event post-dates the epoch, so `min_time=0` there changes nothing."""
    kept = prepared(tmp_path / "kept", dataset, time_origin="epoch", min_time=0, val_split_fraction=1 / 3)
    assert meta(kept)["stats"]["n_before_min_time"] == 0
    assert meta(kept)["n_events"] == 11

    # A reference date in the middle of the data does cut, and only there.
    late = prepared(tmp_path / "late", dataset, time_origin="2010-01-01", min_time=0, val_split_fraction=0.5)
    assert np.asarray(late._store.times).min() >= 0
    assert meta(late)["stats"]["n_before_min_time"] == 4
    assert meta(late)["n_events"] == 11 - 4 - 1  # "c" loses all but one event, so min_events drops them


def test_duckdb_settings_reach_the_connection(tmp_path, dataset, monkeypatch):
    """The three `duckdb_*` fields are the knobs for a corpus larger than memory."""
    from litgpt.life_events.data import source

    seen = {}
    real = source._connect_duckdb

    def spy(*args, **kwargs):
        con = real(*args, **kwargs)
        seen.update(
            zip(
                ("threads", "memory_limit", "temp_directory"),
                con.execute(
                    "SELECT current_setting('threads'), current_setting('memory_limit'), "
                    "current_setting('temp_directory')"
                ).fetchone(),
            )
        )
        return con

    monkeypatch.setattr(source, "_connect_duckdb", spy)
    spill = tmp_path / "spill"
    data = prepared(tmp_path, dataset, duckdb_threads=2, duckdb_memory_limit="512MB", duckdb_temp_directory=spill)
    assert seen["threads"] == 2
    assert seen["memory_limit"] == "488.2 MiB"  # DuckDB's own rendering of 512MB
    assert seen["temp_directory"] == str(spill)
    # And the cache it produced is the usual one.
    assert meta(data)["n_events"] == 11


def test_static_fields_become_tokens_after_the_events(tmp_path, dataset, static_dataset):
    plain = prepared(tmp_path / "plain", dataset)
    data = prepared(tmp_path / "static", static_dataset, static_fields=["GENDER", "MIGRATION"])
    # Statics come last, so they renumber no event.
    assert data.vocab.events == plain.vocab.events
    assert data.vocab.first_static_id == plain.vocab.vocab_size
    assert data.vocab.statics == [
        "GENDER=<unknown>",
        "GENDER=F",
        "GENDER=M",
        "MIGRATION=<unknown>",
        "MIGRATION=none",
        "MIGRATION=second",
    ]
    assert meta(data)["statics"] == ["GENDER", "MIGRATION"]

    store = data._store
    assert store.static_fields == ("GENDER", "MIGRATION")
    assert store.statics.shape == (3, 2)
    labels = {str(i): data.vocab.decode(row.tolist()) for i, row in zip(store.ids, np.asarray(store.statics))}
    assert labels == {
        "a": ["GENDER=F", "MIGRATION=none"],
        "b": ["GENDER=M", "MIGRATION=second"],
        "c": ["GENDER=<unknown>", "MIGRATION=none"],
    }
    # The event stream itself is untouched.
    assert np.array_equal(np.asarray(store.tokens), np.asarray(plain._store.tokens))


def test_without_static_fields_the_statics_are_empty(tmp_path, dataset):
    data = prepared(tmp_path, dataset)
    assert data._store.statics.shape == (3, 0)
    assert data.vocab.statics == [] and data.vocab.static_ids == []


def test_loaded_vocab_maps_unseen_static_values_to_unknown(tmp_path, dataset, static_dataset):
    trained = prepared(tmp_path / "trained", static_dataset, static_fields=["GENDER"])
    event_paths, _ = dataset
    persons = write_persons(tmp_path / "other.parquet", statics={"GENDER": {"a": "X", "b": "M", "c": "F"}})
    data = prepared(
        tmp_path / "other",
        (event_paths, str(persons)),
        static_fields=["GENDER"],
        vocab_path=trained.cache_dir / "vocab.json",
    )
    labels = {str(i): data.vocab.decode(row.tolist())[0] for i, row in zip(data._store.ids, data._store.statics)}
    assert labels == {"a": "GENDER=<unknown>", "b": "GENDER=M", "c": "GENDER=F"}

    # A vocabulary that never had the field cannot place it at all.
    with pytest.raises(ValueError, match="static field 'MIGRATION'"):
        prepared(
            tmp_path / "missing",
            static_dataset,
            static_fields=["MIGRATION"],
            vocab_path=trained.cache_dir / "vocab.json",
        )


def test_static_fields_need_a_person_dataset_and_a_matching_cache(tmp_path, dataset, static_dataset):
    event_paths, _ = dataset
    with pytest.raises(ValueError, match="`person_path` is required with `static_fields`"):
        PlainLifeEvents(event_paths=event_paths, time_origin="epoch", static_fields=["GENDER"])

    prepared(tmp_path, static_dataset, static_fields=["GENDER"])
    stale = make_module(tmp_path, static_dataset)
    with pytest.raises(ValueError, match="was built with static_fields"):
        stale.setup()

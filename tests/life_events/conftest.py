# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
import pytest

from .helpers import STATICS, write_int_id_parquet, write_int_persons, write_parquet, write_persons


@pytest.fixture()
def dataset(tmp_path):
    """Two event datasets with deliberately different NOTES struct schemas."""
    pytest.importorskip("duckdb")
    events_a = write_parquet(
        tmp_path / "events_a.parquet",
        [
            ("a", "2000-01-01", "birth"),
            ("a", "2018-09-01", "study"),
            ("a", "2022-03-01", "job"),
            ("b", "1990-06-15", "birth"),
            ("b", "2010-01-01", "job"),
            ("b", "2015-05-05", "marriage"),
            ("c", "1985-03-20", "birth"),
            ("c", "2005-01-01", "job"),
        ],
        notes_sql="{'a': 1}",
    )
    events_b = write_parquet(
        tmp_path / "events_b.parquet",
        [("a", "2022-03-01", "move"), ("b", "2020-11-11", "move"), ("c", "2011-07-07", "marriage")],
        notes_sql="{'partner': 'x', 'weight': 2.0}",
    )
    return [str(events_a), str(events_b)], str(write_persons(tmp_path / "persons.parquet"))


@pytest.fixture()
def static_dataset(tmp_path, dataset):
    """`dataset`, with a person table that also carries `GENDER` and `MIGRATION`."""
    event_paths, _ = dataset
    return event_paths, str(write_persons(tmp_path / "persons_static.parquet", statics=STATICS))


@pytest.fixture()
def int_dataset(tmp_path):
    """Integer IDs chosen so lexicographic and numeric order disagree: 2 < 10 < 100."""
    pytest.importorskip("duckdb")
    rows = [
        (100, "2000-01-01", "birth"),
        (100, "2018-09-01", "study"),
        (10, "1990-06-15", "birth"),
        (10, "2010-01-01", "job"),
        (2, "1985-03-20", "birth"),
        (2, "2005-01-01", "job"),
    ]
    events = write_int_id_parquet(tmp_path / "int_events.parquet", rows)
    persons = write_int_persons(
        tmp_path / "int_persons.parquet", {100: "2000-01-01", 10: "1990-06-15", 2: "1985-03-20"}
    )
    return [str(events)], str(persons)

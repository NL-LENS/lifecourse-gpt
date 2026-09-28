# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Architecture-agnostic joint life event sequence data caching.

All heavy data-processing operations are performed here.

Events arrive as one or more Parquet datasets sharing the schema
`ID | TIME | EVENT | NOTES`, plus a person-attributes dataset carrying `BIRTHDATE` and,
optionally, static attributes such as gender. `build_event_cache` merges the datasets,
places every event on a time axis, orders co-occurring events alphabetically and writes
one flat, memory-mapped event stream plus per-person offsets and static tokens. See
`store.SequenceStore` for the layout.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path
from typing import Any, Literal

import numpy as np

from litgpt.constants import _DUCKDB_AVAILABLE, _PYARROW_AVAILABLE
from litgpt.life_events.data.vocab import UNKNOWN_STATIC, EventVocab, static_label

__all__ = ["CACHE_VERSION", "TIME_UNIT", "NoteField", "build_event_cache", "summarize", "time_expr"]

CACHE_VERSION = 5
"""Version 4 added `time_offsets.npy`, which re-bases `times.npy` onto any other origin.
Version 5 added `statics.npy`, the static person-level tokens."""

TIME_UNIT = "day"
"""The unit every time coordinate is expressed in.

Whole days rather than a float: a model that compares event times for equality (Delphi's
tie mask does) needs a discrete unit, and a date-resolution silver layer has nothing
finer to offer."""

BIRTH_ORIGIN = "birth"
EPOCH_ORIGIN = "epoch"


def time_expr(origin: str) -> str:
    """SQL placing an event on the time axis, in whole days from `origin`.

    The one place that decides what a time coordinate *means*, so no two queries can
    drift apart on it.

    Args:
        origin: `"birth"` for age, i.e. days since the individual's `BIRTHDATE`;
            `"epoch"` for days since 1970-01-01; or an ISO date to count from.
            Only `"birth"` reads the person dataset.
    """
    if origin == BIRTH_ORIGIN:
        return f"date_diff('{TIME_UNIT}', p.birthdate, s.time)"
    reference = "1970-01-01" if origin == EPOCH_ORIGIN else origin
    try:
        datetime.strptime(reference, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"time_origin must be {BIRTH_ORIGIN!r}, {EPOCH_ORIGIN!r} or an ISO date (YYYY-MM-DD), got {origin!r}"
        ) from None
    return f"date_diff('{TIME_UNIT}', DATE '{reference}', s.time)"


def needs_birthdate(origin: str) -> bool:
    """Whether `origin` requires the person dataset. Only age does."""
    return origin == BIRTH_ORIGIN


def time_offset_expr(origin: str) -> str:
    """SQL for one person's time origin, in whole days since 1970-01-01.

    Every origin is a *date*: a birthdate, the epoch itself, or a fixed reference. So the
    origins differ by a constant per person, and this is that constant. `store.SequenceStore`
    transforms one stored array to any other origin instead of caching a second copy of it.
    See `SequenceStore.time_offsets`.
    """
    if origin == BIRTH_ORIGIN:
        return f"date_diff('{TIME_UNIT}', DATE '1970-01-01', p.birthdate)"
    reference = "1970-01-01" if origin == EPOCH_ORIGIN else origin
    time_expr(origin)  # validates `origin`
    return f"date_diff('{TIME_UNIT}', DATE '1970-01-01', DATE '{reference}')"


STREAM_ROWS = 1 << 20
"""Rows per batch when writing the event cache."""

NOTE_DTYPES: dict[str, Any] = {
    "TINYINT": np.int8,
    "SMALLINT": np.int16,
    "INTEGER": np.int32,
    "BIGINT": np.int64,
    "FLOAT": np.float32,
    "DOUBLE": np.float64,
}
"""SQL types a `NoteField` may be read as, with the array type it is stored in.

Numeric only for now. A categorical attribute (a string) would need a vocabulary of its
own. It can be added here without touching the rest of the pipeline."""


@dataclass(frozen=True)
class NoteField:
    """One attribute read out of the `NOTES` column of every event.

    `NOTES` differs between datasets. A field absent from a dataset or null on a row takes `fill`.

    Args:
        field: Key in `NOTES`. Works whether `NOTES` is a struct, a map or a JSON string.
        dtype: SQL type to cast to, one of `NOTE_DTYPES`. A value that does not cast
            raises rather than being filled, so a mistyped field fails loudly.
        fill: Value for events that carry no such field.
    """

    field: str
    dtype: str = "DOUBLE"
    fill: float = 0.0

    def __post_init__(self) -> None:
        if self.dtype not in NOTE_DTYPES:
            raise ValueError(f"NoteField dtype must be one of {list(NOTE_DTYPES)}, got {self.dtype!r}")
        if np.issubdtype(NOTE_DTYPES[self.dtype], np.integer):
            info = np.iinfo(NOTE_DTYPES[self.dtype])
            if self.fill != int(self.fill) or not info.min <= self.fill <= info.max:
                raise ValueError(f"fill={self.fill} is not representable as {self.dtype}")

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(NOTE_DTYPES[self.dtype])


def note_column(name: str) -> str:
    """The name a note feature carries in SQL, on disk (`<name>.npy`) and in batches."""
    return f"note_{name}"


def _connect_duckdb(
    threads: int | None = None, memory_limit: str | None = None, temp_directory: Path | None = None
) -> Any:
    """Opens an in-memory DuckDB connection, importing the package lazily.

    `litgpt.data.__init__` imports this module eagerly, so a module-level `import duckdb`
    would break `import litgpt.data` for everyone who has not installed the `extra` group.
    """
    if not _DUCKDB_AVAILABLE:
        raise ModuleNotFoundError(
            f"{_DUCKDB_AVAILABLE!s} The life-event data modules read Parquet through DuckDB. "
            "Install it with `pip install duckdb`, or the whole extras group with "
            "`pip install 'litgpt[extra]'`."
        )
    import duckdb

    con = duckdb.connect()
    if threads is not None:
        con.execute(f"SET threads TO {int(threads)}")
    if memory_limit is not None:
        con.execute("SET memory_limit TO ?", [memory_limit])
    if temp_directory is not None:
        # Lets the sort and the join spill to disk on a corpus larger than RAM.
        con.execute("SET temp_directory TO ?", [str(temp_directory)])
    return con


# These quoting functions prevent SQL injection
def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _expand_entry(entry: str) -> str:
    """Turns a directory into the glob DuckDB needs; files and globs pass through."""
    path = Path(entry)
    return str(path / "**" / "*.parquet") if path.is_dir() else str(path)


def _column_types(con: Any, entry: str) -> dict[str, Any]:
    """Column name -> DuckDB type of one dataset. A footer read, not a scan."""
    relation = con.read_parquet(_expand_entry(str(entry)))
    return dict(zip(relation.columns, relation.types))


_INTEGER_ID_TYPES = frozenset({"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER"})
"""Types safe to carry as `BIGINT`. `HUGEINT` and `UBIGINT` are deliberately absent:
both can hold values that do not fit, and a silent overflow would merge two individuals."""


def _id_sql_type(con: Any, entries: Sequence[str], person_path: str | None, id_column: str) -> str:
    """`BIGINT` when every dataset stores `ID` as an integer, `VARCHAR` otherwise.

    Two "silver" layer event datasets may disagree on the type of `ID`, and `VARCHAR` is the one common
    type that cannot lose information.
    """
    for entry in [*entries, *([person_path] if person_path is not None else [])]:
        if str(_column_types(con, entry).get(id_column)) not in _INTEGER_ID_TYPES:
            return "VARCHAR"
    return "BIGINT"


def _note_expr(notes_type: Any, notes_column: str, spec: NoteField) -> str:
    """SQL reading `spec.field` out of one dataset's `NOTES`, cast and filled.

    Dispatches on how that dataset stores `NOTES`, and reads only the one field: binding
    the whole column is exactly what `_event_relation` avoids. A dataset without the
    field, or without a `NOTES` column at all, gets a constant null that is filled with `fill`.
    """
    column = _quote_ident(notes_column)
    key = _quote_literal(spec.field)
    kind = None if notes_type is None else notes_type.id
    if kind == "struct" and spec.field in dict(notes_type.children):
        value = f"struct_extract({column}, {key})"
    elif kind == "map":
        value = f"element_at({column}, {key})[1]"
    elif kind in ("varchar", "json"):
        value = f"json_extract_string({column}, {_quote_literal('$.' + json.dumps(spec.field))})"
    else:
        value = "NULL"
    return f"COALESCE(CAST({value} AS {spec.dtype}), CAST({spec.fill!r} AS {spec.dtype}))"


def _event_relation(
    con: Any,
    entries: Sequence[str],
    id_column: str,
    time_column: str,
    event_column: str,
    id_type: str,
    notes_column: str = "NOTES",
    notes_fields: Mapping[str, NoteField] | None = None,
) -> Any:
    """One projected relation per event dataset, merged with `UNION ALL`.

    Uses a projection because `NOTES` is a dict whose struct schema
    differs between datasets, and binding it across files is at best wasted work and at
    worst wrong: DuckDB binds a plain file list against the *first* file's schema, so a
    second dataset's extra `NOTES` fields are silently dropped. Projecting inside each
    arm means `NOTES` as a whole is never bound; requested `notes_fields` are extracted
    one by one, per dataset, against that dataset's own schema.

    The explicit casts matter because two datasets may disagree on a column's
    type, and every arm has to agree before they can be unioned. `id_type` comes from
    `_id_sql_type`.
    """
    notes_fields = notes_fields or {}
    relations = []
    for entry in entries:
        projection = [
            f"CAST({_quote_ident(id_column)} AS {id_type}) AS id",
            f"CAST({_quote_ident(time_column)} AS DATE) AS time",
            f"CAST({_quote_ident(event_column)} AS VARCHAR) AS event",
        ]
        if notes_fields:
            notes_type = _column_types(con, entry).get(notes_column)
            projection += [
                f"{_note_expr(notes_type, notes_column, spec)} AS {_quote_ident(note_column(name))}"
                for name, spec in notes_fields.items()
            ]
        relations.append(con.read_parquet(_expand_entry(entry)).project(", ".join(projection)))
    # `Relation.union` is UNION ALL; duplicates are collapsed later, and only optionally.
    return reduce(lambda left, right: left.union(right), relations)


def _to_numpy(column: Any, dtype: Any, fill: float = 0) -> np.ndarray:
    """Normalizes a `fetchnumpy` column, which is a `MaskedArray` when DuckDB thinks the
    column is nullable -- even when it holds no nulls."""
    if isinstance(column, np.ma.MaskedArray):
        column = column.filled(fill)
    return np.asarray(column).astype(dtype, copy=False)


def _token_dtype(vocab: EventVocab) -> Any:
    """The narrowest integer type the vocabulary fits in.

    Halves `tokens.npy` for any realistic event vocabulary -- Delphi's is 1270 -- and
    costs nothing at read time, since datasets widen to int64 anyway. The guard is not
    decoration: `uint16` silently wraps on overflow, which would remap event 65536 onto
    the padding token.
    """
    return np.uint16 if vocab.vocab_size <= np.iinfo(np.uint16).max else np.int32


def _stream(con: Any, query: str, dtypes: Mapping[str, Any]) -> Iterator[dict[str, np.ndarray]]:
    """Yields `{column: chunk}` dicts, in batches where pyarrow makes that possible."""
    result = con.execute(query)
    if not _PYARROW_AVAILABLE:
        columns = result.fetchnumpy()
        yield {name: _to_numpy(columns[name], dtype) for name, dtype in dtypes.items()}
        return
    # `to_arrow_reader` since DuckDB 1.3; `fetch_record_batch` is the older spelling,
    # deprecated in 1.5 but the only one available on the `duckdb>=1.0` floor.
    reader = getattr(result, "to_arrow_reader", None) or result.fetch_record_batch
    for batch in reader(STREAM_ROWS):
        yield {name: batch.column(name).to_numpy(zero_copy_only=False) for name in dtypes}


def _write_event_stream(
    con: Any, cache_dir: Path, n_events: int, token_dtype: Any, notes_fields: Mapping[str, NoteField]
) -> int:
    """Streams the ordered event stream into pre-sized `.npy` files.

    If pyarrow is not available, the fallback is `fetchnumpy` which builds the whole result in Python before
    anything is written to the file. Writing through `open_memmap` keeps Python's share to one batch.
    """
    dtypes = {"time": np.int32, "token": token_dtype}
    dtypes |= {note_column(name): spec.np_dtype for name, spec in notes_fields.items()}
    files = {"time": "times.npy", "token": "tokens.npy"}
    files |= {note_column(name): f"{note_column(name)}.npy" for name in notes_fields}
    # `rank`, not `token`, is the tie-break: it is the alphabetical position, which an
    # overridden vocabulary's token order need not match. Ordering by `person_idx` is
    # ordering by id, since the surrogate was assigned in id order. The note columns
    # come last only to make the order of otherwise identical events deterministic.
    notes = [f"e.{_quote_ident(note_column(name))}" for name in notes_fields]
    query = (
        f"SELECT {', '.join(['e.time', 'e.token', *notes])} FROM _events e JOIN _keep k USING (person_idx) "
        f"ORDER BY {', '.join(['e.person_idx', 'e.time', 'e.rank', *notes])}"
    )
    arrays = {
        name: np.lib.format.open_memmap(cache_dir / files[name], mode="w+", dtype=dtype, shape=(n_events,))
        for name, dtype in dtypes.items()
    }
    try:
        written = 0
        for chunk in _stream(con, query, dtypes):
            end = written + chunk["time"].size
            if end > n_events:
                raise RuntimeError(f"event stream is longer than the {n_events} rows the counts promised")
            for name, array in arrays.items():
                array[written:end] = chunk[name]
            written = end
        for array in arrays.values():
            array.flush()
        return written
    finally:
        # Drops the last references, which is what closes the underlying mappings.
        arrays.clear()


def _resolve_vocab(con: Any, specials: Sequence[str], vocab_path: Path | None, time_ok: str) -> tuple[EventVocab, bool]:
    """The vocabulary at `vocab_path` if there is one, else the events found in the data.

    Returns the vocabulary and whether it was loaded. A derived one is not saved here:
    its static labels are only known once the individuals that survive are.
    """
    if vocab_path is not None and Path(vocab_path).is_file():
        return EventVocab.load(Path(vocab_path), specials), True
    # Scanned straight off the Parquet view rather than a materialised table: a
    # distinct aggregate holds only the labels, not the corpus. The join and the time
    # filter mirror `_events`, so the vocabulary has no entry that never survives --
    # a dead entry still costs the model an output it has to learn to suppress.
    # Ordered by DuckDB rather than Python's `sorted()`, so the vocabulary order and
    # the co-occurrence tie-break come from one and the same collation.
    labels = [
        row[0]
        for row in con.execute(
            f"SELECT DISTINCT s.event AS event FROM _silver s JOIN _person p USING (id) "
            f"WHERE s.event IS NOT NULL AND s.time IS NOT NULL AND {time_ok} ORDER BY event"
        ).fetchall()
    ]
    return EventVocab(labels, specials), False


def _static_tokens(
    fields: Sequence[str], values: Mapping[str, np.ndarray], vocab: EventVocab, loaded: bool, specials: Sequence[str]
) -> tuple[np.ndarray, EventVocab]:
    """Encodes each person's static attribute values as tokens, shape `(n_persons, len(fields))`.

    A derived vocabulary gets one label per value that occurs, plus `<unknown>` per field,
    so that prompts encoded later with this vocabulary have somewhere to put a value
    unseen in training. A loaded vocabulary maps an unseen value to that `<unknown>`.
    """
    uniques, inverses = {}, {}
    for field in fields:
        uniques[field], inverses[field] = np.unique(values[field].astype(np.str_), return_inverse=True)
    if not loaded:
        statics = sorted(
            {static_label(field, v) for field in fields for v in [*uniques[field].tolist(), UNKNOWN_STATIC]}
        )
        vocab = EventVocab(vocab.events, specials, statics)
    statics = np.empty((len(values[fields[0]]), len(fields)), dtype=np.int64)
    for j, field in enumerate(fields):
        ids = []
        for value in uniques[field].tolist():
            label = static_label(field, value)
            if label not in vocab.stoi:
                label = static_label(field, UNKNOWN_STATIC)
            if label not in vocab.stoi:
                raise ValueError(
                    f"The vocabulary knows neither {static_label(field, value)!r} nor {label!r}. "
                    f"It was not built with the static field {field!r}."
                )
            ids.append(vocab.stoi[label])
        statics[:, j] = np.asarray(ids, dtype=np.int64)[inverses[field]]
    return statics, vocab


def build_event_cache(
    cache_dir: Path,
    *,
    event_paths: Sequence[str],
    specials: Sequence[str],
    person_path: Path | None = None,
    time_origin: str = BIRTH_ORIGIN,
    min_time: int | None = None,
    vocab_path: Path | None = None,
    id_column: str = "ID",
    time_column: str = "TIME",
    event_column: str = "EVENT",
    notes_column: str = "NOTES",
    birthdate_column: str = "BIRTHDATE",
    notes_fields: Mapping[str, NoteField] | None = None,
    static_fields: Sequence[str] = (),
    min_events: int = 2,
    distinct_events: bool = True,
    on_missing_birthdate: Literal["drop", "error"] = "drop",
    on_unknown_event: Literal["error", "drop"] = "error",
    duckdb_threads: int | None = None,
    duckdb_memory_limit: str | None = None,
    duckdb_temp_directory: Path | None = None,
) -> dict:
    """Builds the sequence cache in `cache_dir` and returns what goes into `meta.json`.

    The arguments mirror the fields of `LifeEventsDataModule`, which documents them.
    `meta.json` is *not* written here: the caller writes it last, as the marker that the
    build completed.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    notes_fields = dict(notes_fields or {})
    for name in notes_fields:
        if not name.isidentifier():
            raise ValueError(f"notes_fields key {name!r} must be a valid identifier, it names a file and a batch key")

    static_fields = list(static_fields)
    if len(set(static_fields)) != len(static_fields):
        raise ValueError(f"static_fields contains duplicates: {static_fields}")

    time_sql = time_expr(time_origin)
    use_birthdate = needs_birthdate(time_origin)
    if use_birthdate and person_path is None:
        raise ValueError(f"time_origin={time_origin!r} measures from BIRTHDATE, so `person_path` is required")
    if static_fields and person_path is None:
        raise ValueError("`static_fields` are read from the person dataset, so `person_path` is required")
    # Carried under positional names, so no column name ever needs quoting past the read.
    static_columns = [f"static_{j}" for j in range(len(static_fields))]

    con = _connect_duckdb(duckdb_threads, duckdb_memory_limit, duckdb_temp_directory)
    try:
        id_type = _id_sql_type(con, event_paths, None if person_path is None else str(person_path), id_column)
        _event_relation(
            con, event_paths, id_column, time_column, event_column, id_type, notes_column, notes_fields
        ).create_view("_silver")
        if person_path is not None:
            statics_sql = "".join(
                f", CAST({_quote_ident(field)} AS VARCHAR) AS {column}"
                for field, column in zip(static_fields, static_columns)
            )
            con.execute(
                f"CREATE TEMP TABLE _persons AS SELECT "
                f"CAST({_quote_ident(id_column)} AS {id_type}) AS id, "
                f"CAST({_quote_ident(birthdate_column)} AS DATE) AS birthdate{statics_sql} "
                f"FROM read_parquet(?)",
                [_expand_entry(str(person_path))],
            )
            (n_duplicate_persons,) = con.execute(
                "SELECT count(*) FROM (SELECT id FROM _persons GROUP BY id HAVING count(*) > 1)"
            ).fetchone()
            if n_duplicate_persons:
                raise ValueError(
                    f"{person_path}: {n_duplicate_persons} IDs appear more than once. "
                    "The join would fan out and silently duplicate every one of their events."
                )
        else:
            # No person dataset: the individuals are whoever appears in the events, and
            # `birthdate` stays NULL. Only a `"birth"` origin would ever read it, and that
            # origin refuses to get here without a person dataset.
            con.execute(
                "CREATE TEMP TABLE _persons AS "
                "SELECT DISTINCT id, CAST(NULL AS DATE) AS birthdate FROM _silver WHERE id IS NOT NULL"
            )

        # An integer surrogate per individual, assigned in id order so that ordering
        # by it is ordering by id. Everything downstream carries this instead of the
        # id string: the only table holding VARCHAR is this one, at one row per
        # individual rather than one per event.
        con.execute(
            f"CREATE TEMP TABLE _person AS SELECT id, birthdate{''.join(', ' + c for c in static_columns)}, "
            "CAST(row_number() OVER (ORDER BY id) - 1 AS INTEGER) AS person_idx FROM _persons"
        )
        con.execute("DROP TABLE _persons")

        # `time_ok` is the exact filter `_events` applies below. The vocabulary and the
        # unknown-event count have to share it, or a row dropped for its time gets
        # reported a second time as an unknown label.
        time_ok = f"{time_sql} IS NOT NULL" if min_time is None else f"{time_sql} >= {int(min_time)}"
        vocab, vocab_loaded = _resolve_vocab(con, specials, vocab_path, time_ok)
        # `rank` is the *alphabetical* position, which `token` only coincides with
        # when the vocabulary was derived from the data. An overridden `vocab_path`
        # may be in any order, and co-occurring events still have to come out
        # alphabetically, so the sort key has to be carried separately.
        con.execute("CREATE TEMP TABLE _vocab_raw(label VARCHAR, token INTEGER)")
        con.executemany(
            "INSERT INTO _vocab_raw VALUES (?, ?)",
            [(label, i + vocab.first_event_id) for i, label in enumerate(vocab.events)],
        )
        con.execute(
            "CREATE TEMP TABLE _vocab AS SELECT label, token, "
            "CAST(row_number() OVER (ORDER BY label) AS INTEGER) AS rank FROM _vocab_raw"
        )
        con.execute("DROP TABLE _vocab_raw")

        # LEFT JOINs, so rows that the birthdate join or the vocabulary would drop can
        # be counted before they disappear. A streaming aggregate over the Parquet
        # scan, with nothing materialised: `count(DISTINCT id)` runs only over rows
        # that already failed, which is a small set unless the data is badly broken.
        # The birthdate counts are only meaningful where a birthdate is read at all;
        # without a person dataset every row would otherwise be reported as missing one.
        no_birthdate = "p.birthdate IS NULL" if use_birthdate else "FALSE"
        before_min_time = "FALSE" if min_time is None else f"{time_sql} < {int(min_time)}"
        stats = dict(
            zip(
                ("n_rows", "n_no_birthdate", "n_persons_no_birthdate", "n_before_min_time", "n_unknown_event"),
                con.execute(
                    f"SELECT count(*), "
                    f"count(*) FILTER (WHERE {no_birthdate}), "
                    f"count(DISTINCT s.id) FILTER (WHERE {no_birthdate}), "
                    f"count(*) FILTER (WHERE {before_min_time}), "
                    f"count(*) FILTER (WHERE {time_ok} AND v.token IS NULL) "
                    f"FROM _silver s "
                    f"LEFT JOIN _person p USING (id) LEFT JOIN _vocab v ON v.label = s.event "
                    f"WHERE s.id IS NOT NULL AND s.time IS NOT NULL AND s.event IS NOT NULL"
                ).fetchone(),
            )
        )
        if stats["n_no_birthdate"] and on_missing_birthdate == "error":
            raise ValueError(
                f"{stats['n_no_birthdate']} events belong to {stats['n_persons_no_birthdate']} individuals "
                f"with no usable BIRTHDATE in {person_path}. "
                'Pass `on_missing_birthdate="drop"` to skip them.'
            )
        if stats["n_unknown_event"] and on_unknown_event == "error":
            raise ValueError(
                f"{stats['n_unknown_event']} events carry a label absent from {vocab_path}. "
                'Pass `on_unknown_event="drop"` to skip them.'
            )

        # The one table with a row per event. Four narrow integers (plus any
        # event attributes in the notes) rather than two VARCHARs and a BIGINT: ~16 bytes a row against
        # ~100, which is most of the memory the build used to need, both here and in
        # the sort below. Note attributes take part in DISTINCT, so two otherwise
        # identical events that disagree on an attribute are both kept.
        distinct = "DISTINCT " if distinct_events else ""
        notes = "".join(f", s.{_quote_ident(note_column(name))}" for name in notes_fields)
        con.execute(
            f"CREATE TEMP TABLE _events AS SELECT {distinct}"
            f"       p.person_idx AS person_idx, "
            f"       CAST({time_sql} AS INTEGER) AS time, v.token AS token, v.rank AS rank{notes} "
            f"FROM _silver s JOIN _person p USING (id) JOIN _vocab v ON v.label = s.event "
            f"WHERE s.time IS NOT NULL AND {time_ok}"
        )

        # The individuals that survive `min_events`, with their event counts -- which
        # are also the offsets. One row per individual, and built entirely in SQL.
        con.execute(
            f"CREATE TEMP TABLE _keep AS SELECT person_idx, count(*) AS n FROM _events "
            f"GROUP BY person_idx HAVING count(*) >= {int(min_events)}"
        )
        # Counted after every other filter, so `min_events` is not credited with
        # individuals the birthdate join or the vocabulary had already removed.
        (n_persons_before_min_events,) = con.execute("SELECT count(DISTINCT person_idx) FROM _events").fetchone()

        per_person = con.execute("SELECT person_idx, n FROM _keep ORDER BY person_idx").fetchnumpy()
        # Back to real ids only at the very end, for the individuals that survived,
        # in whichever type the sources used.
        statics_select = "".join(
            f", COALESCE(p.{column}, {_quote_literal(UNKNOWN_STATIC)}) AS {column}" for column in static_columns
        )
        per_person_columns = con.execute(
            f"SELECT p.id, {time_offset_expr(time_origin)} AS time_offset{statics_select} "
            f"FROM _keep k JOIN _person p USING (person_idx) ORDER BY k.person_idx"
        ).fetchnumpy()
        raw_ids = per_person_columns["id"]
        ids = _to_numpy(raw_ids, np.int64) if id_type == "BIGINT" else np.asarray(raw_ids).astype(np.str_)
        time_offsets = _to_numpy(per_person_columns["time_offset"], np.int64)
        counts = _to_numpy(per_person["n"], np.int64)
        if ids.size == 0:
            raise ValueError(
                f"No usable sequences in {list(event_paths)}: every individual was dropped by the "
                f"birthdate join, the time filter or min_events={min_events}."
            )
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        n_events = int(offsets[-1])

        np.save(cache_dir / "ids.npy", ids, allow_pickle=False)
        np.save(cache_dir / "offsets.npy", offsets, allow_pickle=False)
        np.save(cache_dir / "time_offsets.npy", time_offsets, allow_pickle=False)
        if static_fields:
            values = {
                field: _to_numpy(per_person_columns[column], object)
                for field, column in zip(static_fields, static_columns)
            }
            statics, vocab = _static_tokens(static_fields, values, vocab, vocab_loaded, specials)
        else:
            statics = np.empty((ids.size, 0), dtype=np.int64)
        np.save(cache_dir / "statics.npy", statics.astype(_token_dtype(vocab)), allow_pickle=False)
        # The per-person counts already fix the output shape exactly, so the event
        # stream can be written straight into pre-sized files without ever holding
        # the whole corpus in Python.
        n_written = _write_event_stream(con, cache_dir, n_events, _token_dtype(vocab), notes_fields)
    finally:
        con.close()

    # The offsets come from one query and the stream from another; they line up only
    # because both order by `id`. Checked rather than asserted, since `-O` strips asserts
    # and a silent mismatch would shred every sequence boundary.
    if n_written != n_events:
        raise RuntimeError(f"per-person counts ({n_events}) disagree with the flat event stream ({n_written})")
    vocab.save(cache_dir / "vocab.json")
    if vocab_path is not None and not vocab_loaded:
        vocab.save(Path(vocab_path))

    stats["n_dropped_min_events"] = int(n_persons_before_min_events) - int(ids.size)
    return {
        "version": CACHE_VERSION,
        "n_persons": int(ids.size),
        "n_events": n_events,
        "vocab_size": vocab.vocab_size,
        "stats": {k: int(v) for k, v in stats.items()},
        "time": {"origin": time_origin, "unit": TIME_UNIT, "offset_reference": "1970-01-01"},
        "filters": {"min_events": int(min_events), "min_time": None if min_time is None else int(min_time)},
        "features": {
            name: {"field": spec.field, "dtype": spec.dtype, "fill": spec.fill} for name, spec in notes_fields.items()
        },
        "statics": static_fields,
        "id_type": id_type,
        "sources": {"events": list(event_paths), "persons": None if person_path is None else str(person_path)},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def summarize(meta: dict, cache_dir: Path) -> str:
    """A human-readable summary of what `build_event_cache` kept and dropped."""
    stats, filters = meta["stats"], meta["filters"]
    lines = [
        f"{cache_dir}: {meta['n_persons']:,} individuals / {meta['n_events']:,} events, "
        f"vocab_size={meta['vocab_size']}, time from {meta['time']['origin']} in {meta['time']['unit']}s"
    ]
    if meta["features"]:
        lines.append("  note features: " + ", ".join(meta["features"]))
    if meta["statics"]:
        lines.append("  static fields: " + ", ".join(meta["statics"]))
    dropped = []
    if stats["n_no_birthdate"]:
        dropped.append(
            f"{stats['n_no_birthdate']:,} events without a birthdate ({stats['n_persons_no_birthdate']:,} individuals)"
        )
    if stats["n_before_min_time"]:
        dropped.append(f"{stats['n_before_min_time']:,} before min_time={filters['min_time']}")
    if stats["n_unknown_event"]:
        dropped.append(f"{stats['n_unknown_event']:,} with a label outside the vocabulary")
    if stats["n_dropped_min_events"] > 0:
        dropped.append(f"{stats['n_dropped_min_events']:,} individuals under min_events={filters['min_events']}")
    if dropped:
        lines.append("  dropped: " + ", ".join(dropped))
    lines.append(
        f"  Set `vocab_size={meta['vocab_size']}` and `padded_vocab_size={meta['vocab_size']}` in the model config."
    )
    return "\n".join(lines)

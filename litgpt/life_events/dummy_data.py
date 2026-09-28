# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Generates a small dummy silver layer, for trying the pipeline out end to end.

This data is not realistic on purpose. It creates two event datasets with *different* `NOTES` schemas, a
separate person table with `BIRTHDATE`, dates at daily resolution, and life courses long
enough that windowing and `<no_event>` insertion actually do something. The person table
also carries a static `GENDER`, missing for some, to try out `static_fields` with. Nothing here is
based on real statistics, and no conclusion should be drawn from a model trained on it. Use as follows:

    python -m litgpt.life_events.dummy_data --out_dir data/dummy_life_events

Writes `work_education.parquet`, `civil_registry.parquet` and `persons.parquet`, the
layout `DelphiLifeEvents` expects. See `README.md` for how to train on it.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from litgpt.constants import _PYARROW_AVAILABLE
from litgpt.utils import CLI

__all__ = ["generate"]

AS_OF = date(2026, 1, 1)
"""Observation ends here, so the data does not change with the wall clock."""

DAYS_PER_YEAR = 365.25

# Which registry an event is recorded in. Two datasets with disjoint event types and
# unrelated `NOTES` structs, which is the case the reader has to survive.
WORK_EDUCATION = ("birth", "school_start", "school_end", "university_start", "university_end")
WORK_EVENTS = ("job_start", "job_end", "retirement")
CIVIL_EVENTS = ("move", "marriage", "divorce", "child_birth", "death")


def _dates(rng: np.random.Generator, n: int) -> list[date]:
    """Birthdates spread over the 20th century."""
    start, end = date(1930, 1, 1).toordinal(), date(2005, 12, 31).toordinal()
    return [date.fromordinal(int(o)) for o in rng.integers(start, end, size=n)]


def _at(birth: date, years: float) -> date:
    return birth + timedelta(days=int(round(years * DAYS_PER_YEAR)))


def _life_course(rng: np.random.Generator, birth: date) -> list[tuple[date, str]]:
    """One person's events, in no particular order. Ages are made up but ordered."""
    events = [(birth, "birth")]
    death_age = float(rng.normal(80, 12)) if rng.random() < 0.7 else 200.0

    events.append((_at(birth, 6 + rng.random()), "school_start"))
    school_end = 16 + rng.random() * 3
    events.append((_at(birth, school_end), "school_end"))
    if rng.random() < 0.4:
        events.append((_at(birth, school_end + 0.5), "university_start"))
        events.append((_at(birth, school_end + 3 + rng.random() * 3), "university_end"))

    age = school_end + rng.random() * 5
    for _ in range(int(rng.integers(1, 6))):
        if age > 65:
            break
        events.append((_at(birth, age), "job_start"))
        age += 1 + rng.random() * 12
        events.append((_at(birth, age), "job_end"))
        age += rng.random() * 3
    if death_age > 65:
        events.append((_at(birth, 65 + rng.random() * 5), "retirement"))

    for _ in range(int(rng.integers(1, 7))):
        events.append((_at(birth, 18 + rng.random() * 50), "move"))
    if rng.random() < 0.7:
        married = 22 + rng.random() * 20
        events.append((_at(birth, married), "marriage"))
        if rng.random() < 0.35:
            events.append((_at(birth, married + 1 + rng.random() * 20), "divorce"))
    for _ in range(int(rng.integers(0, 4))):
        events.append((_at(birth, 20 + rng.random() * 25), "child_birth"))
    if death_age < 200:
        events.append((_at(birth, death_age), "death"))

    # Nothing after death or after the observation window closes.
    died = _at(birth, death_age)
    return [(when, what) for when, what in events if when <= min(died, AS_OF) or what == "death" and when <= AS_OF]


def generate(out_dir: Path = Path("data/dummy_life_events"), n_persons: int = 2000, seed: int = 42) -> None:
    """Writes a dummy silver layer to `out_dir`.

    Args:
        out_dir: Directory for the three Parquet files. Created if missing.
        n_persons: How many persons to invent.
        seed: RNG seed, so the same call always gives the same data.
    """
    if not _PYARROW_AVAILABLE:
        raise ModuleNotFoundError(
            f"{_PYARROW_AVAILABLE!s} Writing Parquet needs pyarrow. Install it with `pip install pyarrow`, "
            "or the whole extras group with `pip install 'litgpt[extra]'`."
        )
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    births = _dates(rng, n_persons)
    ids = [f"p{i:06d}" for i in range(n_persons)]
    rows: dict[str, list] = {"work_education": [], "civil_registry": []}
    for person, birth in zip(ids, births):
        for when, what in _life_course(rng, birth):
            if what in WORK_EDUCATION or what in WORK_EVENTS:
                # Only this registry knows about income, and only for paid work.
                notes = {"income": int(rng.integers(20, 90) * 1000) if what in WORK_EVENTS else None}
                rows["work_education"].append((person, when, what, notes))
            else:
                notes = {"municipality": int(rng.integers(1, 400)), "source": "civil"}
                rows["civil_registry"].append((person, when, what, notes))

    schemas = {
        "work_education": pa.struct([("income", pa.int64())]),
        "civil_registry": pa.struct([("municipality", pa.int64()), ("source", pa.string())]),
    }
    for name, records in rows.items():
        table = pa.table(
            {
                "ID": pa.array([r[0] for r in records], pa.string()),
                "TIME": pa.array([r[1] for r in records], pa.date32()),
                "EVENT": pa.array([r[2] for r in records], pa.string()),
                "NOTES": pa.array([r[3] for r in records], schemas[name]),
            }
        )
        pq.write_table(table, out_dir / f"{name}.parquet")

    # Drawn last, so adding it left every other column of the data unchanged.
    genders = [None if rng.random() < 0.02 else str(g) for g in rng.choice(["F", "M"], size=n_persons)]
    pq.write_table(
        pa.table(
            {
                "ID": pa.array(ids, pa.string()),
                "BIRTHDATE": pa.array(births, pa.date32()),
                "GENDER": pa.array(genders, pa.string()),
            }
        ),
        out_dir / "persons.parquet",
    )

    n_events = sum(len(r) for r in rows.values())
    labels = sorted({r[2] for records in rows.values() for r in records})
    print(
        f"{out_dir}: {n_persons:,} persons / {n_events:,} events / {len(labels)} event types\n"
        f"  {', '.join(labels)}\n"
        f"  vocab_size for the model config: {len(labels) + 2} (the events plus <pad> and <no_event>)"
    )


if __name__ == "__main__":
    CLI(generate)

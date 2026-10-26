# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Read access to the cache `build_event_cache` writes."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from litgpt.life_events.data.source import BIRTH_ORIGIN, CACHE_VERSION, EPOCH_ORIGIN, TIME_UNIT, NoteField, note_column
from litgpt.life_events.data.vocab import EventVocab

__all__ = ["SequenceStore"]


@dataclass(frozen=True)
class SequenceStore:
    """The prepared cache: a flat event stream plus per-person boundaries.

    Person `i` owns rows `offsets[i]:offsets[i + 1]` of every per-event array --
    `tokens`, `times` and one array per requested `NOTES` field -- in the order events
    happened, co-occurring events alphabetical.

    Attributes:
        ids: The original `ID` of each person, in id order.
        offsets: Sequence boundaries, `len(ids) + 1` entries.
        tokens: Event token ids.
        times: When each event happened, in whole `time_unit`s from `time_origin`.
        time_origin: What `times` is measured from, e.g. `"birth"` (age) or `"epoch"`.
            A dataset built for one origin should check it rather than assume: the
            numbers look the same either way.
        time_unit: The unit of `times`.
        time_offsets: Each person's own origin, in `time_unit`s since 1970-01-01.
            Every origin is a date, so the origins differ by exactly this constant and
            `times_from` converts between them. For example, a model that needs *both* age and
            calendar time reads one array and shifts it, rather than storing two.
        features: Note event attribute arrays, keyed by their batch name (`note_<name>`).
        feature_fills: The value each note event attribute takes where an event lacks it.
        statics: Each person's static attribute tokens, shape `(len(ids), len(static_fields))`.
            Timeless, so they are not part of the event stream; a dataset decides where
            to put them.
        static_fields: The person-dataset columns `statics` was read from, in column order.
        vocab: The event vocabulary.
    """

    ids: np.ndarray
    offsets: np.ndarray
    tokens: np.ndarray
    times: np.ndarray
    time_origin: str
    time_unit: str
    time_offsets: np.ndarray
    features: dict[str, np.ndarray]
    feature_fills: dict[str, float]
    statics: np.ndarray
    static_fields: tuple[str, ...]
    vocab: EventVocab

    @classmethod
    def load(cls, cache_dir: Path, specials: Sequence[str] | None = None, mmap: bool = True) -> "SequenceStore":
        """Opens a cache. `specials`, when given, must match those it was built with.

        `mmap=False` reads the arrays into memory instead, which leaves no file open: on
        Windows, a memory-mapped file cannot be deleted while the store is alive.
        """
        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "meta.json").read_text())
        if meta.get("version") != CACHE_VERSION:
            raise ValueError(
                f"{cache_dir} was written by cache version {meta.get('version')}, "
                f"this is version {CACHE_VERSION}. Delete it to rebuild."
            )
        # `mmap_mode`: registry-scale token streams never fully enter RAM, and forked
        # dataloader workers share the page cache instead of copying the arrays.
        load = partial(np.load, mmap_mode="r" if mmap else None, allow_pickle=False)
        specs = {name: NoteField(**spec) for name, spec in meta["features"].items()}
        return cls(
            ids=load(cache_dir / "ids.npy"),
            offsets=load(cache_dir / "offsets.npy"),
            tokens=load(cache_dir / "tokens.npy"),
            times=load(cache_dir / "times.npy"),
            time_offsets=load(cache_dir / "time_offsets.npy"),
            time_origin=meta["time"]["origin"],
            time_unit=meta["time"].get("unit", TIME_UNIT),
            features={note_column(name): load(cache_dir / f"{note_column(name)}.npy") for name in specs},
            feature_fills={note_column(name): spec.fill for name, spec in specs.items()},
            statics=load(cache_dir / "statics.npy"),
            static_fields=tuple(meta["statics"]),
            vocab=EventVocab.load(cache_dir / "vocab.json", specials),
        )

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def times_from(self, i: int, times: np.ndarray, origin: str = EPOCH_ORIGIN) -> np.ndarray:
        """Person `i`'s event times, transformed from `time_origin` into `origin`.

        The origins this cache offers are all anchored to a date, so moving between them is
        one addition. That is what makes a second origin free: a model wanting age *and*
        calendar time keeps the stored `times` and calls this for the other one.

        Args:
            i: The person the times belong to. The shift is per person whenever
                either origin is `"birth"`.
            times: Times of that person's events, as `sequence` returns them.
            origin: The origin to express them in: `"birth"`, `"epoch"` or an ISO date.
                `"birth"` is only available on a cache built with a birthdate.

        Returns:
            The same times, measured from `origin`.
        """
        if origin == self.time_origin:
            return times
        if origin == BIRTH_ORIGIN:
            # Only a cache built from birthdates knows them, and that one returned above.
            raise ValueError(
                f"this cache was built with time_origin={self.time_origin!r} and does not know any "
                f"person's birthdate, so times cannot be re-based onto {BIRTH_ORIGIN!r}. Rebuild "
                f"with time_origin={BIRTH_ORIGIN!r}, which keeps every other origin available."
            )
        # Via the epoch, which `time_offsets` is anchored to.
        from_epoch = times + self.time_offsets[i]
        if origin == EPOCH_ORIGIN:
            return from_epoch
        try:
            reference = (np.datetime64(origin, "D") - np.datetime64("1970-01-01", "D")).astype(np.int64)
        except ValueError:
            raise ValueError(
                f"origin must be {BIRTH_ORIGIN!r}, {EPOCH_ORIGIN!r} or an ISO date (YYYY-MM-DD), got {origin!r}"
            ) from None
        return from_epoch - reference

    def sequence(self, i: int) -> dict[str, np.ndarray]:
        """Person `i`'s events as `{"tokens", "times", **features}`, still memory-mapped."""
        start, stop = int(self.offsets[i]), int(self.offsets[i + 1])
        return {
            "tokens": self.tokens[start:stop],
            "times": self.times[start:stop],
            **{name: array[start:stop] for name, array in self.features.items()},
        }

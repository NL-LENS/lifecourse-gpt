# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Abstract life-course event `DataModule`.

Most concrete data processing operations are performed in `source.build_event_cache`.

"""

import json
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from litgpt.data.base import DataModule
from litgpt.life_events.data.source import BIRTH_ORIGIN, NoteField, build_event_cache, needs_birthdate, summarize
from litgpt.life_events.data.store import SequenceStore
from litgpt.life_events.data.vocab import EventVocab

__all__ = ["LifeEventsDataModule"]


@dataclass
class LifeEventsDataModule(DataModule):
    """Abstract base class for life-course event datasets.

    This base class performs all model-agnostic data-related tasks: It reads and merges the preprocessed input "silver"
    layer event datasets, caches the merged dataset, separates the training from the validation split (by person), and
    creates the dataloaders. A subclass decides what a training sample looks like by implementing
    `make_dataset` and `collate_fn`, and declares the reserved tokens its model needs
    in `specials`. See `litgpt.life_events.delphi.data.DelphiLifeEvents`.

    Every event dataset has the schema `ID | TIME | EVENT | NOTES`, with `ID` a 64-bit precision integer, `TIME` an ISO
    date at daily resolution, `EVENT` a string label, and `NOTES` and struct or mapping with potentially heterogeneous
    fields (currently not used). When the datasets are merged, co-occurring events are ordered alphabetically, and each
    event is placed on a time axis in whole days, inferred from `time_origin`. Age (`time_origin="birth"`)
    is one such origin but requires birth dates stored in a person-level dataset. Another origin is absolute time,
    (`time_origin="epoch"` or `time_origin="2000-01-01"`) which needs no person-level dataset.

    Fields are documented below and, being a `DataModule`, all of them are settable from
    the CLI or a YAML config as `--data.<field>`.
    """

    event_paths: list[str] = field(default_factory=list)
    """One entry per event dataset: a Parquet file, a directory, or a glob."""
    person_path: Path | None = None
    """The person-attributes dataset carrying `BIRTHDATE`. Required only for
    `time_origin="birth"`; without it the individuals are whoever appears in the events."""
    cache_dir: Path = Path("data/life_events")
    """Where `prepare_data` writes the preprocessed sequences."""
    time_origin: str = BIRTH_ORIGIN
    """What each event's time is measured from: `"birth"` (age), `"epoch"` (days since
    1970-01-01) or an ISO date. Subclasses whose model can only read one origin restrict
    this through `time_origins`."""
    min_time: int | None = None
    """Drop events before this time coordinate. `None` keeps them all; the default is `0`
    for `time_origin="birth"`, which drops events dated before birth, and `None` for every
    other origin, where negative simply means earlier than the reference date."""
    vocab_path: Path | None = None
    """Load this vocabulary instead of deriving one from the data, so that adding a
    dataset later cannot renumber tokens under a trained checkpoint."""
    force_prepare: bool = False
    """Rebuild the cache even when it already exists."""

    id_column: str = "ID"
    """Name of the individual identifier column, in both dataset kinds."""
    time_column: str = "TIME"
    """Name of the event date column."""
    event_column: str = "EVENT"
    """Name of the event label column."""
    notes_column: str = "NOTES"
    """Name of the per-event attribute column. Read only for `notes_fields`."""
    birthdate_column: str = "BIRTHDATE"
    """Name of the birthdate column in the person dataset."""
    notes_fields: dict[str, NoteField] = field(default_factory=dict)
    """Attributes to read out of `NOTES`, keyed by the name they get in the cache and in
    batches (`note_<name>`). Empty by default: `NOTES` is then never read at all. Whether
    a model uses them is up to the subclass."""

    min_events: int = 2
    """Drop individuals with fewer events than this. Two is the floor: one event has no
    target to predict."""
    distinct_events: bool = True
    """Collapse exact duplicate events, which are what two overlapping silver datasets
    produce."""
    on_missing_birthdate: Literal["drop", "error"] = "drop"
    """What to do about individuals absent from the person table or carrying a null
    `BIRTHDATE`."""
    on_unknown_event: Literal["error", "drop"] = "error"
    """What to do about labels absent from an overridden vocabulary."""

    val_split_fraction: float = 0.05
    """Fraction of *persons* held out. The split is by person, never by row."""
    seed: int = 42
    """Seeds the split, the dataloader shuffling and each dataset's own sampling."""
    num_workers: int = 4
    """Dataloader worker processes."""
    ignore_index: int = -100
    """Label at padded positions, excluded from the loss."""

    duckdb_threads: int | None = None
    """Threads DuckDB may use while building the cache. `None` leaves it to DuckDB, which
    takes every core. Lower it to leave room for other work on a shared machine."""
    duckdb_memory_limit: str | None = None
    """Memory DuckDB may use while building the cache, e.g. `"8GB"`. `None` leaves it to
    DuckDB (about 80% of RAM). Pair it with `duckdb_temp_directory`, or a corpus that
    cannot be sorted within the limit fails instead of spilling."""
    duckdb_temp_directory: Path | None = None
    """Where DuckDB may spill to disk, which is what lets the join and the sort run on a
    corpus larger than the memory limit. Needs room for roughly the size of the event
    table, and should be fast local disk rather than a network mount."""

    vocab: EventVocab | None = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    specials: ClassVar[tuple[str, ...]] = ("<pad>",)
    """Reserved tokens at the front of the vocabulary, padding first."""
    time_origins: ClassVar[tuple[str, ...] | None] = None
    """The `time_origin` values this module's model can read, or `None` for any."""
    model_class_name: ClassVar[str | None] = None
    """The `Config.model_class_name` whose batch layout `collate_fn` produces, if only one
    can read it. `litgpt.pretrain` refuses to pair this module with any other."""

    def __post_init__(self) -> None:
        super().__init__()
        if not self.event_paths:
            raise ValueError("`event_paths` must name at least one event dataset")
        if self.person_path is None and needs_birthdate(self.time_origin):
            raise ValueError(f"`person_path` is required for time_origin={self.time_origin!r}, it supplies BIRTHDATE")
        if self.time_origins is not None and self.time_origin not in self.time_origins:
            raise ValueError(
                f"{type(self).__name__} reads time measured from {list(self.time_origins)}, "
                f"got time_origin={self.time_origin!r}"
            )
        if self.min_time is None and needs_birthdate(self.time_origin):
            # Events dated before birth are almost always data errors, and a model whose
            # time axis starts at birth has nowhere to put them.
            self.min_time = 0
        if not 0.0 < self.val_split_fraction < 1.0:
            raise ValueError(f"val_split_fraction must be in (0, 1), got {self.val_split_fraction}")
        if self.min_events < 2:
            raise ValueError(f"min_events must be at least 2, one event has no target, got {self.min_events}")
        # jsonargparse builds `NoteField`s itself; a plain dict from Python is accepted too.
        self.notes_fields = {
            name: spec if isinstance(spec, NoteField) else NoteField(**spec) for name, spec in self.notes_fields.items()
        }
        self.cache_dir = Path(self.cache_dir)
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self._store: SequenceStore | None = None

    # ---- subclass hooks ------------------------------------------------------------

    @abstractmethod
    def make_dataset(self, store: SequenceStore, indices: np.ndarray, train: bool) -> Dataset:
        """The dataset over the individuals `indices` of `store`, for one split."""

    @abstractmethod
    def collate_fn(self) -> Callable[[list[Any]], Any]:
        """Turns a list of samples into a batch. Must be picklable: no lambdas."""

    # ---- DataModule ----------------------------------------------------------------

    def connect(self, batch_size: int = 1, max_seq_length: int | None = None) -> None:
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    @property
    def vocab_size(self) -> int:
        """Size of the event vocabulary, including the special tokens."""
        if self.vocab is None:
            self.vocab = EventVocab.load(self.cache_dir / "vocab.json", self.specials)
        return self.vocab.vocab_size

    def prepare_data(self) -> None:
        marker = self.cache_dir / "meta.json"
        if marker.is_file() and not self.force_prepare:
            print(
                f"\nWarning: Preprocessed life-event data found in {self.cache_dir}. "
                f"For efficiency, reprocessing is skipped. If the source datasets changed, "
                f"delete it to trigger reprocessing: `rm -rf {self.cache_dir}`\n"
            )
            return
        meta = build_event_cache(
            self.cache_dir,
            event_paths=self.event_paths,
            person_path=self.person_path,
            time_origin=self.time_origin,
            min_time=self.min_time,
            specials=self.specials,
            vocab_path=self.vocab_path,
            id_column=self.id_column,
            time_column=self.time_column,
            event_column=self.event_column,
            notes_column=self.notes_column,
            birthdate_column=self.birthdate_column,
            notes_fields=self.notes_fields,
            min_events=self.min_events,
            distinct_events=self.distinct_events,
            on_missing_birthdate=self.on_missing_birthdate,
            on_unknown_event=self.on_unknown_event,
            duckdb_threads=self.duckdb_threads,
            duckdb_memory_limit=self.duckdb_memory_limit,
            duckdb_temp_directory=self.duckdb_temp_directory,
        )
        # Written last, so a crash mid-build leaves no marker and the next run retries.
        marker.write_text(json.dumps(meta, indent=2))
        self.vocab = None
        print(summarize(meta, self.cache_dir))

    def setup(self, stage: str = "") -> None:
        if self.train_dataset is not None:
            return
        if not (self.cache_dir / "meta.json").is_file():
            raise RuntimeError(f"{self.cache_dir} holds no prepared data. Call `prepare_data()` first.")
        self._store = SequenceStore.load(self.cache_dir, self.specials)
        self.vocab = self._store.vocab

        generator = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(len(self._store), generator=generator).numpy()
        n_val = int(round(len(order) * self.val_split_fraction))
        if n_val == 0:
            raise ValueError(
                f"val_split_fraction={self.val_split_fraction} holds out no individuals from "
                f"{len(order)}. `litgpt.pretrain` always runs a validation pass, so the split "
                "cannot be empty."
            )
        val_idx, train_idx = order[:n_val], order[n_val:]
        self.train_dataset = self.make_dataset(self._store, train_idx, train=True)
        self.val_dataset = self.make_dataset(self._store, val_idx, train=False)

        print(
            f"{self.cache_dir}: {len(self.train_dataset):,} train / {len(self.val_dataset):,} val sequences, "
            f"vocab_size={self.vocab.vocab_size}"
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False)

    def _dataloader(self, dataset: Dataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Call `setup()` before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            pin_memory=True,
            # The dataloader is pickled both to checkpoint the training state and to
            # start worker processes, hence the picklability requirement on the hook.
            collate_fn=self.collate_fn(),
        )

# Derivative files are under copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and
# Economic Innovations (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Delphi-specific data loading and processing.

Covers no-event tokens, windowing and the tie mask.

Unlike the text pretraining modules in `litgpt.data`, which stream a flat token sequence
and let the training loop do the causal shift, this module emits already shifted
batches. Each token carries an age, and the tie mask needs to know the age of a token's
*target*, so inputs, labels and both age vectors have to be aligned in one place. See
`DelphiLifeEvents` for the batch layout.

Adapted from https://github.com/gerstung-lab/Delphi/blob/main/utils.py under the MIT
License: the no-event tokens, the 100-year horizon and the order in which insertion,
sorting and windowing happen all follow `get_batch` there.
"""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from litgpt.life_events.data.datamodule import LifeEventsDataModule
from litgpt.life_events.data.source import BIRTH_ORIGIN
from litgpt.life_events.data.store import SequenceStore
from litgpt.life_events.delphi.mask import build_tie_mask

__all__ = ["DelphiDataset", "DelphiLifeEvents", "delphi_collate_fn", "insert_no_events", "no_event_ages"]

DAYS_PER_YEAR = 365.25
MAX_AGE_DAYS = 36525
"""The 100-year horizon no-event tokens are drawn over, verbatim from Delphi."""

PAD_TOKEN, NO_EVENT_TOKEN = "<pad>", "<no_event>"
SPECIALS = (PAD_TOKEN, NO_EVENT_TOKEN)
PAD_ID, NO_EVENT_ID, FIRST_EVENT_ID = 0, 1, 2
"""Token ids, following Delphi's layout after its `tokens + 1` shift.

`<pad>` only ever fills the right of a short sequence and is excluded from the loss by
`ignore_index`. `<no_event>` is a *real observed outcome*: Delphi's training loss masks
only id 0, so a position whose target is a no-event token contributes to both loss terms.
This is how event-free stretches of the observed window enter the likelihood.
"""

PAD_AGE = -1
"""Age placeholder at padded positions. Never compared against a real age, because
`build_tie_mask` drops padded keys via `pad_mask`; the negative value only keeps it
from silently colliding with a genuine age of 0 if `pad_mask` is ever left out."""


def no_event_ages(
    mode: Literal["regular", "random"], rate: float, end_of_observation: int, rng: np.random.Generator
) -> np.ndarray:
    """Ages, in days, at which no-event tokens go into a sequence observed until `end_of_observation`.

    Args:
        mode: `"regular"` for a fixed grid, `"random"` for uniform ages, as in Delphi.
        rate: Years between no-event tokens.
        end_of_observation: Last age at which the person is known to be observed.
        rng: Source of the random ages. Unused by the regular grid.
    """
    if mode == "regular":
        ages = np.arange(0.0, MAX_AGE_DAYS, DAYS_PER_YEAR * rate) + 1.0
        ages = np.round(ages).astype(np.int64)
    else:
        ages = rng.integers(1, MAX_AGE_DAYS, size=int(100 / rate)).astype(np.int64)
        ages.sort()
    # Delphi's `masked_fill(ages > m, -1)`: nothing is known past the last observed
    # event, so the censored tail is dropped rather than modelled.
    return ages[ages <= end_of_observation]


def insert_no_events(
    tokens: np.ndarray,
    ages: np.ndarray,
    extra: np.ndarray,
    features: dict[str, np.ndarray] | None = None,
    feature_fills: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Merges no-event tokens at the ages `extra` into a person's events.

    Args:
        tokens: Event token ids, in the order the cache gives them.
        ages: Age of each event, in days.
        extra: Ages of the no-event tokens, as `no_event_ages` returns them.
        features: Optional per-event attribute arrays, aligned with `tokens`.
        feature_fills: The value each of `features` takes at an inserted no-event token.

    Returns:
        `(tokens, ages, features)`, sorted by age.
    """
    features = features or {}
    # A day that already carries an event is not a day on which nothing happened,
    # and dropping the collision also settles where the stable sort would put it.
    # Delphi never hits this because its ages are floats.
    extra = extra[~np.isin(extra, ages)]
    tokens = np.concatenate([tokens, np.full(extra.size, NO_EVENT_ID, dtype=tokens.dtype)])
    ages = np.concatenate([ages, extra.astype(ages.dtype)])
    features = {
        name: np.concatenate([array, np.full(extra.size, feature_fills[name], dtype=array.dtype)])
        for name, array in features.items()
    }
    # Stable, so the alphabetical order the cache gave co-occurring events
    # survives, and the appended no-event tokens land after real ones at a tie.
    order = np.argsort(ages, kind="stable")
    return tokens[order], ages[order], {name: array[order] for name, array in features.items()}


class DelphiDataset(Dataset):
    """Delphi-specific life-course event dataset.

    One person per item: inserts no-event tokens, windows, and shifts.

    The store's time axis is read as age in days, which is what `<no_event>` insertion
    assumes: the 100-year horizon is a lifetime, not an interval of calendar time. The
    store records what its times are measured from, and `DelphiLifeEvents` refuses any
    origin but `"birth"`.

    The order of operations follows Delphi's `get_batch`: the end of observation is read
    off the real events *before* anything is inserted, no-event tokens are drawn over a
    100-year horizon and truncated at it, everything is merged and re-sorted by age, and
    only then is a window taken and the causal shift applied.

    Static person-level tokens, if the cache has any, are prepended only after all of
    this, at age 0, so that no window can cut them off. They are inputs only: their own
    labels are set to `ignore_index`. So is the label of the first windowed event
    whenever the window does not start at the beginning, since the wait from age 0 to it
    would claim that nothing happened in between. This mirrors Delphi, whose sex and
    lifestyle tokens open every sequence and are never predicted.

    Note features, if the cache has any, travel with their events through all of this.
    Inserted no-event and static tokens take each feature's fill value. They are emitted
    aligned with `input_ids`, as floats or int64 depending on their type. Delphi itself
    ignores them.

    Args:
        store: The prepared cache.
        indices: Which persons in `store` belong to this split.
        max_seq_length: Length of `input_ids`. `-1` keeps the whole sequence.
        no_event_mode: `"none"`, `"regular"` (a fixed grid) or `"random"` (uniform ages).
        no_event_rate: Years between no-event tokens, Delphi's `no_event_token_rate`.
        select: Which window of a too-long sequence to take, Delphi's `select`.
        seed: Base entropy for the per-sample RNG.
        resample: Draw fresh no-event tokens every epoch. Off for validation, so the
            metric is comparable across evaluations.
        shuffle_statics: Put the static tokens in a random order in every sample, so the
            model cannot learn anything from their order.
        ignore_index: Label of the positions excluded from the loss.
    """

    def __init__(
        self,
        store: SequenceStore,
        indices: np.ndarray,
        max_seq_length: int = -1,
        *,
        no_event_mode: str = "regular",
        no_event_rate: float = 5.0,
        select: str = "random",
        seed: int = 42,
        resample: bool = True,
        shuffle_statics: bool = False,
        ignore_index: int = -100,
    ) -> None:
        n_statics = store.statics.shape[1]
        if 0 < max_seq_length <= n_statics:
            raise ValueError(
                f"max_seq_length={max_seq_length} leaves no room for events after {n_statics} static tokens"
            )
        self.store = store
        self.indices = indices
        self.max_seq_length = max_seq_length
        self.no_event_mode = no_event_mode
        self.no_event_rate = no_event_rate
        self.select = select
        self.seed = seed
        self.resample = resample
        self.shuffle_statics = shuffle_statics
        self.ignore_index = ignore_index
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        """Advances the no-event draw when running without dataloader workers.

        With `num_workers > 0` the epoch enters through the worker seed instead and this
        is unnecessary. `litgpt.pretrain` never calls it, so `num_workers > 0` is
        recommended when training with `no_event_mode="random"`.
        """
        self._epoch = epoch

    def _rng(self, idx: int) -> np.random.Generator:
        """Per-sample RNG: reproducible, and independent of `num_workers`.

        PyTorch draws one `base_seed` per epoch from the dataloader's generator and hands
        worker `w` the seed `base_seed + w`, so subtracting the worker id recovers a value
        that is the same in every worker and different every epoch. Seeding per sample
        rather than caching one generator per worker is what makes a fixed `seed` give the
        same batches whatever `num_workers` is set to.
        """
        if not self.resample:
            epoch_seed = 0
        else:
            info = torch.utils.data.get_worker_info()
            epoch_seed = self._epoch if info is None else info.seed - info.id
        return np.random.default_rng([self.seed, int(epoch_seed) % 2**63, int(idx)])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Draw a sequence from a single person."""
        person = int(self.indices[idx])
        sequence = self.store.sequence(person)
        # `astype` also copies out of the read-only memmap, which `torch.from_numpy` needs.
        tokens = sequence.pop("tokens").astype(np.int64)
        ages = sequence.pop("times").astype(np.int64)
        features = {name: _widen(array) for name, array in sequence.items()}
        rng = self._rng(idx)

        if self.no_event_mode != "none":
            extra = no_event_ages(self.no_event_mode, self.no_event_rate, int(ages[-1]), rng)
            tokens, ages, features = insert_no_events(tokens, ages, extra, features, self.store.feature_fills)

        # One extra raw event, because the last one is only ever a target. The static
        # tokens take their share of the length, so windows get shorter, not batches longer.
        statics = self.store.statics[person].astype(np.int64)
        width = self.max_seq_length + 1 - statics.size if self.max_seq_length > 0 else ages.size
        start = 0
        if ages.size > width:
            if self.select == "left":
                start = 0
            elif self.select == "right":
                start = ages.size - width
            else:
                start = int(rng.integers(0, ages.size - width + 1))
            window = slice(start, start + width)
            tokens, ages = tokens[window], ages[window]
            features = {name: array[window] for name, array in features.items()}

        labels = tokens[1:]
        if statics.size:
            if self.shuffle_statics:
                statics = rng.permutation(statics)
            tokens = np.concatenate([statics, tokens])
            ages = np.concatenate([np.zeros(statics.size, dtype=ages.dtype), ages])
            features = {
                name: np.concatenate([np.full(statics.size, self.store.feature_fills[name], dtype=array.dtype), array])
                for name, array in features.items()
            }
            labels = tokens[1:].copy()
            # Statics are read, never predicted. The first event is a target only when
            # the window starts at birth; otherwise its wait from age 0 would be a lie.
            labels[: statics.size - 1 + int(start > 0)] = self.ignore_index

        return {
            "input_ids": torch.from_numpy(tokens[:-1]),
            "labels": torch.from_numpy(labels),
            "age": torch.from_numpy(ages[:-1]),
            "targets_age": torch.from_numpy(ages[1:]),
            **{name: torch.from_numpy(array[:-1]) for name, array in features.items()},
        }


def _widen(array: np.ndarray) -> np.ndarray:
    """int64 or float32, the two types a batch carries; also copies out of the memmap."""
    return array.astype(np.float32 if np.issubdtype(array.dtype, np.floating) else np.int64)


def delphi_collate_fn(
    samples: list[dict[str, torch.Tensor]],
    pad_id: int = PAD_ID,
    ignore_index: int = -100,
    include_attn_mask: bool = True,
    pad_to_length: int | None = None,
    feature_fills: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pads a list of shifted samples into a batch, optionally with the tie mask.

    `pad_to_length` pads every batch to the same width. `litgpt.pretrain` calls
    `torch.compile(model)` unconditionally, and a ragged `T` forces a recompile per
    distinct shape, so a fixed width is worth the handful of wasted positions. Note
    features are padded with their fill value.
    """
    longest = max(len(sample["input_ids"]) for sample in samples)
    if pad_to_length is not None:
        if longest > pad_to_length:
            raise ValueError(f"Sample of length {longest} exceeds pad_to_length={pad_to_length}")
        longest = pad_to_length

    def pad(key: str, value: float) -> torch.Tensor:
        return torch.stack(
            [torch.nn.functional.pad(sample[key], (0, longest - len(sample[key])), value=value) for sample in samples]
        )

    batch = {
        "input_ids": pad("input_ids", pad_id),
        "labels": pad("labels", ignore_index),
        "age": pad("age", PAD_AGE),
        "targets_age": pad("targets_age", PAD_AGE),
    }
    for name, fill in (feature_fills or {}).items():
        batch[name] = pad(name, fill)
    # `<no_event>` is an ordinary token id, so positions whose target is one stay True
    # here and do contribute to both loss terms. Right-padding is excluded, and so are
    # positions whose target is a static token.
    batch["pad_mask"] = batch["labels"] != ignore_index
    if include_attn_mask:
        # Not `pad_mask`: a static token has no target but must stay visible as a key.
        batch["attn_mask"] = build_tie_mask(batch["age"], batch["targets_age"], batch["input_ids"] != pad_id)
    return batch


@dataclass
class DelphiLifeEvents(LifeEventsDataModule):
    """Life-course event batches in the layout `litgpt.life_events.delphi.model.Delphi` trains on.

    See `LifeEventsDataModule` for how the "silver" layer event datasets are read. This class adds
    Delphi's `<no_event>` token and the tie mask, and opens every sequence with the
    person's `static_fields` tokens at age 0, which are never predicted.

    Batches are dicts of `(B, T)` tensors, plus a `(B, 1, T, T)` mask:
        `input_ids`, `age`      -> `Delphi.forward(idx=..., age=...)`
        `labels`, `targets_age` -> the cross-entropy target and its age
        `pad_mask`              -> `True` at positions in the loss: real tokens whose
                                   target is not padding or a static token
        `attn_mask`             -> causal + padding + tie mask, for `Delphi.forward`
        `note_<name>`           -> one per `notes_fields` entry, aligned with `input_ids`

    Set both `vocab_size` and `padded_vocab_size` in the model config to this module's
    `vocab_size`, and `target_vocab_size` to its `target_vocab_size`. With `static_fields`
    the two differ: the static tokens are inputs only, and `target_vocab_size` keeps them
    out of the total event rate in `chunked_cross_entropy_exp_waiting_time`.

    Two known reporting quirks, neither of which affects the training objective:
    Delphi excludes `<no_event>` from its *validation* metrics and this implementation
    cannot, because `chunked_cross_entropy_exp_waiting_time` takes no ignore-token
    argument -- so validation loss here is not numerically comparable to Delphi's
    published figures, only to our own checkpoints. And `litgpt.pretrain` counts tokens
    as `micro_batch_size * max_seq_length`, which overcounts when sequences are padded.

    `min_time` matters more here than in general. It defaults to 0 for this module's
    only permitted origin, and should stay there: no-event tokens are drawn from
    `[1, 36525]` and can never cover the negative region, so keeping prenatal events
    would make the event evidence and the no-event evidence disagree about the support
    of the process.

    These fields come on top of `LifeEventsDataModule`'s, and are set the same way.
    """

    no_event_mode: Literal["none", "regular", "random"] = "regular"
    """`"none"`, `"regular"` (a fixed grid) or `"random"` (uniform ages), as in Delphi."""
    no_event_rate: float = 5.0
    """Years between no-event tokens."""
    select: Literal["random", "left", "right"] = "random"
    """Window taken from a sequence longer than `max_seq_length + 1`. The validation
    split always uses `"left"`."""
    include_attn_mask: bool = True
    """Build the tie mask in the collate function. Turn this off to train with plain
    causal masking, or to build the mask yourself."""
    shuffle_statics: bool = True
    """Shuffle the order of the static tokens in every training sample. Validation keeps
    the cache's order."""
    pad_to_max_seq_length: bool = True
    """Pad every batch to `max_seq_length` rather than to the longest sample, keeping
    shapes stable for `torch.compile`."""

    specials = SPECIALS
    # The no-event tokens are drawn over a 100-year *lifetime*, and `AgeEncoding` divides
    # by 365.25 to get an age in years. Both are meaningless on any other time axis.
    time_origins = (BIRTH_ORIGIN,)
    model_class_name = "Delphi"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.no_event_rate <= 0:
            raise ValueError(f"no_event_rate must be positive, got {self.no_event_rate}")

    def make_dataset(self, store: SequenceStore, indices: np.ndarray, train: bool) -> DelphiDataset:
        common = dict(
            no_event_mode=self.no_event_mode, no_event_rate=self.no_event_rate, ignore_index=self.ignore_index
        )
        if train:
            return DelphiDataset(
                store,
                indices,
                self.max_seq_length,
                select=self.select,
                seed=self.seed,
                resample=True,
                shuffle_statics=self.shuffle_statics,
                **common,
            )
        # `select="left"` and no resampling, so the metric is comparable across
        # evaluations; `seed + 1` so val never draws the same grid as train for an index.
        return DelphiDataset(
            store, indices, self.max_seq_length, select="left", seed=self.seed + 1, resample=False, **common
        )

    def collate_fn(self) -> partial:
        # `partial`, not a lambda: the collate function must be picklable.
        pad_to_length = self.max_seq_length if self.pad_to_max_seq_length and self.max_seq_length > 0 else None
        return partial(
            delphi_collate_fn,
            pad_id=PAD_ID,
            ignore_index=self.ignore_index,
            include_attn_mask=self.include_attn_mask,
            pad_to_length=pad_to_length,
            feature_fills=dict(self._store.feature_fills) if self._store is not None else None,
        )

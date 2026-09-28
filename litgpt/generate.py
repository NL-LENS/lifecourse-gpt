# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Simulates life-course trajectories from a trained Delphi checkpoint.

Conditions on each person's observed events, optionally only those up to a cutoff age,
samples how their life course continues and writes the result as a silver-layer Parquet
file. For example, to simulate the validation persons of the dummy data from age 50:

    litgpt generate --checkpoint_dir out/pretrain/delphi-2m/final --cutoff_age 50 --termination_events '[death]'

Everything about the model and the data is read from the checkpoint directory, which
`litgpt pretrain` fills with the weights, `model_config.yaml` and `hyperparameters.yaml`.
"""

import importlib
import json
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml

from litgpt.config import Config
from litgpt.constants import _PYARROW_AVAILABLE
from litgpt.life_events.data.source import build_event_cache
from litgpt.life_events.data.store import SequenceStore
from litgpt.life_events.delphi.data import (
    DAYS_PER_YEAR,
    NO_EVENT_TOKEN,
    PAD_AGE,
    PAD_ID,
    DelphiLifeEvents,
    insert_no_events,
    no_event_ages,
)
from litgpt.life_events.delphi.generate import generate_trajectories
from litgpt.life_events.delphi.model import Delphi
from litgpt.utils import extend_checkpoint_dir

__all__ = ["setup"]


def setup(
    checkpoint_dir: Path,
    out_path: Path = Path("out/generate/trajectories.parquet"),
    prompt_paths: list[str] | None = None,
    prompt_person_path: Path | None = None,
    split: Literal["val", "train", "all"] = "val",
    cutoff_age: float | None = None,
    num_samples: int = 1,
    max_new_tokens: int | None = None,
    max_age: float = 85.0,
    termination_events: list[str] | None = None,
    no_repeat: bool = False,
    repeatable_events: list[str] | None = None,
    on_unknown_event: Literal["error", "drop"] = "error",
    batch_size: int = 64,
    seed: int = 42,
    device: str = "auto",
) -> None:
    """Simulates life-course trajectories with a trained Delphi model.

    Prompts come from the prepared cache of the data the checkpoint was trained on, or
    from separate silver-layer datasets given as `prompt_paths`. Either way they get
    no-event tokens inserted the way training batches did, so the model is conditioned
    on sequences like the ones it learned from.

    The output has the silver-layer schema `ID | TIME | EVENT`, one row per event, plus
    `SAMPLE` (which of the `num_samples` trajectories of that person the row belongs to)
    and `GENERATED` (false for the prompt, true for sampled events). No-event tokens are
    kept, with the label `<no_event>`, so each trajectory is exactly the sequence the model
    saw and produced. Filter them out before feeding the file back into the data pipeline,
    which would otherwise read them as events.

    Args:
        checkpoint_dir: A checkpoint directory written by `litgpt pretrain`, e.g.
            `out/pretrain/delphi-2m/final`. Its data config has to point at the prepared
            cache, whose vocabulary the prompts are encoded with.
        out_path: Parquet file to write. Parent directories are created.
        prompt_paths: Silver-layer event datasets to take the prompts from, each a Parquet
            file, a directory or a glob, with the columns the training data used. `None`
            takes them from the checkpoint's prepared cache instead.
        prompt_person_path: The person dataset carrying `BIRTHDATE` for `prompt_paths`.
            Required with `prompt_paths`, since ages are measured from birth.
        split: Which persons of the prepared cache to simulate: the validation split the
            checkpoint held out, the training split, or everyone. Ignored with
            `prompt_paths`, which are always simulated in full.
        cutoff_age: Condition only on events up to this age, in years, and simulate from
            there. The person is taken to be observed and event-free until the cutoff.
            Persons with no event by then are skipped. `None` conditions on everything
            observed and continues from the last event.
        num_samples: Trajectories to sample per person.
        max_new_tokens: Events to sample at most per trajectory, no-event tokens
            included. Defaults to half the model's context. Prompts are cut to their most
            recent `max_seq_length - max_new_tokens` events to make room.
        max_age: Stop a trajectory once it passes this age, in years.
        termination_events: Event labels that end a trajectory, typically `death`.
        no_repeat: Forbid sampling an event a trajectory already contains.
        repeatable_events: Event labels exempt from `no_repeat`. `None` exempts only the
            no-event token.
        on_unknown_event: What to do about labels in `prompt_paths` that the checkpoint's
            vocabulary does not know.
        batch_size: Trajectories sampled in parallel.
        seed: Seeds the sampling, and the no-event ages when the data uses random ones.
        device: `"auto"` (CUDA when available), `"cpu"`, `"cuda"`, `"cuda:1"`, ...
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if prompt_paths is not None and prompt_person_path is None:
        raise ValueError("`prompt_person_path` is required with `prompt_paths`, it supplies BIRTHDATE")
    if not _PYARROW_AVAILABLE:
        raise ModuleNotFoundError(
            f"{_PYARROW_AVAILABLE!s} Writing Parquet needs pyarrow. Install it with `pip install pyarrow`, "
            "or the whole extras group with `pip install 'litgpt[extra]'`."
        )
    import pyarrow as pa
    import pyarrow.parquet as pq

    checkpoint_dir = extend_checkpoint_dir(Path(checkpoint_dir))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    model = load_model(checkpoint_dir, device)
    data = load_data_module(checkpoint_dir / "hyperparameters.yaml")

    max_new_tokens = model.max_seq_length // 2 if max_new_tokens is None else max_new_tokens
    if not 0 < max_new_tokens < model.max_seq_length:
        raise ValueError(
            f"max_new_tokens must be in (0, {model.max_seq_length}), the model's context, to leave room for a "
            f"prompt; got {max_new_tokens}"
        )
    prompt_length = model.max_seq_length - max_new_tokens
    cutoff = None if cutoff_age is None else int(round(cutoff_age * DAYS_PER_YEAR))

    with ExitStack() as stack:
        if prompt_paths is None:
            store, indices = _cached_prompts(data, split)
        else:
            cache_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="litgpt-prompts-")))
            store = _encode_prompts(data, prompt_paths, prompt_person_path, cache_dir, on_unknown_event)
            indices = np.arange(len(store))
        vocab = store.vocab
        if vocab.vocab_size != model.config.vocab_size:
            raise ValueError(
                f"The vocabulary in {data.cache_dir} has {vocab.vocab_size} entries, the model was configured "
                f"for vocab_size={model.config.vocab_size}. The cache no longer matches this checkpoint."
            )
        termination = vocab.encode(termination_events or [])
        repeatable = vocab.encode([NO_EVENT_TOKEN] if repeatable_events is None else repeatable_events)

        prompts = {}
        for i in indices:
            prompt = _build_prompt(store, int(i), data, cutoff, prompt_length, seed)
            if prompt is not None:
                prompts[int(i)] = prompt
        n_skipped = len(indices) - len(prompts)
        rows = [(i, sample) for i in prompts for sample in range(num_samples)]
        print(
            f"Simulating {len(prompts):,} persons x {num_samples} samples"
            + (f" ({n_skipped:,} skipped, no events by age {cutoff_age})" if n_skipped else "")
            + f" on {device}"
        )

        generator = torch.Generator(device=device).manual_seed(seed)
        columns: dict[str, list[np.ndarray]] = {name: [] for name in ("ID", "SAMPLE", "TIME", "EVENT", "GENERATED")}
        for chunk in _chunks(rows, batch_size):
            idx, age = _pad([prompts[i] for i, _ in chunk], device)
            idx, age = generate_trajectories(
                model,
                idx,
                age,
                max_new_tokens,
                max_age=max_age * DAYS_PER_YEAR,
                termination_tokens=termination,
                ignore_tokens=(PAD_ID,),
                no_repeat=no_repeat,
                repeatable_tokens=repeatable,
                generator=generator,
            )
            idx, age = idx.cpu().numpy(), age.cpu().numpy()
            for b, (i, sample) in enumerate(chunk):
                real = idx[b] != PAD_ID
                tokens, ages = idx[b][real], age[b][real]
                birth = np.datetime64("1970-01-01", "D") + int(store.time_offsets[i])
                columns["ID"].append(np.repeat(store.ids[i : i + 1], tokens.size))
                columns["SAMPLE"].append(np.full(tokens.size, sample, dtype=np.int32))
                columns["TIME"].append(birth + ages.astype("timedelta64[D]"))
                columns["EVENT"].append(np.asarray(vocab.itos, dtype=object)[tokens])
                columns["GENERATED"].append(np.arange(tokens.size) >= len(prompts[i][0]))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.concatenate(parts) if parts else np.array([]) for name, parts in columns.items()}
    table = pa.table(
        {
            "ID": pa.array(arrays["ID"], pa.int64() if store.ids.dtype.kind == "i" else pa.string()),
            "TIME": pa.array(arrays["TIME"].astype("datetime64[D]"), pa.date32()),
            "EVENT": pa.array(arrays["EVENT"], pa.string()),
            "SAMPLE": pa.array(arrays["SAMPLE"], pa.int32()),
            "GENERATED": pa.array(arrays["GENERATED"], pa.bool_()),
        }
    )
    pq.write_table(table, out_path)
    n_generated = int(arrays["GENERATED"].sum()) if table.num_rows else 0
    print(f"Wrote {len(rows):,} trajectories, {n_generated:,} sampled events, to {out_path}")


def load_model(checkpoint_dir: Path, device: torch.device) -> Delphi:
    """The Delphi model in `checkpoint_dir`, in eval mode on `device`."""
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    if config.model_class_name != "Delphi":
        raise ValueError(
            f"{checkpoint_dir} holds a {config.model_class_name!r} model; trajectory sampling needs a Delphi one"
        )
    model = Delphi(config)
    state = torch.load(checkpoint_dir / "lit_model.pth", map_location="cpu", mmap=True, weights_only=False)
    # `litgpt pretrain` saves the whole training state, the weights under "model".
    model.load_state_dict(state.get("model", state))
    return model.to(device).eval()


def load_data_module(hyperparameters_path: Path) -> DelphiLifeEvents:
    """Rebuilds the data module the checkpoint was trained with, from its saved hyperparameters."""
    hparams = yaml.safe_load(Path(hyperparameters_path).read_text())
    spec = hparams.get("data")
    if not isinstance(spec, dict) or "class_path" not in spec:
        raise ValueError(f"{hyperparameters_path} has no `data` config to read the vocabulary and prompts from")
    module_name, _, class_name = spec["class_path"].rpartition(".")
    cls = getattr(importlib.import_module(module_name or "litgpt.data"), class_name)
    if not issubclass(cls, DelphiLifeEvents):
        raise ValueError(
            f"{hyperparameters_path} trained on {cls.__name__}, trajectory sampling needs DelphiLifeEvents"
        )
    return cls(**spec.get("init_args", {}))


def _cached_prompts(data: DelphiLifeEvents, split: str) -> tuple[SequenceStore, np.ndarray]:
    """The prepared cache and the persons of `split` in it, split exactly as in training."""
    data.setup()
    store = data.train_dataset.store
    if split == "all":
        return store, np.arange(len(store))
    dataset = data.val_dataset if split == "val" else data.train_dataset
    return store, np.sort(dataset.indices)


def _encode_prompts(
    data: DelphiLifeEvents,
    event_paths: list[str],
    person_path: Path,
    cache_dir: Path,
    on_unknown_event: Literal["error", "drop"],
) -> SequenceStore:
    """Runs `event_paths` through the training data pipeline, with the checkpoint's vocabulary."""
    vocab_path = data.cache_dir / "vocab.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"{vocab_path} is missing; the prompts have to be encoded with the training vocabulary")
    meta = build_event_cache(
        cache_dir,
        event_paths=event_paths,
        specials=data.specials,
        person_path=person_path,
        time_origin=data.time_origin,
        min_time=data.min_time,
        vocab_path=vocab_path,
        id_column=data.id_column,
        time_column=data.time_column,
        event_column=data.event_column,
        birthdate_column=data.birthdate_column,
        # A single event is a usable prompt, unlike a training sequence.
        min_events=1,
        distinct_events=data.distinct_events,
        on_missing_birthdate=data.on_missing_birthdate,
        on_unknown_event=on_unknown_event,
        duckdb_threads=data.duckdb_threads,
        duckdb_memory_limit=data.duckdb_memory_limit,
        duckdb_temp_directory=data.duckdb_temp_directory,
    )
    # `build_event_cache` leaves the completion marker to its caller.
    (cache_dir / "meta.json").write_text(json.dumps(meta))
    return SequenceStore.load(cache_dir, data.specials)


def _build_prompt(
    store: SequenceStore, i: int, data: DelphiLifeEvents, cutoff: int | None, max_length: int, seed: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Person `i`'s conditioning `(tokens, ages)`, or `None` if nothing is observed by `cutoff`.

    Holds the person's most recent events.
    """
    sequence = store.sequence(i)
    tokens, ages = sequence["tokens"].astype(np.int64), sequence["times"].astype(np.int64)
    if cutoff is not None:
        keep = ages <= cutoff
        tokens, ages = tokens[keep], ages[keep]
    if tokens.size == 0:
        return None

    if data.no_event_mode != "none":
        end = int(ages[-1]) if cutoff is None else cutoff
        extra = no_event_ages(data.no_event_mode, data.no_event_rate, end, np.random.default_rng([seed, i]))
        if cutoff is not None:
            # The person is known to be event-free up to the cutoff, and a no-event token
            # is how the model is told so. Without one, sampling would resume from the
            # last event and could place new events inside the observed, quiet stretch.
            extra = np.union1d(extra, [cutoff])
        tokens, ages, _ = insert_no_events(tokens, ages, extra)
    return tokens[-max_length:], ages[-max_length:]


def _chunks(rows: list, size: int) -> Iterator[list]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _pad(prompts: list[tuple[np.ndarray, np.ndarray]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pads prompts into a batch, the layout `generate_trajectories` expects."""
    width = max(tokens.size for tokens, _ in prompts)
    idx = torch.full((len(prompts), width), PAD_ID, dtype=torch.int64)
    age = torch.full((len(prompts), width), PAD_AGE, dtype=torch.int64)
    for b, (tokens, ages) in enumerate(prompts):
        idx[b, : tokens.size] = torch.from_numpy(tokens)
        age[b, : ages.size] = torch.from_numpy(ages)
    return idx.to(device), age.to(device)

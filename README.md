# Life event sequence modelling

This repository is a fork from [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt) that adds GPT-style
sequence models for life-course event data.

## Input data

The input data for training the model comes from a "silver" data layer (Databricks terminology; see https://github.com/NL-LENS/data-pipeline).
The life events will come from different life event dataset files in Parquet format. The schema of these datasets will always be the same:

```
ID | TIME       | EVENT       | NOTES
------------------------------------------------
0  | 2026-01-01 | EVENT_LABEL | {str: Any} 
1  | 2026-01-01 | EVENT_LABEL | {str: Any} 
.. | ...        | ...         | ... 
```

The `ID` column corresponds to the ID of a person to which the sequence belongs. The `TIME` column contains the event time stamp in ISO data format standard (daily resolution). The `EVENT` column contains the labels of the different events. The `NOTES` column contains event attributes (ignored for now).

Time-invariate person attributes are stored in a separate dataset (also in Parquet format).

## Quickstart

First, to create dummy silver layer event datasets in the expected format, for trying out the pipeline end-to-end:

```bash
litgpt dummy_data --out_dir data/dummy_life_events
```

It writes two event datasets with deliberately different `NOTES` schemas
(`work_education.parquet`, `civil_registry.parquet`) and a `persons.parquet` with
`BIRTHDATE`. By default, they contain 2000 invented people, ~27k events, 13 event types. You can pass `--n_persons` to create
larger datasets. It is made up, so nothing trained on it means anything.

Then pretrain Delphi on it:

```bash
litgpt pretrain --config config_hub/pretrain/delphi-2m.yaml
```

The first run preprocesses the Parquet files into `data/dummy_life_events/cache` and
prints what it kept. Later runs reuse that cache, so the cache should be deleted after changing
the source data. Training reports `loss` as the sum of `loss_ce` (which event) and
`loss_dt` (time difference between events), and writes checkpoints to `out/pretrain/delphi-2m`.

To only build the cache, without training, add `--prepare_data_only`:

```bash
litgpt pretrain --config config_hub/pretrain/delphi-2m.yaml --prepare_data_only true
```

It reads the data settings from the same config, prints the summary and exits before any
model, logger, run directory or device is touched. This can run anywhere, including on a
machine with no GPU. The training run afterwards finds the cache and skips the rebuild.
Preparation is worth separating when it is slow, when it belongs on a different machine
than training, or when several runs should share one cache.

To train on real data, `data.init_args.event_paths` and `person_path` should point to 
Parquet files (a directory or a glob works too) and `model_config.vocab_size` and
`padded_vocab_size` should be set to the `vocab_size` that `--prepare_data_only true` prints.

Static person-level attributes such as gender or migration background are columns of the
person dataset, named in `data.init_args.static_fields` (e.g. `--data.init_args.static_fields '[GENDER]'`).
Each value becomes a token `GENDER=F`, and a missing value becomes `GENDER=<unknown>`. These tokens
come last in the vocabulary, so they count towards `vocab_size`. As in Delphi, they open every
sequence at age 0 and are only read by the model: they survive windowing, are never prediction
targets and are shuffled in training (`shuffle_statics`). Set `model_config.target_vocab_size` to the
value the cache summary prints, the vocabulary without the static tokens: only those logits count
as event rates, in the loss and when sampling, so static tokens never add to the total event rate
and `litgpt generate` never samples them.

To sample trajectories from a trained checkpoint:

```bash
litgpt generate out/pretrain/delphi-2m/final --cutoff_age 50 --num_samples 10 --termination_events '[death]'
```

It conditions each held-out person (`--split val`, the default) on what they experienced
up to age 50 and simulates the rest of their life course. The model, the vocabulary and
the persons all come from the checkpoint directory and the cache its config points at.
To simulate other people, pass their silver-layer datasets as `--prompt_paths '[...]'`
and `--prompt_person_path`, which get encoded with the checkpoint's vocabulary. The result,
`out/generate/trajectories.parquet` by default, has the silver-layer columns
`ID | TIME | EVENT`, plus `SAMPLE` and `GENERATED` (prompt or sampled). `<no_event>` rows
are kept, so filter them out before feeding the file back into the data pipeline.
`litgpt generate --help` lists all options.

Or, from Python:

```python
import torch
from litgpt import Config
from litgpt.life_events.data import EventVocab
from litgpt.life_events.delphi import generate_trajectories
from litgpt.life_events.delphi.data import SPECIALS
from litgpt.model import model_class

out = "out/pretrain/delphi-2m/final"
config = Config.from_file(f"{out}/model_config.yaml")
model = model_class(config)(config)  # -> Delphi, per `model_class_name`
model.load_state_dict(torch.load(f"{out}/lit_model.pth", weights_only=False)["model"])

# The cache holds the vocabulary the checkpoint was trained with.
vocab = EventVocab.load("data/dummy_life_events/cache/vocab.json", SPECIALS)
idx = torch.tensor([vocab.encode(["birth", "school_start"])])
age = torch.tensor([[0, 2200]])  # in days, as the model was trained
events, ages = generate_trajectories(
    model, idx, age, max_new_tokens=10, termination_tokens=vocab.encode(["death"]),
    # `<no_event>` is a real outcome the model was trained on, so it gets sampled too.
    # Add it here to draw only actual events: ignore_tokens=(vocab.pad_id, 1).
)
for event, days in zip(vocab.decode(events[0].tolist()), ages[0].tolist()):
    print(f"{days / 365.25:6.1f} years  {event}")
```

## Code layout

All life-event related code lives in `litgpt/life_events/`. The module is split into a model-agnostic data layer and one package per architecture:

```
litgpt/life_events/
  data/                  # knows nothing about any model
    source.py            # silver layer -> cache (DuckDB): merge, time axis, filters, NOTES fields
    store.py             # SequenceStore: memory-mapped tokens, times and note features per individual
    vocab.py             # EventVocab with architecture-defined special tokens
    datamodule.py        # LifeEventsDataModule: prepare_data / split by individual / dataloaders
  delphi/                # everything specific to Delphi
    data.py              # DelphiLifeEvents: <no_event> insertion, windowing, shift, tie-mask collate
    model.py             # Delphi(GPT): age encoding + token dropout
    mask.py              # build_tie_mask, last_visible_index
    loss.py              # competing-risks likelihood (loss_ce + loss_dt)
    generate.py          # generate_trajectories
```

`litgpt/generate.py` wires `generate_trajectories` into the `litgpt generate` command.

Training: `litgpt pretrain delphi-2m --data DelphiLifeEvents --data.event_paths '[...]' --data.person_path ...`. The `delphi-2m` config sets `model_class_name: Delphi`, which is how `litgpt.pretrain` picks the model class and its loss. Note that `--model_config.*` replaces the named config entirely, so when using it also pass `--model_config.model_class_name Delphi`. `pretrain` refuses to pair `DelphiLifeEvents` with any other model class.

### Adding an architecture

1. Subclass `litgpt.model.GPT`. Override `embed(idx, **inputs)` to change the input embedding, and the static methods `model_inputs(batch)`, `compute_loss(logits, batch, config)` and `example_batch(...)` to define what the model reads from a batch and which loss it trains under.
2. Register it in `litgpt.model._MODEL_CLASSES`, or set `model_class_name` to its dotted import path.
3. Subclass `LifeEventsDataModule`: set `specials` (reserved tokens, padding first) and implement `make_dataset(store, indices, train)` and `collate_fn()`. Batches are dicts of `(B, T)` tensors (and `(B, 1, T, T)` masks). They must already be shifted, with the targets under `labels`.
4. If the model only works on one time axis, restrict `time_origins` (Delphi sets `("birth",)`); the base class rejects any other.

### Data module options

Every field of the data module is settable from the CLI as `--data.<field>` or under
`data.init_args` in a YAML config. `litgpt pretrain --data DelphiLifeEvents --print_config`
lists them all with their defaults (`--data.help DelphiLifeEvents` shows descriptions, but
only for the fields a subclass declares itself, not the inherited ones).

When the event datasets exceed the available memory resources, settings that control DuckDB, which does the
merging, joining and sorting in `prepare_data`, can be tuned as follows:

| Field | Default | What it does |
| --- | --- | --- |
| `duckdb_threads` | all cores | Threads for the cache build. Lower it to leave room for other work. |
| `duckdb_memory_limit` | ~80% of RAM | e.g. `8GB`. **Should be paired it with a temp directory**. |
| `duckdb_temp_directory` | none | Where DuckDB spills to disk, which is what lets the sort and the join exceed the memory limit. |

For example:

```bash
litgpt pretrain --config config_hub/pretrain/delphi-2m.yaml --prepare_data_only true \
  --data.duckdb_threads 8 --data.duckdb_memory_limit 8GB --data.duckdb_temp_directory /scratch/duckdb
```

These can also be set with environment variables.

### The time axis

The cache gives every event an integer time coordinate in whole days, but *what it is measured from* differs between model architectures, set by `time_origin`:

| `time_origin` | `store.times` holds | needs `person_path` |
| --- | --- | --- |
| `birth` (default) | age in days | yes, for `BIRTHDATE` |
| `epoch` | days since 1970-01-01 | no |
| an ISO date, e.g. `2000-01-01` | days since that date | no |

Without a person dataset the individuals are simply whoever appears in the events. `min_time` drops events before a given coordinate; it defaults to `0` for `birth` (dropping events dated before birth) and to `None` otherwise, where a negative number just means earlier than the reference date.

A dataset should read `store.time_origin` rather than assume one because the numbers look the same on any axis. Delphi accepts only `birth`, because its `<no_event>` tokens are drawn over a 100-year lifetime and `AgeEncoding` encodes age since birth.

**Models that need more than one axis** (say age *and* calendar time) do not need a second cache, and nothing is stored twice. Every origin is a date, so the axes differ by one constant per individual, which the cache keeps in `time_offsets`:

```python
sequence = store.sequence(i)
age = sequence["times"]                             # the stored axis
calendar = store.times_from(i, age, "epoch")        # days since 1970-01-01
since_2000 = store.times_from(i, age, "2000-01-01")
```

Build such a cache with `time_origin="birth"`: that is the only origin that records each individual's birthdate, so it is the only one from which every other axis is reachable. The dataset then puts both arrays in the batch under whatever names its model reads.

### Using NOTES attributes

`NOTES` is not read unless requested. `notes_fields` maps a attribute name to the field to extract:

```yaml
data:
  class_path: DelphiLifeEvents
  init_args:
    notes_fields:
      income: {field: income, dtype: DOUBLE, fill: 0.0}
```

It works whether `NOTES` is stored as a struct, a map or a JSON string. Each dataset is read against its own schema. An event whose dataset lacks the field, or whose value is null, gets `fill`. Only numeric types are supported for now. The values are cached as `note_<name>.npy` and appear in batches as `note_<name>`, aligned with `input_ids`. Delphi ignores them; a new architecture can consume them in its `embed`.

## Testing

Run tests from the repository root. The `--extra test` option includes the testing
dependencies declared in `pyproject.toml`; `uv run` prepares and uses the project's
virtual environment automatically.

During development, run the tests relevant to your changes. For example, after
changing data preprocessing:

```bash
uv run --extra test pytest tests/life_events/test_source.py -q
```

Before opening a pull request or merging changes, run the full suite:

```bash
uv run --extra test pytest tests/ -q
```

Replace `-q` with `-v` to see individual test names and results.

## Generative AI usage

The code has been partially generated and refined with Claude Code (Opus 5). The authors have manually checked and
validated the code for correctness.

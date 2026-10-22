# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""Architecture-agnostic life-course event data.

This module is responsible for general reading of preprocessed "silver" layer input data, data loading, tokenization,
caching, and building the event vocabulary.

Input event data are expected to be stored in Parquet files with the schema `ID | TIME | EVENT | NOTES`, with
`ID` a 64-bit precision integer, `TIME` an ISO date at daily resolution, `EVENT` a string label, and `NOTES` and
struct or mapping with potentially heterogeneous fields (currently not used).

The data-loading cache is stored in binary numpy files.

"""

from litgpt.life_events.data.datamodule import LifeEventsDataModule
from litgpt.life_events.data.source import BIRTH_ORIGIN, EPOCH_ORIGIN, NoteField, build_event_cache
from litgpt.life_events.data.store import SequenceStore
from litgpt.life_events.data.vocab import EventVocab

__all__ = [
    "BIRTH_ORIGIN",
    "EPOCH_ORIGIN",
    "EventVocab",
    "LifeEventsDataModule",
    "NoteField",
    "SequenceStore",
    "build_event_cache",
]

# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""The event vocabulary shared by every life-course event model architecture."""

import json
from collections.abc import Sequence
from pathlib import Path

__all__ = ["EventVocab"]

VOCAB_VERSION = 1


class EventVocab:
    """Maps categorical life events to token ids.

    Life-course event sequences have a closed vocabulary of discrete event labels.
    The first tokens ids are reserved for `specials`, whose first entry is always the
    padding token at id 0. The remaining ids follow the order in which `events` is given,
    which `build_event_cache` keeps alphabetical. Which specials exist is up to the
    architecture: For example, Delphi reserves `("<pad>", "<no_event>")`.

    Args:
        events: Event names, without the special tokens.
        specials: Reserved tokens, padding first.
    """

    def __init__(self, events: Sequence[str], specials: Sequence[str] = ("<pad>",)) -> None:
        events, specials = list(events), tuple(specials)
        if not specials:
            raise ValueError("`specials` must contain at least the padding token")
        if len(set(events)) != len(events):
            raise ValueError("`events` contains duplicates")
        for special in specials:
            if special in events:
                raise ValueError(f"{special!r} is added automatically and must not appear in `events`")
        self.specials = specials
        self.itos: list[str] = [*specials, *events]
        self.stoi: dict[str, int] = {event: i for i, event in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def first_event_id(self) -> int:
        return len(self.specials)

    @property
    def events(self) -> list[str]:
        return self.itos[self.first_event_id :]

    def encode(self, events: Sequence[str]) -> list[int]:
        try:
            return [self.stoi[event] for event in events]
        except KeyError as e:
            raise ValueError(f"Unknown event {e.args[0]!r}, vocabulary has {self.vocab_size} entries") from None

    def decode(self, ids: Sequence[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": VOCAB_VERSION, "specials": list(self.specials), "events": self.events}
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path, specials: Sequence[str] | None = None) -> "EventVocab":
        """Loads a saved vocabulary, checking its specials when `specials` is given.

        The check is what stops a vocabulary written for one layout of reserved ids from
        being read under another.
        """
        payload = json.loads(Path(path).read_text())
        saved = payload.get("specials")
        expected = saved if specials is None else list(specials)
        if saved is None or saved != expected:
            raise ValueError(
                f"{path} declares specials {saved!r}, expected {expected!r}. "
                "A vocabulary built for a different set of reserved tokens is not compatible; "
                "delete the cache directory to rebuild."
            )
        return cls(payload["events"], specials=saved)

    def __repr__(self) -> str:
        return f"EventVocab(vocab_size={self.vocab_size}, specials={list(self.specials)}, events={self.events})"

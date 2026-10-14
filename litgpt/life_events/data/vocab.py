# Copyright Netherlands eScience Center and Open Data Infrastructure for Social Science and Economic Innovations
# (ODISSEI). Licensed under the Apache License 2.0, see LICENSE file.
# Partially generated and refined with Claude Code (Opus 5).
"""The event vocabulary shared by every life-course event model architecture."""

import json
from collections.abc import Sequence
from pathlib import Path

__all__ = ["EventVocab"]

VOCAB_VERSION = 2
"""Version 2 added `statics`."""

UNKNOWN_STATIC = "<unknown>"
"""The value a static attribute takes for a person whose value is missing."""


def static_label(field: str, value: str) -> str:
    """The vocabulary label of the static attribute `field` taking `value`."""
    return f"{field}={value}"


class EventVocab:
    """Maps categorical life events to token ids.

    Life-course event sequences have a closed vocabulary of discrete event labels.
    The first tokens ids are reserved for `specials`, whose first entry is always the
    padding token at id 0. The remaining ids follow the order in which `events` is given,
    which `build_event_cache` keeps alphabetical. Which specials exist is up to the
    architecture: For example, Delphi reserves `("<pad>", "<no_event>")`.

    Static person-level attributes, such as gender, come last, as labels `"<FIELD>=<value>"`.
    They are tokens a model can read but should never predict, and sit after the events so
    that adding them to a vocabulary does not renumber any event.

    Args:
        events: Event names, without the special tokens.
        specials: Reserved tokens, padding first.
        statics: Static attribute labels, as `static_label` builds them.
    """

    def __init__(
        self, events: Sequence[str], specials: Sequence[str] = ("<pad>",), statics: Sequence[str] = ()
    ) -> None:
        events, specials, statics = list(events), tuple(specials), list(statics)
        if not specials:
            raise ValueError("`specials` must contain at least the padding token")
        if len(set(events) | set(statics)) != len(events) + len(statics):
            raise ValueError("`events` and `statics` contain duplicates")
        for special in specials:
            if special in events or special in statics:
                raise ValueError(f"{special!r} is added automatically and must not appear in `events` or `statics`")
        self.specials = specials
        self.n_events = len(events)
        self.itos: list[str] = [*specials, *events, *statics]
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
    def first_static_id(self) -> int:
        return self.first_event_id + self.n_events

    @property
    def events(self) -> list[str]:
        return self.itos[self.first_event_id : self.first_static_id]

    @property
    def statics(self) -> list[str]:
        return self.itos[self.first_static_id :]

    @property
    def static_ids(self) -> list[int]:
        return list(range(self.first_static_id, self.vocab_size))

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
        payload = {
            "version": VOCAB_VERSION,
            "specials": list(self.specials),
            "events": self.events,
            "statics": self.statics,
        }
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
        return cls(payload["events"], specials=saved, statics=payload.get("statics", []))

    def __repr__(self) -> str:
        return (
            f"EventVocab(vocab_size={self.vocab_size}, specials={list(self.specials)}, events={self.events}, "
            f"statics={self.statics})"
        )

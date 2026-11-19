"""The answer record every tool returns. Facts live here; the accessibility layer only
changes how they are presented, never what they say."""
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Route:
    kind: str                      # image | audio | document | video | code | chat
    task: str                      # count, read_text, transcribe, qa, write_code, ...
    modifiers: list[str] = field(default_factory=list)   # tts, flashcards, captions, card
    source: str = "rules"          # rules | llm | default


@dataclass
class Answer:
    answer: str                                        # short, direct; shown first, scored by eval
    evidence: dict[str, Any] = field(default_factory=dict)   # quote / boxes / timeline / tests
    confidence: float | None = None
    abstain: bool = False                              # True = "the input does not contain this"
    card: dict[str, Any] | None = None                 # filled by the accessibility layer
    route: Route | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)     # warnings shown to the team, not the user

    def to_dict(self) -> dict:
        return asdict(self)

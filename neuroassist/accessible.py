"""Accessibility layer: summary cards, flashcards, captions and read-aloud.

These steps change how an answer is presented, never what it says. Each one is checked against the
source: a card bullet with a number the source never mentions is dropped, and so is a flashcard whose
answer is not supported by the source."""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from . import config, llm
from .tools.text_qa import _norm, unknown_numbers

CARD = {"type": "object", "required": ["title", "one_line", "bullets", "key_terms"], "properties": {
    "title": {"type": "string"},
    "one_line": {"type": "string"},
    "bullets": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    "key_terms": {"type": "array", "maxItems": 4, "items": {
        "type": "object", "required": ["term", "meaning"],
        "properties": {"term": {"type": "string"}, "meaning": {"type": "string"}}}}}}

FLASH = {"type": "object", "required": ["cards"], "properties": {"cards": {
    "type": "array", "maxItems": 6, "items": {
        "type": "object", "required": ["q", "a"], "properties": {"q": {"type": "string"}, "a": {"type": "string"}}}}}}

SYSTEM = ("You rewrite information for students with ADHD or dyslexia. Use short sentences and everyday words. "
          "Use ONLY the facts you are given. Never add facts, numbers or names.")

STOP = set("the a an and or of to in on at for with is are was were be been it this that these those as by from "
           "into than then so but not no yes its their his her they he she we you i".split())


def _clip(text: str, n: int = 6000) -> str:
    return text if len(text) <= n else text[:n] + "\n[...]"


def make_card(question: str, answer: str, source: str) -> dict:
    facts = f"Question: {question}\nAnswer: {answer}\n\nSource:\n<<<\n{_clip(source)}\n>>>"
    r = llm.chat_json(config.CARD_MODEL,
                      facts + "\n\nMake a study card. 'title': 3-6 words. 'one_line': the answer in one plain sentence. "
                      "'bullets': up to 5 key points, each under 15 words. 'key_terms': up to 4 hard words from "
                      "the source, each with a simple meaning.", CARD, system=SYSTEM,
                      num_ctx=config.NUM_CTX_DOC, max_tokens=400)
    known = f"{answer}\n{source}"
    bullets = [b.strip() for b in r.get("bullets", []) if isinstance(b, str) and b.strip()]
    kept = [b for b in bullets if not unknown_numbers(b, known)]
    terms = [t for t in r.get("key_terms", []) if isinstance(t, dict) and t.get("term")
             and _norm(t["term"]) in _norm(known)]
    card = {"title": (r.get("title") or "").strip() or "Summary",
            "one_line": (r.get("one_line") or answer).strip(),
            "bullets": kept, "key_terms": terms}
    if len(kept) < len(bullets):
        card["dropped"] = len(bullets) - len(kept)
    return card


def _supported(text: str, source: str, min_share: float = 0.6) -> bool:
    words = [w for w in _norm(text).split() if w not in STOP and len(w) > 2]
    if not words:
        return bool(_norm(text)) and _norm(text) in _norm(source)
    src = set(_norm(source).split())
    return sum(w in src for w in words) / len(words) >= min_share and not unknown_numbers(text, source)


def make_flashcards(source: str, n: int = 6) -> list[dict]:
    r = llm.chat_json(config.CARD_MODEL,
                      f"Source:\n<<<\n{_clip(source)}\n>>>\n\nWrite up to {n} flashcards that test the key facts in "
                      "the source. 'q': a short question. 'a': a short answer (a few words) taken from the source.",
                      FLASH, system=SYSTEM, num_ctx=config.NUM_CTX_DOC, max_tokens=900)
    cards = [{"q": c["q"].strip(), "a": c["a"].strip()} for c in r.get("cards", [])
             if isinstance(c, dict) and c.get("q") and c.get("a")]
    cards = [c for c in cards if len(c["a"]) > 1 and not re.search(r"[-–—:,(]\s*$", c["a"])
             and c["q"].rstrip().endswith("?")]          # drop cut-off or malformed cards
    return [c for c in cards if _supported(c["a"], source)][:n]


# --- Captions -----------------------------------------------------------------------------------
def _ts(t: float) -> str:
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def to_vtt(speech: list[dict] | None = None, visual: list[dict] | None = None, duration: float | None = None) -> str:
    """speech: Whisper segments {start, end, text}. visual: timeline events {t, description}, which become
    '[Visual] ...' cues lasting until the next event (an audio-description track)."""
    cues = [(s["start"], s["end"], s["text"]) for s in speech or [] if s.get("text")]
    vis = [e for e in visual or [] if e.get("description")]
    for i, e in enumerate(vis):
        end = vis[i + 1]["t"] if i + 1 < len(vis) else (duration or e["t"] + 2.0)
        cues.append((e["t"], max(end, e["t"] + 0.5), f"[Visual] {e['description']}"))
    cues.sort(key=lambda c: c[0])
    return "WEBVTT\n\n" + "\n\n".join(f"{_ts(a)} --> {_ts(b)}\n{t}" for a, b, t in cues) + "\n"


# --- Read aloud (offline) -----------------------------------------------------------------------
_PS_TTS = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
           "$s.Rate = [int]$env:NA_RATE; $s.SetOutputToWaveFile($env:NA_OUT); "
           "$s.Speak([IO.File]::ReadAllText($env:NA_IN)); $s.Dispose()")


def speak(text: str, rate: int = 0) -> str | None:
    """Writes a WAV of `text` with the voices built into Windows (no internet, no model download).
    rate: -10 (slow) .. 10 (fast). Returns the file path, or None if no speech engine is available."""
    text = re.sub(r"[*_#`>|]", " ", text).strip()
    if not text:
        return None
    out = Path(tempfile.mkdtemp(prefix="neuroassist_tts_")) / "answer.wav"
    if sys.platform == "win32":
        src = out.with_suffix(".txt")
        src.write_text(text, encoding="utf-8")
        env = {**os.environ, "NA_IN": str(src), "NA_OUT": str(out), "NA_RATE": str(int(rate))}
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_TTS],
                       env=env, capture_output=True, timeout=120)
    else:
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", 180 + 15 * rate)
            eng.save_to_file(text, str(out))
            eng.runAndWait()
        except Exception:
            return None
    return str(out) if out.exists() and out.stat().st_size > 1000 else None

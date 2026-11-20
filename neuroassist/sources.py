"""Every file as located segments, so any claim can point back to where it came from.

    PDF       one segment per paragraph, located by page          "page 2"
    text      one segment per paragraph, located by line          "line 14"
    code      the whole file                                      "line 1"
    audio     Whisper segments, located by time                   "02:14"
    video     Whisper segments of its audio track + (optional) frame descriptions   "00:03"
    image     one description from the vision model (optional)    "photo"

Linking across files (`related`) and conflict hints (`conflicts`) are plain code, no model:
they are explainable and cannot invent a connection the words do not support."""
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import config
from .ingest import load_document, sniff
from .tools.text_qa import _norm, quote_in_source

STOP = set("""a an the and or of to in on at for with is are was were be been being it its this that these those as
by from into than then so but not no yes their his her they he she we you i your our my me us them will would
can could should must may might do does did done have has had also any all each every some such which who what
when where why how there here about over under up down out more most less very just only own same other""".split())


@dataclass
class Segment:
    file: str
    loc: str                    # human-readable location: "page 2", "line 14", "02:14", "photo"
    text: str
    t: float | None = None      # seconds, for audio/video
    page: int | None = None
    kind: str = "text"          # text | speech | visual
    lines: list | None = None   # [(line number, text)] inside a text paragraph, for exact locations


@dataclass
class Source:
    name: str
    path: str
    kind: str                   # document | code | audio | video | image
    segments: list[Segment] = field(default_factory=list)
    error: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)


def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60:02d}:{t % 60:02d}"


def parse_time(s: str) -> float | None:
    """'12:40' -> 760, '1:02:03' -> 3723, '95' -> 95, '2m10s' -> 130. None if not a time."""
    s = (s or "").strip().lower()
    if not s:
        return None
    m = re.fullmatch(r"(?:(\d+)h)?\s*(?:(\d+)m(?:in)?)?\s*(?:(\d+)s(?:ec)?)?", s)
    if m and any(m.groups()):
        h, mi, se = (int(x) if x else 0 for x in m.groups())
        return h * 3600 + mi * 60 + se
    parts = s.split(":")
    if all(p.isdigit() for p in parts) and 1 <= len(parts) <= 3:
        v = 0
        for p in parts:
            v = v * 60 + int(p)
        return float(v)
    return None


def _paragraphs(text: str, with_lines: bool = False) -> list:
    """(line number, paragraph) for blank-line separated paragraphs; long ones split by sentences.
    with_lines=True adds the paragraph's own [(line number, line)] as a third item."""
    out, line, buf, start = [], 1, [], 1
    for raw in text.splitlines() + [""]:
        if raw.strip():
            if not buf:
                start = line
            buf.append((line, raw.strip()))
        elif buf:
            out.append((start, " ".join(t for _, t in buf), buf))
            buf = []
        line += 1
    res = []
    for ln, p, lines in out:
        if len(p) <= 700:
            res.append((ln, p, lines) if with_lines else (ln, p))
            continue
        sents, chunk = re.split(r"(?<=[.!?])\s+", p), ""
        for s in sents:
            if len(chunk) + len(s) > 600 and chunk:
                res.append((ln, chunk.strip(), lines) if with_lines else (ln, chunk.strip()))
                chunk = ""
            chunk += " " + s
        if chunk.strip():
            res.append((ln, chunk.strip(), lines) if with_lines else (ln, chunk.strip()))
    return res


def _pdf_pages(path) -> list[str]:
    from pypdf import PdfReader
    return [(p.extract_text() or "") for p in PdfReader(str(path)).pages]


def load(path: str, session=None, visual: bool = False) -> Source:
    """`visual=True` also describes photos/video frames with the vision model (slower)."""
    name, kind = Path(path).name, sniff(path)
    src = Source(name, str(path), kind)
    memo = (lambda key, fn, *p: session.memo(path, key, fn, *p)) if session else (lambda key, fn, *p: fn())
    try:
        if kind == "document" and Path(path).read_bytes()[:4] == b"%PDF":
            for i, page in enumerate(memo("pdf_pages", lambda: _pdf_pages(path)), 1):
                src.segments += [Segment(name, f"page {i}", p, page=i) for _, p in _paragraphs(page)]
            if not src.segments:
                src.error = "PDF has no text layer (scanned)"
        elif kind in ("document", "code"):
            text = memo("document", lambda: load_document(path))
            if kind == "code":
                src.segments = [Segment(name, "line 1", text)]
            else:
                src.segments = [Segment(name, f"line {ln}", p, lines=lines)
                                for ln, p, lines in _paragraphs(text, with_lines=True)]
        elif kind in ("audio", "video"):
            from .tools import audio
            try:
                tr = memo("transcript", lambda: audio.transcribe(path), config.WHISPER_MODEL)
                src.segments = [Segment(name, mmss(s["start"]), s["text"], t=s["start"], kind="speech")
                                for s in tr["segments"]]
            except Exception as e:          # e.g. a video with no audio track
                src.error = f"no speech ({type(e).__name__})"
            if kind == "video" and visual:
                from .tools import video
                events, _dur = video.timeline(path, session)
                src.segments += [Segment(name, mmss(e["t"]), e["description"], t=e["t"], kind="visual")
                                 for e in events if e.get("description")]
                src.segments.sort(key=lambda s: s.t or 0)
        elif kind == "image" and visual:
            from .tools import image
            desc = memo("description", lambda: image.describe(path))
            if desc:
                src.segments = [Segment(name, "photo", desc, kind="visual")]
    except Exception as e:
        src.error = f"{type(e).__name__}: {e}"
    return src


# --- Locating and linking ------------------------------------------------------------------------
def _exact_line(seg: Segment, quote: str) -> Segment:
    """Inside a text paragraph, point at the line where the quote starts (not the heading above it)."""
    if not seg.lines or len(seg.lines) < 2:
        return seg
    head = _norm(quote)[:40]
    for ln, text in seg.lines:
        if head[:25] and head[:25] in _norm(text):
            return replace(seg, loc=f"line {ln}")
    return seg


def locate(quote: str, src: Source) -> Segment | None:
    """The segment that contains `quote` (exact or near-exact), else None."""
    for seg in src.segments:
        if quote_in_source(quote, seg.text):
            return _exact_line(seg, quote)
    # A quote spanning two adjacent segments (e.g. two Whisper segments)
    for a, b in zip(src.segments, src.segments[1:]):
        if quote_in_source(quote, a.text + " " + b.text):
            return a
    return None


def _stem(w: str) -> str:
    """Crude suffix stripping so 'computing'/'compute' and 'temperatures'/'temperature' match."""
    for suf in ("ings", "ing", "ed", "es", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def words(text: str) -> set[str]:
    return {_stem(w) for w in _norm(text).replace("_", " ").split()
            if w not in STOP and len(w) > 2 and not w.isdigit()}


def related(text: str, sources: list[Source], exclude: str = "", k: int = 2, min_shared: int = 2,
            min_score: float = 0.25) -> list[tuple[Segment, list[str]]]:
    """Segments in other files that share key words with `text`: [(segment, shared words)].
    Ranked by the share of `text`'s key words they contain. Deliberately simple and explainable."""
    want = words(text)
    if not want:
        return []
    scored = []
    for src in sources:
        if src.name == exclude:
            continue
        for seg in src.segments:
            shared = want & words(seg.text)
            score = len(shared) / len(want)
            if len(shared) >= min_shared and score >= min_score:
                # Show the words as they are written, not their stems ("average", not "averag")
                shown = {_stem(w): w for w in _norm(seg.text).replace("_", " ").split()}
                scored.append((score, _exact_line(seg, ""), sorted(shown.get(s, s) for s in shared)))
    scored.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for _, seg, shared in scored:
        if (seg.file, seg.loc) in seen:
            continue
        seen.add((seg.file, seg.loc))
        out.append((seg, shared))
        if len(out) >= k:
            break
    return out


_DAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_MONTHS = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
FACTS = {
    "day": rf"\b({_DAYS})\b",
    "date": rf"\b(\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})|(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?|\d{{1,2}}/\d{{1,2}})\b",
    "time": r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight)\b",
    "words": r"\b(\d+(?:\s*(?:-|to)\s*\d+)?)\s+words\b",
    "pages": r"\b(\d+(?:\s*(?:-|to)\s*\d+)?)\s+pages?\b",
    "percent": r"\b(\d+)\s*(?:%|percent)",
}


def facts(text: str) -> dict[str, set[str]]:
    t = text.lower()
    return {k: {re.sub(r"\s+", " ", m if isinstance(m, str) else m[0]) for m in re.findall(p, t)}
            for k, p in FACTS.items() if re.search(p, t)}


def conflicts(a: str, b: str) -> list[str]:
    """Deadline-like facts both texts state but with different values, e.g. 'Friday' vs 'Thursday',
    '150 to 200 words' vs '300 words'. Only a hint: shown as 'check both', never resolved."""
    fa, fb = facts(a), facts(b)
    out = []
    for k in fa.keys() & fb.keys():
        if k == "percent":           # several percentages in one passage are normal (grading weights)
            continue
        if fa[k].isdisjoint(fb[k]):
            out.append(f"{k}: {', '.join(sorted(fa[k]))} vs {', '.join(sorted(fb[k]))}")
    return out


def context(seg: Segment, quote: str, width: int = 260) -> tuple[str, str, str]:
    """(before, match, after) around `quote` inside its segment, for showing it in place."""
    text = seg.text
    i = text.lower().find(quote.lower()[:40]) if quote else -1
    if i < 0:
        return "", text[: 2 * width], ""
    j = i + len(quote)
    return text[max(0, i - width):i], text[i:j], text[j:j + width]

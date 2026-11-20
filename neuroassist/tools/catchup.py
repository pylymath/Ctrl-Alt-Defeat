"""Lecture catch-up: "I zoned out at 12:40, catch me up."

Takes the part of a recording (or a video's audio, plus its frame descriptions if already made) from
a time onwards, and returns a short recap where every point carries the timestamp it came from, plus
the instructions said in that window kept word for word. Points whose timestamp is not in the window,
or that mention numbers the window never says, are dropped. It recaps what was SAID; it does not
claim to know what the student missed or understood."""
import re

from .. import config, llm
from ..schema import Answer
from ..sources import mmss, parse_time
from .text_qa import unknown_numbers

TIME = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?|\d+\s*m(?:in(?:utes?)?)?\s*(?:\d+\s*s(?:ec(?:onds?)?)?)?|\d+\s*s(?:ec(?:onds?)?)?)",
                  re.I)
INSTRUCTION = re.compile(r"\b(must|should|need to|have to|make sure|remember|don't|do not|submit|hand in|due|"
                         r"deadline|start with|before|after|read|write|bring|use)\b", re.I)

RECAP = {"type": "object", "required": ["points"], "properties": {"points": {
    "type": "array", "maxItems": 6, "items": {
        "type": "object", "required": ["line", "text"], "properties": {
            "line": {"type": "integer"}, "text": {"type": "string"}}}}}}


def parse_window(question: str) -> tuple[float | None, float | None]:
    """'catch me up from 12:40' -> (760, None); 'between 2:00 and 3:30' -> (120, 210);
    'the last 5 minutes' -> (-300, None) meaning 5 minutes before the end."""
    q = question.lower()
    m = re.search(r"last\s+(\d+)\s*(min|minutes|m|sec|seconds|s)\b", q)
    if m:
        n = int(m.group(1)) * (60 if m.group(2).startswith("m") else 1)
        return -float(n), None
    times = [parse_time(re.sub(r"\s+", "", t)) for t in TIME.findall(q)]
    times = [t for t in times if t is not None]
    if not times:
        return None, None
    if len(times) >= 2 and re.search(r"\b(to|until|till|and|-)\b", q):
        return times[0], times[1]
    return times[0], None


def catch_up(segments: list[dict], start: float, end: float | None = None, visual: list[dict] | None = None) -> dict:
    """segments: Whisper [{start, end, text}]; visual: [{t, description}] (optional).
    Returns {start, end, points: [{t, loc, text}], instructions: [{t, loc, text}], window: [...]}."""
    total = max([s["end"] for s in segments] + [e["t"] for e in visual or []] + [0])
    if start < 0:
        start = max(0.0, total + start)
    end = end if end is not None else total
    win = [s for s in segments if s["end"] > start and s["start"] < end]
    vis = [e for e in visual or [] if start <= e["t"] <= end]
    out = {"start": start, "end": end, "points": [], "instructions": [], "window": win}
    if not win and not vis:
        return out
    # Numbered lines: small models refer to "line 3" far more reliably than to "[02:14]".
    lines = sorted([(s["start"], s["text"]) for s in win] + [(e["t"], f"(on screen) {e['description']}") for e in vis])
    log = "\n".join(f"{i}. [{mmss(t)}] {txt}" for i, (t, txt) in enumerate(lines, 1))
    text = " ".join(txt for _, txt in lines)

    r = llm.chat_json(config.TEXT_MODEL,
                      f"Part of a lecture recording, from {mmss(start)} to {mmss(end)}, one numbered line each:\n"
                      f"{log}\n\nA student lost focus at the start of this part. Recap it in 2 to 6 short points, in "
                      "time order, in plain words of your own, only what the lines say. 'line' = the number of the "
                      "line the point comes from.", RECAP, num_ctx=config.NUM_CTX_DOC, max_tokens=500)
    seen = set()
    for p in r.get("points", []):
        if not isinstance(p, dict) or not p.get("text") or unknown_numbers(p["text"], text):
            continue
        try:
            k = int(str(p.get("line", "")).strip(". "))
        except ValueError:
            continue
        if not 1 <= k <= len(lines) or (k, p["text"]) in seen:
            continue                                      # a line that doesn't exist: dropped
        seen.add((k, p["text"]))
        t = lines[k - 1][0]
        out["points"].append({"t": t, "loc": mmss(t), "text": p["text"].strip()})
    out["points"].sort(key=lambda p: p["t"])
    # Instructions are kept word for word (not rewritten): they are what students act on.
    out["instructions"] = [{"t": s["start"], "loc": mmss(s["start"]), "text": s["text"]}
                           for s in win if INSTRUCTION.search(s["text"])]
    return out


TIME_PHRASE = re.compile(r"\b(?:at|around|near|about|by)?\s*\d{1,2}:\d{2}(?::\d{2})?\b", re.I)
GENERIC = re.compile(r"what (?:did|does|was|is|were)\b.*\b(?:say|said|talk|talking|saying|happen|happening|going on|"
                     r"mention|discuss)|what'?s (?:said|happening)|^\W*$", re.I)


def at_time(question, files, route, session) -> Answer:
    """'What did she say at 1:20?' -> the exact lines around 1:20, with their timestamps (no model).
    'What window size does she suggest around 0:45?' -> a checked answer from just that part, with the
    timestamp of the sentence it came from."""
    from . import audio, video
    from .text_qa import answer_from_text, quote_in_source
    path = files[0]
    t, _ = parse_window(question)
    if t is None or t < 0:
        return Answer("Tell me the time, e.g. 'What did she say at 1:20?'", abstain=True,
                      evidence={"reason": "no time in the question"})
    segments, visual = [], None
    try:
        segments = session.memo(path, "transcript", lambda: audio.transcribe(path), config.WHISPER_MODEL)["segments"]
    except Exception:
        pass
    if route.kind == "video":
        visual, _ = video.timeline(path, session)
    lo, hi = max(0.0, t - 10), t + 20
    win = [s for s in segments if s["end"] > lo and s["start"] < hi]
    vis = [e for e in visual or [] if lo <= e["t"] <= hi]
    lines = sorted([(s["start"], s["text"]) for s in win] + [(e["t"], f"(on screen) {e['description']}") for e in vis])
    ev = {"at_time": {"t": t, "loc": mmss(t), "lines": [{"t": a, "loc": mmss(a), "text": b} for a, b in lines[:12]]}}
    if not lines:
        return Answer(f"Nothing was said or shown around {mmss(t)}.", abstain=True, evidence=ev)
    rest = TIME_PHRASE.sub(" ", question).strip(" ?.,")
    if GENERIC.search(rest) or len(rest.split()) < 3:
        body = "\n".join(f"[{mmss(a)}] {b}" for a, b in lines[:6])
        return Answer(f"Around {mmss(t)}:\n{body}", evidence=ev, confidence=1.0)
    text = "\n".join(b for _, b in lines)
    ans = answer_from_text(rest + "?", text, source="recording")
    quote = ans.evidence.get("quote")
    if quote and not ans.abstain:
        at = next((a for a, b in lines if quote_in_source(quote, b)), lines[0][0])
        ans.answer = f"[{mmss(at)}] {ans.answer}"
    ans.evidence.update(ev)
    return ans


def run(question, files, route, session) -> Answer:
    from . import audio, video
    path = files[0]
    start, end = parse_window(question)
    if start is None:
        return Answer("Tell me where to start, e.g. 'catch me up from 12:40' or 'the last 5 minutes'.",
                      abstain=True, evidence={"reason": "no time in the question"})
    segments, visual, notes = [], None, []
    try:
        tr = session.memo(path, "transcript", lambda: audio.transcribe(path), config.WHISPER_MODEL)
        segments = tr["segments"]
    except Exception as e:
        notes.append(f"no speech ({type(e).__name__})")
    if route.kind == "video":
        visual, _ = video.timeline(path, session)
    res = catch_up(segments, start, end, visual)
    if not res["window"] and not (visual and res["points"]):
        a = Answer(f"Nothing was said between {mmss(res['start'])} and {mmss(res['end'])}.", abstain=True,
                   evidence={"catchup": res})
        a.notes += notes
        return a
    head = f"From {mmss(res['start'])} to {mmss(res['end'])}:"
    body = "\n".join(f"[{p['loc']}] {p['text']}" for p in res["points"]) or \
        "\n".join(f"[{s['loc'] if 'loc' in s else mmss(s['start'])}] {s['text']}" for s in res["window"][:6])
    a = Answer(f"{head}\n{body}", evidence={"catchup": {**res, "window": res["window"][:40]}}, confidence=0.8)
    a.notes += notes
    return a

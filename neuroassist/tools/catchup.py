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
        "type": "object", "required": ["time", "text"], "properties": {
            "time": {"type": "string"}, "text": {"type": "string"}}}}}}


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
    log = "\n".join([f"[{mmss(s['start'])}] {s['text']}" for s in win]
                    + [f"[{mmss(e['t'])}] (on screen) {e['description']}" for e in vis])
    text = " ".join(s["text"] for s in win) + " " + " ".join(e["description"] for e in vis)
    stamps = sorted({s["start"] for s in win} | {e["t"] for e in vis})

    r = llm.chat_json(config.TEXT_MODEL,
                      f"Part of a lecture recording, from {mmss(start)} to {mmss(end)}:\n{log}\n\n"
                      "A student lost focus at the start of this part. Recap it in 2 to 6 short points, in time "
                      "order, plain words, only what is in the text. 'time' = the [mm:ss] of the line the point "
                      "comes from.", RECAP, num_ctx=config.NUM_CTX_DOC, max_tokens=500)
    for p in r.get("points", []):
        if not isinstance(p, dict) or not p.get("text"):
            continue
        t = parse_time(str(p.get("time", "")).strip("[] "))
        if t is None or not stamps or unknown_numbers(p["text"], text):
            continue
        t = min(stamps, key=lambda s: abs(s - t))       # snap to a real line in the window
        if abs(t - (parse_time(str(p.get('time', '')).strip('[] ')) or t)) > 15:
            continue                                      # a time far from any real line: dropped
        out["points"].append({"t": t, "loc": mmss(t), "text": p["text"].strip()})
    out["points"].sort(key=lambda p: p["t"])
    # Instructions are kept word for word (not rewritten): they are what students act on.
    out["instructions"] = [{"t": s["start"], "loc": mmss(s["start"]), "text": s["text"]}
                           for s in win if INSTRUCTION.search(s["text"])]
    return out


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

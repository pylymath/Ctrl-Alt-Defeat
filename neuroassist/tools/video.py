"""Short clips. Sampling depends on the task: counting and motion get dense frames, a plain
description gets sparse ones. Short clips go to the model in one call ("direct"); longer ones are
described in batches, merged into a timeline, and answered from the timeline ("timeline")."""
from .. import config, llm
from ..ingest import extract_frames
from ..schema import Answer

DENSE_TASKS = {"count_events", "count", "motion", "sequence"}

UNSURE = "I could not tell."
ANSWER = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}
FRAMES = {"type": "object", "required": ["frames"], "properties": {"frames": {"type": "array", "items": {
    "type": "object", "required": ["t", "description"], "properties": {
        "t": {"type": "number"}, "description": {"type": "string"}}}}}}

HINTS = {
    "count_events": "Compare consecutive frames carefully and count each event once.",
    "count": "Things may appear one at a time; count every distinct one across all frames.",
    "sequence": "Look at EVERY frame from the first to the last. If colours or things appear one after another, "
                "list them all in the order they appear, then say which was first and which was last.",
    "motion": "Compare positions and sizes between the first and last frames, and name the direction in plain "
              "words (left, right, up, down, toward, away, taller, shorter).",
}

# A full sentence naming the object and its colour is clearer for students than a bare "Up" or "6".
# (The examples are deliberately unlike any practice-pack item, so no answer leaks into the prompt.)
FULL_ANSWER = ("Answer in one full sentence that names the object and its colour, for example 'The purple "
               "triangle slides to the left.' or 'An orange star appears first, then a pink one, and a grey one "
               "appears last.' Write any number as digits (4, not four).")


def _labels(frames) -> str:
    return "\n".join(f"Image {i + 1} is at t={f.t:.1f}s." for i, f in enumerate(frames))


def _direct(question, frames, duration, task) -> Answer:
    prompt = (f"These {len(frames)} images are frames from a {duration:.1f}-second video, in time order.\n"
              f"{_labels(frames)}\n{HINTS.get(task, '')}\n\nQuestion: {question}\n{FULL_ANSWER}")
    r = llm.chat_json(config.VISION_MODEL, prompt, ANSWER, images=[f.jpeg for f in frames],
                      num_ctx=config.NUM_CTX_VIDEO, max_tokens=120)
    return Answer((r.get("answer") or "").strip() or UNSURE,
                  evidence={"frames_used": [round(f.t, 2) for f in frames], "mode": "direct"})


def describe_frames(frames, batch: int = 4) -> list[dict]:
    """One plain description per frame, in batches of `batch` images: [{t, description}]."""
    events = []
    for s in range(0, len(frames), batch):
        chunk = frames[s:s + batch]
        prompt = (f"These are consecutive frames from a video.\n{_labels(chunk)}\n"
                  "For each frame, describe exactly what is visible: objects, colors, positions, any text, "
                  "and what changed since the previous frame. Use the given t values.")
        r = llm.chat_json(config.VISION_MODEL, prompt, FRAMES, images=[f.jpeg for f in chunk],
                          num_ctx=config.NUM_CTX_VIDEO, max_tokens=120 * len(chunk))
        for f, e in zip(chunk, r.get("frames", [])):
            events.append({"t": round(f.t, 2), "description": (e.get("description") or "").strip()})
    return events


def _frames(path, task, session):
    fps = config.VIDEO_FPS_DENSE if task in DENSE_TASKS else config.VIDEO_FPS_SPARSE
    return session.memo(
        path, "frames", lambda: extract_frames(path, fps, config.VIDEO_MAX_FRAMES, config.VIDEO_SIDE),
        fps, config.VIDEO_MAX_FRAMES, config.VIDEO_SIDE)


def timeline(path, session) -> tuple[list[dict], float]:
    """Frame-by-frame descriptions of a whole clip (sparse sampling), cached per file."""
    frames, duration = _frames(path, "describe", session)
    return session.memo(path, "timeline", lambda: describe_frames(frames), len(frames)), duration


def _timeline(question, frames, duration, task, batch: int = 4) -> Answer:
    events = describe_frames(frames, batch)
    log = "\n".join(f"[{e['t']:.1f}s] {e['description']}" for e in events)
    r = llm.chat_json(config.TEXT_MODEL,
                      f"Timeline of a {duration:.1f}-second video, one line per sampled frame:\n{log}\n\n"
                      f"{HINTS.get(task, '')}\nQuestion: {question}\n{FULL_ANSWER}",
                      ANSWER, num_ctx=config.NUM_CTX_DOC, max_tokens=120)
    return Answer((r.get("answer") or "").strip() or UNSURE,
                  evidence={"timeline": events, "mode": "timeline"})


def run(question, files, route, session) -> Answer:
    path = files[0]
    frames, duration = _frames(path, route.task, session)
    mode = config.VIDEO_MODE
    if mode == "auto":
        mode = "direct" if len(frames) <= config.VIDEO_DIRECT_MAX_FRAMES else "timeline"
    ans = (_direct if mode == "direct" else _timeline)(question, frames, duration, route.task)
    if ans.answer == UNSURE:
        # Harness fallback: the 10 failed practice clips were all empty replies, not wrong ones.
        # Try the other mode once (direct <-> timeline) before saying "I could not tell".
        other = "timeline" if mode == "direct" else "direct"
        if other == "direct" and len(frames) > config.VIDEO_DIRECT_MAX_FRAMES:
            step = len(frames) / config.VIDEO_DIRECT_MAX_FRAMES
            frames = [frames[round(i * step)] for i in range(config.VIDEO_DIRECT_MAX_FRAMES - 1)] + [frames[-1]]
        llm.note("fallback", config.VISION_MODEL, f"video {mode} gave no answer, trying {other}")
        second = (_direct if other == "direct" else _timeline)(question, frames, duration, route.task)
        if second.answer != UNSURE:
            second.notes.append(f"{mode} mode gave no answer; answered with {other} mode")
            ans = second
    ans.evidence["duration_s"] = round(duration, 2)
    return ans

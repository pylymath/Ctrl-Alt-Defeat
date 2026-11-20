"""answer(question, files) -> Answer. The one function the UI and the eval both call.

A request becomes a short plan of steps, run in order, each with its own model:

    core      one tool per file (vision / Whisper + text / coder ...), or one per sub-question
    combine   several files or sub-questions -> one answer (text model)
    card      accessible summary card (card model, checked against the source)
    plain     paragraph-by-paragraph plain-language rewrite (numbers checked against each paragraph)
    literal   idioms / vague words / implied expectations explained literally (phrases must be in the source)
    flashcards                          (card model, checked against the source)
    captions  WebVTT: Whisper speech cues + vision-model "[Visual]" cues for video
    tts       the answer (or card) read aloud, offline

Every step is recorded in Answer.steps (name, model, time, ok), so the UI can show what ran."""
import re
import time
from typing import Callable

from . import config, llm, router
from .ingest import file_hash, load_document
from .schema import Answer, Route


class Session:
    """Per-conversation cache. Follow-up questions about the same file reuse its transcript,
    frames or document text instead of reprocessing it."""

    def __init__(self):
        self._cache: dict = {}

    def memo(self, path, name: str, fn: Callable, *params):
        key = (file_hash(path), name, params)
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]


_default_session = Session()
DEFAULT_QA_TASK = "qa"
POST_STEPS = ("card", "plain", "literal", "flashcards", "captions", "tts")


def _handlers():
    from .tools import audio, code, image, text_qa, video
    return {"image": image.run, "video": video.run, "audio": audio.run,
            "document": text_qa.run, "code": code.run, "chat": code.run_chat}


def _model_for(r: Route) -> str:
    if r.kind in ("image", "video"):
        return config.VISION_MODEL
    if r.kind == "audio":
        return f"whisper-{config.WHISPER_MODEL}" if r.task != "identify_sound" else "AST sound classifier"
    if r.task == "catchup":
        return f"whisper-{config.WHISPER_MODEL} + {config.TEXT_MODEL}"
    if r.task == "explain_code":
        return f"python (runs it) + {config.CODE_MODEL}"
    if r.task == "at_time":
        return f"whisper-{config.WHISPER_MODEL} + {config.QA_MODEL}"
    if r.kind == "code" or r.task == "write_code":
        return config.CODE_MODEL
    return config.CARD_MODEL if r.task == "summarize" else config.QA_MODEL


def _step(trace: list, name: str, model: str, fn: Callable):
    """Run one step and record it: model, time, harness events (retries, timeouts, fallbacks) and
    the GPU memory Ollama holds after it, so the UI can show what the step really cost."""
    t, n_events = time.perf_counter(), len(llm.events)

    def record(ok: bool, **extra):
        row = {"step": name, "model": model, "ms": round((time.perf_counter() - t) * 1000), "ok": ok, **extra}
        ev = llm.events[n_events:]
        if ev:
            row["events"] = [f"{e['kind']}: {e['detail']}" for e in ev]
        if config.TRACK_VRAM:
            mem = llm.loaded()
            if mem:
                row["vram_gb"] = round(sum(m["vram_gb"] for m in mem), 1)
                row["loaded"] = [m["name"] for m in mem]
        trace.append(row)

    try:
        out = fn()
        record(True)
        return out
    except Exception as e:
        record(False, error=f"{type(e).__name__}: {e}")
        raise


def split_questions(question: str) -> list[str]:
    """'How many people are there? What does the sign say?' -> two questions. Only splits on '?'
    so a single question with 'and' in it is never broken up."""
    parts = [p.strip() for p in re.split(r"(?<=\?)\s+", question.strip()) if p.strip()]
    if len(parts) < 2 or any(len(p.split()) < 3 for p in parts):
        return [question]
    return parts


TWO_PART = re.compile(r"^(?P<a>.+?),?\s+and\s+(?P<b>(?:who|whom|whose|what|which|where|when|why|how)\b.+\?)\s*$", re.I)


def split_two_part(question: str) -> list[str]:
    """'What instrument did she play, and whose letter did she read?' -> two questions, so a document /
    transcript answer covers BOTH parts (the QA model answers one thing per call)."""
    m = TWO_PART.match(question.strip())
    if not m:
        return [question]
    a, b = m.group("a").strip().rstrip(",") + "?", m.group("b").strip()
    b = b[0].upper() + b[1:]
    if len(a.split()) < 3 or len(b.split()) < 3:
        return [question]
    return [a, b]


def _core(question: str, files: list, session: Session, trace: list, label: str = "answer") -> Answer:
    r = router.route(question, files)
    if r.task == "rewrite":
        parts = [x for x, m in (("in plain language", "plain"), ("with literal meanings", "literal"))
                 if m in r.modifiers]
        if "flashcards" in r.modifiers and not parts:
            ans = Answer("Here are flashcards to quiz yourself. Tap a card to see the answer.")
        else:
            ans = Answer(f"Here is your {r.kind} {' and '.join(parts)}.")
        ans.route = r
        return ans
    handler = _handlers()[r.kind]
    if r.task == "catchup":
        from .tools import catchup
        handler = catchup.run
    elif r.task == "explain_code":
        from .tools import code_explain
        handler = code_explain.run_tool
    elif r.task == "at_time":
        from .tools import catchup
        handler = catchup.at_time
    elif r.task == "transcribe" and r.kind == "video":     # a video's sound track, with timestamps
        from .tools import audio
        handler = audio.run
    ans = _step(trace, f"{label}: {r.kind} / {r.task}", _model_for(r),
                lambda: handler(question, files, r, session))
    ans.route = r
    return ans


ANSWER = {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}


def _combine_files(question: str, parts: list[tuple[str, Answer]], trace: list) -> str:
    facts = "\n".join(f"- From {name}: {a.answer}" for name, a in parts)
    r = _step(trace, "combine answers", config.TEXT_MODEL, lambda: llm.chat_json(
        config.TEXT_MODEL, f"Findings from each file:\n{facts}\n\nQuestion: {question}\n"
        "Answer the question using ONLY these findings, in one or two short sentences.",
        ANSWER, num_ctx=config.NUM_CTX_DOC, max_tokens=200))
    return (r.get("answer") or "").strip() or facts


# --- What the post steps work from --------------------------------------------------------------
def _transcript(path, session) -> dict:
    from .tools import audio
    return session.memo(path, "transcript", lambda: audio.transcribe(path), config.WHISPER_MODEL)


def source_text(ans: Answer, files: list, session: Session) -> str:
    kind = ans.route.kind if ans.route else "chat"
    if not files or kind == "chat":
        return ans.answer
    path = files[0]
    if kind in ("document", "code"):
        return session.memo(path, "document", lambda: load_document(path))
    if kind == "audio":
        tr = _transcript(path, session)
        return tr["text"] if tr["has_speech"] else ans.answer
    if kind == "video":
        events = ans.evidence.get("timeline") or []
        lines = [f"[{e['t']:.1f}s] {e['description']}" for e in events]
        return "\n".join([ans.answer] + lines)
    detail = ans.evidence.get("detail")
    return f"{ans.answer}\n{detail}" if detail else ans.answer


def _captions(ans: Answer, files: list, session: Session, trace: list) -> str | None:
    from . import accessible
    from .tools import video
    kind = ans.route.kind if ans.route else "chat"
    if kind == "audio":
        tr = _step(trace, "captions: speech", f"whisper-{config.WHISPER_MODEL}",
                   lambda: _transcript(files[0], session))
        return accessible.to_vtt(speech=tr["segments"])
    if kind != "video":
        ans.notes.append("captions only apply to audio and video")
        return None
    (events, duration) = _step(trace, "captions: describe frames", config.VISION_MODEL,
                               lambda: video.timeline(files[0], session))
    ans.evidence.setdefault("timeline", events)
    speech = []
    try:      # faster-whisper decodes the clip's own audio track; silent clips just have no speech
        tr = _step(trace, "captions: speech", f"whisper-{config.WHISPER_MODEL}",
                   lambda: _transcript(files[0], session))
        speech = tr["segments"]
    except Exception as e:
        ans.notes.append(f"no speech captions ({type(e).__name__}: {e}); captions describe the picture only")
    if ans.route.task == "describe":
        # Audio-description request: replace the one-line answer with a short step-by-step narration
        # written only from what the frames and the speech showed.
        log = "\n".join([f"[{e['t']:.1f}s] {e['description']}" for e in events]
                        + [f"[{s['start']:.1f}s] (speech) {s['text']}" for s in speech])
        r = _step(trace, "audio description", config.TEXT_MODEL, lambda: llm.chat_json(
            config.TEXT_MODEL, f"Timeline of a {duration:.1f}-second video:\n{log}\n\n"
            "Write an audio description for someone who cannot see the video: 3 to 6 short sentences, "
            "in time order, plain words, only what the timeline shows.",
            ANSWER, num_ctx=config.NUM_CTX_DOC, max_tokens=300))
        if (r.get("answer") or "").strip():
            ans.evidence["short_answer"] = ans.answer
            ans.answer = r["answer"].strip()
    return accessible.to_vtt(speech=speech, visual=events, duration=duration)


def _post(question: str, ans: Answer, files: list, session: Session, wanted: list[str], tts_rate: int):
    from . import accessible
    trace = ans.steps
    src = None
    for name in [s for s in POST_STEPS if s in wanted]:
        try:
            if name in ("card", "flashcards", "plain", "literal") and src is None:
                src = source_text(ans, files, session)
            if name == "card":
                if ans.abstain:
                    continue
                ans.card = _step(trace, "summary card", config.CARD_MODEL,
                                 lambda: accessible.make_card(question, ans.answer, src))
            elif name == "plain":
                ans.plain = _step(trace, "plain-language rewrite", config.CARD_MODEL,
                                  lambda: accessible.plain_rewrite(src))
            elif name == "literal":
                ans.literal = _step(trace, "literal meanings", config.CARD_MODEL,
                                    lambda: accessible.literal_notes(src))
            elif name == "flashcards":
                ans.flashcards = _step(trace, "flashcards", config.CARD_MODEL,
                                       lambda: accessible.make_flashcards(src))
            elif name == "captions" and files:
                ans.captions = _captions(ans, files, session, trace)
            elif name == "tts":
                c = ans.card
                text = (c["one_line"] + ". " + ". ".join(c["bullets"])) if c else ans.answer
                if ans.route and ans.route.task == "write_code":
                    text = "The code is ready. " + (f"It passed {ans.evidence.get('examples_passed')} of "
                                                    f"{ans.evidence.get('examples_total')} examples."
                                                    if ans.evidence.get("examples_total") else "")
                ans.speech_path = _step(trace, "read aloud", "Windows speech (offline)",
                                        lambda: accessible.speak(text, tts_rate))
        except Exception as e:
            ans.notes.append(f"{name} step failed: {type(e).__name__}: {e}")


def answer(question: str, files: list | tuple = (), session: Session | None = None, *,
           extras: bool = True, force: list[str] | tuple = (), tts_rate: int = 0) -> Answer:
    """extras=False runs only the scored answer (the eval uses this). `force` adds output steps
    (card, flashcards, captions, tts) on top of the ones the question asks for."""
    session = session or _default_session
    files = [str(f) for f in files]
    t0 = time.perf_counter()
    trace: list = []

    if len(files) > 1:                          # several files: answer from each, then combine
        parts = [(f.replace("\\", "/").rsplit("/", 1)[-1], _core(question, [f], session, trace, f"file {i + 1}"))
                 for i, f in enumerate(files)]
        ans = Answer(_combine_files(question, parts, trace),
                     evidence={"per_file": [{"file": n, "answer": a.answer, "evidence": a.evidence} for n, a in parts]})
        ans.route = Route("multi", "combine", router.route(question, files[:1]).modifiers, "rules")
        ans.abstain = all(a.abstain for _, a in parts)
    elif len(qs := split_questions(question)) > 1 or (
            files and router.route(question, files).kind in ("document", "audio")
            and router.route(question, files).task == DEFAULT_QA_TASK
            and len(qs := split_two_part(question)) > 1):   # several questions about one input
        subs = [_core(q, files, session, trace, f"question {i + 1}") for i, q in enumerate(qs)]
        ans = Answer("\n".join(f"{i + 1}. {a.answer}" for i, a in enumerate(subs)),
                     evidence={"sub_questions": [{"question": q, "answer": a.answer, "evidence": a.evidence}
                                                 for q, a in zip(qs, subs)]})
        ans.route = Route(subs[0].route.kind, "multi_question",
                          sorted({m for a in subs for m in a.route.modifiers}), "rules")
        ans.abstain = all(a.abstain for a in subs)
    else:
        ans = _core(question, files, session, trace)
    ans.steps = trace

    if extras:
        wanted = list(dict.fromkeys(list(ans.route.modifiers) + list(force)))
        _post(question, ans, files, session, wanted, tts_rate)

    ans.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)
    return ans

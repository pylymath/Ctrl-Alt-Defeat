"""Turn (files, question) into a Route: input kind + one core task + output modifiers.

Rules first (instant, predictable); the resident model is asked only when the rules cannot tell.
Decisions that need to look at the data (e.g. "is there speech in this audio?") are made inside
the tools, not here."""
import re

from . import config, llm
from .ingest import sniff
from .schema import Route

# (task, pattern). First match that is allowed for the input kind wins, so order matters.
TASKS = [
    ("catchup",        r"catch (me )?up|(zoned|spaced|tuned) out|lost (the )?(thread|track|focus)|what did i miss|"
                       r"recap (from|of the last|the last|between)|(from|since|after) \d{1,2}:\d{2}|the last \d+ ?min"),
    ("write_code",     r"\bdef \w+\(|complete (this|the) (python )?function|write (a |some )?(python )?(function|code|script)"),
    ("transcribe",     r"\btranscri|word for word|what (is|are) (he|she|they|the speaker) saying"),
    ("identify_sound", r"what (sound|noise)|what (do|can) (you|i) hear|which (sound|noise)"),
    ("count_events",   r"how many times|bounce|appear.*disappear|\bleft at the end"),
    ("count",          r"\bhow many\b|\bcount\b|number of"),
    ("sequence",       r"\bfirst\b|\blast\b|\bat the (start|end|beginning)\b|\bin the end\b"),
    ("motion",         r"direction|which way|\bmov(e|es|ing)\b|toward|taller|shorter|bigger|smaller|faster"),
    ("read_text",      r"what does (it|this|that|the \w+) say|\bread\b|\btitle\b|\bsign\b|\blabel\b|what('s| is) written"),
    ("identify",       r"what('s| is) (this|that|it)\b|what kind of|what type of|what is (this|the) (item|object|thing)|what is in"),
    ("summarize",      r"summar|simplif|explain|key points|\bnotes\b|tl;?dr|break (it|this) down"),
]

ALLOWED = {
    "image":    {"count", "read_text", "identify", "summarize"},
    "video":    {"catchup", "count_events", "count", "sequence", "motion", "read_text", "summarize"},
    "audio":    {"catchup", "transcribe", "identify_sound", "summarize"},
    "document": {"summarize"},
    "code":     {"write_code", "summarize"},
    "chat":     {"write_code", "summarize"},
}
DEFAULT_TASK = {"image": "describe", "video": "describe", "audio": "qa",
                "document": "qa", "code": "qa", "chat": "qa"}

MODIFIERS = {
    "tts":        r"read (it |this |that )?(aloud|out)|out loud|say it|narrat|speak (it|this)",
    "flashcards": r"flash ?cards?|quiz me|study (guide|pack)|revision cards?",
    "captions":   r"captions?|subtitles?|\.vtt|audio description|accessible description|for (a |someone who is )?(blind|deaf|hard of hearing)",
    "plain":      r"plain (language|english|words)|easier to read|rewrite (it|this) (simply|plainly)|hard to read",
    "literal":    r"literal|what do(es)? (they|it|this|the teacher) (really |actually )?mean|vague|idiom|hidden meaning",
    "card":       r"summar|simplif|key points|break (it|this) down|study (guide|pack|notes)|explain (it|this) simply",
}


def _rules(question: str, kind: str) -> str | None:
    for task, pat in TASKS:
        if task in ALLOWED[kind] and re.search(pat, question, re.I):
            return task
    return None


def _llm_task(question: str, kind: str) -> str | None:
    options = sorted(ALLOWED[kind] | {DEFAULT_TASK[kind]})
    schema = {"type": "object", "properties": {"task": {"enum": options}}, "required": ["task"]}
    prompt = (f"Input type: {kind}\nUser request: {question}\n"
              f"Pick the task that best matches the request from: {', '.join(options)}.")
    try:
        return llm.chat_json(config.TEXT_MODEL, prompt, schema, num_ctx=config.NUM_CTX_ROUTER,
                             max_tokens=20).get("task")
    except Exception:
        return None


def route(question: str, files: list | tuple = ()) -> Route:
    kind = sniff(files[0]) if files else "chat"
    mods = [m for m, pat in MODIFIERS.items() if re.search(pat, question, re.I)]
    task = _rules(question, kind)
    # "Put this in plain language" / "what does the teacher really mean?" asks for a rewrite of the
    # whole input, not a fact from it, so it skips grounded QA (which would answer "does not say").
    if task is None and kind in ("document", "audio", "code") and {"plain", "literal"} & set(mods):
        return Route(kind, "rewrite", mods, "rules")
    if task:
        return Route(kind, task, mods, "rules")
    # The resident model only breaks ties where a wrong default is costly: a bare chat request.
    if kind == "chat" and config.ROUTER_LLM_FALLBACK:
        task = _llm_task(question, kind)
        if task:
            return Route(kind, task, mods, "llm")
    return Route(kind, DEFAULT_TASK[kind], mods, "default")

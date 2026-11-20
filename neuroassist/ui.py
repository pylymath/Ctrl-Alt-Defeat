"""NeuroAssist web UI (runs locally, nothing leaves the laptop).

    python -m neuroassist.ui          # opens http://127.0.0.1:7860
    python -m neuroassist.ui --lan    # also reachable from a phone on the same Wi-Fi (inference stays here)

Three tabs around one assignment:
    1 Understand   ask about any file; catch me up from a timestamp; make a passage easier
    2 Do           help me start (requirements with sources, connected across files), suggested steps,
                   make this smaller, focus mode, a work timer, code checked against its examples
    3 Resume       where was I? (read from the saved record, nothing guessed)"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # fully offline: no usage pings

import html
import re
import tempfile
import threading
import traceback
from pathlib import Path

import gradio as gr

from . import config, llm
from . import workspace as wsp
from .pipeline import Session, answer

OUTPUTS = {"Summary card": "card", "Plain language": "plain", "Literal meanings": "literal",
           "Flashcards": "flashcards", "Captions (audio/video)": "captions", "Read aloud": "tts"}

QUICK = [
    ("Explain simply", "Summarize this simply."),
    ("Plain language", "Rewrite this in plain language."),
    ("What do they really mean?", "What does the teacher really mean? Explain anything vague literally."),
    ("Key points", "What are the key points?"),
    ("Quiz me", "Make flashcards to quiz me on this."),
    ("Audio description", "Give an audio description of this for someone who is blind."),
    ("What does it say?", "What does the text say?"),
    ("Transcribe", "Transcribe this word for word."),
]

FONTS = {
    "Standard": "system-ui, -apple-system, 'Segoe UI', sans-serif",
    "Dyslexia-friendly": "'OpenDyslexic', 'Lexend', Verdana, Tahoma, sans-serif",
    "Large print": "Verdana, Tahoma, sans-serif",
}
BACKGROUNDS = {  # (page, text, accent surface)
    "White": ("#ffffff", "#1d1d1f", "#f2f4f7"),
    "Cream": ("#fbf6e9", "#2b2620", "#f1e8d2"),
    "Pale blue": ("#eef5fb", "#1b2733", "#dbe8f4"),
    "Dark": ("#1e1f22", "#e8e6e3", "#2b2d31"),
}

CSS = """
.na-out { border-radius: 14px; padding: 22px 24px; max-width: 70ch; }
.na-out *, .na-out h3 { color: inherit; font-family: inherit; }
.na-out code { background: #8882; }
.na-answer { font-weight: 600; margin: 0 0 6px; }
.na-badge { display: inline-block; font-size: .8em; padding: 2px 10px; border-radius: 999px; margin-bottom: 10px; }
.na-ok { background: #d9f2e3; color: #14532d; } .na-warn { background: #fde7c7; color: #7a3e00; }
.na-card { border-radius: 12px; padding: 14px 18px; margin-top: 18px; }
.na-card h3 { margin: 0 0 6px; font-size: 1.05em; }
.na-card ul { margin: 8px 0 0 1.1em; padding: 0; } .na-card li { margin-bottom: .45em; }
.na-terms dt { font-weight: 700; } .na-terms dd { margin: 0 0 .5em 0; }
.na-flash details { border-radius: 10px; padding: 10px 14px; margin: 8px 0; cursor: pointer; }
.na-flash summary { font-weight: 600; } .na-flash details[open] summary { margin-bottom: 6px; }
.na-code { white-space: pre-wrap; font-family: Consolas, monospace; font-size: .9em; padding: 12px;
           border-radius: 10px; overflow-x: auto; }
.na-steps { width: 100%; border-collapse: collapse; font-size: .9em; }
.na-steps td, .na-steps th { text-align: left; padding: 4px 8px; border-bottom: 1px solid #8883; }
.na-quote { border-left: 4px solid #6a8dd8; padding: 4px 10px; margin: 6px 0; font-style: italic; }
.na-status { font-size: .9em; opacity: .85; }
.na-plain { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
.na-plain > div { flex: 1 1 260px; border-radius: 10px; padding: 10px 14px; }
.na-orig { opacity: .7; font-size: .9em; }
.na-tag { font-size: .75em; padding: 1px 8px; border-radius: 999px; background: #8883; margin-left: 6px; }
.na-step { display: flex; gap: 10px; align-items: baseline; padding: 6px 0; border-bottom: 1px solid #8882; }
.na-step.done { opacity: .55; text-decoration: line-through; }
.na-step.now { font-weight: 700; }
.na-focus { font-size: 1.5em; font-weight: 700; margin: 10px 0 6px; }
.na-min { font-size: .8em; opacity: .8; white-space: nowrap; }
.na-bar { height: 10px; border-radius: 999px; background: #8883; overflow: hidden; margin: 8px 0 14px; }
.na-bar > div { height: 100%; background: #3a9d5d; }
.na-req { margin: 0 0 14px; }
.na-req summary { cursor: pointer; }
.na-loc { font-size: .78em; padding: 1px 8px; border-radius: 6px; background: #6a8dd833; margin-left: 6px;
          white-space: nowrap; font-style: normal; }
.na-link { font-size: .88em; margin: 4px 0 0 1em; }
.na-conflict { font-size: .88em; margin: 6px 0 0 1em; padding: 4px 10px; border-radius: 8px; background: #fde7c7;
               color: #7a3e00; }
.na-suggest { font-size: .8em; opacity: .75; margin: -4px 0 8px; }
mark.na-hl { background: #ffe58a; color: #1d1d1f; border-radius: 3px; padding: 0 2px; }
.na-timer { font-size: 2em; font-weight: 700; font-variant-numeric: tabular-nums; }
.na-instr { font-size: 1.3em; font-weight: 600; margin: 8px 0; }
"""

_status_cache: dict = {}


def status_html() -> str:
    have = llm.installed_models()
    if not have:
        return ("<div class='na-status'>⚠️ Ollama is not running. Start the Ollama app, then reload this page."
                "</div>")
    need = dict.fromkeys([config.VISION_MODEL, config.TEXT_MODEL, config.QA_MODEL, config.CODE_MODEL, config.CARD_MODEL])

    def present(m):
        return m in have or f"{m}:latest" in have
    parts = [f"{'✅' if present(m) else '❌'} {html.escape(m)}" for m in need]
    missing = [m for m in need if not present(m)]
    tip = f" &nbsp;Missing: run <code>ollama pull {html.escape(missing[0])}</code>" if missing else ""
    return (f"<div class='na-status'>Profile <b>{config.PROFILE}</b> · speech: whisper "
            f"{html.escape(str(config.WHISPER_MODEL))} · {' · '.join(parts)}{tip} · everything runs on this laptop"
            "</div>")


def _esc(s) -> str:
    return html.escape(str(s or ""))


def _bionic(s) -> str:
    """Bold the first half of every word (bionic reading). Escapes the text itself."""
    out = []
    for tok in re.split(r"(\w+)", str(s or "")):
        if tok and re.fullmatch(r"\w+", tok) and not tok.isdigit():
            k = (len(tok) + 1) // 2
            out.append(f"<b>{html.escape(tok[:k])}</b>{html.escape(tok[k:])}")
        else:
            out.append(html.escape(tok))
    return "".join(out)


def _style(font: str, size: int, spacing: float, bg: str) -> tuple[str, str]:
    page, ink, surface = BACKGROUNDS.get(bg, BACKGROUNDS["White"])
    style = (f"background:{page};color:{ink};font-family:{FONTS.get(font, FONTS['Standard'])};"
             f"font-size:{size}px;line-height:{spacing};")
    if font == "Dyslexia-friendly":
        style += "letter-spacing:.04em;word-spacing:.16em;"
    return style, surface


def render(res: dict | None, font: str, size: int, spacing: float, bg: str, bionic: bool = False) -> str:
    T = _bionic if bionic else _esc
    style, surface = _style(font, size, spacing, bg)
    if not res:
        return (f"<div class='na-out' style='{style}'>Add a file (or don't), ask a question, and press "
                "<b>Ask</b>. Answers show up here.</div>")
    if res.get("error"):
        return f"<div class='na-out' style='{style}'><p class='na-answer'>Something went wrong</p>{_esc(res['error'])}</div>"

    route = res.get("route") or {}
    out = [f"<div class='na-out' style='{style}'>"]
    if res.get("abstain"):
        out.append("<span class='na-badge na-warn'>Not in your file</span>")
    elif res.get("confidence") is not None and res["confidence"] >= 0.8:
        out.append("<span class='na-badge na-ok'>Checked against your file</span>")
    if route.get("task") == "write_code":
        ev = res.get("evidence", {})
        if ev.get("examples_total"):
            out.append(f"<span class='na-badge {'na-ok' if ev['examples_passed'] == ev['examples_total'] else 'na-warn'}'>"
                       f"Passed {ev['examples_passed']}/{ev['examples_total']} examples from the prompt</span>")
        out.append(f"<div class='na-code' style='background:{surface}'>{_esc(res['answer'])}</div>")
    else:
        lines = [ln for ln in str(res["answer"]).split("\n") if ln.strip()]
        out.append("".join(f"<p class='na-answer'>{T(ln)}</p>" for ln in lines))

    cu = (res.get("evidence") or {}).get("catchup")
    if cu and cu.get("instructions"):
        out.append(f"<div class='na-card' style='background:{surface}'><h3>📣 Instructions said in this part "
                   "<span style='font-weight:400;font-size:.8em'>(word for word)</span></h3><ul>"
                   + "".join(f"<li><span class='na-loc'>{_esc(i['loc'])}</span> {T(i['text'])}</li>"
                             for i in cu["instructions"]) + "</ul></div>")

    card = res.get("card")
    if card:
        out.append(f"<div class='na-card' style='background:{surface}'><h3>📌 {T(card['title'])}</h3>"
                   f"<div>{T(card['one_line'])}</div>")
        if card.get("bullets"):
            out.append("<ul>" + "".join(f"<li>{T(b)}</li>" for b in card["bullets"]) + "</ul>")
        if card.get("key_terms"):
            out.append("<dl class='na-terms' style='margin-top:10px'>" + "".join(
                f"<dt>{_esc(t['term'])}</dt><dd>{T(t.get('meaning'))}</dd>" for t in card["key_terms"]) + "</dl>")
        out.append("</div>")

    if res.get("plain"):
        out.append("<h3 style='margin:18px 0 0'>✏️ Plain language</h3>")
        for p in res["plain"]:
            tag = "" if p["checked"] else "<span class='na-tag'>kept original: rewrite changed a number</span>"
            out.append(f"<div class='na-plain'><div class='na-orig' style='background:{surface}'>{_esc(p['original'])}"
                       f"</div><div style='background:{surface}'>{T(p['plain'])}{tag}</div></div>")
    if res.get("literal"):
        out.append(f"<div class='na-card' style='background:{surface}'><h3>🔎 What it really means</h3><dl class='na-terms'>"
                   + "".join(f"<dt>“{_esc(i['phrase'])}”<span class='na-tag'>{_esc(i.get('label', i['kind']))}</span></dt>"
                             f"<dd>{T(i['meaning'])}</dd>" for i in res["literal"]) + "</dl></div>")
    elif res.get("literal") == []:
        out.append("<p style='margin-top:14px'>Nothing vague or figurative found.</p>")

    if res.get("flashcards"):
        out.append("<div class='na-flash' style='margin-top:18px'><h3 style='margin:0'>🃏 Flashcards "
                   "<span style='font-weight:400;font-size:.8em'>(tap to flip)</span></h3>")
        out += [f"<details style='background:{surface}'><summary>{T(c['q'])}</summary>{T(c['a'])}</details>"
                for c in res["flashcards"]]
        out.append("</div>")
    elif res.get("flashcards") == []:
        out.append("<p style='margin-top:14px'>No flashcards: none could be checked against the source.</p>")

    if res.get("captions"):
        cues = [c for c in res["captions"].split("\n\n")[1:] if c.strip()]
        out.append(f"<div class='na-card' style='background:{surface}'><h3>💬 Captions</h3>" + "".join(
            f"<div><b>{_esc(c.split(chr(10))[0].split(' --> ')[0][3:8])}</b> {_esc(' '.join(c.split(chr(10))[1:]))}</div>"
            for c in cues[:60]) + "</div>")
    out.append("</div>")
    return "".join(out)


def details_html(res: dict | None) -> str:
    if not res or res.get("error"):
        return _esc((res or {}).get("trace", ""))
    route = res.get("route") or {}
    ev = res.get("evidence") or {}
    out = [f"<p><b>Understood as:</b> {_esc(route.get('kind'))} → {_esc(route.get('task'))}"
           f" (decided by {_esc(route.get('source'))})"]
    if route.get("modifiers"):
        out.append(f" · extras asked for: {_esc(', '.join(route['modifiers']))}")
    out.append("</p><table class='na-steps'><tr><th>#</th><th>Step</th><th>Model</th><th>Time</th>"
               "<th>GPU memory</th></tr>")
    for i, s in enumerate(res.get("steps", []), 1):
        mark = "" if s.get("ok") else " ❌"
        extra = "".join(f"<div class='na-status'>↻ {_esc(e)}</div>" for e in s.get("events", []))
        if s.get("error"):
            extra += f"<div class='na-status'>{_esc(s['error'])}</div>"
        vram = f"{s['vram_gb']:.1f} GB" if s.get("vram_gb") is not None else "–"
        out.append(f"<tr><td>{i}</td><td>{_esc(s['step'])}{mark}{extra}</td><td>{_esc(s['model'])}</td>"
                   f"<td>{s['ms'] / 1000:.1f}s</td><td>{vram}</td></tr>")
    out.append(f"</table><p>Total: {res.get('timings_ms', {}).get('total', 0) / 1000:.1f}s</p>")
    if ev.get("quote"):
        out.append(f"<p><b>Supporting sentence from your file:</b></p><div class='na-quote'>{_esc(ev['quote'])}</div>")
    if ev.get("reason"):
        out.append(f"<p><b>Why no answer:</b> {_esc(ev['reason'])}</p>")
    if ev.get("boxes") is not None:
        out.append(f"<p><b>Objects found:</b> {len(ev['boxes'])} boxes</p>")
    if ev.get("detail"):
        out.append(f"<p><b>Detail:</b> {_esc(ev['detail'])}</p>")
    if ev.get("sounds"):
        out.append("<p><b>Sounds heard:</b> " + ", ".join(
            f"{_esc(s['label'])} ({s['score']:.0%})" for s in ev["sounds"]) + "</p>")
    if ev.get("frames_used"):
        out.append(f"<p><b>Video frames looked at (seconds):</b> {_esc(', '.join(map(str, ev['frames_used'])))}</p>")
    if (ev.get("catchup") or {}).get("window"):
        out.append("<p><b>What was said in that part (transcript):</b></p><ul>" + "".join(
            f"<li>{int(w['start']) // 60:02d}:{int(w['start']) % 60:02d} {_esc(w['text'])}</li>"
            for w in ev["catchup"]["window"]) + "</ul>")
    if ev.get("timeline"):
        out.append("<p><b>Timeline:</b></p><ul>" + "".join(
            f"<li>{e['t']:.1f}s: {_esc(e['description'])}</li>" for e in ev["timeline"]) + "</ul>")
    for key, title in (("per_file", "Per file"), ("sub_questions", "Per question")):
        if ev.get(key):
            out.append(f"<p><b>{title}:</b></p><ul>" + "".join(
                f"<li>{_esc(p.get('file') or p.get('question'))}: {_esc(p['answer'])}</li>" for p in ev[key]) + "</ul>")
    if (res.get("card") or {}).get("dropped"):
        out.append(f"<p>{res['card']['dropped']} card point(s) were removed because they had numbers not in the source.</p>")
    for n in res.get("notes", []):
        out.append(f"<p>ℹ️ {_esc(n)}</p>")
    return "".join(out)


def _paths(files) -> list[str]:
    return [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]


ALL, NONE = "All files together", "No file (just ask)"


def file_choices(files):
    names = [Path(p).name for p in _paths(files)]
    choices = names + ([ALL] if len(names) > 1 else []) + [NONE]
    return gr.update(choices=choices, value=names[0] if names else NONE)


def _pick(files, which) -> list[str]:
    paths = _paths(files)
    if which == NONE:
        return []
    if which == ALL or not which:
        return paths
    return [p for p in paths if Path(p).name == which][:1] or paths[:1]


def _load_ws(name):
    try:
        return wsp.Workspace.load(name) if name else None
    except (FileNotFoundError, ValueError, TypeError):
        return None


def _mmss(sec) -> str:
    sec = max(0, int(sec or 0))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# --- 1 · Understand ---------------------------------------------------------------------------------
def ask(question, files, which, outputs, rate, font, size, spacing, bg, bionic, ws_name, session):
    session = session or Session()
    question = (question or "").strip()
    paths = _pick(files, which)
    if not question:
        question = "Summarize this simply." if paths else ""
    if not question:
        res = {"error": "Type a question first."}
        return (render(res, font, size, spacing, bg, bionic), gr.update(visible=False), gr.update(visible=False), "", res,
                session)
    try:
        a = answer(question, paths, session, force=[OUTPUTS[o] for o in outputs or []], tts_rate=int(rate))
        res = a.to_dict()
        ws = _load_ws(ws_name)
        if ws:                                    # remember it for "Where was I?"
            ws.record(question, a.answer[:300], a.abstain, which if which not in (ALL, NONE) else None)
    except Exception as e:
        msg = str(e)
        if "connect" in msg.lower() or "refused" in msg.lower():
            msg = "Could not reach Ollama. Start the Ollama app and try again. Your saved work is safe."
        res = {"error": msg, "trace": traceback.format_exc()}
    vtt = None
    if res.get("captions"):
        vtt = str(Path(tempfile.mkdtemp(prefix="neuroassist_vtt_")) / "captions.vtt")
        Path(vtt).write_text(res["captions"], encoding="utf-8")
    sp = res.get("speech_path")
    return (render(res, font, size, spacing, bg, bionic), gr.update(value=sp, visible=bool(sp)),
            gr.update(value=vtt, visible=bool(vtt)), details_html(res), res, session)


def catch_me_up(when, files, which, outputs, rate, font, size, spacing, bg, bionic, ws_name, session):
    when = (when or "").strip()
    q = f"Catch me up from {when}" if when else "Catch me up on the last 5 minutes"
    return ask(q, files, which, outputs, rate, font, size, spacing, bg, bionic, ws_name, session)


def _highlight(text: str, unclear: list[dict]) -> str:
    out = html.escape(text)
    for u in unclear:
        pat = re.compile(re.escape(html.escape(u["phrase"])), re.I)
        out = pat.sub(lambda m: f"<mark class='na-hl' title='{html.escape(u['meaning'], quote=True)}'>{m.group(0)}"
                                f"</mark>", out, count=1)
    return out.replace("\n", "<br>")


def make_easier(passage, files, which, session):
    session = session or Session()
    text = (passage or "").strip()
    note = ""
    if not text:
        paths = _pick(files, which)
        if paths:
            from . import sources
            src = sources.load(paths[0], session)
            text = src.text[:2500]
            note = f"No passage selected, so this uses the start of {src.name}."
    if not text:
        raise gr.Error("Paste the passage you want to make easier, or pick a document above.")
    try:
        from . import accessible
        res = accessible.easier(text)
    except Exception as e:
        raise gr.Error(f"Could not process the passage: {e}")
    res["original"], res["note"] = text, note
    return res, 0, session


def render_easier(res, idx, font, size, spacing, bg, bionic=False) -> str:
    T = _bionic if bionic else _esc
    style, surface = _style(font, size, spacing, bg)
    if not res:
        return ""
    out = [f"<div class='na-out' style='{style}'>"]
    if res.get("note"):
        out.append(f"<p class='na-status'>{_esc(res['note'])}</p>")
    if res.get("gist"):
        out.append(f"<p class='na-answer'>💡 {T(res['gist'])}</p>")
    if res.get("details"):
        out.append("<details><summary>More detail</summary><ul>" + "".join(f"<li>{T(d)}</li>" for d in res["details"])
                   + "</ul></details>")
    ins = res.get("instructions") or []
    if ins:
        i = max(0, min(int(idx or 0), len(ins) - 1))
        out.append(f"<div class='na-card' style='background:{surface}'><div class='na-status'>Instruction {i + 1} of "
                   f"{len(ins)} (word for word from the text)</div><div class='na-instr'>{T(ins[i])}</div></div>")
    if res.get("unclear"):
        out.append(f"<div class='na-card' style='background:{surface}'><h3>🔎 Unclear phrases "
                   "<span style='font-weight:400;font-size:.8em'>(highlighted below)</span></h3><dl class='na-terms'>"
                   + "".join(f"<dt>“{_esc(u['phrase'])}”<span class='na-tag'>{_esc(u['label'])}</span></dt>"
                             f"<dd>{T(u['meaning'])}</dd>" for u in res["unclear"]) + "</dl></div>")
    out.append(f"<h3 style='margin-top:18px'>Original, with unclear phrases marked</h3>"
               f"<div class='na-card' style='background:{surface};margin-top:6px'>"
               f"{_highlight(res.get('original', ''), res.get('unclear') or [])}</div>")
    if res.get("plain"):
        out.append("<h3 style='margin-top:18px'>✏️ Plain language, next to the original</h3>")
        for p in res["plain"]:
            tag = "" if p["checked"] else "<span class='na-tag'>kept original: the rewrite changed a number</span>"
            out.append(f"<div class='na-plain'><div class='na-orig' style='background:{surface}'>{_esc(p['original'])}"
                       f"</div><div style='background:{surface}'>{T(p['plain'])}{tag}</div></div>")
    return "".join(out) + "</div>"


def step_instruction(res, idx, delta):
    n = len((res or {}).get("instructions") or [])
    return max(0, min(int(idx or 0) + delta, n - 1)) if n else 0


# --- 2 · Do -----------------------------------------------------------------------------------------
def _req_html(r: dict, T, surface: str) -> str:
    out = [f"<li class='na-req'>{T(r['text'])}<span class='na-loc'>{_esc(r['source'])} · {_esc(r['loc'])}</span>"
           f"<div class='na-quote' style='font-size:.88em'>{_esc(r['quote'])}</div>"]
    for l in r.get("links") or []:
        icon = "👁" if l.get("visual") else "🔗"
        out.append(f"<div class='na-link'>{icon} Also in <b>{_esc(l['file'])}</b> <span class='na-loc'>{_esc(l['loc'])}"
                   f"</span>: “{_esc(l['text'][:180])}{'…' if len(l['text']) > 180 else ''}” "
                   f"<span class='na-status'>({_esc(l['why'])})</span></div>")
    for c in r.get("conflicts") or []:
        out.append(f"<div class='na-conflict'>⚠️ Check both, they differ: {_esc(c)}. "
                   "NeuroAssist does not decide which one is right; ask your teacher.</div>")
    return "".join(out) + "</li>"


def render_plan(ws_name, focus, font, size, spacing, bg, bionic=False) -> str:
    T = _bionic if bionic else _esc
    style, surface = _style(font, size, spacing, bg)
    ws = _load_ws(ws_name)
    if not ws:
        return (f"<div class='na-out' style='{style}'>Add the assignment (PDF, text or a lecture recording) above, "
                "plus any slides, photos or videos that go with it. Give it a name and press <b>Help me start</b>. "
                "You get what it requires, each with the sentence and place it came from, connected across your "
                "files, and a short list of suggested steps.</div>")
    vis = ws.visible_steps()
    done = sum(s["done"] for s in vis)
    nxt = ws.next_step()
    out = [f"<div class='na-out' style='{style}'><p class='na-answer'>🎯 {T(ws.goal)}</p>",
           f"<div class='na-bar'><div style='width:{100 * done / max(len(vis), 1):.0f}%'></div></div>",
           f"<div class='na-status'>{done} of {len(vis)} steps done</div>"]
    if focus:
        if nxt:
            req = ws.requirements[nxt["requirement"]] if nxt.get("requirement") is not None and \
                nxt["requirement"] < len(ws.requirements) else None
            out.append(f"<div class='na-card' style='background:{surface}'><div class='na-status'>Just this one:</div>"
                       f"<div class='na-focus'>{T(nxt['text'])}</div>")
            if req:
                out.append(f"<div class='na-status'>Because the assignment says "
                           f"<span class='na-loc'>{_esc(req['source'])} · {_esc(req['loc'])}</span></div>"
                           f"<div class='na-quote'>{_esc(req['quote'])}</div>")
            out.append("</div>")
        else:
            out.append("<p class='na-focus'>🎉 All steps done.</p>")
        return "".join(out) + "</div>"

    out.append("<h3>Suggested steps</h3><div class='na-suggest'>Suggested by NeuroAssist, not stated in your files. "
               "Add your own or make any of them smaller.</div>")
    if not vis:
        out.append("<p>No steps yet.</p>")
    for i, s in enumerate(vis, 1):
        cls = "done" if s["done"] else ("now" if nxt and s["id"] == nxt["id"] else "")
        mark = "✅" if s["done"] else ("👉" if cls == "now" else "⬜")
        sub = "↳ " if s.get("parent") else ""
        out.append(f"<div class='na-step {cls}'><span>{mark}</span><span>{i}. {sub}{T(s['text'])}</span></div>")

    out.append("<h3 style='margin-top:22px'>What your files say</h3>")
    if ws.requirements:
        for t in wsp.TYPES:
            group = [r for r in ws.requirements if r["type"] == t]
            if group:
                out.append(f"<p style='margin:14px 0 4px'><b>{wsp.TYPE_LABEL[t]}</b></p><ul>"
                           + "".join(_req_html(r, T, surface) for r in group) + "</ul>")
        if not any(r["type"] == "deadline" for r in ws.requirements):
            out.append("<p class='na-status'>⏰ No deadline was found in your files. Ask your teacher.</p>")
    else:
        out.append("<p>No requirement could be matched to a sentence in your files, so none are shown.</p>")
    if ws.dropped:
        out.append(f"<p class='na-status'>{ws.dropped} item(s) were left out because no matching sentence was found "
                   "in your files.</p>")
    errs = [s for s in ws.sources if s.get("error")]
    if errs:
        out.append("<p class='na-status'>" + " · ".join(f"{_esc(s['name'])}: {_esc(s['error'])}" for s in errs) + "</p>")
    return "".join(out) + "</div>"


def step_choices(ws_name):
    ws = _load_ws(ws_name)
    if not ws:
        return gr.update(choices=[], value=None)
    vis = ws.visible_steps()
    choices = [(f"{i}. {'✅ ' if s['done'] else ''}{s['text'][:70]}", s["id"]) for i, s in enumerate(vis, 1)]
    nxt = ws.next_step()
    return gr.update(choices=choices, value=nxt["id"] if nxt else None)


def _ws_names(value=None):
    names = wsp.list_workspaces()
    return gr.update(choices=names, value=value if value in names else (names[0] if names else None))


def start(ws_name, goal, files, visual, session, progress=gr.Progress()):
    session = session or Session()
    name = (ws_name or "").strip() or "My assignment"
    paths = _paths(files)
    if not paths:
        raise gr.Error("Add the assignment file first (PDF, text or a recording).")
    n = [0]

    def say(msg):
        n[0] += 1
        progress(min(0.95, n[0] / (len(paths) * 2 + 3)), desc=msg)
    try:
        wsp.help_me_start(name, paths, goal or "", session, visual=bool(visual), progress=say)
    except Exception as e:
        raise gr.Error(str(e))
    return name, _ws_names(name), session


def smaller(ws_name, sid):
    ws = _load_ws(ws_name)
    if not ws:
        raise gr.Error("Press 'Help me start' first.")
    try:
        wsp.make_smaller(ws, sid)
    except Exception as e:
        raise gr.Error(str(e))
    return ws_name


def set_done(ws_name, sid, done=True):
    ws = _load_ws(ws_name)
    if ws and sid:
        ws.mark_done(sid, done)
    return ws_name


def focus_on(ws_name, sid):
    ws = _load_ws(ws_name)
    if ws and sid:
        ws.focus(sid)
    return ws_name


def add_own(ws_name, text):
    ws = _load_ws(ws_name)
    if not ws:
        raise gr.Error("Press 'Help me start' first.")
    ws.add_step(text or "")
    return ws_name, ""


def timer_start(ws_name, minutes):
    ws = _load_ws(ws_name)
    if not ws:
        raise gr.Error("Press 'Help me start' first.")
    ws.start_timer(int(minutes))
    return timer_html(ws_name)


def timer_stop(ws_name):
    ws = _load_ws(ws_name)
    if ws:
        ws.stop_timer()
    return timer_html(ws_name)


def timer_html(ws_name) -> str:
    ws = _load_ws(ws_name)
    left = ws.timer_left() if ws else None
    if left is None:
        return "<div class='na-status'>No timer running. Pick a length and press Start; it is yours, not a deadline.</div>"
    if left == 0:
        return "<div class='na-timer'>⏰ 00:00</div><div class='na-status'>Time's up. Take a short break, or press ✅ Done.</div>"
    return f"<div class='na-timer'>⏱ {_mmss(left)}</div><div class='na-status'>of your {ws.timer_minutes}-minute timer</div>"


def write_code(prompt, font, size, spacing, bg):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Describe the function, ideally with its signature and an example or two.")
    try:
        res = answer(prompt, [], Session(), extras=False).to_dict()
    except Exception as e:
        res = {"error": str(e)}
    return render(res, font, size, spacing, bg) + (details_html(res) if not res.get("error") else "")


# --- 3 · Resume -------------------------------------------------------------------------------------
def render_resume(ws_name, font, size, spacing, bg, bionic=False) -> str:
    T = _bionic if bionic else _esc
    style, surface = _style(font, size, spacing, bg)
    ws = _load_ws(ws_name)
    if not ws:
        return f"<div class='na-out' style='{style}'>No saved assignment yet. Start one in the <b>Do</b> tab.</div>"
    w = wsp.where_was_i(ws)
    out = [f"<div class='na-out' style='{style}'><span class='na-badge na-ok'>Read from your saved progress, "
           "nothing guessed</span>",
           f"<p class='na-answer'>You're working on: {T(w['goal'])}</p>",
           f"<div class='na-bar'><div style='width:{100 * w['done'] / max(w['total'], 1):.0f}%'></div></div>",
           f"<p>{w['done']} of {w['total']} steps done · last active {_esc(w['last_active'])}</p>"]
    if w["last_done"]:
        out.append(f"<p>✅ Last thing you finished ({_esc(w['last_done_ago'])}): {T(w['last_done'])}</p>")
    if w["next"]:
        out.append(f"<div class='na-card' style='background:{surface}'><div class='na-status'>Your next step "
                   "(suggested):</div>"
                   f"<div class='na-focus'>👉 {T(w['next']['text'])}</div>")
        if w["next_source"]:
            s = w["next_source"]
            out.append(f"<div class='na-status'>It's for <span class='na-loc'>{_esc(s['source'])} · {_esc(s['loc'])}"
                       f"</span></div><div class='na-quote'>{_esc(s['quote'])}</div>")
        out.append("</div>")
    else:
        out.append("<p class='na-focus'>🎉 Every step is done.</p>")
    if w["last_material"]:
        out.append(f"<p>📄 Last file you looked at: <b>{_esc(w['last_material'])}</b></p>")
    if w["last_question"]:
        q = w["last_question"]
        out.append(f"<p>💬 Last thing you asked: “{_esc(q['question'])}”<br>"
                   f"<span class='na-status'>Answer: {_esc(q['answer'])}</span></p>")
    if w["deadlines"]:
        out.append("<p><b>⏰ Deadlines in your files:</b></p><ul>" + "".join(
            f"<li>{T(d['text'])} <span class='na-loc'>{_esc(d['source'])} · {_esc(d['loc'])}</span></li>"
            for d in w["deadlines"]) + "</ul>")
    if w["unresolved"]:
        out.append("<p><b>Still open</b> (your files didn't say; ask your teacher):</p><ul>"
                   + "".join(f"<li>{_esc(u)}</li>" for u in w["unresolved"]) + "</ul>")
    if w["recent"]:
        out.append("<details><summary>Recent activity</summary><ul>" + "".join(
            f"<li>{_esc(e['ago'])}: {_esc(e['event'])}</li>" for e in w["recent"]) + "</ul></details>")
    if w["sources"]:
        out.append(f"<p class='na-status'>Files: {_esc(', '.join(w['sources']))}</p>")
    return "".join(out) + "</div>"


# --- Layout -----------------------------------------------------------------------------------------
def build() -> gr.Blocks:
    with gr.Blocks(title="NeuroAssist") as demo:
        session = gr.State(None)
        last = gr.State(None)
        ws_name = gr.State(None)
        easy_res = gr.State(None)
        easy_idx = gr.State(0)
        gr.Markdown("# 🧠 NeuroAssist\n**Understand** your course materials, get **help to start**, and find your "
                    "way **back** when you lose your place. Everything runs on this laptop; answers come only from "
                    "your files.")
        gr.HTML(status_html)

        saved = wsp.list_workspaces()
        with gr.Row(equal_height=False):
            files = gr.File(label="Your materials (assignment, slides, lecture recording, board photo, video, code)",
                            file_count="multiple", type="filepath", scale=3)
            with gr.Column(scale=2, min_width=260):
                ws_pick = gr.Dropdown(saved, label="Saved assignments", value=saved[0] if saved else None)
                with gr.Accordion("Reading settings", open=False):
                    font = gr.Radio(list(FONTS), value="Standard", label="Font")
                    size = gr.Slider(14, 30, value=18, step=1, label="Text size")
                    spacing = gr.Slider(1.3, 2.4, value=1.7, step=0.1, label="Line spacing")
                    bg = gr.Radio(list(BACKGROUNDS), value="Cream", label="Background")
                    bionic = gr.Checkbox(value=False, label="Bold the start of each word (display option)")
                    rate = gr.Slider(-5, 5, value=0, step=1, label="Reading-aloud speed")
        look = [font, size, spacing, bg]

        with gr.Tabs():
            with gr.Tab("1 · Understand"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        which = gr.Dropdown([NONE], value=NONE, label="Ask about")
                        question = gr.Textbox(label="Your question", lines=3, autofocus=True,
                                              placeholder="e.g. What must I submit?  ·  What does the diagram show?")
                        with gr.Row():
                            quick = [gr.Button(label, size="sm") for label, _ in QUICK]
                        outputs = gr.CheckboxGroup(list(OUTPUTS), label="Also give me", value=[])
                        ask_btn = gr.Button("Ask", variant="primary", size="lg")
                        with gr.Accordion("🧵 I lost the thread: catch me up (recording or video)", open=False):
                            when = gr.Textbox(label="From when?", placeholder="e.g. 12:40  ·  2:00 to 3:30  ·  "
                                                                               "empty = the last 5 minutes")
                            catch_btn = gr.Button("Catch me up")
                    with gr.Column(scale=3, min_width=360):
                        result = gr.HTML(render(None, "Standard", 18, 1.7, "Cream"))
                        speech = gr.Audio(label="Read aloud", type="filepath", interactive=False, autoplay=True,
                                          visible=False)
                        captions = gr.File(label="Captions file (.vtt)", interactive=False, visible=False)
                        with gr.Accordion("How I got this answer", open=False):
                            details = gr.HTML()
                with gr.Accordion("🪄 Make this easier: paste a passage (or leave empty to use the file above)",
                                  open=False):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=320):
                            passage = gr.Textbox(lines=8, label="Passage", placeholder="Paste instructions or a "
                                                                                       "paragraph you find hard")
                            easy_btn = gr.Button("Make this easier", variant="primary")
                            with gr.Row():
                                prev_btn = gr.Button("◀ Previous instruction", size="sm")
                                next_btn = gr.Button("Next instruction ▶", size="sm")
                        with gr.Column(scale=3, min_width=360):
                            easy_out = gr.HTML()

            with gr.Tab("2 · Do"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=320):
                        name_box = gr.Textbox(label="Assignment name", placeholder="e.g. Lab 4")
                        goal = gr.Textbox(label="Anything to add? (optional)",
                                          placeholder="e.g. I only have tonight")
                        visual = gr.Checkbox(value=True, label="Also connect photos and videos (slower)")
                        start_btn = gr.Button("🚀 Help me start", variant="primary", size="lg")
                        step = gr.Radio([], label="Suggested steps (pick one)", value=None)
                        with gr.Row():
                            done_btn = gr.Button("✅ Done", variant="primary")
                            small_btn = gr.Button("🔍 Make this smaller")
                            undo_btn = gr.Button("↩ Not done", size="sm")
                        focus = gr.Checkbox(value=False, label="Focus mode: show only the current step")
                        with gr.Row():
                            own = gr.Textbox(label="Add your own step", scale=3)
                            own_btn = gr.Button("Add", scale=1)
                        with gr.Accordion("⏱ Work timer (yours, not a deadline)", open=False):
                            minutes = gr.Radio([5, 10, 15, 25], value=15, label="Minutes")
                            with gr.Row():
                                t_start = gr.Button("Start")
                                t_stop = gr.Button("Stop", size="sm")
                            timer_box = gr.HTML()
                        with gr.Accordion("💻 Write code for a step (checked against your examples)", open=False):
                            code_prompt = gr.Textbox(lines=8, label="Function to write",
                                                     placeholder='def moving_average(values, window):\n    """...\n'
                                                                 '    >>> moving_average([1, 2, 3, 4], 2)\n'
                                                                 '    [1.5, 2.5, 3.5]\n    """')
                            code_btn = gr.Button("Write it")
                    with gr.Column(scale=3, min_width=360):
                        plan = gr.HTML(render_plan(None, False, "Standard", 18, 1.7, "Cream"))
                        code_out = gr.HTML()

            with gr.Tab("3 · Resume"):
                where_btn = gr.Button("🧭 Where was I?", variant="primary", size="lg")
                resume = gr.HTML()

        # --- wiring ---
        files.change(file_choices, files, which)
        inputs = [question, files, which, outputs, rate, *look, bionic, ws_name, session]
        results = [result, speech, captions, details, last, session]
        ask_btn.click(ask, inputs, results)
        question.submit(ask, inputs, results)
        catch_btn.click(catch_me_up, [when, *inputs[1:]], results)
        for btn, (_, text) in zip(quick, QUICK):
            btn.click(lambda t=text: t, None, question)

        easy_in = [easy_res, easy_idx, *look, bionic]
        easy_btn.click(make_easier, [passage, files, which, session], [easy_res, easy_idx, session]) \
            .then(render_easier, easy_in, easy_out)
        prev_btn.click(lambda r, i: step_instruction(r, i, -1), [easy_res, easy_idx], easy_idx) \
            .then(render_easier, easy_in, easy_out)
        next_btn.click(lambda r, i: step_instruction(r, i, 1), [easy_res, easy_idx], easy_idx) \
            .then(render_easier, easy_in, easy_out)

        plan_in = [ws_name, focus, *look, bionic]

        def refresh(chain):
            return (chain.then(render_plan, plan_in, plan).then(step_choices, ws_name, step)
                    .then(render_resume, [ws_name, *look, bionic], resume).then(timer_html, ws_name, timer_box))

        refresh(start_btn.click(start, [name_box, goal, files, visual, session], [ws_name, ws_pick, session]))
        refresh(ws_pick.input(lambda n: n, ws_pick, ws_name))
        refresh(small_btn.click(smaller, [ws_name, step], ws_name))
        refresh(done_btn.click(set_done, [ws_name, step], ws_name))
        refresh(undo_btn.click(lambda n, s: set_done(n, s, False), [ws_name, step], ws_name))
        refresh(own_btn.click(add_own, [ws_name, own], [ws_name, own]))
        step.input(focus_on, [ws_name, step], ws_name).then(render_plan, plan_in, plan)
        focus.change(render_plan, plan_in, plan)
        t_start.click(timer_start, [ws_name, minutes], timer_box)
        t_stop.click(timer_stop, ws_name, timer_box)
        gr.Timer(1.0).tick(timer_html, ws_name, timer_box, show_progress="hidden")
        code_btn.click(write_code, [code_prompt, *look], code_out)
        where_btn.click(render_resume, [ws_name, *look, bionic], resume)
        refresh(demo.load(lambda n: n, ws_pick, ws_name))

        for ctl in (font, size, spacing, bg, bionic):      # restyle without asking again
            ctl.change(render, [last, *look, bionic], result)
            ctl.change(render_plan, plan_in, plan)
            ctl.change(render_easier, easy_in, easy_out)
    return demo


def _warm():
    try:
        llm.warm(config.VISION_MODEL)
    except Exception:
        pass


def main():
    import sys
    lan = "--lan" in sys.argv
    threading.Thread(target=_warm, daemon=True).start()
    build().launch(server_name="0.0.0.0" if lan else "127.0.0.1", inbrowser=True, css=CSS,
                   theme=gr.themes.Soft(font=["system-ui", "Segoe UI", "sans-serif"],
                                        font_mono=["Consolas", "monospace"]))


if __name__ == "__main__":
    main()

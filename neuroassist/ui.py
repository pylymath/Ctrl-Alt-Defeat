"""NeuroAssist web UI (runs locally, nothing leaves the laptop).

    python -m neuroassist.ui          # opens http://127.0.0.1:7860
"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")   # fully offline: no usage pings

import html
import tempfile
import threading
import traceback
from pathlib import Path

import gradio as gr

from . import config, llm
from .pipeline import Session, answer

OUTPUTS = {"Summary card": "card", "Flashcards": "flashcards", "Captions (audio/video)": "captions",
           "Read aloud": "tts"}

QUICK = [
    ("Explain simply", "Summarize this simply."),
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
"""

_status_cache: dict = {}


def status_html() -> str:
    have = llm.installed_models()
    if not have:
        return ("<div class='na-status'>⚠️ Ollama is not running. Start the Ollama app, then reload this page."
                "</div>")
    need = dict.fromkeys([config.VISION_MODEL, config.TEXT_MODEL, config.CODE_MODEL, config.CARD_MODEL])

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


def render(res: dict | None, font: str, size: int, spacing: float, bg: str) -> str:
    page, ink, surface = BACKGROUNDS.get(bg, BACKGROUNDS["White"])
    style = (f"background:{page};color:{ink};font-family:{FONTS.get(font, FONTS['Standard'])};"
             f"font-size:{size}px;line-height:{spacing};")
    if font == "Dyslexia-friendly":
        style += "letter-spacing:.04em;word-spacing:.16em;"
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
        out.append("".join(f"<p class='na-answer'>{_esc(ln)}</p>" for ln in lines))

    card = res.get("card")
    if card:
        out.append(f"<div class='na-card' style='background:{surface}'><h3>📌 {_esc(card['title'])}</h3>"
                   f"<div>{_esc(card['one_line'])}</div>")
        if card.get("bullets"):
            out.append("<ul>" + "".join(f"<li>{_esc(b)}</li>" for b in card["bullets"]) + "</ul>")
        if card.get("key_terms"):
            out.append("<dl class='na-terms' style='margin-top:10px'>" + "".join(
                f"<dt>{_esc(t['term'])}</dt><dd>{_esc(t.get('meaning'))}</dd>" for t in card["key_terms"]) + "</dl>")
        out.append("</div>")

    if res.get("flashcards"):
        out.append("<div class='na-flash' style='margin-top:18px'><h3 style='margin:0'>🃏 Flashcards "
                   "<span style='font-weight:400;font-size:.8em'>(tap to flip)</span></h3>")
        out += [f"<details style='background:{surface}'><summary>{_esc(c['q'])}</summary>{_esc(c['a'])}</details>"
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
    out.append("</p><table class='na-steps'><tr><th>#</th><th>Step</th><th>Model</th><th>Time</th></tr>")
    for i, s in enumerate(res.get("steps", []), 1):
        mark = "" if s.get("ok") else " ❌"
        out.append(f"<tr><td>{i}</td><td>{_esc(s['step'])}{mark}</td><td>{_esc(s['model'])}</td>"
                   f"<td>{s['ms'] / 1000:.1f}s</td></tr>")
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


def ask(question, files, outputs, rate, font, size, spacing, bg, session):
    session = session or Session()
    question = (question or "").strip()
    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]
    if not question:
        question = "Summarize this simply." if paths else ""
    if not question:
        res = {"error": "Type a question first."}
        return (render(res, font, size, spacing, bg), gr.update(visible=False), gr.update(visible=False), "", res,
                session)
    try:
        a = answer(question, paths, session, force=[OUTPUTS[o] for o in outputs or []], tts_rate=int(rate))
        res = a.to_dict()
    except Exception as e:
        msg = str(e)
        if "connect" in msg.lower() or "refused" in msg.lower():
            msg = "Could not reach Ollama. Start the Ollama app and try again."
        res = {"error": msg, "trace": traceback.format_exc()}
    vtt = None
    if res.get("captions"):
        vtt = str(Path(tempfile.mkdtemp(prefix="neuroassist_vtt_")) / "captions.vtt")
        Path(vtt).write_text(res["captions"], encoding="utf-8")
    sp = res.get("speech_path")
    return (render(res, font, size, spacing, bg), gr.update(value=sp, visible=bool(sp)),
            gr.update(value=vtt, visible=bool(vtt)), details_html(res), res, session)


def build() -> gr.Blocks:
    with gr.Blocks(title="NeuroAssist") as demo:
        session = gr.State(None)
        last = gr.State(None)
        gr.Markdown("# 🧠 NeuroAssist\nAsk about a photo, video, recording, document or code. "
                    "Answers come only from your file, and say so when the file doesn't contain the answer.")
        status = gr.HTML(status_html)

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=320):
                files = gr.File(label="Your files (photo, video, audio, PDF, text, code)", file_count="multiple",
                                type="filepath")
                question = gr.Textbox(label="Your question", lines=3, autofocus=True,
                                      placeholder="e.g. How many people are in this photo?  ·  Quiz me on this chapter")
                with gr.Row():
                    quick = [gr.Button(label, size="sm") for label, _ in QUICK]
                outputs = gr.CheckboxGroup(list(OUTPUTS), label="Also give me", value=[])
                ask_btn = gr.Button("Ask", variant="primary", size="lg")
                with gr.Accordion("Reading settings", open=False):
                    font = gr.Radio(list(FONTS), value="Standard", label="Font")
                    size = gr.Slider(14, 30, value=18, step=1, label="Text size")
                    spacing = gr.Slider(1.3, 2.4, value=1.7, step=0.1, label="Line spacing")
                    bg = gr.Radio(list(BACKGROUNDS), value="Cream", label="Background")
                    rate = gr.Slider(-5, 5, value=0, step=1, label="Reading-aloud speed")

            with gr.Column(scale=3, min_width=360):
                result = gr.HTML(render(None, "Standard", 18, 1.7, "Cream"))
                speech = gr.Audio(label="Read aloud", type="filepath", interactive=False, autoplay=True, visible=False)
                captions = gr.File(label="Captions file (.vtt)", interactive=False, visible=False)
                with gr.Accordion("How I got this answer", open=False):
                    details = gr.HTML()

        inputs = [question, files, outputs, rate, font, size, spacing, bg, session]
        results = [result, speech, captions, details, last, session]
        ask_btn.click(ask, inputs, results)
        question.submit(ask, inputs, results)
        for btn, (_, text) in zip(quick, QUICK):
            btn.click(lambda t=text: t, None, question)
        for ctl in (font, size, spacing, bg):      # restyle the last answer without asking again
            ctl.change(render, [last, font, size, spacing, bg], result)
    return demo


def _warm():
    try:
        llm.warm(config.VISION_MODEL)
    except Exception:
        pass


def main():
    threading.Thread(target=_warm, daemon=True).start()
    build().launch(server_name="127.0.0.1", inbrowser=True, css=CSS, theme=gr.themes.Soft(font=["system-ui", "Segoe UI", "sans-serif"], font_mono=["Consolas", "monospace"]))


if __name__ == "__main__":
    main()

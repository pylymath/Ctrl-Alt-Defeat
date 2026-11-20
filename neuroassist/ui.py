"""Percie web UI (runs locally, nothing leaves the laptop).

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

# Four everyday actions up front; the rest one click away under "More".
QUICK = [
    ("Explain simply", "Summarize this simply."),
    ("Key points", "What are the key points?"),
    ("Quiz me", "Make flashcards to quiz me on this."),
    ("What do they mean?", "What does the teacher really mean? Explain anything vague literally."),
]
MORE_QUICK = [
    ("Explain and run the code", "Explain this code and run it."),
    ("Fix the code", "Fix this code."),
    ("Plain language", "Rewrite this in plain language."),
    ("Audio description", "Give an audio description of this for someone who is blind."),
    ("What does it say?", "What does the text say?"),
    ("Transcribe", "Transcribe this word for word."),
]

SANS = "'Segoe UI Variable Text', 'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', sans-serif"
SERIF = "'Iowan Old Style', 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif"
FONTS = {
    "Standard": SANS,
    "Dyslexia-friendly": "'OpenDyslexic', 'Lexend', Verdana, Tahoma, sans-serif",
    "Large print": "Verdana, Tahoma, sans-serif",
    "Serif": SERIF,
}
BACKGROUNDS = {  # reading surface: (page, text, inner surface). Low glare, never pure black on white.
    "Auto": ("var(--na-page)", "var(--na-ink)", "var(--na-surface)"),      # follows light / dark mode
    "Cream": ("#FBF8F1", "#2F2B26", "#F3EEE3"),
    "White": ("#FFFFFF", "#2B2926", "#F5F3EE"),
    "Pale blue": ("#F1F5F8", "#24303A", "#E4ECF2"),
    "Sage": ("#F1F5F1", "#28302B", "#E3ECE5"),
    "Dark": ("#22211F", "#ECE8E1", "#2D2B28"),
}

# Calm palette: warm paper, soft sage accent, clay for the few things that need attention.
INK, MUTED, LINE = "#2F2B26", "#7A7369", "#E7E2D8"
SAGE, SAGE_DEEP, SAGE_SOFT = "#5C7C6D", "#4E6B5E", "#E6EEE9"
CLAY_SOFT, CLAY_INK = "#F6E6DC", "#8A4B2F"

CSS = """
:root { --na-page: #FBF8F1; --na-surface: #F3EEE3; --na-ink: #2F2B26; --na-muted: #7A7369; --na-line: #E7E2D8; --na-sage: #5C7C6D; --na-sage-soft: #E6EEE9;
        --na-clay-soft: #F6E6DC; --na-clay-ink: #8A4B2F; --na-serif: 'Iowan Old Style', 'Palatino Linotype', Palatino,
        'Book Antiqua', Georgia, serif; }
.dark { --na-page: #262523; --na-surface: #302E2B; --na-ink: #ECE8E1; --na-muted: #A39E95; --na-line: #3A3835; --na-sage: #8FB3A2; --na-sage-soft: #2C3631;
        --na-clay-soft: #3B2C24; --na-clay-ink: #E7B79C; }
.na-main { max-width: 1100px; width: 100%; margin: 0 auto; }
footer { display: none !important; }

/* header */
.na-head { padding: 18px 4px 6px; }
.na-head h1 { font-family: var(--na-serif); font-weight: 700; font-size: 3rem; line-height: 1.05; letter-spacing: -.025em;
              margin: 0; background: linear-gradient(100deg, #2F2B26 10%, #4E6B5E 95%); -webkit-background-clip: text;
              background-clip: text; color: transparent; }
.dark .na-head h1 { background: linear-gradient(100deg, #F3EFE8 10%, #A9C9B9 95%); -webkit-background-clip: text;
                    background-clip: text; }
.na-head p { font-size: 1.05rem !important; }
.na-head p { margin: 4px 0 0; color: var(--na-muted); font-size: 1rem; }
.na-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.na-pill { font-size: .78rem; padding: 3px 10px; border-radius: 999px; background: var(--na-sage-soft); color: var(--na-ink); }
.na-pill.off { background: var(--na-clay-soft); color: var(--na-clay-ink); }

/* tabs: quiet underline, no boxes */
.tab-wrapper, .tab-container { border: none !important; }
button[role="tab"] { font-size: 1rem !important; color: var(--na-muted) !important; border: none !important;
                     background: transparent !important; padding: 10px 16px !important; }
button[role="tab"][aria-selected="true"] { color: var(--na-ink) !important; font-weight: 600 !important;
                     box-shadow: inset 0 -2px 0 var(--na-sage) !important; }

/* composer: one calm box for typing, the paperclip and drag-and-drop */
.na-composer { border-radius: 20px !important; box-shadow: 0 6px 24px rgba(60,50,40,.07) !important;
               border: 1px solid var(--na-line) !important; }
.na-composer textarea { font-size: 1.05rem !important; line-height: 1.5 !important; }
.na-composer button[aria-label*="ubmit"], .na-composer .submit-button { border-radius: 999px !important; }
.na-under { align-items: end !important; margin-top: 4px; }

/* chips */
.na-chips { gap: 8px !important; flex-wrap: wrap !important; }
.na-chips button { border-radius: 999px !important; font-weight: 500 !important; min-width: 0 !important;
                   padding: 6px 14px !important; flex: 0 1 auto !important; white-space: nowrap !important; }

/* reading surface */
.na-out { border-radius: 18px; padding: 24px 28px; max-width: 72ch; border: 1px solid var(--na-line);
          box-shadow: 0 1px 2px rgba(60,50,40,.04); }
.na-out *, .na-out h3 { color: inherit; font-family: inherit; }
.na-out h3 { font-weight: 600; font-size: 1.02em; margin: 0 0 8px; }
.na-out code { background: #8881; padding: 0 4px; border-radius: 4px; }
.na-empty { color: var(--na-muted); }
.na-answer { font-weight: 500; margin: 0 0 8px; }
.na-badge { display: inline-block; font-size: .78em; padding: 3px 12px; border-radius: 999px; margin-bottom: 12px;
            font-weight: 500; }
.na-ok { background: #E3EEE7; color: #2F5A45; } .na-warn { background: #F6E6DC; color: #8A4B2F; }
.na-card { border-radius: 14px; padding: 16px 20px; margin-top: 18px; }
.na-card ul { margin: 8px 0 0 1.1em; padding: 0; } .na-card li { margin-bottom: .5em; }
.na-terms dt { font-weight: 600; } .na-terms dd { margin: 0 0 .6em 0; }
.na-flash details { border-radius: 12px; padding: 12px 16px; margin: 8px 0; cursor: pointer; }
.na-flash summary { font-weight: 500; } .na-flash details[open] summary { margin-bottom: 6px; }
.na-code { white-space: pre-wrap; font-family: Consolas, 'Cascadia Mono', monospace; font-size: .88em; padding: 14px;
           border-radius: 12px; overflow-x: auto; }
.na-steps { width: 100%; border-collapse: collapse; font-size: .9em; }
.na-steps td, .na-steps th { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--na-line); }
.na-quote { border-left: 3px solid #A9C2B5; padding: 4px 12px; margin: 6px 0; color: inherit; opacity: .9; }
.na-status { font-size: .88em; opacity: .75; }
.na-plain { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
.na-plain > div { flex: 1 1 260px; border-radius: 12px; padding: 12px 16px; }
.na-orig { opacity: .72; font-size: .92em; }
.na-tag { font-size: .72em; padding: 2px 9px; border-radius: 999px; background: #8881; margin-left: 6px;
          font-weight: 500; white-space: nowrap; }
.na-step { display: flex; gap: 12px; align-items: baseline; padding: 9px 0; border-bottom: 1px solid #8881; }
.na-step .na-dot { width: 1.2em; text-align: center; opacity: .8; }
.na-step.done { opacity: .5; text-decoration: line-through; }
.na-step.now { font-weight: 600; }
.na-focus { font-family: var(--na-serif); font-size: 1.55em; font-weight: 500; line-height: 1.35; margin: 10px 0 8px; }
.na-bar { height: 6px; border-radius: 999px; background: #8882; overflow: hidden; margin: 10px 0 6px; }
.na-bar > div { height: 100%; background: linear-gradient(90deg, #8FB3A2, #5C7C6D); border-radius: 999px; }
.na-goal { font-family: var(--na-serif); font-size: 1.3em; font-weight: 500; margin: 0 0 4px; }
.na-req { margin: 0 0 16px; }
.na-loc { font-size: .74em; padding: 2px 8px; border-radius: 6px; background: #5C7C6D1f; margin-left: 6px;
          white-space: nowrap; font-weight: 500; }
.na-link { font-size: .88em; margin: 6px 0 0 1em; opacity: .9; }
.na-conflict { font-size: .9em; margin: 8px 0 0 1em; padding: 8px 12px; border-radius: 10px; background: #F6E6DC;
               color: #7A4128; }
.na-suggest { font-size: .82em; opacity: .7; margin: -2px 0 10px; }
.na-section { font-size: .78em; text-transform: uppercase; letter-spacing: .08em; opacity: .6; margin: 24px 0 8px; }
mark.na-hl { background: #F3E2B8; color: #2F2B26; border-radius: 4px; padding: 0 3px; }
.na-timer { font-family: var(--na-serif); font-size: 2.2em; font-variant-numeric: tabular-nums; }
.na-instr { font-family: var(--na-serif); font-size: 1.35em; line-height: 1.4; margin: 8px 0; }

/* code with line numbers; the line an error came from is marked */
.na-numbered { white-space: normal; padding: 10px 0; }
.na-cl { white-space: pre; padding: 0 14px 0 0; }
.na-cl span { display: inline-block; width: 2.6em; text-align: right; padding-right: 12px; opacity: .45;
              user-select: none; }
.na-cl.na-bad { background: #F6E6DC; color: #7A4128; }

/* bionic reading: normal weight everywhere, so the bold word-starts actually stand out */
.na-bio, .na-bio * { font-weight: 400 !important; }
.na-bio b { font-weight: 700 !important; }
/* ...and on the rest of the page (bionic.js wraps labels, buttons, tabs, chips, placeholders in .pb) */
.percie-bionic .pb, .percie-bionic .pb-ph { font-weight: 400 !important; }
.percie-bionic .pb b, .percie-bionic .pb-ph b { font-weight: 700 !important; }
"""


def theme():
    """Calm, low-glare theme. System fonts only (works offline)."""
    sage = gr.themes.Color(c50="#F1F6F3", c100="#E6EEE9", c200="#CFDDD5", c300="#AFC6BA", c400="#8FB3A2",
                           c500="#5C7C6D", c600="#4E6B5E", c700="#41594E", c800="#34473E", c900="#28372F",
                           c950="#1C2621")
    t = gr.themes.Base(primary_hue=sage, secondary_hue=sage, neutral_hue=gr.themes.colors.stone,
                       radius_size=gr.themes.sizes.radius_lg, spacing_size=gr.themes.sizes.spacing_lg,
                       text_size=gr.themes.sizes.text_md, font=[SANS], font_mono=["Consolas", "monospace"])
    return t.set(
        body_background_fill=("radial-gradient(1100px 520px at 0% 0%, #FDFBF6 0%, rgba(253,251,246,0) 60%), "
                              "radial-gradient(900px 600px at 100% 100%, #EAF0EB 0%, rgba(234,240,235,0) 60%), "
                              "#F6F4EE"),
        body_background_fill_dark=("radial-gradient(1100px 520px at 0% 0%, #2A2926 0%, rgba(42,41,38,0) 60%), "
                                   "radial-gradient(900px 600px at 100% 100%, #222A26 0%, rgba(34,42,38,0) 60%), "
                                   "#1C1B19"),
        body_text_color=INK, body_text_color_dark="#ECE8E1",
        body_text_color_subdued=MUTED, body_text_color_subdued_dark="#A39E95",
        block_background_fill="rgba(255,255,255,0.66)", block_background_fill_dark="rgba(45,43,40,0.66)",
        block_border_color=LINE, block_border_color_dark="#3A3835", block_border_width="1px",
        block_shadow="0 1px 2px rgba(60,50,40,0.04)", block_shadow_dark="none",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_text_color=MUTED, block_label_text_color_dark="#A39E95", block_label_border_width="0px",
        block_title_text_color=MUTED, block_title_text_weight="500",
        panel_background_fill="transparent", panel_background_fill_dark="transparent",
        panel_border_width="0px",
        border_color_primary=LINE, border_color_primary_dark="#3A3835",
        input_background_fill="#FFFFFF", input_background_fill_dark="#262522",
        input_border_color=LINE, input_border_color_dark="#3A3835",
        input_border_color_focus="#9DB5A8", input_border_color_focus_dark="#6F8F80",
        input_shadow_focus="0 0 0 3px rgba(92,124,109,0.14)",
        button_primary_background_fill=SAGE, button_primary_background_fill_hover=SAGE_DEEP,
        button_primary_background_fill_dark="#8FB3A2", button_primary_background_fill_hover_dark="#A5C3B4",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#1C1B19",
        button_primary_border_color="transparent", button_primary_border_color_dark="transparent",
        button_secondary_background_fill="#F1EEE7", button_secondary_background_fill_hover="#E8E3D9",
        button_secondary_background_fill_dark="#2D2B28", button_secondary_background_fill_hover_dark="#36332F",
        button_secondary_text_color="#3D3832", button_secondary_text_color_dark="#E3DED6",
        button_secondary_border_color="#E2DDD2", button_secondary_border_color_dark="#3A3835",
        button_large_text_weight="500",
        color_accent=SAGE, color_accent_soft=SAGE_SOFT, color_accent_soft_dark="#2C3631",
        checkbox_background_color_selected=SAGE, checkbox_background_color_selected_dark="#8FB3A2",
        checkbox_border_color_selected=SAGE, slider_color=SAGE, slider_color_dark="#8FB3A2",
        loader_color=SAGE, shadow_drop="none", shadow_drop_lg="0 10px 30px rgba(60,50,40,0.07)",
    )


_status_cache: dict = {}


def status_html() -> str:
    have = llm.installed_models()
    if not have:
        return ("<div class='na-pills'><span class='na-pill off'>Ollama isn't running. Open the Ollama app, then "
                "reload this page.</span></div>")
    need = dict.fromkeys([config.VISION_MODEL, config.QA_MODEL, config.CODE_MODEL])

    def present(m):
        return m in have or f"{m}:latest" in have or m.removesuffix(":latest") in have
    pills = [f"<span class='na-pill{'' if present(m) else ' off'}'>{'' if present(m) else 'missing: '}"
             f"{html.escape(m.removesuffix(':latest'))}</span>" for m in need]
    pills.append(f"<span class='na-pill'>whisper-{html.escape(str(config.WHISPER_MODEL))}</span>")
    pills.append("<span class='na-pill'>offline · on this laptop</span>")
    return "<div class='na-pills'>" + "".join(pills) + "</div>"


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


def _bionic_view(fn):
    """Renderers whose last argument is `bionic`: tag the reading surface so CSS drops the base weight to
    normal and only the bolded word-starts stand out (bold-on-semibold was invisible)."""
    import functools
    import inspect
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        out = fn(*args, **kwargs)
        if sig.bind(*args, **kwargs).arguments.get("bionic"):
            out = out.replace("class='na-out'", "class='na-out na-bio'")
        return out
    return wrap


def _style(font: str, size: int, spacing: float, bg: str) -> tuple[str, str]:
    page, ink, surface = BACKGROUNDS.get(bg, BACKGROUNDS["Auto"])
    style = (f"background:{page};color:{ink};font-family:{FONTS.get(font, FONTS['Standard'])};"
             f"font-size:{size}px;line-height:{spacing};")
    if font == "Dyslexia-friendly":
        style += "letter-spacing:.04em;word-spacing:.16em;"
    return style, surface


def _code_lines(code: str, surface: str, bad: int | None = None) -> str:
    rows = "".join(f"<div class='na-cl{' na-bad' if i == bad else ''}'><span>{i}</span>{_esc(l) or ' '}</div>"
                   for i, l in enumerate(code.splitlines(), 1))
    return f"<div class='na-code na-numbered' style='background:{surface}'>{rows}</div>"


def _run_block(r: dict, T, surface: str, label: str) -> str:
    """What really happened when the code ran: output, error, and each docstring example."""
    out = []
    ex = r.get("examples") or []
    n_ok = sum(e["ok"] for e in ex)
    if r.get("error"):
        out.append(f"<span class='na-badge na-warn'>{label}: error"
                   f"{' on line ' + str(r['error_line']) if r.get('error_line') else ''}</span> ")
    else:
        out.append(f"<span class='na-badge na-ok'>{label}: finished in {r.get('ms', 0) / 1000:.2f} s</span> ")
    if ex:
        out.append(f"<span class='na-badge {'na-ok' if n_ok == len(ex) else 'na-warn'}'>"
                   f"Examples: {n_ok} of {len(ex)} pass</span>")
    out.append("<div class='na-section' style='margin-top:10px'>Output</div>"
               f"<div class='na-code' style='background:{surface}'>{_esc(r.get('stdout') or '(nothing printed)')}"
               + (f"\n{_esc(r['error'])}" if r.get("error") else "") + "</div>")
    if ex:
        out.append("<div class='na-section'>Examples from the docstring</div><table class='na-steps'>"
                   "<tr><th></th><th>Call</th><th>Expected</th><th>Got</th></tr>" + "".join(
                       f"<tr><td>{'✓' if e['ok'] else '✗'}</td><td><code>{_esc(e['expr'])}</code></td>"
                       f"<td><code>{_esc(e['expected'])}</code></td><td><code>{_esc(e['actual'])}</code></td></tr>"
                       for e in ex) + "</table>")
    return "".join(out)


def _code_report(ev: dict, T, surface: str) -> str:
    r = ev.get("run") or {}
    out = [f"<div class='na-status' style='margin-top:6px'>Code from {_esc(ev.get('where', 'your file'))}</div>"]
    if ev.get("fixed") is not None:
        fr = ev.get("fixed_run") or {}
        out.append("<div class='na-section'>Before</div>" + _run_block(r, T, surface, "Original"))
        out.append("<div class='na-section'>Corrected code</div>" + _code_lines(ev["fixed"], surface, fr.get("error_line")))
        out.append(_run_block(fr, T, surface, "Corrected version"))
        return "".join(out)
    out.append(_code_lines(ev["code"], surface, r.get("error_line")))
    out.append(_run_block(r, T, surface, "Ran it on this laptop"))
    x = ev.get("explanation") or {}
    if x.get("steps"):
        out.append("<div class='na-section'>Step by step</div>" + "".join(
            f"<div class='na-step'><span class='na-loc' style='margin:0'>line {_esc(s['lines'])}</span>"
            f"<span>{T(s['what'])}</span></div>" for s in x["steps"]))
    if x.get("issues"):
        out.append(f"<div class='na-card' style='background:{surface}'><h3>What the run shows is wrong</h3><ul>"
                   + "".join(f"<li>{T(i)}</li>" for i in x["issues"]) + "</ul>"
                   "<div class='na-status'>Ask <b>Fix this code</b> to get a corrected version, checked against the "
                   "same examples.</div></div>")
    return "".join(out)


@_bionic_view
def render(res: dict | None, font: str, size: int, spacing: float, bg: str, bionic: bool = False) -> str:
    T = _bionic if bionic else _esc
    style, surface = _style(font, size, spacing, bg)
    if not res:
        # Run through T too, so every reading setting (bionic included) visibly applies before the first answer.
        return (f"<div class='na-out' style='{style}'><span class='na-empty'>"
                + T("Add a file, or don't. Ask a question, and your answer appears here, with the sentence it "
                    "came from.") + "</span></div>")
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
    if route.get("task") == "explain_code" and (res.get("evidence") or {}).get("code"):
        out.append(_code_report(res["evidence"], T, surface))

    cu = (res.get("evidence") or {}).get("catchup")
    if cu and cu.get("instructions"):
        out.append(f"<div class='na-card' style='background:{surface}'><h3>Instructions said in this part "
                   "<span style='font-weight:400;font-size:.8em'>(word for word)</span></h3><ul>"
                   + "".join(f"<li><span class='na-loc'>{_esc(i['loc'])}</span> {T(i['text'])}</li>"
                             for i in cu["instructions"]) + "</ul></div>")

    card = res.get("card")
    if card:
        out.append(f"<div class='na-card' style='background:{surface}'><h3>{T(card['title'])}</h3>"
                   f"<div>{T(card['one_line'])}</div>")
        if card.get("bullets"):
            out.append("<ul>" + "".join(f"<li>{T(b)}</li>" for b in card["bullets"]) + "</ul>")
        if card.get("key_terms"):
            out.append("<dl class='na-terms' style='margin-top:10px'>" + "".join(
                f"<dt>{_esc(t['term'])}</dt><dd>{T(t.get('meaning'))}</dd>" for t in card["key_terms"]) + "</dl>")
        out.append("</div>")

    if res.get("plain"):
        out.append("<h3 style='margin:18px 0 0'>Plain language</h3>")
        for p in res["plain"]:
            tag = "" if p["checked"] else "<span class='na-tag'>kept original: rewrite changed a number</span>"
            out.append(f"<div class='na-plain'><div class='na-orig' style='background:{surface}'>{_esc(p['original'])}"
                       f"</div><div style='background:{surface}'>{T(p['plain'])}{tag}</div></div>")
    if res.get("literal"):
        out.append(f"<div class='na-card' style='background:{surface}'><h3>What it really means</h3><dl class='na-terms'>"
                   + "".join(f"<dt>“{_esc(i['phrase'])}”<span class='na-tag'>{_esc(i.get('label', i['kind']))}</span></dt>"
                             f"<dd>{T(i['meaning'])}</dd>" for i in res["literal"]) + "</dl></div>")
    elif res.get("literal") == []:
        out.append("<p style='margin-top:14px'>Nothing vague or figurative found.</p>")

    if res.get("flashcards"):
        out.append("<div class='na-flash' style='margin-top:18px'><h3 style='margin:0'>Flashcards "
                   "<span style='font-weight:400;font-size:.8em'>(tap to flip)</span></h3>")
        out += [f"<details style='background:{surface}'><summary>{T(c['q'])}</summary>{T(c['a'])}</details>"
                for c in res["flashcards"]]
        out.append("</div>")
    elif res.get("flashcards") == []:
        out.append("<p style='margin-top:14px'>No flashcards: none could be checked against the source.</p>")

    if res.get("captions"):
        cues = [c for c in res["captions"].split("\n\n")[1:] if c.strip()]
        out.append(f"<div class='na-card' style='background:{surface}'><h3>Captions</h3>" + "".join(
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
        mark = "" if s.get("ok") else " (failed)"
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
    def one(f):
        if isinstance(f, str):
            return f
        if isinstance(f, dict):
            return f.get("path") or f.get("name") or ""
        return getattr(f, "path", None) or getattr(f, "name", str(f))
    return [p for p in (one(f) for f in (files or [])) if p]


ALL, NONE = "All files together", "No file (just ask)"


def file_choices(files):
    names = [Path(p).name for p in _paths(files)]
    choices = names + ([ALL] if len(names) > 1 else []) + [NONE]
    return gr.update(choices=choices, value=names[0] if names else NONE)


def sync_files(box, current):
    """Keep the attached files (from the composer) in state; refresh 'Ask about' only when they change,
    so typing never resets the user's choice."""
    new = _paths((box or {}).get("files"))
    if new == (current or []):
        return gr.skip(), gr.skip()
    return new, file_choices(new)


def submit(box, files, which, outputs, rate, font, size, spacing, bg, bionic, ws_name, session):
    """Composer submit: answer, clear the typed text (keep the files), and show the Understand tab."""
    question = ((box or {}).get("text") or "").strip()
    out = ask(question, files, which, outputs, rate, font, size, spacing, bg, bionic, ws_name, session)
    kept = {"text": "", "files": (box or {}).get("files", [])}
    return (*out, kept, gr.Tabs(selected="understand"))


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


@_bionic_view
def render_easier(res, idx, font, size, spacing, bg, bionic=False) -> str:
    T = _bionic if bionic else _esc
    style, surface = _style(font, size, spacing, bg)
    if not res:
        return ""
    out = [f"<div class='na-out' style='{style}'>"]
    if res.get("note"):
        out.append(f"<p class='na-status'>{_esc(res['note'])}</p>")
    if res.get("gist"):
        out.append(f"<p class='na-goal'>{T(res['gist'])}</p>")
    if res.get("details"):
        out.append("<details><summary>More detail</summary><ul>" + "".join(f"<li>{T(d)}</li>" for d in res["details"])
                   + "</ul></details>")
    ins = res.get("instructions") or []
    if ins:
        i = max(0, min(int(idx or 0), len(ins) - 1))
        out.append(f"<div class='na-card' style='background:{surface}'><div class='na-status'>Instruction {i + 1} of "
                   f"{len(ins)} (word for word from the text)</div><div class='na-instr'>{T(ins[i])}</div></div>")
    if res.get("unclear"):
        out.append(f"<div class='na-card' style='background:{surface}'><h3>Unclear phrases "
                   "<span style='font-weight:400;font-size:.8em'>(highlighted below)</span></h3><dl class='na-terms'>"
                   + "".join(f"<dt>“{_esc(u['phrase'])}”<span class='na-tag'>{_esc(u['label'])}</span></dt>"
                             f"<dd>{T(u['meaning'])}</dd>" for u in res["unclear"]) + "</dl></div>")
    out.append(f"<h3 style='margin-top:18px'>Original, with unclear phrases marked</h3>"
               f"<div class='na-card' style='background:{surface};margin-top:6px'>"
               f"{_highlight(res.get('original', ''), res.get('unclear') or [])}</div>")
    if res.get("plain"):
        out.append("<h3 style='margin-top:18px'>Plain language, next to the original</h3>")
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
        icon = "◉" if l.get("visual") else "↗"
        out.append(f"<div class='na-link'>{icon} Also in <b>{_esc(l['file'])}</b> <span class='na-loc'>{_esc(l['loc'])}"
                   f"</span>: “{_esc(l['text'][:180])}{'…' if len(l['text']) > 180 else ''}” "
                   f"<span class='na-status'>({_esc(l['why'])})</span></div>")
    for c in r.get("conflicts") or []:
        out.append(f"<div class='na-conflict'>Check both, they differ: {_esc(c)}. "
                   "Percie does not decide which one is right; ask your teacher.</div>")
    return "".join(out) + "</li>"


@_bionic_view
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
    out = [f"<div class='na-out' style='{style}'><p class='na-goal'>{T(ws.goal)}</p>",
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
            out.append("<p class='na-focus'>All steps done. Nice work.</p>")
        return "".join(out) + "</div>"

    out.append("<div class='na-section'>Suggested steps</div><div class='na-suggest'>Suggested by Percie, not stated in your files. "
               "Add your own or make any of them smaller.</div>")
    if not vis:
        out.append("<p>No steps yet.</p>")
    for i, s in enumerate(vis, 1):
        cls = "done" if s["done"] else ("now" if nxt and s["id"] == nxt["id"] else "")
        mark = "✓" if s["done"] else ("→" if cls == "now" else "○")
        sub = "↳ " if s.get("parent") else ""
        out.append(f"<div class='na-step {cls}'><span class='na-dot'>{mark}</span><span>{i}. {sub}{T(s['text'])}</span></div>")

    out.append("<div class='na-section'>What your files say</div>")
    if ws.requirements:
        for t in wsp.TYPES:
            group = [r for r in ws.requirements if r["type"] == t]
            if group:
                out.append(f"<p style='margin:14px 0 4px'><b>{wsp.TYPE_LABEL[t]}</b></p><ul>"
                           + "".join(_req_html(r, T, surface) for r in group) + "</ul>")
        if not any(r["type"] == "deadline" for r in ws.requirements):
            out.append("<p class='na-status'>No deadline was found in your files. Ask your teacher.</p>")
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
    choices = [(f"{i}. {'✓ ' if s['done'] else ''}{s['text'][:70]}", s["id"]) for i, s in enumerate(vis, 1)]
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
        return "<div class='na-status'>No timer running. Pick a length and press Start. It's yours, not a deadline.</div>"
    if left == 0:
        return "<div class='na-timer'>00:00</div><div class='na-status'>Time's up. Take a short break, or press Done.</div>"
    return f"<div class='na-timer'>{_mmss(left)}</div><div class='na-status'>of your {ws.timer_minutes}-minute timer</div>"


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
@_bionic_view
def render_resume(ws_name, font, size, spacing, bg, bionic=False) -> str:
    T = _bionic if bionic else _esc
    style, surface = _style(font, size, spacing, bg)
    ws = _load_ws(ws_name)
    if not ws:
        return f"<div class='na-out' style='{style}'>No saved assignment yet. Start one in the <b>Do</b> tab.</div>"
    w = wsp.where_was_i(ws)
    out = [f"<div class='na-out' style='{style}'><span class='na-badge na-ok'>Read from your saved progress, "
           "nothing guessed</span>",
           f"<p class='na-status'>You're working on</p><p class='na-goal'>{T(w['goal'])}</p>",
           f"<div class='na-bar'><div style='width:{100 * w['done'] / max(w['total'], 1):.0f}%'></div></div>",
           f"<p>{w['done']} of {w['total']} steps done · last active {_esc(w['last_active'])}</p>"]
    if w["last_done"]:
        out.append(f"<p>Last thing you finished ({_esc(w['last_done_ago'])}): {T(w['last_done'])}</p>")
    if w["next"]:
        out.append(f"<div class='na-card' style='background:{surface}'><div class='na-status'>Your next step "
                   "(suggested):</div>"
                   f"<div class='na-focus'>{T(w['next']['text'])}</div>")
        if w["next_source"]:
            s = w["next_source"]
            out.append(f"<div class='na-status'>It's for <span class='na-loc'>{_esc(s['source'])} · {_esc(s['loc'])}"
                       f"</span></div><div class='na-quote'>{_esc(s['quote'])}</div>")
        out.append("</div>")
    else:
        out.append("<p class='na-focus'>Every step is done.</p>")
    if w["last_material"]:
        out.append(f"<p>Last file you looked at: <b>{_esc(w['last_material'])}</b></p>")
    if w["last_question"]:
        q = w["last_question"]
        out.append(f"<p>Last thing you asked: “{_esc(q['question'])}”<br>"
                   f"<span class='na-status'>Answer: {_esc(q['answer'])}</span></p>")
    if w["deadlines"]:
        out.append("<div class='na-section'>Deadlines in your files</div><ul>" + "".join(
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


# --- Layout ---------------------------------------------------------------------------------------
def build() -> gr.Blocks:
    look_default = ("Standard", 18, 1.7, "Auto")
    with gr.Blocks(title="Percie") as demo:
        session = gr.State(None)
        last = gr.State(None)
        ws_name = gr.State(None)
        easy_res = gr.State(None)
        easy_idx = gr.State(0)
        saved = wsp.list_workspaces()

        # --- Sidebar: everything about HOW things look and which assignment is open --------------------
        with gr.Sidebar(open=True, width=300):
            gr.HTML("<div class='na-section' style='margin-top:4px'>Assignment</div>")
            ws_pick = gr.Dropdown(saved, label="Open a saved assignment", value=saved[0] if saved else None)
            gr.HTML("<div class='na-section'>Reading</div>")
            font = gr.Radio(list(FONTS), value="Standard", label="Font")
            size = gr.Slider(14, 30, value=18, step=1, label="Text size")
            spacing = gr.Slider(1.3, 2.4, value=1.7, step=0.1, label="Line spacing")
            bg = gr.Radio(list(BACKGROUNDS), value="Auto", label="Background")
            bionic = gr.Checkbox(value=False, label="Bionic reading (bold word starts)")
            rate = gr.Slider(-5, 5, value=0, step=1, label="Read-aloud speed")
            gr.HTML("<div class='na-section'>Running locally</div>")
            gr.HTML(status_html)
        look = [font, size, spacing, bg]

        with gr.Column(elem_classes="na-main"):
            # --- Header + materials ------------------------------------------------------------------------
            gr.HTML("<div class='na-head'><h1>Percie</h1><p>Understand your materials, get started, "
                    "and find your place again. Nothing leaves this laptop.</p></div>")
            # --- Composer: type, click the paperclip, or drop files onto it. Shared by all three tabs. --------
            box = gr.MultimodalTextbox(file_count="multiple", show_label=False, lines=2, max_lines=8, autofocus=True,
                                       submit_btn="Ask", elem_classes="na-composer",
                                       placeholder="Ask anything, or drop your files here: assignment, slides, "
                                                   "recording, photo, video or code")
            files = gr.State([])
            with gr.Row(equal_height=True, elem_classes="na-under"):
                which = gr.Dropdown([NONE], value=NONE, label="Ask about", scale=1, min_width=220)
                with gr.Column(scale=3):
                    with gr.Row(elem_classes="na-chips"):
                        quick = [gr.Button(label, size="sm", variant="secondary") for label, _ in QUICK]

            with gr.Tabs(selected="understand") as tabs:
                # ---- Understand ----
                with gr.Tab("Understand", id="understand"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=3, min_width=360):
                            result = gr.HTML(render(None, *look_default))
                            speech = gr.Audio(label="Read aloud", type="filepath", interactive=False, autoplay=True,
                                              visible=False)
                            captions = gr.File(label="Captions (.vtt)", interactive=False, visible=False)
                            with gr.Accordion("How I got this answer", open=False):
                                details = gr.HTML()
                        with gr.Column(scale=2, min_width=300):
                            with gr.Accordion("I lost the thread (catch me up)", open=False):
                                when = gr.Textbox(label="From when in the recording?",
                                                  placeholder="12:40   ·   2:00 to 3:30   ·   empty = last 5 minutes")
                                catch_btn = gr.Button("Catch me up", variant="secondary")
                            with gr.Accordion("More", open=False):
                                with gr.Row(elem_classes="na-chips"):
                                    more_quick = [gr.Button(label, size="sm", variant="secondary")
                                                  for label, _ in MORE_QUICK]
                                outputs = gr.CheckboxGroup(list(OUTPUTS), label="Also give me", value=[])
                    with gr.Accordion("Make this easier: paste a passage, or leave it empty to use the attached file",
                                      open=False):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=2, min_width=320):
                                passage = gr.Textbox(lines=7, show_label=False,
                                                     placeholder="Paste instructions or a paragraph you find hard")
                                easy_btn = gr.Button("Make this easier", variant="primary")
                                with gr.Row():
                                    prev_btn = gr.Button("Previous instruction", size="sm", variant="secondary")
                                    next_btn = gr.Button("Next instruction", size="sm", variant="secondary")
                            with gr.Column(scale=3, min_width=360):
                                easy_out = gr.HTML()

                # ---- Do ----
                with gr.Tab("Do", id="do"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=2, min_width=320):
                            name_box = gr.Textbox(label="Assignment name", placeholder="e.g. Lab 4")
                            goal = gr.Textbox(label="Anything to add? (optional)", placeholder="e.g. I only have tonight")
                            visual = gr.Checkbox(value=True, label="Also connect photos and videos (slower)")
                            start_btn = gr.Button("Help me start", variant="primary", size="lg")
                            step = gr.Radio([], label="Your steps", value=None)
                            with gr.Row():
                                done_btn = gr.Button("Done", variant="primary")
                                small_btn = gr.Button("Make this smaller", variant="secondary")
                            focus = gr.Checkbox(value=False, label="Focus mode: show only this step")
                            with gr.Accordion("More", open=False):
                                undo_btn = gr.Button("Mark as not done", size="sm", variant="secondary")
                                with gr.Row():
                                    own = gr.Textbox(show_label=False, placeholder="Add your own step", scale=3)
                                    own_btn = gr.Button("Add", scale=1, variant="secondary")
                            with gr.Accordion("Timer (yours, not a deadline)", open=False):
                                minutes = gr.Radio([5, 10, 15, 25], value=15, label="Minutes")
                                with gr.Row():
                                    t_start = gr.Button("Start", variant="secondary")
                                    t_stop = gr.Button("Stop", size="sm", variant="secondary")
                                timer_box = gr.HTML()
                            with gr.Accordion("Write code for a step", open=False):
                                code_prompt = gr.Textbox(lines=8, show_label=False,
                                                         placeholder='def moving_average(values, window):\n    """...\n'
                                                                     '    >>> moving_average([1, 2, 3, 4], 2)\n'
                                                                     '    [1.5, 2.5, 3.5]\n    """')
                                code_btn = gr.Button("Write it and check the examples", variant="secondary")
                        with gr.Column(scale=3, min_width=360):
                            plan = gr.HTML(render_plan(None, False, *look_default))
                            code_out = gr.HTML()

                # ---- Resume ----
                with gr.Tab("Resume", id="resume"):
                    with gr.Row():
                        with gr.Column(scale=1, min_width=240):
                            where_btn = gr.Button("Where was I?", variant="primary", size="lg")
                        with gr.Column(scale=3):
                            resume = gr.HTML()

        # --- wiring --------------------------------------------------------------------------------------
        # Files attached in the composer (paperclip or drag-and-drop) are kept until removed with their x,
        # so follow-up questions and Help me start use them too.
        box.change(sync_files, [box, files], [files, which], show_progress="hidden")
        inputs = [box, files, which, outputs, rate, *look, bionic, ws_name, session]
        results = [result, speech, captions, details, last, session, box, tabs]
        # After an answer arrives, draw it again with the reading settings as they are NOW (they may have
        # been changed while the model was still working).
        redraw = dict(fn=render, inputs=[last, *look, bionic], outputs=result, show_progress="hidden")
        box.submit(submit, inputs, results).then(**redraw)
        catch_btn.click(catch_me_up, [when, *inputs[1:]], results[:6]).then(**redraw)
        for btn, (_, text) in list(zip(quick, QUICK)) + list(zip(more_quick, MORE_QUICK)):
            btn.click(lambda v, t=text: {"text": t, "files": (v or {}).get("files", [])}, box, box)

        easy_in = [easy_res, easy_idx, *look, bionic]
        easy_btn.click(make_easier, [passage, files, which, session], [easy_res, easy_idx, session]) \
            .then(render_easier, easy_in, easy_out)
        prev_btn.click(lambda r, i: step_instruction(r, i, -1), [easy_res, easy_idx], easy_idx) \
            .then(render_easier, easy_in, easy_out)
        next_btn.click(lambda r, i: step_instruction(r, i, 1), [easy_res, easy_idx], easy_idx) \
            .then(render_easier, easy_in, easy_out)

        plan_in = [ws_name, focus, *look, bionic]
        resume_in = [ws_name, *look, bionic]

        def refresh(chain):
            return (chain.then(render_plan, plan_in, plan).then(step_choices, ws_name, step)
                    .then(render_resume, resume_in, resume).then(timer_html, ws_name, timer_box))

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
        where_btn.click(render_resume, resume_in, resume)
        refresh(demo.load(lambda n: n, ws_pick, ws_name))

        # Every reading setting re-draws every view at once (answer, easier, plan, resume)
        for ctl in (font, size, spacing, bg, bionic):
            ctl.change(render, [last, *look, bionic], result, show_progress="hidden")
            ctl.change(render_easier, easy_in, easy_out, show_progress="hidden")
            ctl.change(render_plan, plan_in, plan, show_progress="hidden")
            ctl.change(render_resume, resume_in, resume, show_progress="hidden")
        # ...and the rest of the page (labels, buttons, tabs, chips, placeholders), in the browser
        bionic.change(None, bionic, None, js="(v) => { if (window.percieBionic) window.percieBionic(v); }")
    return demo


HEAD = "<script>" + (Path(__file__).with_name("bionic.js")).read_text(encoding="utf-8") + "</script>"


def launch(demo: gr.Blocks, **kw):
    """One place for the look: theme, CSS and the page-wide bionic script."""
    return demo.launch(css=CSS, theme=theme(), head=HEAD, **kw)


def _warm():
    try:
        llm.warm(config.VISION_MODEL)
    except Exception:
        pass


def main():
    import sys
    lan = "--lan" in sys.argv
    threading.Thread(target=_warm, daemon=True).start()
    launch(build(), server_name="0.0.0.0" if lan else "127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    main()

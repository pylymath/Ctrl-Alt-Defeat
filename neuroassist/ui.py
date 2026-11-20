"""Percie web UI (runs locally, nothing leaves the laptop).

    python -m neuroassist.ui          # opens http://127.0.0.1:7860
    python -m neuroassist.ui --lan    # also reachable from a phone on the same Wi-Fi (inference stays here)

Three tabs around one assignment:
    1 Understand   ask about any file; catch me up from a timestamp; make a passage easier
    2 Do           help me start (requirements with sources, connected across files), suggested steps,
                   break down steps, one step at a time, a work timer, code checked against its examples
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
.na-split { align-items: flex-start !important; gap: 32px !important; max-width: 1320px; margin: 0 auto !important; }
.na-left { position: sticky !important; top: 10px; align-self: flex-start !important; }
.na-right { height: calc(100vh - 28px); overflow-y: auto !important; padding: 6px 8px 24px 0; }
/* children keep their natural height (no squeezing); the pane scrolls instead */
.na-right > *, .na-right .gr-group > *, .na-right .gr-group > * > * { flex-shrink: 0 !important; }
.na-right .gr-group { overflow: visible !important; }
/* attached files as tags */
/* flip cards: one side at a time */
.na-fc { display: block; position: relative; cursor: pointer; border-radius: 14px; padding: 16px 18px; margin: 10px 0;
         border: 1px solid var(--na-line); min-height: 3.2em; user-select: none; }
.na-fc input { position: absolute; opacity: 0; pointer-events: none; }
.na-fc small { display: block; font-size: .72em; opacity: .55; margin-bottom: 6px; letter-spacing: .03em; }
.na-fc .na-fc-front, .na-fc .na-fc-back { display: block; }
.na-fc .na-fc-back { display: none; }
.na-fc input:checked ~ .na-fc-front { display: none; }
.na-fc input:checked ~ .na-fc-back { display: block; }
.na-fc:has(input:checked) { border-color: #A9C2B5; }

/* clickable step checklist: ticked steps are crossed out */
.na-steplist label { border-radius: 10px !important; padding: 8px 12px !important; width: 100%; }
.na-steplist label:has(input:checked) span { text-decoration: line-through; opacity: .55; }
.na-tr { padding: 4px 0; border-bottom: 1px solid #8881; }
.na-tr .na-loc { margin: 0 8px 0 0; }
.na-tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 2px 8px; }
.na-tagfile { display: inline-flex; align-items: center; gap: 6px; font-size: .8rem; padding: 4px 10px;
              border-radius: 999px; background: var(--na-sage-soft); color: var(--na-ink); max-width: 100%; }
.na-tagfile b { font-weight: 600; font-size: .68rem; letter-spacing: .04em; opacity: .7; }
.na-tagfile span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 220px; }
.na-hint { font-size: .8rem; color: var(--na-muted); margin: 2px 4px 6px; }
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
.na-mic { border-radius: 14px !important; }

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
    if res and "_easier" in res:
        return render_easier(res["_easier"], 0, font, size, spacing, bg, bionic)
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
    elif route.get("task") == "transcribe" and (res.get("evidence") or {}).get("segments"):
        # Transcript: one line per sentence, each with the time it was said.
        out.append(f"<h3>{T('Transcript')}</h3>" + "".join(
            f"<div class='na-tr'><span class='na-loc'>{int(s['start']) // 60:02d}:{int(s['start']) % 60:02d}</span> "
            f"{T(s['text'])}</div>" for s in res["evidence"]["segments"]))
    else:
        lines = [ln for ln in str(res["answer"]).split("\n") if ln.strip()]
        out.append("".join(f"<p class='na-answer'>{T(ln)}</p>" for ln in lines))
    at = (res.get("evidence") or {}).get("at_time")
    if at and at.get("lines") and not str(res.get("answer", "")).startswith("Around"):
        out.append(f"<div class='na-card' style='background:{surface}'><h3>{T('What was said around')} {at['loc']}</h3>"
                   + "".join(f"<div class='na-tr'><span class='na-loc'>{l['loc']}</span> {T(l['text'])}</div>"
                             for l in at["lines"]) + "</div>")
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
        # Real flip cards: the front shows ONLY the question; tap to turn it over and see ONLY the answer.
        n = len(res["flashcards"])
        out.append("<div class='na-flash' style='margin-top:18px'><h3 style='margin:0'>Flashcards "
                   "<span style='font-weight:400;font-size:.8em'>(tap a card to flip it)</span></h3>")
        out += [f"<label class='na-fc' style='background:{surface}'><input type='checkbox'>"
                f"<span class='na-fc-front'><small>Card {i} of {n} · question</small>{T(c['q'])}</span>"
                f"<span class='na-fc-back'><small>Card {i} of {n} · answer</small>{T(c['a'])}</span></label>"
                for i, c in enumerate(res["flashcards"], 1)]
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


def submit(box, files, which, outputs, rate, font, size, spacing, bg, bionic, ws_name, session, read_aloud=False):
    """Composer submit: answer, clear the typed text (keep the files), and show the Understand tab."""
    question = ((box or {}).get("text") or "").strip()
    outputs = list(outputs or [])
    if read_aloud and "Read aloud" not in outputs:
        outputs.append("Read aloud")
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
        out.append(f"<div class='na-card' style='background:{surface}'><h3>{T('What to do, one at a time')}</h3>"
                   "<ol style='margin:6px 0 0 1.2em;padding:0'>" + "".join(f"<li style='margin-bottom:.5em'>{T(i)}</li>"
                                                                          for i in ins) +
                   f"</ol><div class='na-status'>{T('Word for word from the text.')}</div></div>")
    if res.get("unclear"):
        out.append(f"<div class='na-card' style='background:{surface}'><h3>Unclear phrases "
                   "<span style='font-weight:400;font-size:.8em'>(highlighted below)</span></h3><dl class='na-terms'>"
                   + "".join(f"<dt>“{_esc(u['phrase'])}”<span class='na-tag'>{_esc(u['label'])}</span></dt>"
                             f"<dd>{T(u['meaning'])}</dd>" for u in res["unclear"]) + "</dl></div>")
    out.append(f"<h3 style='margin-top:18px'>Original, with unclear phrases marked</h3>"
               f"<div class='na-card' style='background:{surface};margin-top:6px'>"
               f"{_highlight(res.get('original', ''), res.get('unclear') or [])}</div>")
    good = [p for p in res.get("plain") or [] if p["checked"]]
    if good:
        out.append(f"<details style='margin-top:18px'><summary>{T('Plain language, next to the original')} "
                   f"({len(good)})</summary>")
        for p in good:
            out.append(f"<div class='na-plain'><div class='na-orig' style='background:{surface}'>{T(p['original'])}"
                       f"</div><div style='background:{surface}'>{T(p['plain'])}</div></div>")
        skipped = len(res.get("plain") or []) - len(good)
        if skipped:
            out.append(f"<p class='na-status'>{skipped} paragraph(s) left as they are: the simpler version changed a "
                       "number or a condition.</p>")
        out.append("</details>")
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
    """One step at a time: the step to do NOW, big; what's next, small; everything else folded away."""
    T = _bionic if bionic else _esc
    style, surface = _style(font, size, spacing, bg)
    ws = _load_ws(ws_name)
    if not ws:
        return (f"<div class='na-out' style='{style}'><span class='na-empty'>"
                + T("Attach your assignment in the box on the left (and any recordings, slides or photos), then press "
                    "Help me start. You'll get one small step at a time.") + "</span></div>")
    vis = ws.visible_steps()
    tops = [s for s in ws.steps if not s.get("parent")]      # breaking a step down never adds to the count
    done = sum(s["done"] for s in tops)
    nxt = ws.next_step()
    out = [f"<div class='na-out' style='{style}'><p class='na-goal'>{T(ws.goal)}</p>",
           f"<div class='na-bar'><div style='width:{100 * done / max(len(tops), 1):.0f}%'></div></div>",
           f"<div class='na-status'>{done} of {len(tops)} steps done</div>"]
    if nxt:
        req = ws.requirements[nxt["requirement"]] if nxt.get("requirement") is not None and \
            nxt["requirement"] < len(ws.requirements) else None
        label = "Your step now"
        if nxt.get("parent"):
            parts = [k for k in ws.steps if k.get("parent") == nxt["parent"]]
            parent = ws.step(nxt["parent"])
            label = f"Part {parts.index(nxt) + 1} of {len(parts)} of: " + (parent["text"] if parent else "")
        out.append(f"<div class='na-card' style='background:{surface}'><div class='na-status'>{T(label)}</div>"
                   f"<div class='na-focus'>{T(nxt['text'])}</div>")
        if req:
            out.append(f"<div class='na-status'>Because your file says "
                       f"<span class='na-loc'>{_esc(req['source'])} · {_esc(req['loc'])}</span></div>"
                       f"<div class='na-quote'>{T(req['quote'])}</div>")
        out.append("</div>")
        rest = [s for s in vis if not s["done"] and s is not nxt]
        if rest:
            out.append(f"<p class='na-status' style='margin-top:12px'>Up next: {T(rest[0]['text'])}</p>")
    else:
        out.append("<p class='na-focus'>" + T("All steps done. Nice work.") + "</p>")

    conflicts = [c for r in ws.requirements for c in r.get("conflicts") or []]
    if conflicts:
        out.append(f"<div class='na-conflict' style='margin:16px 0 0'>{T('Check with your teacher:')} "
                   f"{_esc(conflicts[0])}. {T('Your files disagree, so Percie shows both and does not pick one.')}</div>")

    if not focus:
        # The steps themselves are a clickable checklist below this card (tick / untick).
        out.append(f"<div class='na-suggest' style='margin-top:14px'>"
                   f"{T('Your steps are listed below: click one to tick it off or untick it. Suggested by Percie, not stated in your files.')}</div>")
        if ws.requirements:
            groups = "".join(
                f"<p style='margin:12px 0 4px'><b>{wsp.TYPE_LABEL[t]}</b></p><ul>"
                + "".join(_req_html(r, T, surface) for r in ws.requirements if r["type"] == t) + "</ul>"
                for t in wsp.TYPES if any(r["type"] == t for r in ws.requirements))
            out.append(f"<details style='margin-top:10px'><summary>{T('What your files say')} "
                       f"({len(ws.requirements)})</summary>{groups}</details>")
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
# One composer and one row of actions on the left (fixed); everything that grows goes in the right
# pane, which is the only part that scrolls. Actions that only make sense for some files (code, a
# recording) appear only when such a file is attached, so there are fewer choices on screen.
ACTIONS = {   # label: (kind, question)
    "Explain simply": ("ask", "Summarize this simply."),
    "Key points": ("ask", "What are the key points?"),
    "Make it easier": ("easier", None),
    "What does it mean?": ("ask", "What does the teacher really mean? Explain anything vague literally."),
    "Quiz me": ("ask", "Make flashcards to quiz me on this."),
    "Catch me up": ("catchup", None),
    "Transcript": ("ask", "Transcribe this word for word."),
    "Run the code": ("ask", "Explain this code and run it."),
    "Fix the code": ("ask", "Fix this code."),
    "Write the code": ("write", None),
}
CORE = ["Explain simply", "Key points", "Make it easier", "What does it mean?", "Quiz me"]
WHEN_RECORDING, WHEN_CODE = ["Transcript", "Catch me up"], ["Run the code", "Fix the code", "Write the code"]

CODE_TEXT = re.compile(r"^\s*(def \w+\s*\(|class \w+|import \w|from [\w.]+ import |for .+:\s*$|while .+:\s*$|"
                       r"print\(|return\b|>>> )", re.M)
CODE_WORDS = re.compile(r"\b(code|function|script|program|python|def)\b", re.I)


def looks_like_code(text: str) -> bool:
    return bool(text) and ("```" in text or bool(CODE_TEXT.search(text)))


_kinds_cache: dict = {}


def _file_kinds(paths):
    key = tuple(paths)
    if key not in _kinds_cache:
        _kinds_cache.clear()
        _kinds_cache[key] = _file_kinds_uncached(paths)
    return _kinds_cache[key]


def _file_kinds_uncached(paths):
    from .ingest import sniff
    from .tools.code_explain import find_snippets
    has_av = has_code = False
    for p in paths:
        try:
            k = sniff(p)
            has_av |= k in ("audio", "video")
            if k == "code":
                has_code = True
            elif k == "document" and not has_code:
                from .ingest import load_document
                has_code = bool(find_snippets(load_document(p)))
        except Exception:
            continue
    return has_av, has_code


KIND_TAG = {"document": "DOC", "code": "CODE", "audio": "AUDIO", "video": "VIDEO", "image": "IMAGE"}


def file_tags(paths) -> str:
    """Attached files as calm tags: type + name. Remove a file with its x inside the box."""
    if not paths:
        return ""
    from .ingest import sniff
    tags = []
    for p in paths:
        try:
            kind = KIND_TAG.get(sniff(p), "FILE")
        except Exception:
            kind = "FILE"
        name = Path(p).name
        tags.append(f"<span class='na-tagfile' title='{html.escape(name, quote=True)}'><b>{kind}</b>"
                    f"<span>{html.escape(name)}</span></span>")
    return "<div class='na-tags'>" + "".join(tags) + "</div>"


def _plan_matches(ws_name, paths) -> bool:
    ws = _load_ws(ws_name)
    return bool(ws) and {s["name"] for s in ws.sources} == {Path(p).name for p in paths}


def sync_files(box, current, ws_name=None):
    """Composer changed. A microphone recording (Gradio names it audio.wav) is a spoken question: it is
    transcribed locally into the text box and removed from the files. Other files are the student's
    materials: kept in state; 'Ask about' and the file-specific actions update only when they change."""
    box = box or {}
    raw = list(box.get("files") or [])
    voice = [f for f in raw if Path(_paths([f])[0]).name.lower() == "audio.wav"]
    box_out = gr.skip()
    if voice:
        from .tools import audio
        words = ""
        try:
            words = (audio.transcribe(_paths(voice[-1:])[0]).get("text") or "").strip()
        except Exception:
            pass
        if not words:
            gr.Warning("I didn't catch any words. Try again a little closer to the microphone.")
        raw = [f for f in raw if f not in voice]
        box_out = {"text": f"{(box.get('text') or '').strip()} {words}".strip(), "files": raw}
    paths = _paths(raw)
    text = (box_out["text"] if voice else box.get("text")) or ""
    # Code actions show for a file with code AND for code (or a request to write code) typed or pasted in the box.
    show_code = looks_like_code(text) or bool(CODE_WORDS.search(text))
    if paths == (current or []):
        return (gr.skip(), gr.skip(), gr.skip(), gr.update(visible=show_code or _file_kinds(paths)[1]), box_out,
                gr.skip(), gr.skip())
    # New materials: a plan made from other files is set aside (still saved, openable from the sidebar),
    # so Do and Resume never show stale steps. Do makes a fresh plan when opened.
    ws_out = gr.skip() if (not paths or _plan_matches(ws_name, paths)) else None
    has_av, has_code = _file_kinds(paths)
    names = [Path(p).name for p in paths]
    which = gr.update(choices=names + ([ALL] if len(names) > 1 else []) + [NONE],
                      value=names[0] if names else NONE, visible=len(names) > 1)
    return (paths, which, gr.update(visible=has_av), gr.update(visible=has_code or show_code), box_out,
            file_tags(paths), ws_out)


def act(label, box, files, which, rate, font, size, spacing, bg, bionic, ws_name, session, read_aloud):
    """One click = one finished result on the attached file. No extra choices or boxes."""
    kind, question = ACTIONS[label]
    outputs = ["Read aloud"] if read_aloud else []
    typed = ((box or {}).get("text") or "").strip()
    if label in ("Run the code", "Fix the code") and looks_like_code(typed):
        # Code pasted in the box wins over attached files: run / fix exactly what the student pasted.
        code = typed if "```" in typed else f"```python\n{typed}\n```"
        return ask(f"{question}\n{code}", [], NONE, outputs, rate, font, size, spacing, bg, bionic, ws_name, session)
    if kind == "write":
        if not typed:
            raise gr.Error("Type what the function should do in the box (or paste its def line with a few "
                           ">>> examples), then press Write the code.")
        prompt = typed if re.search(r"\bdef \w+\s*\(", typed) else f"Write a Python function: {typed}"
        return ask(prompt, [], NONE, outputs, rate, font, size, spacing, bg, bionic, ws_name, session)
    if kind == "ask":
        return ask(question, files, which, outputs, rate, font, size, spacing, bg, bionic, ws_name, session)
    if kind == "catchup":   # a time typed in the box ("12:40") is used; otherwise the last 5 minutes
        when = typed if re.search(r"\d", typed) else ""
        return catch_me_up(when, files, which, outputs, rate, font, size, spacing, bg, bionic, ws_name, session)
    res, _, session = make_easier(typed if len(typed) > 60 else "", files, which, session)   # a pasted passage, or the file
    wrapped = {"_easier": res}
    return (render(wrapped, font, size, spacing, bg, bionic), gr.update(visible=False), gr.update(visible=False),
            "", wrapped, session)


def do_start(files, session, progress=gr.Progress()):
    session = session or Session()
    paths = _paths(files)
    if not paths:
        raise gr.Error("Attach your assignment first: use the clip in the box on the left.")
    docs = [p for p in paths if Path(p).suffix.lower() in (".pdf", ".txt", ".md", ".docx")] or paths
    name = Path(docs[0]).stem.replace("_", " ").strip() or "My assignment"
    n = [0]

    def say(msg):
        n[0] += 1
        progress(min(0.95, n[0] / (len(paths) * 2 + 3)), desc=msg)
    try:
        wsp.help_me_start(name, paths, "", session, visual=True, progress=say)
    except Exception as e:
        raise gr.Error(str(e))
    return name, _ws_names(name), session


def auto_start(files, ws_name, session, progress=gr.Progress()):
    """Opening Do with files attached and no plan for them: make the plan straight away."""
    paths = _paths(files)
    if not paths or _plan_matches(ws_name, paths):
        return gr.skip(), gr.skip(), gr.skip()
    return do_start(files, session, progress)


def step_list(ws_name):
    """The steps as a checklist: ticked = done. Broken-down parts are indented under their step."""
    ws = _load_ws(ws_name)
    if not ws or not ws.steps:
        return gr.update(choices=[], value=[], visible=False)
    vis = ws.visible_steps()
    choices = [(("    ↳ " if s.get("parent") else "") + s["text"], s["id"]) for s in vis]
    return gr.update(choices=choices, value=[s["id"] for s in vis if s["done"]], visible=True)


def set_steps(ws_name, ticked):
    """Click a step to tick it off or untick it (undo). The first unticked step becomes the current one."""
    ws = _load_ws(ws_name)
    if not ws:
        return ws_name
    ticked = set(ticked or [])
    reopened = False
    for s in ws.visible_steps():
        if s["done"] != (s["id"] in ticked):
            s["done"] = s["id"] in ticked
            reopened |= not s["done"]
            ws.event(f"{'finished' if s['done'] else 'reopened'}: {s['text']}")
    for top in [s for s in ws.steps if not s.get("parent")]:
        kids = [k for k in ws.steps if k.get("parent") == top["id"]]
        if kids:
            top["done"] = all(k["done"] for k in kids)
    cur = ws.step(ws.current)
    if reopened or not cur or cur["done"]:      # undo: the earliest unticked step is the one to do now
        nxt = next((s for s in ws.visible_steps() if not s["done"]), None)
        ws.current = nxt["id"] if nxt else None
    ws.save()
    return ws_name


def _with_ws(ws_name, fn):
    ws = _load_ws(ws_name)
    if not ws:
        raise gr.Error("Press Help me start first.")
    try:
        fn(ws)
    except ValueError as e:
        raise gr.Error(str(e))
    return ws_name


def do_done(ws_name):
    return _with_ws(ws_name, lambda ws: ws.next_step() and ws.mark_done(ws.next_step()["id"]))


def do_smaller(ws_name):
    return _with_ws(ws_name, lambda ws: wsp.make_smaller(ws, None))


def do_skip(ws_name):
    return _with_ws(ws_name, lambda ws: ws.skip())


def do_timer(ws_name):
    ws = _load_ws(ws_name)
    if not ws:
        raise gr.Error("Press Help me start first.")
    ws.stop_timer() if ws.timer_left() else ws.start_timer(15)
    return timer_html(ws_name)


def build() -> gr.Blocks:
    look_default = ("Standard", 18, 1.7, "Auto")
    with gr.Blocks(title="Percie") as demo:
        session = gr.State(None)
        last = gr.State(None)
        ws_name = gr.State(None)
        files = gr.State([])
        saved = wsp.list_workspaces()

        # --- Sidebar: settings only, closed until you want it ----------------------------------------
        with gr.Sidebar(open=False, width=290):
            gr.HTML("<div class='na-section' style='margin-top:4px'>Reading</div>")
            font = gr.Radio(list(FONTS), value="Standard", label="Font")
            size = gr.Slider(14, 30, value=18, step=1, label="Text size")
            spacing = gr.Slider(1.3, 2.4, value=1.7, step=0.1, label="Line spacing")
            bg = gr.Radio(list(BACKGROUNDS), value="Auto", label="Background")
            bionic = gr.Checkbox(value=False, label="Bionic reading (bold word starts)")
            read_aloud = gr.Checkbox(value=False, label="Read answers aloud")
            rate = gr.Slider(-5, 5, value=0, step=1, label="Read-aloud speed")
            gr.HTML("<div class='na-section'>Saved assignments</div>")
            ws_pick = gr.Dropdown(saved, show_label=False, value=saved[0] if saved else None)
            gr.HTML("<div class='na-section'>Running locally</div>")
            gr.HTML(status_html)
        look = [font, size, spacing, bg]

        with gr.Row(equal_height=False, elem_classes="na-split"):
            # --- Left: fixed. Ask, attach, speak; pick one action. ------------------------------------
            with gr.Column(scale=2, min_width=340, elem_classes="na-left"):
                gr.HTML("<div class='na-head'><h1>Percie</h1><p>Understand it. Start it. Pick up where you "
                        "left off. All on this laptop.</p></div>")
                box = gr.MultimodalTextbox(file_count="multiple", sources=["upload", "microphone"], show_label=False,
                                           lines=2, max_lines=6, autofocus=True, submit_btn="Ask",
                                           elem_classes="na-composer",
                                           placeholder="Ask anything, or attach your files with the clip")
                tags = gr.HTML()
                gr.HTML("<div class='na-hint'>Enter sends · Shift+Enter for a new line · mic to speak</div>")
                which = gr.Dropdown([NONE], value=NONE, label="Ask about", visible=False)
                with gr.Tabs(selected="understand") as tabs:
                    with gr.Tab("Understand", id="understand") as tab_u:
                        with gr.Row(elem_classes="na-chips"):
                            core = [gr.Button(lbl, size="sm", variant="secondary") for lbl in CORE]
                        with gr.Row(elem_classes="na-chips", visible=False) as rec_row:
                            rec = [gr.Button(lbl, size="sm", variant="secondary") for lbl in WHEN_RECORDING]
                        with gr.Row(elem_classes="na-chips", visible=False) as code_row:
                            code = [gr.Button(lbl, size="sm", variant="secondary") for lbl in WHEN_CODE]
                        gr.HTML("<div class='na-hint'>Catch me up: type a time like 12:40 first, or leave the box "
                                "empty for the last 5 minutes. Make it easier: paste a passage, or use your file.</div>")
                    with gr.Tab("Do", id="do") as tab_d:
                        start_btn = gr.Button("Help me start", variant="primary", size="lg")
                        with gr.Row():
                            done_btn = gr.Button("Done", variant="primary")
                            small_btn = gr.Button("Break down steps", variant="secondary")
                            skip_btn = gr.Button("Skip for now", variant="secondary")
                        timer_btn = gr.Button("15-minute focus timer (start / stop)", size="sm", variant="secondary")
                        timer_box = gr.HTML()
                    with gr.Tab("Resume", id="resume") as tab_r:
                        where_btn = gr.Button("Where was I?", variant="primary", size="lg")
                        gr.HTML("<div class='na-hint'>Read from your saved progress. Nothing is guessed.</div>")

            # --- Right: the only part that scrolls --------------------------------------------------------
            with gr.Column(scale=3, min_width=380, elem_classes="na-right"):
                with gr.Group(visible=True) as pane_u:
                    result = gr.HTML(render(None, *look_default))
                    speech = gr.Audio(label="Read aloud", type="filepath", interactive=False, autoplay=True,
                                      visible=False)
                    captions = gr.File(label="Captions (.vtt)", interactive=False, visible=False)
                    with gr.Accordion("How I got this answer", open=False):
                        details = gr.HTML()
                with gr.Column(visible=False) as pane_d:
                    plan = gr.HTML(render_plan(None, False, *look_default))
                    steps_cb = gr.CheckboxGroup([], value=[], label="Your steps (click to tick off or untick)",
                                                elem_classes="na-steplist", visible=False)
                resume = gr.HTML(visible=False)

        # --- wiring --------------------------------------------------------------------------------------
        box.change(sync_files, [box, files, ws_name], [files, which, rec_row, code_row, box, tags, ws_name],
                   show_progress="hidden")
        redraw = dict(fn=render, inputs=[last, *look, bionic], outputs=result, show_progress="hidden")
        show_u = dict(fn=lambda: (gr.Tabs(selected="understand"), gr.update(visible=True), gr.update(visible=False),
                                  gr.update(visible=False)), inputs=None, outputs=[tabs, pane_u, pane_d, resume])
        results = [result, speech, captions, details, last, session]
        common = [box, files, which, rate, *look, bionic, ws_name, session, read_aloud]
        box.submit(submit, [box, files, which, gr.State([]), rate, *look, bionic, ws_name, session, read_aloud],
                   [*results, box, tabs]).then(**redraw)
        box.submit(**show_u)
        for btn in core + rec + code:
            btn.click(act, [gr.State(btn.value), *common], results).then(**redraw)

        plan_in = [ws_name, gr.State(False), *look, bionic]
        resume_in = [ws_name, *look, bionic]

        def refresh(chain):
            return (chain.then(render_plan, plan_in, plan).then(step_list, ws_name, steps_cb)
                    .then(render_resume, resume_in, resume)
                    .then(timer_html, ws_name, timer_box))

        refresh(start_btn.click(do_start, [files, session], [ws_name, ws_pick, session]))
        refresh(done_btn.click(do_done, ws_name, ws_name))
        refresh(small_btn.click(do_smaller, ws_name, ws_name))
        refresh(skip_btn.click(do_skip, ws_name, ws_name))
        refresh(ws_pick.input(lambda n: n, ws_pick, ws_name))
        refresh(steps_cb.input(set_steps, [ws_name, steps_cb], ws_name))
        timer_btn.click(do_timer, ws_name, timer_box)
        gr.Timer(1.0).tick(timer_html, ws_name, timer_box, show_progress="hidden")
        where_btn.click(render_resume, resume_in, resume)
        refresh(demo.load(lambda n: n, ws_pick, ws_name))

        # Tabs switch what the right pane shows
        tab_u.select(lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
                     None, [pane_u, pane_d, resume])
        refresh(tab_d.select(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
                             None, [pane_u, pane_d, resume]).then(render_plan, plan_in, plan)
                .then(auto_start, [files, ws_name, session], [ws_name, ws_pick, session]))
        tab_r.select(lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
                     None, [pane_u, pane_d, resume]).then(render_resume, resume_in, resume)

        # Every reading setting re-draws every view at once
        for ctl in (font, size, spacing, bg, bionic):
            ctl.change(render, [last, *look, bionic], result, show_progress="hidden")
            ctl.change(render_plan, plan_in, plan, show_progress="hidden")
            ctl.change(render_resume, resume_in, resume, show_progress="hidden")
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

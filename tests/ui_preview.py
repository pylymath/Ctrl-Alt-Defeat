"""Open the UI with canned answers (no Ollama, no GPU) to check layout, colours and reading settings.

    python tests/ui_preview.py        # http://127.0.0.1:7861
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NEUROASSIST_TRACK_VRAM", "0")

from neuroassist import ui                     # noqa: E402
from neuroassist.schema import Answer, Route  # noqa: E402


def fake_answer(question, files=(), session=None, **kw):
    a = Answer("You must submit lab4.py with a function called moving_average, a plot saved as lab4_plot.png, "
               "and a short reflection of 150 to 200 words.",
               evidence={"quote": "You must submit a single Python file named lab4.py that contains a function "
                                  "called moving_average."}, confidence=0.9)
    a.route = Route("document", "qa", [], "rules")
    a.card = {"title": "What to hand in", "one_line": "Three things: code, a plot and a reflection.",
              "bullets": ["lab4.py with moving_average", "lab4_plot.png with raw and smooth lines",
                          "A reflection of 150 to 200 words"], "key_terms": []}
    a.steps = [{"step": "answer: document / qa", "model": "neuro-doc-qa", "ms": 2100, "ok": True, "vram_gb": 6.8}]
    return a


ui.answer = fake_answer
if __name__ == "__main__":
    ui.build().launch(server_name="127.0.0.1", server_port=7861, inbrowser=False, css=ui.CSS, theme=ui.theme())

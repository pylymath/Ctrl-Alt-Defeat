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


def _code_demo(question):
    """Real run of the demo assignment's starter code; only the explanation / fix text is canned."""
    from neuroassist import llm, pipeline
    llm.chat_json = lambda *a, **k: {
        "summary": "It tries to average each group of 3 readings, but it also averages the short groups at "
                   "the end, so the last numbers are too small.",
        "steps": [{"lines": "8", "what": "Starts an empty list for the results."},
                  {"lines": "9-11", "what": "For every position, takes the next `window` readings and divides "
                                            "their sum by `window`."},
                  {"lines": "12", "what": "Returns the list of averages."},
                  {"lines": "15-16", "what": "Runs it on five readings and prints the result."}],
        "issues": ["Line 9 loops over every position, so near the end the chunk has fewer than `window` "
                   "readings. Dividing by `window` anyway makes the last averages too small (16.7 and 8.0)."]}
    llm.chat = lambda *a, **k: ('```python\ndef moving_average(values, window):\n    """Average of each full '
                                'window of readings.\n    >>> moving_average([1, 2, 3, 4], 2)\n    [1.5, 2.5, 3.5]\n'
                                '    >>> moving_average([1, 2], 5)\n    []\n    """\n    result = []\n'
                                '    for i in range(len(values) - window + 1):\n        chunk = values[i:i + window]\n'
                                '        result.append(sum(chunk) / window)\n    return result\n```')
    return pipeline.answer(question, [str(ROOT / "assets" / "demo" / "assignment.txt")], pipeline.Session(),
                           extras=False)


def fake_answer(question, files=(), session=None, **kw):
    if "flashcard" in question.lower():
        a = Answer("Here are flashcards to quiz yourself. Tap a card to see the answer.")
        a.route = Route("document", "rewrite", ["flashcards"], "rules")
        a.flashcards = [{"q": "What file must you submit?", "a": "lab4.py"},
                        {"q": "What window size should you start with?", "a": "Three"},
                        {"q": "When is the lab due, according to the PDF?", "a": "Friday at 5 pm"}]
        return a
    if "code" in question.lower():
        return _code_demo(question)
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

from neuroassist import workspace as _wsp  # noqa: E402


def _fake_start(name, files, goal_hint="", session=None, visual=True, progress=None):
    """Preview only: reuse the saved sample plan for whatever files are attached."""
    ws = _wsp.Workspace.load("Lab 4")
    ws.name = name
    ws.sources = [{"name": Path(f).name, "path": f, "kind": "document", "error": ""} for f in files]
    for s in ws.steps:
        s["done"] = False
    ws.current = ws.visible_steps()[0]["id"] if ws.steps else None
    return ws.save()


_wsp.help_me_start = _fake_start

from neuroassist import accessible  # noqa: E402
accessible.easier = lambda text: {
    "gist": "Lab 4: write a moving-average program, plot it, and explain the window size.",
    "details": ["Hand in lab4.py, a plot and a short reflection", "Remove missing rows before averaging",
                "Due Friday at 5 pm; late work loses marks"],
    "plain": [{"original": text[:220], "plain": "Sensors are noisy. You will smooth the readings. Then you plot them.",
               "checked": True}],
    "unclear": [{"phrase": "go the extra mile", "kind": "idiom", "label": "Common meaning",
                 "meaning": "Do more than you have to. Here it means the optional bonus."},
                {"phrase": "in a timely manner", "kind": "vague", "label": "Not exact",
                 "meaning": "The text does not say how soon. Ask your teacher if you need a date."}],
    "instructions": accessible.instructions(text)}
if __name__ == "__main__":
    ui.launch(ui.build(), server_name="127.0.0.1", server_port=7861, inbrowser=False)

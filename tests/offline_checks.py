"""Offline checks: the harness, workspace, linking, catch-up and 'make this easier' logic, with a FAKE
model, so they run on any laptop in seconds with no Ollama and no GPU.

    python tests/offline_checks.py

They test our code (grounding checks, locating, linking, conflicts, retries, saving), not answer
quality: that is what eval/run_eval.py on the demo laptop is for."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NEUROASSIST_WORKSPACES", tempfile.mkdtemp(prefix="na_ws_"))
os.environ["NEUROASSIST_TRACK_VRAM"] = "0"
os.environ.setdefault("NEUROASSIST_PROFILE", "standard")

from neuroassist import accessible, config, llm, sources, workspace as wsp   # noqa: E402
from neuroassist.tools import catchup, video                                    # noqa: E402

DEMO = ROOT / "assets" / "demo"
ASSIGNMENT = (DEMO / "assignment.txt").read_text(encoding="utf-8")
passed = failed = 0


def check(name, cond, info=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}  {info}")


# --- 1. Harness: empty replies, thinking fallback, bad JSON ------------------------------------------
print("harness")


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, **kw):
        self.calls.append(kw)
        content, thinking = self.replies.pop(0)
        return SimpleNamespace(message=SimpleNamespace(content=content, thinking=thinking), done_reason="stop")


real_client = llm._client
llm._client = FakeClient([("", None), ('{"answer": "right"}', None)])
check("empty reply is retried", llm.chat_json("m", "q", {"type": "object", "required": ["answer"],
                                                         "properties": {"answer": {"type": "string"}}}) == {"answer": "right"})
check("retry used a different temperature", llm._client.calls[1]["options"]["temperature"] > 0)
llm._client = FakeClient([("", '{"answer": "from thinking"}')])
check("reply in `thinking` is used", llm.chat("m", "q", schema={"type": "object"}) == '{"answer": "from thinking"}')
llm._client = FakeClient([("not json", None), ('{"a": 1}', None)])
check("invalid JSON retried", llm.chat_json("m", "q", {"type": "object", "required": ["a"], "properties": {}}) == {"a": 1})
llm._client = FakeClient([('{"found": false, "quote": "", "answer": ""}', None)])
check("valid abstention is NOT retried", llm.chat_json(
    "m", "q", {"type": "object", "required": ["found", "quote", "answer"], "properties": {}})["found"] is False
      and len(llm._client.calls) == 1)
check("context pinned per model (no reloads)",
      config.pinned_ctx(config.VISION_MODEL, 1024) == config.pinned_ctx(config.VISION_MODEL, 12288))
llm._client = real_client


# --- Fake model for the features ---------------------------------------------------------------------
def fake_json(model, prompt, schema, **kw):
    props = schema.get("properties", {})
    if "items" in props and "goal" in props:                       # requirements
        if "Recording" in prompt or "recording transcript" in prompt.lower():
            return {"goal": "", "items": [
                {"type": "constraint", "text": "Remove missing rows before computing the moving average",
                 "quote": "Remove the rows with missing temperatures before you compute the moving average, not after."},
                {"type": "deadline", "text": "Submit by Thursday",
                 "quote": "Remember everything is due on Thursday at 5 pm."}]}
        return {"goal": "Write a program that smooths sensor readings with a moving average.", "items": [
            {"type": "deliverable", "text": "Submit lab4.py with a moving_average function",
             "quote": "You must submit a single Python file named lab4.py that contains a function called moving_average."},
            {"type": "deliverable", "text": "Submit a plot saved as lab4_plot.png",
             "quote": "You must also submit a plot of the raw and smoothed readings saved as lab4_plot.png."},
            {"type": "deadline", "text": "Submit by Friday 5 pm",
             "quote": "Submit everything on the course page by Friday 5 pm."},
            {"type": "constraint", "text": "Reflection of 150 to 200 words",
             "quote": "Write a short reflection of 150 to 200 words explaining what the window size changes in the plot."},
            {"type": "deliverable", "text": "Add a weighted moving average line",
             "quote": "If you want to go the extra mile, add a second line to the plot using a weighted moving average."},
            {"type": "deliverable", "text": "Cite 5 references",                       # invented: must be dropped
             "quote": "You must cite five references in APA style."},
            {"type": "deliverable", "text": "Submit lab4.py with a moving_average function",   # duplicate
             "quote": "You must submit a single Python file named lab4.py that contains a function called moving_average."},
        ]}
    if "steps" in props:
        return {"steps": [{"text": "Open the assignment and the CSV file", "requirement": 0},
                          {"text": "Write the moving_average function", "requirement": 0},
                          {"text": "Spend 45 minutes on the plot", "requirement": 1},     # invented number: dropped
                          {"text": "Make the plot and save it as lab4_plot.png", "requirement": 1}]}
    if "actions" in props:
        return {"actions": ["Open lab4.py", "Write the function signature", "Add the loop"]}
    if "points" in props:
        return {"points": [{"time": "00:05", "text": "Remove missing rows first."},
                           {"time": "00:09", "text": "Start with a window size of three."},
                           {"time": "07:30", "text": "Something never said."},             # not in window: dropped
                           {"time": "00:09", "text": "Use 12 readings."}]}                 # unknown number: dropped
    if "gist" in props:
        return {"gist": "Lab 4 asks you to smooth sensor data and submit code, a plot and a reflection.",
                "details": ["Submit lab4.py", "Plot raw and smoothed data", "Reflection: 150 to 200 words",
                            "Use 300 words"]}                                              # unknown number: dropped
    if "plain" in props:
        return {"plain": "Short plain version."}
    if "items" in props and "phrase" in str(schema):
        return {"items": [{"phrase": "go the extra mile", "kind": "idiom", "meaning": "Do more than required."},
                          {"phrase": "in a timely manner", "kind": "vague", "meaning": "The text does not say how soon."},
                          {"phrase": "never in the text", "kind": "vague", "meaning": "x"}]}   # not in source: dropped
    if "answer" in props:
        return {"answer": ""}
    return {}


llm.chat_json = fake_json

TRANSCRIPT = {"has_speech": True, "text": "", "segments": [
    {"start": 0.0, "end": 4.0, "text": "Hi everyone, quick note about lab four."},
    {"start": 4.0, "end": 8.5, "text": "Remove the rows with missing temperatures before you compute the moving average, not after."},
    {"start": 8.5, "end": 11.0, "text": "And start with a window size of three."},
    {"start": 11.0, "end": 14.0, "text": "Remember everything is due on Thursday at 5 pm."}]}
TRANSCRIPT["text"] = " ".join(s["text"] for s in TRANSCRIPT["segments"])
from neuroassist.tools import audio  # noqa: E402
audio.transcribe = lambda path: TRANSCRIPT

# --- 2. Sources ---------------------------------------------------------------------------------------
print("sources")
doc = sources.load(str(DEMO / "assignment.txt"))
check("text file split into located paragraphs", len(doc.segments) >= 8 and doc.segments[0].loc.startswith("line "))
seg = sources.locate("Submit everything on the course page by Friday 5 pm.", doc)
check("quote located to its exact line (not the heading)", seg is not None and seg.loc == "line 20", seg and seg.loc)
rec = sources.load(str(DEMO / "voice_note.wav"))
check("recording located by timestamps", [s.loc for s in rec.segments] == ["00:00", "00:04", "00:08", "00:11"])
check("parse_time", (sources.parse_time("12:40"), sources.parse_time("1:02:03"), sources.parse_time("2m10s"))
      == (760, 3723, 130))
check("conflict found: Friday vs Thursday",
      any(c.startswith("day") for c in sources.conflicts("due Friday 5 pm", "due on Thursday at 5 pm")))
check("no conflict when they agree", sources.conflicts("due Friday", "submit by friday") == [])
check("grading percentages are not conflicts", sources.conflicts("50 percent", "20 percent") == [])

# --- 3. Help me start + connect ----------------------------------------------------------------------
print("help me start / connect my materials")
ws = wsp.help_me_start("Lab 4", [str(DEMO / "assignment.txt"), str(DEMO / "voice_note.wav")], visual=False)
texts = [r["text"] for r in ws.requirements]
check("invented requirement dropped (no matching sentence)", "Cite 5 references" not in texts and ws.dropped >= 1)
check("duplicate requirement kept once", texts.count("Submit lab4.py with a moving_average function") == 1)
check("bonus tagged optional", any(r["type"] == "optional" and "weighted" in r["text"] for r in ws.requirements))
check("recording requirement located by timestamp",
      any(r["source"] == "voice_note.wav" and r["loc"] == "00:04" for r in ws.requirements))
dl = [r for r in ws.requirements if r["type"] == "deadline"]
check("deadlines from both files", {r["source"] for r in dl} == {"assignment.txt", "voice_note.wav"}, dl)
check("Friday vs Thursday flagged on BOTH deadlines",
      all(any("friday" in c and "thursday" in c for c in r["conflicts"]) for r in dl), [r["conflicts"] for r in dl])
check("recording requirement linked to the document passage",
      any(r["source"] == "voice_note.wav" and r["type"] == "constraint" and l["file"] == "assignment.txt"
          for r in ws.requirements for l in r["links"]),
      [(r["text"], [l["file"] for l in r["links"]]) for r in ws.requirements])
check("deadlines linked both ways", all(any(l["why"] == "another deadline" or l["file"] != r["source"]
                                            for l in r["links"]) for r in dl))
check("step with invented number dropped", not any("45" in s["text"] for s in ws.steps))
check("steps saved and first is current", ws.steps and ws.current == ws.steps[0]["id"])
check("no time estimates stored on steps", all("minutes" not in s for s in ws.steps))

# --- 4. Make smaller, done, where was I ---------------------------------------------------------------
print("make smaller / done / where was I")
first = ws.next_step()
wsp.make_smaller(ws, first["id"])
vis = ws.visible_steps()
check("step replaced by its smaller steps", vis[0]["text"] == "Open lab4.py" and first not in vis)
ws.mark_done(vis[0]["id"])
check("done moves to the next step", ws.next_step()["text"] == "Write the function signature")
for k in [s for s in ws.steps if s.get("parent") == first["id"]]:
    ws.mark_done(k["id"])
check("parent done when all its parts are", ws.step(first["id"])["done"])
ws.record("How many references are required?", "The document does not say.", True, "assignment.txt")
ws.start_timer(10)
w = wsp.where_was_i(wsp.Workspace.load("Lab 4"))
check("reloads from disk", w["goal"].startswith("Write a program"))
check("where was I: last finished + next + open question + last file",
      w["last_done"] and w["next"] and w["unresolved"] == ["How many references are required?"]
      and w["last_material"] == "assignment.txt")
check("timer running", 590 <= wsp.Workspace.load("Lab 4").timer_left() <= 600)

# --- 5. Catch-up --------------------------------------------------------------------------------------
print("catch-up")
check("parse window", (catchup.parse_window("catch me up from 0:04"), catchup.parse_window("between 2:00 and 3:30"),
                       catchup.parse_window("the last 5 minutes")) == ((4, None), (120, 210), (-300, None)))
res = catchup.catch_up(TRANSCRIPT["segments"], 4)
check("points keep real timestamps; invented ones dropped", [p["loc"] for p in res["points"]] == ["00:04", "00:08"],
      res["points"])
check("instructions kept word for word", any("missing temperatures" in i["text"] for i in res["instructions"]))
check("window excludes earlier speech", all(s["start"] >= 4 for s in res["window"]))

# --- 6. Make this easier -------------------------------------------------------------------------------
print("make this easier")
e = accessible.easier(ASSIGNMENT)
check("gist kept", e["gist"].startswith("Lab 4"))
check("detail with an invented number dropped", "Use 300 words" not in e["details"])
check("unclear phrase not in the text dropped", [u["phrase"] for u in e["unclear"]] == ["go the extra mile",
                                                                                         "in a timely manner"])
check("idiom labelled as common meaning", e["unclear"][0]["label"] == "Common meaning")
check("plain rewrite that drops numbers is not shown",
      any(not p["checked"] for p in e["plain"]) and all(p["plain"] == p["original"] for p in e["plain"] if not p["checked"]))
check("instructions one at a time", len(e["instructions"]) >= 5)

# --- 7. Video fallback -----------------------------------------------------------------------------------
print("video fallback")
calls = []
video._direct = lambda q, f, d, t: (calls.append("direct"), video.Answer(video.UNSURE))[1]
video._timeline = lambda q, f, d, t, batch=4: (calls.append("timeline"), video.Answer("It moves right."))[1]
video._frames = lambda path, task, session: ([object()] * 4, 3.0)
config.VIDEO_MODE = "auto"
a = video.run("Which way?", ["x.mp4"], SimpleNamespace(task="motion"), None)
check("empty direct answer falls back to timeline", calls == ["direct", "timeline"] and a.answer == "It moves right.")

# --- 8. Through the real pipeline (router -> tool -> trace) ----------------------------------------------
print("pipeline")
from neuroassist import pipeline  # noqa: E402
_prev = llm.chat_json


def qa_json(model, prompt, schema, **kw):
    if "found" in schema.get("properties", {}):
        return {"found": True, "quote": "You must submit a single Python file named lab4.py that contains a "
                                        "function called moving_average.", "answer": "lab4.py"}
    return _prev(model, prompt, schema, **kw)


llm.chat_json = qa_json
a = pipeline.answer("What must I submit?", [str(DEMO / "assignment.txt")], pipeline.Session(), extras=False)
check("document QA answer with its checked quote", a.answer == "lab4.py" and a.evidence.get("quote"))
check("trace records the step", a.steps and a.steps[0]["ok"] and "document / qa" in a.steps[0]["step"])
a = pipeline.answer("I zoned out, catch me up from 0:04", [str(DEMO / "voice_note.wav")], pipeline.Session())
check("catch-up routed through the pipeline", a.route.task == "catchup" and "[00:04]" in a.answer, a.answer)
a = pipeline.answer("Catch me up", [str(DEMO / "voice_note.wav")], pipeline.Session())
check("catch-up without a time asks for one", a.abstain and "12:40" in a.answer)
a = pipeline.answer("Rewrite this in plain language", [str(DEMO / "assignment.txt")], pipeline.Session())
check("plain-language request skips QA and returns the rewrite", a.route.task == "rewrite" and a.plain)
llm.chat_json = _prev

# --- 9. Installer pieces (no downloads) ------------------------------------------------------------------
print("installer")
from neuroassist import installer  # noqa: E402
tmp = Path(tempfile.mkdtemp(prefix="na_inst_"))
(tmp / "neuro-doc-qa").mkdir()
(tmp / "neuro-doc-qa" / "unsloth.Q4_K_M.gguf").write_bytes(b"x")
check("finds the fine-tune folder", installer.find_finetune(tmp) == tmp / "neuro-doc-qa")
mf = installer.MODELFILE.format(gguf="unsloth.Q4_K_M.gguf")
check("fallback Modelfile is ChatML and points at the gguf",
      mf.startswith("FROM ./unsloth.Q4_K_M.gguf") and "<|im_start|>assistant" in mf and "{{ .Content }}" in mf)
created = []
installer.installed = lambda: set()
installer.subprocess = SimpleNamespace(run=lambda cmd, cwd=None: (created.append((cmd, cwd)), SimpleNamespace(returncode=0))[1])
check("imports with ollama create", installer.import_finetune(tmp) == "neuro-doc-qa:latest"
      and created[0][0][:3] == ["ollama", "create", "neuro-doc-qa"])
check("writes its own Modelfile when none came with it", (tmp / "neuro-doc-qa" / "Modelfile.neuroassist").exists())
installer.ENV_FILE = tmp / "neuroassist.env"
installer.write_settings({"NEUROASSIST_PROFILE": "lite", "NEUROASSIST_QA_MODEL": "neuro-doc-qa:latest"})
installer.write_settings({"NEUROASSIST_QA_MODEL": ""})
vals = config._read_settings(installer.ENV_FILE)
check("settings file merges and keeps earlier values", vals.get("NEUROASSIST_PROFILE") == "lite"
      and vals.get("NEUROASSIST_QA_MODEL") == "", vals)
config._SETTINGS["NEUROASSIST_DEMO_X"] = "from-file"
check("settings file is read", config._env("DEMO_X", "default") == "from-file")
os.environ["NEUROASSIST_DEMO_X"] = "from-env"
check("environment variable wins over the file", config._env("DEMO_X", "default") == "from-env")
os.environ["NEUROASSIST_DEMO_X"] = ""
check("empty value means 'use the default'", config._env("DEMO_X", "default") == "from-file")

# --- 10. UI renders ----------------------------------------------------------------------------------------
print("ui")
from neuroassist import ui  # noqa: E402
look = ("Dyslexia-friendly", 18, 1.7, "Cream")
for name, html_ in [("plan", ui.render_plan("Lab 4", False, *look, True)),
                    ("focus", ui.render_plan("Lab 4", True, *look)),
                    ("resume", ui.render_resume("Lab 4", *look)),
                    ("easier", ui.render_easier({**e, "original": ASSIGNMENT}, 2, *look)),
                    ("timer", ui.timer_html("Lab 4"))]:
    check(f"renders {name}", isinstance(html_, str) and len(html_) > 50 and "Traceback" not in html_)
plan_html = ui.render_plan("Lab 4", False, *look)
check("plan shows conflict and links", "Check both" in plan_html and "Also in" in plan_html)
check("plan labels steps as suggested", "Suggested by Percie" in plan_html)
bio = ui.render({"answer": "You must submit lab4.py", "route": {}, "evidence": {}}, "Standard", 18, 1.7, "Auto", True)
check("bionic reading: word starts bolded and surface tagged", "na-out na-bio" in bio and "<b>Yo</b>u" in bio)
check("bionic reading off: plain text", "<b>" not in ui.render({"answer": "You must", "route": {}, "evidence": {}},
                                                               "Standard", 18, 1.7, "Auto", False))
check("bionic applies before the first answer too", "<b>" in ui.render(None, "Standard", 18, 1.7, "Auto", True))
check("easier highlights unclear phrases", "<mark" in ui.render_easier({**e, "original": ASSIGNMENT}, 0, *look))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)

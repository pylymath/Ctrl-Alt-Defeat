"""Workspace: one assignment, its materials, what it requires, suggested small steps, and where the
student is. Saved as JSON on this laptop (~/.neuroassist/workspaces/<name>.json), never uploaded.

    Help me start         each file -> requirements, every one with an exact quote checked in code and
                          its location (page / line / timestamp); typed as deliverable, deadline,
                          constraint, grading or optional. Then SUGGESTED steps, kept apart from them.
    Connect my materials  each requirement -> passages in the other files that share its key words,
                          and "check both" hints when they state different deadlines/word counts.
                          Nothing is merged or overridden: all sources are shown.
    Break down steps      one step -> 2-3 smaller parts (the step count stays the same)
    Where was I?          read back from the saved record. No model call, nothing guessed.

What is stated by a file (requirements, quotes) and what is suggested by the assistant (steps) are
labelled differently everywhere."""
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config, llm, sources
from .tools.text_qa import unknown_numbers

HOME = Path(config._env("WORKSPACES", str(Path.home() / ".neuroassist" / "workspaces")))

SYSTEM = ("You help students with ADHD, autism or dyslexia start and finish assignments. Use ONLY the "
          "text you are given. Never invent requirements, deadlines, numbers or names.")

TYPES = ["deliverable", "deadline", "constraint", "grading", "optional"]
TYPE_LABEL = {"deliverable": "What to hand in", "deadline": "Deadlines", "constraint": "Rules and limits",
              "grading": "How it is marked", "optional": "Optional / bonus"}

REQS = {"type": "object", "required": ["goal", "items"], "properties": {
    "goal": {"type": "string"},
    "items": {"type": "array", "maxItems": 12, "items": {
        "type": "object", "required": ["type", "text", "quote"], "properties": {
            "type": {"enum": TYPES}, "text": {"type": "string"}, "quote": {"type": "string"}}}}}}

STEPS = {"type": "object", "required": ["steps"], "properties": {"steps": {
    "type": "array", "maxItems": 5, "items": {
        "type": "object", "required": ["text", "requirement"], "properties": {
            "text": {"type": "string"}, "requirement": {"type": "integer"}}}}}}

SMALLER = {"type": "object", "required": ["actions"], "properties": {"actions": {
    "type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "string"}}}}


@dataclass
class Workspace:
    name: str
    goal: str = ""
    sources: list[dict] = field(default_factory=list)        # {name, path, kind, error}
    requirements: list[dict] = field(default_factory=list)   # {type, text, quote, source, loc, links, conflicts}
    steps: list[dict] = field(default_factory=list)          # {id, text, done, requirement, parent}
    unresolved: list[str] = field(default_factory=list)      # questions the files could not answer
    history: list[dict] = field(default_factory=list)        # {t, question, answer, abstain, file}
    current: str | None = None                               # id of the step the student is on
    last_material: str | None = None                        # last file the student asked about
    timer_end: float | None = None                           # a work timer the student started
    timer_minutes: int | None = None
    log: list[dict] = field(default_factory=list)            # {t, event}
    dropped: int = 0                                         # requirements dropped: no matching sentence
    updated: float = 0.0

    # --- persistence ---------------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return HOME / f"{_slug(self.name)}.json"

    def save(self) -> "Workspace":
        self.updated = time.time()
        HOME.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(self.path)
        return self

    @classmethod
    def load(cls, name: str) -> "Workspace":
        d = json.loads((HOME / f"{_slug(name)}.json").read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def event(self, text: str):
        self.log.append({"t": time.time(), "event": text})
        self.log = self.log[-80:]

    # --- steps ---------------------------------------------------------------------------------
    def step(self, sid: str | None) -> dict | None:
        return next((s for s in self.steps if s["id"] == sid), None) if sid else None

    def visible_steps(self) -> list[dict]:
        """Steps in order, each broken-down step replaced by its smaller steps."""
        out = []
        for s in self.steps:
            if s.get("parent"):
                continue
            kids = [k for k in self.steps if k.get("parent") == s["id"]]
            out += kids or [s]
        return out

    def next_step(self) -> dict | None:
        vis = self.visible_steps()
        cur = self.step(self.current)
        if cur and not cur["done"] and cur in vis:
            return cur
        return next((s for s in vis if not s["done"]), None)

    def mark_done(self, sid: str, done: bool = True):
        s = self.step(sid)
        if not s:
            return
        s["done"] = done
        self.event(f"{'finished' if done else 'reopened'}: {s['text']}")
        if s.get("parent"):
            parent = self.step(s["parent"])
            if parent:
                parent["done"] = all(k["done"] for k in self.steps if k.get("parent") == s["parent"])
        nxt = self.next_step()
        self.current = nxt["id"] if nxt else None
        self.save()

    def focus(self, sid: str):
        if self.step(sid):
            self.current = sid
            self.event(f"switched to: {self.step(sid)['text']}")
            self.save()

    def skip(self):
        """Move on to the next unfinished step without marking this one done (it stays in the list)."""
        vis = [s for s in self.visible_steps() if not s["done"]]
        if len(vis) < 2:
            return
        cur = self.next_step()
        i = vis.index(cur) if cur in vis else -1
        self.current = vis[(i + 1) % len(vis)]["id"]
        self.event(f"skipped for now: {cur['text'] if cur else ''}")
        self.save()

    def add_step(self, text: str):
        text = text.strip()
        if text:
            self.steps.append(_new_step(text, None))
            self.event(f"added own step: {text}")
            self.save()

    # --- activity ------------------------------------------------------------------------------
    def record(self, question: str, answer: str, abstain: bool, file: str | None = None):
        self.history.append({"t": time.time(), "question": question, "answer": answer, "abstain": abstain,
                             "file": file})
        self.history = self.history[-30:]
        if file:
            self.last_material = file
        if abstain and question not in self.unresolved:
            self.unresolved.append(question)
        self.save()

    def start_timer(self, minutes: int):
        self.timer_minutes = int(minutes)
        self.timer_end = time.time() + 60 * int(minutes)
        self.event(f"started a {minutes}-minute timer")
        self.save()

    def stop_timer(self):
        self.timer_end = None
        self.save()

    def timer_left(self) -> int | None:
        if not self.timer_end:
            return None
        return max(0, int(self.timer_end - time.time()))


def _slug(name: str) -> str:
    return re.sub(r"[^\w-]+", "-", name.strip().lower()).strip("-") or "assignment"


def _new_step(text: str, requirement, parent=None) -> dict:
    return {"id": uuid.uuid4().hex[:8], "text": text.strip(), "done": False, "requirement": requirement,
            "parent": parent}


def list_workspaces() -> list[str]:
    if not HOME.exists():
        return []
    items = []
    for p in HOME.glob("*.json"):
        try:
            items.append((p.stat().st_mtime, json.loads(p.read_text(encoding="utf-8"))["name"]))
        except Exception:
            continue
    return [n for _, n in sorted(items, reverse=True)]


def _clip(t: str, n: int) -> str:
    return t if len(t) <= n else t[:n] + "\n[...]"


def _similar(a: str, b: str) -> bool:
    wa, wb = sources.words(a), sources.words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.6


# --- Help me start + Connect my materials ---------------------------------------------------------
def extract_requirements(src: sources.Source) -> tuple[str, list[dict], int]:
    """(goal, requirements, dropped) for one file. Each requirement must quote a sentence that is
    really in the file; its location comes from the segment that contains the quote."""
    if not src.text.strip():
        return "", [], 0
    what = "recording transcript" if src.kind in ("audio", "video") else "document"
    r = llm.chat_json(config.TEXT_MODEL,
                      f"{what.capitalize()} '{src.name}':\n<<<\n{_clip(src.text, 6000)}\n>>>\n\n"
                      "List everything this asks the student to hand in or do, and every deadline, rule or "
                      "limit, and how it is marked. For each item: 'type' = deliverable, deadline, constraint, "
                      "grading, or optional (if the text says it is optional, a bonus, or not required); "
                      "'text' = the item in plain words; 'quote' = the exact sentence from the text that states "
                      "it. If the text gives no exact deadline, do not make one up. "
                      "'goal' = what the assignment is for, in one plain sentence (empty if not an assignment).",
                      REQS, system=SYSTEM, num_ctx=config.NUM_CTX_DOC, max_tokens=1200)
    out, dropped = [], 0
    for it in r.get("items", []):
        if not isinstance(it, dict) or not it.get("text") or not it.get("quote"):
            continue
        seg = sources.locate(it["quote"], src)
        if not seg or unknown_numbers(it["text"], src.text):
            dropped += 1            # no supporting sentence, or a number the file never says
            continue
        kind = it.get("type") if it.get("type") in TYPES else "deliverable"
        if kind == "optional" or re.search(r"\b(optional|bonus|not required|extra credit)\b",
                                            it["quote"] + " " + seg.text, re.I):
            kind = "optional"
        out.append({"type": kind, "text": it["text"].strip(), "quote": it["quote"].strip(),
                    "source": src.name, "loc": seg.loc, "t": seg.t, "links": [], "conflicts": []})
    return (r.get("goal") or "").strip(), out, dropped


def connect(reqs: list[dict], srcs: list[sources.Source]) -> list[dict]:
    """Merge the same requirement found in two files, then link each requirement to passages in the
    other files and flag deadline/length facts that disagree. Plain code: no model call."""
    merged: list[dict] = []
    for r in reqs:
        twin = next((m for m in merged if m["type"] == r["type"] and _similar(m["text"], r["text"])), None)
        if twin:
            if twin["source"] != r["source"]:
                twin["links"].append({"file": r["source"], "loc": r["loc"], "text": r["quote"], "t": r["t"],
                                      "why": "states the same requirement"})
                twin["conflicts"] += [f"{c} ({twin['source']} {twin['loc']} vs {r['source']} {r['loc']})"
                                      for c in sources.conflicts(twin["quote"], r["quote"])]
            continue                     # the same requirement twice in one file: keep the first
        merged.append(r)
    for r in merged:
        have = {(l["file"], l["loc"]) for l in r["links"]}
        for seg, shared in sources.related(r["text"] + " " + r["quote"], srcs, exclude=r["source"]):
            if (seg.file, seg.loc) in have:
                continue
            r["links"].append({"file": seg.file, "loc": seg.loc, "text": seg.text[:300], "t": seg.t,
                               "why": "mentions " + ", ".join(shared[:4]),
                               "visual": seg.kind == "visual"})
            if r["type"] == "constraint":
                r["conflicts"] += [f"{c} ({r['source']} {r['loc']} vs {seg.file} {seg.loc})"
                                   for c in sources.conflicts(r["quote"], seg.text)]
    # Deadlines from different files are always compared, whatever words they share, and a
    # disagreement is shown on BOTH of them.
    dls = [r for r in merged if r["type"] == "deadline"]
    for i, a in enumerate(dls):
        for b in dls[i + 1:]:
            if a["source"] == b["source"]:
                continue
            for c in sources.conflicts(a["quote"], b["quote"]):
                where = f"({a['source']} {a['loc']} vs {b['source']} {b['loc']})"
                for r, other in ((a, b), (b, a)):
                    r["conflicts"].append(f"{c} {where}")
                    # Drop a word-overlap link to the same passage, then link the deadline itself
                    r["links"] = [l for l in r["links"] if not (
                        l["file"] == other["source"] and (l["loc"] == other["loc"] or other["quote"] in l["text"]))]
                    r["links"].append({"file": other["source"], "loc": other["loc"], "text": other["quote"],
                                       "t": other["t"], "why": "another deadline"})
    for r in merged:
        r["conflicts"] = list(dict.fromkeys(r["conflicts"]))
    return merged


def help_me_start(name: str, files: list[str], goal_hint: str = "", session=None, visual: bool = True,
                  progress=None) -> Workspace:
    say = progress or (lambda msg: None)
    srcs = []
    for f in files:
        say(f"Reading {Path(f).name}")
        srcs.append(sources.load(f, session, visual=visual))
    readable = [s for s in srcs if s.kind in ("document", "code", "audio", "video") and s.text.strip()]
    if not readable:
        raise ValueError("Add the assignment as a document (PDF/text) or a recording first. "
                         "Photos and videos can be added as extra materials.")
    goals, reqs, dropped = [], [], 0
    for s in readable:
        if s.kind == "code":
            continue
        say(f"Finding requirements in {s.name}")
        g, rs, d = extract_requirements(s)
        if g:
            goals.append(g)
        reqs += rs
        dropped += d
    say("Connecting your materials")
    reqs = connect(reqs, srcs)
    order = {t: i for i, t in enumerate(TYPES)}
    reqs.sort(key=lambda r: order[r["type"]])
    goal = goals[0] if goals else (goal_hint or "Finish the assignment")

    steps = []
    todo = [(i, r) for i, r in enumerate(reqs) if r["type"] in ("deliverable", "optional", "constraint")]
    if todo:
        say("Suggesting first steps")
        listing = "\n".join(f"{i}. {r['text']}{' (optional)' if r['type'] == 'optional' else ''}" for i, r in todo)
        hint = f"\nThe student adds: {goal_hint}" if goal_hint else ""
        s = llm.chat_json(config.TEXT_MODEL,
                          f"Assignment goal: {goal}\nRequirements:\n{listing}{hint}\n\nSuggest AT MOST 5 steps: a short checklist of "
                          "concrete steps in a sensible order. Each step is ONE action that starts with a verb and "
                          "says exactly what to open, write or check. Make the first step very small and easy. "
                          "'requirement' = the number of the requirement it serves. Optional items go last. "
                          "Do not give time estimates or dates.",
                          STEPS, system=SYSTEM, num_ctx=config.NUM_CTX_DOC, max_tokens=900)
        known = goal + listing + "\n".join(s.text for s in readable)
        valid = {i for i, _ in todo}
        for st in s.get("steps", []):
            if isinstance(st, dict) and st.get("text") and not unknown_numbers(st["text"], known):
                req = st.get("requirement")
                steps.append(_new_step(st["text"], req if req in valid else None))
        steps = steps[:5]                       # never more than 5: a short list is easier to start

    ws = Workspace(name=name, goal=goal, requirements=reqs, steps=steps, dropped=dropped,
                   sources=[{"name": s.name, "path": s.path, "kind": s.kind, "error": s.error} for s in srcs])
    ws.current = steps[0]["id"] if steps else None
    ws.event(f"started: {len(reqs)} requirements from {len(readable)} file(s), {len(steps)} suggested steps")
    return ws.save()


# --- Break down steps ------------------------------------------------------------------------------
def make_smaller(ws: Workspace, sid: str | None = None) -> Workspace:
    s = ws.step(sid) or ws.next_step()
    if not s:
        raise ValueError("No step to break down. Press 'Help me start' first.")
    req = ws.requirements[s["requirement"]] if s.get("requirement") is not None and \
        s["requirement"] < len(ws.requirements) else None
    ctx = f"\nIt is for this requirement: {req['text']} (the file says: \"{req['quote']}\")" if req else ""
    r = llm.chat_json(config.TEXT_MODEL,
                      f"Assignment goal: {ws.goal}\nStep: {s['text']}{ctx}\n\nThis step feels too big to start. "
                      "Break it into 2 or 3 smaller actions, each starting with a verb. The first one must be "
                      "almost effortless (e.g. open the file, write the heading).",
                      SMALLER, system=SYSTEM, num_ctx=config.NUM_CTX_DOC, max_tokens=400)
    parent_id = s.get("parent") or s["id"]
    kids = [_new_step(a, s.get("requirement"), parent=parent_id)
            for a in r.get("actions", [])[:3] if isinstance(a, str) and a.strip()]
    if len(kids) < 2:
        raise ValueError("Could not break this step down. Try again, or add your own step.")
    idx = ws.steps.index(s)
    ws.steps[idx + 1:idx + 1] = kids
    if s.get("parent"):                 # a small step broken down again: its parts replace it
        ws.steps.remove(s)
    ws.current = kids[0]["id"]
    ws.event(f"made smaller: {s['text']}")
    return ws.save()


# --- Where was I? ---------------------------------------------------------------------------------
def ago(t: float) -> str:
    m = int((time.time() - t) // 60)
    if m < 1:
        return "just now"
    if m < 60:
        return f"{m} min ago"
    h = m // 60
    return f"{h} h ago" if h < 48 else f"{h // 24} days ago"


def where_was_i(ws: Workspace) -> dict:
    """Everything here is read from the saved record; nothing is generated."""
    vis = ws.visible_steps()
    nxt = ws.next_step()
    req = ws.requirements[nxt["requirement"]] if nxt and nxt.get("requirement") is not None and \
        nxt["requirement"] < len(ws.requirements) else None
    last_done = next((e for e in reversed(ws.log) if e["event"].startswith("finished")), None)
    deadlines = [r for r in ws.requirements if r["type"] == "deadline"]
    return {
        "goal": ws.goal,
        "done": sum(s["done"] for s in vis), "total": len(vis),
        "last_active": ago(ws.updated) if ws.updated else "",
        "last_done": last_done["event"].removeprefix("finished: ") if last_done else None,
        "last_done_ago": ago(last_done["t"]) if last_done else None,
        "next": nxt, "next_source": req,
        "last_question": ws.history[-1] if ws.history else None,
        "last_material": ws.last_material,
        "unresolved": ws.unresolved[-5:],
        "deadlines": deadlines,
        "recent": [{"event": e["event"], "ago": ago(e["t"])} for e in ws.log[-5:]][::-1],
        "sources": [s["name"] for s in ws.sources],
    }

"""Scoring for the practice pack. Deliberately lenient where references are full sentences:
every prediction and reference is written to the results CSV so the team can review by hand.
Nothing in here is ever imported by the app."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
STOP = set("a an the is are was were it this that of to in on and or with by at for from as be "
           "there its it's sound sounds like".split())
COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray", "grey", "brown"}
MOTION_GROUPS = [{"left", "leftward", "leftwards"}, {"right", "rightward", "rightwards"},
                 {"up", "upward", "upwards", "rises", "rising", "upper", "top"},
                 {"down", "downward", "downwards", "falls", "falling", "bottom"},
                 {"taller", "grows", "growing", "higher", "increases"}, {"shorter", "shrinks", "lower", "decreases"},
                 {"toward", "towards", "closer", "approach", "together"}, {"touch", "touches", "collide", "meet", "hit"}]


def norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(w for w in s.split() if w not in {"a", "an", "the"})


def stem(w: str) -> str:
    for suf in ("ing", "es", "ed", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def content_stems(s: str) -> set[str]:
    return {stem(w) for w in norm(s).split() if w not in STOP}


def ints(s: str) -> list[int]:
    out = []
    for tok in re.findall(r"[a-z]+|\d+", str(s).lower()):
        if tok.isdigit():
            out.append(int(tok))
        elif tok in NUM_WORDS:
            out.append(NUM_WORDS[tok])
    return out


def token_f1(pred: str, ref: str) -> float:
    p, r = norm(pred).split(), norm(ref).split()
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if not p or not r or not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def contains(pred: str, ref: str) -> bool:
    p, r = norm(pred), norm(ref)
    return bool(r) and (f" {r} " in f" {p} " or (len(p) >= 3 and f" {p} " in f" {r} "))


def wer(pred: str, ref: str) -> float:
    p, r = norm(pred).split(), norm(ref).split()
    if not r:
        return 0.0 if not p else 1.0
    d = list(range(len(p) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, pw in enumerate(p, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != pw))
    return d[len(p)] / len(r)


# --- Per modality: each returns (passed: bool, score: float, how: str) -------------------------
def score_image(q, pred, ans):
    ref = ans["reference_answer"]
    if re.search(r"\bhow many\b", q, re.I):
        p, r = ints(pred), ints(ref)
        ok = bool(p and r and p[0] == r[0])
        return ok, float(ok), "count"
    ok = contains(pred, ref) or token_f1(pred, ref) >= 0.5
    return ok, max(token_f1(pred, ref), float(contains(pred, ref))), "match"


def score_audio(q, pred, ans):
    ref = ans["reference_answer"]
    if re.search(r"transcri", q, re.I):
        w = wer(pred, ans.get("transcript") or ref)
        return w <= 0.2, round(1 - min(w, 1), 3), "1-WER"
    if re.search(r"what (sound|noise)", q, re.I):
        ok = bool(content_stems(ref) & content_stems(pred))
        return ok, float(ok), "label overlap"
    r = content_stems(ref)
    rec = len(r & content_stems(pred)) / len(r) if r else 0.0
    return rec >= 0.5, round(rec, 3), "key-word recall"


ABSTAIN_RE = re.compile(r"does not say|doesn.t say|not (in|mentioned|stated|found)|no information|cannot find", re.I)


def score_text(q, pred, ans, abstained):
    ref = ans["reference_answer"]
    said_no = abstained or bool(ABSTAIN_RE.search(pred))
    if ans.get("answerable") is False:
        return said_no, float(said_no), "abstain"
    if said_no:
        return False, 0.0, "wrongly abstained"
    f1 = token_f1(pred, ref)
    ok = contains(pred, ref) or f1 >= 0.5
    return ok, max(f1, float(contains(pred, ref))), "match"


def score_video(q, pred, ans):
    ref = ans["reference_answer"]
    if re.search(r"how many|what number", q, re.I):
        # "one at a time" / "one by one" are not counts, and "t=0.8s" timestamps are not answers.
        def clean(s):
            s = re.sub(r"\bone (at a time|by one|after (another|the other))\b", " ", str(s), flags=re.I)
            return re.sub(r"\bt\s*=\s*\d+(\.\d+)?\s*s?\b|\b\d+\.\d+\s*s\b", " ", s, flags=re.I)
        r, p = ints(clean(ref)), ints(clean(pred))
        # The reference's final number is the answer ("5 appear, 2 disappear, so 3 are left"); the prediction
        # is judged by the number it leads with (any number appearing somewhere is not enough).
        ok = bool(r and p) and p[0] == r[-1]
        return ok, float(ok), "number"
    pl = pred.lower()
    ref_colors = sorted((m.start(), m.group(0)) for c in COLORS for m in re.finditer(rf"\b{c}\b", ref.lower()))
    asks_colour = bool(re.search(r"colou?r", q, re.I))
    if asks_colour and re.search(r"\bfirst\b", q, re.I) and re.search(r"\blast\b", q, re.I) and len(ref_colors) >= 2:
        # "What colour appears first and last?": the first and last colours must both be named, in that order.
        # The colours in between are not asked for, so they are not required.
        first, last = ref_colors[0][1], ref_colors[-1][1]
        pf, pz = re.search(rf"\b{first}\b", pl), re.search(rf"\b{last}\b", pl)
        ok = bool(pf and pz and pf.start() < pz.start())
        return ok, float(ok), "first/last in order"
    # Colours are required only when the question is about colour ("Which way does the square move?" is
    # answered by the direction, whether or not the answer repeats the square's colour).
    keys = [{c} for c in dict.fromkeys(c for _, c in ref_colors)] if asks_colour else []
    keys += [{w.lower()} for w in re.findall(r'"([^"]+)"', ref)]
    keys += [g for g in MOTION_GROUPS if any(re.search(rf"\b{w}\b", ref.lower()) for w in g)]
    if not keys:
        f1 = token_f1(pred, ref)
        return f1 >= 0.4, round(f1, 3), "token F1"
    hit = sum(any(re.search(rf"\b{re.escape(w)}\b", pl) for w in g) for g in keys)
    rec = hit / len(keys)
    return rec >= 0.67, round(rec, 3), "key-term recall"


def score_code(code: str, test_file: Path, timeout: float = 15.0):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "run_test.py"
        f.write_text(code + "\n\n" + test_file.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, 0.0, "unit tests (timeout)"
    ok = p.returncode == 0
    return ok, float(ok), "unit tests" if ok else "unit tests: " + (p.stderr.strip().splitlines() or ["fail"])[-1][:120]

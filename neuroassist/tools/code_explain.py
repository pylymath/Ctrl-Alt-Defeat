"""Explain code that is IN a document (or a .py file, or pasted in the question): find it, RUN it,
check the examples written in its docstring, then explain it using what really happened.

    "Explain this code and run it"   -> real output, examples passed/failed (with the actual values),
                                        the error and its line, then a line-by-line explanation
    "Fix this code"                  -> a corrected version from the coder model, checked against the
                                        SAME examples, so "it works now" is measured, not claimed

Everything the student sees about running the code (output, errors, pass/fail, actual values) comes
from actually running it, not from the model. The model only explains, and its line references are
checked against the real line numbers.

Note: the snippet runs in a separate Python process (isolated mode, temp folder, no input, 5 s limit).
That stops hangs and crashes; it is not a security sandbox, so only run code from your own files."""
import ast
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import config, llm
from ..ingest import load_document, sniff
from ..schema import Answer

MARK = "__PERCIE_EXAMPLE__"
FENCE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```", re.S | re.I)
PYLINE = re.compile(r"^\s*(def |class |import |from \w|for |while |if |elif |else:|try:|except|return\b|print\(|"
                    r"@|#|with |[A-Za-z_][\w\[\]., ]*\s*[+\-*/]?=[^=])")


# --- Finding the code -------------------------------------------------------------------------------
def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def find_snippets(text: str) -> list[dict]:
    """Code blocks in a document: ```fenced``` blocks first, else runs of lines that look like Python
    (and parse). Each: {code, line} with the 1-based line where it starts."""
    out = []
    for m in FENCE.finditer(text):
        code = m.group(1).strip("\n")
        if code.strip() and _parses(code):
            out.append({"code": code, "line": text[:m.start()].count("\n") + 2})
    if out:
        return out
    lines, i = text.splitlines(), 0
    while i < len(lines):
        if PYLINE.match(lines[i]) and not lines[i].strip().startswith("#"):
            j = i
            while j + 1 < len(lines) and (lines[j + 1].strip() == "" or lines[j + 1].startswith((" ", "\t"))
                                          or PYLINE.match(lines[j + 1])):
                j += 1
            block = "\n".join(lines[i:j + 1]).strip("\n")
            # drop a common indent (e.g. code indented 4 spaces inside a text document)
            ind = min((len(l) - len(l.lstrip()) for l in block.splitlines() if l.strip()), default=0)
            block = "\n".join(l[ind:] for l in block.splitlines())
            if len(block.splitlines()) >= 2 and _parses(block):
                out.append({"code": block, "line": i + 1})
            i = j + 1
        else:
            i += 1
    return out


def pick(snippets: list[dict], question: str) -> dict | None:
    """The snippet the question is about: one whose function name is mentioned, else the first."""
    for s in snippets:
        names = re.findall(r"^\s*def (\w+)", s["code"], re.M)
        if any(re.search(rf"\b{re.escape(n)}\b", question) for n in names):
            return s
    return snippets[0] if snippets else None


def code_from(question: str, files: list, session=None) -> tuple[dict | None, str]:
    """(snippet, where it came from) from a .py file, a document, or the question itself."""
    if files:
        path = files[0]
        name = Path(path).name
        text = (session.memo(path, "document", lambda: load_document(path)) if session else load_document(path))
        if sniff(path) == "code":
            return {"code": text.strip("\n"), "line": 1}, name
        s = pick(find_snippets(text), question)
        return s, (f"{name}, line {s['line']}" if s else name)
    s = pick(find_snippets(question), question)
    return s, "your question"


# --- Running it ---------------------------------------------------------------------------------------
def examples(code: str) -> list[tuple[str, str]]:
    """(expression, expected) pairs from '>>> expr' lines followed by an expected-value line."""
    lines = code.splitlines()
    out = []
    for i, raw in enumerate(lines):
        s = raw.strip()
        if s.startswith(">>>"):
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt and not nxt.startswith((">>>", '"""', "'''")):
                out.append((s[3:].strip(), nxt))
    return out


def _error_line(stderr: str, filename: str) -> int | None:
    hits = re.findall(rf'File ".*{re.escape(filename)}", line (\d+)', stderr)
    return int(hits[-1]) if hits else None


def run(code: str, timeout: float = 5.0) -> dict:
    """Run the snippet as a script, then (separately) its docstring examples.
    Returns {stdout, stderr, ok, error, error_line, ms, examples: [{expr, expected, actual, ok}]}."""
    res = {"stdout": "", "stderr": "", "ok": False, "error": "", "error_line": None, "ms": 0, "examples": []}
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "snippet.py"
        f.write_text(code, encoding="utf-8")
        t = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, text=True,
                               timeout=timeout, stdin=subprocess.DEVNULL)
            res.update(stdout=p.stdout[-4000:], stderr=p.stderr[-4000:], ok=p.returncode == 0)
            if p.returncode:
                last = (p.stderr.strip().splitlines() or ["crashed"])[-1]
                res.update(error=last, error_line=_error_line(p.stderr, "snippet.py"))
        except subprocess.TimeoutExpired:
            res.update(error=f"Still running after {timeout:.0f} s (an endless loop?), so it was stopped.")
        res["ms"] = round((time.perf_counter() - t) * 1000)

        ex = examples(code)
        if ex:
            # Define everything without running top-level prints/calls a second time: keep only
            # imports, defs and classes from the snippet.
            tree = ast.parse(code)
            defs = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                                           ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                                                           ast.AnnAssign))]
            lib = ast.unparse(ast.Module(body=defs, type_ignores=[]))
            h = [lib, "", "import sys"]
            for k, (expr, want) in enumerate(ex):
                h += ["try:",
                      f"    _a = ({expr})",
                      f"    _w = ({want})",
                      f"    print({MARK!r}, {k}, _a == _w, repr(_a))",
                      "except Exception as _e:",
                      f"    print({MARK!r}, {k}, False, type(_e).__name__ + ': ' + str(_e))"]
            g = Path(d) / "examples.py"
            g.write_text("\n".join(h), encoding="utf-8")
            try:
                q = subprocess.run([sys.executable, "-I", str(g)], cwd=d, capture_output=True, text=True,
                                   timeout=timeout, stdin=subprocess.DEVNULL)
                got = {}
                for line in q.stdout.splitlines():
                    if line.startswith(MARK):
                        _, k, ok, actual = line.split(" ", 3)
                        got[int(k)] = (ok == "True", actual)
            except subprocess.TimeoutExpired:
                got = {}
            res["examples"] = [{"expr": e, "expected": w, "ok": got.get(k, (False, ""))[0],
                                "actual": got.get(k, (False, "did not finish"))[1]} for k, (e, w) in enumerate(ex)]
    return res


# --- Explaining it ------------------------------------------------------------------------------------
EXPLAIN = {"type": "object", "required": ["summary", "steps", "issues"], "properties": {
    "summary": {"type": "string"},
    "steps": {"type": "array", "maxItems": 8, "items": {"type": "object", "required": ["lines", "what"],
                                                          "properties": {"lines": {"type": "string"},
                                                                         "what": {"type": "string"}}}},
    "issues": {"type": "array", "maxItems": 4, "items": {"type": "string"}}}}

SYSTEM = ("You explain Python code to a student with ADHD or dyslexia: short sentences, plain words, one idea "
          "per step. Base everything on the code and on the REAL run results you are given.")


def numbered(code: str) -> str:
    return "\n".join(f"{i:>3} | {l}" for i, l in enumerate(code.splitlines(), 1))


def _results_text(r: dict) -> str:
    parts = [f"Output:\n{r['stdout'] or '(nothing printed)'}"]
    if r["error"]:
        parts.append(f"Error: {r['error']}" + (f" (line {r['error_line']})" if r["error_line"] else ""))
    for e in r["examples"]:
        parts.append(f"Example {e['expr']}: expected {e['expected']}, got {e['actual']} -> "
                     f"{'pass' if e['ok'] else 'FAIL'}")
    return "\n".join(parts)


def explain(code: str, r: dict) -> dict:
    n = len(code.splitlines())
    x = llm.chat_json(config.CODE_MODEL,
                      f"Code (with line numbers):\n{numbered(code)}\n\nWhat happened when it was run:\n"
                      f"{_results_text(r)}\n\n'summary': what the code does, in one or two sentences. "
                      "'steps': walk through it in order; 'lines' = the line numbers (e.g. '3-5'), 'what' = what "
                      "those lines do. 'issues': real problems shown by the run results (failed examples, errors), "
                      "each saying which line causes it. Empty if it all works.",
                      EXPLAIN, system=SYSTEM, num_ctx=config.NUM_CTX_CODE, max_tokens=700)

    def valid(ref: str) -> bool:
        nums = [int(v) for v in re.findall(r"\d+", ref or "")]
        return bool(nums) and all(1 <= v <= n for v in nums)
    steps = [s for s in x.get("steps", []) if isinstance(s, dict) and s.get("what") and valid(s.get("lines", ""))]
    issues = [i for i in x.get("issues", []) if isinstance(i, str) and i.strip()]
    if not (r["error"] or any(not e["ok"] for e in r["examples"])):
        issues = []                 # the run showed no problem: don't let the model invent one
    return {"summary": (x.get("summary") or "").strip(), "steps": steps, "issues": issues}


# --- Fixing it ----------------------------------------------------------------------------------------
def fix(code: str, r: dict) -> tuple[str, dict]:
    """A corrected version from the coder model, run against the SAME examples. Returns (code, run)."""
    from .code import extract_code
    prompt = (f"This Python code has a problem.\n```python\n{code}\n```\nWhat happened when it was run:\n"
              f"{_results_text(r)}\n\nRewrite it so every example in the docstring passes. Keep the same function "
              "names, docstring and examples. Reply with the full corrected code only.")
    new = extract_code(llm.chat(config.CODE_MODEL, prompt, system="You are an expert Python programmer.",
                                num_ctx=config.NUM_CTX_CODE, max_tokens=1024))
    if not _parses(new):
        return new, {"stdout": "", "stderr": "", "ok": False, "error": "The fix is not valid Python.",
                     "error_line": None, "ms": 0, "examples": []}
    if not examples(new) and examples(code):          # model dropped the docstring: test with the originals
        new_run = run(new + "\n\n" + _examples_block(code))
        return new, new_run
    return new, run(new)


def _examples_block(code: str) -> str:
    ex = examples(code)
    body = "\n".join(f"    >>> {e}\n    {w}" for e, w in ex)
    return f'def _percie_examples():\n    """\n{body}\n    """\n'


# --- Tool entry point ------------------------------------------------------------------------------------
FIX_WORDS = re.compile(r"\b(fix|correct|repair|debug|make it work)\b", re.I)


def run_tool(question, files, route, session) -> Answer:
    s, where = code_from(question, files, session)
    if not s:
        return Answer("I couldn't find any Python code in that file. Put the code in a .py file, or in the "
                      "document between ``` marks.", abstain=True, evidence={"reason": "no code found"})
    code = s["code"]
    before = run(code)
    ev = {"code": code, "where": where, "run": before}
    if FIX_WORDS.search(question):
        new, after = fix(code, before)
        ev.update(fixed=new, fixed_run=after)
        ex = after["examples"]
        if ex:
            n_ok = sum(e["ok"] for e in ex)
            head = (f"Fixed: the corrected code passes {n_ok} of {len(ex)} examples." if n_ok == len(ex) else
                    f"I tried a fix, but it passes only {n_ok} of {len(ex)} examples.")
        else:
            head = "Here is a corrected version; it runs without errors." if after["ok"] else \
                "I tried a fix, but it still fails: " + after["error"]
        return Answer(head, evidence=ev, confidence=1.0 if ex and all(e["ok"] for e in ex) else 0.5)
    ev["explanation"] = explain(code, before)
    summary = ev["explanation"]["summary"] or "Here is what the code does and what happened when it ran."
    return Answer(summary, evidence=ev, confidence=None)

"""Write working Python from a prompt, then check it against the examples in its own docstring
and retry once with the error. Only the prompt's own examples are used (never the hidden tests).

Note: generated code runs in a separate process with a timeout. That stops hangs and crashes,
but it is not a security sandbox."""
import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import config, llm
from ..schema import Answer

SYSTEM = ("You are an expert Python programmer. Reply with ONE complete Python code block that contains "
          "the full function, including its signature and any imports it needs. No explanation, no tests, "
          "no example usage.")


def entry_point(prompt: str) -> str | None:
    names = re.findall(r"^def (\w+)\s*\(", prompt, re.M)
    return names[-1] if names else None


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return (max(blocks, key=len) if blocks else text).strip()


def _complete_stub(prompt_code: str, code: str, name: str) -> str:
    """If the model returned only a function body, graft it onto the prompt's signature."""
    if f"def {name}" in code:
        return code
    body = "\n".join("    " + line if line.strip() else line for line in code.splitlines())
    return prompt_code.rstrip() + "\n" + body


def docstring_examples(prompt: str, name: str) -> list[str]:
    """Turn the examples written in the prompt into assert statements."""
    lines, asserts = prompt.splitlines(), []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith(">>>"):
            expr = line[3:].strip()
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt and not nxt.startswith(">>>") and not nxt.startswith('"""') and not nxt.startswith("'''"):
                asserts.append(f"assert ({expr}) == ({nxt}), {expr!r}")
            elif "==" in expr:
                asserts.append(f"assert {expr}, {expr!r}")
            continue
        m = re.match(rf"^({re.escape(name)}\(.*\))\s*(?:==>|=>|➞|->|==|should return|returns?)\s*(.+?)\.?$", line)
        if m:
            asserts.append(f"assert ({m.group(1)}) == ({m.group(2)}), {m.group(1)!r}")
    valid = []
    for a in asserts:
        try:
            ast.parse(a)
            valid.append(a)
        except SyntaxError:
            pass
    return valid


def run_checks(code: str, checks: list[str], timeout: float = 5.0) -> tuple[int, int, str]:
    """Returns (passed, total, first error)."""
    if not checks:
        return 0, 0, ""
    harness = [code, "", "import sys", "_passed, _errors = 0, []"]
    for c in checks:
        harness += ["try:", f"    {c}", "    _passed += 1", "except Exception as e:",
                    "    _errors.append(f'{type(e).__name__}: {e}')"]
    harness += [f"print(_passed, {len(checks)})", "print(_errors[0] if _errors else '')"]
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "check.py"
        f.write_text("\n".join(harness), encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0, len(checks), f"timed out after {timeout:.0f}s (infinite loop?)"
    if p.returncode != 0:
        return 0, len(checks), (p.stderr.strip().splitlines() or ["crashed"])[-1]
    out = p.stdout.strip().splitlines()
    passed, total = (int(x) for x in out[0].split())
    return passed, total, out[1] if len(out) > 1 else ""


def write_code(question: str) -> Answer:
    name = entry_point(question) or ""
    start = re.search(r"^(from |import |def )", question, re.M)   # signature (+ imports) from the prompt
    stub = question[start.start():] if start else ""
    checks = docstring_examples(question, name) if name else []
    prompt, attempts = question, []
    for attempt in range(config.CODE_RETRIES + 1):
        code = extract_code(llm.chat(config.CODE_MODEL, prompt, system=SYSTEM,
                                     num_ctx=config.NUM_CTX_CODE, max_tokens=1024))
        if name:
            code = _complete_stub(stub, code, name)
        passed, total, err = run_checks(code, checks)
        attempts.append({"passed": passed, "total": total, "error": err})
        if not err:
            break
        prompt = (f"{question}\n\nThis attempt failed a check from the docstring:\n```python\n{code}\n```\n"
                  f"Error: {err}\nWrite the corrected full function.")
    ev = {"entry_point": name, "examples_passed": passed, "examples_total": total, "attempts": attempts}
    return Answer(code, evidence=ev, confidence=(passed / total) if total else None)


def run(question, files, route, session) -> Answer:
    if files:   # a code file was attached: put it in front of the question
        question = f"```python\n{Path(files[0]).read_text(encoding='utf-8', errors='replace')}\n```\n\n{question}"
    return write_code(question)


def run_chat(question, files, route, session) -> Answer:
    """No file attached. Code requests go to the coder; anything else to the main model."""
    if route.task == "write_code":
        return write_code(question)
    return Answer(llm.chat(config.TEXT_MODEL, question, num_ctx=config.NUM_CTX_DOC, max_tokens=400).strip())

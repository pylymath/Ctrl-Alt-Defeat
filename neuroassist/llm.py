"""Thin Ollama wrapper: fixed options, per-model keep_alive, JSON-schema output, timing."""
import json
import re
import time

import ollama

from . import config

_client = ollama.Client(host=config.OLLAMA_HOST)
_rejects_think: set[str] = set()   # models that error on the `think` flag


def chat(model: str, prompt: str, *, images=None, system: str | None = None, schema: dict | None = None,
         num_ctx: int = 2048, temperature: float = 0.0, max_tokens: int | None = None) -> str:
    """One-shot chat. `images` is a list of JPEG bytes or file paths, in order."""
    msgs = [{"role": "system", "content": system}] if system else []
    user = {"role": "user", "content": prompt}
    if images:
        user["images"] = list(images)
    msgs.append(user)
    opts = {"temperature": temperature, "num_ctx": num_ctx}
    if max_tokens:
        opts["num_predict"] = max_tokens
    kw = dict(model=model, messages=msgs, options=opts, keep_alive=config.keep_alive(model))
    if schema:
        kw["format"] = schema
    if model not in _rejects_think:
        kw["think"] = False      # Qwen3 family: skip hidden reasoning, it is slow on CPU
    try:
        r = _client.chat(**kw)
    except ollama.ResponseError as e:
        if "think" in kw and "think" in str(e).lower():
            _rejects_think.add(model)
            kw.pop("think")
            r = _client.chat(**kw)
        elif e.status_code == 404:
            raise RuntimeError(f"Model '{model}' is not pulled. Run: ollama pull {model}") from e
        else:
            raise
    text = r.message.content or ""
    # Some Ollama versions put the whole reply in `thinking` (and leave `content` empty) when
    # think=False is combined with a JSON format, notably for Qwen3-VL. That text IS the answer.
    if not text.strip() and kw.get("think") is False:
        text = getattr(r.message, "thinking", None) or ""
    return text


def chat_json(model: str, prompt: str, schema: dict, **kw) -> dict:
    """Chat constrained to `schema`. Falls back to extracting the first {...} block."""
    text = chat(model, prompt, schema=schema, **kw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def warm(model: str) -> float:
    """Load a model into memory ahead of time (call at app start). Returns seconds taken."""
    t = time.perf_counter()
    _client.generate(model=model, prompt="", keep_alive=config.keep_alive(model))
    return time.perf_counter() - t


def installed_models() -> set[str]:
    try:
        return {m.model for m in _client.list().models}
    except Exception:
        return set()

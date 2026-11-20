"""Thin Ollama wrapper with the harness every model call goes through.

    fixed options      temperature 0, one pinned context size per model (no reloads), keep_alive
    JSON schemas       structured output, with a fallback that extracts the first {...} block
    empty replies      retried (EMPTY_RETRIES) instead of being shown as an answer
    timeouts           LLM_TIMEOUT seconds per call, reported as a clear error
    events             retries / timeouts / fallbacks are recorded so the UI can show them

`events` is a plain list the pipeline reads around each step ("How I got this answer")."""
import json
import re
import time

import httpx
import ollama

from . import config

_client = ollama.Client(host=config.OLLAMA_HOST, timeout=config.LLM_TIMEOUT)
_rejects_think: set[str] = set()   # models that error on the `think` flag
events: list[dict] = []            # harness events, newest last


def note(kind: str, model: str = "", detail: str = ""):
    events.append({"t": time.time(), "kind": kind, "model": model, "detail": detail})
    del events[:-200]


class ModelTimeout(RuntimeError):
    pass


def _call(kw: dict):
    model = kw["model"]
    try:
        return _client.chat(**kw)
    except ollama.ResponseError as e:
        if "think" in kw and "think" in str(e).lower():
            _rejects_think.add(model)
            kw.pop("think")
            return _client.chat(**kw)
        if e.status_code == 404:
            raise RuntimeError(f"Model '{model}' is not pulled. Run: ollama pull {model}") from e
        raise
    except (httpx.TimeoutException, TimeoutError) as e:
        note("timeout", model, f"no reply within {config.LLM_TIMEOUT:.0f}s")
        raise ModelTimeout(f"{model} did not answer within {config.LLM_TIMEOUT:.0f} s. "
                           "Try a shorter file or ask again.") from e


def _text(r, kw) -> str:
    text = r.message.content or ""
    # Some Ollama versions put the whole reply in `thinking` (and leave `content` empty) when
    # think=False is combined with a JSON format, notably for Qwen3-VL. That text IS the answer.
    if not text.strip() and kw.get("think") is False:
        text = getattr(r.message, "thinking", None) or ""
    return text


def chat(model: str, prompt: str, *, images=None, system: str | None = None, schema: dict | None = None,
         num_ctx: int = 2048, temperature: float = 0.0, max_tokens: int | None = None,
         retries: int | None = None) -> str:
    """One-shot chat. `images` is a list of JPEG bytes or file paths, in order. An empty reply is
    retried `retries` times (default EMPTY_RETRIES), the retry with a little temperature so it
    does not repeat the same empty decode."""
    msgs = [{"role": "system", "content": system}] if system else []
    user = {"role": "user", "content": prompt}
    if images:
        user["images"] = list(images)
    msgs.append(user)
    opts = {"temperature": temperature, "num_ctx": config.pinned_ctx(model, num_ctx)}
    if max_tokens:
        opts["num_predict"] = max_tokens
    kw = dict(model=model, messages=msgs, options=opts, keep_alive=config.keep_alive(model))
    if schema:
        kw["format"] = schema
    if model not in _rejects_think:
        kw["think"] = False      # Qwen3 family: skip hidden reasoning, it is slow and eats the token budget
    retries = config.EMPTY_RETRIES if retries is None else retries
    text = ""
    for attempt in range(retries + 1):
        r = _call(kw)
        text = _text(r, kw)
        if text.strip():
            if attempt:
                note("retry_ok", model, f"empty reply, answered on retry {attempt}")
            return text
        note("empty", model, f"empty reply (done_reason={getattr(r, 'done_reason', '?')})")
        kw["options"] = {**opts, "temperature": 0.3, "seed": 7 + attempt}
    return text


def _has_required(d: dict, schema: dict) -> bool:
    return all(k in d for k in schema.get("required", []))


def chat_json(model: str, prompt: str, schema: dict, **kw) -> dict:
    """Chat constrained to `schema`. Falls back to extracting the first {...} block. A reply that
    is not valid JSON or misses required fields is retried once before returning {}."""
    retries = kw.pop("retries", config.EMPTY_RETRIES)
    for attempt in range(retries + 1):
        text = chat(model, prompt, schema=schema, retries=0 if attempt else retries, **kw)
        d = _parse(text)
        if d and _has_required(d, schema):
            if attempt:
                note("retry_ok", model, "invalid JSON, fixed on retry")
            return d
        if attempt < retries:
            note("bad_json", model, (text or "")[:80] or "empty")
            kw["temperature"] = 0.3
    return d or {}


def _parse(text: str) -> dict:
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                return d if isinstance(d, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def warm(model: str) -> float:
    """Load a model into memory ahead of time (call at app start). Returns seconds taken."""
    t = time.perf_counter()
    _client.generate(model=model, prompt="", keep_alive=config.keep_alive(model),
                     options={"num_ctx": config.pinned_ctx(model, 0)})
    return time.perf_counter() - t


def installed_models() -> set[str]:
    try:
        return {m.model for m in _client.list().models}
    except Exception:
        return set()


def loaded() -> list[dict]:
    """Models Ollama has in memory right now: [{name, vram_gb, ram_gb, ctx}]. Cheap HTTP call."""
    try:
        out = []
        for m in _client.ps().models:
            size, vram = (m.size or 0) / 2**30, (m.size_vram or 0) / 2**30
            out.append({"name": m.model, "vram_gb": round(vram, 1), "ram_gb": round(max(size - vram, 0), 1),
                        "ctx": getattr(m, "context_length", None)})
        return out
    except Exception:
        return []

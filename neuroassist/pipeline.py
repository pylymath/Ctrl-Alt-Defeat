"""answer(question, files) -> Answer. The one function the UI and the eval both call."""
import time
from typing import Callable

from . import router
from .ingest import file_hash
from .schema import Answer


class Session:
    """Per-conversation cache. Follow-up questions about the same file reuse its transcript,
    frames or document text instead of reprocessing it."""

    def __init__(self):
        self._cache: dict = {}

    def memo(self, path, name: str, fn: Callable, *params):
        key = (file_hash(path), name, params)
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]


_default_session = Session()


def _handlers():
    from .tools import audio, code, image, text_qa, video
    return {"image": image.run, "video": video.run, "audio": audio.run,
            "document": text_qa.run, "code": code.run, "chat": code.run_chat}


def answer(question: str, files: list | tuple = (), session: Session | None = None) -> Answer:
    session = session or _default_session
    t0 = time.perf_counter()
    r = router.route(question, files)
    t_route = time.perf_counter()
    ans = _handlers()[r.kind](question, list(files), r, session)
    ans.route = r
    for m in r.modifiers:          # output steps; implemented in the accessibility layer (next)
        ans.notes.append(f"modifier '{m}' requested but not implemented yet")
    ans.timings_ms["route"] = round((t_route - t0) * 1000, 1)
    ans.timings_ms["total"] = round((time.perf_counter() - t0) * 1000, 1)
    return ans

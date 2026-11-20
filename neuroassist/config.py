"""All tunables in one place. Each value comes from, in order:
    1. an environment variable NEUROASSIST_<NAME>   (the eval uses these to compare settings)
    2. neuroassist.env in the repo root             (written by install.ps1 / install.sh; git-ignored)
    3. the default below
so a fresh machine needs no global changes (no setx)."""
import os
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parents[1] / "neuroassist.env"


def _read_settings(path: Path = SETTINGS_FILE) -> dict[str, str]:
    out = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"')
    except FileNotFoundError:
        pass
    return out


_SETTINGS = _read_settings()


def _get(name: str):
    """NEUROASSIST_<name> from the environment, else from neuroassist.env. Empty means unset."""
    v = os.environ.get("NEUROASSIST_" + name)
    if v in (None, ""):                     # an empty variable (e.g. setx NAME "") falls through to the file
        v = _SETTINGS.get("NEUROASSIST_" + name)
    return v if v not in (None, "") else None


def _total_ram_gb() -> float:
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "aev")]

        m = _MemStatus()
        m.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.total / 2**30
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except Exception:
            return 16.0


# "lite" = CPU-only / <12 GB RAM laptops. "standard" = a 6 GB+ GPU or 16 GB+ RAM.
PROFILES = {
    "lite": dict(vision="qwen3-vl:2b-instruct", whisper="base", video_max_frames=8, video_side=336,
                 image_side=896, video_fps_dense=2.0, video_fps_sparse=0.5),
    "standard": dict(vision="qwen3-vl:4b-instruct", whisper="small", video_max_frames=16, video_side=448,
                     image_side=1280, video_fps_dense=4.0, video_fps_sparse=1.0),
}
PROFILE = _get("PROFILE") or ("lite" if _total_ram_gb() < 12 else "standard")
_P = PROFILES[PROFILE]


def _env(name, default, cast=str):
    v = _get(name)
    return cast(v) if v is not None else default


# --- Models -------------------------------------------------------------------------------
VISION_MODEL = _env("VISION_MODEL", _P["vision"])          # images + video, stays loaded
TEXT_MODEL = _env("TEXT_MODEL", VISION_MODEL)               # documents, transcripts, cards, router
# Grounded question answering only (strict JSON: found / quote / answer). Point it at a QA fine-tune,
# e.g. NEUROASSIST_QA_MODEL=neuro-doc-qa:latest. Summaries, cards, video timelines and chat keep using
# TEXT_MODEL, because a QA-only fine-tune is bad at everything else.
QA_MODEL = _env("QA_MODEL", TEXT_MODEL)
CODE_MODEL = _env("CODE_MODEL", "qwen2.5-coder:3b")         # swapped in only for code
# Writes the accessible summary cards. Point it at the Colab fine-tune once imported into Ollama,
# e.g. NEUROASSIST_CARD_MODEL=neuro-assist. Defaults to the main model (no swap).
CARD_MODEL = _env("CARD_MODEL", TEXT_MODEL)
WHISPER_MODEL = _env("WHISPER_MODEL", _P["whisper"])        # faster-whisper size, CPU int8
SOUND_MODEL = _env("SOUND_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593")  # AudioSet, not ESC-50
OLLAMA_HOST = _env("OLLAMA_HOST", "http://127.0.0.1:11434")

RESIDENT_KEEP_ALIVE = _env("RESIDENT_KEEP_ALIVE", "30m")
SWAPPED_KEEP_ALIVE = _env("SWAPPED_KEEP_ALIVE", "0")


def keep_alive(model: str):
    """Main model stays loaded; anything else is unloaded right after use."""
    return RESIDENT_KEEP_ALIVE if model in (VISION_MODEL, TEXT_MODEL, QA_MODEL) else int(SWAPPED_KEEP_ALIVE)


# --- Context sizes (tokens). Smaller = less memory and faster prompt processing. ---------------
NUM_CTX_ROUTER = _env("NUM_CTX_ROUTER", 1024, int)
NUM_CTX_DOC = _env("NUM_CTX_DOC", 4096, int)
NUM_CTX_IMAGE = _env("NUM_CTX_IMAGE", 4096, int)
NUM_CTX_VIDEO = _env("NUM_CTX_VIDEO", 12288, int)   # 16 frames of Qwen3-VL need ~7.5k
NUM_CTX_CODE = _env("NUM_CTX_CODE", 4096, int)


def pinned_ctx(model: str, requested: int) -> int:
    """Ollama reloads a model whenever num_ctx changes. The vision model is called with 1k (router),
    4k (images, documents) and 12k (video) contexts, so without this every switch between a photo
    and a video cost a full reload. Each model gets one fixed context: the largest it is used with."""
    if not PIN_CTX:
        return requested
    if model in (VISION_MODEL, TEXT_MODEL):
        return max(requested, NUM_CTX_VIDEO if model == VISION_MODEL else NUM_CTX_DOC)
    return max(requested, NUM_CTX_DOC)


# --- Harness -----------------------------------------------------------------------------------
PIN_CTX = _env("PIN_CTX", "1") == "1"
LLM_TIMEOUT = _env("LLM_TIMEOUT", 180.0, float)     # seconds per model call before giving up
EMPTY_RETRIES = _env("EMPTY_RETRIES", 1, int)        # retries when a model returns nothing usable
TRACK_VRAM = _env("TRACK_VRAM", "1") == "1"          # record GPU memory per step (Ollama /api/ps)

# --- Behaviour switches the eval compares ----------------------------------------------------
COUNT_MODE = _env("COUNT_MODE", "boxes")          # "boxes" (count boxes in code) | "direct"
VIDEO_MODE = _env("VIDEO_MODE", "auto")           # "direct" | "timeline" | "auto"
VIDEO_DIRECT_MAX_FRAMES = _env("VIDEO_DIRECT_MAX_FRAMES", 8, int)
VIDEO_MAX_FRAMES = _env("VIDEO_MAX_FRAMES", _P["video_max_frames"], int)
VIDEO_SIDE = _env("VIDEO_SIDE", _P["video_side"], int)
VIDEO_FPS_DENSE = _env("VIDEO_FPS_DENSE", _P["video_fps_dense"], float)
VIDEO_FPS_SPARSE = _env("VIDEO_FPS_SPARSE", _P["video_fps_sparse"], float)
IMAGE_SIDE = _env("IMAGE_SIDE", _P["image_side"], int)
CODE_RETRIES = _env("CODE_RETRIES", 1, int)
NO_SPEECH_THRESHOLD = _env("NO_SPEECH_THRESHOLD", 0.6, float)
SOUND_CONFIDENT = _env("SOUND_CONFIDENT", 0.30, float)
ROUTER_LLM_FALLBACK = _env("ROUTER_LLM_FALLBACK", "1") == "1"

# --- Data (never inside the repo) --------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(_env("DATA", str(REPO_ROOT.parent / "archive (1)")))

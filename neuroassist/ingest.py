"""Detect input type from file bytes and normalize every input before any model sees it."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sniff(path: str | Path) -> str:
    """Input type from the file's first bytes, falling back to the extension."""
    p = Path(path)
    h = p.read_bytes()[:16]
    if h[4:8] == b"ftyp" or h[:4] == b"\x1aE\xdf\xa3":          # mp4/mov, webm/mkv
        return "video"
    if h[:4] == b"RIFF":                                          # WAV / WebP / AVI share RIFF
        return {b"WAVE": "audio", b"WEBP": "image"}.get(h[8:12], "video")
    if h[:3] == b"ID3" or h[:4] in (b"fLaC", b"OggS") or h[:2] == b"\xff\xfb":
        return "audio"
    if h[:3] == b"\xff\xd8\xff" or h[:4] == b"\x89PNG" or h[:6] in (b"GIF87a", b"GIF89a"):
        return "image"
    if h[:4] == b"%PDF":
        return "document"
    if p.suffix.lower() in (".py", ".js", ".ts", ".java", ".c", ".cpp", ".ipynb"):
        return "code"
    return "document"


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# --- Documents ------------------------------------------------------------------------------
def load_document(path: str | Path) -> str:
    p = Path(path)
    if p.read_bytes()[:4] == b"%PDF":
        from pypdf import PdfReader
        text = "\n\n".join((page.extract_text() or "") for page in PdfReader(str(p)).pages).strip()
        if not text:
            raise ValueError("PDF has no text layer (scanned). TODO: render pages and send to the vision model.")
        return text
    return p.read_text(encoding="utf-8", errors="replace")


# --- Images ---------------------------------------------------------------------------------
def _resize(img, side: int):
    import cv2
    h, w = img.shape[:2]
    s = side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def _jpeg(img, quality: int = 90) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


def prepare_image(path: str | Path, side: int) -> bytes:
    """Decode any image, cap the long side (fewer image tokens = faster), re-encode as JPEG."""
    import cv2
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)   # unicode-path safe
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return _jpeg(_resize(img, side))


# --- Video ----------------------------------------------------------------------------------
@dataclass
class Frame:
    t: float          # seconds from start
    jpeg: bytes


def extract_frames(path: str | Path, fps: float, max_frames: int, side: int,
                   dedupe: float = 1.5) -> tuple[list[Frame], float]:
    """Sample at `fps`, always keep first and last frame, drop near-duplicates, cap the count.
    Streams the video so a 30 s 720p clip never sits in RAM whole."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(native / fps))
    picked, last, i = [], None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        last = (i, f)
        if i % step == 0:
            picked.append((i, _resize(f, side)))
        i += 1
    cap.release()
    if last is None:
        raise ValueError(f"Could not read video: {path}")
    if picked[-1][0] != last[0]:
        picked.append((last[0], _resize(last[1], side)))
    duration = (last[0] + 1) / native

    kept, prev = [], None
    for j, (idx, img) in enumerate(picked):
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
        is_last = j == len(picked) - 1
        if prev is not None and not is_last and np.abs(small - prev).mean() < dedupe:
            continue
        kept.append((idx, img))
        prev = small
    if len(kept) > max_frames:
        sel = sorted(set(np.linspace(0, len(kept) - 1, max_frames).round().astype(int)))
        kept = [kept[s] for s in sel]
    return [Frame(t=idx / native, jpeg=_jpeg(img)) for idx, img in kept], duration


# --- Audio ----------------------------------------------------------------------------------
def load_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Mono float32 at 16 kHz. Uses faster-whisper's decoder (any format) if available."""
    try:
        from faster_whisper import decode_audio
        return decode_audio(str(path), sampling_rate=sr)
    except ImportError:
        import wave
        with wave.open(str(path)) as w:
            ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(w.getnframes())
        a = np.frombuffer(raw, {1: np.uint8, 2: np.int16, 4: np.int32}[width]).astype(np.float32)
        a = (a - 128) / 128 if width == 1 else a / float(2 ** (8 * width - 1))
        if ch > 1:
            a = a.reshape(-1, ch).mean(1)
        if rate != sr:
            a = np.interp(np.arange(0, len(a) * sr / rate) * rate / sr, np.arange(len(a)), a)
        return a.astype(np.float32)

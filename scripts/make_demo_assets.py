"""Builds the demo pack in assets/demo/ from scratch (our own content, never practice-pack data).

    python scripts/make_demo_assets.py

    assignment.txt   written by hand (already in the folder)
    diagram.png      a whiteboard-style flow diagram of the lab's pipeline       -> image
    voice_note.wav   the TA's spoken instruction, then a chime                   -> audio (speech + sound)
    process.mp4      a moving-average window sliding along a row of readings     -> video
"""
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "assets" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, center, scale=0.9, color=(40, 40, 40), thick=2):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    cv2.putText(img, s, (int(center[0] - w / 2), int(center[1] + h / 2)), FONT, scale, color, thick, cv2.LINE_AA)


def diagram():
    img = np.full((620, 1100, 3), 250, np.uint8)
    rng = np.random.default_rng(0)                     # faint whiteboard smudges
    for _ in range(40):
        c = tuple(int(x) for x in rng.integers(225, 245, 3))
        cv2.circle(img, (int(rng.integers(0, 1100)), int(rng.integers(0, 620))), int(rng.integers(20, 80)), c, -1)
    img = cv2.GaussianBlur(img, (41, 41), 0)
    _text(img, "Lab 4 pipeline", (550, 50), 1.2, (30, 30, 30), 3)
    boxes = [("Read CSV", (40, 140)), ("Remove missing", (300, 140)), ("Moving average", (580, 140)),
             ("Plot raw + smooth", (840, 140))]
    colors = [(200, 120, 40), (60, 140, 220), (60, 170, 80), (170, 80, 170)]   # BGR: blue, orange, green, purple
    for (label, (x, y)), col in zip(boxes, colors):
        cv2.rectangle(img, (x, y), (x + 220, y + 100), col, 4, cv2.LINE_AA)
        _text(img, label, (x + 110, y + 50), 0.75, col, 2)
    for i in range(3):
        x0 = boxes[i][1][0] + 220
        cv2.arrowedLine(img, (x0 + 5, 190), (boxes[i + 1][1][0] - 5, 190), (50, 50, 50), 3, cv2.LINE_AA, tipLength=0.25)
    # Loop back from the plot to the window size: "try another window"
    cv2.rectangle(img, (580, 380), (800, 470), (40, 40, 200), 4, cv2.LINE_AA)
    _text(img, "Window size", (690, 425), 0.8, (40, 40, 200), 2)
    cv2.arrowedLine(img, (690, 380), (690, 245), (40, 40, 200), 3, cv2.LINE_AA, tipLength=0.08)
    cv2.line(img, (950, 245), (950, 520), (50, 50, 50), 3, cv2.LINE_AA)
    cv2.line(img, (950, 520), (690, 520), (50, 50, 50), 3, cv2.LINE_AA)
    cv2.arrowedLine(img, (690, 520), (690, 475), (50, 50, 50), 3, cv2.LINE_AA, tipLength=0.4)
    _text(img, "try another window", (820, 555), 0.7, (50, 50, 50), 2)
    _text(img, "tests check this step", (560, 300), 0.6, (60, 170, 80), 2)
    cv2.imwrite(str(OUT / "diagram.png"), img)


def video(fps=10):
    vals = [21, 23, 22, 26, 24, 25, 29, 27]
    w, h = 960, 400
    path = OUT / "process.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cell = 100
    x0 = (w - cell * len(vals)) // 2

    def base(highlight=None, results=()):
        img = np.full((h, w, 3), 255, np.uint8)
        _text(img, "Moving average, window = 3", (w // 2, 40), 1.0, (30, 30, 30), 2)
        for i, v in enumerate(vals):
            x = x0 + i * cell
            cv2.rectangle(img, (x, 110), (x + cell, 190), (90, 90, 90), 2)
            _text(img, str(v), (x + cell // 2, 150), 1.0)
        for j, r in enumerate(results):
            x = x0 + (j + 1) * cell
            cv2.rectangle(img, (x, 270), (x + cell, 340), (60, 170, 80), -1)
            _text(img, f"{r:.1f}", (x + cell // 2, 305), 0.9, (255, 255, 255), 2)
        if highlight is not None:
            x = x0 + highlight * cell
            cv2.rectangle(img, (x - 4, 104), (x + 3 * cell + 4, 196), (40, 40, 220), 5)
        return img

    for _ in range(fps):                                   # 1 s: just the readings
        vw.write(base())
    results = []
    for s in range(len(vals) - 2):                         # window slides right, one result each step
        for _ in range(fps // 2):
            vw.write(base(s, results))
        results.append(sum(vals[s:s + 3]) / 3)
        for _ in range(fps // 2):
            vw.write(base(s, results))
    for _ in range(int(1.5 * fps)):                        # end: all results, no window
        vw.write(base(None, results))
    vw.release()


def _tts_wav(text: str, path: Path):
    text = text.replace("'", "''")          # PowerShell single-quoted string: ' is written as ''
    if sys.platform != "win32":
        raise SystemExit("voice note needs Windows speech (or record one yourself as assets/demo/voice_note.wav)")
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$s.Rate = -1; $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{path}', $fmt); $s.Speak('{text}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=True)


def _read16(path: Path) -> np.ndarray:
    with wave.open(str(path)) as f:
        return np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768


def voice_note(sr=16000):
    with tempfile.TemporaryDirectory() as d:
        speech_path = Path(d) / "speech.wav"
        _tts_wav("Hi everyone, quick note about lab four. Remove the rows with missing temperatures before you "
                 "compute the moving average, not after. And start with a window size of three. "
                 "One change: because of the holiday, lab four is now due on Thursday at 5 pm. "
                 "When you hear the bell, the lab is open for questions.", speech_path)
        speech = _read16(speech_path)
    t = np.arange(int(2.5 * sr)) / sr
    bell = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((880, 1.0), (1760, 0.5), (2640, 0.25), (3520, 0.12)))
    bell = bell * np.exp(-2.2 * t)
    bell = np.concatenate([bell, 0.8 * np.roll(bell, 0)[: len(bell)]])     # ding-ding
    audio = np.concatenate([speech, np.zeros(int(0.6 * sr)), 0.6 * bell / np.abs(bell).max(), np.zeros(sr // 2)])
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(OUT / "voice_note.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


LECTURE = (
    "Okay, let's get started. Today is about smoothing noisy data, which you need for lab four. "
    "Sensors are noisy. If you plot raw temperature readings, the line jumps around even when the real "
    "temperature barely changes. A moving average fixes that. You take a window of readings, say three, "
    "average them, then slide the window one step to the right and do it again. "
    "So the window size is the key choice. A small window follows the data closely but stays noisy. "
    "A large window is very smooth, but it reacts late to real changes. That trade-off is exactly what "
    "your reflection should explain. "
    "Two practical things. First, clean the data before you smooth it. Missing values will break the "
    "average, so drop those rows first. Second, when the window is bigger than the list, return an empty "
    "list instead of crashing. Write a test for that case. "
    "One more thing. For the plot, put the raw readings and the smoothed line on the same chart, with a "
    "legend, so I can compare them. "
    "Next week we will look at weighted moving averages, where recent readings count more. That is the "
    "bonus part of the lab, and it is optional. "
    "That's it for today. Office hours are on Wednesday afternoon if you get stuck."
)


def lecture(sr=16000):
    """About 90 seconds of 'lecture' with instructions spread through it, for 'catch me up from 0:45'."""
    _tts_wav(LECTURE, OUT / "lecture.wav")


if __name__ == "__main__":
    diagram()
    video()
    voice_note()
    lecture()
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:<16} {p.stat().st_size / 1024:8.1f} KB")

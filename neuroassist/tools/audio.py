"""Audio: speech goes to Whisper, everything else to an AudioSet sound classifier.
Whisper alone invents words on rain or a chainsaw, so the two paths are kept separate.
Both run on the CPU so the GPU stays free for the language model."""
from .. import config
from ..ingest import load_audio
from ..schema import Answer
from .text_qa import answer_from_text

_whisper = None
_sound = None


def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper


def _get_sound():
    global _sound
    if _sound is None:
        from transformers import pipeline
        _sound = pipeline("audio-classification", model=config.SOUND_MODEL, device="cpu")
    return _sound


def transcribe(path) -> dict:
    segs, _info = _get_whisper().transcribe(str(path), beam_size=5, vad_filter=True, language="en")
    segs = [s for s in segs if s.text.strip()]
    speech = [s for s in segs if s.no_speech_prob < config.NO_SPEECH_THRESHOLD]
    return {"text": " ".join(s.text.strip() for s in speech),
            "segments": [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in speech],
            "has_speech": bool(speech)}


def classify_sound(path) -> list[dict]:
    preds = _get_sound()({"raw": load_audio(path), "sampling_rate": 16000}, top_k=5)
    return [{"label": p["label"], "score": round(float(p["score"]), 3)} for p in preds]


def _sound_answer(labels: list[dict]) -> Answer:
    if not labels:
        return Answer("I could not identify the sound.", confidence=0.0)
    top = labels[0]
    if top["score"] >= config.SOUND_CONFIDENT or len(labels) < 2:
        return Answer(f"{top['label']}.", evidence={"sounds": labels}, confidence=top["score"])
    return Answer(f"It sounds like {top['label'].lower()} or {labels[1]['label'].lower()}.",
                  evidence={"sounds": labels}, confidence=top["score"])


def run(question, files, route, session) -> Answer:
    path = files[0]
    if route.task == "identify_sound":
        return _sound_answer(session.memo(path, "sounds", lambda: classify_sound(path)))
    tr = session.memo(path, "transcript", lambda: transcribe(path), config.WHISPER_MODEL)
    if not tr["has_speech"]:           # no speech found: answer from what the clip sounds like
        ans = _sound_answer(session.memo(path, "sounds", lambda: classify_sound(path)))
        ans.notes.append("no speech detected; answered with the sound classifier")
        return ans
    if route.task == "transcribe":
        return Answer(tr["text"], evidence={"segments": tr["segments"]})
    ans = answer_from_text(question, tr["text"], source="recording")
    ans.evidence["segments"] = tr["segments"]
    return ans

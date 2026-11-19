"""Run ONCE while online on the demo laptop:  python -m neuroassist.prefetch
Downloads the Whisper and sound-classifier weights into the local cache, checks the Ollama models
are pulled, and loads the main model. After this, everything works with Wi-Fi off."""
import time

from . import config, llm


def main():
    print(f"profile={config.PROFILE}")
    have = llm.installed_models()
    for m in sorted({config.VISION_MODEL, config.TEXT_MODEL, config.CODE_MODEL, config.CARD_MODEL}):
        print(f"  ollama {m:<28} {'OK' if m in have else 'MISSING -> ollama pull ' + m}")

    t = time.perf_counter()
    from faster_whisper import WhisperModel
    WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    print(f"  whisper {config.WHISPER_MODEL:<27} OK ({time.perf_counter() - t:.0f}s)")

    t = time.perf_counter()
    from transformers import pipeline
    pipeline("audio-classification", model=config.SOUND_MODEL, device="cpu")
    print(f"  sound   {config.SOUND_MODEL:<27} OK ({time.perf_counter() - t:.0f}s)")

    if config.VISION_MODEL in have:
        print(f"  warm    {config.VISION_MODEL:<27} loaded in {llm.warm(config.VISION_MODEL):.1f}s")


if __name__ == "__main__":
    main()

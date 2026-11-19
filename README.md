# NeuroAssist

Offline multimodal assistant for students with ADHD and dyslexia. One interface for photos,
audio, documents, code and short videos; every model runs locally through Ollama, with no internet.

**Status:** core pipeline, router and eval script done. UI, accessibility layer (cards, TTS,
captions) and the video-description workflow are next.

## How it works

```
upload + question → ingest (detect type from bytes, normalize, hash)
                  → router (rules first, model fallback) → Route(kind, task, modifiers)
                  → tool for that input type → Answer(answer, evidence, confidence, abstain)
                  → accessibility layer (next): answer first, card, TTS, captions
```

| Input | Model | Where | Guardrail / trick |
|---|---|---|---|
| Image | `qwen3-vl:4b` (stays loaded) | GPU | Counting asks for one bounding box per object, counted in code |
| Video | `qwen3-vl:4b` | GPU | Frame rate depends on the question; first and last frame always kept |
| Document | `qwen3-vl:4b` as a text model | GPU | Must quote its source sentence; code checks the quote exists, else "The document does not say." |
| Code | `qwen2.5-coder:3b` (loaded only for code) | GPU | Generated code is run against the prompt's own examples; one retry with the error |
| Speech | faster-whisper `small`, int8 | CPU | Voice-activity filter; no speech → sound classifier |
| Sounds | AST trained on AudioSet (527 sound types) | CPU | Not trained on ESC-50, the source of the practice clips |

Settings live in [`neuroassist/config.py`](neuroassist/config.py). Every value can be overridden with
an `NEUROASSIST_*` environment variable. Machines with under 12 GB RAM get the `lite` profile
automatically (`qwen3-vl:2b`, Whisper `base`, fewer video frames).

## Setup on the demo laptop (Legion)

1. **Ollama memory settings.** Set these once in PowerShell, then quit and reopen the Ollama app:
   ```powershell
   setx OLLAMA_FLASH_ATTENTION 1
   setx OLLAMA_KV_CACHE_TYPE q8_0
   setx OLLAMA_MAX_LOADED_MODELS 1
   setx OLLAMA_NUM_PARALLEL 1
   ```
2. **Models:**
   ```powershell
   ollama pull qwen3-vl:4b
   ollama pull qwen2.5-coder:3b
   ```
   With 12 GB+ of VRAM, also try `qwen3-vl:8b` via `$env:NEUROASSIST_VISION_MODEL="qwen3-vl:8b"` and keep
   whichever scores better.
3. **Python** (3.10+):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Data:** keep the practice pack outside the repo and point to it:
   `$env:NEUROASSIST_DATA="C:\path\to\practice_pack"`. Never commit it (see `.gitignore`).
5. **While still online, run once:** `python -m neuroassist.prefetch`
   It downloads the Whisper and sound-classifier weights. Without this step, audio fails with Wi-Fi off.

## Eval

```powershell
python eval\run_eval.py                          # all 100 items
python eval\run_eval.py --modality image video   # some input types
python eval\run_eval.py --ids p_vid_009
$env:NEUROASSIST_COUNT_MODE="direct"; python eval\run_eval.py --modality image --tag direct
```

Results go to `eval/results/` as a CSV (every prediction next to its reference) and a Markdown
summary table for the slides. The folder is git-ignored because it contains reference answers.
The app itself never reads `answers.json`.

Settings worth comparing: `COUNT_MODE` (`boxes` / `direct`), `VIDEO_MODE` (`direct` / `timeline`),
`TEXT_MODEL` (e.g. `qwen2.5:3b`), `VIDEO_FPS_DENSE`, `WHISPER_MODEL`.

## Using the Colab fine-tune (card writer)

The fine-tuned model only writes the accessible summary cards; facts always come from the base models.

1. Copy the `.gguf` from Drive into a folder on the Legion, next to a file named `Modelfile`. If Unsloth
   did not create one, use this (Qwen2.5 chat format):
   ```
   FROM ./neuro_assist_model-q4_k_m.gguf
   TEMPLATE """{{- if .System }}<|im_start|>system
   {{ .System }}<|im_end|>
   {{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
   {{ .Content }}<|im_end|>
   {{ end }}<|im_start|>assistant
   """
   PARAMETER stop "<|im_end|>"
   PARAMETER stop "<|im_start|>"
   PARAMETER temperature 0.2
   ```
2. `ollama create neuro-assist -f Modelfile`
3. `$env:NEUROASSIST_CARD_MODEL="neuro-assist"`

## Quick manual test

```powershell
python -m neuroassist "How many people are in this image?" C:\path\to\photo.jpg
```

## Known limits

- Generated code runs in a separate process with a timeout. That isn't a security sandbox.
- Scanned PDFs with no text layer aren't handled yet.
- Video understanding is limited by how many frames fit in context; very fast events can fall between frames.
- Sound labels are limited to AudioSet's 527 types.

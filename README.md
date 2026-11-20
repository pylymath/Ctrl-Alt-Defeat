# NeuroAssist

Offline multimodal assistant for students with ADHD and dyslexia. One interface for photos,
audio, documents, code and short videos; every model runs locally through Ollama, with no internet.

**Run it:** `python -m neuroassist.ui` opens the app at http://127.0.0.1:7860 (local only, Gradio
analytics off).

## How it works

```
upload + question → ingest (detect type from bytes, normalize, hash)
                  → router (rules first, model fallback) → Route(kind, task, modifiers)
                  → tool for that input type → Answer(answer, evidence, confidence, abstain)
                  → accessibility steps: summary card, flashcards, captions, read aloud
```

### Multi-step requests

`pipeline.answer()` turns each request into a short plan and records every step (model, time) in
`Answer.steps`; the UI shows it under "How I got this answer".

| Request | Steps, in order |
|---|---|
| "What's the main argument? Read it aloud." (PDF) | doc QA (text model) → read aloud (Windows speech) |
| "What does the lecturer say about X?" (audio) | Whisper → doc QA over the transcript (text model) |
| "Summarize this and quiz me" (PDF) | summary (card model) → card → flashcards (card model, checked against the PDF) |
| "Give an audio description of this for someone who is blind" (video) | vision answer → per-frame descriptions (vision) → Whisper on the clip's audio → step-by-step narration (text model) → WebVTT captions |
| "How many people are there? What does the sign say?" (photo) | split on `?` → one vision call per question → numbered answers |
| Two or more files | one tool per file → combine (text model) |
| A long video | frame descriptions in batches (vision) → answer from the timeline (text model) |

Output steps come from words in the question (`router.MODIFIERS`: "summarize", "quiz me", "captions",
"read aloud"...) or from the "Also give me" boxes in the UI. They only change presentation: card points
with numbers the source never mentions are dropped, and flashcards whose answer is not in the source are
dropped. The eval calls `answer(..., extras=False)`, so it scores only the answer itself.

### Accessibility in the UI

Answer first in large text, then the card; flashcards that flip on tap; read aloud at an adjustable speed
(Windows' built-in voices, no download); captions as a `.vtt` file; font (standard, dyslexia-friendly
spacing, large print), text size, line spacing and background tint (white, cream, pale blue, dark) restyle
the current answer instantly. Badges say when an answer was checked against the file, or that the file
does not contain it.

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
   ollama pull qwen3-vl:4b-instruct   # -instruct: the plain `4b` tag is the thinking model
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

## Using the Colab fine-tunes

The notebooks in `finetune/` save everything to Google Drive under `neuroassist-finetunes/`. On the Legion:

1. **Download** that Drive folder, keeping its layout, e.g. to `C:\models\neuroassist-finetunes`.
   Only these files are needed (skip `lora/`, `hf/` and any f16 `.gguf` to save space):

   | Folder | Files the app needs | Loaded by |
   |---|---|---|
   | `neuro-card-writer/` | `*q4_k_m.gguf` + `Modelfile` | Ollama |
   | `neuro-coder/` | `*q4_k_m.gguf` + `Modelfile` | Ollama |
   | `neuro-doc-qa/` | `*q4_k_m.gguf` + `Modelfile` | Ollama |
   | `neuro-vision/merged/` | the whole folder (safetensors + `Modelfile`) | Ollama, quantized on import |
   | `neuro-whisper-small/ct2/` | the whole folder (`model.bin`, tokenizer, configs) | faster-whisper, directly |

   Whisper is not a GGUF: `ct2/` is the CTranslate2 int8 copy the app loads; `hf/` is the plain
   Hugging Face backup and is not used.
2. **Import** into Ollama: `.\scripts\import_finetunes.ps1 -ModelsDir "C:\models\neuroassist-finetunes"`.
   It skips anything missing and prints the `$env:` lines that switch the app to each fine-tune.
3. **Keep a fine-tune only if it wins on the practice pack.** Run the eval once with the stock models
   and once with the fine-tune, for the modality it affects, and compare the two `eval/results/*.md`:
   ```
   python eval/run_eval.py --modality code --tag stock
   $env:NEUROASSIST_CODE_MODEL="neuro-coder"
   python eval/run_eval.py --modality code --tag finetuned
   ```
   Doc QA → `--modality text`; vision → `--modality image video` (with the COUNT/VIDEO settings the
   script prints); Whisper → `--modality audio`. The card writer only affects summaries; check those by eye.
4. **Make the winners permanent** with `setx NAME "value"` (applies to new terminal windows), then rerun
   `python -m neuroassist.prefetch` to load-check everything.

Memory: the main vision model (also `TEXT_MODEL`) and `QA_MODEL` stay loaded; the coder and card writer are
loaded only when used. With the doc-QA fine-tune as `QA_MODEL` that is two resident models (about 6 GB of VRAM).

`QA_MODEL` is used only for grounded document/transcript questions. Do not point `TEXT_MODEL` at a QA
fine-tune: summaries, cards, flashcards, video timelines and chat all use `TEXT_MODEL`.

## Quick manual test (no UI)

```powershell
python -m neuroassist "How many people are in this image?" C:\path\to\photo.jpg
```

## Known limits

- Generated code runs in a separate process with a timeout. That isn't a security sandbox.
- Scanned PDFs with no text layer aren't handled yet.
- Video understanding is limited by how many frames fit in context; very fast events can fall between frames.
- Sound labels are limited to AudioSet's 527 types.

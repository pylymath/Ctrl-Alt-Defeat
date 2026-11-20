# Percie

> The code keeps its working name: the Python package `neuroassist/`, settings `NEUROASSIST_*` /
> `neuroassist.env`, and the model `neuro-doc-qa`. Only the product name changed, so existing setups keep working.

Offline multimodal assistant for students with ADHD and dyslexia. One interface for photos,
audio, documents, code and short videos; every model runs locally through Ollama, with no internet.

**Install:** see [Install](#install-any-windows-mac-or-linux-computer). **Run it:** `run.bat` or `python -m neuroassist.ui` opens the app at http://127.0.0.1:7860 (local only, Gradio
analytics off). `--lan` also serves it to phones on the same Wi-Fi; inference stays on the laptop.

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

### Interface

A calm, low-glare design: warm paper background with a soft gradient, one muted sage accent, a serif heading,
system fonts (works offline), light and dark mode. Four everyday actions up front; everything else is one
click away under **More**. Reading settings sit in a sidebar and restyle every view at once: font
(including dyslexia-friendly), size, line spacing, background (**Auto** follows light/dark), **bionic reading**
(bold word starts, shown at normal base weight so they stand out), and read-aloud speed.
`python tests/ui_preview.py` opens it with canned answers (no models) to check the layout.

### Workspace: Understand · Do · Resume

Built around one assignment and all of its materials. Saved as JSON in `~/.neuroassist/workspaces/`,
never uploaded. Labels keep three things apart: **what a file says** (quotes, with their place),
**rewritten wording** (plain language, gists), and **suggestions** (steps).

| Feature | What it does | What keeps it honest |
|---|---|---|
| **Help me start** (Do) | Requirements from every file, grouped into hand in / deadlines / rules / marking / optional, then *suggested* steps | each requirement must quote a sentence that code finds in the file, with its place (`page 2`, `line 20`, `00:11`); no time estimates; "no deadline found, ask your teacher" |
| **Connect my materials** (Do) | Links each requirement to passages in the other files (recording, slides, photo/video descriptions); flags deadlines and length limits that disagree | linking is plain word overlap, so it shows *why* ("mentions moving, average"); conflicts read **"check both"**, and the app never picks one |
| **Break down steps / one step at a time / timer** (Do) | Splits the current step into 2-3 parts (shown as 'Part 1 of 3'; the step count stays the same); shows one step and the sentence it serves; a work timer the student starts | the timer is the student's own, not a deadline |
| **I lost the thread** (Understand) | "Catch me up from 12:40" / "between 2:00 and 3:30" / "the last 5 minutes" on a recording or video | each recap point keeps a real timestamp from that window (others are dropped); instructions are quoted word for word |
| **Where was I?** (Resume) | Goal, progress, last step finished, next step and its source, last file and question, deadlines, open questions, recent activity | read from the saved record with **no model call** |
| **Make this easier** (Understand) | For a passage: a one-line gist with details folded away, instructions one at a time, unclear phrases highlighted in the original, plain language beside the original | idioms get their *common meaning*; vague words get "the text does not specify…"; no sarcasm or intent detection; rewrites that change a number aren't shown |

### Harness (the code around every model call)

- **Empty replies are retried**, then the other video mode is tried (direct ↔ timeline), before anything says
  "I could not tell". All 10 failed practice videos were empty replies.
- **Fixed context per model.** Ollama reloads a model whenever its context size changes, and the vision model was called
  with 1k/4k/12k contexts.
- **Time limit per call** (`NEUROASSIST_LLM_TIMEOUT`, default 180 s), with a clear message; saved work is kept.
- **Every step is recorded** in "How I got this answer": model, time, **GPU memory**, and any retry or fallback.
- `scripts/ollama_settings.ps1`: one request at a time, at most 2 models loaded, flash attention, 8-bit context memory.
- `tests/offline_checks.py`: 54 checks of the above using a stand-in model, with no Ollama or GPU (about 10 s).

`DEMO.md` is the 5-minute demo script; `assets/demo/` holds its files.

## Install (any Windows, Mac or Linux computer)

One command sets everything up. You need internet for this step only; after it, the app works offline.

**Windows** (PowerShell, in the repo folder):
```powershell
git clone https://github.com/pylymath/Ctrl-Alt-Defeat
cd Ctrl-Alt-Defeat
powershell -ExecutionPolicy Bypass -File install.ps1
run.bat
```
**Mac / Linux:**
```bash
git clone https://github.com/pylymath/Ctrl-Alt-Defeat && cd Ctrl-Alt-Defeat
chmod +x install.sh run.sh && ./install.sh && ./run.sh
```

What the installer does (`install.ps1` / `install.sh` → `python -m neuroassist.installer`):

1. Finds Python 3.10–3.12 and Ollama. If either is missing, it offers to install it (winget on Windows).
2. Creates `.venv` and installs the packages (CPU-only PyTorch, so it's small).
3. Starts Ollama if needed, with low-memory settings (one request at a time, 8-bit context memory).
4. Picks a profile for the hardware: a 6 GB+ NVIDIA GPU, Apple silicon with 16 GB+, or 24 GB+ RAM gets
   `qwen3-vl:4b-instruct`; anything smaller gets `qwen3-vl:2b-instruct`. Then it pulls that model and `qwen2.5-coder:3b`.
5. Imports our fine-tuned document-QA model if you give it one:
   `install.ps1 -Finetunes D:\neuroassist-finetunes` (a folder or USB stick), or
   `install.ps1 -HfRepo USER/neuroassist-finetunes` (Hugging Face). Without one, document QA uses the stock model.
6. Downloads Whisper and the sound classifier, so audio works offline.
7. Writes `neuroassist.env` (this machine's settings; git-ignored, and nothing global is changed), then asks
   one live test question and prints the time and GPU memory.

`run.bat` / `./run.sh` checks the status and opens the app at http://127.0.0.1:7860. Add `--lan` to open it from
a phone on the same Wi-Fi; inference stays on the computer. To see what's installed:
`python -m neuroassist.installer --check`.

**Sharing the fine-tune.** The GGUF (~2 GB) is too big for GitHub. Upload the `neuro-doc-qa` folder (the `.gguf` and
its `Modelfile`) to a Hugging Face model repo, for example from Colab:
`huggingface-cli upload USER/neuroassist-finetunes /content/drive/MyDrive/neuroassist-finetunes/neuro-doc-qa neuro-doc-qa`.
Check the base model's licence before making it public (Qwen2.5-3B uses the Qwen research licence).

**Practice data** (eval only): keep it outside the repo and set `NEUROASSIST_DATA=C:\path\to\practice_pack`,
or add that line to `neuroassist.env`. Never commit it (see `.gitignore`).

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

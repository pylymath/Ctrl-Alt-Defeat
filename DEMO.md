# Demo script: 5 minutes, all five input types, one assignment

**Headline:** understand my materials, help me start, help me get back on track. All five input types
work underneath, all on this laptop.

The demo files are in `assets/demo/`. We made them ourselves with `scripts/make_demo_assets.py`; nothing comes from the practice pack.

| File | Input type | Used for |
|---|---|---|
| `assignment.txt` | text | requirements, "not in your file", make this easier, Help me start |
| `diagram.png` | image | describe a diagram and what its arrows connect |
| `voice_note.wav` | audio | the TA's spoken instructions (one moves the deadline to **Thursday**), then a bell |
| `process.mp4` | video | what happens first, and what happens next |
| (typed prompt) | code | writes a function and runs the prompt's own examples |

Start the app with `python -m neuroassist.ui`, then upload all four files at the top.

## 0:00–0:25 · The person and the problem
"For us, the problem isn't always understanding the material. It's keeping the instructions, the explanation
and the next step connected, and finding our place again after losing focus. NeuroAssist does that on this
laptop, without sending course files anywhere." Point at the status line: every model is local.

## 0:25–1:10 · Text (1 · Understand, *Ask about*: `assignment.txt`)
1. Ask `What must I submit?`. You get the answer plus the sentence it came from (open "How I got this answer").
2. Ask `How many references are required?`. You get **"The document does not say."**, with the *Not in your file* badge.
3. Open **🪄 Make this easier** and press it with the box left empty, so it uses the file. Show:
   - the one-line gist, with "More detail" folded away;
   - the instructions **one at a time** (press *Next instruction*);
   - unclear phrases **highlighted in the original**. "go the extra mile" is labelled *Common meaning*;
     "in a timely manner" is labelled *Not exact: the text doesn't say how soon*.

## 1:10–1:35 · Image (*Ask about*: `diagram.png`)
Ask `Describe this diagram and explain what the arrows connect.`, with **Read aloud** ticked.

## 1:35–2:10 · Audio (*Ask about*: `voice_note.wav`)
1. Ask `What does the TA say to do before computing the moving average?`. Answer: remove the rows with missing temperatures.
2. Ask `What sound is at the end?`. You get a bell/chime, shown as a prediction with its score.
3. Open **🧵 I lost the thread**, type `0:08`, and press **Catch me up**. You get a recap, each point with its
   timestamp, and the instructions said in that part **word for word**.

## 2:10–2:45 · Video (*Ask about*: `process.mp4`)
Ask `What happens first, and what happens next?`. A window slides along the readings, and a result appears
at each step. "How I got this answer" shows the frames or timeline it used.

## 2:45–3:25 · Code (2 · Do → **💻 Write code for a step**)
```python
def moving_average(values, window):
    """Return the moving averages of `values` over `window` items.
    If the window is larger than the list, return an empty list.
    >>> moving_average([1, 2, 3, 4], 2)
    [1.5, 2.5, 3.5]
    >>> moving_average([5, 5, 5], 3)
    [5.0]
    >>> moving_average([1, 2], 5)
    []
    """
```
The badge reads "Passed 3/3 examples from the prompt". Say: **it passes these examples; that is not proof it's correct.**

## 3:25–4:25 · The signature moment (2 · Do, then 3 · Resume)
1. Name it `Lab 4` and press **🚀 Help me start**. You get:
   - **What your files say**, grouped into What to hand in / Deadlines / Rules and limits / How it's marked /
     Optional. Each item shows the sentence and the exact place it came from (`assignment.txt · line 20`,
     `voice_note.wav · 00:11`).
   - **⚠️ Check both, they differ: Friday vs Thursday.** The PDF and the TA's voice note disagree.
     NeuroAssist shows both and doesn't decide which is right. *This is "connect my materials".*
   - **Suggested steps**, labelled as suggestions and kept separate from the requirements.
2. Press **🔍 Make this smaller** on a step. It's replaced by smaller actions.
3. Tick **Focus mode**. Only the current step shows, with the sentence it serves.
4. Start a **10-minute timer**, then press **✅ Done** once.
5. Go to **3 · Resume** and press **🧭 Where was I?**. It shows the last step finished, the next step and its source,
   the last file and question, both deadlines, and "Still open: How many references are required?".
   Say: **"This is read from the saved record. Nothing is guessed."**

## 4:25–5:00 · Evidence and close
Show one slide with the eval results by input type and the local timings. "How I got this answer" also shows
time, GPU memory and any retries for each step. Then show this table:

| Component | What we found | Decision |
|---|---|---|
| Document QA | fine-tune: text 5→16/20, audio 14→17/20 | keep the fine-tune, for QA only |
| Code | fine-tune 15/20 vs stock 19/20 | keep the stock coder |
| Speech | fine-tune tied with stock | keep stock Whisper |
| General text | the QA-only model hurt summaries, cards and video | separate QA_MODEL from TEXT_MODEL |
| Video | the 10 failures were empty replies, not wrong answers | retry, then fall back to the other video mode |
| Vision fine-tune | Ollama can't import Qwen2.5-VL safetensors | not deployed; stock vision |
| GPU memory | 12–14 GB from 4× context memory and model reloads | 1 request at a time, fixed context per model |

## Before going on stage
- Run `scripts\ollama_settings.ps1` once, then restart Ollama.
- Run the whole script once on the Legion and time it. If *Help me start* (with photos/videos) or the
  video question is slow, run them before the demo and say on stage that they were processed beforehand.
  Keep at least one generation live.
- Start with a fresh assignment name so "Where was I?" shows only today's run. Saved assignments are in
  `%USERPROFILE%\.neuroassist\workspaces\`.
- Phone (optional): run `python -m neuroassist.ui --lan` and open `http://<laptop-ip>:7860` on the same Wi-Fi.
  Inference stays on the laptop. Use it only on a trusted network, and ask the organizers whether it counts
  for the web/mobile bonus.

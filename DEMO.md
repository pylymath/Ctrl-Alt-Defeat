# Presenting NeuroAssist

Two versions: **5 minutes** for round one, and **15 minutes** for the final. The 15-minute one is the
5-minute one plus three extra sections, so you only learn one demo.

**Roles:** one **Driver** at the keyboard, who clicks and doesn't talk, and one or two **Speakers**, who talk and don't click.
**One sentence to remember:** *"NeuroAssist helps students understand their materials, get started, and get back on
track, all on this laptop."*

## Before you start (5 minutes, backstage)
1. `run.bat`. The status line must show every model ✅.
2. Upload the 4 files in `assets/demo/` (`assignment.txt`, `diagram.png`, `voice_note.wav`, `process.mp4`).
3. In **2 · Do**, press **Help me start** once with the name `Lab 4`, so the slow part is already done.
   Say on stage that it was prepared beforehand.
4. Reading settings: font *Dyslexia-friendly*, background *Cream*, text size 20 (it reads well on a projector).
5. Ask one warm-up question so the model is loaded.

---

## Round one: 5 minutes

| Time | Driver clicks | Speaker says (short version) |
|---|---|---|
| **0:00** | Nothing; the app is on screen | "Many of us lose track of what an assignment wants, or lose our place in a lecture. That's not a reading problem, it's a connecting problem. NeuroAssist connects your materials, gives you a first step, and gets you back on track, offline." |
| **0:30** | **1 · Understand**, *Ask about* `assignment.txt` → `What must I submit?` → open *How I got this answer* | "It answers only from your file and shows the sentence it used." |
| **0:50** | Ask `How many references are required?` | "If the file doesn't say, it says so: *Not in your file*. No made-up answers." |
| **1:05** | *Ask about* `diagram.png` → `What do the arrows connect?` (tick **Read aloud**) | "Photos and diagrams, read aloud if you want." |
| **1:30** | *Ask about* `voice_note.wav` → `What sound is at the end?`, then **🧵 Catch me up** from `0:08` | "Recordings: it identifies sounds, and if you zoned out, it catches you up from a timestamp, with the TA's instructions word for word." |
| **2:10** | *Ask about* `process.mp4` → `What happens first, and what happens next?` | "Video: what happened, in order." |
| **2:35** | **2 · Do** → **💻 Write code** → paste the `moving_average` prompt (below) | "It writes code and runs the examples from your prompt: 3 of 3 pass. That means it passes these examples, not that it's proven correct." |
| **3:10** | Scroll the **Do** tab: requirements, then the **⚠️ Check both: Friday vs Thursday** box | "Here's the main feature. The PDF says Friday. The TA's voice note says Thursday. NeuroAssist found both, shows where each came from, and doesn't pick one. Ask your teacher." |
| **3:40** | **🔍 Make this smaller** on step 1 → tick **Focus mode** → **✅ Done** | "Too big to start? Make it smaller. Focus mode shows one step at a time." |
| **4:05** | **3 · Resume** → **🧭 Where was I?** | "Come back tomorrow: your last step, your next step, and the question still open. All read from what you saved. Nothing is guessed." |
| **4:30** | Switch to the results slide | "All five input types run on one laptop. What we tested and kept is on this slide. Thank you." |

**Code prompt to paste:**
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

**If something fails live:** say *"This is the fallback we built for exactly this"*. Open *How I got this answer*,
where retries and errors are listed, and move on to the next row. Don't debug on stage.

---

## The final: 15 minutes

| Time | Section | What to show / say |
|---|---|---|
| **0–1** | **Hook** | The same opening as round one, plus one honest personal line about why this team built it. |
| **1–3** | **Who it's for, and the problem** | A computing student with ADHD, autism or dyslexia. Three moments where they get stuck: *"What does this assignment actually want?"*, *"I zoned out, what did I miss?"*, *"I stopped yesterday, where was I?"*. Why offline matters: course files and recordings never leave the laptop. |
| **3–8** | **Live demo** | The 5-minute demo, slower, plus these extras: **🪄 Make this easier** on the assignment (gist, then instructions one at a time, unclear phrases highlighted: "go the extra mile" → *common meaning*, "in a timely manner" → *the text doesn't say how soon*); the **work timer**; **Add your own step**; reading settings changed live. |
| **8–11** | **How it works** (architecture slide) | One diagram: *file → router (rules, no model) → one model per job → checks → answer + evidence*. Models: Qwen3-VL 4B (vision and text), our fine-tuned Qwen2.5-3B for document Q&A, Qwen2.5-Coder 3B, Whisper small, the AST sound classifier. Then the **checks**: quotes must be found in the file, code runs the prompt's examples, invented numbers are dropped, empty replies are retried, then video switches mode. Open *How I got this answer* to show each step's model, time and GPU memory. |
| **11–13** | **Evidence** (results slide + "what we tested and kept") | The eval table by input type. Fine-tuning: document Q&A went 5→16/20 on text, so we kept it; the coder got worse and Whisper tied, so we kept stock. The video failures were empty replies, which we fixed with a retry and a fallback. GPU memory was 12–14 GB, reduced with fixed context sizes and one request at a time. *Use the real numbers from the 8 am Legion run.* |
| **13–14** | **Limits, honestly** | Linking files uses shared words, not understanding. Video is short clips only. A "check both" warning is a hint, not a ruling. We tested with our team only: a few people, early results. |
| **14–15** | **Close** | "Understand your materials, start, and get back on track, offline. Anyone can install it with one command." Show the `install.ps1` line. |

---

## Judges' questions: short answers

| Question | Answer |
|---|---|
| Why these models? | Small enough to run together on one laptop GPU. We measured each against stock and kept a fine-tune only where it won (document Q&A). |
| What does your fine-tune actually improve? | Answering from a document with an exact quote: 5→16 of 20 on text, 14→17 of 20 on audio transcripts. |
| How do you catch made-up answers? | The model must quote the sentence it used, and code checks that the sentence is really in the file. Numbers the file never mentions are rejected. No match means "Not in your file". |
| What don't the checks guarantee? | That the reasoning is right. A real quote can still be misread. That's why we always show the quote, so the student can check. |
| What does video miss? | It samples up to 16 frames, so very fast events between frames can be missed. Clips longer than a few minutes aren't the target. |
| How much memory? | Measure on the Legion with `ollama ps` and quote it: *"__ GB GPU while answering"*. It's shown per step in the app. |
| What if processing fails? | Retry, then the other video mode, then a clear message. Saved work is never lost: it's a file on disk. |
| Is anything sent online? | No. After setup, everything works with Wi-Fi off. Gradio analytics are turned off too. |
| Could we run it ourselves? | Yes: `install.ps1` (Windows) or `install.sh` (Mac/Linux) sets everything up, then `run.bat` (see the README). |
| Did you train on the test data? | No. Fine-tuning used public datasets (SQuAD 2.0 etc.). The practice pack was only used to measure. |

## Demo files
`assets/demo/` holds our own files, made with `scripts/make_demo_assets.py`. Nothing comes from the practice pack.
The voice note deliberately moves the deadline to Thursday, so the "check both" warning has something to show.

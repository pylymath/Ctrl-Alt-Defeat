# Presenting Percie: two separate talks

- **Round 1 (5 min, every team): breadth.** One assignment, all five input types, and the signature moment.
- **Final (15 min, shortlisted teams): depth.** A different story ("a week with Percie"), different
  interactions, proof it's offline, how it works, the evidence, and the limits. **Nothing from Round 1 is repeated**
  except one 10-second callback.

**Roles:** one **Driver** at the keyboard (clicks, doesn't talk) and one or two **Speakers** (talk, don't click).
**The line to remember:** *"Percie helps students understand their materials, get started, and get back on
track, all on this laptop."*

Files are in `assets\demo\`. We made them all ourselves:
- `assignment.txt`
- `diagram.png`
- `voice_note.wav`: the TA moves the deadline to Thursday
- `lecture.wav`: a 94-second lecture
- `process.mp4`

The app has a **sidebar on the left** for reading settings and the open assignment, and three tabs: **Understand · Do · Resume**.

## Backstage, before either talk (5 min)
1. `run.bat`. The sidebar's "Running locally" pills show every model, with none marked "missing".
2. Sidebar: font *Dyslexia-friendly*, text size 20 (readable on a projector), background *Auto*.
3. Upload the files that talk needs (listed below), and ask one warm-up question so the model is loaded.
4. In **Do**, press **Help me start** once with a fresh name, so the slow part is done. Say on stage that it was
   "prepared before the session".

---

# ROUND 1 (5 minutes): "Lab 4: one assignment, five kinds of input"

Upload: `assignment.txt`, `diagram.png`, `voice_note.wav`, `process.mp4`. **Help me start** is done backstage as `Lab 4`.

| Time | Driver clicks | Speaker says |
|---|---|---|
| **0:00** | Nothing; the app is on screen | "Many of us lose track of what an assignment really wants, or lose our place in a lecture. That isn't a reading problem; it's a connecting problem. Percie connects your materials, gives you a first step, and gets you back on track, offline." |
| **0:30** | **Understand**, *Ask about* `assignment.txt` → `What must I submit?` → open **How I got this answer** | "Text: it answers only from your file and shows the sentence it used." |
| **0:50** | `How many references are required?` | "When the file doesn't say, it says so: *Not in your file*. It never makes an answer up." |
| **1:05** | *Ask about* `diagram.png` → `What do the arrows connect?` | "Images: a diagram from the board, explained." |
| **1:30** | *Ask about* `voice_note.wav` → `What sound is at the end?` | "Audio: it identifies sounds as well as speech. That's a bell." |
| **1:50** | *Ask about* `process.mp4` → `What happens first, and what happens next?` | "Video: what happened, in order." |
| **2:15** | **Do** → **Write code for a step** → paste the prompt below → **Write it and check the examples** | "Code: it writes the function and runs the examples in your prompt. 3 of 3 pass, which means it passes these examples, not that it's proven correct." |
| **2:50** | Scroll to **What your files say** → the **Check both: Friday vs Thursday** box | "This is the core feature. The assignment says Friday. The TA's voice note says Thursday. Percie found both, shows exactly where each came from, and doesn't pick one for you." |
| **3:25** | **Make this smaller** on the first step → tick **Focus mode** → **Done** | "Too big to start? Make it smaller. Focus mode shows one step at a time." |
| **3:55** | **Resume** → **Where was I?** | "Come back tomorrow: your last step, your next step, and the question you still need to ask. All read from what you saved. Nothing is guessed." |
| **4:25** | Results slide | "All five input types, one laptop, no internet. We tested every model choice and kept only what measured better. Thank you." |

**Code prompt (Round 1):**
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

---

# FINAL (15 minutes): "A week with Percie"

A different story with different interactions. Upload **all five** files: `lecture.wav`, `assignment.txt`,
`diagram.png`, `voice_note.wav`, `process.mp4`. Backstage, **Help me start** as `Week demo`, with "Also connect
photos and videos" ticked.

| Time | Section | Driver | Speaker |
|---|---|---|---|
| **0:00–1:00** | **Hook + proof it's offline** | **Turn Wi-Fi off** in the taskbar, where the audience can see it | "Last round you saw one assignment. Now, a week. And to prove nothing leaves this laptop, Wi-Fi is now off." (One honest line about why the team cares.) |
| **1:00–2:00** | **Who it's for** | Slide: three stuck moments | "A computing student with ADHD, autism or dyslexia gets stuck in three places: *What does this actually want?* *I zoned out, what did I miss?* *I stopped yesterday, where was I?*" |
| **2:00–3:30** | **Monday: the lecture** | *Ask about* `lecture.wav` → **I lost the thread** → `0:40` → **Catch me up**. Then **More** → tick *Read aloud* and *Captions* → chip **Explain simply** → **Ask** | "She zoned out 40 seconds in. *Catch me up* gives every point with its timestamp, and the instructions word for word. Captions download as a file, and it reads aloud at her speed." |
| **3:30–4:30** | **Tuesday: quiz** | Same file → chip **Quiz me** → **Ask** → flip two flashcards | "Flashcards from the lecture. Any card whose answer isn't in the recording is dropped." |
| **4:30–6:00** | **Wednesday: the assignment is hard to read** | *Ask about* `assignment.txt` → **Make this easier** (box empty) → **Next instruction** twice → in the sidebar, tick **Bionic reading** and change the background | "One-line gist first, details folded away. Instructions one at a time, word for word. Unclear phrases are highlighted: 'go the extra mile' gets its *common meaning*; 'in a timely manner' gets *the text doesn't say how soon*. We never guess what the teacher meant. And every reading setting changes the whole app at once." |
| **6:00–7:30** | **Thursday: start the work** | **Do** → open `Week demo` in the sidebar → show the links to the lecture and the diagram → **Check both** → **More** → add your own step `Email the TA about the deadline` → **Timer** → 15 min → **Start** | "Every requirement is connected across the PDF, the lecture, the TA's note and the board photo, with the reason for each link. The deadline conflict becomes her own first step. The timer is hers, not a deadline." |
| **7:30–8:30** | **Friday: code + video** | **Write code for a step** with the prompt below; then **Understand**, *Ask about* `process.mp4` → **More** → chip **Audio description** → **Ask** | "The cleaning function, checked against its examples. And the video gets an audio description, with captions, for a blind classmate." |
| **8:30–9:00** | **Saturday: back after a break** | **Resume** → **Where was I?** | "A 10-second callback to Round 1: everything is read from her saved progress." |
| **9:00–11:30** | **How it works** | Architecture slide → then open **How I got this answer** on any answer (models, time, GPU memory, retries) | "The router uses rules, not a model: each file goes to one specialist model, then **checks**. Quotes must exist in the file. Numbers the file never says are dropped. Code runs its examples. Empty replies are retried, and video switches mode. Each model keeps a fixed context, so it never reloads. This table is live, per step." |
| **11:30–13:30** | **Evidence** | Results slide + the "What we tested and kept" table (below) | "Practice-pack scores by input type (*use this morning's numbers*). We fine-tuned five models; one clearly won (document Q&A, 5→16 of 20 on text), so it's the only one we ship. The video failures were empty replies, not wrong answers; the retry fixed them. GPU memory went from 12–14 GB to __ GB." |
| **13:30–14:30** | **Limits + next** | Slide | "Links use shared words, not understanding. Video is short clips. 'Check both' is a hint, not a ruling. We've tested with our own team only. Next: tests with more students, scanned PDFs, and a phone view over local Wi-Fi." |
| **14:30–15:00** | **Close** | Show the one-line install in the README | "One command installs it on any laptop. Understand, start, get back on track, offline. Thank you." |

**Code prompt (Final, different from Round 1):**
```python
def drop_missing(readings):
    """Remove readings that are None. Keep the order.
    >>> drop_missing([21.5, None, 22.0, None])
    [21.5, 22.0]
    >>> drop_missing([])
    []
    >>> drop_missing([None])
    []
    """
```

**"What we tested and kept" (Final, minutes 11:30–13:30):**

| Component | What we found | Decision |
|---|---|---|
| Document Q&A | fine-tune: text 5→16/20, audio transcripts 14→17/20 | ship the fine-tune, for Q&A only |
| Code | fine-tune 15/20 vs stock 19/20 | keep stock |
| Speech | fine-tune tied with stock | keep stock Whisper |
| Vision | fine-tune trained, but Ollama can't import it | don't claim it; stock vision |
| General text | the Q&A-only model made summaries worse | split the Q&A model from the text model |
| Video | the 10 failures were empty replies | retry, then switch video mode |
| Memory | 12–14 GB from 4× context memory and reloads | 1 request at a time, fixed contexts |

**If something fails live (either talk):** say *"That's exactly why we built the fallbacks"*, open **How I got
this answer** (retries and errors are listed there), and move to the next row. Don't debug on stage.

---

# Judges' questions: 15 likely ones, with short answers

| # | Question | Answer (about 20 seconds) |
|---|---|---|
| 1 | **Why run it locally instead of using a cloud model?** | Course files, recordings and notes are personal, and many students can't upload them. Local means private, free, and working without Wi-Fi; we showed that live. |
| 2 | **Why several small models instead of one big one?** | One laptop GPU. Each job gets a model that's good at it and small enough to share memory: vision (Qwen3-VL 4B), document Q&A (our fine-tuned Qwen2.5-3B), code (Qwen2.5-Coder 3B), speech (Whisper small), sounds (AST). |
| 3 | **How does it decide which model to use?** | Rules, not a model: the file's bytes give the type, and the question's wording gives the task. That's instant and predictable. A model is only asked when there's no file and the rules can't tell. |
| 4 | **What did fine-tuning actually improve?** | Answering from a document with an exact quote: text 5→16 of 20, audio transcripts 14→17 of 20 on the practice pack. We also fine-tuned code, speech and vision; they didn't beat stock, so we don't ship them. |
| 5 | **How do you stop it making things up?** | The model must return the exact sentence it used, and code checks that the sentence is in the file. A question mentioning numbers the file never contains is refused. Requirements without a matching sentence are dropped, and we show how many. |
| 6 | **What do those checks NOT guarantee?** | That the reasoning is right. A real quote can still be misread. That's why the quote is always shown, so the student can check it in one glance. |
| 7 | **How do you know this helps neurodivergent students?** | Honestly: early testing with our own team, a small number of people. The features target well-known difficulties: starting tasks, keeping place, ambiguous wording. Each one is designed so it can't mislead (quotes, no guessing). A proper study with more students is the next step. |
| 8 | **How fast is it, and how much memory?** | (Use this morning's numbers.) About __ s per answer and __ GB of GPU memory, shown live per step in "How I got this answer". Ollama runs one request at a time with 8-bit context memory. |
| 9 | **What happens when something fails?** | It retries an empty or broken reply, tries the other video mode, then gives a clear message. Saved work is a file on disk and is never lost. Each model call has a time limit, so nothing hangs. |
| 10 | **What are the video limits?** | Up to 16 frames, sampled more densely for counting and motion. Very fast events between frames can be missed. It's built for short clips, not hour-long recordings. For lectures, the audio track goes through Whisper. |
| 11 | **Did you train on the test data?** | No. Fine-tuning used public datasets (SQuAD 2.0 and others). The practice pack was only used for measuring and isn't in the repo. |
| 12 | **How is this different from ChatGPT or NotebookLM?** | Offline and free. Every claim points to a sentence and its page, line or timestamp. It flags conflicting instructions instead of resolving them. It saves your place and reads it back without guessing. And it's designed for accessibility from the start: calm colours, dyslexia font, bionic reading, read-aloud. |
| 13 | **What if two of the student's files disagree?** | It shows both, with where each came from, labelled "check both", and never decides. In the demo, the PDF says Friday and the TA says Thursday. |
| 14 | **Could we run it on our own laptop?** | Yes: `install.ps1` (Windows) or `install.sh` (Mac/Linux). It picks smaller models for smaller machines, downloads everything once, then works offline. `run.bat` opens it. |
| 15 | **What would you do with another month?** | Test with more students; read scanned PDFs page by page with the vision model; a phone view over local Wi-Fi; and train the document model on assignment-style instructions, the gap we measured. |

**Ground rules for answers:** say "we measured" only with a number from our eval. If you don't know, say "we
haven't measured that". Never claim it diagnoses, detects attention, or reads emotions; it doesn't.

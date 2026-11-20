# Morning setup on the Legion (target: 20 minutes, then rehearse)

**Golden rules (yesterday's problems came from breaking these)**
1. **One command at a time.** Paste one line, press Enter, and wait for the prompt (`PS C:\...>`) before the next.
2. **Never run plain `python` or `pip`.** Always `.venv\Scripts\python.exe ...`. (Yesterday `pip` went outside the venv.)
3. **Don't edit any code** (no Notepad fixes). If something's wrong, screenshot it and send it to Claude.
4. **Don't install or upgrade any packages** beyond what the script does. Don't turn off Smart App Control.
5. **Don't touch the old folder.** It is the fallback and keeps working whatever happens to the new one.
6. **Red `[FAIL]` line?** Stop, screenshot the whole window, and look up the fix in the table below.

---

## Tonight (Yash, 2 minutes)
- Upload `C:\Users\uctel\Downloads\percie-v2.zip` (about 3 MB) to the team Google Drive.
- (Optional) Upload the fine-tune to Hugging Face (steps in the README, "Sharing the fine-tune"). It isn't needed for the
  Legion, because the fine-tune is already in its Ollama.

## 8:00 · Step 1: put the new version next to the old one (2 min)
1. On the Legion, download `percie-v2.zip` from Drive.
2. Right-click it → **Extract All…** → set the folder to `C:\Users\fepes\Documents\Hackathon 11.26` → **Extract**.
3. You now have **two** folders side by side:
   - `Ctrl-Alt-Defeat-main`: the old version (fallback, don't touch)
   - `Ctrl-Alt-Defeat-v2`: the new version

## Step 2: open PowerShell **in the new folder** (30 s)
Open `Ctrl-Alt-Defeat-v2` in File Explorer, click the **address bar**, type `powershell`, and press Enter.
A window opens that's already in the right folder (no `cd`, so no trouble with the space in "Hackathon 11.26").

## Step 3: one setup command (3–6 min)
```
powershell -ExecutionPolicy Bypass -File scripts\morning.ps1
```
It reuses what already works and installs nothing new:
- **Python:** links the old folder's `.venv`.
- **Old settings:** clears the `setx NEUROASSIST_...` values from yesterday (they now live in `neuroassist.env`).
- **Ollama:** sets its memory settings and restarts it.
- **Models:** checks them, finds `neuro-doc-qa`, loads Whisper and the sound model.
- **Tests:** asks one live test question, then runs the 62 logic checks.

**Good output ends like this** (your seconds and GB will differ):
```
[OK  ] Test answer in 12s, 7.9 GB on the GPU: 'lab4.py'
62 passed, 0 failed
READY. Next:
```
Write down the seconds and GB from the "Test answer" line: those numbers go on the results slide.

| If you see | Do this |
|---|---|
| `[FAIL] Run this from the NEW (v2) folder` | Close the window and redo Step 2 inside `Ctrl-Alt-Defeat-v2`. |
| `[FAIL] No .venv in the old folder` | The old folder has another name. Run: `powershell -ExecutionPolicy Bypass -File scripts\morning.ps1 -OldRepo "C:\full\path\to\old folder"` |
| `[FAIL] Ollama did not start` | Start **Ollama** from the Start menu, wait 10 s, then run the Step 3 command again. |
| `Downloading qwen3-vl:4b-instruct` | Normal if it wasn't pulled yet (about 3 GB, needs Wi-Fi). Let it finish. |
| `Test answer ... [FAIL]` (wrong answer, but no error) | Carry on; the app still works. Tell Claude what it printed. |
| `An Application Control policy has blocked this file` | Screenshot it and send it to Claude. Use the fallback (below) for practice meanwhile. |
| Anything else red | Screenshot it and send it to Claude. Use the fallback meanwhile. |

## Step 4: open the app (1 min)
```
run.bat
```
The browser opens at http://127.0.0.1:7860. Check:
- the sidebar's **Running locally** pills list the models, and none says "missing";
- the tabs **Understand · Do · Resume** are there, and the colours are calm (warm paper, sage green);
- tick **Bionic reading** in the sidebar: the first half of each word turns bold straight away.

Quick test: drag `assets\demo\assignment.txt` onto the big box at the top (or click its paperclip), type
`What must I submit?` and press Enter. The answer should mention lab4.py, and the file should stay attached.

Voice test: under the box, **Or say it** → **Record** (allow the microphone when the browser asks) → say
"What must I submit?" → **Stop**. Your words appear in the box within a few seconds; press Enter.
(Tick **Read answers aloud** to hear the answer.)

## Step 5: run the eval in a SECOND PowerShell window (25–35 min)
Open another PowerShell in the v2 folder (Step 2 again), then:
```
.venv\Scripts\python.exe eval\run_eval.py --tag v2
```
- **Don't use the app while this runs** (they share the GPU). Work on slides meanwhile.
- At the end it prints a table (Image / Audio / Text / Video / Code). Take a photo of it; it goes on the results slide.
- Compare it with yesterday's 58/100 (video 6/15, text 5/20). Video should be clearly higher now. If it isn't, send the table to Claude.

## Step 6: GPU memory number for the judges (1 min)
While the app is answering a question:
```
ollama ps
```
Note the SIZE and "100% GPU". That's the "how much memory?" answer.

## Step 7: rehearse (from ~9:00)
Follow `DEMO.md`, **Round 1**, with a stopwatch, twice. Then the Final once.
Before each run: in **Do**, press **Help me start** with a fresh name (`Lab 4 run 2`, ...).

## 11:00 · Decision
- **Everything worked in rehearsal:** submit v2. Yash, on the dev laptop, tells Claude "push v2 to main", or
  runs `git push origin v2:main` himself.
- **v2 is broken:** demo from the old folder (fallback below), and submit `main` as it is.
- **Code deadline: 12:00 noon.**

---

## Fallback: the old version (works without anything above)
Open PowerShell in the **old** folder (`Ctrl-Alt-Defeat-main`, Step 2 method), then:
```
$env:NEUROASSIST_QA_MODEL="neuro-doc-qa:latest"; .venv\Scripts\python.exe -m neuroassist.ui
```
(Step 3 cleared yesterday's saved settings, so this one-line form sets the QA model for that window.)

## Handy one-liners
| Need | Command (inside the v2 folder) |
|---|---|
| What's installed and working? | `.venv\Scripts\python.exe -m neuroassist.installer --check` |
| The logic checks only (10 s) | `.venv\Scripts\python.exe tests\offline_checks.py` |
| Eval for one input type | `.venv\Scripts\python.exe eval\run_eval.py --modality video --tag v2-video` |
| Stop the app | Click the PowerShell window, then press `Ctrl+C` |
| Phone demo (same Wi-Fi) | `run.bat --lan`, then open `http://<laptop IP>:7860` on the phone (IP: `ipconfig`) |

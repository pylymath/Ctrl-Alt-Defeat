"""Generates the five Colab fine-tuning notebooks in this folder.

    python finetune/build_notebooks.py

Prompts and output formats are copied from the app's own tool code at build time, so every
fine-tuned model is trained on exactly what the app will send it. Re-run this after changing a
prompt in neuroassist/tools/*.py.

Every dataset here is a TRAIN split. All practice items come from validation/test splits
(COCO val2017, VizWiz val, SQuAD 2.0 dev, LibriSpeech test-clean, ESC-10, HumanEval)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from neuroassist import router                          # noqa: E402
from neuroassist.tools import code as app_code          # noqa: E402
from neuroassist.tools import image as app_image        # noqa: E402
from neuroassist.tools import text_qa as app_text       # noqa: E402
from neuroassist.tools import video as app_video        # noqa: E402


def fill(src: str, **values) -> str:
    for k, v in values.items():
        src = src.replace(f"@@{k}@@", repr(v))
    assert "@@" not in src, src[src.index("@@") - 40: src.index("@@") + 40]
    return src


def write(name: str, cells: list[tuple[str, str]], gpu: str = "A100"):
    nb = {"cells": [], "nbformat": 4, "nbformat_minor": 0, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": gpu, "machine_shape": "hm"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}}
    for kind, src in cells:
        lines = src.strip("\n").split("\n")
        c = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", name, len(cells), "cells")


md = lambda s: ("markdown", s)      # noqa: E731
py = lambda s: ("code", s)          # noqa: E731

RULE = ("⚠️ **Hackathon rule:** train only on TRAIN splits. Every practice item comes from a validation/test "
        "split, and the judges' items almost certainly do too. Never train on the practice pack.")

# ============================================================================================
# Shared cells for the three text models (card writer, coder, document Q&A)
# ============================================================================================
GPU_DRIVE = r'''
import torch, os, json, random, re
assert torch.cuda.is_available(), "No GPU. Runtime → Change runtime type → GPU"
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name}, {gpu.total_memory / 2**30:.1f} GB")
if gpu.total_memory < 20 * 2**30 and BATCH_SIZE > 2:      # T4 / L4: keep the effective batch size
    GRAD_ACCUM, BATCH_SIZE = GRAD_ACCUM * BATCH_SIZE // 2, 2
    print(f"Small GPU: BATCH_SIZE=2, GRAD_ACCUM={GRAD_ACCUM}")
random.seed(SEED)

OUT_DIR = f"/content/{RUN_NAME}"
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")      # mounted first, so nothing is lost if a later cell fails
    OUT_DIR = f"{DRIVE_DIR}/{RUN_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)
print("Saving to", OUT_DIR)
'''

LOAD_TEXT_MODEL = r'''
from unsloth import FastLanguageModel, is_bf16_supported

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED)
model.print_trainable_parameters()


def chat(user, assistant=None, system=None):
    """One training/eval conversation in the app's format."""
    msgs = [{"role": "system", "content": system if system is not None else SYSTEM_PROMPT},
            {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def generate_batch(conversations, max_new_tokens=256, batch_size=16):
    """Greedy answers for a list of conversations (without the assistant turn)."""
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    outs = []
    for i in range(0, len(conversations), batch_size):
        prompts = [tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                   for c in conversations[i:i + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        outs += tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    tokenizer.padding_side = "right"
    FastLanguageModel.for_training(model)
    return [o.strip() for o in outs]
'''

TO_TEXT = r'''
from datasets import Dataset

texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in examples]
lengths = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
keep = [t for t, l in zip(texts, lengths) if l <= MAX_SEQ_LENGTH]
print(f"{len(keep)} examples ({len(texts) - len(keep)} dropped as longer than {MAX_SEQ_LENGTH} tokens), "
      f"median {sorted(lengths)[len(lengths) // 2]} tokens")
data = Dataset.from_dict({"text": keep}).shuffle(seed=SEED)
split = data.train_test_split(test_size=min(0.02, 500 / len(data)), seed=SEED)
train_ds, val_ds = split["train"], split["test"]
print(f"train {len(train_ds)} · validation-loss set {len(val_ds)}")
print("\n--- one training example ---\n" + train_ds[0]["text"][:1500])
'''

BASELINE = r'''
print("Scoring the base model before training (this is the number to beat)...")
before = evaluate_model()
print(json.dumps(before, indent=2))
'''

TRAIN_TEXT = r'''
import inspect
from trl import SFTTrainer, SFTConfig

wanted = dict(
    output_dir=f"/content/checkpoints/{RUN_NAME}",
    dataset_text_field="text", max_length=MAX_SEQ_LENGTH, max_seq_length=MAX_SEQ_LENGTH,
    packing=False, dataset_num_proc=2,
    per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS, max_steps=MAX_STEPS if MAX_STEPS else -1,
    learning_rate=LEARNING_RATE, warmup_ratio=0.05, lr_scheduler_type="cosine",
    optim="adamw_8bit", weight_decay=0.01,
    bf16=is_bf16_supported(), fp16=not is_bf16_supported(),
    logging_steps=10, eval_strategy="steps", eval_steps=100, save_strategy="no",
    seed=SEED, report_to="none")
accepted = inspect.signature(SFTConfig.__init__).parameters      # TRL renames args between versions
if "max_length" in accepted:
    wanted.pop("max_seq_length")
args = SFTConfig(**{k: v for k, v in wanted.items() if k in accepted})

kw = dict(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=args)
kw["processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
   else "tokenizer"] = tokenizer
trainer = SFTTrainer(**kw)

# Loss only on the assistant's answer. Turn markers are read from this model's chat template.
from unsloth.chat_templates import train_on_responses_only
probe = tokenizer.apply_chat_template(
    [{"role": "user", "content": "@@U@@"}, {"role": "assistant", "content": "@@A@@"}], tokenize=False)
def _marker(text, words):
    idx = max(text.rfind(w) for w in words)
    return text[text.rfind("<", 0, idx):]
user_part = _marker(probe[:probe.index("@@U@@")], ["user"])
asst_part = _marker(probe[probe.index("@@U@@") + 5:probe.index("@@A@@")], ["assistant", "model"])
print(f"training on answers only: {user_part!r} … {asst_part!r}")
trainer = train_on_responses_only(trainer, instruction_part=user_part, response_part=asst_part)

stats = trainer.train()
print(f"done in {stats.metrics['train_runtime'] / 60:.1f} min · train loss {stats.metrics['train_loss']:.3f}")
'''

COMPARE = r'''
after = evaluate_model()
print(f"{'metric':<28}{'before':>10}{'after':>10}")
for k in before:
    b, a = before[k], after[k]
    print(f"{k:<28}{b:>10.3f}{a:>10.3f}" if isinstance(b, float) else f"{k:<28}{b!s:>10}{a!s:>10}")
json.dump({"model": MODEL_NAME, "before": before, "after": after}, open(f"{OUT_DIR}/scores.json", "w"), indent=2)
print(f"\nKeep this fine-tune only if it also beats the stock model in the app's eval (eval/run_eval.py).")
'''

EXPORT_TEXT = r'''
import glob, shutil

model.save_pretrained(f"{OUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUT_DIR}/lora")
print("LoRA adapters saved:", f"{OUT_DIR}/lora")

gguf_dir = f"/content/{RUN_NAME}_gguf"
model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=GGUF_QUANT)
found = [f for f in glob.glob(f"/content/{RUN_NAME}_gguf*/**/*", recursive=True)
         if f.endswith(".gguf") or os.path.basename(f) == "Modelfile"]
for f in found:
    shutil.copy(f, OUT_DIR)
    print("copied", os.path.basename(f), f"({os.path.getsize(f) / 2**20:.0f} MB)")
ggufs = sorted(os.path.basename(f) for f in found if f.endswith(".gguf"))
assert ggufs, "No .gguf was produced; check the export log above."
gguf = next((g for g in ggufs if GGUF_QUANT.lower() in g.lower()), ggufs[0])

# Our own Modelfile, pointing at the quantized file. Qwen models use the ChatML format.
assert "qwen" in MODEL_NAME.lower(), "Modelfile below is for Qwen; write one for this model family."
modelfile = f"""FROM ./{gguf}
TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0
"""
open(f"{OUT_DIR}/Modelfile", "w").write(modelfile)
print(f"""
Done. On the demo laptop, copy {gguf} and Modelfile from Drive into one folder, then:
    ollama create {OLLAMA_NAME} -f Modelfile
and run the app / eval with:
    $env:{APP_SETTING}="{OLLAMA_NAME}"
""")
'''


def text_notebook(filename, title, intro, install_extra, settings, data, cells_after_data=None):
    cells = [
        md(f"# {title}\n\n{intro}\n\n**Run:** `Runtime → Change runtime type → A100 GPU` (T4 works, slower), "
           f"then `Runtime → Run all`. Only the ⚙️ Settings cell is meant to be edited.\n\n{RULE}"),
        md("## 1. Install"),
        py("# Unsloth installs matching torch / transformers / trl / peft / bitsandbytes.\n"
           "# If a new release breaks, pin the previous one, e.g. unsloth==2026.11.6\n"
           "!pip install -q unsloth" + (f"\n!pip install -q {install_extra}" if install_extra else "")),
        md("## 2. ⚙️ Settings"),
        py(settings),
        md("## 3. GPU and Google Drive"),
        py(GPU_DRIVE),
        md("## 4. Load the model and add LoRA adapters"),
        py(LOAD_TEXT_MODEL),
        md("## 5. Build the training data and the held-out test"),
        py(data),
        md("## 6. Format with the model's chat template"),
        py(TO_TEXT),
        md("## 7. Score the base model (before)"),
        py(BASELINE),
        md("## 8. Train"),
        py(TRAIN_TEXT),
        md("## 9. Score again (after) and compare"),
        py(COMPARE),
        md("## 10. Save adapters, export GGUF and a Modelfile for Ollama"),
        py(EXPORT_TEXT),
    ]
    write(filename, cells)


# --------------------------------------------------------------------------------------------
# 01 Card writer (sentence simplifier)
# --------------------------------------------------------------------------------------------
text_notebook(
    "01_card_writer.ipynb",
    "NeuroAssist fine-tune 1/5 — Card writer (sentence simplifier)",
    "Teaches **Qwen2.5-1.5B-Instruct** to rewrite sentences in plain, short language. The app builds the "
    "accessible card from these rewrites; facts always come from the base models.\n\n"
    "**Data:** ASSET `validation` (2,000 sentences × up to 10 human simplifications) + WikiSplit `train` "
    "(sentence splitting). **Scored on:** ASSET `test` (never trained on) with SARI and reading grade level.",
    "evaluate sacrebleu sacremoses textstat",
    fill(r'''
MODEL_NAME  = "unsloth/Qwen2.5-1.5B-Instruct"
RUN_NAME    = "neuro-card-writer"
OLLAMA_NAME = "neuro-card-writer"       # name for `ollama create`
APP_SETTING = "NEUROASSIST_CARD_MODEL"  # how the app picks it up

# The app sends exactly these prompts. Change both here and in the app together.
SYSTEM_PROMPT   = "You are NeuroAssist, a reading assistant for students with dyslexia and ADHD. Rewrite the text in short, plain sentences. Keep every fact. Add nothing."
SIMPLIFY_PROMPT = "Rewrite this in simpler words:\n{input}"
SPLIT_PROMPT    = "Split this into shorter sentences:\n{input}"

ASSET_REFS_PER_SENTENCE = 4      # of ASSET's 10 references per sentence (2,000 sentences)
WIKISPLIT_ROWS          = 6000
EVAL_ROWS               = 359    # ASSET test sentences used for SARI (all = 359)

MAX_SEQ_LENGTH = 1024
LORA_RANK, LORA_ALPHA = 16, 16
EPOCHS, MAX_STEPS = 1, None      # MAX_STEPS=30 for a quick smoke test
LEARNING_RATE = 2e-4
BATCH_SIZE, GRAD_ACCUM = 16, 1
SEED = 3407
SAVE_TO_DRIVE, DRIVE_DIR = True, "/content/drive/MyDrive/neuroassist-finetunes"
GGUF_QUANT = "q4_k_m"
'''),
    r'''
from datasets import load_dataset

examples = []
asset = load_dataset("facebook/asset", "simplification", split="validation")
for row in asset:
    src = row["original"].strip()
    refs = [r.strip() for r in row["simplifications"] if r.strip() and r.strip() != src]
    for ref in random.sample(refs, min(ASSET_REFS_PER_SENTENCE, len(refs))):
        examples.append(chat(SIMPLIFY_PROMPT.format(input=src), ref))
n_asset = len(examples)

wiki = (load_dataset("cl-nagoya/wikisplit", split="train", streaming=True)
        .select_columns(["complex", "simple"]).shuffle(seed=SEED, buffer_size=20000))
for row in wiki.take(WIKISPLIT_ROWS):
    examples.append(chat(SPLIT_PROMPT.format(input=row["complex"].strip()), row["simple"].strip()))
random.shuffle(examples)
print(f"ASSET pairs {n_asset} · WikiSplit pairs {len(examples) - n_asset}")

# Held-out test: ASSET test split, never used for training.
import evaluate as hf_evaluate, textstat
sari = hf_evaluate.load("sari")
asset_test = load_dataset("facebook/asset", "simplification", split="test").select(range(EVAL_ROWS))

def evaluate_model():
    sources = [r["original"] for r in asset_test]
    preds = generate_batch([chat(SIMPLIFY_PROMPT.format(input=s)) for s in sources], max_new_tokens=128)
    refs = [r["simplifications"] for r in asset_test]
    return {"SARI (higher=better)": round(sari.compute(sources=sources, predictions=preds, references=refs)["sari"], 2) * 1.0,
            "grade level source": round(textstat.flesch_kincaid_grade(" ".join(sources)), 2) * 1.0,
            "grade level output": round(textstat.flesch_kincaid_grade(" ".join(preds)), 2) * 1.0,
            "example": preds[0][:60]}
''')

# --------------------------------------------------------------------------------------------
# 02 Coder
# --------------------------------------------------------------------------------------------
text_notebook(
    "02_coder.ipynb",
    "NeuroAssist fine-tune 2/5 — Coder",
    "Fine-tunes **Qwen2.5-Coder-3B-Instruct** on Python problems, with the app's exact system prompt and "
    "the practice pack's prompt style (`Complete this Python function:` + a stub with examples).\n\n"
    "**Data:** MBPP `full` train+validation+prompt (474 problems, turned into stub prompts) + "
    "`Vezora/Tested-143k-Python-Alpaca` (tested Python solutions). **HumanEval is filtered out** of "
    "the training data. **Scored on:** MBPP `test` with its unit tests.\n\n"
    "The stock model already scores 17/20 on the practice items, so the bar is high. Keep the fine-tune "
    "only if the app's eval improves.",
    "",
    fill(r'''
MODEL_NAME  = "unsloth/Qwen2.5-Coder-3B-Instruct"
RUN_NAME    = "neuro-coder"
OLLAMA_NAME = "neuro-coder"
APP_SETTING = "NEUROASSIST_CODE_MODEL"

SYSTEM_PROMPT = @@CODE_SYSTEM@@   # copied from the app (neuroassist/tools/code.py)

VEZORA_ROWS = 6000          # tested instruction → solution pairs
EVAL_ROWS   = 150           # MBPP test problems used for pass@1

MAX_SEQ_LENGTH = 2048
LORA_RANK, LORA_ALPHA = 16, 16
EPOCHS, MAX_STEPS = 1, None
LEARNING_RATE = 1e-4        # lower than usual: protect what the model already does well
BATCH_SIZE, GRAD_ACCUM = 8, 2
SEED = 3407
SAVE_TO_DRIVE, DRIVE_DIR = True, "/content/drive/MyDrive/neuroassist-finetunes"
GGUF_QUANT = "q4_k_m"
''', CODE_SYSTEM=app_code.SYSTEM),
    r'''
import ast, subprocess, sys, tempfile
from datasets import load_dataset

# ---- HumanEval decontamination (HumanEval is used ONLY to filter it out) ----
def _toks(s):
    return re.findall(r"\w+|[^\w\s]", s)
humaneval = load_dataset("openai/openai_humaneval", split="test")
HE_NAMES = {r["entry_point"] for r in humaneval if len(r["entry_point"]) >= 8}
HE_GRAMS = set()
for r in humaneval:
    t = _toks(r["prompt"] + r["canonical_solution"])
    HE_GRAMS |= {" ".join(t[i:i + 12]) for i in range(len(t) - 11)}

def contaminated(text):
    if any(re.search(rf"\b{re.escape(n)}\b", text) for n in HE_NAMES):
        return True
    t = _toks(text)
    return any(" ".join(t[i:i + 12]) in HE_GRAMS for i in range(len(t) - 11))

def fence(code):
    return f"```python\n{code.strip()}\n```"

def mbpp_prompt(row):
    """MBPP problem → the practice pack's 'Complete this Python function' format."""
    code = row["code"].replace("\r\n", "\n").replace("\t", "    ")
    m = re.search(r"assert\s+(?:not\s+)?\(?\s*(\w+)\s*\(", row["test_list"][0])
    sig = m and re.search(rf"^def {m.group(1)}\s*\(.*?\)\s*(?:->.*?)?:", code, re.M | re.S)
    if not sig:
        return None, code
    shown = []
    for t in row["test_list"][:2]:
        mm = re.match(r"assert\s+(.*?)\s*==\s*(.*)$", t.strip())
        if mm:
            shown.append(f"    >>> {mm.group(1)}\n    {mm.group(2)}")
    imports = "\n".join(l for l in code.splitlines() if l.startswith(("import ", "from ")))
    stub = (imports + "\n\n\n" if imports else "") + sig.group(0) + '\n    """ ' + row["text"].strip() + "\n" \
           + "\n".join(shown) + '\n    """\n'
    return "Complete this Python function:\n\n\n" + stub, code

examples, dropped = [], 0
for split_name in ("train", "validation", "prompt"):
    for row in load_dataset("google-research-datasets/mbpp", "full", split=split_name):
        prompt, code = mbpp_prompt(row)
        if not prompt:
            continue
        if contaminated(prompt + code):
            dropped += 1
            continue
        examples.append(chat(prompt, fence(code)))
n_mbpp = len(examples)

def extract(out):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", out, re.S)
    code = max(blocks, key=len) if blocks else out
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code if "def " in code and len(code) < 4000 else None

vez = load_dataset("Vezora/Tested-143k-Python-Alpaca", split="train", streaming=True).shuffle(seed=SEED, buffer_size=20000)
for row in vez:
    if len(examples) - n_mbpp >= VEZORA_ROWS:
        break
    instr = (row["instruction"] + ("\n\n" + row["input"] if row["input"] else "")).strip()
    code = extract(row["output"])
    if not code:
        continue
    if contaminated(instr + code):
        dropped += 1
        continue
    examples.append(chat(instr, fence(code)))
random.shuffle(examples)
print(f"MBPP {n_mbpp} · Vezora {len(examples) - n_mbpp} · dropped as HumanEval-like: {dropped}")

# ---- Held-out test: MBPP test split with its unit tests ----
mbpp_test = [r for r in load_dataset("google-research-datasets/mbpp", "full", split="test")
             if mbpp_prompt(r)[0]][:EVAL_ROWS]

def passes(code, row):
    src = (row["test_setup_code"] or "") + "\n" + code + "\n\n" + "\n".join(row["test_list"])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        return subprocess.run([sys.executable, f.name], capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False

def evaluate_model():
    outs = generate_batch([chat(mbpp_prompt(r)[0]) for r in mbpp_test], max_new_tokens=512, batch_size=16)
    ok = sum(passes(extract(o) or "", r) for o, r in zip(outs, mbpp_test))
    return {"MBPP test pass@1": round(ok / len(mbpp_test), 3), "passed": f"{ok}/{len(mbpp_test)}"}
''')

# --------------------------------------------------------------------------------------------
# 03 Document Q&A
# --------------------------------------------------------------------------------------------
text_notebook(
    "03_document_qa.ipynb",
    "NeuroAssist fine-tune 3/5 — Document Q&A (answer or say \"not there\")",
    "Fine-tunes **Qwen2.5-3B-Instruct** to answer questions about a document in the app's exact JSON "
    "format `{found, quote, answer}`, and to say `found: false` when the document doesn't contain the "
    "answer. The app then checks the quote really is in the document.\n\n"
    "**Data:** SQuAD 2.0 **train** (the practice documents come from the *dev* split, so there's no overlap). "
    "Each document is the question's paragraph plus a few unrelated paragraphs, to match the practice "
    "pack's multi-paragraph documents. **Scored on:** held-out SQuAD train articles not used for training.\n\n"
    "Ask the organizers if SQuAD *train* is OK, since the practice documents come from SQuAD *dev*.",
    "",
    fill(r'''
MODEL_NAME  = "unsloth/Qwen2.5-3B-Instruct"
RUN_NAME    = "neuro-doc-qa"
OLLAMA_NAME = "neuro-doc-qa"
APP_SETTING = "NEUROASSIST_TEXT_MODEL"

# Copied from the app (neuroassist/tools/text_qa.py). Must match exactly.
SYSTEM_TEMPLATE = @@SYSTEM@@
PROMPT_TEMPLATE = @@PROMPT@@
SYSTEM_PROMPT = SYSTEM_TEMPLATE.format(source="document")

TRAIN_ROWS           = 8000
UNANSWERABLE_SHARE   = 0.40     # practice pack: 5 of 20
EXTRA_PARAGRAPHS     = (0, 3)   # unrelated paragraphs added around the real one
HELDOUT_ARTICLES     = 0.05     # share of SQuAD train articles kept out for scoring
EVAL_ROWS            = 300

MAX_SEQ_LENGTH = 3072
LORA_RANK, LORA_ALPHA = 16, 16
EPOCHS, MAX_STEPS = 1, None
LEARNING_RATE = 2e-4
BATCH_SIZE, GRAD_ACCUM = 8, 2
SEED = 3407
SAVE_TO_DRIVE, DRIVE_DIR = True, "/content/drive/MyDrive/neuroassist-finetunes"
GGUF_QUANT = "q4_k_m"
''', SYSTEM=app_text.SYSTEM, PROMPT=app_text.PROMPT),
    r'''
from collections import defaultdict
from datasets import load_dataset

squad = load_dataset("rajpurkar/squad_v2", split="train")
titles = sorted(set(squad["title"]))
random.shuffle(titles)
heldout = set(titles[: max(1, int(len(titles) * HELDOUT_ARTICLES))])
paragraphs = sorted(set(squad["context"]))

SENT = re.compile(r"[^.!?]+(?:[.!?]+[\"')\]]*)?\s*")
def sentence_at(context, start):
    for m in SENT.finditer(context):
        if m.start() <= start < m.end():
            return m.group(0).strip()
    return None

def build(row):
    """(conversation, target dict) in the app's format, or None if unusable."""
    k = random.randint(*EXTRA_PARAGRAPHS)
    others = [p for p in random.sample(paragraphs, k + 3) if p != row["context"]][:k]
    paras = others + [row["context"]]
    random.shuffle(paras)
    doc = "\n\n".join(paras)
    if row["answers"]["text"]:
        ans, start = row["answers"]["text"][0], row["answers"]["answer_start"][0]
        quote = sentence_at(row["context"], start)
        if not quote or ans not in quote:
            return None
        target = {"found": True, "quote": quote, "answer": ans}
    else:
        target = {"found": False, "quote": "", "answer": ""}
    user = PROMPT_TEMPLATE.format(Source="Document", source="document", doc=doc, question=row["question"])
    return user, target, doc

def sample(rows, n):
    ans = [r for r in rows if r["answers"]["text"]]
    no = [r for r in rows if not r["answers"]["text"]]
    n_no = int(n * UNANSWERABLE_SHARE)
    picked = random.sample(no, min(n_no, len(no))) + random.sample(ans, min(n - n_no, len(ans)))
    random.shuffle(picked)
    return picked

train_rows = [r for r in squad if r["title"] not in heldout]
test_rows = [r for r in squad if r["title"] in heldout]
examples = []
for r in sample(train_rows, TRAIN_ROWS):
    b = build(r)
    if b:
        examples.append(chat(b[0], json.dumps(b[1], ensure_ascii=False)))
print(f"{len(examples)} training examples · {len(heldout)} held-out articles")

eval_set = [b for b in (build(r) for r in sample(test_rows, EVAL_ROWS)) if b]

def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()

def evaluate_model():
    outs = generate_batch([chat(u) for u, _, _ in eval_set], max_new_tokens=200)
    valid = ans_ok = ans_n = no_ok = no_n = quote_ok = 0
    for out, (_, gold, doc) in zip(outs, eval_set):
        try:
            p = json.loads(out); valid += 1
        except Exception:
            p = {}
        if gold["found"]:
            ans_n += 1
            hit = bool(p.get("found")) and _norm(gold["answer"]) in _norm(p.get("answer", "")) or \
                  (p.get("answer") and _norm(p["answer"]) in _norm(gold["answer"]))
            ans_ok += bool(hit)
            quote_ok += bool(p.get("quote")) and _norm(p["quote"]) in _norm(doc)
        else:
            no_n += 1
            no_ok += not p.get("found", False)
    return {"valid JSON": round(valid / len(outs), 3),
            "answerable correct": round(ans_ok / max(ans_n, 1), 3),
            "quote really in doc": round(quote_ok / max(ans_n, 1), 3),
            "unanswerable -> not found": round(no_ok / max(no_n, 1), 3)}
''')

# ============================================================================================
# 04 Vision + video
# ============================================================================================
IMAGE_PROMPTS = dict(app_image.PROMPTS)
VIDEO_HINTS = dict(app_video.HINTS)
ROUTER_TASKS = [(t, p) for t, p in router.TASKS]
ALLOWED_IMAGE = sorted(router.ALLOWED["image"])
ALLOWED_VIDEO = sorted(router.ALLOWED["video"])
DENSE_TASKS = sorted(app_video.DENSE_TASKS)

VISION_SETTINGS = fill(r'''
MODEL_NAME  = "unsloth/Qwen2.5-VL-3B-Instruct"
RUN_NAME    = "neuro-vision"
OLLAMA_NAME = "neuro-vision"
APP_SETTING = "NEUROASSIST_VISION_MODEL"

# FIRST RUN: keep True. Trains 10 steps, exports, and loads the result in Ollama inside Colab,
# proving the whole path works before you spend an hour training. Then set False and Run all.
SMOKE_TEST = True

# Image data: Cauldron TRAIN subsets (examples per subset)
CAULDRON = {"tallyqa": 2500,   # counting  (COCO train / Visual Genome images)
            "textvqa": 2000,   # reading text in photos
            "st_vqa": 1500,    # reading scene text
            "vqav2": 2000}     # general questions (COCO train2014 images)
# Video data: synthetic clips generated in this notebook (shapes, motion, signs, counting)
SYNTH_VIDEOS = 3000
EVAL_IMAGES, EVAL_VIDEOS = 100, 100

# Must match the app's "standard" profile (neuroassist/config.py)
VIDEO_FPS_DENSE, VIDEO_FPS_SPARSE = 4.0, 1.0
VIDEO_MAX_FRAMES, VIDEO_SIDE = 16, 448
IMAGE_SIDE = 768                 # training images are capped here to save memory

MAX_SEQ_LENGTH = 6144            # 16 frames x ~256 tokens + text
LORA_RANK, LORA_ALPHA = 16, 16
EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE, GRAD_ACCUM = 2, 4
SEED = 3407
SAVE_TO_DRIVE, DRIVE_DIR = True, "/content/drive/MyDrive/neuroassist-finetunes"

# ---- Copied from the app at build time (do not edit; re-run finetune/build_notebooks.py) ----
IMAGE_PROMPTS = @@IMAGE_PROMPTS@@
COUNT_SUFFIX  = " Count carefully, including small, partly hidden or background ones."
VIDEO_HINTS   = @@VIDEO_HINTS@@
ROUTER_TASKS  = @@ROUTER_TASKS@@
ALLOWED_IMAGE, ALLOWED_VIDEO, DENSE_TASKS = set(@@ALLOWED_IMAGE@@), set(@@ALLOWED_VIDEO@@), set(@@DENSE_TASKS@@)
''', IMAGE_PROMPTS=IMAGE_PROMPTS, VIDEO_HINTS=VIDEO_HINTS, ROUTER_TASKS=ROUTER_TASKS,
                       ALLOWED_IMAGE=ALLOWED_IMAGE, ALLOWED_VIDEO=ALLOWED_VIDEO, DENSE_TASKS=DENSE_TASKS)

APP_FORMATS = r'''
import io, json, random, re
import numpy as np, cv2
from PIL import Image

def app_task(question, allowed, default="describe"):
    """Same rules as the app's router (neuroassist/router.py)."""
    for task, pat in ROUTER_TASKS:
        if task in allowed and re.search(pat, question, re.I):
            return task
    return default

NUM_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
def to_int(s):
    s = s.strip().lower().rstrip(".")
    return int(s) if s.isdigit() else NUM_WORDS.get(s)

def image_example(question, answer):
    """(prompt, target) exactly as the app asks and parses (neuroassist/tools/image.py)."""
    task = app_task(question, ALLOWED_IMAGE)
    if task == "count":                     # trains the app's COUNT_MODE=direct path
        n = to_int(answer)
        if n is None:
            return None
        return question + COUNT_SUFFIX, json.dumps({"count": n, "detail": ""})
    prompt = IMAGE_PROMPTS.get(task, IMAGE_PROMPTS["describe"]).format(q=question)
    return prompt, json.dumps({"answer": answer, "detail": ""})

def video_prompt(question, times, duration, task):
    """Exactly the app's direct video prompt (neuroassist/tools/video.py)."""
    labels = "\n".join(f"Image {i + 1} is at t={t:.1f}s." for i, t in enumerate(times))
    return (f"These {len(times)} images are frames from a {duration:.1f}-second video, in time order.\n"
            f"{labels}\n{VIDEO_HINTS.get(task, '')}\n\nQuestion: {question}\n"
            "Answer in one short, direct sentence.")

def frame_changed(a, b, mean_thresh=1.5, local_thresh=30.0):
    """Same test as the app (neuroassist/ingest.py): whole-picture change OR a sharp local change."""
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
    if np.abs(cv2.resize(ga, (32, 32)).astype(np.float32) - cv2.resize(gb, (32, 32)).astype(np.float32)).mean() >= mean_thresh:
        return True
    d = np.abs(cv2.resize(ga, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
               - cv2.resize(gb, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32))
    return float(np.percentile(d, 99)) > local_thresh

def sample_frames(frames, native_fps, fps, max_frames, side):
    """Same sampling as the app (neuroassist/ingest.py): step, keep last, drop unchanged, cap."""
    step = max(1, round(native_fps / fps))
    idx = [i for i in range(len(frames)) if i % step == 0]
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)
    kept = []
    for j, i in enumerate(idx):
        if kept and j != len(idx) - 1 and not frame_changed(frames[kept[-1]], frames[i]):
            continue
        kept.append(i)
    if len(kept) > max_frames:
        kept = [kept[s] for s in sorted(set(np.linspace(0, len(kept) - 1, max_frames).round().astype(int)))]
    out = []
    for i in kept:
        f = frames[i]; h, w = f.shape[:2]; s = side / max(h, w)
        if s < 1:
            f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        out.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    return out, [i / native_fps for i in kept], len(frames) / native_fps

def to_jpeg(img, side):
    img = img.convert("RGB"); img.thumbnail((side, side))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=90); return buf.getvalue()
'''

SYNTH = r'''
# Synthetic clips with known answers. Our own generator, not the organizers' clips: random shapes,
# colors, sizes, speeds, backgrounds and wording, so the model learns the skills, not the clips.
W, H, NATIVE_FPS = 640, 360, 10
COLORS = {"red": (40, 40, 230), "blue": (220, 90, 30), "green": (60, 170, 40), "yellow": (30, 210, 240),
          "orange": (20, 140, 250), "purple": (170, 50, 140), "pink": (180, 120, 250), "black": (20, 20, 20)}
BGS = {"white": (245, 245, 245), "gray": (150, 150, 150), "light blue": (240, 220, 190), "beige": (200, 230, 240)}
SHAPES = ["circle", "square", "triangle"]
WORD_PAIRS = [("OPEN", "CLOSED"), ("PUSH", "PULL"), ("STOP", "GO"), ("IN", "OUT"), ("ON", "OFF"), ("YES", "NO"),
              ("ENTER", "EXIT"), ("HOT", "COLD"), ("UP", "DOWN"), ("SALE", "SOLD"), ("WALK", "WAIT"), ("FULL", "EMPTY")]

def canvas(bg):
    return np.full((H, W, 3), bg, np.uint8)

def draw(img, shape, cx, cy, r, color):
    cx, cy, r = int(cx), int(cy), int(r)
    if shape == "circle":
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    elif shape == "square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
    else:
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)

def text(img, s, scale=3.0, color=(20, 20, 20)):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 6)
    cv2.putText(img, s, ((W - tw) // 2, (H + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 6, cv2.LINE_AA)

def plural(n, word):
    return word if n == 1 else word + "s"

def grid_positions(rng, n, r):
    cols = max(1, (W - 40) // (2 * r + 20)); rows = max(1, (H - 40) // (2 * r + 20))
    cells = rng.sample([(c, q) for c in range(cols) for q in range(rows)], n)
    return [(40 + r + c * (2 * r + 20), 40 + r + q * (2 * r + 20)) for c, q in cells]

def clip(rng):
    """Returns (frames_bgr, question, answer). Ten templates, varied wording."""
    t = rng.randrange(10)
    bg_name = rng.choice(list(BGS)); bg = BGS[bg_name]
    cols = [c for c in COLORS if not (bg_name == "white" and c == "yellow")]
    shape, color = rng.choice(SHAPES), rng.choice(cols)
    r = rng.randint(22, 40)
    frames = []
    if t == 0:                                            # shapes appear one at a time
        n = rng.randint(2, 9); pos = grid_positions(rng, n, r); per = rng.randint(4, 8)
        for k in range(n * per + per):
            img = canvas(bg)
            for i in range(min(n, k // per + 1) if k >= per else 0):
                draw(img, shape, *pos[i], r, COLORS[color])
            frames.append(img)
        q = rng.choice(["How many shapes appear in this video?", f"How many {shape}s appear?",
                        "Count the shapes that appear in the video."])
        a = f"{n} {plural(n, shape)} appear one at a time."
    elif t == 1:                                          # moves in one direction
        d = rng.choice(["left", "right", "up", "down"]); steps = rng.randint(20, 40)
        x0, y0 = (W * 0.75, H / 2) if d == "left" else (W * 0.25, H / 2) if d == "right" else \
                 (W / 2, H * 0.75) if d == "up" else (W / 2, H * 0.25)
        dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[d]
        span = (W if dx else H) * 0.5
        for k in range(steps):
            img = canvas(bg); f = k / (steps - 1)
            draw(img, shape, x0 + dx * span * f, y0 + dy * span * f, r, COLORS[color]); frames.append(img)
        q = rng.choice([f"Which direction does the {shape} move?", f"Which way is the {color} {shape} moving?",
                        f"In which direction does the {shape} move?"])
        a = f"The {color} {shape} moves {d}."
    elif t == 2:                                          # three colors in order
        c3 = rng.sample(cols, 3); per = rng.randint(8, 14)
        for c in c3:
            for _ in range(per):
                img = canvas(bg); draw(img, shape, W / 2, H / 2, r * 2, COLORS[c]); frames.append(img)
        q = rng.choice(["What color appears first, and what color appears last?",
                        "Which color is shown first and which is shown last?"])
        a = f"A {c3[0]} {shape} appears first, then {c3[1]}, and {c3[2]} appears last."
    elif t == 3:                                          # sign text changes
        w1, w2 = rng.choice(WORD_PAIRS)
        if rng.random() < 0.5:
            w1, w2 = w2, w1
        per = rng.randint(10, 20)
        for w in (w1, w2):
            for _ in range(per):
                img = canvas(bg); cv2.rectangle(img, (120, 90), (W - 120, H - 90), (255, 255, 255), -1)
                cv2.rectangle(img, (120, 90), (W - 120, H - 90), (30, 30, 30), 4); text(img, w, 2.4); frames.append(img)
        q = rng.choice(["What does the sign say at the start and at the end?",
                        "What does the sign say first, and what does it say at the end?"])
        a = f'The sign says "{w1}" at the start and "{w2}" at the end.'
    elif t == 4:                                          # bar grows or shrinks
        grow = rng.random() < 0.5; steps = rng.randint(20, 40); lo, hi = rng.randint(20, 80), rng.randint(220, 320)
        for k in range(steps):
            f = k / (steps - 1); h = lo + (hi - lo) * (f if grow else 1 - f)
            img = canvas(bg); cv2.rectangle(img, (W // 2 - 40, int(H - 20 - h)), (W // 2 + 40, H - 20), COLORS[color], -1)
            frames.append(img)
        q = rng.choice(["Does the bar get taller or shorter?", "Is the bar getting taller or shorter?"])
        a = f"The bar gets {'taller' if grow else 'shorter'}."
    elif t == 5:                                          # two shapes meet / move apart
        meet = rng.random() < 0.6; steps = rng.randint(20, 35); c2 = rng.choice([c for c in cols if c != color])
        for k in range(steps):
            f = k / (steps - 1); gap = (W * 0.35) * ((1 - f) if meet else f) + (r if meet else r * 1.2)
            img = canvas(bg)
            draw(img, shape, W / 2 - gap, H / 2, r, COLORS[color]); draw(img, shape, W / 2 + gap, H / 2, r, COLORS[c2])
            frames.append(img)
        q = rng.choice(["What happens to the two shapes?", "Describe what the two shapes do."])
        a = ("The two shapes move toward each other until they touch." if meet
             else "The two shapes move away from each other.")
    elif t == 6:                                          # countdown / count up
        k = rng.randint(3, 9); down = rng.random() < 0.7; per = rng.randint(6, 10)
        seq = list(range(k, 0, -1)) if down else list(range(1, k + 1))
        for v in seq:
            for _ in range(per):
                img = canvas(bg); text(img, str(v), 5.0, COLORS[color]); frames.append(img)
        q = rng.choice(["What number is shown last?", "What is the last number shown?"])
        a = (f"A countdown from {k} to 1; the last number shown is 1." if down
             else f"The numbers count up from 1 to {k}; the last number shown is {k}.")
    elif t == 7:                                          # appear, then some disappear
        n = rng.randint(3, 8); m = rng.randint(1, n - 1); pos = grid_positions(rng, n, r); per = rng.randint(4, 7)
        shown = []
        for i in range(n):
            shown.append(i)
            for _ in range(per):
                img = canvas(bg); [draw(img, shape, *pos[j], r, COLORS[color]) for j in shown]; frames.append(img)
        for i in rng.sample(range(n), m):
            shown.remove(i)
            for _ in range(per):
                img = canvas(bg); [draw(img, shape, *pos[j], r, COLORS[color]) for j in shown]; frames.append(img)
        left = n - m
        q = rng.choice(["Several shapes appear and then some disappear. How many shapes are left at the end?",
                        "How many shapes are left at the end?"])
        a = (f"{n} {plural(n, shape)} appear, {m} {'disappears' if m == 1 else 'disappear'}, "
             f"so {left} {'is' if left == 1 else 'are'} left at the end.")
    elif t == 8:                                          # ball bounces k times
        k = rng.randint(1, 5); per = rng.randint(10, 14); floor = H - 30; top = rng.randint(60, 140)
        for b in range(k):
            for s in range(per):
                f = s / per; y = floor - r - (floor - r - top) * (1 - (2 * f - 1) ** 2)
                img = canvas(bg); cv2.line(img, (0, floor), (W, floor), (60, 60, 60), 3)
                draw(img, "circle", 80 + (W - 160) * (b * per + s) / (k * per), y, r, COLORS[color]); frames.append(img)
        q = rng.choice(["How many times does the ball bounce?", "How many bounces does the ball make?"])
        a = f"The ball bounces {k} {plural(k, 'time')}."
    else:                                                 # background color changes
        c1, c2 = rng.sample(list(COLORS), 2); per = rng.randint(10, 20)
        for c in (c1, c2):
            for _ in range(per):
                frames.append(canvas(COLORS[c]))
        if rng.random() < 0.5:
            q, a = "What color is the background at the end?", f"The background ends {c2}."
        else:
            q, a = "What color is the background at the start?", f"The background starts {c1}."
    return frames, q, a

def video_example(seed):
    rng = random.Random(seed)
    frames, q, a = clip(rng)
    task = app_task(q, ALLOWED_VIDEO)
    fps = VIDEO_FPS_DENSE if task in DENSE_TASKS else VIDEO_FPS_SPARSE
    imgs, times, dur = sample_frames(frames, NATIVE_FPS, fps, VIDEO_MAX_FRAMES, VIDEO_SIDE)
    return imgs, video_prompt(q, times, dur, task), json.dumps({"answer": a}), a

imgs, prompt, target, _ = video_example(1)
print(prompt, "\n→", target, f"\n({len(imgs)} frames, {imgs[0].size})")
imgs[0]
'''

VISION_DATA = r'''
import torch
from datasets import load_dataset

def cauldron(subset, n):
    """n (jpeg, prompt, target) examples from a Cauldron TRAIN subset, in the app's formats."""
    out = []
    ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True).shuffle(seed=SEED, buffer_size=1000)
    for row in ds:
        if len(out) >= n:
            break
        if len(row["images"]) != 1:
            continue
        turn = random.choice(row["texts"])
        q = turn["user"].split("\n")[0].strip()          # drop Cauldron's "Give a very brief answer." line
        a = turn["assistant"].strip().rstrip(".")
        made = image_example(q, a)
        if made:
            out.append((to_jpeg(row["images"][0], IMAGE_SIDE),) + made + (a,))
    print(f"{subset:<8} {len(out)}")
    return out

scale = 0.02 if SMOKE_TEST else 1.0
image_items, image_eval = [], []
for subset, n in CAULDRON.items():
    rows = cauldron(subset, int(n * scale) + (EVAL_IMAGES // len(CAULDRON)))
    image_eval += rows[: EVAL_IMAGES // len(CAULDRON)]
    image_items += rows[EVAL_IMAGES // len(CAULDRON):]
video_seeds = list(range(int(SYNTH_VIDEOS * scale) + 10))
video_eval_seeds = list(range(10**6, 10**6 + EVAL_VIDEOS))      # different seeds = unseen clips

class Mixed(torch.utils.data.Dataset):
    """Built lazily: images stay as JPEG bytes, clips are rendered on demand."""
    def __init__(self):
        self.items = [("img", i) for i in range(len(image_items))] + [("vid", s) for s in video_seeds]
        random.Random(SEED).shuffle(self.items)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        kind, ref = self.items[i]
        if kind == "img":
            jpg, prompt, target, _ = image_items[ref]
            images = [Image.open(io.BytesIO(jpg)).convert("RGB")]
        else:
            images, prompt, target, _ = video_example(ref)
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        return {"messages": [{"role": "user", "content": content},
                             {"role": "assistant", "content": [{"type": "text", "text": target}]}]}

train_ds = Mixed()
print(f"training examples: {len(image_items)} images + {len(video_seeds)} clips = {len(train_ds)}")
'''

VISION_LOAD = r'''
import os
assert torch.cuda.is_available(), "No GPU. Runtime → Change runtime type → GPU"
print(torch.cuda.get_device_name(0))
OUT_DIR = f"/content/{RUN_NAME}"
if SAVE_TO_DRIVE and not SMOKE_TEST:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = f"{DRIVE_DIR}/{RUN_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)

from unsloth import FastVisionModel, is_bf16_supported
model, tokenizer = FastVisionModel.from_pretrained(MODEL_NAME, load_in_4bit=True, use_gradient_checkpointing="unsloth")
model = FastVisionModel.get_peft_model(
    model, finetune_vision_layers=True, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0, bias="none", random_state=SEED)
'''

VISION_EVAL = r'''
NUM_RE = re.compile(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.I)
KEY_WORDS = set(COLORS) | {"left", "right", "up", "down", "taller", "shorter", "toward", "away", "touch"}

def ask(images, prompt, max_new_tokens=96):
    msgs = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]}]
    text_in = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    enc = tokenizer(images=images, text=[text_in], add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

def field(out, key):
    try:
        return str(json.loads(out).get(key, ""))
    except Exception:
        return out

def video_correct(pred, ref):
    keys = [w for w in re.findall(r"[a-z]+|\d+", ref.lower()) if w in KEY_WORDS or w.isdigit()]
    keys += [w.lower() for w in re.findall(r'"([^"]+)"', ref)]
    p = pred.lower()
    if re.search(r"how many|what number|last number", ref.lower()) or not keys:
        return bool(NUM_RE.findall(ref)) and NUM_RE.findall(ref)[-1].lower() in [n.lower() for n in NUM_RE.findall(p)]
    return all(re.search(rf"\b{re.escape(k)}\b", p) for k in keys)

def evaluate_model():
    FastVisionModel.for_inference(model)
    img_ok = 0
    for jpg, prompt, target, ans in image_eval:
        out = ask([Image.open(io.BytesIO(jpg)).convert("RGB")], prompt)
        gold = json.loads(target)
        got = field(out, "count" if "count" in gold else "answer").strip().lower().rstrip(".")
        img_ok += got == str(gold.get("count", gold.get("answer"))).lower()
    vid_ok = 0
    for s in video_eval_seeds[: 20 if SMOKE_TEST else len(video_eval_seeds)]:
        images, prompt, _, ref = video_example(s)
        vid_ok += video_correct(field(ask(images, prompt), "answer"), ref)
    FastVisionModel.for_training(model)
    n_vid = 20 if SMOKE_TEST else len(video_eval_seeds)
    return {"image exact match": round(img_ok / len(image_eval), 3),
            "synthetic video correct": round(vid_ok / n_vid, 3)}

print("Scoring the base model before training...")
before = evaluate_model()
print(before)
'''

VISION_TRAIN = r'''
import inspect
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

try:      # answers-only loss (newer Unsloth); falls back to full-sequence loss
    collator = UnslothVisionDataCollator(model, tokenizer, train_on_responses_only=True,
                                         instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
except TypeError:
    collator = UnslothVisionDataCollator(model, tokenizer)

wanted = dict(
    output_dir=f"/content/checkpoints/{RUN_NAME}",
    per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS, max_steps=10 if SMOKE_TEST else -1,
    learning_rate=LEARNING_RATE, warmup_ratio=0.03, lr_scheduler_type="cosine",
    optim="adamw_8bit", weight_decay=0.01, bf16=is_bf16_supported(), fp16=not is_bf16_supported(),
    logging_steps=5 if SMOKE_TEST else 20, save_strategy="no", seed=SEED, report_to="none",
    remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
    max_length=MAX_SEQ_LENGTH, max_seq_length=MAX_SEQ_LENGTH, dataloader_num_workers=2)
accepted = inspect.signature(SFTConfig.__init__).parameters
if "max_length" in accepted:
    wanted.pop("max_seq_length")
args = SFTConfig(**{k: v for k, v in wanted.items() if k in accepted})
kw = dict(model=model, train_dataset=train_ds, data_collator=collator, args=args)
kw["processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
   else "tokenizer"] = tokenizer
FastVisionModel.for_training(model)
trainer = SFTTrainer(**kw)
stats = trainer.train()
print(f"done in {stats.metrics['train_runtime'] / 60:.1f} min · loss {stats.metrics['train_loss']:.3f}")

after = evaluate_model()
print(f"{'metric':<28}{'before':>8}{'after':>8}")
for k in before:
    print(f"{k:<28}{before[k]:>8.3f}{after[k]:>8.3f}")
json.dump({"model": MODEL_NAME, "before": before, "after": after}, open(f"{OUT_DIR}/scores.json", "w"), indent=2)
'''

VISION_EXPORT = r'''
model.save_pretrained(f"{OUT_DIR}/lora"); tokenizer.save_pretrained(f"{OUT_DIR}/lora")
merged = f"{OUT_DIR}/merged"
model.save_pretrained_merged(merged, tokenizer, save_method="merged_16bit")   # full weights, ~7.5 GB
open(f"{merged}/Modelfile", "w").write("FROM .\nPARAMETER temperature 0\n")
print("Merged model + Modelfile in", merged)
print(f"""
On the demo laptop, copy the whole 'merged' folder from Drive, then inside it:
    ollama create {OLLAMA_NAME} --quantize q4_K_M -f Modelfile
and run the app / eval with:
    $env:{APP_SETTING}="{OLLAMA_NAME}"
    $env:NEUROASSIST_COUNT_MODE="direct"        # this model was trained on direct counts
    $env:NEUROASSIST_VIDEO_MODE="direct"        # and on all frames in one call
    $env:NEUROASSIST_VIDEO_DIRECT_MAX_FRAMES="16"
""")
'''

VISION_OLLAMA_CHECK = r'''
# Proves Ollama can import and run the exported model, using Ollama installed inside Colab.
# If this fails, do NOT spend an hour training: keep the stock qwen3-vl:4b in the app instead.
import subprocess, time, base64, urllib.request
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
server = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)
r = subprocess.run(["ollama", "create", OLLAMA_NAME, "--quantize", "q4_K_M", "-f", "Modelfile"],
                   cwd=merged, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
assert r.returncode == 0, "ollama create failed: see the log above"
images, prompt, _, ref = video_example(10**6)
buf = io.BytesIO(); images[0].save(buf, "JPEG")
body = json.dumps({"model": OLLAMA_NAME, "stream": False, "format": "json",
                   "messages": [{"role": "user", "content": "What is in this image? Reply as JSON with key answer.",
                                 "images": [base64.b64encode(buf.getvalue()).decode()]}]}).encode()
reply = json.loads(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:11434/api/chat", body, {"Content-Type": "application/json"}), timeout=300).read())
print("Ollama answered:", reply["message"]["content"])
print("\n✅ Export path works. Set SMOKE_TEST = False in Settings and Run all for the real training.")
'''

write("04_vision_video.ipynb", [
    md("# NeuroAssist fine-tune 4/5 — Images + video (Qwen2.5-VL-3B)\n\n"
       "Fine-tunes **Qwen2.5-VL-3B-Instruct** on counting, reading text, general questions and short video "
       "clips, using the app's exact prompts and JSON formats.\n\n"
       "**Data:**\n"
       "- Images: HuggingFaceM4 Cauldron **train** subsets `tallyqa` (counting), `textvqa` + `st_vqa` "
       "(reading text), `vqav2` (general).\n"
       "- Video: synthetic clips generated in this notebook with known answers: shapes appearing, "
       "moving, bouncing, signs changing, countdowns. The frames are sampled the same way the app samples them.\n\n"
       "**Scored on:** held-out Cauldron images and unseen synthetic clips.\n\n"
       "**Do this first:** run once with `SMOKE_TEST = True` (about 10 minutes). It trains 10 steps, exports, "
       "and loads the model in Ollama inside Colab. Only if that works, set it to `False` and run the real "
       "training (about 1–2 h on an A100).\n\n"
       "**Keep it only if it beats stock `qwen3-vl:4b`** on the app's image and video eval.\n\n" + RULE),
    md("## 1. Install"),
    py("!pip install -q unsloth\n!pip install -q opencv-python-headless"),
    md("## 2. ⚙️ Settings"),
    py(VISION_SETTINGS),
    md("## 3. App formats (prompts, routing, frame sampling)"),
    py(APP_FORMATS),
    md("## 4. Synthetic video generator (preview below)"),
    py(SYNTH),
    md("## 5. Load the training data"),
    py(VISION_DATA),
    md("## 6. Load the model and add LoRA adapters"),
    py(VISION_LOAD),
    md("## 7. Score the base model (before)"),
    py(VISION_EVAL),
    md("## 8. Train, then score again"),
    py(VISION_TRAIN),
    md("## 9. Save adapters and the merged model for Ollama"),
    py(VISION_EXPORT),
    md("## 10. Check the export loads in Ollama (runs inside Colab)"),
    py(VISION_OLLAMA_CHECK),
])

# ============================================================================================
# 05 Speech (Whisper)
# ============================================================================================
write("05_speech_whisper.ipynb", [
    md("# NeuroAssist fine-tune 5/5 — Speech-to-text (Whisper small)\n\n"
       "Fine-tunes **openai/whisper-small** on LibriSpeech **train-clean-100**, then converts it to the "
       "CTranslate2 format that the app's faster-whisper loads.\n\n"
       "The practice recordings come from LibriSpeech *test-clean*, so this notebook never touches test-clean. "
       "It scores on *validation-clean* (word error rate, lower is better).\n\n"
       "**Expect a small gain at best.** Stock Whisper small already makes only about 3–4 word errors per "
       "100 here. LibriSpeech transcripts are uppercase with no punctuation, so they're converted to "
       "sentence case; the model may still lose some punctuation. Keep it only if the WER drops **and** "
       "transcripts still read well."),
    md("## 1. Install"),
    py("!pip install -q -U transformers datasets accelerate jiwer soundfile ctranslate2 faster-whisper"),
    md("## 2. ⚙️ Settings"),
    py(r'''
BASE_MODEL  = "openai/whisper-small"
RUN_NAME    = "neuro-whisper-small"
APP_SETTING = "NEUROASSIST_WHISPER_MODEL"

TRAIN_ROWS  = 6000           # utterances from LibriSpeech clean/train.100 (28,539 available)
EVAL_ROWS   = 400            # from clean/validation (never test: the practice audio is test-clean)
MAX_STEPS   = 1000           # ~2.7 epochs over 6,000 clips at batch 16
LEARNING_RATE = 1e-5
BATCH_SIZE  = 16
WARMUP_STEPS = 100
EVAL_EVERY  = 250
SEED = 3407
SAVE_TO_DRIVE, DRIVE_DIR = True, "/content/drive/MyDrive/neuroassist-finetunes"
'''),
    md("## 3. GPU and Google Drive"),
    py(r'''
import torch, os, io, json, random
import numpy as np
assert torch.cuda.is_available(), "No GPU. Runtime → Change runtime type → GPU"
print(torch.cuda.get_device_name(0))
random.seed(SEED)
OUT_DIR = f"/content/{RUN_NAME}"
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = f"{DRIVE_DIR}/{RUN_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)
'''),
    md("## 4. Load LibriSpeech (train-clean-100 and validation-clean)"),
    py(r'''
import soundfile as sf
from datasets import load_dataset, Audio

def sentence_case(t):
    t = t.strip().lower()
    t = re.sub(r"\bi\b", "I", t)
    return (t[:1].upper() + t[1:] + ".") if t else t

import re
def librispeech(split, n, shuffle):
    ds = load_dataset("openslr/librispeech_asr", "clean", split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))       # decode ourselves: no extra codec packages
    if shuffle:
        ds = ds.shuffle(seed=SEED, buffer_size=2000)
    rows = []
    for r in ds.take(n):
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if sr != 16000:
            a = np.interp(np.arange(0, len(a) * 16000 / sr) * sr / 16000, np.arange(len(a)), a).astype(np.float32)
        rows.append({"audio": a, "text": sentence_case(r["text"])})
    return rows

train_rows = librispeech("train.100", TRAIN_ROWS, shuffle=True)
eval_rows = librispeech("validation", EVAL_ROWS, shuffle=False)
print(len(train_rows), "train ·", len(eval_rows), "eval ·", "example:", train_rows[0]["text"][:80])
'''),
    md("## 5. Model, processor and batching"),
    py(r'''
from dataclasses import dataclass
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False

class Rows(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

@dataclass
class Collator:
    def __call__(self, batch):
        feats = processor.feature_extractor([b["audio"] for b in batch], sampling_rate=16000,
                                            return_tensors="pt").input_features
        lab = processor.tokenizer([b["text"] for b in batch], padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == model.config.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}

import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
normalize = BasicTextNormalizer()

def compute_metrics(pred):
    ids, out = pred.label_ids, pred.predictions
    ids[ids == -100] = processor.tokenizer.pad_token_id
    out[out == -100] = processor.tokenizer.pad_token_id
    hyp = processor.tokenizer.batch_decode(out, skip_special_tokens=True)
    ref = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
    pairs = [(normalize(r), normalize(h)) for r, h in zip(ref, hyp) if normalize(r).strip()]
    return {"wer": jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])}
'''),
    md("## 6. Train (with word error rate before and after)"),
    py(r'''
import inspect
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

wanted = dict(output_dir=f"/content/checkpoints/{RUN_NAME}", per_device_train_batch_size=BATCH_SIZE,
              per_device_eval_batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, warmup_steps=WARMUP_STEPS,
              max_steps=MAX_STEPS, gradient_checkpointing=True, bf16=torch.cuda.is_bf16_supported(),
              fp16=not torch.cuda.is_bf16_supported(), eval_strategy="steps", eval_steps=EVAL_EVERY,
              predict_with_generate=True, generation_max_length=225, logging_steps=25, save_strategy="no",
              report_to="none", remove_unused_columns=False, seed=SEED)
accepted = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
args = Seq2SeqTrainingArguments(**{k: v for k, v in wanted.items() if k in accepted})
kw = dict(model=model, args=args, train_dataset=Rows(train_rows), eval_dataset=Rows(eval_rows),
          data_collator=Collator(), compute_metrics=compute_metrics)
kw["processing_class" if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
   else "tokenizer"] = processor.feature_extractor
trainer = Seq2SeqTrainer(**kw)

before = trainer.evaluate()["eval_wer"]
print(f"WER before: {before:.4f}")
trainer.train()
after = trainer.evaluate()["eval_wer"]
print(f"WER before {before:.4f} → after {after:.4f}  ({'better' if after < before else 'NOT better: keep stock Whisper'})")
json.dump({"model": BASE_MODEL, "wer_before": before, "wer_after": after}, open(f"{OUT_DIR}/scores.json", "w"), indent=2)
'''),
    md("## 7. Convert for faster-whisper and test it"),
    py(r'''
import subprocess, shutil
merged = f"/content/{RUN_NAME}_hf"
model.save_pretrained(merged); processor.save_pretrained(merged)
ct2 = f"/content/{RUN_NAME}_ct2"
shutil.rmtree(ct2, ignore_errors=True)
r = subprocess.run(["ct2-transformers-converter", "--model", merged, "--output_dir", ct2, "--quantization", "int8",
                    "--copy_files", "tokenizer.json", "preprocessor_config.json"], capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
assert r.returncode == 0, "Conversion failed (see log). The Hugging Face copy is still saved below."

from faster_whisper import WhisperModel
fw = WhisperModel(ct2, device="cpu", compute_type="int8")     # same settings as the app
sf.write("/content/sample.wav", eval_rows[0]["audio"], 16000)
segs, _ = fw.transcribe("/content/sample.wav", beam_size=5, vad_filter=True, language="en")
print("REF:", eval_rows[0]["text"]); print("OUT:", " ".join(s.text.strip() for s in segs))

shutil.copytree(ct2, f"{OUT_DIR}/ct2", dirs_exist_ok=True)
shutil.copytree(merged, f"{OUT_DIR}/hf", dirs_exist_ok=True)
print(f"""
Saved to {OUT_DIR}/ct2. On the demo laptop, copy that folder (e.g. to C:\\models\\{RUN_NAME}) and set:
    $env:{APP_SETTING}="C:\\models\\{RUN_NAME}"
""")
'''),
])

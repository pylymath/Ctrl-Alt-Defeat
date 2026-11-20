"""Model setup for a fresh machine. install.ps1 / install.sh run this after creating the venv:

    python -m neuroassist.installer                         # stock models only
    python -m neuroassist.installer --finetunes D:\\models   # + our fine-tunes from a folder / USB stick
    python -m neuroassist.installer --hf-repo USER/neuroassist-finetunes   # + fine-tunes from Hugging Face
    python -m neuroassist.installer --check                 # only report what is installed and working

Steps: make sure Ollama is running -> pick a profile for this hardware -> pull the stock models ->
import the fine-tuned document-QA model (if given) -> download Whisper + the sound classifier ->
write neuroassist.env -> load the main model -> one live test question. Safe to run again: finished
steps are skipped."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "neuroassist.env"
FINETUNE = "neuro-doc-qa"            # the one fine-tune that beat stock (text 5->16/20, audio 14->17/20)

MODELFILE = '''FROM ./{gguf}
TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0
'''

# Settings for an Ollama we start ourselves (same as scripts/ollama_settings.ps1)
OLLAMA_ENV = {"OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "2", "OLLAMA_FLASH_ATTENTION": "1",
              "OLLAMA_KV_CACHE_TYPE": "q8_0"}


def say(msg: str, ok: bool | None = None):
    mark = {True: "OK  ", False: "FAIL", None: "..  "}[ok]
    print(f"[{mark}] {msg}", flush=True)


# --- Ollama ---------------------------------------------------------------------------------------
def host() -> str:
    h = os.environ.get("NEUROASSIST_OLLAMA_HOST") or "http://127.0.0.1:11434"
    return h.replace("0.0.0.0", "127.0.0.1")


def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(host() + "/api/version", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_ollama() -> bool:
    if ollama_up():
        say("Ollama is running", True)
        return True
    exe = shutil.which("ollama")
    if not exe:
        say("Ollama is not installed. Install it from https://ollama.com/download and run this again.", False)
        return False
    say("Starting Ollama (one request at a time, 8-bit context memory)")
    env = {**os.environ, **OLLAMA_ENV, "OLLAMA_HOST": "127.0.0.1:11434"}
    flags = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0     # DETACHED_PROCESS | NEW_PROCESS_GROUP
    subprocess.Popen([exe, "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=flags, start_new_session=sys.platform != "win32")
    for _ in range(40):
        time.sleep(0.5)
        if ollama_up():
            say("Ollama started", True)
            return True
    say("Ollama did not start. Open the Ollama app, then run this again.", False)
    return False


def installed() -> set[str]:
    with urllib.request.urlopen(host() + "/api/tags", timeout=10) as r:
        names = {m["name"] for m in json.load(r)["models"]}
    return names | {n.removesuffix(":latest") for n in names}


def pull(model: str):
    if model in installed():
        say(f"{model} already pulled", True)
        return
    say(f"Downloading {model} (one time, a few GB)")
    r = subprocess.run(["ollama", "pull", model])
    if r.returncode:
        raise RuntimeError(f"ollama pull {model} failed")
    say(f"{model} downloaded", True)


# --- Hardware -------------------------------------------------------------------------------------
def gpu_vram_gb() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(float(x) for x in out.split()) / 1024
    except Exception:
        return 0.0


def pick_profile(forced: str | None) -> str:
    from .config import _total_ram_gb
    ram, vram = _total_ram_gb(), gpu_vram_gb()
    apple = sys.platform == "darwin" and ram >= 16          # Apple silicon: unified memory runs the 4B model well
    # A 6 GB+ NVIDIA GPU or a big-memory machine gets the 4B vision model; anything else the 2B (CPU-friendly).
    prof = forced or ("standard" if vram >= 6 or apple or ram >= 24 else "lite")
    say(f"Hardware: {ram:.0f} GB RAM, {vram:.0f} GB NVIDIA GPU memory -> profile '{prof}'"
        + ("" if forced else " (change with --profile)"), True)
    return prof


# --- Fine-tune --------------------------------------------------------------------------------------
def find_finetune(folder: Path) -> Path | None:
    """The folder holding neuro-doc-qa's .gguf (directly, or in a 'neuro-doc-qa' subfolder)."""
    for d in [folder / FINETUNE, folder]:
        if d.is_dir() and list(d.glob("*.gguf")):
            return d
    hits = [p.parent for p in folder.rglob("*.gguf") if FINETUNE in str(p).lower()]
    return hits[0] if hits else None


def download_finetune(repo: str) -> Path:
    from huggingface_hub import snapshot_download
    say(f"Downloading {FINETUNE} from Hugging Face ({repo})")
    dest = ROOT / "models" / "finetunes"
    snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=[f"{FINETUNE}/*", "*.gguf", "Modelfile"])
    return dest


def import_finetune(folder: Path) -> str | None:
    d = find_finetune(folder)
    if not d:
        say(f"No {FINETUNE} .gguf found under {folder}; using the stock model for document QA", False)
        return None
    if FINETUNE in installed():
        say(f"{FINETUNE} already imported", True)
        return f"{FINETUNE}:latest"
    gguf = sorted(d.glob("*.gguf"), key=lambda p: ("q4_k_m" not in p.name.lower(), p.name))[0]
    mf = d / "Modelfile"
    if not mf.exists() or gguf.name not in mf.read_text(encoding="utf-8", errors="ignore"):
        mf = d / "Modelfile.neuroassist"
        mf.write_text(MODELFILE.format(gguf=gguf.name), encoding="utf-8")
    say(f"Importing {FINETUNE} from {gguf.name}")
    r = subprocess.run(["ollama", "create", FINETUNE, "-f", mf.name], cwd=d)
    if r.returncode:
        say(f"Import failed; using the stock model for document QA", False)
        return None
    say(f"{FINETUNE} imported", True)
    return f"{FINETUNE}:latest"


# --- Settings + checks ------------------------------------------------------------------------------
def write_settings(values: dict[str, str]):
    old = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                old[k.strip()] = v.strip()
    old.update({k: v for k, v in values.items() if v is not None})
    lines = ["# Written by the installer. Environment variables with the same name override these."]
    lines += [f"{k}={v}" for k, v in sorted(old.items())]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"Settings saved to {ENV_FILE.name}", True)


def prefetch_audio():
    from . import config
    t = time.perf_counter()
    from faster_whisper import WhisperModel
    WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    say(f"Speech model whisper-{config.WHISPER_MODEL} ready ({time.perf_counter() - t:.0f}s)", True)
    try:
        t = time.perf_counter()
        from transformers import pipeline
        pipeline("audio-classification", model=config.SOUND_MODEL, device="cpu")
        say(f"Sound classifier ready ({time.perf_counter() - t:.0f}s)", True)
    except Exception as e:
        say(f"Sound classifier not available ({type(e).__name__}); speech still works", False)


def smoke_test() -> bool:
    """One real question through the whole pipeline, on the demo assignment."""
    from . import llm
    from .pipeline import Session, answer
    say("Loading the main model and asking one test question")
    t = time.perf_counter()
    try:
        a = answer("What file must I submit?", [str(ROOT / "assets" / "demo" / "assignment.txt")], Session(),
                   extras=False)
    except Exception as e:
        say(f"Test question failed: {type(e).__name__}: {e}", False)
        return False
    mem = llm.loaded()
    gpu = f", {sum(m['vram_gb'] for m in mem):.1f} GB on the GPU" if mem else ""
    ok = "lab4" in a.answer.lower()
    say(f"Test answer in {time.perf_counter() - t:.0f}s{gpu}: {a.answer!r}", ok)
    return ok


def check():
    from . import config
    up = ollama_up()
    say(f"Ollama at {host()}", up)
    if up:
        have = installed()
        for m in dict.fromkeys([config.VISION_MODEL, config.TEXT_MODEL, config.QA_MODEL, config.CODE_MODEL]):
            say(f"model {m}", m in have or m.removesuffix(":latest") in have)
    say(f"profile {config.PROFILE}; settings file {'found' if ENV_FILE.exists() else 'not written yet'}", True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finetunes", type=Path, help="folder with our fine-tuned models (e.g. from Drive or USB)")
    ap.add_argument("--hf-repo", help="Hugging Face repo with the fine-tuned models")
    ap.add_argument("--profile", choices=["lite", "standard"], help="force a profile instead of detecting")
    ap.add_argument("--data", help="practice-pack folder for the eval (saved in neuroassist.env)")
    ap.add_argument("--skip-test", action="store_true", help="skip the live test question")
    ap.add_argument("--check", action="store_true", help="only report status")
    a = ap.parse_args(argv)
    if a.check:
        check()
        return 0

    print("\nNeuroAssist setup: everything runs on this computer; downloads happen only now.\n")
    if not ensure_ollama():
        return 1
    profile = pick_profile(a.profile)
    os.environ["NEUROASSIST_PROFILE"] = profile
    from . import config
    if config.PROFILE != profile:              # config was imported before the profile was chosen
        import importlib
        importlib.reload(config)
    settings = {"NEUROASSIST_PROFILE": profile}
    if a.data:
        settings["NEUROASSIST_DATA"] = a.data
    try:
        pull(config.PROFILES[profile]["vision"])
        pull(config.CODE_MODEL)
    except RuntimeError as e:
        say(str(e), False)
        return 1

    src = a.finetunes or (download_finetune(a.hf_repo) if a.hf_repo else None)
    qa = import_finetune(src) if src else None
    if not qa and FINETUNE in installed():          # imported earlier (e.g. by scripts/import_finetunes.ps1)
        qa = f"{FINETUNE}:latest"
        say(f"Using {qa}, already in Ollama", True)
    if not qa and not src:
        say("No fine-tunes given: document QA uses the stock model (add --finetunes or --hf-repo)", None)
    settings["NEUROASSIST_QA_MODEL"] = qa or ""
    write_settings(settings)
    if qa:
        os.environ["NEUROASSIST_QA_MODEL"] = qa
    import importlib
    from . import config as cfg
    importlib.reload(cfg)

    prefetch_audio()
    ok = True if a.skip_test else smoke_test()
    print("\nDone." + ("" if ok else " (the test question did not pass; the app still opens, see above)"))
    print("Start the app:  run.bat   (Windows)   or   ./run.sh   (Mac/Linux)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

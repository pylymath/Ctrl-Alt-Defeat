#!/usr/bin/env bash
# NeuroAssist one-step setup for macOS / Linux. From the repo folder:
#
#     ./install.sh
#     ./install.sh --finetunes /path/to/neuroassist-finetunes
#     ./install.sh --hf-repo USER/neuroassist-finetunes
#
# Same steps as install.ps1: Python 3.10-3.12 + Ollama, .venv, packages, models, settings, one test question.
# Then run:  ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(not (3,10) <= sys.version_info[:2] <= (3,12))'; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10-3.12 was not found. Install it (macOS: brew install python@3.12) and run this again."; exit 1
fi
echo "[OK  ] Python: $PY"

if ! command -v ollama >/dev/null; then
  echo "Ollama (runs the AI models locally) was not found."
  if [ "$(uname)" = "Darwin" ]; then
    echo "Install it from https://ollama.com/download (or: brew install ollama) and run this again."; exit 1
  fi
  read -r -p "Install Ollama with its official script (https://ollama.com/install.sh)? [y/N] " a
  if [[ "$a" =~ ^[Yy] ]]; then curl -fsSL https://ollama.com/install.sh | sh; else exit 1; fi
fi
echo "[OK  ] Ollama: $(ollama --version)"

[ -x .venv/bin/python ] || { echo "[..  ] Creating .venv"; "$PY" -m venv .venv; }
echo "[..  ] Installing Python packages (first time: a few minutes)"
.venv/bin/python -m pip install --upgrade pip --quiet
if [ "$(uname)" = "Linux" ]; then
  .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi
.venv/bin/python -m pip install -r requirements.txt --quiet
echo "[OK  ] Python packages installed"

exec .venv/bin/python -m neuroassist.installer "$@"

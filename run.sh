#!/usr/bin/env bash
# Starts NeuroAssist (after install.sh). Opens http://127.0.0.1:7860.  ./run.sh --lan  also serves phones on the Wi-Fi.
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Not installed yet. Run: ./install.sh"; exit 1; }
.venv/bin/python -m neuroassist.installer --check
exec .venv/bin/python -m neuroassist.ui "$@"

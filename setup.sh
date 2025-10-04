#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
echo "Venv ready. Activate with: source .venv/bin/activate"



#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

echo "Environment installed. Run:"
echo "  export PYTHONPATH=\"$PWD/src\""
echo "  python scripts/verify_setup.py"

#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "==> Creating virtual environment..."
python3 -m venv .venv

echo "==> Installing dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q

echo "==> Installing utils package (editable)..."
.venv/bin/pip install -e . -q

echo "==> Setting up .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "    .env created — fill in your API key before running notebooks"
else
    echo "    .env already exists, skipping"
fi

echo ""
echo "Done! Next steps:"
echo "  1. source .venv/bin/activate"
echo "  2. Edit .env and set LLM_MODEL + API key"
echo "  3. jupyter notebook"

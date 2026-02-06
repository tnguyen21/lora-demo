#!/usr/bin/env bash
# One-liner setup for workshop attendees.
# Usage: bash scripts/setup_env.sh

set -euo pipefail

echo "=== Workshop Environment Setup ==="

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install deps
echo "Creating Python 3.12 virtual environment..."
uv venv --python 3.12

echo "Installing dependencies..."
source .venv/bin/activate
uv pip install -e ".[dev]"

# Check for API key
if [ -z "${TINKER_API_KEY:-}" ]; then
    echo ""
    echo "WARNING: TINKER_API_KEY not set."
    echo "  export TINKER_API_KEY='your-key-here'"
    echo ""
    echo "You can still run cost_comparison.py scripts without it."
fi

echo ""
echo "=== Setup complete! ==="
echo "Activate with: source .venv/bin/activate"
echo ""
echo "Try running:"
echo "  python use_cases/text/01_ticket_routing/cost_comparison.py"

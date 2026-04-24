#!/bin/bash

set -e

echo "============================================"
echo "  MCP Explorer - Setup"
echo "============================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN} $1${NC}"; }
warn() { echo -e "${YELLOW}  $1${NC}"; }
err()  { echo -e "${RED} $1${NC}"; }

# Python
echo "1. Checking Python..."
if ! command -v python3 &>/dev/null; then err "Python 3 not found."; exit 1; fi
ok "Python $(python3 --version | awk '{print $2}')"

# Virtual env
echo ""
echo "2. Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ok "venv created"
else
    ok "venv already exists"
fi
source venv/bin/activate

# Install packages
echo ""
echo "3. Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "flask, requests, mcp installed"

# Seed database
echo ""
echo "4. Seeding database..."
python3 seed_db.py
ok "bookstore.db ready"

# Ollama check
echo ""
echo "5. Checking Ollama..."
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama is running"
else
    warn "Ollama not detected on localhost:11434"
    warn "Start it with Docker:"
    warn "  docker run -d -p 11434:11434 -v ollama:/root/.ollama ollama/ollama"
    warn "Then pull the model:"
    warn "  docker exec \$(docker ps -qf ancestor=ollama/ollama) ollama pull qwen2:0.5b"
    echo ""
    warn "Re-run this script or just start the app once Ollama is up."
fi

# Pull model if Ollama is live
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo ""
    echo "6. Pulling qwen2:0.5b (skip if already downloaded)..."
    curl -sf http://localhost:11434/api/pull -d '{"name":"qwen2:0.5b"}' | tail -1 || warn "Pull failed — model may already exist"
    ok "Model ready"
fi

echo ""
echo "============================================"
echo -e "${GREEN} Setup Complete!${NC}"
echo "============================================"
echo ""
echo "  Run the app:"
echo "    source venv/bin/activate"
echo "    python3 app.py"
echo ""
echo "  Then open: http://localhost:5050"
echo ""
echo "  Note: for better tool-calling, try MODEL_NAME=qwen2.5:3b python3 app.py"
echo ""

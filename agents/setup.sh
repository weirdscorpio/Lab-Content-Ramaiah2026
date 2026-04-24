#!/bin/bash

set -e

echo "============================================"
echo "  AI Agents Explorer - Setup"
echo "============================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN} $1${NC}"; }
warn() { echo -e "${YELLOW}  $1${NC}"; }
err()  { echo -e "${RED} $1${NC}"; }

echo "1. Checking Python..."
if ! command -v python3 &>/dev/null; then err "Python 3 not found."; exit 1; fi
ok "Python $(python3 --version | awk '{print $2}')"

echo ""
echo "2. Setting up virtual environment..."
if [ ! -d "venv" ]; then python3 -m venv venv && ok "venv created"; else ok "venv exists"; fi
source venv/bin/activate

echo ""
echo "3. Installing packages (flask, langgraph, langchain-ollama)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Packages installed"

echo ""
echo "4. Seeding movies database..."
python3 seed_db.py
ok "movies.db ready"

echo ""
echo "5. Checking Ollama..."
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama running"
    echo ""
    echo "5. Pulling llama3.2:3b (skip if already present)..."
    curl -sf http://localhost:11434/api/pull -d '{"name":"llama3.2:3b"}' | tail -1 || warn "Pull may have failed — model may already exist"
    ok "Model ready"
else
    warn "Ollama not found on localhost:11434"
    warn "Start it: docker run -d -p 11434:11434 -v ollama:/root/.ollama ollama/ollama"
    warn "Then pull: docker exec \$(docker ps -qf ancestor=ollama/ollama) ollama pull llama3.2:3b"
fi

echo ""
echo "============================================"
echo -e "${GREEN} Setup Complete!${NC}"
echo "============================================"
echo ""
echo "  Run:  source venv/bin/activate && python3 app.py"
echo "  Open: http://localhost:5051"
echo ""

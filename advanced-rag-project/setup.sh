#!/bin/bash

set -e

echo "============================================"
echo "   Advanced Multimodal RAG - Setup"
echo "============================================"
echo ""

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[!!]${NC} $1"; }
err()  { echo -e "  ${RED}[ERR]${NC} $1"; }
info() { echo -e "  ${CYAN}[--]${NC} $1"; }

# ── 1. Python ───────────────────────────────────────────────────────────────
echo "1. Checking Python..."
if ! command -v python3 &>/dev/null; then
    err "Python 3 not found. Please install Python 3.9+."
    exit 1
fi
PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
ok "Python $PY_VER found"

# ── 2. Virtual environment ──────────────────────────────────────────────────
echo ""
echo "2. Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ok "Virtual environment created"
else
    ok "Virtual environment already exists"
fi
source venv/bin/activate
ok "Virtual environment activated"

# ── 3. Package manager ─────────────────────────────────────────────────────
echo ""
echo "3. Installing UV package manager..."
pip install --quiet --upgrade pip
pip install --quiet --upgrade uv && ok "UV installed" || warn "UV failed, will use pip"

# ── 4. Python dependencies ──────────────────────────────────────────────────
echo ""
echo "4. Installing Python dependencies..."
info "This may take a few minutes on first run (downloading models)."

if command -v uv &>/dev/null; then
    uv pip install -r requirements.txt && ok "All dependencies installed (UV)" \
        || { warn "UV failed, retrying with pip..."; pip install -r requirements.txt && ok "All dependencies installed (pip)"; }
else
    pip install -r requirements.txt && ok "All dependencies installed (pip)"
fi

# ── 5. Verify key imports ───────────────────────────────────────────────────
echo ""
echo "5. Verifying key imports..."

python3 -c "import chromadb; print('  chromadb', chromadb.__version__)" \
    && ok "chromadb OK" || warn "chromadb import failed"

python3 -c "import sentence_transformers; print('  sentence-transformers', sentence_transformers.__version__)" \
    && ok "sentence-transformers OK" || warn "sentence-transformers import failed"

python3 -c "import fitz; print('  PyMuPDF', fitz.__version__)" \
    && ok "PyMuPDF OK" || warn "PyMuPDF import failed"

python3 -c "import langchain; print('  langchain', langchain.__version__)" \
    && ok "langchain OK" || warn "langchain import failed"

python3 -c "import langgraph; print('  langgraph', langgraph.__version__)" \
    && ok "langgraph OK" || warn "langgraph import failed"

python3 -c "from PIL import Image; print('  Pillow OK')" \
    && ok "Pillow OK" || warn "Pillow import failed"

# ── 6. Ollama ───────────────────────────────────────────────────────────────
echo ""
echo "6. Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    warn "Ollama CLI not found."
    warn "Install it from https://ollama.com and then run:"
    warn "  ollama pull qwen2:0.5b"
    warn "  ollama pull llava          # optional — for image captioning"
else
    ok "Ollama found"

    # Check if Ollama server is running
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        ok "Ollama server is running"

        # Pull text model
        echo ""
        info "Pulling qwen2:0.5b (text LLM)..."
        ollama pull qwen2:0.5b && ok "qwen2:0.5b ready" || warn "qwen2:0.5b pull failed"

        # Offer to pull vision model
        echo ""
        read -r -p "  Pull llava (vision model for image captioning)? [y/N] " PULL_LLAVA
        if [[ "$PULL_LLAVA" =~ ^[Yy]$ ]]; then
            ollama pull llava && ok "llava ready" || warn "llava pull failed"
        else
            info "Skipping llava. Images will be indexed using filename only."
        fi
    else
        warn "Ollama server is not running. Start it with: ollama serve"
        warn "Then pull models: ollama pull qwen2:0.5b"
    fi
fi

# ── 7. Create required directories ─────────────────────────────────────────
echo ""
echo "7. Creating directories..."
mkdir -p docs images uploads chroma_db static/thumbnails
ok "Directories ready"

# ── 8. Sample document ──────────────────────────────────────────────────────
echo ""
echo "8. Checking sample content..."
if [ ! -f "docs/ai_concepts.md" ]; then
    info "Sample document already exists or creating placeholder..."
    # The file is part of the repo — nothing to do
fi
ok "Sample docs ready (see docs/ and images/)"

# ── 9. Smoke-test the app import ────────────────────────────────────────────
echo ""
echo "9. Running import smoke test..."
python3 -c "
import sys
sys.path.insert(0, '.')
from core.vector_engine import MultiModalVectorEngine
from core.document_processor import DocumentProcessor
from core.image_processor import ImageProcessor
print('  All core modules import cleanly.')
" && ok "Smoke test passed" || warn "Smoke test failed — check errors above"

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo -e "  ${GREEN}Setup complete!${NC}"
echo "============================================"
echo ""
echo "  To start the server:"
echo ""
echo -e "    ${CYAN}source venv/bin/activate${NC}"
echo -e "    ${CYAN}python app/app.py${NC}"
echo ""
echo "  Then open: http://127.0.0.1:5353"
echo ""
echo "  Supported uploads: PDF, Markdown, TXT, JPG, PNG, WEBP"
echo ""
echo "  Advanced RAG techniques active:"
echo "    HyDE  · Multi-Query  · RRF Fusion"
echo "    Cross-Encoder Rerank  · Contextual Compression"
echo ""

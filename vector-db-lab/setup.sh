#!/bin/bash

set -e

echo "============================================"
echo "🔍 Vector Databases - Lab Setup"
echo "============================================"
echo ""
echo "Setting up your super simple environment..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${GREEN} $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}  $1${NC}"
}

print_error() {
    echo -e "${RED} $1${NC}"
}

# Check Python availability
echo ""
echo "1. Checking Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Python $PYTHON_VERSION available"
else
    print_error "Python 3 not found. Please install Python 3."
    exit 1
fi

# Create virtual environment
echo ""
echo "2. Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "3. Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

# Install uv first (fast Python package manager)
echo ""
echo "4. Installing UV package manager..."
pip install --quiet --upgrade uv
if [ $? -eq 0 ]; then
    print_status "UV package manager installed"
else
    print_warning "UV installation failed, falling back to pip"
    USE_PIP=true
fi

# Install packages using UV (or pip as fallback)
echo ""
echo "5. Installing Python packages..."
echo "   - numpy (for vector math)"
echo "   - chromadb (latest version for production vector database)"
echo "   - sentence-transformers (for real AI embeddings)"
echo "   - Building educational concepts with production tools"

if [ -z "$USE_PIP" ]; then
    # Use UV for faster installation
    echo "   Using UV for fast package installation..."
    uv pip install --upgrade numpy chromadb sentence-transformers
    
    if [ $? -eq 0 ]; then
        print_status "All packages installed successfully with UV"
    else
        print_warning "UV installation failed, retrying with pip..."
        pip install --quiet --upgrade numpy chromadb sentence-transformers
        print_status "All packages installed successfully with pip"
    fi
else
    # Fallback to pip
    pip install --quiet --upgrade numpy chromadb sentence-transformers
    print_status "All packages installed successfully with pip"
fi

# Test ChromaDB installation
echo ""
echo "6. Testing ChromaDB installation..."
python3 -c "import chromadb; print('ChromaDB version:', chromadb.__version__)" 2>/dev/null && \
    print_status "ChromaDB working correctly" || \
    print_warning "ChromaDB test failed - may still work"

# Create completion marker for KodeKloud tests
echo ""
echo "7. Creating setup completion marker..."
echo "Vector Database ChromaDB Setup Complete" > setup-complete.txt
echo "Created: $(date)" >> setup-complete.txt
echo "ChromaDB Version: $(python3 -c 'import chromadb; print(chromadb.__version__)' 2>/dev/null || echo 'unknown')" >> setup-complete.txt
print_status "Setup completion marker created"


# Final setup summary
echo ""
echo "============================================"
echo -e "${GREEN} Setup Complete※${NC}"
echo "============================================"
echo ""

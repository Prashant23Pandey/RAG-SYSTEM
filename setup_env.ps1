# ============================================
# Agentic RAG System - Environment Setup Guide
# ============================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Agentic RAG System - Environment Setup    " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create virtual environment
Write-Host "[1/5] Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($?) { Write-Host "  -> Virtual environment created successfully" -ForegroundColor Green }
else { Write-Host "  -> Failed to create virtual environment" -ForegroundColor Red; exit 1 }

# Step 2: Activate virtual environment
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1
if ($?) { Write-Host "  -> Virtual environment activated" -ForegroundColor Green }
else { Write-Host "  -> Failed to activate virtual environment" -ForegroundColor Red; exit 1 }

# Step 3: Upgrade pip
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($?) { Write-Host "  -> pip upgraded" -ForegroundColor Green }

# Step 4: Install requirements
Write-Host "[4/5] Installing project dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($?) { Write-Host "  -> Dependencies installed successfully" -ForegroundColor Green }
else { Write-Host "  -> Failed to install dependencies" -ForegroundColor Red; exit 1 }

# Step 5: Check Ollama and pull models
Write-Host "[5/5] Checking Ollama and pulling models..." -ForegroundColor Yellow
$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaPath) {
    Write-Host "  -> Ollama found at: $($ollamaPath.Source)" -ForegroundColor Green
    Write-Host "  -> Pulling llama3.2 model (this may take a while)..." -ForegroundColor Yellow
    ollama pull llama3.2
    Write-Host "  -> Pulling nomic-embed-text model..." -ForegroundColor Yellow
    ollama pull nomic-embed-text
    Write-Host "  -> Models pulled successfully" -ForegroundColor Green
} else {
    Write-Host "  -> Ollama not found. Please install from https://ollama.com" -ForegroundColor Red
    Write-Host "  -> After installing, run: ollama pull llama3.2" -ForegroundColor Yellow
    Write-Host "  -> Then run: ollama pull nomic-embed-text" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!                          " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment in future sessions:" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "To start Ollama server:" -ForegroundColor White
Write-Host "  ollama serve" -ForegroundColor Yellow
Write-Host ""
Write-Host "To run the system:" -ForegroundColor White
Write-Host "  python main.py" -ForegroundColor Yellow

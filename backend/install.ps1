# PowerShell script to install backend dependencies
# Run this from the backend directory

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Emotion Voice API - Dependency Installation  " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "Error: requirements.txt not found!" -ForegroundColor Red
    Write-Host "Make sure you're running this from the backend directory." -ForegroundColor Yellow
    exit 1
}

# Check if venv exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Installing dependencies (this may take several minutes)..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "  Installation completed successfully!         " -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Start the server: uvicorn main:app --reload" -ForegroundColor White
    Write-Host "  2. Open browser: http://127.0.0.1:8000/docs" -ForegroundColor White
    Write-Host "  3. The server will run in DUMMY MODE until you train your model" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Installation failed!" -ForegroundColor Red
    Write-Host "Try running manually: pip install -r requirements.txt" -ForegroundColor Yellow
}

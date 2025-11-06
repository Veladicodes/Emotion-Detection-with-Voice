# Phase 4: Start Backend Server
# Quick start script for Windows PowerShell

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "  EMOTION VOICE API - PHASE 4" -ForegroundColor Green
Write-Host "  Starting Production Server..." -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor Cyan

# Check if in correct directory
if (-Not (Test-Path "main.py")) {
    Write-Host "❌ Error: main.py not found" -ForegroundColor Red
    Write-Host "Run this script from the backend directory:" -ForegroundColor Yellow
    Write-Host "  cd 'D:\Emotion Voice\backend'" -ForegroundColor White
    Write-Host "  .\start_server.ps1" -ForegroundColor White
    exit 1
}

# Check if model exists
$modelPath = "D:\Emotion Voice\phase3_outputs\final_export\best_efficientnet_v2_m+vgg16+densenet121.pt"
if (-Not (Test-Path $modelPath)) {
    Write-Host "" 
    Write-Host "⚠️  WARNING: Model not found!" -ForegroundColor Yellow
    Write-Host "Expected: $modelPath" -ForegroundColor White
    Write-Host ""
    Write-Host "Server will start in DUMMY MODE with random predictions." -ForegroundColor Yellow
    Write-Host "Complete Phase 3 model export first for real predictions." -ForegroundColor Yellow
    Write-Host ""
    Start-Sleep -Seconds 3
}

# Check if python-multipart is installed
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Cyan
$multipartInstalled = python -c "import multipart" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing python-multipart..." -ForegroundColor Yellow
    pip install python-multipart --quiet
}

Write-Host "✅ Dependencies OK" -ForegroundColor Green
Write-Host ""

# Start server
Write-Host "Starting Uvicorn server..." -ForegroundColor Cyan
Write-Host "API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "Press CTRL+C to stop" -ForegroundColor Yellow
Write-Host ""

# Run uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload


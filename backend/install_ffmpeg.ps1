# Quick FFmpeg Installation Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Installing FFmpeg for WebM Support" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Chocolatey is installed
$chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue

if ($chocoInstalled) {
    Write-Host "Installing ffmpeg via Chocolatey..." -ForegroundColor Yellow
    choco install ffmpeg -y
    Write-Host ""
    Write-Host "Done! Close and reopen your terminal." -ForegroundColor Green
} else {
    Write-Host "Chocolatey not found." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Install ffmpeg manually:" -ForegroundColor Cyan
    Write-Host "1. Download from: https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor White
    Write-Host "2. Extract to C:\ffmpeg" -ForegroundColor White
    Write-Host "3. Add C:\ffmpeg\bin to PATH" -ForegroundColor White
    Write-Host ""
    Write-Host "Or install Chocolatey first:" -ForegroundColor Cyan
    Write-Host "https://chocolatey.org/install" -ForegroundColor White
}


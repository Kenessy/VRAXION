# IGNITE_EXPERT_INFINITE.ps1
# Launches VRAXION_INFINITE.py and a live log tail window.

$workDir = "G:\AI\mirror\VRAXION"
if (-not (Test-Path $workDir)) {
    Write-Error "Project directory $workDir not found."
    return
}

Set-Location $workDir

Write-Host "Launching VRAXION Infinite Engine..." -ForegroundColor Cyan
$engineCmd = "cd '$workDir'; python VRAXION_INFINITE.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $engineCmd

Write-Host "Waiting for Log Initialization..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "Launching Telemetry Window..." -ForegroundColor Green
$logPath = Join-Path $workDir "logs\expert_infinite.log"
$telemetryCmd = "cd '$workDir'; Get-Content '$logPath' -Wait -Tail 30"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $telemetryCmd

Write-Host "Ignition complete. Monitor the telemetry window." -ForegroundColor Green

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$proofDir = Join-Path $root "proofs"
New-Item -ItemType Directory -Path $proofDir -Force | Out-Null

$logPath = Join-Path $proofDir "assoc_clean_proof.log"
$summaryOut = Join-Path $proofDir "assoc_clean_proof_summary.json"

# Core environment (CPU + synthetic, no downloads)
$env:VAR_COMPUTE_DEVICE = "cpu"
$env:VAR_LOGGING_PATH = $logPath
$env:TP6_OFFLINE_ONLY = "1"
$env:TP6_SYNTH = "1"
$env:TP6_SYNTH_MODE = "assoc_clean"
$env:TP6_SYNTH_LEN = "8"
$env:TP6_ASSOC_KEYS = "2"
$env:TP6_ASSOC_PAIRS = "1"
$env:TP6_MAX_SAMPLES = "512"
$env:TP6_BATCH_SIZE = "32"
$env:TP6_MAX_STEPS = "800"
$env:TP6_PTR_UPDATE_EVERY = "8"

# Pointer + readout settings (from documented micro proof)
$env:TP6_PTR_SOFT_GATE = "1"
$env:PARAM_POINTER_FORWARD_STEP_PROB = "0.05"
$env:TP6_PTR_INERTIA = "0.1"
$env:TP6_PTR_DEADZONE = "0"
$env:TP6_PTR_NO_ROUND = "1"
$env:TP6_SOFT_READOUT = "1"
$env:TP6_LMOVE = "0"

# Governors/overrides tuned for clean micro proof
$env:TP6_PTR_UPDATE_GOV = "1"
$env:TP6_PANIC = "0"
$env:TP6_THERMO = "0"
$env:TP6_SHARD_BATCH = "0"
$env:TP6_SHARD_ADAPT = "0"
$env:TP6_TRACTION_LOG = "0"
$env:TP6_DEBUG_EVERY = "10"

Write-Host "Running assoc_clean micro proof..." -ForegroundColor Cyan
Write-Host "Log: $logPath" -ForegroundColor Cyan

python (Join-Path $root "tournament_phase6.py")

$summarySrc = Join-Path $root "summaries\\current\\tournament_phase6_summary.json"
if (Test-Path $summarySrc) {
    Copy-Item $summarySrc $summaryOut -Force
    Write-Host "Summary saved: $summaryOut" -ForegroundColor Green
} else {
    Write-Warning "Summary not found: $summarySrc"
}

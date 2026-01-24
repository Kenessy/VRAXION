# run_expert_endurance.ps1
# VRAXION: 16-Expert Endurance Run (Synthetic Data Mode)

# 1. Unlock 16 Experts + bf16
$env:TP6_EXPERT_HEADS = "16"
$env:TP6_PRECISION = "bf16"
$env:TP6_RESUME = "0"

# 2. Synthetic dataset (assoc_byte)
$env:TP6_SYNTH = "1"
$env:TP6_SYNTH_MODE = "assoc_byte"
$env:TP6_SYNTH_LEN = "512"
$env:TP6_ASSOC_KEYS = "64"
$env:TP6_ASSOC_PAIRS = "4"
$env:PILOT_OFFLINE = "1"

# 3. Endurance bounds + logging
$env:TP6_MAX_STEPS = "1000"
$env:TP6_WALL = "600"
$env:TP6_SAVE_EVERY_STEPS = "50"
$env:VAR_LOGGING_PATH = "logs/expert_endurance.log"
$env:TP6_CKPT = "checkpoints/expert_endurance/expert_v1.pt"

# 4. Ensure directories exist
New-Item -ItemType Directory -Force -Path "checkpoints/expert_endurance" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host ">>> STARTING VRAXION: EXPERT ENDURANCE (16 HEADS) <<<" -ForegroundColor Green
Write-Host "    Task: Synthetic Assoc Byte"
Write-Host "    Mode: bfloat16 | Device: CUDA"
Write-Host "    Steps: 1000 | Wall: 600s | Log: logs/expert_endurance.log"
python tournament_phase6.py

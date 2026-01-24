# run_expert_demo.ps1
# VRAXION: 16-Expert Audit Run (Synthetic Data Mode)

# 1. Unlock the 16 Experts
$env:TP6_EXPERT_HEADS = "16"

# 2. Force Pro Precision
$env:TP6_PRECISION = "bf16"

# 3. SAFETY: Force Fresh Start
$env:TP6_RESUME = "0"

# 4. Correct Paths
$env:TP6_CKPT = "checkpoints/expert_demo/expert_v1.pt"
$env:VAR_LOGGING_PATH = "logs/expert_demo.log"

# 5. SYNTHETIC DATA (Fast, pure logic test)
$env:TP6_SYNTH = "1"
$env:TP6_SYNTH_MODE = "assoc_byte"
$env:TP6_SYNTH_LEN = "512"
$env:TP6_ASSOC_KEYS = "64"
$env:TP6_ASSOC_PAIRS = "4"
$env:TP6_OFFLINE_ONLY = "1"

# 6. Ensure directories exist
New-Item -ItemType Directory -Force -Path "checkpoints/expert_demo" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

Write-Host ">>> STARTING VRAXION: EXPERT MODE (16 HEADS) <<<" -ForegroundColor Green
Write-Host "    Task: Synthetic Assoc Byte (Logic Test)"
Write-Host "    Mode: bfloat16 | Device: CUDA"
python tournament_phase6.py

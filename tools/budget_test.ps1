$ErrorActionPreference = 'Stop'
$repo = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location $repo

$env:VAR_COMPUTE_DEVICE = 'cpu'
$env:CUDA_VISIBLE_DEVICES = ''
$env:TP6_PRECISION = 'fp64'
$env:TP6_PTR_DTYPE = 'fp64'
$env:OMP_NUM_THREADS = '24'
$env:MKL_NUM_THREADS = '24'

$env:TP6_RESUME = '1'
$env:TP6_CKPT = 'checkpoints/mitosis/tier4_mix_mitosis_sentinel_round12_split.pt'
$env:TP6_SAVE_EVERY_STEPS = '100'
$env:TP6_EVAL_EVERY_STEPS = '10'
$env:TP6_EVAL_AT_CHECKPOINT = '0'
$env:TP6_PRINT_EVERY = '10'
$env:TP6_SHARD_ADAPT_EVERY = '1024'
$env:TP6_FORCE_CADENCE_1 = '0'

$env:TP6_SYNTH = '1'
$env:TP6_SYNTH_MODE = 'assoc_mix'
$env:TP6_SYNTH_LEN = '32'
$env:TP6_ASSOC_KEYS = '4'
$env:TP6_ASSOC_VAL_RANGE = '4'
$env:TP6_ASSOC_PAIRS = '1'
$env:TP6_ASSOC_MIX_OFFSET = '4096'
$env:TP6_ASSOC_MIX_DOMAIN_TOKEN = '0'

$env:TP6_MITOSIS = '0'
$env:TP6_METABOLIC_HUNGER = '0'

$env:TP6_RING_LEN = '128'
$env:TP6_EXPERT_HEADS = '21'
$env:TP6_PTR_INERTIA_OVERRIDE = '0.3'
$env:TP6_SCALE_INIT = '0.05'
$env:TP6_UPDATE_SCALE = '0.05'
$env:TP6_LR = '0.01'
$env:TP6_SCALE_MIN = '0.0'

$env:TP6_ANCHOR_CONF_MIN = '0.5'
$env:TP6_ANCHOR_MIN_STEP = '0.1'

$env:TP6_MAX_STEPS = '0'
$env:TP6_IGNORE_MAX_STEPS = '0'
$env:TP6_WALL = '120'
$env:TP6_OFFLINE_ONLY = '0'

$env:TP6_EXPERT_BUDGET = '2'
$env:TP6_USAGE_LAMBDA = '0.0'
$env:TP6_USAGE_LAMBDA_MAX = '0.2'
$env:TP6_USAGE_LAMBDA_ETA = '0.05'
$env:TP6_USAGE_GATE_CONF = '0.75'
$env:TP6_USAGE_GATE_EVAL = '0'
$env:TP6_USAGE_EMA_BETA = '0.9'
$env:TP6_USAGE_REMAP = '1'
$env:TP6_USAGE_REMAP_EVERY = '10'
$env:TP6_USAGE_REMAP_MODE = 'round_robin'

$env:VAR_LOGGING_PATH = 'logs/budget_test3.log'
$env:VAR_LIVE_TRACE_PATH = 'traces/budget_test3.jsonl'
$env:VAR_LIVE_TRACE_EVERY_N_STEPS = '10'

New-Item -ItemType Directory -Force -Path 'logs','traces' | Out-Null
python tournament_phase6.py

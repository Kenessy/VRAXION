[CmdletBinding()]
param(
    # Pass-through args for vrx_sync_linear_projects.py, e.g.:
    #   .\vrx_sync.ps1 sync --apply
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"

# Secrets policy:
# - This script NEVER prints the Linear API key.
# - Store the key outside the repo (recommended) or point to a file via VRX_LINEAR_API_KEY_FILE.
#
# Default key file location:
#   %USERPROFILE%\.vraxion\secrets\linear_api_key.txt

$keyFile = $env:VRX_LINEAR_API_KEY_FILE
if ([string]::IsNullOrWhiteSpace($keyFile)) {
    $keyFile = Join-Path $HOME ".vraxion\secrets\linear_api_key.txt"
}

if (-not (Test-Path -LiteralPath $keyFile)) {
    Write-Error ("Linear API key file not found: {0}`n" -f $keyFile) `
        + "Set VRX_LINEAR_API_KEY_FILE to an existing file, or create:`n" `
        + (Join-Path $HOME ".vraxion\secrets\linear_api_key.txt")
    exit 2
}

$env:LINEAR_API_KEY = (Get-Content -LiteralPath $keyFile -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($env:LINEAR_API_KEY)) {
    Write-Error ("Linear API key file is empty: {0}" -f $keyFile)
    exit 2
}

$scriptPath = Join-Path $PSScriptRoot "vrx_sync_linear_projects.py"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    Write-Error ("Missing sync script: {0}" -f $scriptPath)
    exit 2
}

python $scriptPath @Args

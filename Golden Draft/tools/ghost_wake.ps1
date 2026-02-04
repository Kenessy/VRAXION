<#
Ghost Hand: loopback trigger for Codex/VRAXION long runs.

This script watches for a JSON trigger file (wake_trigger.json) written by
Golden Draft/tools/vraxion_lab_supervisor.py and, after a delay, focuses the
target PowerShell window and "presses Enter" by sending a memo string.

Notes:
- For reliable focus, set your PowerShell window title to a stable string:
    $host.UI.RawUI.WindowTitle = "VRAXION_CLI"
  and set `--wake-window-title VRAXION_CLI` in the supervisor.
- This script is intentionally dumb and robust. All "intelligence" lives in the
  supervisor / agent; this only provides the physical wake-up signal.
#>

param(
  [string]$TriggerPath = "S:\AI\work\VRAXION_DEV\bench_vault\wake_trigger.json",
  [int]$PollSeconds = 2,
  [int]$FocusDelayMs = 250
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Log([string]$msg) {
  $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
  Write-Host ("[{0}] {1}" -f $ts, $msg)
}

function Try-AppActivate([string]$title) {
  try {
    Add-Type -AssemblyName Microsoft.VisualBasic | Out-Null
    [Microsoft.VisualBasic.Interaction]::AppActivate($title) | Out-Null
    Start-Sleep -Milliseconds $FocusDelayMs
    return $true
  } catch {
    return $false
  }
}

function Send-Memo([string]$memo) {
  Add-Type -AssemblyName System.Windows.Forms | Out-Null
  # SendWait is best-effort; keep memo single-line ASCII for reliability.
  [System.Windows.Forms.SendKeys]::SendWait($memo)
  Start-Sleep -Milliseconds 50
  [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
}

Write-Log ("ghost_wake online. watching: {0}" -f $TriggerPath)

while ($true) {
  if (-not (Test-Path -LiteralPath $TriggerPath)) {
    Start-Sleep -Seconds $PollSeconds
    continue
  }

  $raw = Get-Content -LiteralPath $TriggerPath -Raw
  if (-not $raw) {
    Start-Sleep -Seconds $PollSeconds
    continue
  }

  try {
    $j = $raw | ConvertFrom-Json
  } catch {
    Write-Log "trigger json parse failed; will retry"
    Start-Sleep -Seconds $PollSeconds
    continue
  }

  $wakeAfter = 0
  try { $wakeAfter = [int]$j.wake_after_s } catch { $wakeAfter = 0 }
  if ($wakeAfter -lt 0) { $wakeAfter = 0 }

  $memo = ""
  try { $memo = [string]$j.memo } catch { $memo = "" }
  if (-not $memo) { $memo = "[ghost_wake] ping" }

  $title = ""
  try { $title = [string]$j.window_title } catch { $title = "" }

  Write-Log ("trigger detected (reason={0}) wake_after_s={1}" -f $j.reason, $wakeAfter)
  Start-Sleep -Seconds $wakeAfter

  if ($title) {
    $ok = Try-AppActivate $title
    if (-not $ok) {
      Write-Log ("AppActivate failed for title '{0}'. sending to current active window." -f $title)
    }
  } else {
    Write-Log "no window_title set; sending to current active window."
  }

  try {
    Send-Memo $memo
    Write-Log ("sent memo: {0}" -f $memo)
  } catch {
    Write-Log ("SendKeys failed: {0}" -f $_.Exception.Message)
  }

  try {
    $dir = Split-Path -Parent $TriggerPath
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
    $dst = Join-Path $dir ("wake_trigger.processed_{0}.json" -f $stamp)
    Move-Item -LiteralPath $TriggerPath -Destination $dst -Force
    Write-Log ("archived trigger -> {0}" -f $dst)
  } catch {
    Write-Log ("failed to archive trigger: {0}" -f $_.Exception.Message)
  }
}


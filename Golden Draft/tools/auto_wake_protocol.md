# VRAXION Loopback Protocol (Supervisor + Ghost Hand)

Goal: keep long VRAXION runs alive (and restartable) even when the Codex chat turn
ends, and "wake" the chat only when something meaningful happens.

This uses two pieces:

1) `Golden Draft/tools/vraxion_lab_supervisor.py`
2) `Golden Draft/tools/ghost_wake.ps1`

## 1) One-time setup (recommended)

In the PowerShell window where you run Codex/VRAXION, set a stable window title:

```powershell
$host.UI.RawUI.WindowTitle = "VRAXION_CLI"
```

## 2) Start Ghost Hand (separate window)

Run this in a separate PowerShell window (keep it open):

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "S:\AI\work\VRAXION_DEV\Golden Draft\tools\ghost_wake.ps1"
```

It watches for:

`S:\AI\work\VRAXION_DEV\bench_vault\wake_trigger.json`

## 3) Start a supervised long run

Example: run the repulsion sweep under a supervisor (auto-restart on crash):

```powershell
Set-Location "S:\AI\work\VRAXION_DEV"
python -u "Golden Draft\tools\vraxion_lab_supervisor.py" `
  --job-name "assoc_repulsion_sweep" `
  --wake-window-title "VRAXION_CLI" `
  --wake-after-s 720 `
  --watchdog-no-output-s 120 `
  --max-restarts 20 `
  -- `
  python -u "Golden Draft\tools\sweep_assoc_repulsion.py"
```

When the run finishes (or crashes), the supervisor writes a wake trigger and the
Ghost Hand will type a memo + press Enter into the VRAXION_CLI window.

## Trigger format

The supervisor writes a JSON file like:

```json
{
  "version": 1,
  "wake_after_s": 720,
  "memo": "[vraxion_lab_supervisor] DONE ...",
  "reason": "done",
  "job_root": "S:\\AI\\work\\VRAXION_DEV\\bench_vault\\jobs\\...",
  "window_title": "VRAXION_CLI"
}
```

Ghost Hand moves it to `wake_trigger.processed_*.json` after sending it.


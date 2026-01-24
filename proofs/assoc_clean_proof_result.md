Assoc_clean micro proof (CPU)

Run command:
- powershell -File tools\run_proof_assoc_clean.ps1

Artifacts:
- Log: proofs\assoc_clean_proof.log

Results (parsed from log):
- Min loss: 0.0021 at step 2610
- Last loss before stop: 0.0304 at step 2882
- Run stopped manually (Ctrl+C) after demonstrating convergence.

Example log line (min loss):
[02:19:59] synth | absolute_hallway | step 2610 | loss 0.0021 | t=104.2s | ctrl(inertia=0.10, deadzone=0.00, walk=0.05, cadence=1, scale=1.000, cap=1.000, raw_delta=206.119, ptr[flip=0.833; dwell=1.00; dwell_max=1; delta=38.0538; delta_raw=206.1194])

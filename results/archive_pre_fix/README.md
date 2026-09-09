# ⚠️ INVALID RESULTS — produced by buggy code, kept only for comparison

**Do not use anything in this directory.** Every file here was generated before the
correctness fixes of 2026-09-09 (see the repo README, "Fixes applied"). They are
kept so the before/after can be compared, not because they are usable.

Specifically, these outputs are affected by:

- **The active-learning posterior double-counted evidence.** Re-multiplied the whole
  history at every timestep, so the log-odds saturated to numerical 0/1 and the
  belief could never revise. Visible as the flat plateaus in the accuracy curves —
  a patient misclassified early stayed misclassified forever.
- **The policies extrapolated from the wrong state** (`z_pred` rather than `z_filt`),
  discarding the most recent observation when scoring candidate probes.
- **EM never converged.** It silently exhausted `max_iter=100` on every run, so all
  parameter estimates here are badly under-converged (e.g. λ̂ = 0.40 against a true
  0.80). Its M-step was also inconsistent with its E-step, so the log-likelihood
  could *decrease* — it did, on 446 of 469 steps for one group.
- **No seeds.** None of these numbers can be reproduced.
- **The `random` baseline replayed one identical probe sequence** for every patient,
  and the policies were compared on different patient noise draws rather than paired.

| File | What it was |
|---|---|
| `experiment1_output.txt` | Experiment 1 log, 10 Jul |
| `experiment1_output2.txt` | Experiment 1 log, 21 Jul |
| `experiment2_mixed_noise_sigmaw_low.png` | The pre-fix version of the figure now in `results/` — the direct before/after pair |
| `experiment2_high_noise.png`, `experiment2_gamma_high_group0.png` | Experiment 2 under other regimes, 21 Jul; no post-fix equivalent has been generated |
| `true_trajectories.png`, `predicted_trajectories_after_estimating_parameters.png`, `test_predicted_trajectories.png` | Trajectory plots, 26 May |
| `analia_active_learning_strategies_50patients_30timesteps.png` | Early active-learning figure, 3 Jul |

The notebook's own embedded outputs are stale for the same reasons and have not
been regenerated.

# Results

Everything in this directory is current — produced after the 2026-09-09 fixes, with
explicit seeds, so every file can be regenerated bit-identically. Anything from
before those fixes is in `archive_pre_fix/` and is **not usable**; see the README
there for why.

## Experiment 1 — classification accuracy and parameter recovery across 8 regimes

Regenerate with: `MPLBACKEND=Agg python -u experiment1.py > results/experiment1_log.txt 2>&1`

| File | Contents |
|---|---|
| `experiment1_accuracy.csv` | One row per regime: our accuracy, both logistic-regression baselines, the train/test sizes and the group split |
| `experiment1_parameters.csv` | One row per (regime, group): true vs estimated α, λ, β, γ, σ_w, σ_e; **plus the identified combinations** `b0`, `b1`, `noise_var`, `noise_acov1`; plus EM convergence status, iteration count and log-likelihood |
| `experiment1_log.txt` | Full stdout, including the summary table at the end |

**Read `b0`/`b1`/`noise_var`/`noise_acov1`, not the raw α, β, σ_w.** Only five
combinations of the six parameters are identifiable — α, β and σ_w are defined
only up to a common scale (repo README > Known limitations). The `*_est` vs
`*_true` pairs for the identified columns are the meaningful recovery check.

## Experiment 1 — seed sensitivity

Regenerate with: `MPLBACKEND=Agg python -u experiment1_seed_sweep.py > results/experiment1_seed_sweep_log.txt 2>&1`

| File | Contents |
|---|---|
| `experiment1_seed_sweep.csv` | One row per (regime, seed) over 5 seeds: accuracy for all three methods, group split, how many of the 2 EM fits converged |
| `experiment1_seed_sweep_log.txt` | Per-regime mean and sd across seeds |

Use this to judge whether a change in a single-regime number between two runs is
real. A difference smaller than that regime's seed-to-seed `sd` is sampling noise.

## Experiment 2 — active-learning policy comparison

Regenerate with: `MPLBACKEND=Agg python -u experiment2.py > results/experiment2_log.txt 2>&1` (~90 s)

| File | Contents |
|---|---|
| `experiment2_mixed_noise_sigmaw_low.png` | The accuracy-vs-timestep figure, ±1 SEM bands |
| `experiment2_mixed_noise_sigmaw_low.csv` | Accuracy and SEM per timestep for all three policies — the numbers behind the figure |
| `experiment2_log.txt` | Full stdout: EM estimates, per-timestep table, steps-to-threshold summary |

Filenames carry the regime `TAG` set at the top of `experiment2.py`, so running a
different regime writes alongside these rather than overwriting them. Only the
`mixed_noise_sigmaw_low` regime has been rerun post-fix; the other two regimes in
`archive_pre_fix/` have no current equivalent.

# Results

Everything in this directory is current — regenerated 2026-09-11, after both the
2026-09-09 correctness fixes and the 2026-09-11 `P0 = 0` alignment (repo README >
Fixes applied 2026-09-11), with explicit seeds, so every file can be regenerated
bit-identically. Anything from before those fixes is in `archive_pre_fix/` and is
**not usable**; see the README there for why.

Because `P0` changed in `kalman_filter`, `infer_patient` and `_OnlineSSM`, every
file here moved except `experiment1_parameters.csv`, which is byte-identical: EM
already used `P0 = 0`, so the fits themselves never changed — only what the
classifier and the active-learning filters did with them.

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
| `experiment2_mixed_noise_sigmaw_low.png` | The accuracy-vs-timestep figure, ±1 SEM bands. Untitled and unsubtitled — `n`, `T`, the regime and the seed are in the log beside it, for the caption |
| `experiment2_mixed_noise_sigmaw_low.csv` | Accuracy and SEM per timestep for all three policies — the numbers behind the figure |
| `experiment2_log.txt` | Full stdout: EM estimates, per-timestep table, steps-to-threshold summary |

Filenames carry the regime `TAG` set at the top of `experiment2.py`, so running a
different regime writes alongside these rather than overwriting them. Only the
`mixed_noise_sigmaw_low` regime has been rerun post-fix; the other two regimes in
`archive_pre_fix/` have no current equivalent.

**The figures and the CSVs name the policies differently, on purpose.** The
figures show the display labels from `POLICY_LABELS` (`Random selection`,
`Minimum entropy`, `Mutual information`); the CSV columns and the sweep's
`policy` field keep the internal identifiers (`random`, `entropy minimization`,
`mutual information`). Keeping the two separate means a figure can be relabelled
without touching the CSV columns that the sweep's cross-check looks up by name.

Both identifiers and labels now describe the same thing: the policy probes where
the expected posterior Shannon entropy is *lowest*. Files written before
2026-09-11 use the old identifier `uncertainty sampling` for it — the rename
preserved Monte Carlo stream index 1, so those older numbers are still directly
comparable.

## Experiment 2 — seed sensitivity

Regenerate with: `MPLBACKEND=Agg python -u experiment2_seed_sweep.py > results/experiment2_seed_sweep_log.txt 2>&1` (~8 min)

| File | Contents |
|---|---|
| `experiment2_seed_sweep.csv` | One row per (seed, policy, timestep) over 5 seeds: accuracy, within-seed SEM across patients, and how many of that seed's 2 EM fits converged |
| `experiment2_seed_sweep_mixed_noise_sigmaw_low.png` | Mean curve per policy across the 5 seeds, banded by ±1 **sd across seeds** — not the SEM bands of the single-seed figure |
| `experiment2_seed_sweep_log.txt` | Per-seed steps-to-threshold, mean accuracy, and the paired gap over `random` |

Each seed redraws the training set (`seed=(seed, 0)`, so its own pair of EM fits),
plus every patient's noise and probes (`seed*100 + i`, disjoint across seeds).
The group assignment stays `i % 2` — 50 patients per group by construction, since
balance here is a design choice rather than something worth adding variance over.

Sweep seed 0 is arranged to reproduce `experiment2.py` exactly (identical training
seed, patient seeds 0–99), and the script asserts that against
`experiment2_mixed_noise_sigmaw_low.csv` at the end of every run. If that
cross-check ever reports `DIFFERS`, the two scripts have drifted apart and the
sweep no longer describes the single-seed result.

**This is the file to quote a policy comparison from.** `experiment2.py` alone
cannot separate a better policy from a luckier training set; the `sd` and paired
`wins` columns here can.

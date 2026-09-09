# mediators — Active Discovery of Latent Dynamic Mediators

Each patient `n` has an input signal `x_t`, an observation `y_t`, and a hidden
treatment-group label `c_n ∈ {0, 1}`. The group determines *which causal pathway*
generated `y_t`, via a group-specific latent mediation process `z_t`. Two
linear-Gaussian state-space models, indexed by `j`:

```
state:        z_t = α_j x_t + λ_j z_{t-1} + w_t,    w ~ N(0, σ_w²)
observation:  y_t = β_j z_t  + γ_j x_t     + e_t,   e ~ N(0, σ_e²)
```

The project asks two questions: can we recover the group from `(x, y)` alone, and
can *actively choosing* `x_t` identify it in fewer time steps than random dosing?

## Layout

| Path | What it is |
|---|---|
| `active_discovery_of_latent_dynamic_mediators.ipynb` | Main narrative notebook: the three phases end to end |
| `utils/data_generation.py` | `generate_synthetic_samples` (batch, phase 1) and `PatientSimulator` (one step at a time, phase 3) |
| `utils/kalman_filter.py`, `utils/rts_smoother.py` | Forward filter and RTS backward smoother |
| `utils/em_algorithm.py` | `em_ssm`: per-group parameter estimation |
| `utils/infer_patient.py` | Group posterior from the marginal likelihood under each model |
| `utils/active_learning.py` | The active-learning loop and the online per-group filter |
| `utils/monte_carlo_uncertainty_sampling.py` | Uncertainty-sampling policy (minimise posterior entropy) |
| `utils/monte_carlo_mutual_information.py` | Mutual-information policy (BALD) |
| `utils/baseline.py` | Logistic-regression baselines |
| `utils/plots.py` | All figures |
| `experiment1.py` | Classification accuracy across 8 noise / γ regimes |
| `experiment2.py` | Active-learning policy comparison; writes `results/*.png` + `*.csv` |
| `experiment1_seed_sweep.py` | Refits experiment 1 across 5 seeds to measure how much of each regime's accuracy is just the data draw |

**The three phases.** (1) *Training* — EM per group, labels known. (2)
*Classification* — run the filter under both models, compare marginal
likelihoods, take the MAP label. (3) *Active learning* — pick `x_{t+1}` to
identify the group as fast as possible.

## Running it

Conda env `hyperparameters` (Python 3.12.6; needs numpy, scipy, scikit-learn,
matplotlib) — this is also the notebook's kernel:

```bash
PY=~/RDS/miniconda3/envs/hyperparameters/bin/python
MPLBACKEND=Agg $PY -u experiment2.py           > results/experiment2_log.txt 2>&1   # ~90 s
MPLBACKEND=Agg $PY -u experiment1.py           > results/experiment1_log.txt 2>&1
MPLBACKEND=Agg $PY -u experiment1_seed_sweep.py > results/experiment1_seed_sweep_log.txt 2>&1
```

All outputs land in `results/` — see **`results/README.md`** for what each file
contains. Pre-fix outputs are quarantined in `results/archive_pre_fix/`.

`experiment2.py` also logs the accuracy curves to `results/experiment2_<TAG>.csv`
and prints a per-timestep table plus steps-to-threshold summary.

## Reproducibility

Every stochastic component takes an explicit seed derived from the script's
`SEED`, and each gets its **own** stream, so components are independent of one
another:

| Component | Seed | Notes |
|---|---|---|
| Training / test data | `generate_synthetic_samples(seed=...)` | `(SEED, regime, 0)` train, `(SEED, regime, 1)` test — distinct per regime, and train ≠ test |
| Patient `i`'s noise | `PatientSimulator(seed=i)` | Same realisation for all policies — a *paired* comparison |
| Patient `i`'s probes | `run_patient_with_active_learning(seed=i)` | The `random` policy's `x` sequence |
| Monte Carlo draws | derived from the same `seed` | One stream **per (patient, policy)** |

Verified properties (all hold):

1. A rerun is bit-identical.
2. Each policy gives the same result run **alone** as inside the full comparison.
3. Results are unaffected by other consumers of the global `np.random` state.
4. The order policies appear in the loop does not matter.

Properties 2–4 did **not** hold before: the Monte Carlo policies drew from
NumPy's *global legacy* `RandomState` (which is what `scipy.stats.*.rvs` falls
back to, and what `np.random.seed` seeds). A whole-script rerun did reproduce,
but only by accident of call ordering — running `mutual information` on its own
gave a different answer than running it third in the loop, and adding a policy or
changing `samples_size` silently changed every downstream result. The policies now
take an explicit `rng=` argument.

`em_ssm`, `kalman_filter`, `rts_smoother` and `infer_patient` are deterministic.
`np.random.seed(SEED)` is still called in both scripts purely as a belt-and-braces
default for anything added later that forgets to take a seed; nothing relies on it.

## Fixes applied 2026-09-09

**Active learning (`utils/active_learning.py`).**

1. *The class posterior double-counted evidence.* The loop combined the running
   posterior with `_innovations_log_likelihood` over the **full** history, but
   that function already accumulates the whole sequence — so every timestep
   re-multiplied all previous evidence. Log-odds grew quadratically in `t` and
   saturated to numerical 0/1, which is where the `divide by zero encountered in
   log` warnings came from. Once saturated, the belief could never revise, so a
   patient misclassified early stayed misclassified forever (visible as the flat
   plateaus in the old figures).
2. *The policies extrapolated from the wrong state.* They were handed
   `z_pred[-1]` = `E[z_t | y_{1:t-1}]`, then advanced one more step — silently
   discarding the most recent observation, and overstating the predictive
   variance (`P_pred` > `P_filt`), which flattens the differences between
   candidate probes.

Both are fixed by `_OnlineSSM`, one stateful filter per group that consumes
`(x_t, y_t)` pairs incrementally. It accumulates `log p(y_{1:t} | M_j)` from the
one-step innovations (so the posterior combines with the *base* prior, no double
counting) and tracks the filtered state `z_{t|t}` (available online — filtering
only looks backwards; it is *smoothing* that needs the future). Verified to match
`_innovations_log_likelihood` and `kalman_filter` to machine precision, and it
drops the belief update from O(T²) to O(T) — ~11× faster at T=30.

**EM (`utils/em_algorithm.py`).** EM was silently exhausting `max_iter=100` on
every run ever recorded, so all previously reported parameter estimates were
badly under-converged (e.g. λ̂ = 0.40 against a true 0.80).

3. *Returns the highest-likelihood iterate, not the last one.* The two failure
   modes pull opposite ways — one group needed >2000 iterations to approach the
   optimum, another peaked at iteration 24 — so no single `max_iter` is correct.
4. *`tol` is now relative.* The old `tol=1e-4` was absolute, against a
   log-likelihood of order −3×10³, i.e. ~1e-8 relative. It essentially never
   tripped, which is why EM always ran to `max_iter`.
5. *Convergence is now reported.* A `RuntimeWarning` on non-convergence, plus an
   optional `return_info=True` giving `converged / n_iter / best_iter /
   best_log_lik / n_decreases`.
6. *`sigma_e_init` defaults to `np.std(Y)`.* The hard-coded `0.2` against a true
   `σ_e ≈ 1.0` converged to a measurably worse optimum (1.2 nats, with λ̂ less
   than half its better value).
7. *The M-step is now a real M-step.* It summed `t=1..T-1`, treating `z_0` as a
   free initial value, while the filter imposed `z_{-1} ~ N(0, 1)` and scored a
   `t=0` transition. Maximising a different objective than you score removes EM's
   monotonicity guarantee — and it showed: the log-likelihood **decreased on 446
   of 469 steps** for group 1, peaking at iteration 24 and then drifting downhill
   until the tolerance check mistook the slowing decline for convergence. Both
   sides now use `z_{-1} = 0` exactly (`_P0 = 0`, matching what both data
   generators actually do) and the M-step includes `t=0`.

Result: **0 decreasing steps** in both groups, group 1 converges in **19**
iterations to a *better* optimum than the old 470-iteration run, and σ_e is
recovered to within 0.3%.

**Experiment scripts.** Seeded (`np.random.seed(SEED)`) so runs reproduce;
`experiment2.py` gained the per-timestep log and CSV export; `experiment1.py` no
longer rebinds its `params` loop variable to the EM output.

*Active-learning comparison is now paired.* `PatientSimulator(..., seed=i)` gives
every policy the same noise realisation for patient `i` (common random numbers),
and `run_patient_with_active_learning(..., seed=i)` varies the `random` policy's
probe sequence across patients — with the default `seed=0` every patient received
an **identical** `x` sequence, which understated the baseline.

**Plots (`utils/plots.py`).** `plot_accuracy_active_learning` keeps its old
signature and gains `title` / `subtitle` / `chance`. Fixed categorical palette
(assigned in dict order, never cycled, so a policy keeps its colour if another is
dropped), 2px round-capped lines, error bands as a 12% wash, recessive solid
hairline grid, dashing reserved for the chance threshold, text in ink tokens
rather than series colours, and a subtitle carrying `n`, `T`, the noise regime,
the band definition and the seed so the figure is self-describing.

## Known limitations

- **Only 5 of the 6 parameters are identifiable.** Substituting the state
  equation into the observation equation gives the reduced form
  `y_t = λ y_{t-1} + (βα+γ) x_t − γλ x_{t-1} + (β w_t + e_t − λ e_{t-1})`, so the
  data pins down `λ`, `(βα+γ)`, `γλ` and two noise moments — five quantities for
  six parameters. The leftover is an exact one-dimensional ridge:
  `(α, β, σ_w) → (cα, β/c, cσ_w)` leaves the likelihood unchanged.
  *Classification is unaffected* (the marginal likelihood is invariant), but α, β
  and σ_w individually are **not recoverable** and must not be read as mediation
  strengths. Worse, the direct effect `γ` and the mediated effect `βα` enter
  together at lag 0; only the `−γλ x_{t-1}` term separates them, and that vanishes
  as `λ → 0`. Fix by pinning the scale (`β ≡ 1` or `σ_w ≡ 1`) and reporting the
  invariants, or by adding an observation channel that breaks the tie.
- **Non-convergence on the flat direction is expected, not alarming.** 5 of the
  16 group fits in `experiment1.py` still warn at `max_iter=1000` (all in the
  mixed-noise / `gamma_low_high` regimes, each with a last log-likelihood change
  of ~1e-4) because EM crawls along the scale ridge. The
  *identified* quantities are already stable — λ, b0, b1 and the noise moments
  agree to <0.005 between iteration 1000 and 2000 — while the raw parameters keep
  drifting. Read the invariants, not the raw values.
- **The logistic-regression baseline is weak.** No feature standardisation,
  default `C=1.0`, 100 correlated raw features from 100 samples — hence ~0.5
  accuracy on the full-time-series variant. Add `StandardScaler` + a small CV over
  `C`, and give the moments variant a lag-1 *cross*-covariance
  `cov(x_{t-1}, y_t)`, which is precisely the mediation signal. The headline
  comparison is vulnerable until then.
- **The notebook's committed outputs are stale.** Its *source* now carries the
  same seeds and fixes as the scripts, but the stored outputs were produced before
  any of them (`divide by zero` warnings and all), so the figures and the 0.80
  accuracy in it are not what the code now produces. It needs re-executing.
- **EM and inference disagree about the initial state.** EM now uses
  `z_{-1} = 0` exactly (correct — it is what the generators do), but
  `kalman_filter`, `infer_patient` and `_innovations_log_likelihood` still default
  to `P0 = 1.0`, which overstates z's stationary variance (≈0.28 for group 0).
  It is a startup transient that washes out, but aligning them would be tidier —
  and would require rerunning experiment 2.
- **The Monte Carlo policies still dominate the runtime**, though far less than
  they did. `draw_samples_from_mixture_distribution` was calling
  `scipy.stats.norm.rvs` once per sample in a list comprehension — ~30 µs of
  argument validation and dispatch to produce a single ~30 ns number. It is now
  one vectorised `rng.normal` call (104× faster on its own, ~27× end-to-end),
  which is also what made the per-policy seeding possible. The remaining cost is
  the *density* loops in `evaluate_Shannon_entropy` /
  `evaluate_mutual_information`, which still evaluate one Gaussian per sample in
  a list comprehension; those have no randomness, so vectorising them is pure
  speed with no effect on results.
- **No tests.** The invariants checked by hand — online filter ≡ batch filter, EM
  monotonicity, EM recovering the identified quantities on long low-noise data —
  are exactly what a small test module should pin.
- **Dead code**: `class_posterior`, `choose_next_x_v2`, `_filter_to_last_state`
  and `_predict_y_moments` in `active_learning.py` are all now unreachable.

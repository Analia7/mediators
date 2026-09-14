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
| `utils/monte_carlo_entropy_minimization.py` | Entropy-minimization policy (minimise expected posterior entropy) |
| `utils/monte_carlo_mutual_information.py` | Mutual-information policy (BALD) |
| `utils/baseline.py` | Logistic-regression baselines |
| `utils/plots.py` | All figures |
| `experiment1.py` | Classification accuracy across 8 noise / γ regimes |
| `experiment2.py` | Active-learning policy comparison; writes `results/*.png` + `*.csv` |
| `experiment1_seed_sweep.py` | Refits experiment 1 across 5 seeds to measure how much of each regime's accuracy is just the data draw |
| `experiment2_seed_sweep.py` | Reruns experiment 2 across 5 seeds to put a spread on the policy comparison; seed 0 reproduces `experiment2.py` exactly, and the script checks that |

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
MPLBACKEND=Agg $PY -u experiment2_seed_sweep.py > results/experiment2_seed_sweep_log.txt 2>&1   # ~8 min
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
hairline grid, dashing reserved for the chance threshold, and text in ink tokens
rather than series colours.

*Titles and subtitles are opt-in.* Every figure here is captioned in a document,
so `plot_scatter_by_group`, `plot_mean_std`, `plot_trajectories` and
`plot_accuracy_active_learning` all draw no title and no subtitle unless one is
passed. What the figure must carry itself stays on the figure: axis and row
labels, the per-column headings (`Group 0 · Patient 3 (correct)`), the legend,
and the per-panel log-likelihood / coverage annotations. Run configuration —
`n`, `T`, the regime, the seed, the band definition — belongs to the caption, and
is recorded in the CSVs and logs under `results/` either way.

*Policy display names are separate from policy identifiers.* Figures label the
policies `Random selection` / `Minimum entropy` / `Mutual information` via
`POLICY_LABELS` in `utils/active_learning.py`, while the identifiers used as dict
keys, CSV columns and Monte Carlo stream indices stay lowercase (`random`,
`entropy minimization`, `mutual information`). Both halves now describe the same
thing — the policy minimises the expected posterior Shannon entropy — since the
identifier was renamed from `uncertainty sampling` on 2026-09-11 (see below).

*`plot_trajectories` is now a predicted-vs-actual figure.* It draws **one column
per group** instead of overlaying the groups on shared axes — the columns hold
*different patients*, so the old layout invited a comparison that means nothing,
and four noisy lines per axis at `T = 50` was unreadable. Inside a column the
encoding is fixed and identical in every panel: blue solid = actual, orange
dashed = the fitted model's prediction, orange wash = the ±1 SD predictive band.
The band is the point of the rewrite — the prediction is open-loop from `x`
alone, so `y` carries `σ_e ≈ 0.9` of observation noise that no parameter estimate
can track, and the mean line alone makes an honest fit look bad. Rows share a
y-scale so the columns stay comparable.

`x_t` is not plotted — it is i.i.d. noise in this generator, so its own panel
says nothing, and it is implicit in both predictions anyway. Each panel is
annotated with a log-likelihood and the share of actual points inside the band,
and the two rows' likelihoods are **different quantities**: the observation row
gives the marginal `log p(y_{1:T} | M_j)` from the Kalman innovations — exactly
what `infer_patient` compares, so with a uniform prior the difference between two
columns is the classification's log-odds — while the latent row gives
`log p(z_{1:T} | M_j)` for the *true* path under the drawn predictive, since a
latent series has no marginal likelihood to quote. The annotation names the
variable (`log p(y)`, `log p(z)`) so the two are not read as comparable.

Column headings carry the classifier's verdict in parentheses: `(correct)` when
the column's model, the MAP label from that same likelihood comparison, and the
patient's true group all agree, `(incorrect)` otherwise. On a figure of one
patient per group that reads as whether the patient was classified correctly; on
one patient under both models it marks which column the likelihood ratio picked
and whether it was right. When `groups` holds the model rather than the patient's
group — the duplicated-patient case — pass `true_group=`.

## Notebook outputs refreshed 2026-09-10

`active_discovery_of_latent_dynamic_mediators.ipynb` was re-executed top to
bottom on the fixed code (kernel `hyperparameters`, `SEED = 0`), replacing
outputs that predated every fix above.

**It also now runs a named regime.** Every phase draws from
`MIXED_NOISE_SIGMA_LOW` (`utils/data_generation.py`) — experiment 1's
`mixed_noise_sigma_low`, and the regime `experiment2.py` runs. Previously the
notebook passed no parameters at all and so silently used the generator's
defaults, which differ in `σ_e` (0.9 against 1.0): a slightly *easier* problem
than any regime the scripts report, and therefore not comparable with either of
them. Phase 3 also had to be given `params=` explicitly, because
`PatientSimulator` falls back to `DEFAULT_PARAMS` (`σ_e = 0.9`) — so the
active-learning section had been simulating a different process from the
training and test data it was built on.

Because the notebook's training draw is `seed=(0, 0)` on the same regime,
**its EM fits are now identical to `experiment2.py`'s** — α̂ = 0.492, λ̂ = 0.715,
β̂ = 0.739, γ̂ = 0.131 for group 0, matching `results/experiment2_log.txt`
digit for digit. That equality is the cheapest available check that the notebook
and the script agree about phase 1.

| | Stale output (old code, old regime) | Now |
|---|---|---|
| Group 1 `λ̂` | 0.14 | 0.29 (true 0.40; group 0: 0.72 against 0.80) |
| Our approach | 0.80 | **0.74** |
| Baseline, full series | 0.55 | 0.42 |
| Baseline, moments | 0.68 | 0.47 |
| Active learning | flat plateaus, `divide by zero` warnings | both active policies lead `Random selection` throughout: `Minimum entropy` reaches 1.00 at step 11 and holds it, `Mutual information` at step 14, while `Random selection` first touches 1.00 only at step 21 and then falls back below it |

The accuracy gap over the baseline therefore *widens* — the baseline was the main
beneficiary of the old numbers, and it is the weak component (see limitations).
0.74 sits just above this regime's five-seed mean of 0.730 ± 0.036
(`results/experiment1_seed_sweep.csv`), so it is an ordinary draw for the regime
rather than a notably good or bad one; note that experiment 1 reporting 0.74 for
the same regime is a coincidence of two different data draws, not a match to
check against (it uses `seed=(0, 2, ·)`, the notebook `seed=(0, ·)`).

Two things in the output are expected rather than wrong: EM warns
non-convergence for training group 1 (`max_iter=1000`, last change ~3e-4) because
it is crawling the scale ridge, and the α̂/β̂/σ̂_w it reports for either group are
individually unidentified — read `λ`, `(βα+γ)` and the noise moments instead.

## Fixes applied 2026-09-11

**The initial state prior is now consistent across the codebase (`P0 = 0`).**
EM already assumed `z_{-1} = 0` exactly with no uncertainty (`_Z0`, `_P0` in
`utils/em_algorithm.py`) — which is what both generators do — but every *consumer*
of that fit defaulted to `P0 = 1.0`: `kalman_filter`, `_innovations_log_likelihood`
and `_OnlineSSM`. All three now default to `0.0`.

This had been filed below as a tidiness issue, on the grounds that a startup
transient washes out. It does not wash out of a *ratio*. `infer_patient`
classifies by differencing two marginal log-likelihoods, and the mismatched prior
costs each model a different amount — on experiment 2's fits, 1.29 nats for
group 0 against 0.01 nats for group 1. The residue was a systematic **+0.056 nat
tilt of the log-odds toward group 1**, which moved 5 of 100 test predictions.

Verified after the change, on experiment 2's training draw:

* `em_ssm`'s E-step log-likelihood now equals `_innovations_log_likelihood`
  summed over patients, for both groups (agreement to 1e-11, i.e. float summation
  order only). EM maximises exactly the quantity inference scores.
* `_OnlineSSM` still matches the batch filter exactly — same accumulated
  log-likelihood, same filtered `z`, `P` to 12 dp.
* `_e_step`'s sufficient statistics and log-likelihood were checked against the
  exact joint Gaussian posterior of `z_{0:T-1} | y` built in closed form
  (`z = M⁻¹(αx + w)`, `M = I − λL`): `E[z_t]`, `E[z_t²]` and the lag-one
  `E[z_t z_{t-1}]` all agree to ~2e-16, and `log p(y)` exactly.

**What moved.** Every artifact in `results/` and the notebook were regenerated.
EM is untouched by this change, so `experiment1_parameters.csv` is byte-identical
and the notebook's fits still match `experiment2.py`'s digit for digit.

| | Before | After |
|---|---|---|
| Notebook, our approach | 0.75 | 0.74 |
| Experiment 1, per-regime accuracy | — | 4 of 8 regimes moved, all by 0.01–0.02 |
| Experiment 1, five-seed means | — | −0.008 to +0.016, every one inside its own sd |
| Experiment 2, paired gap over random | ME +0.049 ± 0.002, MI +0.023 ± 0.004 | ME +0.050 ± 0.002, MI +0.026 ± 0.009 |

**The correctness argument is the reason for this change, not an accuracy gain.**
On seed 0 the four affected experiment-1 regimes all moved *down*, which looks
like a bias being removed — but across five seeds the direction is not systematic
(2–3 of 5 seeds move up in each regime) and the means shift by less than their own
spread. What did change consistently is the seed-to-seed sd, which **rose** in all
four affected regimes (median across regimes 0.008 → 0.013, `mixed_noise_sigma_low`
0.022 → 0.036). Sharper early likelihoods make borderline patients flip more
readily. Experiment 2's policy ordering and every conclusion drawn from it are
unchanged, and the sweep's seed-0 cross-check against `experiment2.py` still
passes.

## Renamed 2026-09-11: `uncertainty sampling` → `entropy minimization`

The policy is named for what it does. It picks the probe that **minimises** the
expected posterior Shannon entropy; "uncertainty sampling" conventionally means
the opposite — probing where the model is *least* certain — so the old name
described the wrong algorithm.

| Was | Now |
|---|---|
| `utils/monte_carlo_uncertainty_sampling.py` | `utils/monte_carlo_entropy_minimization.py` |
| `select_next_x_uncertainty_sampling` | `select_next_x_entropy_minimization` |
| identifier `"uncertainty sampling"` | identifier `"entropy minimization"` |

The figure label is unchanged: `POLICY_LABELS` still maps this policy to
`Minimum entropy`, which already said the right thing, so no committed figure's
legend moved.

**The rename changed no numbers.** The identifier indexes a Monte Carlo stream
through `_POLICY_STREAM`, and what `np.random.default_rng` actually consumes is
the stream *integer*, not the string — so preserving index `1` preserves every
draw. Verified: `results/experiment2_mixed_noise_sigmaw_low.csv` is bit-identical
across the rename (max |delta| = 0.0 over the whole numeric body); only the column
headers moved. The seed sweep's `policy` field and its seed-0 cross-check follow
the same rule, and both CSVs were regenerated so the cross-check still resolves
its columns by name.

`results/archive_pre_fix/` and any file written before 2026-09-11 still carry the
old identifier. Because stream index 1 was preserved, those numbers remain
directly comparable with current ones.

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
- **Non-convergence on the flat direction is expected, not alarming.** 2 of the
  16 group fits in `experiment1.py` warn at `max_iter=1000` — group 1 of
  `mixed_noise_sigma_low` and of `mixed_noise_sigma_high`, each with a last
  log-likelihood change of ~1e-4 — because EM crawls along the scale ridge. The
  *identified* quantities are already stable — λ, b0, b1 and the noise moments
  agree to <0.005 between iteration 1000 and 2000 — while the raw parameters keep
  drifting. Read the invariants, not the raw values.
- **The logistic-regression baseline is weak.** No feature standardisation,
  default `C=1.0`, 100 correlated raw features from 100 samples — hence ~0.5
  accuracy on the full-time-series variant. Add `StandardScaler` + a small CV over
  `C`, and give the moments variant a lag-1 *cross*-covariance
  `cov(x_{t-1}, y_t)`, which is precisely the mediation signal. The headline
  comparison is vulnerable until then.
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

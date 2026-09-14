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


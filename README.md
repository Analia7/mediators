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
| `utils/monte_carlo_entropy_minimization.py` | Entropy-minimization policy (minimize expected posterior entropy) |
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

## Method

### Phase 1 -- EM algorithm

Both steps use the same prior on the initial state that the data generators use:
a trajectory starts from $z_0 = 0$ **exactly**, so
$\mathbb{E}[z_0] = \mathbb{E}[z_0^2] = \mathbb{E}[z_1 z_0] = 0$ and $P_0 = 0$.

**E-step.** For each patient, run the Kalman filter from $(z_0 = 0, P_0 = 0)$ and
then the RTS smoother, and accumulate the sufficient statistics

$$
\mathbb{E}[z_t] = \hat{z}_{t|T}, \qquad
\mathbb{E}[z_t^2] = P_{t|T} + \hat{z}_{t|T}^2, \qquad
\mathbb{E}[z_t z_{t-1}] = G_t P_{t|T} + \hat{z}_{t|T}\, \hat{z}_{t-1|T},
$$

where $G_t = \lambda P_{t-1|t-1} / P_{t|t-1}$ is the lag-one smoother gain and
$\mathbb{E}[z_1 z_0] = 0$. The same pass returns the marginal log-likelihood
$\log p(y_{1:T})$ from the innovations, which is what the convergence check
monitors.

**M-step.** Maximize the expected complete-data log-likelihood $Q(\theta_j \mid \theta_j^{\text{old}})$
to estimate the parameters. To ease notation, the subscript $j$ is dropped below.
**All sums run over every transition $t = 1, \ldots, T$ and all $N$ patients in
the group**, with the $t = 1$ term contributing through $z_0 = 0$.

Differentiating $Q$ with respect to $\alpha$ and $\lambda$ and setting both
equations to zero yields the $2 \times 2$ linear system

$$
\begin{bmatrix}
\sum x_t^2 & \sum x_t\, \mathbb{E}[z_{t-1}] \\
\sum x_t\, \mathbb{E}[z_{t-1}] & \sum \mathbb{E}[z_{t-1}^2]
\end{bmatrix}
\begin{bmatrix} \alpha \\ \lambda \end{bmatrix}
=
\begin{bmatrix}
\sum x_t\, \mathbb{E}[z_t] \\
\sum \mathbb{E}[z_t z_{t-1}]
\end{bmatrix}.
$$

This is weighted least squares — regressing $\mathbb{E}[z_t]$ onto $x_t$ and
$\mathbb{E}[z_{t-1}]$.

After substituting the updated $\alpha$ and $\lambda$, the optimal variance is
the mean squared residual of the state equation under the smoothed posterior:

$$
\sigma_w^2 = \frac{1}{NT} \sum_{n=1}^{N} \sum_{t=1}^{T}
\mathbb{E}\!\left[ (z_t - \alpha x_t - \lambda z_{t-1})^2 \right],
$$

with the square expanded by linearity of expectation:

$$
\begin{aligned}
\mathbb{E}\!\left[ (z_t - \alpha x_t - \lambda z_{t-1})^2 \right]
&= \mathbb{E}[z_t^2]
 - 2\alpha\, x_t\, \mathbb{E}[z_t]
 - 2\lambda\, \mathbb{E}[z_t z_{t-1}] \\
&\quad + \alpha^2 x_t^2
 + 2\alpha\lambda\, x_t\, \mathbb{E}[z_{t-1}]
 + \lambda^2\, \mathbb{E}[z_{t-1}^2].
\end{aligned}
$$

The observation-equation parameters follow analogously, from the residual
$y_t - \beta z_t - \gamma x_t$: the linear system

$$
\begin{bmatrix}
\sum \mathbb{E}[z_t^2] & \sum x_t\, \mathbb{E}[z_t] \\
\sum x_t\, \mathbb{E}[z_t] & \sum x_t^2
\end{bmatrix}
\begin{bmatrix} \beta \\ \gamma \end{bmatrix}
=
\begin{bmatrix}
\sum y_t\, \mathbb{E}[z_t] \\
\sum y_t\, x_t
\end{bmatrix},
$$

followed by

$$
\sigma_e^2 = \frac{1}{NT} \sum_{n=1}^{N} \sum_{t=1}^{T}
\mathbb{E}\!\left[ (y_t - \beta z_t - \gamma x_t)^2 \right].
$$

Both variances are floored at $10^{-6}$ before the square root, so a collapsing
noise estimate cannot divide by zero on the next filter pass.

### Phase 2 -- inference

For each patient $n$:

1. Run the Kalman filter to infer $z_{1:T}$ under **both** models, $c_n = 0$ and
   $c_n = 1$, accumulating the marginal likelihood $p^j(y_{1:T})$ of each.
2. Infer $c_n$ from those marginal likelihoods. Under a uniform prior the
   normalized marginal likelihood *is* the class probability.


### Phase 3 — active sampling

#### Entropy minimization

Choosing the probe by active sampling requires the expected posterior entropy

$$
\mathbb{E}_{y_t \sim p(y_t \mid x_t)}\!\left[ H(c \mid x_t, y_t) \right]
$$

at each time step, where $H$ is the conditional Shannon entropy. Several policies
follow from it; we take the next action to be the one leaving the class posterior
as sharp as possible, i.e. **minimum expected entropy**:

$$
x_t^{*} = \arg\min_{x_t} \ \mathbb{E}_{y_t \sim p(y_t \mid x_t)}\!\left[ H(c \mid x_t, y_t) \right].
$$

The marginal distribution of $y_t$ is a mixture of Gaussians, so this expectation
has no closed form and is approximated by Monte Carlo.

**Algorithm 1 — active learning via entropy minimization for SSMs**
(`utils/monte_carlo_entropy_minimization.py`)

**Input:** prior $P(c = 1 \mid y_{1:t-1})$, KF state estimates $\hat{z}^{(c)}_{t-1|t-1}$,
KF variances $P^{(c)}_{t-1|t-1}$, sample size $N$, candidate inputs $\mathcal{X}$.

1. Compute the predictive variance for each group $c \in \{0, 1\}$:

   $$
   S^{(c)}_{t|t-1} \gets \beta_c^2 \left( \lambda_c^2 P^{(c)}_{t-1|t-1} + \sigma_{w,c}^2 \right) + \sigma_{e,c}^2.
   $$

2. For each candidate $x_t \in \mathcal{X}$ (steps 2–6), compute the predictive
   mean for each group $c \in \{0, 1\}$:

   $$
   \hat{y}^{(c)}_{t|t-1} \gets (\beta_c \alpha_c + \gamma_c)\, x_t + \beta_c \lambda_c \hat{z}^{(c)}_{t-1|t-1}.
   $$

3. Draw $N$ samples $\{y^{(s)}_t\}_{s=1}^{N}$ from the mixture
   $p(y_t \mid x_t, y_{1:t-1})$: for $s = 1, \ldots, N$, sample
   $c^{(s)} \sim \mathrm{Bernoulli}\left( P(c = 1 \mid y_{1:t-1}) \right)$ and then
   $y^{(s)}_t \sim \mathcal{N}\left( \hat{y}^{(c^{(s)})}_{t|t-1},\, S^{(c^{(s)})}_{t|t-1} \right)$.

4. Evaluate each sample under both group models, $c \in \{0, 1\}$:

   $$
   p(y^{(s)}_t \mid x_t, y_{1:t-1}, c) = \frac{1}{\sqrt{2\pi S^{(c)}_{t|t-1}}} \exp\left( -\frac{\left( y^{(s)}_t - \hat{y}^{(c)}_{t|t-1} \right)^2}{2 S^{(c)}_{t|t-1}} \right).
   $$

5. Turn each into a posterior and take its Shannon entropy:

   $$
   P(c = 1 \mid x_t, y^{(s)}_t, y_{1:t-1}) = \frac{p(y^{(s)}_t \mid x_t, y_{1:t-1}, c = 1)\, P(c = 1 \mid y_{1:t-1})}{\sum_{c'} p(y^{(s)}_t \mid x_t, y_{1:t-1}, c')\, P(c' \mid y_{1:t-1})},
   $$

   $$
   H^{(s)} \gets - \sum_{c \in \{0, 1\}} P(c \mid x_t, y^{(s)}_t, y_{1:t-1}) \ln P(c \mid x_t, y^{(s)}_t, y_{1:t-1}).
   $$

6. Approximate the expected entropy for this candidate:
   $\bar{H}_x \gets \frac{1}{N} \sum_{s=1}^{N} H^{(s)}$.

**Output:** the selected input $x_t^{*} = \arg\min_{x_t \in \mathcal{X}} \bar{H}_x$.

#### Mutual information

An alternative is to maximize the mutual information $I(c; y_t \mid x_t)$ between
the latent treatment group and the future outcome, i.e. the action with the
largest expected reduction in the Shannon entropy of $c$:

$$
x_t^{*} = \arg\max_{x_t} \Big[ H(y_t \mid x_t) - \mathbb{E}_{c \sim P(c)}\big[ H(y_t \mid x_t, c) \big] \Big],
$$

where $H(y_t \mid \cdot)$ is differential entropy. The second term is an
expectation of entropies of pure Gaussians and simplifies analytically to
$\frac{1}{2} \sum_c P(c) \ln(2\pi e \sigma_c^2)$. The total predictive entropy
$H(y_t \mid x_t)$ has no closed form — $y_t$ is marginally a mixture of Gaussians
— so it is again approximated by sampling the predictive mixture at each
candidate $x_t$.

**Algorithm 2 — active learning via mutual information for SSMs**
(`utils/monte_carlo_mutual_information.py`)

**Input:** prior $P(c = 1 \mid y_{1:t-1})$, KF state estimates $\hat{z}^{(c)}_{t-1|t-1}$,
KF variances $P^{(c)}_{t-1|t-1}$, sample size $N$, candidate inputs $\mathcal{X}$.

1. Compute the predictive variance for each group $c \in \{0, 1\}$:

   $$
   S^{(c)}_{t|t-1} \gets \beta_c^2 \left( \lambda_c^2 P^{(c)}_{t-1|t-1} + \sigma_{w,c}^2 \right) + \sigma_{e,c}^2.
   $$

2. Compute the analytic noise entropy:

   $$
   H_{\mathrm{noise}} \gets \frac{1}{2} \sum_{c \in \{0, 1\}} P(c \mid y_{1:t-1}) \ln \left( 2\pi e\, S^{(c)}_{t|t-1} \right).
   $$

3. For each candidate $x_t \in \mathcal{X}$ (steps 3–6), compute the predictive
   mean for each group $c \in \{0, 1\}$:

   $$
   \hat{y}^{(c)}_{t|t-1} \gets (\beta_c \alpha_c + \gamma_c)\, x_t + \beta_c \lambda_c \hat{z}^{(c)}_{t-1|t-1}.
   $$

4. Draw $N$ samples $\{y^{(s)}_t\}_{s=1}^{N}$ from the mixture
   $p(y_t \mid x_t, y_{1:t-1})$: for $s = 1, \ldots, N$, sample
   $c^{(s)} \sim \mathrm{Bernoulli}\left( P(c = 1 \mid y_{1:t-1}) \right)$ and then
   $y^{(s)}_t \sim \mathcal{N}\left( \hat{y}^{(c^{(s)})}_{t|t-1},\, S^{(c^{(s)})}_{t|t-1} \right)$.

5. Evaluate the full mixture density at each sample and average the log to get
   the predictive entropy:

   $$
   p(y^{(s)}_t \mid x_t, y_{1:t-1}) \gets \sum_{c \in \{0, 1\}} P(c \mid y_{1:t-1})\, \mathcal{N}\left( y^{(s)}_t \mid \hat{y}^{(c)}_{t|t-1},\, S^{(c)}_{t|t-1} \right),
   $$

   $$
   \hat{H}_x \gets - \frac{1}{N} \sum_{s=1}^{N} \ln p(y^{(s)}_t \mid x_t, y_{1:t-1}).
   $$

6. Estimated mutual information: $I_x \gets \hat{H}_x - H_{\mathrm{noise}}$.

**Output:** the selected input $x_t^{*} = \arg\max_{x_t \in \mathcal{X}} I_x$.

### References

1. R. H. Shumway and D. S. Stoffer, *An approach to time series smoothing and
   forecasting using the EM algorithm*, 1982.
2. S. Särkkä, *Bayesian Filtering and Smoothing* — appendix (and Ch. 13).
3. V. Elvira and É. Chouzenoux, *GraphEM*, 2022.
4. B. Settles, *Active Learning Literature Survey*, 2009.

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
| Patient `i`'s noise | `PatientSimulator(seed=i)` | Same realization for all policies — a *paired* comparison |
| Patient `i`'s probes | `run_patient_with_active_learning(seed=i)` | The `random` policy's `x` sequence |
| Monte Carlo draws | derived from the same `seed` | One stream **per (patient, policy)** |


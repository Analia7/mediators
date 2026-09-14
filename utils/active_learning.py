import numpy as np
from utils.kalman_filter import kalman_filter
from utils.monte_carlo_mutual_information import select_next_x_mutual_information
from utils.monte_carlo_entropy_minimization import select_next_x_entropy_minimization
from utils.infer_patient import _innovations_log_likelihood


# Stream index per policy, so each policy's Monte Carlo draws are independent and
# reproducible on their own -- run one policy in isolation and it behaves exactly
# as it did inside the full comparison. Keyed by an explicit table rather than
# hash(policy): Python randomises string hashing per process, which would make
# runs irreproducible.
_POLICY_STREAM = {"random": 0, "entropy minimization": 1, "mutual information": 2}

# Display names for figures. The keys above are the *identifiers*: they name the
# CSV columns, key every stored result, and index the Monte Carlo streams via the
# table above. What the RNG actually consumes is the stream *integer*, not the
# string, so an identifier can be renamed without changing a single draw -- but
# only if its integer is preserved, and only if every CSV written under the old
# name is regenerated, because the cross-check in experiment2_seed_sweep.py looks
# up columns by name. This policy was called "uncertainty sampling" until
# 2026-09-11; it was renamed because the old term conventionally means probing
# where the model is *most* uncertain, which is the opposite of what it does --
# it picks the probe minimising the expected posterior Shannon entropy. Stream
# index 1 was kept, so the rename left every number identical.
POLICY_LABELS = {
    "random": "Random selection",
    "entropy minimization": "Minimum entropy",
    "mutual information": "Mutual information",
}

def class_posterior(X, Y, estimated_params, prior_1=0.5):
    X = np.asarray(X)
    Y = np.asarray(Y)
 
    prior_0 = 1.0 - prior_1
 
    # --- Marginal log-likelihoods under each model ---
    log_lik_0 = _innovations_log_likelihood(Y, X, estimated_params[0])
    log_lik_1 = _innovations_log_likelihood(Y, X, estimated_params[1])
 
    # --- Posterior in log-space for numerical stability ---
    # log unnormalised posteriors
    log_post_0 = log_lik_0 + np.log(prior_0)
    log_post_1 = log_lik_1 + np.log(prior_1)
 
    # log-sum-exp normalisation
    log_normaliser = np.logaddexp(log_post_0, log_post_1)
    prob_1 = np.exp(log_post_1 - log_normaliser)
    return prob_1

class _OnlineSSM:
    """
    Per-group Kalman filter that consumes one (x_t, y_t) pair at a time.

    Unlike calling `kalman_filter` on the whole history at every timestep, this
    keeps the recursion's state, so it gives us two things the active-learning
    loop needs:

      * `log_lik` : the accumulated marginal log-likelihood log p(y_{1:t} | M_j),
        built up from the one-step-ahead innovations. Combining this with the
        base prior gives the class posterior directly (no double counting).
      * `z`, `P`  : the *filtered* state E[z_t | y_{1:t}] and its variance.
        Filtering only ever looks backwards, so this is available online — it is
        smoothing (see `rts_smoother`) that would need the future.
    """

    def __init__(self, params, z0=0.0, P0=0.0):
        self.p = params
        self.z, self.P = float(z0), float(P0)
        self.log_lik = 0.0

    def update(self, x_t, y_t):
        """Advance one step with the chosen x_t and the observed y_t."""
        p = self.p

        # --- Predict z_t from z_{t-1|t-1} ---
        z_pred = p['alpha'] * x_t + p['lam'] * self.z
        P_pred = p['lam'] ** 2 * self.P + p['sigma_w'] ** 2

        # --- Innovation, and its contribution to log p(y_{1:t}) ---
        v = y_t - p['beta'] * z_pred - p['gamma'] * x_t
        S = p['beta'] ** 2 * P_pred + p['sigma_e'] ** 2
        self.log_lik += -0.5 * (np.log(2 * np.pi * S) + v ** 2 / S)

        # --- Update to the filtered state z_{t|t} ---
        K = (p['beta'] * P_pred) / S
        self.z = z_pred + K * v
        self.P = (1 - K * p['beta']) * P_pred


def steps_to_threshold(accuracy, threshold):
    """
    First time step at which an accuracy curve reaches `threshold` and never
    drops back below it, 1-indexed to match the figures. None if it never does.

    Sustained rather than first-crossing on purpose: with 50-100 patients a
    single curve wobbles by ~1/n per step, so a first crossing can be one
    patient flipping. Shared by experiment2.py and its seed sweep, which have to
    quote the same number.
    """
    ok = np.asarray(accuracy) >= threshold
    if not ok.any():
        return None
    t = len(ok)
    while t > 0 and ok[t - 1]:
        t -= 1
    return t + 1 if t < len(ok) else None


def run_patient_with_active_learning(patient, estimated_params, candidates,
                               policy="entropy minimization", T=50, prior_1=0.5, seed=0):
    """
    Parameters
    ----------
    patient : Patient
        The patient object to run the simulation on.
    estimated_params : dict
        Estimated parameters for both groups. Should contain keys 0 and 1, each mapping to a dict of parameters.
    candidates : array-like, shape (n_candidates,)
        Candidate input signals to choose from for x_{t+1}.
    policy : str, optional
        The policy to use for selecting the next input signal. Options are "random", "entropy minimization" or "mutual information"
    T: int, optional
        The number of timesteps to run the simulation for. Defaults to 50.
    prior_1 : float, optional
        Prior probability of belonging to group 1, P(c_n = 1). Defaults to 0.5.
    seed : int, optional
        Seeds this patient's two independent streams: the probe sequence used by
        the "random" policy, and the Monte Carlo draws used by the two active
        policies. Vary it per patient -- a shared seed gives every patient the
        same probe sequence. Defaults to 0.

    Returns
    -------
    probs : np.ndarray, shape (T,)
        probs[t] is the posterior P(c_n = 1 | y_{1:t}) after t observations, so
        probs[0] is the prior.
    correct : np.ndarray of bool, shape (T,)
        Whether the MAP label at each timestep matches the patient's true group.
    """
    if policy not in _POLICY_STREAM:
        raise ValueError(f"unknown policy {policy!r}")

    true_group = patient.group

    # if policy is "random"
    x_rng = np.random.default_rng(10_000 + seed)
    # Monte Carlo stream for the active policies, distinct per (patient, policy).
    mc_rng = np.random.default_rng([20_000 + seed, _POLICY_STREAM[policy]])
    lo, hi = float(candidates.min()), float(candidates.max())

    probs   = np.empty(T)
    correct = np.empty(T, dtype=bool)

    # One running filter per group: accumulates log p(y_{1:t} | M_j) and tracks
    # the filtered state that the policies extrapolate from.
    filters = {group_label: _OnlineSSM(estimated_params[group_label])
               for group_label in [0, 1]}

    # t = 0: nothing observed yet, so the belief is just the prior. Probe at
    # random, since no candidate is informative under an empty history.
    probs[0]   = prior_1
    correct[0] = (int(prior_1 > 0.5) == true_group)
    next_x     = x_rng.uniform(lo, hi)

    for t in range(T):
        y_t = patient.step(next_x)
        for f in filters.values():
            f.update(next_x, y_t)

        if t == T - 1:
            break

        # --- Posterior after t+1 observations, in log-space ---
        # The accumulated log_lik already covers the whole history, so this is
        # combined with the *base* prior. Folding in the previous posterior
        # instead would count the same evidence once per timestep.
        log_post_0 = np.log(1.0 - prior_1) + filters[0].log_lik
        log_post_1 = np.log(prior_1)       + filters[1].log_lik
        posterior_1 = np.exp(log_post_1 - np.logaddexp(log_post_0, log_post_1))

        probs[t + 1]   = posterior_1
        correct[t + 1] = (int(posterior_1 > 0.5) == true_group)

        # --- Choose x_{t+1} ---
        # The policies advance one step from here, so hand them the filtered
        # state z_{t|t}: using the predicted state z_{t|t-1} would throw away
        # the observation we just made.
        z_means = {group_label: filters[group_label].z for group_label in [0, 1]}
        z_vars  = {group_label: filters[group_label].P for group_label in [0, 1]}

        if policy == "random":
            next_x = x_rng.uniform(lo, hi)
        elif policy == "entropy minimization":
            next_x = select_next_x_entropy_minimization(candidates, estimated_params, z_means, z_vars, prior_1=posterior_1, samples_size=50, rng=mc_rng)

        elif policy == "mutual information":
            next_x = select_next_x_mutual_information(candidates, estimated_params, z_means, z_vars, prior_1=posterior_1, samples_size=50, rng=mc_rng)
        else:
            raise ValueError(f"unknown policy {policy!r}")

    return probs, correct


























def _filter_to_last_state(X, Y, params):
    """Filtered state z_{T|T} and variance P_{T|T} after the existing data.
    With no data, fall back to the filter's prior init (0, 0) — the same
    prior _innovations_log_likelihood uses — so early timesteps don't crash."""
    if len(Y) == 0:
        return 0.0, 0.0
    z_filt, P_filt, _, _ = kalman_filter(Y, X,
                                         params['alpha'], params['lam'],
                                         params['beta'],  params['gamma'],
                                         params['sigma_w'], params['sigma_e'])
    return z_filt[-1], P_filt[-1]


def _predict_y_moments(params, z_last, P_last, next_x):
    """One-step-ahead predictive mean and variance of y_{T+1} given next_x.

        z_pred = alpha*next_x + lam*z_last
        P_pred = lam^2 * P_last + sigma_w^2
        mu     = beta*z_pred + gamma*next_x
        var    = beta^2 * P_pred + sigma_e^2
    """
    Q = params['sigma_w'] ** 2
    R = params['sigma_e'] ** 2
    z_pred = params['alpha'] * next_x + params['lam'] * z_last
    P_pred = params['lam'] ** 2 * P_last + Q
    mu  = params['beta'] * z_pred + params['gamma'] * next_x
    var = params['beta'] ** 2 * P_pred + R
    return mu, var

def choose_next_x_v2(X, Y, estimated_params, candidates, timestep,
                  prior_1=0.5, criterion="skl"):
    """Pick the candidate input that makes the two groups' one-step-ahead
    predictions of y disagree most. prior_1 is unused here (the criterion is
    prior-free) but kept for call-signature compatibility."""
    X = np.asarray(X[:timestep])
    Y = np.asarray(Y[:timestep])

    # Filter over existing data ONCE per group — independent of the candidate.
    z0, P0 = _filter_to_last_state(X, Y, estimated_params[0])
    z1, P1 = _filter_to_last_state(X, Y, estimated_params[1])

    if criterion == "separation":          # (mu0 - mu1)^2 / (s0^2 + s1^2)
        def score_fn(m0, v0, m1, v1):
            return (m0 - m1) ** 2 / (v0 + v1)
    elif criterion == "skl":               # symmetric KL between the two Gaussians
        def score_fn(m0, v0, m1, v1):
            d2 = (m0 - m1) ** 2
            return 0.5 * ((v0 + d2) / v1 + (v1 + d2) / v0) - 1.0
    else:
        raise ValueError(f"unknown criterion {criterion!r}")

    best_candidate, best_score = None, -np.inf
    for candidate in candidates:
        m0, v0 = _predict_y_moments(estimated_params[0], z0, P0, candidate)
        m1, v1 = _predict_y_moments(estimated_params[1], z1, P1, candidate)
        s = score_fn(m0, v0, m1, v1)
        if s > best_score:
            best_score, best_candidate = s, candidate
    return best_candidate

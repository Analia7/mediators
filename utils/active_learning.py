import numpy as np
from utils.kalman_filter import kalman_filter
from utils.infer_patient import _innovations_log_likelihood


def estimate_next_y(X, Y, estimated_params, next_x, group_label):
    """
    Estimate the next observation y_{T+1} for a patient given their data and the estimated parameters.

    Parameters
    ----------
    X : array-like, shape (T,)
        Input signals x_t for the patient.
    Y : array-like, shape (T,)
        Observations y_t for the patient.
    estimated_params : dict
        Estimated parameters for both groups. Should contain keys 0 and 1, each mapping to a dict of parameters.
    next_x : float
        The next input signal x_{T+1} for which we want to estimate y_{T+1}.
    group_label : int
        The group label (0 or 1) indicating which group's parameters to use for estimation.

    Returns
    -------
    estimated_y_next : float
        The estimated next observation y_{T+1}.
    """
    params = estimated_params[group_label]
    
    # Run Kalman filter on the existing data to get the last state estimate
    z_filt, P_filt, z_pred, P_pred = kalman_filter(Y, X, 
                                                   params['alpha'], 
                                                   params['lam'], 
                                                   params['beta'], 
                                                   params['gamma'], 
                                                   params['sigma_w'], 
                                                   params['sigma_e'])
    
    # Get the last filtered state and its variance
    z_last = z_filt[-1]
    P_last = P_filt[-1]
    
    # Predict the next state using the state equation
    z_next_pred = params['alpha'] * next_x + params['lam'] * z_last
    
    # Predict the next observation using the observation equation
    estimated_y_next = params['beta'] * z_next_pred + params['gamma'] * next_x
    
    return estimated_y_next

def predicted_probability_group1(X, Y, estimated_params, next_x, timestep, prior_1=0.5):
    # trim the input arrays to the current timestep + 1
    X = np.asarray(X)
    Y = np.asarray(Y)
 
    prior_0 = 1.0 - prior_1
    
    # create two arrays such that one contains the last y under group 0 and the other under group 1
    Y_est_group_0 = np.copy(Y)
    Y_est_group_1 = np.copy(Y)

    y_next_0 = estimate_next_y(X, Y, estimated_params, next_x, group_label=0)
    y_next_1 = estimate_next_y(X, Y, estimated_params, next_x, group_label=1)
    Y_est_group_0 = np.append(Y_est_group_0, y_next_0)
    Y_est_group_1 = np.append(Y_est_group_1, y_next_1)
    
    X = np.append(X, next_x)  # Append the candidate next_x to the input signals

    # --- Marginal log-likelihoods under each model ---
    log_lik_0 = _innovations_log_likelihood(Y_est_group_0, X, estimated_params[0])
    log_lik_1 = _innovations_log_likelihood(Y_est_group_1, X, estimated_params[1])
 
    # --- Posterior in log-space for numerical stability ---
    # log unnormalised posteriors
    log_post_0 = log_lik_0 + np.log(prior_0)
    log_post_1 = log_lik_1 + np.log(prior_1)
 
    # log-sum-exp normalisation
    log_normaliser = np.logaddexp(log_post_0, log_post_1)
    prob_1 = np.exp(log_post_1 - log_normaliser)
    return prob_1

def choose_next_x(X, Y, estimated_params, candidates, timestep, prior_1=0.5):
    """
    Choose the next input signal x_{T+1} from a set of candidates based on the current patient data and estimated parameters.

    Parameters
    ----------
    X : array-like, shape (T,)
        Input signals x_t for the patient.
    Y : array-like, shape (T,)
        Observations y_t for the patient.
    estimated_params : dict
        Estimated parameters for both groups. Should contain keys 0 and 1, each mapping to a dict of parameters.
    candidates : array-like, shape (n_candidates,)
        Candidate input signals to choose from for x_{T+1}.
    timestep : int
        The current timestep T (0-indexed).
    prior_1 : float
        Prior probability of belonging to group 1, P(c_n = 1). Defaults to 0.5.

    Returns
    -------
    best_candidate : float
        The candidate input signal that maximizes the expected information gain or minimizes uncertainty.
    """
    best_candidate = None
    best_prob_diff = np.inf

    for candidate in candidates:
        prob_1 = predicted_probability_group1(X, Y, estimated_params, candidate, timestep, prior_1)
        prob_diff = abs(prob_1 - 0.5)  # We want to minimize the difference from 0.5 (uncertainty)

        if prob_diff < best_prob_diff:
            best_prob_diff = prob_diff
            best_candidate = candidate

    return best_candidate

def class_posterior(X, Y, estimated_params, timestep, prior_1=0.5):
    X = X[:timestep]
    Y = Y[:timestep]
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

def run_patient_with_active_learning(patient, estimated_params, candidates,
                               policy="active", T=50, prior_1=0.5, seed=0):
    true_group = patient.group

    # if policy is "random"
    x_rng = np.random.default_rng(10_000 + seed)
    lo, hi = float(candidates.min()), float(candidates.max())

    probs   = np.empty(T)
    correct = np.empty(T, dtype=bool)
    

    for t in range(T):
        X, _, Y = patient.history()

        if t == 0:
            probs[t] = prior_1
            correct[t] = (int(prior_1 > 0.5) == true_group)
        else:
            p1 = class_posterior(X, Y, estimated_params, t, prior_1)
            probs[t] = p1
            correct[t] = (int(p1 > 0.5) == true_group)

        # choose next x
        if policy == "random":
            next_x = x_rng.uniform(lo, hi)
        elif policy == "active":
            if len(Y) == 0:                       # no data yet → can't inform the choice
                next_x = x_rng.uniform(lo, hi)
            else:
                next_x = choose_next_x(X, Y, estimated_params, candidates, t, prior_1)
        else:
            raise ValueError(f"unknown policy {policy!r}")

        patient.step(next_x)

    return probs, correct


























def _filter_to_last_state(X, Y, params):
    """Filtered state z_{T|T} and variance P_{T|T} after the existing data.
    With no data, fall back to the filter's prior init (0, 1) — the same
    prior _innovations_log_likelihood uses — so early timesteps don't crash."""
    if len(Y) == 0:
        return 0.0, 1.0
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

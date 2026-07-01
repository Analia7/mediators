import numpy as np
from utils.kalman_filter import kalman_filter
 
 
def _innovations_log_likelihood(Y, X, params):
    """
    Compute the marginal log-likelihood log p(y_{1:T}) using the Kalman filter
    innovations sequence.
 
    At each step t, the innovation is:
        v_t = y_t - beta * z_t^pred - gamma * x_t  ~  N(0, S_t)
    where S_t = beta^2 * P_t^pred + sigma_e^2 is the innovation variance.
 
    The marginal log-likelihood is:
        log p(y_{1:T}) = sum_t [ -0.5 * log(2 * pi * S_t) - 0.5 * v_t^2 / S_t ]
    """
    alpha   = params["alpha"]
    lam     = params["lam"]
    beta    = params["beta"]
    gamma   = params["gamma"]
    sigma_w = params["sigma_w"]
    sigma_e = params["sigma_e"]
 
    T = len(Y)
    Q = sigma_w ** 2
    R = sigma_e ** 2
 
    log_lik = 0.0
    z_prev  = 0.0
    P_prev  = 1.0
 
    for t in range(T):
        # Predict
        z_pred = alpha * X[t] + lam * z_prev
        P_pred = lam ** 2 * P_prev + Q
 
        # Innovation and its variance
        v_t = Y[t] - beta * z_pred - gamma * X[t]
        S_t = beta ** 2 * P_pred + R
 
        log_lik += -0.5 * (np.log(2 * np.pi * S_t) + v_t ** 2 / S_t)
 
        # Update (needed to carry forward z_prev, P_prev)
        K      = (beta * P_pred) / S_t
        z_prev = z_pred + K * v_t
        P_prev = (1 - K * beta) * P_pred
 
    return log_lik
 
 
def infer_patient(X, Y, params_0, params_1, prior_1=0.5):
    """
    Infer the group label and latent trajectory z_{1:T} for a single patient.
 
    Runs the Kalman filter under both group models, computes the marginal
    likelihood p(y_{1:T}) under each via the innovations sequence, then
    forms the posterior group probability using Bayes' rule:
 
        P(c_n = 1 | y_{1:T}) =
            p^1(y_{1:T}) * prior_1
            -----------------------------------------------
            p^1(y_{1:T}) * prior_1 + p^0(y_{1:T}) * (1 - prior_1)
 
    The filtered trajectory z_{1:T} is returned under the most probable group.
 
    Parameters
    ----------
    X : array-like, shape (T,)
        Input signal for the patient.
    Y : array-like, shape (T,)
        Observations for the patient.
    params_0 : dict
        Estimated parameters for group 0.
        Keys: alpha, lam, beta, gamma, sigma_w, sigma_e.
    params_1 : dict
        Estimated parameters for group 1.
        Keys: alpha, lam, beta, gamma, sigma_w, sigma_e.
    prior_1 : float
        Prior probability of belonging to group 1, P(c_n = 1).
        Defaults to 0.5 (uniform / uninformative prior).
 
    Returns
    -------
    prob_1 : float
        Posterior probability P(c_n = 1 | y_{1:T}).
    c_n : int
        Inferred group label (0 or 1) — the most probable group.
    z_filt : np.ndarray, shape (T,)
        Kalman-filtered latent trajectory under the inferred group's model.
    P_filt : np.ndarray, shape (T,)
        Filtered state variances under the inferred group's model.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
 
    prior_0 = 1.0 - prior_1
 
    # --- Marginal log-likelihoods under each model ---
    log_lik_0 = _innovations_log_likelihood(Y, X, params_0)
    log_lik_1 = _innovations_log_likelihood(Y, X, params_1)
 
    # --- Posterior in log-space for numerical stability ---
    # log unnormalised posteriors
    log_post_0 = log_lik_0 + np.log(prior_0)
    log_post_1 = log_lik_1 + np.log(prior_1)
 
    # log-sum-exp normalization
    log_normalizer = np.logaddexp(log_post_0, log_post_1)
    prob_1 = np.exp(log_post_1 - log_normalizer)
    c_n    = int(prob_1 >= 0.5)
 
    # --- Kalman filter under the inferred group's model ---
    params_inferred = params_1 if c_n == 1 else params_0
    z_filt, P_filt, _, _ = kalman_filter(
        Y, X,
        alpha   = params_inferred["alpha"],
        lam     = params_inferred["lam"],
        beta    = params_inferred["beta"],
        gamma   = params_inferred["gamma"],
        sigma_w = params_inferred["sigma_w"],
        sigma_e = params_inferred["sigma_e"],
    )
 
    return prob_1, c_n, z_filt, P_filt
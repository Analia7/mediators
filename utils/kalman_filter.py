import numpy as np
 
 
def kalman_filter(Y, X, alpha, lam, beta, gamma, sigma_w, sigma_e, z0=0.0, P0=0.0):
    """
    Kalman filter for the SSM:
      State:       z_t = alpha * x_t + lam * z_{t-1} + w_t,   w ~ N(0, sigma_w^2)
      Observation: y_t = beta  * z_t + gamma * x_t   + e_t,   e ~ N(0, sigma_e^2)
 
    Parameters
    ----------
    Y : array-like, shape (T,)
        Observations y_t.
    X : array-like, shape (T,)
        Input signals x_t.
    alpha : float
        State equation coefficient on x_t.
    lam : float
        State equation coefficient on z_{t-1} (transition).
    beta : float
        Observation equation coefficient on z_t.
    gamma : float
        Observation equation coefficient on x_t.
    sigma_w : float
        Standard deviation of state noise w_t.
    sigma_e : float
        Standard deviation of observation noise e_t.
    z0 : float
        Initial state mean. Defaults to 0.0.
    P0 : float
        Initial state variance. Defaults to 0.0: both data generators start every
        trajectory from z_{-1} = 0 exactly, so the initial state carries no
        uncertainty. This must match what em_algorithm assumes (_Z0, _P0) -- a
        filter that scores a different initial prior than EM maximised reports a
        different log-likelihood for the same parameters, and since
        classification compares two such log-likelihoods, the mismatch does not
        cancel. The old default of 1.0 tilted the log-odds toward whichever
        model had the larger startup transient.
 
    Returns
    -------
    z_filt : np.ndarray, shape (T,)
        Filtered state means E[z_t | y_{1:t}].
    P_filt : np.ndarray, shape (T,)
        Filtered state variances Var[z_t | y_{1:t}].
    z_pred : np.ndarray, shape (T,)
        Predicted state means E[z_t | y_{1:t-1}].
    P_pred : np.ndarray, shape (T,)
        Predicted state variances Var[z_t | y_{1:t-1}].
    """
    Y = np.asarray(Y)
    X = np.asarray(X)
    T = len(Y)
 
    Q = sigma_w ** 2
    R = sigma_e ** 2
 
    z_filt = np.zeros(T)
    P_filt = np.zeros(T)
    z_pred = np.zeros(T)
    P_pred = np.zeros(T)
 
    z_prev = z0
    P_prev = P0
 
    for t in range(T):
        # --- Predict ---
        z_pred[t] = alpha * X[t] + lam * z_prev
        P_pred[t] = lam ** 2 * P_prev + Q
 
        # --- Update ---
        y_hat = beta * z_pred[t] + gamma * X[t]
        S = beta ** 2 * P_pred[t] + R
        K = (beta * P_pred[t]) / S
 
        z_filt[t] = z_pred[t] + K * (Y[t] - y_hat)
        P_filt[t] = (1 - K * beta) * P_pred[t]
 
        z_prev = z_filt[t]
        P_prev = P_filt[t]
 
    return z_filt, P_filt, z_pred, P_pred
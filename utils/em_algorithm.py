import numpy as np
from utils.kalman_filter import kalman_filter
from utils.rts_smoother import rts_smoother
 
 
def _e_step(X, Y, alpha, lam, beta, gamma, sigma_w, sigma_e):
    """
    E-step: run Kalman filter + RTS smoother for each patient.
 
    Returns smoothed sufficient statistics aggregated across all patients:
        Ez   : (n_patients, T)  E[z_t | Y]
        Ez2  : (n_patients, T)  E[z_t^2 | Y]  = P_smooth_t + Ez_t^2
        Ezz  : (n_patients, T)  E[z_t * z_{t-1} | Y]  (t >= 1)
    and the total log-likelihood.
    """
    n_patients, T = X.shape
 
    Ez  = np.zeros((n_patients, T))
    Ez2 = np.zeros((n_patients, T))
    Ezz = np.zeros((n_patients, T))      # index t holds E[z_t * z_{t-1}]
 
    log_lik = 0.0
    Q = sigma_w ** 2
    R = sigma_e ** 2
 
    for n in range(n_patients):
        z_filt, P_filt, z_pred, P_pred = kalman_filter(
            Y[n], X[n], alpha, lam, beta, gamma, sigma_w, sigma_e
        )
        z_smooth, P_smooth = rts_smoother(z_filt, P_filt, z_pred, P_pred, lam)
 
        Ez[n]  = z_smooth
        Ez2[n] = P_smooth + z_smooth ** 2
 
        # Smoothed cross-covariance E[z_t * z_{t-1}] via the lag-one smoother:
        # G_t = P_filt[t-1] * lam / P_pred[t]
        # Cov[z_t, z_{t-1} | Y] = G_t * P_smooth[t]
        # E[z_t * z_{t-1}] = Cov + Ez[t] * Ez[t-1]
        for t in range(1, T):
            G = (P_filt[t - 1] * lam) / P_pred[t]
            cov_cross = G * P_smooth[t]
            Ezz[n, t] = cov_cross + z_smooth[t] * z_smooth[t - 1]
 
        # Innovations log-likelihood (proxy; full complete-data LL computed in M-step)
        for t in range(T):
            innov = Y[n, t] - (beta * z_pred[t] + gamma * X[n, t])
            S = beta ** 2 * P_pred[t] + R
            log_lik += -0.5 * (np.log(2 * np.pi * S) + innov ** 2 / S)
 
    return Ez, Ez2, Ezz, log_lik
 
 
def _m_step(X, Y, Ez, Ez2, Ezz):
    """
    M-step: closed-form parameter updates given smoothed sufficient statistics.
 
    State equation:  z_t = alpha * x_t + lam * z_{t-1} + w_t
    Observation eq:  y_t = beta  * z_t + gamma * x_t   + e_t
    """
    n_patients, T = X.shape
 
    # ---- Update alpha and lam (state equation, t >= 1) ----
    # Minimise sum_n sum_{t=1}^{T} E[(z_t - alpha*x_t - lam*z_{t-1})^2]
    # -> solve 2x2 linear system in [alpha, lam]
    #
    # [ sum x_t^2        sum x_t*E[z_{t-1}]  ] [alpha]   [ sum x_t*E[z_t]         ]
    # [ sum x_t*E[z_{t-1}]  sum E[z_{t-1}^2] ] [lam  ] = [ sum E[z_t * z_{t-1}]  ]
 
    A = np.zeros((2, 2))
    b = np.zeros(2)
 
    for n in range(n_patients):
        xt  = X[n, 1:]          # x_t,   t=1..T-1
        ezt = Ez[n, 1:]         # E[z_t]
        ezt1 = Ez[n, :-1]       # E[z_{t-1}]
        ez2t1 = Ez2[n, :-1]     # E[z_{t-1}^2]
        ezzt  = Ezz[n, 1:]      # E[z_t * z_{t-1}]
 
        A[0, 0] += np.sum(xt ** 2)
        A[0, 1] += np.sum(xt * ezt1)
        A[1, 0] += np.sum(xt * ezt1)
        A[1, 1] += np.sum(ez2t1)
        b[0]    += np.sum(xt * ezt)
        b[1]    += np.sum(ezzt)
 
    alpha_new, lam_new = np.linalg.solve(A, b)
 
    # ---- Update sigma_w ----
    # Q = (1 / N*(T-1)) * sum_n sum_{t=1}^{T} E[(z_t - alpha*x_t - lam*z_{t-1})^2]
    sigma_w_sq = 0.0
    count_w = 0
    for n in range(n_patients):
        xt   = X[n, 1:]
        ezt  = Ez[n, 1:]
        ezt1 = Ez[n, :-1]
        ez2t1 = Ez2[n, :-1]
        ezzt  = Ezz[n, 1:]
        ez2t  = Ez2[n, 1:]
 
        sigma_w_sq += (
            np.sum(ez2t)
            - 2 * alpha_new * np.sum(xt * ezt)
            - 2 * lam_new   * np.sum(ezzt)
            + alpha_new ** 2 * np.sum(xt ** 2)
            + 2 * alpha_new * lam_new * np.sum(xt * ezt1)
            + lam_new ** 2  * np.sum(ez2t1)
        )
        count_w += T - 1
 
    sigma_w_new = np.sqrt(max(sigma_w_sq / count_w, 1e-6))
 
    # ---- Update beta and gamma (observation equation) ----
    # Minimise sum_n sum_t E[(y_t - beta*z_t - gamma*x_t)^2]
    # -> solve 2x2 linear system in [beta, gamma]
    #
    # [ sum E[z_t^2]   sum x_t*E[z_t] ] [beta ]   [ sum y_t*E[z_t] ]
    # [ sum x_t*E[z_t] sum x_t^2      ] [gamma] = [ sum y_t*x_t    ]
 
    C = np.zeros((2, 2))
    d = np.zeros(2)
 
    for n in range(n_patients):
        xt   = X[n]
        yt   = Y[n]
        ezt  = Ez[n]
        ez2t = Ez2[n]
 
        C[0, 0] += np.sum(ez2t)
        C[0, 1] += np.sum(xt * ezt)
        C[1, 0] += np.sum(xt * ezt)
        C[1, 1] += np.sum(xt ** 2)
        d[0]    += np.sum(yt * ezt)
        d[1]    += np.sum(yt * xt)
 
    beta_new, gamma_new = np.linalg.solve(C, d)
 
    # ---- Update sigma_e ----
    sigma_e_sq = 0.0
    count_e = 0
    for n in range(n_patients):
        xt   = X[n]
        yt   = Y[n]
        ezt  = Ez[n]
        ez2t = Ez2[n]
 
        sigma_e_sq += (
            np.sum(yt ** 2)
            - 2 * beta_new  * np.sum(yt * ezt)
            - 2 * gamma_new * np.sum(yt * xt)
            + beta_new ** 2  * np.sum(ez2t)
            + 2 * beta_new * gamma_new * np.sum(xt * ezt)
            + gamma_new ** 2 * np.sum(xt ** 2)
        )
        count_e += T
 
    sigma_e_new = np.sqrt(max(sigma_e_sq / count_e, 1e-6))
 
    return alpha_new, lam_new, beta_new, gamma_new, sigma_w_new, sigma_e_new
 
 
def em_ssm(
    X,
    Y,
    alpha_init,
    lam_init,
    beta_init,
    gamma_init,
    sigma_w_init,
    sigma_e_init,
    max_iter=100,
    tol=1e-4,
):
    """
    EM algorithm for a group-specific SSM.
 
    Runs the E-step (Kalman filter + RTS smoother per patient) and M-step
    (closed-form parameter updates) until convergence.
 
    Parameters
    ----------
    X : np.ndarray, shape (n_patients, T)
        Input signals for all patients in the group.
    Y : np.ndarray, shape (n_patients, T)
        Observations for all patients in the group.
    alpha_init, lam_init, beta_init, gamma_init : float
        Initial parameter values.
    sigma_w_init, sigma_e_init : float
        Initial noise standard deviations.
    max_iter : int
        Maximum number of EM iterations. Defaults to 100.
    tol : float
        Convergence threshold on the change in log-likelihood. Defaults to 1e-4.
 
    Returns
    -------
    params : dict
        Estimated parameters: alpha, lam, beta, gamma, sigma_w, sigma_e.
    log_likelihoods : list of float
        Log-likelihood after each E-step (useful for convergence diagnostics).
    """
    alpha, lam, beta, gamma, sigma_w, sigma_e = (
        alpha_init, lam_init, beta_init, gamma_init, sigma_w_init, sigma_e_init
    )
 
    log_likelihoods = []
 
    for i in range(max_iter):
        # E-step
        Ez, Ez2, Ezz, log_lik = _e_step(X, Y, alpha, lam, beta, gamma, sigma_w, sigma_e)
        log_likelihoods.append(log_lik)
 
        # Check convergence
        if i > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print(f"Converged at iteration {i + 1}")
            break
 
        # M-step
        alpha, lam, beta, gamma, sigma_w, sigma_e = _m_step(X, Y, Ez, Ez2, Ezz)
 
    params = {
        "alpha": alpha,
        "lam": lam,
        "beta": beta,
        "gamma": gamma,
        "sigma_w": sigma_w,
        "sigma_e": sigma_e,
    }
 
    return params, log_likelihoods
import warnings

import numpy as np
from utils.kalman_filter import kalman_filter
from utils.rts_smoother import rts_smoother


# Both data generators (generate_synthetic_samples, PatientSimulator) start a
# trajectory from z_{-1} = 0 exactly, so the initial state carries no uncertainty.
# EM uses that prior on both sides: the E-step filters from (z0=0, P0=0) and the
# M-step includes the t=0 transition with E[z_{-1}] = E[z_{-1}^2] = 0. The two
# MUST agree -- if the M-step maximises a different objective than the E-step
# scores, the algorithm is not an EM and can decrease the log-likelihood, which
# is what it used to do (the M-step summed t=1..T-1, i.e. it treated z_0 as a
# free initial value, while the filter imposed z_{-1} ~ N(0, 1)).
_Z0 = 0.0
_P0 = 0.0


def _e_step(X, Y, alpha, lam, beta, gamma, sigma_w, sigma_e):
    """
    E-step: run Kalman filter + RTS smoother for each patient.

    Returns smoothed sufficient statistics aggregated across all patients:
        Ez   : (n_patients, T)  E[z_t | Y]
        Ez2  : (n_patients, T)  E[z_t^2 | Y]  = P_smooth_t + Ez_t^2
        Ezz  : (n_patients, T)  E[z_t * z_{t-1} | Y]; index 0 is E[z_0 * z_{-1}]
                                which is 0 because z_{-1} = 0.
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
            Y[n], X[n], alpha, lam, beta, gamma, sigma_w, sigma_e,
            z0=_Z0, P0=_P0
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

        # Marginal log-likelihood log p(y_{1:T}) from the innovations sequence.
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

    Both sums run over the full t = 0..T-1, with z_{-1} = 0 contributing
    E[z_{-1}] = E[z_{-1}^2] = E[z_0 z_{-1}] = 0 at t = 0. Dropping t = 0 would
    make this the maximiser of a different model than the E-step scores.
    """
    n_patients, T = X.shape

    # ---- Update alpha and lam (state equation) ----
    # Minimise sum_n sum_{t=0}^{T-1} E[(z_t - alpha*x_t - lam*z_{t-1})^2]
    # -> solve 2x2 linear system in [alpha, lam]
    #
    # [ sum x_t^2           sum x_t*E[z_{t-1}] ] [alpha]   [ sum x_t*E[z_t]       ]
    # [ sum x_t*E[z_{t-1}]  sum E[z_{t-1}^2]   ] [lam  ] = [ sum E[z_t * z_{t-1}] ]

    A = np.zeros((2, 2))
    b = np.zeros(2)

    for n in range(n_patients):
        xt    = X[n]                                    # x_t,   t=0..T-1
        ezt   = Ez[n]                                   # E[z_t]
        ezt1  = np.concatenate(([0.0], Ez[n, :-1]))     # E[z_{t-1}], 0 at t=0
        ez2t1 = np.concatenate(([0.0], Ez2[n, :-1]))    # E[z_{t-1}^2], 0 at t=0
        ezzt  = Ezz[n]                                  # E[z_t z_{t-1}], 0 at t=0

        A[0, 0] += np.sum(xt ** 2)
        A[0, 1] += np.sum(xt * ezt1)
        A[1, 0] += np.sum(xt * ezt1)
        A[1, 1] += np.sum(ez2t1)
        b[0]    += np.sum(xt * ezt)
        b[1]    += np.sum(ezzt)

    alpha_new, lam_new = np.linalg.solve(A, b)

    # ---- Update sigma_w ----
    # Q = (1 / N*T) * sum_n sum_{t=0}^{T-1} E[(z_t - alpha*x_t - lam*z_{t-1})^2]
    sigma_w_sq = 0.0
    count_w = 0
    for n in range(n_patients):
        xt    = X[n]
        ezt   = Ez[n]
        ez2t  = Ez2[n]
        ezt1  = np.concatenate(([0.0], Ez[n, :-1]))
        ez2t1 = np.concatenate(([0.0], Ez2[n, :-1]))
        ezzt  = Ezz[n]

        sigma_w_sq += (
            np.sum(ez2t)
            - 2 * alpha_new * np.sum(xt * ezt)
            - 2 * lam_new   * np.sum(ezzt)
            + alpha_new ** 2 * np.sum(xt ** 2)
            + 2 * alpha_new * lam_new * np.sum(xt * ezt1)
            + lam_new ** 2  * np.sum(ez2t1)
        )
        count_w += T

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
    alpha_init=0.5,
    lam_init=0.5,
    beta_init=0.5,
    gamma_init=0.5,
    sigma_w_init=0.2,
    sigma_e_init=None,
    max_iter=1000,
    tol=1e-8,
    verbose=True,
    return_info=False,
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
    sigma_w_init : float
        Initial state-noise standard deviation.
    sigma_e_init : float or None
        Initial observation-noise standard deviation. None (the default) starts
        from np.std(Y), which matters: a fixed small value such as 0.2 against a
        true sigma_e near 1.0 converges to a measurably worse optimum.
    max_iter : int
        Maximum number of EM iterations. Defaults to 1000.
    tol : float
        Convergence threshold on the log-likelihood change, **relative** to the
        current log-likelihood. An absolute threshold is meaningless here because
        |log-likelihood| grows with n_patients * T, so the old absolute 1e-4
        amounted to ~1e-8 relative and essentially never tripped.
    verbose : bool
        Print the convergence outcome.
    return_info : bool
        Also return a diagnostics dict (see Returns). Off by default so the
        historical two-value unpacking keeps working.

    Returns
    -------
    params : dict
        Parameters at the **highest-likelihood iterate**, not the last one:
        alpha, lam, beta, gamma, sigma_w, sigma_e.
    log_likelihoods : list of float
        Log-likelihood after each E-step (useful for convergence diagnostics).
    info : dict, only when return_info=True
        converged, n_iter, best_iter, best_log_lik, final_log_lik, n_decreases.
    """
    if sigma_e_init is None:
        sigma_e_init = float(np.std(Y))

    alpha, lam, beta, gamma, sigma_w, sigma_e = (
        alpha_init, lam_init, beta_init, gamma_init, sigma_w_init, sigma_e_init
    )

    log_likelihoods = []
    best_log_lik = -np.inf
    best_params = None
    best_iter = 0
    converged = False
    n_iter = 0

    for i in range(max_iter):
        # E-step
        Ez, Ez2, Ezz, log_lik = _e_step(X, Y, alpha, lam, beta, gamma, sigma_w, sigma_e)
        log_likelihoods.append(log_lik)
        n_iter = i + 1

        # Keep the best iterate seen. The two failure modes pull in opposite
        # directions -- one group needed >2000 iterations to get near the optimum,
        # another peaked at iteration 24 -- so no single max_iter is right and the
        # final iterate is not necessarily the best one.
        if log_lik > best_log_lik:
            best_log_lik = log_lik
            best_iter = n_iter
            best_params = {
                "alpha": alpha, "lam": lam, "beta": beta,
                "gamma": gamma, "sigma_w": sigma_w, "sigma_e": sigma_e,
            }

        # Check convergence
        if i > 0:
            scale = max(1.0, abs(log_likelihoods[-2]))
            if abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol * scale:
                converged = True
                if verbose:
                    print(f"Converged at iteration {n_iter}")
                break

        # M-step
        alpha, lam, beta, gamma, sigma_w, sigma_e = _m_step(X, Y, Ez, Ez2, Ezz)

    n_decreases = int((np.diff(log_likelihoods) < 0).sum()) if len(log_likelihoods) > 1 else 0

    if not converged:
        warnings.warn(
            f"em_ssm did not converge in {n_iter} iterations "
            f"(last log-likelihood change "
            f"{abs(log_likelihoods[-1] - log_likelihoods[-2]) if n_iter > 1 else float('nan'):.3e}, "
            f"relative tol {tol:g}). Returning the best iterate (iteration "
            f"{best_iter}, log-likelihood {best_log_lik:.4f}).",
            RuntimeWarning,
            stacklevel=2,
        )
    if n_decreases and verbose:
        # Should be impossible for an exact EM; kept as a tripwire.
        print(f"  warning: log-likelihood decreased on {n_decreases}/"
              f"{len(log_likelihoods) - 1} EM steps")

    if return_info:
        info = {
            "converged": converged,
            "n_iter": n_iter,
            "best_iter": best_iter,
            "best_log_lik": best_log_lik,
            "final_log_lik": log_likelihoods[-1],
            "n_decreases": n_decreases,
        }
        return best_params, log_likelihoods, info

    return best_params, log_likelihoods

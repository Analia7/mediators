import numpy as np


# ---------------------------------------------------------------------------
# Kalman filter + RTS smoother for one patient, one group
# ---------------------------------------------------------------------------

def kalman_filter(Y, X, alpha, lam, beta, gamma, sigma_w, sigma_e,
                  z0=0.0, P0=1.0):
    """
    Forward Kalman filter for the scalar input-driven SSM:

        z_t = alpha*x_t + lam*z_{t-1} + w_t,   w ~ N(0, sigma_w^2)
        y_t = beta*z_t  + gamma*x_t   + e_t,   e ~ N(0, sigma_e^2)

    Returns
    -------
    z_filt  : (T,)  filtered means    z_{t|t}
    P_filt  : (T,)  filtered variances P_{t|t}
    z_pred  : (T,)  predicted means   z_{t|t-1}
    P_pred  : (T,)  predicted variances P_{t|t-1}
    K       : (T,)  Kalman gains
    log_lik : float total log-likelihood log p(Y | X, theta)
    """
    T        = len(Y)
    sigma_w2 = sigma_w ** 2
    sigma_e2 = sigma_e ** 2

    z_filt  = np.zeros(T);  P_filt = np.zeros(T)
    z_pred  = np.zeros(T);  P_pred = np.zeros(T)
    K       = np.zeros(T)
    log_lik = 0.0

    z_prev, P_prev = z0, P0

    for t in range(T):
        # predict
        zp = alpha * X[t] + lam * z_prev       # z_{t|t-1}
        Pp = lam**2 * P_prev + sigma_w2         # P_{t|t-1}

        # innovation and its variance
        innov = Y[t] - (beta * zp + gamma * X[t])
        S     = beta**2 * Pp + sigma_e2

        # log-likelihood contribution (innovations form, eq. 18 in paper)
        log_lik += -0.5 * (np.log(2 * np.pi * S) + innov**2 / S)

        # update
        k  = Pp * beta / S                      # Kalman gain
        zf = zp + k * innov                     # z_{t|t}
        Pf = (1 - k * beta) * Pp               # P_{t|t}

        z_pred[t] = zp;  P_pred[t] = Pp
        z_filt[t] = zf;  P_filt[t] = Pf
        K[t]      = k
        z_prev, P_prev = zf, Pf

    return z_filt, P_filt, z_pred, P_pred, K, log_lik


def rts_smoother(z_filt, P_filt, z_pred, P_pred, lam):
    """
    RTS backward smoother (equations A8-A12 in Shumway & Stoffer 1982).

    Returns
    -------
    z_smooth : (T,)  smoothed means     z_{t|T}
    P_smooth : (T,)  smoothed variances P_{t|T}
    P_cross  : (T,)  lag-1 cross-covariances P_{t,t-1|T}
                     P_cross[t] = Cov(z_t, z_{t-1} | Y_{1:T}), valid for t >= 1
    """
    T        = len(z_filt)
    z_smooth = np.zeros(T)
    P_smooth = np.zeros(T)
    P_cross  = np.zeros(T)
    J        = np.zeros(T - 1)

    # initialise at final time
    z_smooth[-1] = z_filt[-1]
    P_smooth[-1] = P_filt[-1]

    # backward sweep
    for t in range(T - 2, -1, -1):
        Jt         = P_filt[t] * lam / P_pred[t + 1]   # smoother gain (eq. A8)
        J[t]       = Jt
        z_smooth[t] = z_filt[t] + Jt * (z_smooth[t + 1] - z_pred[t + 1])    # A9
        P_smooth[t] = P_filt[t] + Jt**2 * (P_smooth[t + 1] - P_pred[t + 1]) # A10

    # lag-1 cross-covariances P_{t,t-1|T} = P_{t|T} * J_{t-1}  (eq. A11)
    for t in range(1, T):
        P_cross[t] = P_smooth[t] * J[t - 1]

    return z_smooth, P_smooth, P_cross


# ---------------------------------------------------------------------------
# Sufficient statistics for one patient
# ---------------------------------------------------------------------------

def sufficient_stats_one_patient(Y, X, z_smooth, P_smooth, P_cross):
    """
    Accumulate the sufficient statistics needed for the M-step for a single patient.

    The M-step solves two separate weighted least-squares problems:

      State eq:  z_t = alpha*x_t + lam*z_{t-1} + w_t    (t = 1..T)
      Obs   eq:  y_t = beta*z_t  + gamma*x_t   + e_t    (t = 0..T)

    Because z_t is latent, second moments enter via the smoother:
      E[z_t^2]        = P_{t|T} + z_{t|T}^2
      E[z_t * z_{t-1}] = P_{t,t-1|T} + z_{t|T} * z_{t-1|T}
    """
    Ez2    = P_smooth + z_smooth**2                             # E[z_t^2]
    Ezzt_1 = P_cross[1:] + z_smooth[1:] * z_smooth[:-1]        # E[z_t * z_{t-1}]

    s = {}

    # --- state equation (pairs t=1..T, indexed as curr/prev) ---
    s['sum_Ez2_prev']  = np.sum(Ez2[:-1])                      # Σ E[z_{t-1}^2]
    s['sum_Ez2_curr']  = np.sum(Ez2[1:])                       # Σ E[z_t^2]
    s['sum_Ezzt_1']    = np.sum(Ezzt_1)                        # Σ E[z_t z_{t-1}]
    s['sum_x2_curr']   = np.sum(X[1:]**2)                      # Σ x_t^2
    s['sum_xEz_curr']  = np.sum(X[1:] * z_smooth[1:])         # Σ x_t E[z_t]
    s['sum_xEz_prev']  = np.sum(X[1:] * z_smooth[:-1])        # Σ x_t E[z_{t-1}]
    s['T_state']       = len(Y) - 1

    # --- observation equation (all t=0..T) ---
    s['sum_y_Ez']      = np.sum(Y * z_smooth)                  # Σ y_t E[z_t]
    s['sum_y_x']       = np.sum(Y * X)                         # Σ y_t x_t
    s['sum_y2']        = np.sum(Y**2)                          # Σ y_t^2
    s['sum_Ez2_obs']   = np.sum(Ez2)                           # Σ E[z_t^2]
    s['sum_x2_obs']    = np.sum(X**2)                          # Σ x_t^2
    s['sum_xEz_obs']   = np.sum(X * z_smooth)                  # Σ x_t E[z_t]
    s['T_obs']         = len(Y)

    return s


def accumulate_stats(stats_list):
    """Sum sufficient statistics across all patients in a group."""
    total = {k: 0.0 for k in stats_list[0]}
    for s in stats_list:
        for k in total:
            total[k] += s[k]
    return total


# ---------------------------------------------------------------------------
# M-step: closed-form parameter updates
# ---------------------------------------------------------------------------

def m_step(S):
    """
    Given accumulated sufficient statistics S (summed over all patients
    in a group), update parameters via closed-form WLS.

    State equation normal equations for [alpha, lam]:

        [ Σx_t^2          Σ x_t E[z_{t-1}] ] [alpha]   [ Σ x_t E[z_t]       ]
        [ Σ x_t E[z_{t-1}]  Σ E[z_{t-1}^2] ] [lam  ] = [ Σ E[z_t z_{t-1}]   ]

    Observation equation normal equations for [beta, gamma]:

        [ Σ E[z_t^2]    Σ x_t E[z_t] ] [beta ]   [ Σ y_t E[z_t] ]
        [ Σ x_t E[z_t]  Σ x_t^2      ] [gamma] = [ Σ y_t x_t    ]

    The E[·] terms come from the Kalman smoother, which accounts for the
    uncertainty in z when computing regression targets (this is the key
    correction vs naive regression on z_smooth).
    """
    # --- state equation ---
    A_state = np.array([
        [S['sum_x2_curr'],   S['sum_xEz_prev']],
        [S['sum_xEz_prev'],  S['sum_Ez2_prev'] ]
    ])
    b_state = np.array([S['sum_xEz_curr'], S['sum_Ezzt_1']])
    alpha_new, lam_new = np.linalg.solve(A_state, b_state)

    resid_state = (
          S['sum_Ez2_curr']
        - 2 * alpha_new * S['sum_xEz_curr']
        - 2 * lam_new   * S['sum_Ezzt_1']
        + alpha_new**2  * S['sum_x2_curr']
        + 2 * alpha_new * lam_new * S['sum_xEz_prev']
        + lam_new**2    * S['sum_Ez2_prev']
    )
    sigma_w2_new = max(resid_state / S['T_state'], 1e-6)

    # --- observation equation ---
    A_obs = np.array([
        [S['sum_Ez2_obs'],  S['sum_xEz_obs']],
        [S['sum_xEz_obs'],  S['sum_x2_obs'] ]
    ])
    b_obs = np.array([S['sum_y_Ez'], S['sum_y_x']])
    beta_new, gamma_new = np.linalg.solve(A_obs, b_obs)

    resid_obs = (
          S['sum_y2']
        - 2 * beta_new  * S['sum_y_Ez']
        - 2 * gamma_new * S['sum_y_x']
        + beta_new**2   * S['sum_Ez2_obs']
        + 2 * beta_new  * gamma_new * S['sum_xEz_obs']
        + gamma_new**2  * S['sum_x2_obs']
    )
    sigma_e2_new = max(resid_obs / S['T_obs'], 1e-6)

    return dict(
        alpha=alpha_new, lam=lam_new, sigma_w=np.sqrt(sigma_w2_new),
        beta=beta_new,   gamma=gamma_new, sigma_e=np.sqrt(sigma_e2_new)
    )


# ---------------------------------------------------------------------------
# EM for one group
# ---------------------------------------------------------------------------

def em_one_group(Y_group, X_group, theta_init, max_iter=100, tol=1e-4,
                 verbose=True):
    """
    Run EM for a single group given labelled patients.

    One EM iteration =
      E-step: run Kalman filter + RTS smoother on ALL patients,
              accumulate sufficient statistics
      M-step: one closed-form parameter update from the accumulated stats

    Parameters
    ----------
    Y_group    : (N, T)  observations
    X_group    : (N, T)  inputs
    theta_init : dict    keys: alpha, lam, beta, gamma, sigma_w, sigma_e

    Returns
    -------
    theta    : dict   estimated parameters
    log_liks : list   total log-likelihood at each iteration
    """
    theta    = theta_init.copy()
    log_liks = []

    for iteration in range(max_iter):
        stats_list = []
        total_ll   = 0.0

        # E-step: filter + smooth every patient, collect sufficient stats
        for Y, X in zip(Y_group, X_group):
            z_filt, P_filt, z_pred, P_pred, K, ll = kalman_filter(
                Y, X, **theta
            )
            z_smooth, P_smooth, P_cross = rts_smoother(
                z_filt, P_filt, z_pred, P_pred, theta['lam']
            )
            stats_list.append(
                sufficient_stats_one_patient(Y, X, z_smooth, P_smooth, P_cross)
            )
            total_ll += ll

        log_liks.append(total_ll)

        # M-step: one update from stats summed over all patients
        S     = accumulate_stats(stats_list)
        theta = m_step(S)

        # convergence
        if iteration > 0 and abs(log_liks[-1] - log_liks[-2]) < tol:
            if verbose:
                print(f"  Converged at iteration {iteration + 1}")
            break

    return theta, log_liks


# ---------------------------------------------------------------------------
# Full training: fit both groups independently
# ---------------------------------------------------------------------------

def fit_mixture_ssm(j_train, X_train, Y_train, theta_init=None,
                    max_iter=100, tol=1e-4, verbose=True):
    """
    Fit group-specific SSM parameters given known group labels.

    Because group labels are observed, the mixture EM reduces to two
    independent SSM fits — one per group. No E-step over group assignments
    is needed; patients are hard-assigned to their true group.

    Parameters
    ----------
    j_train    : (N,)    group labels in {0, 1}
    X_train    : (N, T)  input signals
    Y_train    : (N, T)  observations
    theta_init : dict or None
        None  -> initialise from data moments (lag-1 autocorrelation, stdev)
        dict  -> keys 0 and 1, each a parameter dict

    Returns
    -------
    thetas   : {0: theta_0, 1: theta_1}
    log_liks : {0: [...],   1: [...]}
    """
    results_theta   = {}
    results_logliks = {}

    for group in [0, 1]:
        mask = (j_train == group)
        Y_g  = Y_train[mask]
        X_g  = X_train[mask]
        N_g  = Y_g.shape[0]

        if verbose:
            print(f"\nFitting group {group}  ({N_g} patients)")

        if theta_init is None:
            # data-driven initialisation
            lag1 = float(np.mean([
                np.corrcoef(Y_g[n, :-1], Y_g[n, 1:])[0, 1]
                for n in range(N_g)
            ]))
            init = dict(
                alpha=0.1,
                lam=np.clip(lag1, 0.1, 0.95),
                beta=0.5,
                gamma=0.1,
                sigma_w=np.std(Y_g) * 0.3,
                sigma_e=np.std(Y_g) * 0.3,
            )
        else:
            init = theta_init[group]

        theta, lls = em_one_group(Y_g, X_g, init,
                                   max_iter=max_iter, tol=tol,
                                   verbose=verbose)

        results_theta[group]   = theta
        results_logliks[group] = lls

        if verbose:
            print(f"  Final log-lik : {lls[-1]:.2f}")
            print(f"  alpha={theta['alpha']:.3f}  lam={theta['lam']:.3f}  "
                  f"beta={theta['beta']:.3f}  gamma={theta['gamma']:.3f}  "
                  f"sigma_w={theta['sigma_w']:.3f}  sigma_e={theta['sigma_e']:.3f}")

    return results_theta, results_logliks


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def generate_synthetic_samples(
        n_patients=100, T=50,
        alpha_0=0.6, lambda_0=0.8, beta_0=0.5, gamma_0=0.2,
        sigma_w_0=0.1, sigma_e_0=0.1,
        alpha_1=0.3, lambda_1=0.4, beta_1=0.9, gamma_1=0.5,
        sigma_w_1=0.1, sigma_e_1=0.1,
    ):
        params = {
            0: dict(alpha=alpha_0, lam=lambda_0, beta=beta_0,
                    gamma=gamma_0, sigma_w=sigma_w_0, sigma_e=sigma_e_0),
            1: dict(alpha=alpha_1, lam=lambda_1, beta=beta_1,
                    gamma=gamma_1, sigma_w=sigma_w_1, sigma_e=sigma_e_1),
        }
        j = np.random.choice([0, 1], size=n_patients)
        X = np.zeros((n_patients, T))
        Z = np.zeros((n_patients, T))
        Y = np.zeros((n_patients, T))
        for n in range(n_patients):
            p = params[j[n]]
            X[n] = np.cumsum(np.random.randn(T)) * 0.5
            z_prev = 0.0
            for t in range(T):
                w   = np.random.normal(0, p['sigma_w'])
                e   = np.random.normal(0, p['sigma_e'])
                z_t = p['alpha'] * X[n, t] + p['lam'] * z_prev + w
                y_t = p['beta']  * z_t     + p['gamma'] * X[n, t] + e
                Z[n, t] = z_t;  Y[n, t] = y_t;  z_prev = z_t
        return j, X, Z, Y

    np.random.seed(42)
    j_train, X_train, Z_train, Y_train = generate_synthetic_samples(
        n_patients=200, T=50
    )

    TRUE = {
        0: dict(alpha=0.6, lam=0.8, beta=0.5, gamma=0.2, sigma_w=0.1, sigma_e=0.1),
        1: dict(alpha=0.3, lam=0.4, beta=0.9, gamma=0.5, sigma_w=0.1, sigma_e=0.1),
    }

    print("True parameters:")
    for g in [0, 1]:
        p = TRUE[g]
        print(f"  Group {g}: alpha={p['alpha']}  lam={p['lam']}  "
              f"beta={p['beta']}  gamma={p['gamma']}  "
              f"sigma_w={p['sigma_w']}  sigma_e={p['sigma_e']}")

    # Run with default init
    thetas, log_liks = fit_mixture_ssm(
        j_train, X_train, Y_train, max_iter=200, tol=1e-5
    )

    print("\nRecovery summary:")
    for g in [0, 1]:
        print(f"\n  Group {g}:")
        for k in ['alpha', 'lam', 'beta', 'gamma', 'sigma_w', 'sigma_e']:
            print(f"    {k:8s}  true={TRUE[g][k]:.3f}  est={thetas[g][k]:.3f}")

    print()
    print("NOTE — identifiability of alpha and beta:")
    print("  With random-walk inputs and small sigma_w, z_t is nearly a")
    print("  deterministic filtered version of x_t (corr ~ 0.96).")
    print("  The Kalman smoother cannot separate 'how much of y came via z'")
    print("  vs 'directly from x', so alpha and beta are only identified")
    print("  through their product alpha*beta.")
    print("  lam, gamma, sigma_e recover well because lam governs time dynamics")
    print("  (independent of x) and gamma is the direct x->y coefficient.")
    for g in [0, 1]:
        t_ab = TRUE[g]['alpha'] * TRUE[g]['beta']
        e_ab = thetas[g]['alpha'] * thetas[g]['beta']
        print(f"  Group {g}: true alpha*beta={t_ab:.3f}  est alpha*beta={e_ab:.3f}")

 
# ---------------------------------------------------------------------------
# Phase 2: inference for a new patient
# ---------------------------------------------------------------------------
 
def infer_patient(Y, X, theta_0, theta_1):
    """
    Given a new patient's observations Y and inputs X, infer:
      - group assignment j* (hard, no prior)
      - filtered trajectory z_{t|t} and uncertainty P_{t|t}
      - smoothed trajectory z_{t|T} and uncertainty P_{t|T}
 
    All outputs are under the winning group's parameters.
 
    Parameters
    ----------
    Y, X     : (T,)  observations and inputs for one patient
    theta_0  : dict  estimated parameters for group 0
    theta_1  : dict  estimated parameters for group 1
 
    Returns
    -------
    result : dict with keys
        'group'     : int    hard group assignment (0 or 1)
        'log_liks'  : dict   {'0': ll_0, '1': ll_1}  (useful for confidence)
        'z_filt'    : (T,)   filtered means   z_{t|t}
        'P_filt'    : (T,)   filtered variances P_{t|t}
        'z_smooth'  : (T,)   smoothed means   z_{t|T}
        'P_smooth'  : (T,)   smoothed variances P_{t|T}
    """
    # --- Step 1: run Kalman filter under both groups ---
    out_0 = kalman_filter(Y, X, **theta_0)
    out_1 = kalman_filter(Y, X, **theta_1)
 
    ll_0 = out_0[-1]   # log p(Y | X, theta_0)
    ll_1 = out_1[-1]   # log p(Y | X, theta_1)
 
    # --- Step 2: hard group assignment by log-likelihood ---
    group = 0 if ll_0 >= ll_1 else 1
 
    # --- Step 3: smoother under winning group ---
    if group == 0:
        z_filt, P_filt, z_pred, P_pred, K, _ = out_0
        theta_win = theta_0
    else:
        z_filt, P_filt, z_pred, P_pred, K, _ = out_1
        theta_win = theta_1
 
    z_smooth, P_smooth, _ = rts_smoother(
        z_filt, P_filt, z_pred, P_pred, theta_win['lam']
    )
 
    return dict(
        group    = group,
        log_liks = {0: ll_0, 1: ll_1},
        z_filt   = z_filt,
        P_filt   = P_filt,
        z_smooth = z_smooth,
        P_smooth = P_smooth,
    )
 
 
def infer_patients(Y_test, X_test, theta_0, theta_1):
    """
    Run infer_patient over a batch of N patients.
 
    Parameters
    ----------
    Y_test   : (N, T)
    X_test   : (N, T)
    theta_0  : dict
    theta_1  : dict
 
    Returns
    -------
    results  : list of N dicts (one per patient, same keys as infer_patient)
    """
    return [
        infer_patient(Y_test[n], X_test[n], theta_0, theta_1)
        for n in range(len(Y_test))
    ]
 
 
# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
 
    def generate_synthetic_samples(
        n_patients=100, T=50,
        alpha_0=0.6, lambda_0=0.8, beta_0=0.5, gamma_0=0.2,
        sigma_w_0=0.1, sigma_e_0=0.1,
        alpha_1=0.3, lambda_1=0.4, beta_1=0.9, gamma_1=0.5,
        sigma_w_1=0.1, sigma_e_1=0.1,
    ):
        params = {
            0: dict(alpha=alpha_0, lam=lambda_0, beta=beta_0,
                    gamma=gamma_0, sigma_w=sigma_w_0, sigma_e=sigma_e_0),
            1: dict(alpha=alpha_1, lam=lambda_1, beta=beta_1,
                    gamma=gamma_1, sigma_w=sigma_w_1, sigma_e=sigma_e_1),
        }
        j = np.random.choice([0, 1], size=n_patients)
        X = np.zeros((n_patients, T))
        Z = np.zeros((n_patients, T))
        Y = np.zeros((n_patients, T))
        for n in range(n_patients):
            p = params[j[n]]
            X[n] = np.cumsum(np.random.randn(T)) * 0.5
            z_prev = 0.0
            for t in range(T):
                w   = np.random.normal(0, p['sigma_w'])
                e   = np.random.normal(0, p['sigma_e'])
                z_t = p['alpha'] * X[n, t] + p['lam'] * z_prev + w
                y_t = p['beta']  * z_t     + p['gamma'] * X[n, t] + e
                Z[n, t] = z_t;  Y[n, t] = y_t;  z_prev = z_t
        return j, X, Z, Y
 
    np.random.seed(42)
    j_train, X_train, Z_train, Y_train = generate_synthetic_samples(
        n_patients=200, T=50
    )
 
    TRUE = {
        0: dict(alpha=0.6, lam=0.8, beta=0.5, gamma=0.2, sigma_w=0.1, sigma_e=0.1),
        1: dict(alpha=0.3, lam=0.4, beta=0.9, gamma=0.5, sigma_w=0.1, sigma_e=0.1),
    }
 
    print("True parameters:")
    for g in [0, 1]:
        p = TRUE[g]
        print(f"  Group {g}: alpha={p['alpha']}  lam={p['lam']}  "
              f"beta={p['beta']}  gamma={p['gamma']}  "
              f"sigma_w={p['sigma_w']}  sigma_e={p['sigma_e']}")
 
    # Run with default init
    thetas, log_liks = fit_mixture_ssm(
        j_train, X_train, Y_train, max_iter=200, tol=1e-5
    )
 
    print("\nRecovery summary:")
    for g in [0, 1]:
        print(f"\n  Group {g}:")
        for k in ['alpha', 'lam', 'beta', 'gamma', 'sigma_w', 'sigma_e']:
            print(f"    {k:8s}  true={TRUE[g][k]:.3f}  est={thetas[g][k]:.3f}")
 
    # -----------------------------------------------------------------------
    # Phase 2: infer group labels and z trajectories on held-out patients
    # -----------------------------------------------------------------------
 
    np.random.seed(99)
    j_test, X_test, Z_test, Y_test = generate_synthetic_samples(
        n_patients=50, T=50
    )
 
    results = infer_patients(Y_test, X_test, thetas[0], thetas[1])
 
    # group assignment accuracy
    j_pred = np.array([r['group'] for r in results])
    acc    = np.mean(j_pred == j_test)
    print(f"\nPhase 2 — group assignment accuracy: {acc:.2%}  ({int(acc*50)}/50)")
 
    # z trajectory recovery: MAE vs true Z, separated by group
    for g in [0, 1]:
        idx = np.where(j_test == g)[0]
        if len(idx) == 0:
            continue
        mae_filt   = np.mean([np.mean(np.abs(results[n]['z_filt']   - Z_test[n])) for n in idx])
        mae_smooth = np.mean([np.mean(np.abs(results[n]['z_smooth'] - Z_test[n])) for n in idx])
        print(f"  Group {g} ({len(idx)} patients):"
              f"  MAE filtered={mae_filt:.4f}  MAE smoothed={mae_smooth:.4f}")
 
    # detailed view of one example patient
    n = np.where(j_test == 0)[0][0]
    r = results[n]
    T = Y_test.shape[1]
    print(f"\nExample patient (true group={j_test[n]}, assigned={r['group']}):")
    print(f"  log-lik group 0: {r['log_liks'][0]:.2f}")
    print(f"  log-lik group 1: {r['log_liks'][1]:.2f}")
    print(f"  {'t':>4}  {'z_true':>8}  {'z_filt':>8}  {'z_smooth':>9}"
          f"  {'±2σ_filt':>9}  {'±2σ_smooth':>11}")
    for t in range(0, T, 5):
        print(f"  {t:>4}  {Z_test[n,t]:>8.3f}  {r['z_filt'][t]:>8.3f}"
              f"  {r['z_smooth'][t]:>9.3f}"
              f"  {2*r['P_filt'][t]**0.5:>9.3f}  {2*r['P_smooth'][t]**0.5:>11.3f}")
 
    print()
    print("NOTE — identifiability of alpha and beta:")
    print("  With random-walk inputs and small sigma_w, z_t is nearly a")
    print("  deterministic filtered version of x_t (corr ~ 0.96).")
    print("  The Kalman smoother cannot separate 'how much of y came via z'")
    print("  vs 'directly from x', so alpha and beta are only identified")
    print("  through their product alpha*beta.")
    print("  lam, gamma, sigma_e recover well because lam governs time dynamics")
    print("  (independent of x) and gamma is the direct x->y coefficient.")
    for g in [0, 1]:
        t_ab = TRUE[g]['alpha'] * TRUE[g]['beta']
        e_ab = thetas[g]['alpha'] * thetas[g]['beta']
        print(f"  Group {g}: true alpha*beta={t_ab:.3f}  est alpha*beta={e_ab:.3f}")
# generate synthetic samples
import numpy as np

def generate_synthetic_samples(
    n_patients=100,
    T=50,
    # Group 0 SSM parameters
    alpha_0=0.6, lambda_0=0.8, beta_0=0.5, gamma_0=0.2,
    sigma_w_0=0.1, sigma_e_0=0.1,
    # Group 1 SSM parameters
    alpha_1=0.3, lambda_1=0.4, beta_1=0.9, gamma_1=0.5,
    sigma_w_1=0.1, sigma_e_1=0.1,
):
    """
    Generates synthetic patient data from two group-specific SSMs:

      State:       z_t = alpha_j * x_t + lambda_j * z_{t-1} + w_t,   w ~ N(0, sigma_w)
      Observation: y_t = beta_j  * z_t  + gamma_j  * x_t   + e_t,   e ~ N(0, sigma_e)

    Returns:
      j  : (n_patients,)       group labels in {0, 1}
      X  : (n_patients, T)     input signals x_t
      Z  : (n_patients, T)     latent states z_t
      Y  : (n_patients, T)     observations y_t
    """
    # Pack parameters per group for clean indexing
    params = {
        0: dict(alpha=alpha_0, lam=lambda_0, beta=beta_0,
                gamma=gamma_0, sigma_w=sigma_w_0, sigma_e=sigma_e_0),
        1: dict(alpha=alpha_1, lam=lambda_1, beta=beta_1,
                gamma=gamma_1, sigma_w=sigma_w_1, sigma_e=sigma_e_1),
    }

    # Assign group labels randomly
    j = np.random.choice([0, 1], size=n_patients)

    # Pre-allocate outputs
    X = np.zeros((n_patients, T))
    Z = np.zeros((n_patients, T))
    Y = np.zeros((n_patients, T))

    for n in range(n_patients):
        p = params[j[n]]

        # Input signal x_t (was random walk, swap for your real signal if available)
        # currently, it is just random
        X[n] = np.random.randn(T) * 0.5

        z_prev = 0.0
        for t in range(T):
            w = np.random.normal(0, p['sigma_w'])
            e = np.random.normal(0, p['sigma_e'])

            z_t = p['alpha'] * X[n, t] + p['lam'] * z_prev + w
            y_t = p['beta']  * z_t     + p['gamma'] * X[n, t] + e

            Z[n, t] = z_t
            Y[n, t] = y_t
            z_prev  = z_t

    return j, X, Z, Y
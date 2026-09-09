# generate synthetic samples
import numpy as np

# For phase 1: estimating the SSM parameters
def generate_synthetic_samples(
    n_patients=100,
    T=50,
    # Group 0 SSM parameters
    alpha_0=0.6, lambda_0=0.8, beta_0=0.5, gamma_0=0.2,
    sigma_w_0=0.1, sigma_e_0=0.9,
    # Group 1 SSM parameters
    alpha_1=0.3, lambda_1=0.4, beta_1=0.9, gamma_1=0.5,
    sigma_w_1=0.1, sigma_e_1=0.9,
    seed=None,
):
    """
    Generates synthetic patient data from two group-specific SSMs:

      State:       z_t = alpha_j * x_t + lambda_j * z_{t-1} + w_t,   w ~ N(0, sigma_w)
      Observation: y_t = beta_j  * z_t  + gamma_j  * x_t   + e_t,   e ~ N(0, sigma_e)

    seed : int, sequence of int, or None
      Pass a seed for an isolated, independently reproducible stream. None keeps
      the historical behaviour of drawing from the global np.random state, so
      np.random.seed(...) still works -- but that state is shared with every
      other consumer, so prefer an explicit seed.

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

    rng = np.random if seed is None else np.random.default_rng(seed)

    # Assign group labels randomly
    j = rng.choice([0, 1], size=n_patients)

    # Pre-allocate outputs
    X = np.zeros((n_patients, T))
    Z = np.zeros((n_patients, T))
    Y = np.zeros((n_patients, T))

    for n in range(n_patients):
        p = params[j[n]]

        # Input signal x_t (was random walk, swap for your real signal if available)
        # currently, it is just random
        X[n] = rng.standard_normal(T) * 0.5

        z_prev = 0.0
        for t in range(T):
            w = rng.normal(0, p['sigma_w'])
            e = rng.normal(0, p['sigma_e'])

            z_t = p['alpha'] * X[n, t] + p['lam'] * z_prev + w
            y_t = p['beta']  * z_t     + p['gamma'] * X[n, t] + e

            Z[n, t] = z_t
            Y[n, t] = y_t
            z_prev  = z_t

    return j, X, Z, Y

DEFAULT_PARAMS = {
    0: dict(alpha=0.6, lam=0.8, beta=0.5, gamma=0.2, sigma_w=0.1, sigma_e=0.9),
    1: dict(alpha=0.3, lam=0.4, beta=0.9, gamma=0.5, sigma_w=0.1, sigma_e=0.9),
}
 
# For phase 2 (active learning): simulating one patient at a time, one time step at a time
class PatientSimulator:
    """
    A stateful simulator for one patient that advances one time step per call.
 
    Typical active-learning loop:
 
        sim = PatientSimulator(group=1, seed=0)
        for t in range(T):
            x_t = my_policy(sim.history())   # YOU choose x_t
            y_t = sim.step(x_t)              # observe the result
    """
 
    def __init__(self, group=None, params=None, z0=0.0, seed=None):
        """
        group  : 0 or 1. If None, drawn uniformly at random (like the batch code).
        params : optional override of {group: {...}}; defaults to DEFAULT_PARAMS.
        z0     : initial latent state z_{-1}.
        seed   : per-patient RNG seed for reproducible noise / group draw.
        """
        self.rng = np.random.default_rng(seed)
 
        if group is None:
            group = int(self.rng.integers(0, 2))
        self.group = int(group)
 
        self.params = (params or DEFAULT_PARAMS)[self.group]
        self.z0 = float(z0)
        self.reset()
 
    def reset(self):
        """Restart the trajectory (keeps the same group, params, and RNG state)."""
        self.t = 0
        self.z_prev = self.z0
        self._X, self._Z, self._Y = [], [], []
        return self
 
    def step(self, x_t):
        """
        Advance one time step using the x_t you provide.
 
        Returns y_t, the (noisy) observation. The latent state z_t is hidden in a
        real active-learning setting, but it is recorded and accessible via
        .history() / .Z for evaluation.
        """
        p = self.params
 
        w = self.rng.normal(0.0, p['sigma_w'])
        e = self.rng.normal(0.0, p['sigma_e'])
 
        z_t = p['alpha'] * x_t + p['lam'] * self.z_prev + w
        y_t = p['beta'] * z_t + p['gamma'] * x_t + e
 
        self._X.append(float(x_t))
        self._Z.append(z_t)
        self._Y.append(y_t)
 
        self.z_prev = z_t
        self.t += 1
 
        return y_t
 
    # ---- accessors ---------------------------------------------------------
    @property
    def X(self):
        return np.array(self._X)
 
    @property
    def Z(self):
        return np.array(self._Z)
 
    @property
    def Y(self):
        return np.array(self._Y)
 
    def history(self):
        """Return (X, Z, Y) arrays of everything observed so far."""
        return self.X, self.Z, self.Y
import matplotlib.pyplot as plt
import numpy as np

def plot_ssm_trajectories(X_train, Z_train, Y_train, j_train, groups=[0, 1],
                          estimated_params=None):
    """
    Plot SSM trajectories, optionally overlaying predictions from estimated parameters.

    Parameters
    ----------
    X_train, Z_train, Y_train : (n_patients, T) arrays
    j_train                   : (n_patients,) group labels
    groups                    : list of group indices to plot
    estimated_params          : dict of {group: {alpha, lam, beta, gamma, sigma_w, sigma_e}}
                                If provided, predicted trajectories are overlaid.
    """

    GROUP_COLORS = {0: '#4C9BE8', 1: '#E8844C'}
    DARK_FACTOR  = 0.5

    def darken(hex_color, factor=DARK_FACTOR):
        hex_color = hex_color.lstrip('#')
        r, g, b = [int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)]
        return (r * factor, g * factor, b * factor)

    def compute_z_est(X, p):
        """Compute estimated latent state trajectory from parameters (no noise = mean trajectory)."""
        T = X.shape[0]
        Z_est = np.zeros(T)
        z_prev = 0.0
        for t in range(T):
            z_t = p['alpha'] * X[t] + p['lam'] * z_prev
            Z_est[t] = z_t
            z_prev   = z_t
        return Z_est

    def compute_y_pred(X, Z_est, p):
        """Compute predicted observations using estimated Z and parameters."""
        return p['beta'] * Z_est + p['gamma'] * X

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for group in groups:
        idx   = np.where(j_train == group)[0][0]
        color = GROUP_COLORS.get(group, f'C{group}')

        # Ground truth
        axes[0].plot(X_train[idx], color=color, label=f'Group {group}')
        axes[1].plot(Z_train[idx], color=color, label=f'Group {group} (true)')
        axes[2].plot(Y_train[idx], color=color, label=f'Group {group} (true)')

        # Predictions
        if estimated_params is not None:
            p      = estimated_params[group]
            dark   = darken(color)
            Z_est  = compute_z_est(X_train[idx], p)
            Y_pred = compute_y_pred(X_train[idx], Z_est, p)

            axes[1].plot(Z_est,  color=dark, linestyle='--', label=f'Group {group} (est)')
            axes[2].plot(Y_pred, color=dark, linestyle='--', label=f'Group {group} (pred)')

    axes[0].set_ylabel('Input X');       axes[0].legend()
    axes[1].set_ylabel('Latent Z');      axes[1].legend()
    axes[2].set_ylabel('Observation Y'); axes[2].legend()
    axes[2].set_xlabel('Time')
    plt.suptitle('SSM Trajectories by Group')
    plt.tight_layout()
    plt.show()
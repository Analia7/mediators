import matplotlib.pyplot as plt
import numpy as np
 
 
def plot_scatter_by_group(
    X,
    Y,
    groups,
    group_labels=None,
    xlabel="Input Signal X",
    ylabel="Observation Y",
    title="Synthetic Patient Data Colored by Group",
    figsize=(12, 6),
    alpha=0.5,
    output_path=False,
):
    """
    Scatter plot of (X, Y) data points colored by group membership.
    Note that temporality is not represented in this plot, so it is not suitable for visualizing time series dynamics.
    Instead, it is meant to show how the two groups differ in their overall distribution of X
 
    Parameters
    ----------
    X : array-like, shape (n_patients, T)
        Input signal values.
    Y : array-like, shape (n_patients, T)
        Observation values.
    groups : array-like, shape (n_patients,)
        Group label per patient.
    group_labels : dict, optional
        Mapping from group value to display label.
        e.g. {0: "Control", 1: "Treatment"}
        Defaults to "Group {value}" for each unique group.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    figsize : tuple
        Figure size as (width, height).
    alpha : float
        Marker transparency (0–1).
    output_path : str or False
        If a file path is provided (e.g. "plots/my_chart.png"), the figure is
        saved there instead of displayed. Supports any format matplotlib accepts
        (.png, .pdf, .svg, etc.). Defaults to False (display only).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    groups = np.asarray(groups)
 
    unique_groups = sorted(set(groups))
 
    fig, ax = plt.subplots(figsize=figsize)
 
    for group in unique_groups:
        label = (
            group_labels[group]
            if group_labels and group in group_labels
            else f"Group {group}"
        )
        idx = groups == group
        ax.scatter(X[idx].flatten(), Y[idx].flatten(), alpha=alpha, label=label)
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
 
    if output_path:
        fig.savefig(output_path)
    else:
        plt.show()
 
    plt.close(fig)


def _compute_estimated_trajectories(X_patient, params):
    """
    Compute noiseless estimated Z and Y for a single patient using SSM equations:
      z_t = alpha * x_t + lam * z_{t-1}
      y_t = beta  * z_t + gamma * x_t
    """
    T = len(X_patient)
    Z_est = np.zeros(T)
    Y_est = np.zeros(T)
 
    z_prev = 0.0
    for t in range(T):
        z_t = params["alpha"] * X_patient[t] + params["lam"] * z_prev
        y_t = params["beta"] * z_t + params["gamma"] * X_patient[t]
        Z_est[t] = z_t
        Y_est[t] = y_t
        z_prev = z_t
 
    return Z_est, Y_est
 
 
def plot_trajectories(
    X,
    Z,
    Y,
    groups,
    sample_indices=None,
    group_labels=None,
    estimated_params=None,
    xlabel="Time",
    title="SSM Trajectories by Group",
    figsize=(12, 8),
    output_path=False,
):
    """
    Plot SSM trajectories (X, Z, Y over time) for one sample per group.
    Optionally overlays estimated Z and Y computed from estimated_params.
 
    Parameters
    ----------
    X : array-like, shape (n_patients, T)
        Input signal values.
    Z : array-like, shape (n_patients, T)
        Latent state values (ground truth).
    Y : array-like, shape (n_patients, T)
        Observation values (ground truth).
    groups : array-like, shape (n_patients,)
        Group label per patient.
    sample_indices : dict, optional
        Mapping from group value to the patient index to plot.
        e.g. {0: 5, 1: 12} plots patient 5 for group 0, patient 12 for group 1.
        Defaults to the first patient in each group.
    group_labels : dict, optional
        Mapping from group value to display label.
        e.g. {0: "Control", 1: "Treatment"}
        Defaults to "Group {value}" for each unique group.
    estimated_params : dict, optional
        Per-group parameter estimates used to compute predicted Z and Y.
        e.g. {
            0: {"alpha": 0.6, "lam": 0.8, "beta": 0.5, "gamma": 0.2},
            1: {"alpha": 0.3, "lam": 0.4, "beta": 0.9, "gamma": 0.5},
        }
        Predicted trajectories are plotted as darker dashed lines alongside
        the ground truth. Defaults to None (no predictions shown).
    xlabel : str
        X-axis label (shown on the bottom subplot).
    title : str
        Figure title.
    figsize : tuple
        Figure size as (width, height).
    output_path : str or False
        If a file path is provided (e.g. "plots/trajectories.png"), the figure
        is saved there instead of displayed. Supports any format matplotlib
        accepts (.png, .pdf, .svg, etc.). Defaults to False (display only).
    """
    X = np.asarray(X)
    Z = np.asarray(Z)
    Y = np.asarray(Y)
    groups = np.asarray(groups)
 
    unique_groups = sorted(set(groups))
 
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
 
    for group in unique_groups:
        label = (
            group_labels[group]
            if group_labels and group in group_labels
            else f"Group {group}"
        )
 
        # Pick the sample to plot: user-specified or first in the group
        group_indices = np.where(groups == group)[0]
        idx = sample_indices[group] if sample_indices and group in sample_indices else group_indices[0]
 
        # Ground truth — lighter, solid
        color = f"C{unique_groups.index(group)}"
        axes[0].plot(X[idx], label=label, color=color, alpha=0.6)
        axes[1].plot(Z[idx], label=f"{label} (true)", color=color, alpha=0.6)
        axes[2].plot(Y[idx], label=f"{label} (true)", color=color, alpha=0.6)
 
        # Estimated trajectories — darker, dashed
        if estimated_params and group in estimated_params:
            Z_est, Y_est = _compute_estimated_trajectories(X[idx], estimated_params[group])
            dark_color = f"C{unique_groups.index(group)}"
            axes[1].plot(Z_est, label=f"{label} (est)", color=dark_color, linestyle="--", alpha=1.0)
            axes[2].plot(Y_est, label=f"{label} (est)", color=dark_color, linestyle="--", alpha=1.0)
 
    axes[0].set_ylabel("Input X")
    axes[0].legend()
    axes[1].set_ylabel("Latent Z")
    axes[1].legend()
    axes[2].set_ylabel("Observation Y")
    axes[2].set_xlabel(xlabel)
    axes[2].legend()
 
    plt.suptitle(title)
    plt.tight_layout()
 
    if output_path:
        fig.savefig(output_path)
    else:
        plt.show()
 
    plt.close(fig)


def plot_mean_std(
    Y,
    groups,
    group_labels=None,
    xlabel="Time",
    ylabel="Y",
    title="Mean ± 1 SD of Observations by Group",
    figsize=(12, 4),
    std_multiplier=1,
    output_path=False,
):
    """
    Plot mean ± std of Y over time, shaded by group.
 
    Parameters
    ----------
    Y : array-like, shape (n_patients, T)
        Observation values.
    groups : array-like, shape (n_patients,)
        Group label per patient.
    group_labels : dict, optional
        Mapping from group value to display label.
        e.g. {0: "Control", 1: "Treatment"}
        Defaults to "Group {value}" for each unique group.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    figsize : tuple
        Figure size as (width, height).
    std_multiplier : float
        Number of standard deviations for the shaded band. Defaults to 1.
    output_path : str or False
        If a file path is provided (e.g. "plots/mean_std.png"), the figure is
        saved there instead of displayed. Supports any format matplotlib accepts
        (.png, .pdf, .svg, etc.). Defaults to False (display only).
    """
    Y = np.asarray(Y)
    groups = np.asarray(groups)
 
    unique_groups = sorted(set(groups))
    T = Y.shape[1]
    t = np.arange(T)
 
    fig, ax = plt.subplots(figsize=figsize)
 
    for group in unique_groups:
        label = (
            group_labels[group]
            if group_labels and group in group_labels
            else f"Group {group}"
        )
        idx = groups == group
        mean_y = Y[idx].mean(axis=0)
        std_y = Y[idx].std(axis=0)
 
        ax.plot(t, mean_y, label=label)
        ax.fill_between(t, mean_y - std_multiplier * std_y, mean_y + std_multiplier * std_y, alpha=0.2)
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
 
    if output_path:
        fig.savefig(output_path)
    else:
        plt.show()
 
    plt.close(fig)

def plot_accuracy_active_learning(curves, sems=None, save_path=None):
    """
    curves : dict like {"active": acc_active, "random": acc_random}
    sems   : optional dict of same shape, for +/- shaded bands
    """
    T = len(next(iter(curves.values())))     # length of any curve
    steps = np.arange(1, T + 1)

    plt.figure(figsize=(8, 5))
    for label, acc in curves.items():
        line, = plt.plot(steps, acc, label=label, lw=2)
        if sems is not None and label in sems:
            plt.fill_between(steps, acc - sems[label], acc + sems[label],
                             color=line.get_color(), alpha=0.2)
    plt.axhline(0.5, ls="--", color="gray")
    plt.xlabel("time step")
    plt.ylabel("fraction correct")
    plt.title(f"{T} steps")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

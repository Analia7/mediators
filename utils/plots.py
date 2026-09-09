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

# Categorical slots 1-3 of the reference palette, in fixed order (never cycled).
# These three validate on the all-pairs CVD / normal-vision gates in both modes.
_SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua

# Text and chrome tokens: marks carry the series colour, text never does.
_INK_PRIMARY   = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED     = "#8a8984"
_GRID          = "#e8e7e4"
_AXIS          = "#c9c8c4"


def plot_accuracy_active_learning(curves, sems=None, save_path=None,
                                  title=None, subtitle=None, chance=0.5):
    """
    curves   : dict like {"random": acc_random, "uncertainty sampling": acc_us}
               Iteration order fixes the colour assignment, so a policy keeps its
               colour even if another is dropped from the comparison.
    sems     : optional dict of same shape, for +/- 1 SEM shaded bands
    save_path: write a PNG here instead of showing the figure
    title    : headline; defaults to a plain description
    subtitle : one line of run configuration (n patients, noise regime, ...)
    chance   : y value of the reference line, or None to omit it
    """
    T = len(next(iter(curves.values())))     # length of any curve
    steps = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Recessive horizontal grid: solid hairlines one step off the surface, behind
    # the data. Never dashed - dashing reads as "threshold" when it is just a grid.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRID, lw=0.8, ls="-")
    ax.xaxis.grid(False)

    # The chance line *is* a threshold, so here the dashes are meaningful.
    if chance is not None:
        ax.axhline(chance, ls="--", lw=1, color=_INK_MUTED, zorder=1)
        ax.annotate(f"chance ({chance:g})", xy=(1, chance), xytext=(9, 6),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=9, color=_INK_MUTED)

    for i, (label, acc) in enumerate(curves.items()):
        colour = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        acc = np.asarray(acc)
        if sems is not None and label in sems:
            sem = np.asarray(sems[label])
            # A wash, not a saturated block, so overlapping bands stay readable.
            ax.fill_between(steps, acc - sem, acc + sem,
                            color=colour, alpha=0.12, lw=0, zorder=2)
        ax.plot(steps, acc, color=colour, lw=2, zorder=3,
                solid_capstyle="round", solid_joinstyle="round", label=label)

    ax.set_xlim(1, T)
    lo = min(float(np.min(np.asarray(a))) for a in curves.values())
    ax.set_ylim(min(lo - 0.04, (chance or 1.0) - 0.06), 1.02)

    # Clean tick values; tabular figures because they align vertically.
    xticks = [1] + [t for t in range(5, T + 1, 5)]
    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.tick_params(colors=_INK_SECONDARY, labelsize=10, length=0)

    ax.set_xlabel("time step", fontsize=11, color=_INK_SECONDARY, labelpad=8)
    ax.set_ylabel("fraction correct", fontsize=11, color=_INK_SECONDARY, labelpad=8)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.8)

    if title is None:
        title = "Group identification accuracy over time"
    ax.set_title(title, fontsize=12, color=_INK_PRIMARY, loc="left",
                 pad=30 if subtitle else 12)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xytext=(0, 7), xycoords="axes fraction",
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=10, color=_INK_SECONDARY)

    # Legend always present for >=2 series; its text wears an ink token, with the
    # short colour key beside it carrying identity.
    # Centre-right: with the curves converging at the top and the chance rule at
    # the bottom, this band is the one part of the plot that is reliably empty.
    leg = ax.legend(loc="center right", fontsize=10, handlelength=1.6,
                    borderpad=0.9, labelspacing=0.6,
                    frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    for text in leg.get_texts():
        text.set_color(_INK_SECONDARY)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor="white")
    else:
        plt.show()

    plt.close(fig)

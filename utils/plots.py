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

def plot_trajectories(
    X,
    Z,
    Y,
    groups,
    sample_indices=None,
    group_labels=None,
    xlabel="Time",
    title="SSM Trajectories by Group",
    figsize=(12, 8),
    output_path=False,
):
    """
    Plot SSM trajectories (X, Z, Y over time) for one sample per group.
 
    Parameters
    ----------
    X : array-like, shape (n_patients, T)
        Input signal values.
    Z : array-like, shape (n_patients, T)
        Latent state values.
    Y : array-like, shape (n_patients, T)
        Observation values.
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
        if sample_indices and group in sample_indices:
            idx = sample_indices[group]
        else:
            idx = group_indices[0]
 
        axes[0].plot(X[idx], label=label)
        axes[1].plot(Z[idx], label=label)
        axes[2].plot(Y[idx], label=label)
 
    axes[0].set_ylabel("Input X")
    axes[0].legend()
    axes[1].set_ylabel("Latent Z")
    axes[2].set_ylabel("Observation Y")
    axes[2].set_xlabel(xlabel)
 
    plt.suptitle(title)
    plt.tight_layout()
 
    if output_path:
        fig.savefig(output_path)
    else:
        plt.show()
 
    plt.close(fig)
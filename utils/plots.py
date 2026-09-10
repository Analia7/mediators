import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.infer_patient import _innovations_log_likelihood


# Categorical slots 1-3 of the reference palette, in fixed order (never cycled).
# These three validate on the all-pairs CVD / normal-vision gates in both modes.
_SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua

# Text and chrome tokens: marks carry the series colour, text never does.
_INK_PRIMARY   = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED     = "#8a8984"
_GRID          = "#e8e7e4"
_AXIS          = "#c9c8c4"

# Fixed roles in every predicted-vs-actual figure, so the encoding reads the
# same in each panel: slot 1 is what happened, slot 2 is what the model says.
_ACTUAL_COLOUR = _SERIES_COLORS[0]
_MODEL_COLOUR  = _SERIES_COLORS[1]

 
 
def plot_scatter_by_group(
    X,
    Y,
    groups,
    group_labels=None,
    xlabel="Input signal X",
    ylabel="Observation Y",
    title=None,
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
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
 
    if output_path:
        fig.savefig(output_path)
    else:
        plt.show()
 
    plt.close(fig)


def _predict_trajectories(X_patient, params):
    """
    Open-loop prediction of z_t and y_t for one patient from x_{1:T} alone.

    Runs the fitted SSM forward with the noise set to its mean (zero) and
    propagates the predictive variance alongside it:

      mean:      z_t = alpha x_t + lam z_{t-1},        y_t = beta z_t + gamma x_t
      variance:  V^z_t = lam^2 V^z_{t-1} + sigma_w^2,  V^y_t = beta^2 V^z_t + sigma_e^2

    This is a *prediction*, not a fit: no y is consumed, so the model is being
    asked to reproduce the observations it has never seen. The variance is what
    makes that comparison fair — y_t carries observation noise of size sigma_e
    that no amount of parameter accuracy can predict, so the band, not the
    mean line, is what the actual trajectory should be judged against.

    Returns (Z_mean, Z_sd, Y_mean, Y_sd), each of shape (T,). The two sd arrays
    are all-zero when the params carry no noise scales.
    """
    X_patient = np.asarray(X_patient, dtype=float)
    T = len(X_patient)

    alpha, lam = params["alpha"], params["lam"]
    beta, gamma = params["beta"], params["gamma"]
    var_w = params.get("sigma_w", 0.0) ** 2
    var_e = params.get("sigma_e", 0.0) ** 2

    Z_mean, Z_var = np.zeros(T), np.zeros(T)
    Y_mean, Y_var = np.zeros(T), np.zeros(T)

    z_prev, v_prev = 0.0, 0.0          # z_{-1} = 0 exactly, as both generators do
    for t in range(T):
        z_t = alpha * X_patient[t] + lam * z_prev
        v_z = lam ** 2 * v_prev + var_w

        Z_mean[t], Z_var[t] = z_t, v_z
        Y_mean[t] = beta * z_t + gamma * X_patient[t]
        Y_var[t] = beta ** 2 * v_z + var_e

        z_prev, v_prev = z_t, v_z

    return Z_mean, np.sqrt(Z_var), Y_mean, np.sqrt(Y_var)


def _predictive_log_lik(actual, mean, sd):
    """
    log p(actual | model) under the plotted open-loop predictive Gaussian,
    summed over t. Used for the latent row, where there is no observed series
    and so no marginal likelihood to quote -- only how probable the true path
    is under what the model predicted.
    """
    var = np.asarray(sd, dtype=float) ** 2
    return float(-0.5 * np.sum(np.log(2 * np.pi * var) + (actual - mean) ** 2 / var))


def _panel_stats(symbol, log_lik, actual, mean, sd, band_sd):
    """
    The panel's log-likelihood, and the share of actual points in the band.

    The variable is named in the annotation -- log p(y) is a marginal likelihood
    and log p(z) a density against ground truth, so the two rows' numbers are
    not comparable and should not look as though they are.
    """
    covered = float(np.mean(np.abs(actual - mean) <= band_sd * sd))
    return f"log p({symbol}) {log_lik:+.1f}   ·   {covered:.0%} inside the band"


def _classifier_verdict(x, y, estimated_params, column_group, true_group):
    """
    Whether this column's model is the one the evidence picks for this patient,
    *and* whether that pick is the truth — the same comparison `infer_patient`
    makes, so the heading agrees with the reported classification.

    Correct means all three agree: the column's model, the MAP label, and the
    patient's actual group. So on a figure of one patient under both models the
    winning column reads correct and the other incorrect, while on a figure of
    one patient per group each column reads whether that patient was classified
    correctly. Returns None when the comparison cannot be made (fewer than two
    models, or params without the noise scales the likelihood needs).
    """
    if len(estimated_params) < 2 or true_group is None:
        return None
    if not all("sigma_w" in p and "sigma_e" in p for p in estimated_params.values()):
        return None

    log_liks = {group: _innovations_log_likelihood(y, x, params)
                for group, params in estimated_params.items()}
    map_label = max(log_liks, key=log_liks.get)      # equal priors, as infer_patient
    return column_group == map_label == true_group


def plot_trajectories(
    X,
    Z,
    Y,
    groups,
    true_group=None,
    sample_indices=None,
    group_labels=None,
    estimated_params=None,
    xlabel="Time step",
    title=None,
    subtitle=None,
    figsize=None,
    band_sd=1.0,
    output_path=False,
    dpi=200,
):
    """
    Predicted vs. actual trajectories for one sample patient per group.

    Small multiples: one **column per group**, two stacked rows (latent z_t,
    observation y_t). Groups get their own column rather than sharing an axis,
    because the columns hold *different patients* — overlaying them invites a
    comparison that means nothing, and at T=50 four noisy lines per axis is a
    hairball. Rows share a y-scale so the columns stay comparable. `x_t` is not
    drawn: it is i.i.d. noise here, so its own panel says nothing, and it is
    already implicit in both predictions.

    Within a column the colour encoding is fixed and reads the same in both
    panels: **blue solid = actual**, **orange dashed = the fitted model's
    prediction**, with an orange wash for the +/- band_sd predictive band.

    Each panel is annotated with a log-likelihood, and the two are *different
    quantities* by necessity:

      * observation row — the marginal `log p(y_{1:T} | M_j)` from the Kalman
        innovations, i.e. exactly what `infer_patient` compares between the two
        models to pick a group. With a uniform prior the difference between the
        two columns' values is the classification's log-odds.
      * latent row — `log p(z_{1:T} | M_j)` of the *true* path under the drawn
        open-loop predictive. z is latent, so there is no marginal likelihood to
        quote; this scores the prediction against ground truth, which only
        simulation makes available.

    Parameters
    ----------
    X, Z, Y : array-like, shape (n_patients, T)
        Input signals, ground-truth latent states, and observations. X is not
        plotted, but drives the prediction and the observation-row likelihood.
    groups : array-like, shape (n_patients,)
        Group label per patient. One column is drawn per distinct value.
    true_group : int, optional
        The plotted patient's actual group, for the `(correct)` / `(incorrect)`
        mark on the column headings. Needed only when `groups` does not carry it
        — showing one patient under both models duplicates the patient and puts
        the *model* index in `groups`. Defaults to reading it from `groups`.
    sample_indices : dict, optional
        Mapping from group value to the patient index to plot.
        e.g. {0: 5, 1: 12} plots patient 5 for group 0, patient 12 for group 1.
        Defaults to the first patient in each group.
    group_labels : dict, optional
        Mapping from group value to the column heading, e.g.
        {0: "Control", 1: "Treatment"}. Defaults to "Group {value} · Patient
        {index}". Pass this when the columns are not different patients — the
        same patient scored under both fitted models, say.
    estimated_params : dict, optional
        Per-group parameter estimates used to predict z and y, e.g.
        {0: {"alpha": .6, "lam": .8, "beta": .5, "gamma": .2, "sigma_w": .1,
             "sigma_e": .9}, 1: {...}}.
        `sigma_w` / `sigma_e` are optional; without them the predictive band is
        omitted. Defaults to None — ground truth only, no predictions.
    xlabel : str
        X-axis label (shown on the bottom row only).
    title : str, optional
        Headline. Defaults to None — no title, on the assumption that a caption
        carries it.
    subtitle : str, optional
        One line of context under the title. Defaults to None — no subtitle, on
        the assumption that a caption carries it.
    figsize : tuple, optional
        Figure size as (width, height). Defaults to a width that scales with the
        number of columns.
    band_sd : float
        Width of the predictive band, in standard deviations. Defaults to 1.
    output_path : str or False
        If a file path is provided (e.g. "plots/trajectories.png"), the figure
        is saved there instead of displayed. Supports any format matplotlib
        accepts (.png, .pdf, .svg, etc.). Defaults to False (display only).
    dpi : int
        Resolution used when saving. Defaults to 200.
    """
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    Y = np.asarray(Y, dtype=float)
    groups = np.asarray(groups)

    unique_groups = sorted(set(groups.tolist()))
    n_cols = len(unique_groups)
    T = X.shape[1]
    t = np.arange(T)

    show_pred = bool(estimated_params)

    if figsize is None:
        figsize = (5.6 * n_cols + 1.0, 5.4)

    fig, axes = plt.subplots(
        2, n_cols, figsize=figsize, sharex=True, sharey="row", squeeze=False
    )

    pending_labels = []

    for col, group in enumerate(unique_groups):
        # Pick the sample to plot: user-specified or first in the group
        group_indices = np.where(groups == group)[0]
        idx = (
            sample_indices[group]
            if sample_indices and group in sample_indices
            else group_indices[0]
        )

        ax_z, ax_y = axes[0][col], axes[1][col]

        params = estimated_params.get(group) if show_pred else None
        if params is not None:
            Z_mean, Z_sd, Y_mean, Y_sd = _predict_trajectories(X[idx], params)
            # Two different log-likelihoods, because the two rows are different
            # kinds of quantity. y is observed, so its row quotes the *marginal*
            # log p(y_{1:T} | M_j) from the Kalman innovations -- the statistic
            # infer_patient compares to pick a group, so the number in the figure
            # is the number that decides the classification. z is latent and has
            # no marginal likelihood; its row scores the true path against the
            # open-loop predictive that is drawn.
            # Both need the noise scales: without them the predictive has no
            # spread, so there is no density to score. Guard the computation, not
            # just the annotation -- a zero variance divides by zero first.
            scored = ("sigma_w" in params and "sigma_e" in params
                      and np.all(Z_sd > 0) and np.all(Y_sd > 0))
            log_lik_z = _predictive_log_lik(Z[idx], Z_mean, Z_sd) if scored else None
            log_lik_y = (_innovations_log_likelihood(Y[idx], X[idx], params)
                         if scored else None)
        else:
            Z_mean = Z_sd = Y_mean = Y_sd = None
            log_lik_z = log_lik_y = None

        for symbol, ax, actual, mean, sd, log_lik in (
            ("z", ax_z, Z[idx], Z_mean, Z_sd, log_lik_z),
            ("y", ax_y, Y[idx], Y_mean, Y_sd, log_lik_y),
        ):
            if mean is not None and np.any(sd > 0):
                # A wash, not a saturated block: the actual line stays readable
                # where it runs through the band, which is most of the time.
                ax.fill_between(t, mean - band_sd * sd, mean + band_sd * sd,
                                color=_MODEL_COLOUR, alpha=0.12, lw=0, zorder=2)
            ax.plot(t, actual, color=_ACTUAL_COLOUR, lw=1.8, zorder=4,
                    solid_capstyle="round", solid_joinstyle="round")
            if mean is not None:
                ax.plot(t, mean, color=_MODEL_COLOUR, lw=1.8, ls="--",
                        dash_capstyle="round", zorder=3)
            if log_lik is not None:
                ax.annotate(
                    _panel_stats(symbol, log_lik, actual, mean, sd, band_sd),
                    xy=(0, 1), xytext=(4, -5), xycoords="axes fraction",
                    textcoords="offset points", ha="left", va="top",
                    fontsize=9, color=_INK_SECONDARY, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="none", alpha=0.85),
                )

        # Direct labels supplement the legend, on the first column's latent
        # panel only — one clean instance beats a label in every panel. They go
        # at the time step where the two curves are furthest apart, so they need
        # no collision handling between them. Deferred until the y-limits are
        # final, because which side of its curve a label can sit on depends on
        # how much room is left there.
        if col == 0 and Z_mean is not None:
            gap = Z[idx] - Z_mean
            t_star = int(np.argmax(np.abs(gap)))
            actual_is_above = gap[t_star] > 0
            pending_labels = [
                (ax_z, t_star, Z[idx][t_star], "Actual", actual_is_above),
                (ax_z, t_star, Z_mean[t_star], "Predicted", not actual_is_above),
            ]

        heading = (
            group_labels[group]
            if group_labels and group in group_labels
            else f"Group {group} · Patient {idx}"
        )
        if show_pred:
            # groups[idx] is the column's model, which is the patient's true
            # group only when the columns really are different patients; pass
            # true_group when they are not.
            verdict = _classifier_verdict(
                X[idx], Y[idx], estimated_params, group,
                groups[idx] if true_group is None else true_group,
            )
            if verdict is not None:
                heading += " (correct)" if verdict else " (incorrect)"
        ax_z.set_title(heading, fontsize=11, color=_INK_PRIMARY, loc="left", pad=8)

    # --- chrome: recessive grid, a zero rule, ink-token text, two spines
    row_labels = ("Latent  $z_t$", "Observation  $y_t$")
    for row in range(2):
        for col in range(n_cols):
            ax = axes[row][col]
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color=_GRID, lw=0.8, ls="-")
            ax.xaxis.grid(False)
            ax.axhline(0, color=_AXIS, lw=0.8, zorder=1)
            ax.set_xlim(0, max(T - 1, 1))   # max(): a single time step is not a range
            ax.tick_params(colors=_INK_SECONDARY, labelsize=9, length=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(_AXIS)
                ax.spines[side].set_linewidth(0.8)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10.5,
                              color=_INK_SECONDARY, labelpad=8)
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=10.5, color=_INK_SECONDARY,
                              labelpad=6)

    # No title and no subtitle by default: the caption carries both. Either can
    # be passed in for a standalone figure.
    if title:
        fig.suptitle(title, fontsize=13, color=_INK_PRIMARY, x=0.008, ha="left",
                     y=0.992, va="top")
    # The legend occupies the top-right of the header, so the subtitle only gets
    # the top row when there is no title *and* no legend to collide with.
    has_top_row = bool(title) or show_pred
    if subtitle:
        fig.text(0.008, 0.952 if has_top_row else 0.988, subtitle, fontsize=10,
                 color=_INK_SECONDARY, ha="left", va="top")

    # Legend always present for the two-series rows; the grey input row is a
    # single series, named by its row label, so it stays out of the key.
    if show_pred:
        handles = [
            Line2D([], [], color=_ACTUAL_COLOUR, lw=1.8, label="Actual"),
            Line2D([], [], color=_MODEL_COLOUR, lw=1.8, ls="--",
                   label="Model prediction"),
        ]
        if any(
            "sigma_w" in p or "sigma_e" in p for p in estimated_params.values()
        ):
            handles.append(
                # 0.26, not the 0.12 used on the plot: the swatch is ~40x smaller
                # than the band, and a wash that reads correctly at panel size
                # disappears at legend size.
                Patch(facecolor=_MODEL_COLOUR, alpha=0.26,
                      label=f"Prediction ±{band_sd:g} SD")
            )
        leg = fig.legend(handles=handles, loc="upper right",
                         bbox_to_anchor=(0.998, 0.995), ncol=len(handles),
                         fontsize=10, handlelength=1.8, frameon=False,
                         borderaxespad=0.0, columnspacing=1.6)
        for text in leg.get_texts():
            text.set_color(_INK_SECONDARY)

    # Reserve only the rows the header actually occupies: row 1 is the title
    # and/or the legend, row 2 the subtitle.
    header_rows = int(has_top_row) + int(bool(subtitle))
    fig.tight_layout(rect=(0, 0, 1, (0.99, 0.96, 0.94)[header_rows]))

    # Direct labels last: a label 16pt off a curve near the top or bottom of its
    # panel would land outside it, and only now — after tight_layout has settled
    # the panel geometry — is there a panel height to measure that against.
    _LEAD_PT = 16.0        # leader length
    for ax, t_star, value, text, up in pending_labels:
        lo, hi = ax.get_ylim()
        room_above = (hi - value) / (hi - lo)
        panel_pt = ax.get_position().height * fig.get_figheight() * 72.0
        needed = (_LEAD_PT + 11.0) / panel_pt      # leader + one line of type
        fits_up, fits_down = room_above >= needed, (1.0 - room_above) >= needed
        if up and not fits_up:
            up = True if not fits_down else False
        elif not up and not fits_down:
            up = fits_up
        if not (fits_up or fits_down):
            # The panel is too short for a label at all. Drop it rather than
            # park it on top of the other curve — the legend still carries
            # identity, and a label pointing at the wrong line is worse than none.
            continue
        ax.annotate(
            text,
            xy=(t_star, value),
            xytext=(0, _LEAD_PT if up else -_LEAD_PT),
            textcoords="offset points",
            ha="center", va="bottom" if up else "top",
            fontsize=9, color=_INK_SECONDARY, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=0.8),
            # A leader, not an arrow: the two curves run close together, so the
            # label has to say which one it belongs to.
            arrowprops=dict(arrowstyle="-", color=_AXIS, lw=0.8, shrinkA=1,
                            shrinkB=1),
        )

    if output_path:
        fig.savefig(output_path, dpi=dpi, facecolor="white")
    else:
        plt.show()

    plt.close(fig)


def plot_mean_std(
    Y,
    groups,
    group_labels=None,
    xlabel="Time",
    ylabel="Y",
    title=None,
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
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
 
    if output_path:
        fig.savefig(output_path)
    else:
        plt.show()
 
    plt.close(fig)

def plot_accuracy_active_learning(curves, sems=None, save_path=None,
                                  title=None, subtitle=None, chance=0.5):
    """
    curves   : dict like {"Random": acc_random, "Minimum entropy": acc_me}
               Keys are used verbatim as legend labels, so pass display names
               (see POLICY_LABELS in utils.active_learning) rather than the
               internal policy identifiers. Iteration order fixes the colour
               assignment, so a policy keeps its colour even if another is
               dropped from the comparison.
    sems     : optional dict of same shape, for +/- 1 SEM shaded bands
    save_path: write a PNG here instead of showing the figure
    title    : headline; defaults to a plain description
    subtitle : one line of run configuration (n patients, noise regime, ...);
               None by default, on the assumption a caption carries it
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
        ax.annotate(f"Chance ({chance:g})", xy=(1, chance), xytext=(9, 6),
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

    ax.set_xlabel("Time step", fontsize=11, color=_INK_SECONDARY, labelpad=8)
    ax.set_ylabel("Fraction correct", fontsize=11, color=_INK_SECONDARY, labelpad=8)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.8)

    # Untitled by default: these figures go into documents where the caption
    # carries the headline. Pass title= to opt back in. With no title, an empty
    # one still reserves the space the subtitle sits in -- an annotation placed
    # in offset points does not reliably claim room from tight_layout.
    if title:
        ax.set_title(title, fontsize=12, color=_INK_PRIMARY, loc="left",
                     pad=30 if subtitle else 12)
    elif subtitle:
        ax.set_title("", pad=20)
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

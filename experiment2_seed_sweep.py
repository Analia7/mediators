"""
How much of experiment 2's policy comparison is just the data draw?

`experiment2.py` runs one seed. Its headline — entropy minimization holds >=0.90
from step 4, mutual information from step 10, random from step 13 — therefore has
no error bar behind it, so there is no way to tell a real ordering of the
policies from one lucky training set. This refits and reruns the whole thing
across several seeds and reports the mean and spread of each policy's curve.

Each seed redraws everything that is random:

  * the training set              seed=(seed, 0)   -> its own pair of EM fits
  * patient i's noise             seed = seed*N + i
  * patient i's probes / MC draws  same, split into per-policy streams

The patient seed is offset by seed*N_PATIENTS so that no two sweep seeds share a
patient, and so that **sweep seed 0 reproduces `experiment2.py` exactly** (its
patient seeds are 0..99). That equality is checked against the committed CSV at
the end of the run.

What is deliberately *not* random: the group assignment stays `i % 2`, so every
seed has exactly 50 patients per group. Balance is a design choice here, not a
draw, and letting it vary would add variance that says nothing about the
policies.

Writes results/experiment2_seed_sweep.csv (one row per seed x policy x timestep).
"""
import contextlib
import io
import os
import warnings

import numpy as np

from utils.data_generation import generate_synthetic_samples, PatientSimulator
from utils.em_algorithm import em_ssm
from utils.active_learning import run_patient_with_active_learning, steps_to_threshold, POLICY_LABELS
from utils.plots import plot_accuracy_active_learning

SEEDS = [0, 1, 2, 3, 4]
N_PATIENTS = 100
T = 30
POLICIES = ["random", "entropy minimization", "mutual information"]
CANDIDATES = np.linspace(-4, 4, 33)
THRESHOLDS = [0.90, 0.95]

# Reuse experiment 2's regime and TAG rather than duplicating them: execute the
# file only up to the point where it starts generating data. Its banner print
# sits inside that prefix, so swallow stdout — this log is the sweep's, not a
# half-run of experiment 2.
_src = open("experiment2.py").read()
_ns = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(_src[: _src.index('print(f"\\nRunning experiment with parameter set')],
         {"np": np}, _ns)
TAG = _ns["TAG"]
GEN_PARAMS = _ns["params"]

# The simulator's format, derived from the generative settings rather than
# retyped, so the two can never disagree about which regime is being run.
TRUE_PARAMS = {
    group: dict(
        alpha=GEN_PARAMS[f"alpha_{group}"], lam=GEN_PARAMS[f"lambda_{group}"],
        beta=GEN_PARAMS[f"beta_{group}"], gamma=GEN_PARAMS[f"gamma_{group}"],
        sigma_w=GEN_PARAMS[f"sigma_w_{group}"], sigma_e=GEN_PARAMS[f"sigma_e_{group}"],
    )
    for group in (0, 1)
}

warnings.simplefilter("ignore")   # EM non-convergence on the flat ridge is expected

rows = []
curves = {pol: [] for pol in POLICIES}      # per seed: the (T,) accuracy curve
em_converged = {}

print(f"Experiment 2 seed sweep · regime {TAG} · {len(SEEDS)} seeds · "
      f"{N_PATIENTS} patients · T={T}")
print(f"true params: {TRUE_PARAMS}\n")

for seed in SEEDS:
    j_train, X_train, _, Y_train = generate_synthetic_samples(
        n_patients=100, T=50, seed=(seed, 0), **GEN_PARAMS)

    # --- this seed's own EM fits: the policies and the filters run on these ---
    estimated_params = {}
    n_converged = 0
    for group in (0, 1):
        idx = j_train == group
        estimated_params[group], _, info = em_ssm(
            X_train[idx], Y_train[idx],
            alpha_init=0.5, lam_init=0.5, beta_init=0.5, gamma_init=0.5,
            sigma_w_init=0.2, verbose=False, return_info=True)
        n_converged += int(info["converged"])
    em_converged[seed] = n_converged

    # --- the paired policy comparison, exactly as experiment2.py runs it ---
    correct = {pol: np.zeros((N_PATIENTS, T), dtype=bool) for pol in POLICIES}
    for i in range(N_PATIENTS):
        patient_seed = seed * N_PATIENTS + i
        for policy in POLICIES:
            patient = PatientSimulator(i % 2, params=TRUE_PARAMS, seed=patient_seed)
            _, correct_t = run_patient_with_active_learning(
                patient, estimated_params, CANDIDATES,
                policy=policy, T=T, seed=patient_seed)
            correct[policy][i] = correct_t

    line = f"  seed {seed}  (EM converged {n_converged}/2)  "
    for policy in POLICIES:
        acc = correct[policy].mean(axis=0)
        sem = correct[policy].std(axis=0) / np.sqrt(N_PATIENTS)
        curves[policy].append(acc)
        for t in range(T):
            rows.append({"seed": seed, "policy": policy, "timestep": t + 1,
                         "accuracy": float(acc[t]), "sem": float(sem[t]),
                         "em_groups_converged": n_converged})
        line += f"{policy}: mean {acc.mean():.3f}  "
    print(line, flush=True)

curves = {pol: np.array(v) for pol, v in curves.items()}    # (n_seeds, T) each


def fmt_steps(x):
    return f"{x:>7}" if x is not None else f"{'never':>7}"


def mean_sd(values):
    """Mean and sd over the seeds that reached the threshold at all."""
    hit = [v for v in values if v is not None]
    if not hit:
        return "      -", "     -", len(values)
    a = np.array(hit, dtype=float)
    # One seed reaching the threshold has no spread to report — an sd of 0.00
    # there would read as "perfectly consistent", the opposite of the truth.
    sd = f"{a.std(ddof=1):>6.2f}" if len(a) > 1 else f"{'-':>6}"
    return f"{a.mean():>7.1f}", sd, len(values) - len(hit)


# ---- steps to a sustained threshold: the headline, per seed ------------------
for threshold in THRESHOLDS:
    print(f"\nSteps to a sustained accuracy >= {threshold:g} "
          f"(1-indexed; step 1 is the prior, so the floor is 2):")
    print(f"  {'policy':<24}" + "".join(f"{f'seed {s}':>7}" for s in SEEDS)
          + f"{'mean':>7}{'sd':>7}{'never':>7}")
    for policy in POLICIES:
        per_seed = [steps_to_threshold(curves[policy][k], threshold)
                    for k in range(len(SEEDS))]
        mean, sd, n_never = mean_sd(per_seed)
        print(f"  {policy:<24}" + "".join(fmt_steps(x) for x in per_seed)
              + f"{mean}{sd}{n_never:>7}")

# ---- mean accuracy over the run, and the paired gap over random -------------
print(f"\nMean accuracy over all {T} steps, per seed:")
print(f"  {'policy':<24}" + "".join(f"{f'seed {s}':>7}" for s in SEEDS)
      + f"{'mean':>7}{'sd':>7}")
for policy in POLICIES:
    m = curves[policy].mean(axis=1)
    print(f"  {policy:<24}" + "".join(f"{x:>7.3f}" for x in m)
          + f"{m.mean():>7.3f}{m.std(ddof=1):>7.3f}")

print("\nPaired gap over random, per seed (same patients, same noise, same probes"
      " at t=0):")
print(f"  {'policy':<24}" + "".join(f"{f'seed {s}':>7}" for s in SEEDS)
      + f"{'mean':>7}{'sd':>7}{'wins':>7}")
for policy in POLICIES[1:]:
    delta = curves[policy].mean(axis=1) - curves["random"].mean(axis=1)
    print(f"  {policy:<24}" + "".join(f"{x:>+7.3f}" for x in delta)
          + f"{delta.mean():>+7.3f}{delta.std(ddof=1):>7.3f}"
          + f"{int((delta > 0).sum()):>5}/{len(SEEDS)}")
print("  (a gap smaller than its own sd is not evidence of a better policy)")

# ---- write the rows out ------------------------------------------------------
path = "results/experiment2_seed_sweep.csv"
cols = list(rows[0].keys())
with open(path, "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])
                         for c in cols) + "\n")
print(f"\nWrote {path}  ({len(rows)} rows)")

# ---- the figure: mean across seeds, banded by the across-seed spread --------
mean_curves = {pol: curves[pol].mean(axis=0) for pol in POLICIES}
sd_curves = {pol: curves[pol].std(axis=0, ddof=1) for pol in POLICIES}
# Figures carry the display names; the CSV above keeps the internal identifiers.
plot_accuracy_active_learning(
    {POLICY_LABELS[p]: mean_curves[p] for p in POLICIES},
    {POLICY_LABELS[p]: sd_curves[p] for p in POLICIES},
    # PDF for the write-up -- vector, so it stays sharp at any reproduction
    # size; PNG alongside it for quick previewing.
    save_path=[f"results/experiment2_seed_sweep_{TAG}.pdf",
               f"results/experiment2_seed_sweep_{TAG}.png"],
    # Larger type than the single-run figure's default: this one is the summary
    # panel, so it is the one that gets reproduced small.
    font_scale=1.7,
)
print(f"Wrote results/experiment2_seed_sweep_{TAG}.pdf and .png")

# ---- cross-check: sweep seed 0 must reproduce experiment2.py exactly --------
single = f"results/experiment2_{TAG}.csv"
if os.path.exists(single):
    # Indexed by column position, not by name: the policy names contain spaces,
    # and numpy's `names=True` silently rewrites those to underscores.
    with open(single) as f:
        header = f.readline().strip().split(",")
    ref = np.loadtxt(single, delimiter=",", skiprows=1)
    worst = max(float(np.abs(curves[pol][0] - ref[:, header.index(f"{pol}_accuracy")]).max())
                for pol in POLICIES)
    verdict = "matches" if worst == 0.0 else f"DIFFERS (max |delta| {worst:.6g})"
    print(f"\nCross-check vs {single}: seed 0 {verdict}")
else:
    print(f"\nCross-check skipped: {single} not found")

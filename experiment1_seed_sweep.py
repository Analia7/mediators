"""
How much of experiment 1's per-regime accuracy is just the data draw?

Refits every regime across several seeds and reports the mean and spread. Any
single-seed number should be read against the `sd` column here: a difference
between two runs that sits inside it is sampling noise, not a change in method.

Writes results/experiment1_seed_sweep.csv (one row per regime x seed).
"""
import warnings

import numpy as np

from utils.data_generation import generate_synthetic_samples
from utils.em_algorithm import em_ssm
from utils.infer_patient import infer_patient
from utils.baseline import logistic_regression_prediction

SEEDS = [0, 1, 2, 3, 4]

# Reuse experiment 1's regime definitions rather than duplicating them: execute
# the file only up to its main loop.
_src = open("experiment1.py").read()
_ns = {}
exec(_src[: _src.index("# The observable process is ARX(1)")], {"np": np}, _ns)
parameter_sets = _ns["parameter_sets"]

rows = []
warnings.simplefilter("ignore")   # non-convergence on the flat ridge is expected

print(f"{'regime':<24}" + "".join(f"{f'seed {s}':>8}" for s in SEEDS)
      + f"{'mean':>8}{'sd':>7}   group split per seed")

for regime_index, (name, params) in enumerate(parameter_sets.items()):
    accs, splits = [], []
    for seed in SEEDS:
        j_tr, X_tr, _, Y_tr = generate_synthetic_samples(
            n_patients=100, T=50, seed=(seed, regime_index, 0), **params)
        j_te, X_te, _, Y_te = generate_synthetic_samples(
            n_patients=100, T=50, seed=(seed, regime_index, 1), **params)

        est = {}
        n_converged = 0
        for group in (0, 1):
            idx = j_tr == group
            est[group], _, info = em_ssm(
                X_tr[idx], Y_tr[idx],
                alpha_init=0.5, lam_init=0.5, beta_init=0.5, gamma_init=0.5,
                sigma_w_init=0.2, verbose=False, return_info=True)
            n_converged += int(info["converged"])

        pred = np.array([infer_patient(X_te[n], Y_te[n], est[0], est[1])[1]
                         for n in range(len(j_te))])
        acc = float(np.mean(pred == j_te))
        acc_full = float(np.mean(logistic_regression_prediction(
            X_tr, Y_tr, j_tr, X_te, Y_te, variant='full time series') == j_te))
        acc_mom = float(np.mean(logistic_regression_prediction(
            X_tr, Y_tr, j_tr, X_te, Y_te, variant='distribution moments') == j_te))

        accs.append(acc)
        splits.append(f"{int((j_tr == 0).sum())}/{int((j_tr == 1).sum())}")
        rows.append({"regime": name, "seed": seed,
                     "n_train_group0": int((j_tr == 0).sum()),
                     "n_train_group1": int((j_tr == 1).sum()),
                     "em_groups_converged": n_converged,
                     "accuracy_ssm": acc,
                     "accuracy_lr_full_timeseries": acc_full,
                     "accuracy_lr_moments": acc_mom})

    a = np.array(accs)
    print(f"{name:<24}" + "".join(f"{x:>8.2f}" for x in a)
          + f"{a.mean():>8.3f}{a.std(ddof=1):>7.3f}   " + " ".join(splits), flush=True)

path = "results/experiment1_seed_sweep.csv"
cols = list(rows[0].keys())
with open(path, "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])
                         for c in cols) + "\n")
print(f"\nWrote {path}  ({len(rows)} rows)")

by_regime = {}
for r in rows:
    by_regime.setdefault(r["regime"], []).append(r["accuracy_ssm"])
sds = {k: float(np.std(v, ddof=1)) for k, v in by_regime.items()}
print(f"Seed-to-seed sd of our accuracy: median {np.median(list(sds.values())):.3f}, "
      f"max {max(sds.values()):.3f} ({max(sds, key=sds.get)})")

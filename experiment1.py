from utils.data_generation import generate_synthetic_samples, PatientSimulator
from utils.plots import plot_scatter_by_group, plot_trajectories, plot_mean_std, plot_accuracy_active_learning
from utils.em_algorithm import em_ssm
from utils.infer_patient import infer_patient
from utils.active_learning import run_patient_with_active_learning
from utils.baseline import logistic_regression_prediction
import numpy as np

# Each regime draws its train and test sets from explicit, distinct seeds derived
# from SEED, so every regime is reproducible and independent of the others (and of
# whatever else consumes randomness). The global seed is a belt-and-braces default.
SEED = 0
np.random.seed(SEED)

# parameters for synthetic data generation experiments
# low noise (both sigma_w and sigma_e are low)
low_noise_params = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.2,
    'sigma_w_0': 0.1, 'sigma_e_0': 0.1,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.5,
    'sigma_w_1': 0.1, 'sigma_e_1': 0.1,
}

# high noise (both sigma_w and sigma_e are high)
high_noise_params = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.2,
    'sigma_w_0': 1.0, 'sigma_e_0': 1.0,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.5,
    'sigma_w_1': 1.0, 'sigma_e_1': 1.0,
}

# mixed (sigma_w is low, sigma_e is high)
mixed_noise_params_sigma_low = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.2,
    'sigma_w_0': 0.1, 'sigma_e_0': 1.0,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.5,
    'sigma_w_1': 0.1, 'sigma_e_1': 1.0,
}

# mixed (sigma_w is high, sigma_e is low)
mixed_noise_params_sigma_high = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.2,
    'sigma_w_0': 1.0, 'sigma_e_0': 0.1,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.5,
    'sigma_w_1': 1.0, 'sigma_e_1': 0.1,
}

# noise is average, gamma is high for both groups
gamma_high_params_both_groups = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.9,
    'sigma_w_0': 0.4, 'sigma_e_0': 0.4,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.9,
    'sigma_w_1': 0.4, 'sigma_e_1': 0.4,
}

# noise is average, gamma is low for both groups
gamma_low_params_both_groups = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.1,
    'sigma_w_0': 0.4, 'sigma_e_0': 0.4,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.1,
    'sigma_w_1': 0.4, 'sigma_e_1': 0.4,
}

# noise is average, gamma is high for group 0 and low for group 1
gamma_high_low_params = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.9,
    'sigma_w_0': 0.4, 'sigma_e_0': 0.4,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.1,
    'sigma_w_1': 0.4, 'sigma_e_1': 0.4,
}

# noise is average, gamma is low for group 0 and high for group 1
gamma_low_high_params = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.1,
    'sigma_w_0': 0.4, 'sigma_e_0': 0.4,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.9,
    'sigma_w_1': 0.4, 'sigma_e_1': 0.4,
}   

parameter_sets = {
    'low_noise': low_noise_params,
    'high_noise': high_noise_params,
    'mixed_noise_sigma_low': mixed_noise_params_sigma_low,
    'mixed_noise_sigma_high': mixed_noise_params_sigma_high,
    'gamma_high_both_groups': gamma_high_params_both_groups,
    'gamma_low_both_groups': gamma_low_params_both_groups,
    'gamma_high_low': gamma_high_low_params,
    'gamma_low_high': gamma_low_high_params,
}

# The observable process is ARX(1) with MA(1) noise, so only these combinations
# are identifiable (see README > Known limitations). Report them alongside the raw
# parameters -- alpha/beta/sigma_w individually are not recoverable, being defined
# only up to a common scale.
def identified(alpha, lam, beta, gamma, sigma_w, sigma_e):
    return {
        "b0":           beta * alpha + gamma,      # coefficient on x_t
        "b1":          -gamma * lam,               # coefficient on x_{t-1}
        "noise_var":    beta ** 2 * sigma_w ** 2 + (1 + lam ** 2) * sigma_e ** 2,
        "noise_acov1": -lam * sigma_e ** 2,
    }

PARAM_KEYS = ["alpha", "lam", "beta", "gamma", "sigma_w", "sigma_e"]
ID_KEYS = ["b0", "b1", "noise_var", "noise_acov1"]

accuracy_rows = []
parameter_rows = []

for regime_index, (name, params) in enumerate(parameter_sets.items()):
    print(f"\nRunning experiment with parameter set: {name}")
    # distinct seeds per regime, and train != test
    j_train, X_train, Z_train, Y_train = generate_synthetic_samples(
        n_patients=100, T=50, seed=(SEED, regime_index, 0), **params)
    j_test, X_test, Z_test, Y_test = generate_synthetic_samples(
        n_patients=100, T=50, seed=(SEED, regime_index, 1), **params)

    # estimate parameters for each group using EM algorithm
    estimated_params = {}
    for group in [0, 1]:
        idx = j_train == group
        X_group = X_train[idx]
        Y_group = Y_train[idx]

        # est_params, not params: `params` is the loop variable holding this
        # regime's generative settings, and rebinding it here was a live footgun.
        est_params, log_liks, info = em_ssm(
            X_group, Y_group,
            alpha_init=0.5, lam_init=0.5, beta_init=0.5, gamma_init=0.5,
            sigma_w_init=0.2,   # sigma_e_init defaults to np.std(Y)
            return_info=True,
        )

        estimated_params[group] = est_params

        print(f"\nGroup {group} estimated params:")
        for k, v in est_params.items():
            print(f"  {k}: {v:.4f}")

        true_params = {
            "alpha": params[f"alpha_{group}"], "lam": params[f"lambda_{group}"],
            "beta": params[f"beta_{group}"],   "gamma": params[f"gamma_{group}"],
            "sigma_w": params[f"sigma_w_{group}"], "sigma_e": params[f"sigma_e_{group}"],
        }
        id_true = identified(**true_params)
        id_est = identified(**{k: est_params[k] for k in PARAM_KEYS})

        row = {"regime": name, "group": group, "n_patients": int(idx.sum()),
               "converged": info["converged"], "n_iter": info["n_iter"],
               "best_iter": info["best_iter"], "log_lik": info["best_log_lik"]}
        for k in PARAM_KEYS:
            row[f"{k}_true"] = true_params[k]
            row[f"{k}_est"] = est_params[k]
        for k in ID_KEYS:
            row[f"{k}_true"] = id_true[k]
            row[f"{k}_est"] = id_est[k]
        parameter_rows.append(row)
    model_predictions = []
    for n in range(len(j_test)):
        prob_1, c_n, _, _ = infer_patient(X_test[n], Y_test[n], estimated_params[0], estimated_params[1])
        model_predictions.append(c_n)

    model_predictions = np.array(model_predictions)
    accuracy = np.mean(model_predictions == j_test)
    print("Our approach:")
    print(f"Accuracy: {accuracy:.4f}  ({int(accuracy * len(j_test))}/{len(j_test)} correct)")


    # Run logistic regression on full time series
    predictions_full = logistic_regression_prediction(X_train, Y_train, j_train, X_test, Y_test, variant='full time series')
    accuracy_full = np.mean(predictions_full == j_test)
    print(f"Logistic Regression (Full Time Series) Accuracy: {accuracy_full:.4f}")

    # Run logistic regression on distribution moments
    predictions_moments = logistic_regression_prediction(X_train, Y_train, j_train, X_test, Y_test, variant='distribution moments')
    accuracy_moments = np.mean(predictions_moments == j_test)
    print(f"Logistic Regression (Distribution Moments) Accuracy: {accuracy_moments:.4f}")

    accuracy_rows.append({
        "regime": name, "seed": SEED, "n_train": len(j_train), "n_test": len(j_test),
        "n_train_group0": int((j_train == 0).sum()), "n_train_group1": int((j_train == 1).sum()),
        "accuracy_ssm": accuracy,
        "accuracy_lr_full_timeseries": accuracy_full,
        "accuracy_lr_moments": accuracy_moments,
    })

# ---- write the results out, so the numbers exist outside this log ------------
def write_csv(path, rows):
    cols = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"Wrote {path}  ({len(rows)} rows)")

print("\n" + "=" * 76)
print(f"{'regime':<24}{'ours':>6}{'LR-full':>9}{'LR-mom':>8}   {'split':>9}{'EM conv':>10}")
for a in accuracy_rows:
    conv = [p["converged"] for p in parameter_rows if p["regime"] == a["regime"]]
    print(f"{a['regime']:<24}{a['accuracy_ssm']:>6.2f}{a['accuracy_lr_full_timeseries']:>9.2f}"
          f"{a['accuracy_lr_moments']:>8.2f}   "
          f"{a['n_train_group0']:>4}/{a['n_train_group1']:<4}{sum(conv):>8}/{len(conv)}")
print(f"{'mean':<24}{np.mean([a['accuracy_ssm'] for a in accuracy_rows]):>6.2f}"
      f"{np.mean([a['accuracy_lr_full_timeseries'] for a in accuracy_rows]):>9.2f}"
      f"{np.mean([a['accuracy_lr_moments'] for a in accuracy_rows]):>8.2f}")
print("=" * 76 + "\n")

write_csv("results/experiment1_accuracy.csv", accuracy_rows)
write_csv("results/experiment1_parameters.csv", parameter_rows)

# def generate_synthetic_samples(
#     n_patients=100,
#     T=50,
#     # Group 0 SSM parameters
#     alpha_0=0.6, lambda_0=0.8, beta_0=0.5, gamma_0=0.2,
#     sigma_w_0=0.1, sigma_e_0=0.9,
#     # Group 1 SSM parameters
#     alpha_1=0.3, lambda_1=0.4, beta_1=0.9, gamma_1=0.5,
#     sigma_w_1=0.1, sigma_e_1=0.9,
# )

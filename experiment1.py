from utils.data_generation import generate_synthetic_samples, PatientSimulator
from utils.plots import plot_scatter_by_group, plot_trajectories, plot_mean_std, plot_accuracy_active_learning
from utils.em_algorithm import em_ssm
from utils.infer_patient import infer_patient
from utils.active_learning import run_patient_with_active_learning
from utils.baseline import logistic_regression_prediction
import numpy as np

# parameters for synthetic data generation experiments
# low noise (both sigma_w and sigma_e are low)
low_noise_params = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.2,
    'sigma_w_0': 0.1, 'sigma_e_0': 0.25,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.5,
    'sigma_w_1': 0.1, 'sigma_e_1': 0.25,
}

# high noise (both sigma_w and sigma_e are high)
high_noise_params = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.2,
    'sigma_w_0': 0.5, 'sigma_e_0': 1.0,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.5,
    'sigma_w_1': 0.5, 'sigma_e_1': 1.0,
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
    'sigma_w_0': 0.5, 'sigma_e_0': 0.25,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.5,
    'sigma_w_1': 0.5, 'sigma_e_1': 0.25,
}

# noise is average, gamma is high for both groups
gamma_high_params_both_groups = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.9,
    'sigma_w_0': 0.25, 'sigma_e_0': 0.4,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.9,
    'sigma_w_1': 0.25, 'sigma_e_1': 0.4,
}

# noise is average, gamma is low for both groups
gamma_low_params_both_groups = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.1,
    'sigma_w_0': 0.25, 'sigma_e_0': 0.4,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.1,
    'sigma_w_1': 0.25, 'sigma_e_1': 0.4,
}

# noise is average, gamma is high for group 0 and low for group 1
gamma_high_low_params = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.9,
    'sigma_w_0': 0.25, 'sigma_e_0': 0.4,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.1,
    'sigma_w_1': 0.25, 'sigma_e_1': 0.4,
}

parameter_sets = {
    'low_noise': low_noise_params,
    'high_noise': high_noise_params,
    'mixed_noise_sigma_low': mixed_noise_params_sigma_low,
    'mixed_noise_sigma_high': mixed_noise_params_sigma_high,
    'gamma_high_both_groups': gamma_high_params_both_groups,
    'gamma_low_both_groups': gamma_low_params_both_groups,
    'gamma_high_low': gamma_high_low_params,
}

for name, params in parameter_sets.items():
    print(f"\nRunning experiment with parameter set: {name}")
    j_train, X_train, Z_train, Y_train = generate_synthetic_samples(n_patients=100, T=50, **params)
    j_test, X_test, Z_test, Y_test = generate_synthetic_samples(n_patients=100, T=50, **params)

    # estimate parameters for each group using EM algorithm
    estimated_params = {}
    for group in [0, 1]:
        idx = j_train == group
        X_group = X_train[idx]
        Y_group = Y_train[idx]

        params, log_liks = em_ssm(
            X_group, Y_group,
            alpha_init=0.5, lam_init=0.5, beta_init=0.5, gamma_init=0.5,
            sigma_w_init=0.2, sigma_e_init=0.2,
        )

        estimated_params[group] = params

        print(f"\nGroup {group} estimated params:")
        for k, v in params.items():
            print(f"  {k}: {v:.4f}")
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

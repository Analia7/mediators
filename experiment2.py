from utils.data_generation import generate_synthetic_samples, PatientSimulator
from utils.plots import plot_scatter_by_group, plot_trajectories, plot_mean_std, plot_accuracy_active_learning
from utils.em_algorithm import em_ssm
from utils.infer_patient import infer_patient
from utils.active_learning import run_patient_with_active_learning
from utils.baseline import logistic_regression_prediction
import numpy as np

print("\nRunning experiment 2: Active Learning with EM-estimated parameters")
# parameters for synthetic data generation experiments
# mixed (sigma_w is low, sigma_e is high)
params = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.2,
    'sigma_w_0': 0.1, 'sigma_e_0': 1.0,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.5,
    'sigma_w_1': 0.1, 'sigma_e_1': 1.0,
}
  
print(f"\nRunning experiment with parameter set: {params}")
j_train, X_train, Z_train, Y_train = generate_synthetic_samples(n_patients=100, T=50, **params)

# estimate parameters for each group using EM algorithm
estimated_params = {}
for group in [0, 1]:
    idx = j_train == group
    X_group = X_train[idx]
    Y_group = Y_train[idx]


    est_params, log_liks = em_ssm(
        X_group, Y_group,
        alpha_init=0.5, lam_init=0.5, beta_init=0.5, gamma_init=0.5,
        sigma_w_init=0.2, sigma_e_init=0.2,
        )

    estimated_params[group] = est_params

    print(f"\nGroup {group} estimated params:")
    for k, v in est_params.items():
        print(f"  {k}: {v:.4f}")

# rewrite params dictionary to match the expected format for PatientSimulator
params = {
    0: dict(alpha=0.6, lam=0.8, beta=0.5, gamma=0.2, sigma_w=0.1, sigma_e=1.0),
    1: dict(alpha=0.3, lam=0.4, beta=0.9, gamma=0.5, sigma_w=0.1, sigma_e=1.0),
}
n_patients = 100
T = 30
policies = ["random", "uncertainty sampling", "mutual information"]
# to account for time step 0
correct = {pol: np.zeros((n_patients, T), dtype=bool) for pol in policies}
candidates = np.linspace(-4, 4, 33)

for i in range(n_patients):
    true_group = i % 2
    #patient = PatientSimulator(true_group)

    for policy in policies:
        patient = PatientSimulator(true_group, params=params)
        _, correct_t = run_patient_with_active_learning(patient, estimated_params, candidates, policy=policy, T=T)
        correct[policy][i] = correct_t

accuracy = {pol: correct[pol].mean(axis=0) for pol in policies} 
sem = {pol: correct[pol].std(axis=0)/np.sqrt(n_patients) for pol in policies} # standard error of the mean

plot_accuracy_active_learning(accuracy, sem, save_path="results/experiment2_mixed_noise_sigmaw_low.png")
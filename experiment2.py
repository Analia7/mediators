from utils.data_generation import generate_synthetic_samples, PatientSimulator
from utils.plots import plot_scatter_by_group, plot_trajectories, plot_mean_std, plot_accuracy_active_learning
from utils.em_algorithm import em_ssm
from utils.infer_patient import infer_patient
from utils.active_learning import run_patient_with_active_learning, steps_to_threshold, POLICY_LABELS
from utils.baseline import logistic_regression_prediction
import numpy as np

print("\nRunning experiment 2: Active Learning with EM-estimated parameters")

# Every stochastic component below takes an explicit seed derived from SEED, so a
# rerun reproduces exactly and each component is independent of the others:
#   - the training set                        seed=(SEED, 0)
#   - patient i's noise realisation           seed=i          (shared across policies)
#   - patient i's probe sequence / MC draws    seed=i          (per-policy streams)
# The global np.random.seed is a belt-and-braces default for anything added later
# that forgets to take a seed; nothing here relies on it.
SEED = 0
np.random.seed(SEED)

TAG = "mixed_noise_sigmaw_low"

# parameters for synthetic data generation experiments
# mixed (sigma_w is low, sigma_e is high)
params = {
    'alpha_0': 0.6, 'lambda_0': 0.8, 'beta_0': 0.5, 'gamma_0': 0.2,
    'sigma_w_0': 0.1, 'sigma_e_0': 1.0,
    'alpha_1': 0.3, 'lambda_1': 0.4, 'beta_1': 0.9, 'gamma_1': 0.5,
    'sigma_w_1': 0.1, 'sigma_e_1': 1.0,
}
  
print(f"\nRunning experiment with parameter set: {params}")
j_train, X_train, Z_train, Y_train = generate_synthetic_samples(n_patients=100, T=50, seed=(SEED, 0), **params)

# estimate parameters for each group using EM algorithm
estimated_params = {}
for group in [0, 1]:
    idx = j_train == group
    X_group = X_train[idx]
    Y_group = Y_train[idx]


    est_params, log_liks = em_ssm(
        X_group, Y_group,
        alpha_init=0.5, lam_init=0.5, beta_init=0.5, gamma_init=0.5,
        sigma_w_init=0.2,   # sigma_e_init defaults to np.std(Y)
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

    for policy in policies:
        # seed=i on the simulator gives every policy the same noise realisation for
        # patient i (common random numbers), so the comparison is paired rather than
        # confounded by which patient each policy happened to draw. seed=i on the
        # run varies the "random" policy's probe sequence across patients -- with the
        # default seed=0 every patient got an identical x sequence.
        patient = PatientSimulator(true_group, params=params, seed=i)
        _, correct_t = run_patient_with_active_learning(
            patient, estimated_params, candidates, policy=policy, T=T, seed=i)
        correct[policy][i] = correct_t

accuracy = {pol: correct[pol].mean(axis=0) for pol in policies} 
sem = {pol: correct[pol].std(axis=0)/np.sqrt(n_patients) for pol in policies} # standard error of the mean

# ---- log the curves, so the numbers exist outside the PNG ------------------
print(f"\nAccuracy per time step (n_patients={n_patients}, +/- 1 SEM):")
header = "  t  " + "".join(f"{pol:>28}" for pol in policies)
print(header)
print("  " + "-" * (len(header) - 2))
for t in range(T):
    row = f"  {t + 1:<3}"
    for pol in policies:
        row += f"{accuracy[pol][t]:>21.3f} +/-{sem[pol][t]:5.3f}"
    print(row)

print("\nSummary:")
print(f"  {'policy':<24}{'final':>8}{'mean':>8}{'>=0.90':>9}{'>=0.95':>9}{'>=0.99':>9}")
for pol in policies:
    a = accuracy[pol]
    def fmt(x):
        return f"{x:>9}" if x is not None else f"{'never':>9}"
    print(f"  {pol:<24}{a[-1]:>8.3f}{a.mean():>8.3f}"
          f"{fmt(steps_to_threshold(a, 0.90))}{fmt(steps_to_threshold(a, 0.95))}{fmt(steps_to_threshold(a, 0.99))}")
print("  ('>=x' = first time step from which accuracy stays at or above x)")

csv_path = f"results/experiment2_{TAG}.csv"
cols = ["timestep"] + [f"{p}_accuracy" for p in policies] + [f"{p}_sem" for p in policies]
table = np.column_stack(
    [np.arange(1, T + 1)] + [accuracy[p] for p in policies] + [sem[p] for p in policies])
np.savetxt(csv_path, table, delimiter=",", header=",".join(cols), comments="", fmt="%.6f")
print(f"\nWrote {csv_path}")

# Figures carry the display names; the CSV above keeps the internal identifiers.
plot_accuracy_active_learning(
    {POLICY_LABELS[p]: accuracy[p] for p in policies},
    {POLICY_LABELS[p]: sem[p] for p in policies},
    save_path=f"results/experiment2_{TAG}.png",
    subtitle=(f"{n_patients} patients, T={T}, mixed noise (sigma_w=0.1, sigma_e=1.0); "
              f"bands are +/- 1 SEM; seed={SEED}"),
)

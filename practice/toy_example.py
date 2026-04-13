"""Harder toy example for Kalman vs Particle filtering.

System model:
    x_k = A x_{k-1} + B u_{k-1} + w_k
    y_k = H x_k + v_k

This version introduces:
- piecewise control regimes,
- occasional large process disturbances,
- missing measurements,
- outlier measurements.
"""

from kalman_filter import KalmanFilter
from particle_filter import ParticleFilter
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.5], [1.0]])
Q = np.array([[0.02, 0.0], [0.0, 0.02]])
H = np.array([[1.0, 0.0], [0.0, 1.0]])
R = np.array([[0.12, 0.0], [0.0, 0.12]])
x_0 = np.array([0.0, 0.0])
P_0 = np.eye(2)

num_steps = 160
u = np.zeros((num_steps, 1))

u[:45] = 0.15
u[45:90] = -0.25
u[90:130] = 0.35
u[130:] = -0.05
u += 0.05 * np.sin(np.linspace(0.0, 8.0 * np.pi, num_steps)).reshape(-1, 1)

true_states = np.zeros((num_steps, 2))
measurements = np.zeros((num_steps, 2))
measurement_available = np.ones(num_steps, dtype=bool)

for k in range(num_steps):
    prev_state = x_0 if k == 0 else true_states[k - 1]

    process_noise = rng.multivariate_normal(np.zeros(2), Q)
    if rng.random() < 0.07:
        process_noise += rng.multivariate_normal(np.zeros(2), 12.0 * Q)

    true_states[k] = A @ prev_state + B @ u[k] + process_noise

    if rng.random() < 0.12:
        measurement_available[k] = False
        measurements[k] = np.array([np.nan, np.nan])
        continue

    measurement_noise = rng.multivariate_normal(np.zeros(2), R)
    measurement = H @ true_states[k] + measurement_noise
    if rng.random() < 0.08:
        measurement += rng.multivariate_normal(np.zeros(2), 20.0 * R)
    measurements[k] = measurement

kf = KalmanFilter(A, B, Q, H, R, x_0, P_0)
pf = ParticleFilter(num_particles=1500, A=A, B=B, Q=Q, H=H, R=R, x_0=x_0)

kf_estimates = np.zeros((num_steps, 2))
pf_estimates = np.zeros((num_steps, 2))
kf_std = np.zeros((num_steps, 2))

x_kf = x_0.copy()
P_kf = P_0.copy()

for k in range(num_steps):
    control = u[k - 1] if k > 0 else u[0]

    x_minus, P_minus = kf.predict(x_kf, P_kf, control)
    if measurement_available[k]:
        x_kf, P_kf = kf.update(x_minus, P_minus, measurements[k])
    else:
        x_kf, P_kf = x_minus, P_minus

    kf_estimates[k] = x_kf
    kf_std[k] = np.sqrt(np.diag(P_kf))

    pf.predict(control)
    if measurement_available[k]:
        pf.update(measurements[k])
    pf_estimates[k] = np.average(pf.particles, axis=0, weights=pf.weights)

kf_rmse = np.sqrt(np.mean((kf_estimates - true_states) ** 2, axis=0))
pf_rmse = np.sqrt(np.mean((pf_estimates - true_states) ** 2, axis=0))

print(f"Measurement dropouts: {(~measurement_available).sum()} / {num_steps}")
print(f"KF RMSE [position, velocity]: {kf_rmse}")
print(f"PF RMSE [position, velocity]: {pf_rmse}")

time = np.arange(num_steps)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
state_names = ["Position x[0]", "Velocity x[1]"]

for idx, ax in enumerate(axes):
    ax.plot(time, true_states[:, idx], color="black", linewidth=1.5, label="True")
    ax.plot(time, kf_estimates[:, idx], color="tab:blue", label="Kalman")
    ax.plot(time, pf_estimates[:, idx], color="tab:orange", label="Particle")
    ax.fill_between(
        time,
        kf_estimates[:, idx] - 2.0 * kf_std[:, idx],
        kf_estimates[:, idx] + 2.0 * kf_std[:, idx],
        color="tab:blue",
        alpha=0.15,
        label="Kalman ±2σ" if idx == 0 else None,
    )
    ax.scatter(
        time[measurement_available],
        measurements[measurement_available, idx],
        s=12,
        color="gray",
        alpha=0.5,
        label="Measurements" if idx == 0 else None,
    )
    ax.set_ylabel(state_names[idx])
    ax.grid(alpha=0.25)

axes[0].legend(loc="upper left")
axes[0].set_title("Harder state estimation scenario")
axes[1].set_xlabel("Time step")

fig.tight_layout()
fig.savefig("state_estimation_harder.png", dpi=150)
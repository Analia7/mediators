"""Nonlinear state estimation example with a particle filter.

System model (nonlinear):
    x_k = 0.5 x_{k-1} + 25 x_{k-1} / (1 + x_{k-1}^2) + 8 cos(1.2 k) + w_k
    y_k = x_k^2 / 20 + v_k

This is a classic nonlinear benchmark where a standard linear Kalman filter
is not directly applicable without linearization.

Kalman filter applied after linearization approxmation of system.
"""

import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(42)

num_steps = 120
q_var = 10.0  # process noise variance
r_var = 1.0   # measurement noise variance
num_particles = 3000


def f_state(x_prev, k):
    return 0.5 * x_prev + 25.0 * x_prev / (1.0 + x_prev**2) + 8.0 * np.cos(1.2 * k)


def h_meas(x):
    return x**2 / 20.0


true_x = np.zeros(num_steps)
measurements = np.zeros(num_steps)

x0 = 0.1
true_x[0] = x0
measurements[0] = h_meas(true_x[0]) + rng.normal(0.0, np.sqrt(r_var))

for k in range(1, num_steps):
    process_noise = rng.normal(0.0, np.sqrt(q_var))
    true_x[k] = f_state(true_x[k - 1], k) + process_noise

    meas_noise = rng.normal(0.0, np.sqrt(r_var))
    measurements[k] = h_meas(true_x[k]) + meas_noise


particles = rng.normal(loc=0.0, scale=5.0, size=num_particles)
weights = np.ones(num_particles) / num_particles
estimates = np.zeros(num_steps)


def gaussian_pdf(value, mean, var):
    scale = np.sqrt(2.0 * np.pi * var)
    exponent = -0.5 * ((value - mean) ** 2) / var
    return np.exp(exponent) / scale


for k in range(num_steps):
    if k > 0:
        process_noise = rng.normal(0.0, np.sqrt(q_var), size=num_particles)
        particles = f_state(particles, k) + process_noise

    pred_meas = h_meas(particles)
    likelihood = gaussian_pdf(measurements[k], pred_meas, r_var)
    weights *= likelihood

    weight_sum = np.sum(weights)
    if weight_sum == 0 or not np.isfinite(weight_sum):
        weights.fill(1.0 / num_particles)
    else:
        weights /= weight_sum

    estimates[k] = np.sum(weights * particles)

    n_eff = 1.0 / np.sum(weights**2)
    if n_eff < num_particles / 2.0:
        indices = rng.choice(num_particles, size=num_particles, p=weights)
        particles = particles[indices]
        weights.fill(1.0 / num_particles)


rmse = np.sqrt(np.mean((estimates - true_x) ** 2))
print(f"Nonlinear PF RMSE: {rmse:.4f}")


time = np.arange(num_steps)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

axes[0].plot(time, true_x, color="black", linewidth=1.6, label="True state")
axes[0].plot(time, estimates, color="tab:orange", linewidth=1.2, label="PF estimate")
axes[0].set_ylabel("x")
axes[0].set_title("Nonlinear system state estimation")
axes[0].grid(alpha=0.25)
axes[0].legend(loc="upper right")

axes[1].plot(time, measurements, color="tab:blue", linewidth=1.0, label="Measurement")
axes[1].set_ylabel("y")
axes[1].set_xlabel("Time step")
axes[1].grid(alpha=0.25)
axes[1].legend(loc="upper right")

fig.tight_layout()
fig.savefig("nonlinear_state_estimation.png", dpi=150)

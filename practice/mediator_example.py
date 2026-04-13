"""
The system is defined as following:

- there is a treatment variable u that is binary
- there is a mediator variable x that is continuous and unobserved and is affected by the treatment variable
- there is an outcome variable y that is continuous and observed and is affected by the mediator variable

A real life example of this could be a drug trial where:
- u is whether the patient receives the drug or a placebo
- x is the level of a certain biomarker that is affected by the drug and is not directly observed
- y is the health outcome of the patient that is affected by the biomarker and is observed

We use the particle filter and the Kalman filter (on the linearized system) to estimate the unobserved mediator variable x over time based on the observed outcome variable y and the treatment variable u.

The nonlinear system is defined in the diabetes_class.py file, and the Kalman filter and particle filter implementations are in their respective files. The code in this file will simulate the system, apply both filters, and compare their estimates of the hidden mediator variable x over time.

"""
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilter
from particle_filter import ParticleFilter
from diabetes_class import InsulinGlucoseSystem

# Set random seed for reproducibility
rng = np.random.default_rng(123)

# Values for treatment variable u (binary)
num_steps = 100
u = rng.integers(0, 2, size=num_steps)  # Random binary treatment assignment

# True mediator variable x (unobserved)
system = InsulinGlucoseSystem()
true_x = np.zeros(num_steps)
true_x[0] = 0.0  # Initial insulin level
for t in range(1, num_steps):
    true_x[t] = system.transition_function(true_x[t - 1], u[t - 1])

# Observed outcome variable y
y = np.zeros(num_steps)
for t in range(num_steps):
    y[t] = system.observation_function(true_x[t])

# Kalman Filter setup (using linearized system)
kf = KalmanFilter(
    A=np.array([[system.alpha]]),  # State transition matrix
    B=np.array([[system.beta]]),   # Control input matrix
    Q=np.array([[system.proc_std**2]]),  # Process noise covariance
    H=np.array([[-(system.G_base * system.gamma) / (1 + system.gamma * true_x[t])**2]]),  # Measurement matrix (Jacobian)
    R=np.array([[system.obs_std**2]]),  # Measurement noise covariance
    x_0=np.array([0.0]),  # Initial state estimate
    P_0=np.array([[1.0]])  # Initial error covariance
)
# Particle Filter setup
pf = ParticleFilter(
    num_particles=1000,
    A=np.array([[system.alpha]]),
    B=np.array([[system.beta]]),
    Q=np.array([[system.proc_std**2]]),
    H=np.array([[-(system.G_base * system.gamma) / (1 + system.gamma * true_x[t])**2]]),
    R=np.array([[system.obs_std**2]]),
    x_0=np.array([0.0])
)

# Run Kalman Filter and Particle Filter
kf_estimates = np.zeros(num_steps)
pf_estimates = np.zeros(num_steps)
kf_std = np.zeros(num_steps)
x_kf = np.array([0.0])  # Initial state estimate for Kalman Filter
P_kf = np.array([[1.0]])  # Initial error covariance for Kalman Filter
for t in range(num_steps):
    # Kalman Filter prediction and update
    x_minus, P_minus = kf.predict(x_kf, P_kf, u[t])
    x_kf, P_kf = kf.update(x_minus, P_minus, np.array([y[t]]))
    kf_estimates[t] = x_kf[0]
    kf_std[t] = np.sqrt(P_kf[0, 0])

    # Particle Filter prediction and update
    pf.predict(u[t])
    pf.update(np.array([y[t]]))
    pf_estimates[t] = np.mean(pf.particles)

# Plot results
# mention what the system models in the plot title
plt.figure(figsize=(12, 6))
plt.plot(true_x, label='True Mediator (x)', color='black', linewidth=1.5)
plt.plot(kf_estimates, label='Kalman Filter Estimate', color='blue')
plt.fill_between(range(num_steps), kf_estimates - 2*kf_std, kf_estimates + 2*kf_std, color='blue', alpha=0.2)
plt.plot(pf_estimates, label='Particle Filter Estimate', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Mediator Level (x)')
plt.title('Estimation of Hidden Mediator Variable x (internal insulin) over Time')
plt.legend()
plt.grid()
plt.savefig('mediator_estimation.png', dpi=300)
plt.show()

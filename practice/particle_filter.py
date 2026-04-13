"""
Particle Filter implementation for state estimation.

The system is defined as:
x_k = A*x_{k-1} + B*u_{k-1} + w_k
y_k = H*x_k + v_k
"""
import numpy as np


class ParticleFilter:
    def __init__(self, num_particles, A, B, Q, H, R, x_0):
        """
        Initialize the Particle Filter parameters.
        """
        self.num_particles = num_particles
        self.A = A
        self.B = B
        self.Q = Q
        self.H = H
        self.R = R
        self.x_0 = x_0

        # Initialize particles and weights
        self.particles = np.random.multivariate_normal(x_0, Q, num_particles)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, u_prev):
        """
        Predict the next state of the particles.
        
        Generate process noise for each particle and update their states based on the system dynamics.
        """
        u_prev = np.atleast_1d(u_prev)

        # Generate process noise
        w = np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q, self.num_particles)
        
        # Update particle states
        for i in range(self.num_particles):
            self.particles[i] = self.A @ self.particles[i] + self.B @ u_prev + w[i]

    def update(self, y):
        """
        Update the weights of the particles based on the new measurement.

        Using observed output y_k and the weights w_{t-1}^i calculate the wight update, using
        w_t^i = p(y_k | x_k^i) * w_{t-1}^i
        where p(y_k | x_k^i) is the likelihood of the measurement given the particle's state, which can be calculated using the measurement model and the measurement noise covariance R.
        After updating the weights, normalize them so that they sum to 1.
        """
        for i in range(self.num_particles):
            # Calculate the likelihood of the measurement given the particle's state
            likelihood = self.gaussian_likelihood(y, self.H @ self.particles[i], self.R)
            self.weights[i] *= likelihood
        
        # Normalize weights
        self.weights /= np.sum(self.weights)

        # Resample particles based on their weights if N_eff is below a threshold of N/3
        N_eff = 1.0 / np.sum(self.weights**2)
        if N_eff < self.num_particles / 3:
            self.resample()

    def gaussian_likelihood(self, y, mean, cov):
        """
        Calculate the Gaussian likelihood of a measurement given a mean and covariance.
        """
        d = len(y)
        cov_inv = np.linalg.inv(cov)
        diff = y - mean
        exponent = -0.5 * diff.T @ cov_inv @ diff
        return (1.0 / np.sqrt((2 * np.pi)**d * np.linalg.det(cov))) * np.exp(exponent)

    def resample(self):
        """
        Resample the particles based on their weights.
        """
        # Perform systematic resampling
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

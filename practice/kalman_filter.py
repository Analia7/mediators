"""
Kalman Filter implementation for a simple linear system.

The system is defined as:
x_k = A*x_{k-1} + B*u_{k-1} + w_k
y_k = H*x_k + v_k
"""

import numpy as np

class KalmanFilter:
    def __init__(self, A, B, Q, H, R, x_0, P_0):
        """
        Initialize the Kalman Filter parameters.
        """
        self.A = A # State transition matrix
        self.B = B # Control input matrix
        self.Q = Q # Process noise covariance (for x)
        self.H = H # Measurement matrix
        self.R = R # Measurement noise covariance
        self.x_0 = x_0 # Initial state estimate
        self.P_0 = P_0 # Initial error covariance

    def predict(self, x_prev, P_prev, u_prev=None):
        """
        Predict the next state and error covariance.

        If u_prev is None, assumes zero control input.
        """
        control_effect = 0 if u_prev is None else self.B @ np.atleast_1d(u_prev)
        x_minus = self.A @ x_prev + control_effect
        P_minus = self.Q + self.A @ P_prev @ self.A.T
        return x_minus, P_minus

    def update(self, x_minus, P_minus, y):
        """
        Update the state estimate and error covariance with the new measurement.
        """
        # Kalman Gain
        K = P_minus@self.H.T@np.linalg.inv(self.H@P_minus@self.H.T+self.R)

        # Update state estimate and error covariance
        x = x_minus + K@(y - self.H@x_minus)
        P = (np.eye(len(P_minus))- K@self.H)@P_minus
        return x, P


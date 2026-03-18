import numpy as np

class KalmanFilter:
    def __init__(self, A, Q, H, R, x_0, P_0):
        """
        Initialize the Kalman Filter parameters.
        """
        self.A = A
        self.Q = Q
        self.H = H
        self.R = R
        self.x_0 = x_0
        self.P_0 = P_0

    def predict(self, x_prev, P_prev):
        """
        Predict the next state and error covariance.
        """
        x_minus = self.A@x_prev
        P_minus = self.Q + self.A@P_prev@self.A.T
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


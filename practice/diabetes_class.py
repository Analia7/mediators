import numpy as np

class InsulinGlucoseSystem:
    """
    State-Space Model for Insulin-Glucose Dynamics
    ----------------------------------------------
    This system models a hidden mediator (Insulin) and an observed 
    outcome (Glucose) using nonlinear dynamics.

    1. State Transition (The Hidden Mediator):
       x_{t+1} = alpha * x_t + beta * u_t + w_t
       
       Where:
       - x: Hidden Insulin concentration (continuous)
       - u: Binary treatment (1 = Bolus, 0 = None)
       - alpha: Decay rate (persistence of insulin)
       - beta: Absorption coefficient (impact of treatment)
       - w: Process noise (metabolic variability) ~ N(0, proc_std^2)

    2. Observation Function (The Nonlinear Outcome):
       y_t = G_base / (1 + gamma * x_t) + v_t
       
       Where:
       - y: Observed Glucose reading (continuous)
       - G_base: Patient's baseline glucose level
       - gamma: Insulin sensitivity (how much x affects y)
       - v: Measurement noise (sensor error) ~ N(0, obs_std^2)

    3. Linearization (The Jacobian for Kalman Filtering):
       H = dy/dx = - (G_base * gamma) / (1 + gamma * x)^2
       
       This derivative represents the local slope of the nonlinear 
       observation curve at a specific insulin level x.
    """
    def __init__(self):
        
        # Parameters
        self.alpha = 0.95      # Insulin decay (5% per time step)
        self.beta = 2.0        # Impact of binary treatment u
        self.gamma = 0.1       # Insulin sensitivity
        self.G_base = 120.0    # Baseline glucose (mg/dL)
        
        # Noise levels
        self.proc_std = 0.1    # w_t
        self.obs_std = 2.0     # v_t

    def transition_function(self, x_prev, u):
        """
        Updates the hidden mediator x (Insulin Level).
        u is binary (0 or 1).
        """
        noise = np.random.normal(0, self.proc_std)
        x_next = self.alpha * x_prev + self.beta * u + noise
        return max(0, x_next) # Insulin cannot be negative

    def observation_function(self, x):
        """
        Calculates the observed outcome y (Glucose).
        This is the nonlinear part.
        """
        noise = np.random.normal(0, self.obs_std)
        # Nonlinear relationship: Glucose drops as insulin x increases
        y = self.G_base / (1 + self.gamma * x) + noise
        return y

    def get_jacobian(self, x_pred):
        """
        Linearization for the Kalman Filter.
        Derivative of y with respect to x: H = dy/dx
        """
        # Using quotient rule derivative of G_base / (1 + gamma * x)
        H = - (self.G_base * self.gamma) / (1 + self.gamma * x_pred)**2
        return np.array([[H]])

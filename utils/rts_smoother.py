import numpy as np

def rts_smoother(z_filt, P_filt, z_pred, P_pred, lam):
    """
    Rauch-Tung-Striebel (RTS) backward smoothing pass.
 
    Takes the outputs of the Kalman filter and refines the state estimates
    using future observations.
 
    Parameters
    ----------
    z_filt : array-like, shape (T,)
        Filtered state means from the Kalman filter.
    P_filt : array-like, shape (T,)
        Filtered state variances from the Kalman filter.
    z_pred : array-like, shape (T,)
        Predicted state means from the Kalman filter.
    P_pred : array-like, shape (T,)
        Predicted state variances from the Kalman filter.
    lam : float
        State transition coefficient on z_{t-1} (must match the Kalman filter).
 
    Returns
    -------
    z_smooth : np.ndarray, shape (T,)
        Smoothed state means E[z_t | y_{1:T}].
    P_smooth : np.ndarray, shape (T,)
        Smoothed state variances Var[z_t | y_{1:T}].
    """
    z_filt = np.asarray(z_filt)
    P_filt = np.asarray(P_filt)
    z_pred = np.asarray(z_pred)
    P_pred = np.asarray(P_pred)
 
    T = len(z_filt)
    z_smooth = np.zeros(T)
    P_smooth = np.zeros(T)
 
    # Initialise at the last filtered estimate
    z_smooth[-1] = z_filt[-1]
    P_smooth[-1] = P_filt[-1]
 
    for t in range(T - 2, -1, -1):
        G = (P_filt[t] * lam) / P_pred[t + 1]
        z_smooth[t] = z_filt[t] + G * (z_smooth[t + 1] - z_pred[t + 1])
        P_smooth[t] = P_filt[t] + G ** 2 * (P_smooth[t + 1] - P_pred[t + 1])
 
    return z_smooth, P_smooth
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import norm

def compute_predicted_means_vars(estimated_params, candidate, z_pred_means, z_pred_vars):
    """
    Compute the predicted means and variances of y_{t+1} under both group models given the candidate x_{t+1}

    Parameters
    ----------
    estimated_params : dict
        Estimated parameters for both groups. Should contain keys 0 and 1, each mapping to a dict of parameters.
    candidate : float
        The candidate input signal x_{t+1} for which we want to compute the predicted means and variances of y_{t+1}.
    z_pred_means : dict
        Predicted means of the latent state z_{t} under both group models. Should contain
        keys 0 and 1, each mapping to a float.
    z_pred_vars : dict
        Predicted variances of the latent state z_{t} under both group models. Should
        contain keys 0 and 1, each mapping to a float.
    
    Returns
    -------
    y_pred_means : dict
        Predicted means of y_{t+1} under both group models. Should contain keys 0 and 1, each mapping to a float.
    y_pred_vars : dict
        Predicted variances of y_{t+1} under both group models. Should contain keys 0 and 1, each mapping to a float.
    """
    y_pred_means = {}
    y_pred_vars = {}

    for i in range(2):  # For both groups 0 and 1
        params = estimated_params[i]
        alpha, lam, beta, gamma = params["alpha"], params["lam"], params["beta"], params["gamma"]
        sigma_w, sigma_e = params["sigma_w"], params["sigma_e"]

        # Predicted mean and variance of z_{t+1} given the candidate x_{t+1}
        z_pred_mean = alpha * candidate + lam * z_pred_means[i]
        z_pred_var = lam**2 * z_pred_vars[i] + sigma_w**2

        # Predicted mean and variance of y_{t+1} given the predicted z_{t+1}
        y_pred_mean = beta * z_pred_mean + gamma * candidate
        y_pred_var = beta**2 * z_pred_var + sigma_e**2

        y_pred_means[i] = y_pred_mean
        y_pred_vars[i] = y_pred_var

    return y_pred_means, y_pred_vars

def draw_samples_from_mixture_distribution(y_pred_means, y_pred_vars, prior_1, samples_size):
    """
    Draw samples from the mixture distribution of y_{t+1} given the predicted means and variances
        
    Parameters
    ----------
    y_pred_means : dict
        Predicted means of y_{t+1} under both group models. Should contain keys 0 and 1, each mapping to a float.
    y_pred_vars : dict
        Predicted variances of y_{t+1} under both group models. Should contain keys 0 and 1, each mapping to a float.
    prior_1 : float
        Prior probability of belonging to group 1, P(c_n = 1).
    samples_size : int
        Number of samples to draw from the mixture distribution.

    Returns
    -------
    group_label_samples : np.ndarray, shape (samples_size,)
        Sampled group labels (0 or 1) based on the prior probabilities.
    y_samples : np.ndarray, shape (samples_size,)
        Sampled y_{t+1} values from the mixture distribution based on the sampled group
    """
    # sample group labels based on the prior probabilities
    group_label_samples = bernoulli.rvs(prior_1, size=samples_size)

    # sample y_{t+1} from the predicted distributions based on the sampled group labels
    y_samples = np.array([
        norm.rvs(loc=y_pred_means[group_label], scale=np.sqrt(y_pred_vars[group_label]))
        for group_label in group_label_samples
    ])
    return group_label_samples, y_samples

def evaluate_Shannon_entropy(x_candidate, y_pred_means, y_pred_vars, group_label_samples, y_samples, prior_1):
    """
    Evaluate the Shannon entropy for a candidate x_{t+1} based on the sampled group labels and y_{t+1} values.
    
    Parameters
    ----------
    x_candidate : float
        The candidate input signal x_{t+1} for which we want to evaluate the Shannon entropy.
    y_pred_means : dict
        Predicted means of y_{t+1} under both group models. Should contain keys 0 and 1, each mapping to a float.
    y_pred_vars : dict
        Predicted variances of y_{t+1} under both group models. Should contain keys 0 and 1, each mapping to a float.
    group_label_samples : np.ndarray, shape (samples_size,)
        Sampled group labels (0 or 1) based on the prior probabilities.
    y_samples : np.ndarray, shape (samples_size,)
        Sampled y_{t+1} values from the mixture distribution based on the sampled group labels.
    prior_1 : float
        Prior probability of belonging to group 1, P(c_n = 1).
    """
    # evaluate the Shannon entropy for each sample
    # compute probability of y_t+1 given the sampled group labels, candidate x, and the predicted mean of y_t+1
    # p(y^(s)_t+1 | c^(s)_n, x_candidate, y_1:t) = (1/sqrt(2 * pi * pred_var^2)) * exp(- (y^(s)_t+1 - pred_mean)^2 / (2 * pred_var^2))
    # for groups 0 and 1
    # for group 0: p(y^(s)_t+1 | c^(s)_n=0, x_candidate, y_1:t) = (1/sqrt(2 * pi * pred_var_0^2)) * exp(- (y^(s)_t+1 - pred_mean_0)^2 / (2 * pred_var_0^2)) 

    probs_y_samples_group0 = np.array([
        (1 / np.sqrt(2 * np.pi * y_pred_vars[0])) *
        np.exp(- (y_sample - y_pred_means[0])**2 / (2 * y_pred_vars[0]))
        for y_sample in y_samples
    ])
    # for group 1: p(y^(s)_t+1 | c^(s)_n=1, x_candidate, y_1:t) = (1/sqrt(2 * pi * pred_var_1^2)) * exp(- (y^(s)_t+1 - pred_mean_1)^2 / (2 * pred_var_1^2))
    probs_y_samples_group1 = np.array([
        (1 / np.sqrt(2 * np.pi * y_pred_vars[1])) *
        np.exp(- (y_sample - y_pred_means[1])**2 / (2 * y_pred_vars[1]))
        for y_sample in y_samples
    ])
    # compute the posterior for each sample
    # P(c^(s)_n | y^(s)_t+1, x_candidate, y_1:t) = p(y^(s)_t+1 | c^(s)_n, x_candidate, y_1:t) * P(c^(s)_n) / sum_{c_n} p(y^(s)_t+1 | c_n, x_candidate, y_1:t) * P(c_n)
    posterior_samples = np.array([
        (probs_y_samples_group0[s] * (1 - prior_1)) / 
        (probs_y_samples_group0[s] * (1 - prior_1) + probs_y_samples_group1[s] * prior_1)
        if group_label_samples[s] == 0 else
        (probs_y_samples_group1[s] * prior_1) / 
        (probs_y_samples_group0[s] * (1 - prior_1) + probs_y_samples_group1[s] * prior_1)
        for s in range(len(y_samples))
    ])

    # compute the Shannon entropy for each sample
    # +1e-10 is added to avoid log(0)
    shannon_entropy_samples = -(posterior_samples * np.log(posterior_samples + 1e-10) + (1 - posterior_samples) * np.log(1 - posterior_samples + 1e-10))

    # return average Shannon entropy across all samples
    return np.mean(shannon_entropy_samples)


def select_next_x_uncertainty_sampling(x_candidates, estimated_params, z_pred_means, z_pred_vars, prior_1=0.5, samples_size=50):
    """
    Choose the next input signal x_{t+1} from a set of candidates based on the current patient data and estimated parameters.

    Parameters
    ----------
    x_candidates : array-like, shape (n_candidates,)
        Candidate input signals x_{t+1} for the patient.
    estimated_params : dict
        Estimated parameters for both groups.
    z_pred_means : dict
        Predicted means of the latent state z_{t} under both group models. Should contain keys 0 and 1, each mapping to a float.
    z_pred_vars : dict
        Predicted variances of the latent state z_{t} under both group models. Should contain keys 0 and 1, each mapping to a float.
    prior_1 : float
        Prior probability of belonging to group 1, P(c_n = 1).
    samples_size : int
        Number of samples to draw from the mixture distribution for uncertainty estimation.

    Returns
    -------
    best_candidate : float
        The candidate input signal that maximizes uncertainty.
    """
    best_candidate = None
    max_entropy = -np.inf

    for candidate in x_candidates:
        # compute the predicted means and variances of y_{t+1} under both group models given the candidate x_{t+1}
        y_pred_means, y_pred_vars = compute_predicted_means_vars(estimated_params, candidate, z_pred_means, z_pred_vars)

        # draw samples from the mixture distribution of y_{t+1} given the predicted means and variances
        group_label_samples, y_samples = draw_samples_from_mixture_distribution(y_pred_means, y_pred_vars, prior_1, samples_size)

        # evaluate the Shannon entropy for the candidate x_{t+1}
        entropy = evaluate_Shannon_entropy(candidate, y_pred_means, y_pred_vars, group_label_samples, y_samples, prior_1)

        if entropy > max_entropy:
            max_entropy = entropy
            best_candidate = candidate
    return best_candidate


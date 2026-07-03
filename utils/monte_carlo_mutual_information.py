import numpy as np
from utils.monte_carlo_uncertainty_sampling import compute_predicted_means_vars, draw_samples_from_mixture_distribution

def compute_analytic_noise_entropy(y_pred_vars, prior_1):
    """
    Compute the analytic noise entropy of y_{t+1} given the predicted variances under both group models and the prior probability of group 1.

    Parameters
    ----------
    y_pred_vars : dict
        Predicted variances of y_{t+1} under both group models. Should contain keys 0 and 1, each mapping to a float.
    prior_1 : float
        Prior probability of group 1.

    Returns
    -------
    noise_entropy : float
        The analytic noise entropy of y_{t+1}.
    """
    prior_0 = 1 - prior_1
    priors = np.array([prior_0, prior_1])
    group_entropies = np.array([
        0.5 * np.log(2 * np.pi * np.e * y_pred_vars[i]) for i in range(2)
    ])
    noise_entropy = np.sum(priors * group_entropies) 
    return noise_entropy

def evaluate_mutual_information(y_pred_means, y_pred_vars, y_samples, prior_1):
    """
    Evaluate the mutual information between the candidate x_{t+1} and the predicted y_{t+1} values based on the sampled y_{t+1} values.

    Parameters
    ----------
    y_pred_means : dict
        Predicted means of y_{t+1} under both group models. Should contain keys 0 and 1, each mapping to a float.
    y_pred_vars : dict
        Predicted variances of y_{t+1} under both group models. Should contain keys 0 and 1, each mapping to a float.
    y_samples : np.ndarray, shape (samples_size,)
        Sampled y_{t+1} values from the mixture distribution based on the sampled group labels.
    prior_1 : float
        Prior probability of belonging to group 1, P(c_n = 1).
    """
    # evaluate predictive entropy for all samples
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

    probs_y_weighted = np.array([
        probs_y_samples_group0[s] * (1 - prior_1) + probs_y_samples_group1[s] * prior_1
        for s in range(len(y_samples))
    ])

    # approximate predictive entropy
    predictive_entropy = -np.mean(np.log(probs_y_weighted))

    noise_entropy = compute_analytic_noise_entropy(y_pred_vars, prior_1)

    estimated_mutual_information = predictive_entropy - noise_entropy

    return estimated_mutual_information


def select_next_x_mutual_information(x_candidates, estimated_params, z_pred_means, z_pred_vars, prior_1=0.5, samples_size=50):
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
        Number of samples to draw from the mixture distribution for mutual information estimation.

    Returns
    -------
    best_candidate : float
        The candidate input signal that maximizes mutual information.
    """
    best_candidate = None
    max_mutual_information = -np.inf

    for candidate in x_candidates:
        # compute the predicted means and variances of y_{t+1} under both group models given the candidate x_{t+1}
        y_pred_means, y_pred_vars = compute_predicted_means_vars(estimated_params, candidate, z_pred_means, z_pred_vars)

        # draw samples from the mixture distribution of y_{t+1} given the predicted means and variances
        _, y_samples = draw_samples_from_mixture_distribution(y_pred_means, y_pred_vars, prior_1, samples_size)

        # evaluate the mutual information for the candidate x_{t+1}
        mutual_information = evaluate_mutual_information(y_pred_means, y_pred_vars, y_samples, prior_1)

        if mutual_information > max_mutual_information:
            max_mutual_information = mutual_information
            best_candidate = candidate
    return best_candidate


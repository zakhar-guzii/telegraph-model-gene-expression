import numpy as np


def sample_steady_state(k_on, k_off, k_syn, k_deg, n_rep) -> dict:
    """Task 8: Draws independent samples from the steady-state joint distribution
    of the telegraph model and computes their empirical sample moments.

    Uses the exact hierarchical Beta-Poisson-Bernoulli mixture representation
    of the steady-state system to efficiently generate paired samples of gene
    states and RNA counts without performing a costly time-course simulation.

    Args:
        k_on (float): Rate constant for gene activation (OFF -> ON).
        k_off (float): Rate constant for gene inactivation (ON -> OFF).
        k_syn (float): RNA synthesis rate (only when gene is ON).
        k_deg (float): RNA degradation rate per single molecule.
        n_rep (int): Total number of independent steady-state samples to draw.

    Returns:
        dict: A dictionary containing the generated samples and their computed moments:
            - "G_samples" : 1D array of size (n_rep,) containing binary gene states.
            - "R_samples" : 1D array of size (n_rep,) containing RNA molecule counts.
            - "mu_G"      : Steady-state sample mean of the gene state.
            - "mu_R"      : Steady-state sample mean of the RNA count.
            - "sigma_G"   : Steady-state sample standard deviation of the gene state.
            - "sigma_R"   : Steady-state sample standard deviation of the RNA count.
            - "cov_RG"    : Steady-state sample covariance between RNA and gene state.
    """
    a = k_on / k_deg
    b = k_off / k_deg

    x = np.random.beta(a, b, size=n_rep)

    G_samples = (np.random.random(size=n_rep) < x).astype(int)

    lambda_R = x * (k_syn / k_deg)
    R_samples = np.random.poisson(lambda_R)

    mu_G = np.mean(G_samples)
    mu_R = np.mean(R_samples)

    sigma_G = np.std(G_samples, ddof=1)
    sigma_R = np.std(R_samples, ddof=1)

    cov_matrix = np.cov(G_samples, R_samples, ddof=1)
    cov_RG = cov_matrix[0, 1]

    return {
        "G_samples": G_samples,
        "R_samples": R_samples,
        "mu_G": mu_G,
        "mu_R": mu_R,
        "sigma_G": sigma_G,
        "sigma_R": sigma_R,
        "cov_RG": cov_RG,
    }
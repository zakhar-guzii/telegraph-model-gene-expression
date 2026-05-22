import numpy as np


def simulate_telegraph(
    k_on, k_off, k_syn, k_deg, t0, g0, r0, n_sim, n_rep
) -> np.ndarray:
    """Task 1: Simulates stochastic gene expression trajectories using the Gillespie SSA.

    Implements the continuous-time Markov chain to track the binary state of a gene
    (ON/OFF) and discrete RNA molecule counts.

    Args:
        k_on (float): Rate constant for gene activation (OFF -> ON).
        k_off (float): Rate constant for gene inactivation (ON -> OFF).
        k_syn (float): RNA synthesis rate (only when gene is ON).
        k_deg (float): RNA degradation rate per single molecule.
        t0 (float): Initial simulation time.
        g0 (int): Initial gene state; 0 = OFF, 1 = ON.
        r0 (int): Initial number of RNA molecules.
        n_sim (int): Total number of reaction events to simulate per trajectory.
        n_rep (int): Total number of independent parallel trajectories.

    Returns:
        np.ndarray: A 3D float/int array of shape (n_sim + 1, n_rep, 3), where:
            - [i, j, 0] = Simulation time at step i, trajectory j.
            - [i, j, 1] = Gene state (0 or 1) at step i, trajectory j.
            - [i, j, 2] = RNA  count at step i, trajectory j.
    """
    data = np.zeros((n_sim + 1, n_rep, 3))

    data[0, :, 0] = t0
    data[0, :, 1] = g0
    data[0, :, 2] = r0

    # Outer loop: each trajectory is independent
    for j in range(n_rep):
        t = t0
        g = g0
        r = r0

        # Inner loop: advance one reaction event per iteration.
        for i in range(1, n_sim + 1):
            a1 = k_on * (1 - g)
            a2 = k_off * g
            a3 = k_syn * g
            a4 = k_deg * r
            a0 = a1 + a2 + a3 + a4

            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)

            tau = (1 / a0) * np.log(1 / r1)
            threshold = r2 * a0

            if threshold < a1:
                g = 1
            elif threshold < (a1 + a2):
                g = 0
            elif threshold < (a1 + a2 + a3):
                r += 1
            else:
                r -= 1

            t += tau
            data[i, j, 0] = t
            data[i, j, 1] = g
            data[i, j, 2] = r

    return data


def compute_sample_moments(data) -> np.array:
    """Task 2: sapmle statistics of gene state and RNA count across trajectories

    For each recorded time step, calculates the cross-trajectory mean, standard
    deviation, and covariance of the gene state G and RNA count R.

     Args:
        data (np.ndarray): Simulation output of shape ``(n_sim + 1, n_rep, 3)``
            as returned by :func:`simulate_telegraph`, where axis-2 contains:

            - [:, :, 0] : Simulation time.
            - [:, :, 1] : Gene state (0 = OFF, 1 = ON).
            - [:, :, 2] : RNA molecule count.

    Returns:
        dict: A dictionary with one 1-D array of length ``n_sim + 1`` per key:

            - "time"    : Mean simulation time across trajectories at each step.
            - "mu_G"    : Sample mean of gene state  E[G].
            - "mu_R"   : Sample mean of RNA count   E[R].
            - "sigma_G" : Sample std  of gene state  sqrt(Var[G]).
            - "sigma_R" : Sample std  of RNA count   sqrt(Var[R]).
            - "cov_RG"  : Sample covariance          Cov(R, G).
    """
    t_data = data[:, :, 0]
    G = data[:, :, 1]
    R = data[:, :, 2]

    mean_t = np.mean(t_data, axis=1)
    mu_G = np.mean(G, axis=1)
    mu_R = np.mean(R, axis=1)

    sigma_G = np.std(G, axis=1, ddof=1)  # Is it need to do Bessel's correction?
    sigma_R = np.std(R, axis=1, ddof=1)

    cov_RG = np.sum((G - mu_G[:, None]) * (R - mu_R[:, None]), axis=1) / (
        G.shape[1] - 1
    )

    return {
        "time": mean_t,
        "mu_G": mu_G,
        "mu_R": mu_R,
        "sigma_G": sigma_G,
        "sigma_R": sigma_R,
        "cov_RG": cov_RG,
    }


def show_sample_moments():
    pass

import numpy as np


def simulate_telegraph(
    k_on, k_off, k_syn, k_deg, t0, g0, r0, n_sim, n_rep
) -> np.ndarray:
    """Simulates stochastic gene expression trajectories using the Gillespie SSA.

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

    for j in range(n_rep):
        t = t0
        g = g0
        r = r0

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


def compute_sample_moments(data) -> dict:
    """Computes sample statistics of gene state and RNA count across trajectories

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


def compute_time_grid_moments(data, n_points=500):
    """Compute moments on a fixed time grid (unbiased estimator).

    Unlike :func:`compute_sample_moments`, which averages at step indices
    (biased because more events occur when the gene is ON), this function
    interpolates each trajectory to a common set of time points using
    piece-wise constant interpolation, then averages across trajectories.

    Args:
        data (np.ndarray): Simulation output of shape ``(n_sim + 1, n_rep, 3)``
            as returned by :func:`simulate_telegraph`.
        n_points (int): Number of equally-spaced time points in the grid.

    Returns:
        dict: Same keys as :func:`compute_sample_moments`:
            ``time``, ``mu_G``, ``mu_R``, ``sigma_G``, ``sigma_R``, ``cov_RG``.
    """
    n_rep = data.shape[1]

    # Common time range: from 0 to the earliest trajectory end (with 2% margin)
    t_max = data[-1, :, 0].min()
    t_grid = np.linspace(0, t_max * 0.98, n_points)

    G_grid = np.zeros((n_points, n_rep))
    R_grid = np.zeros((n_points, n_rep))

    for j in range(n_rep):
        t_j = data[:, j, 0]
        G_j = data[:, j, 1]
        R_j = data[:, j, 2]
        # Piece-wise constant: state just before each grid point
        idx = np.searchsorted(t_j, t_grid, side="right") - 1
        idx = np.clip(idx, 0, len(t_j) - 1)
        G_grid[:, j] = G_j[idx]
        R_grid[:, j] = R_j[idx]

    mu_G = G_grid.mean(axis=1)
    mu_R = R_grid.mean(axis=1)

    return {
        "time": t_grid,
        "mu_G": mu_G,
        "mu_R": mu_R,
        "sigma_G": G_grid.std(axis=1, ddof=1),
        "sigma_R": R_grid.std(axis=1, ddof=1),
        "cov_RG": np.array([
            np.cov(G_grid[i], R_grid[i])[0, 1] for i in range(n_points)
        ]),
    }

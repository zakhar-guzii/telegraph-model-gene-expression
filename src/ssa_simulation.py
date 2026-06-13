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
        np.ndarray: A 3D array of shape (n_sim + 1, n_rep, 3), where:
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


def compute_sample_moments(data, t_end=None, n_grid=1000, time_step=None) -> dict:
    """Computes sample statistics of gene state and RNA count on a real-time grid.

    Each Gillespie trajectory advances by a random time step, so the same step
    index corresponds to different wall-clock times across trajectories. To get
    moments at genuine, evenly-spaced times, every trajectory is first resampled
    onto a shared uniform time grid using zero-order hold (ZOH): the state is held
    constant between reaction events, which is exact for the piecewise-constant SSA
    process. Cross-trajectory mean, std, and covariance are then taken at each grid
    time.

    Args:
        data (np.ndarray): Simulation output of shape (n_sim + 1, n_rep, 3) from
            :func:`simulate_telegraph` (``[i, j, 0]`` time, ``[i, j, 1]`` gene state,
            ``[i, j, 2]`` RNA count).
        t_end (float, optional): End time of the shared grid. Defaults to the
            earliest-finishing trajectory's last time (no extrapolation). Set it equal
            to the ODE solver's ``t_end`` for aligned SSA-vs-ODE comparison; trajectories
            that ended earlier hold their last value to ``t_end``.
        n_grid (int): Number of evenly-spaced grid points (default 1000, matching the
            ODE ``t_eval`` grid). Ignored when ``time_step`` is given.
        time_step (float, optional): Grid spacing in seconds. When given, overrides
            ``n_grid`` with ``n_grid = int((t_end - t0) / time_step) + 1`` and the grid
            steps in exact increments of ``time_step``.

    Returns:
        dict: 1-D arrays of length n_grid:
            - "time"    : The evenly-spaced real-time grid.
            - "mu_G"    : Sample mean of gene state E[G].
            - "mu_R"    : Sample mean of RNA count E[R].
            - "sigma_G" : Sample standard deviation of gene state.
            - "sigma_R" : Sample standard deviation of RNA count.
            - "cov_RG"  : Sample covariance Cov(R, G).
    """
    t_data = data[:, :, 0]
    G = data[:, :, 1]
    R = data[:, :, 2]

    n_records, n_rep = G.shape

    t0 = t_data[0, 0]  # initial time, identical across trajectories

    # Shared grid end: default to the earliest-finishing trajectory.
    if t_end is None:
        t_end = t_data[-1, :].min()

    # Build the uniform real-time grid.
    if time_step is not None:
        n_grid = int((t_end - t0) / time_step) + 1
        grid = t0 + np.arange(n_grid) * time_step
    else:
        grid = np.linspace(t0, t_end, n_grid)

    # Zero-order-hold resampling: for each grid time, take the most recent event value.
    G_grid = np.empty((grid.size, n_rep))
    R_grid = np.empty((grid.size, n_rep))
    for j in range(n_rep):
        idx = np.searchsorted(t_data[:, j], grid, side="right") - 1
        idx = np.clip(idx, 0, n_records - 1)  # hold first value at t0, last value past end
        G_grid[:, j] = G[idx, j]
        R_grid[:, j] = R[idx, j]

    # Cross-trajectory moments at each grid time.
    mu_G = np.mean(G_grid, axis=1)
    mu_R = np.mean(R_grid, axis=1)
    sigma_G = np.std(G_grid, axis=1, ddof=1)
    sigma_R = np.std(R_grid, axis=1, ddof=1)
    cov_RG = np.sum(
        (G_grid - mu_G[:, None]) * (R_grid - mu_R[:, None]), axis=1
    ) / (n_rep - 1)

    return {
        "time": grid,
        "mu_G": mu_G,
        "mu_R": mu_R,
        "sigma_G": sigma_G,
        "sigma_R": sigma_R,
        "cov_RG": cov_RG,
    }

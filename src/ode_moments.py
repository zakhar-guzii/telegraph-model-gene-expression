import numpy as np
from scipy.integrate import solve_ivp


def solve_ode_moments(k_on, k_off, k_syn, k_deg, t0, g0, r0, t_end):
    """Solve the closed ODE system for the first and second moments of a
    two-state (telegraph) gene expression model.

    Returns:
        t: 1-D array of 1000 time points.
        y: 2-D array (4 x 1000) — rows are μ_G, μ_R, σ²_R, C_RG.
    """
    def rhs(t, y):
        mu_g, mu_r, sigma2_r, c_rg = y

        sigma2_g = mu_g * (1.0 - mu_g) 

        d_mu_g     = k_on * (1.0 - mu_g) - k_off * mu_g
        d_mu_r     = k_syn * mu_g - k_deg * mu_r
        d_sigma2_r = 2.0 * k_syn * c_rg - 2.0 * k_deg * sigma2_r + k_syn * mu_g + k_deg * mu_r
        d_c_rg     = k_syn * sigma2_g - (k_on + k_off + k_deg) * c_rg

        return [d_mu_g, d_mu_r, d_sigma2_r, d_c_rg]

    t_eval = np.linspace(t0, t_end, 1000)
    sol = solve_ivp(rhs, (t0, t_end), [g0, r0, 0.0, 0.0], t_eval=t_eval, method="RK45")

    return sol.t, sol.y
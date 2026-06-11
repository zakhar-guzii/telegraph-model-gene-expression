# Telegraph Model of Gene Expression

A computational study of the two-state (telegraph) model of stochastic gene
expression. The project implements and cross-validates three complementary
descriptions of the same system: an exact stochastic simulation (Gillespie SSA),
a closed system of moment ordinary differential equations (ODEs), and a direct
sampler for the exact steady-state distribution. Empirical moments from the
stochastic simulation are compared against the deterministic moment dynamics and
the analytical steady state to confirm that all three agree.

This repository is the deliverable for Project 8 of the *Experimental
Mathematics* course (KSE, Applied Mathematics, Spring 2026).

## Problem Formulation

The telegraph model describes a single gene that switches stochastically between
an inactive (OFF) and an active (ON) state and transcribes mRNA only while
active. The system is defined by four reactions:

| Reaction            | Effect              | Propensity        |
| ------------------- | ------------------- | ----------------- |
| Gene activation     | OFF -> ON           | `k_on * (1 - G)`  |
| Gene inactivation   | ON -> OFF           | `k_off * G`       |
| RNA synthesis       | R -> R + 1 (ON only)| `k_syn * G`       |
| RNA degradation     | R -> R - 1          | `k_deg * R`       |

Here `G` is the binary gene state (0 = OFF, 1 = ON) and `R` is the mRNA molecule
count. The four rate constants `k_on`, `k_off`, `k_syn`, `k_deg` fully specify
the dynamics. The quantities of interest are the time evolution and steady-state
values of the first and second moments: the mean gene activity, the mean mRNA
count, their variances, and the gene-RNA covariance.

## Approaches

The project investigates the model from three independent directions.

### 1. Stochastic Simulation Algorithm (SSA)

An exact event-driven Gillespie simulation that generates individual
trajectories of `(G, R)` over time. Ensemble averaging across many independent
trajectories yields empirical estimates of the moment dynamics.

### 2. Moment ODEs

A closed system of four ODEs for the moments `(mu_G, mu_R, sigma2_R, C_RG)`.
Because the gene state is binary, its variance closes exactly as
`sigma2_G = mu_G * (1 - mu_G)`, which makes the moment equations exact rather
than an approximation. The system is integrated numerically with an explicit
Runge-Kutta scheme.

### 3. Exact Steady-State Sampling

A direct sampler for the steady-state joint distribution based on its
hierarchical Beta-Poisson-Bernoulli mixture representation. This draws
independent steady-state samples of `(G, R)` without running a time course,
providing an efficient analytical reference for the long-time limit.

## Repository Structure

```
.
├── src/
│   ├── ssa_simulation.py      # Gillespie SSA and cross-trajectory sample moments
│   ├── ode_moments.py         # Closed moment ODE system and its solver
│   ├── steady_state.py        # Exact steady-state sampler (Beta-Poisson-Bernoulli)
│   ├── ssa_visualization.py   # Plotly figures for SSA moment dynamics
│   └── ode_visualization.py   # Plotly figures for ODE moment dynamics
└── notebooks/
    ├── ssa_parameter_exploration.ipynb  # SSA behaviour across parameter regimes
    ├── ssa_vs_ode.ipynb                 # Side-by-side SSA vs ODE comparison
    └── validation.ipynb                 # Convergence, steady-state, and edge-case checks
```

## Modules

### `src/ssa_simulation.py`

- `simulate_telegraph(k_on, k_off, k_syn, k_deg, t0, g0, r0, n_sim, n_rep)` —
  runs `n_rep` independent Gillespie trajectories of `n_sim` reaction events
  each. Returns a `(n_sim + 1, n_rep, 3)` array of time, gene state, and RNA
  count.
- `compute_sample_moments(data)` — computes cross-trajectory mean, standard
  deviation, and covariance of `G` and `R` at each recorded step.

### `src/ode_moments.py`

- `solve_ode_moments(k_on, k_off, k_syn, k_deg, t0, g0, r0, t_end)` — integrates
  the closed moment ODE system and returns the time grid and the trajectories of
  `(mu_G, mu_R, sigma2_R, C_RG)`.

### `src/steady_state.py`

- `sample_steady_state(k_on, k_off, k_syn, k_deg, n_rep)` — draws independent
  samples from the exact steady-state distribution and returns the samples
  together with their empirical moments.

### `src/ssa_visualization.py`, `src/ode_visualization.py`

- `show_sample_moments(moments, title)` and
  `show_ode_moments(t, y, title, analytical)` — render three-panel Plotly
  figures showing the mean gene activity, mean mRNA count with fluctuation
  bands, and the gene-RNA covariance.

## Notebooks

- **`ssa_parameter_exploration.ipynb`** — studies how the SSA moments respond to
  the switching regime (slow vs. fast promoter switching), the expression level
  (`k_syn / k_deg`), the initial conditions, and the numerical parameters
  (ensemble size and run length).
- **`ssa_vs_ode.ipynb`** — overlays the stochastic SSA estimates and the
  deterministic ODE moments for matched parameter sets, confirming that the SSA
  means converge to the exact moment dynamics.
- **`validation.ipynb`** — three validation checks: (1) convergence of the
  steady-state sampler to the ODE prediction as the sample size grows, with the
  expected `1/sqrt(N)` error decay; (2) agreement of the sampler with the settled
  ODE steady state across parameter regimes; (3) analytical edge cases
  (constitutive expression with `k_off = 0`, giving Poisson statistics; and no
  degradation with `k_deg = 0`, giving linear mRNA growth).

## Requirements

- Python 3.11 or newer
- NumPy
- SciPy
- Plotly
- Jupyter (to run the notebooks)

Install the dependencies with:

```bash
pip install numpy scipy plotly jupyter
```

## Usage

The source modules are plain functions and can be imported directly. A minimal
example comparing the stochastic and deterministic moment dynamics:

```python
import sys
sys.path.append("src")

from ssa_simulation import simulate_telegraph, compute_sample_moments
from ode_moments import solve_ode_moments
from ssa_visualization import show_sample_moments
from ode_visualization import show_ode_moments

params = dict(k_on=0.5, k_off=0.5, k_syn=20.0, k_deg=1.0)

# Stochastic simulation
data = simulate_telegraph(**params, t0=0, g0=0, r0=0, n_sim=2000, n_rep=1000)
ssa_moments = compute_sample_moments(data)
show_sample_moments(ssa_moments)

# Deterministic moment ODEs
t, y = solve_ode_moments(**params, t0=0, g0=0, r0=0, t_end=10)
show_ode_moments(t, y)
```

For the full analysis, open the notebooks in the `notebooks/` directory and run
them top to bottom.

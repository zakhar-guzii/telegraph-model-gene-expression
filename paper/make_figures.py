r"""Generate the PNG figures used in the paper.

All figures are written to ``paper/figures`` (the folder the LaTeX source pulls
from via ``\graphicspath``). Run with::

    python paper/make_figures.py

Notes
-----
* The SSA-vs-ODE overlay is built by the reusable :func:`composite_ssa_ode`
  helper below. It mirrors the interactive ``show_combined_moments`` in
  ``src/comparison_visualization.py`` (SSA in green with an empirical +/-sigma
  band, ODE in dashed blue) but returns a *saved* figure instead of calling
  ``fig.show()``, so it can run headless. It additionally overlays the
  analytical standard deviation ``mu_R +/- sigma_R`` taken straight from the
  moment-ODE solver on top of the stochastic band.
"""
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO, "src"))

from ssa_simulation import simulate_telegraph, compute_sample_moments
from ode_moments import solve_ode_moments
from steady_state import sample_steady_state

OUT = os.path.join(REPO, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

np.random.seed(20260617)

PARAMS = dict(k_on=0.5, k_off=0.5, k_syn=20.0, k_deg=1.0)
T_END = 10.0
N_SIM = 2000
N_REP = 1000

# --- palette ---------------------------------------------------------------
SSA_COLOR = "#10B981"                     # SSA mean lines (green)
SSA_FILL = "rgba(16, 185, 129, 0.12)"     # SSA empirical +/-sigma band
ODE_COLOR = "#1E40AF"                      # ODE mean lines (dark blue)
SIGMA_COLOR = "#F59E0B"                    # analytical ODE +/-sigma_R (amber)

GENE_COLOR = "#3B82F6"
GENE_FILL = "rgba(59, 130, 246, 0.07)"
RNA_COLOR = "#EF4444"
RNA_FILL = "rgba(239, 68, 68, 0.07)"
COV_COLOR = "#8B5CF6"


def band(fig, x, lo, hi, row, fill=SSA_FILL):
    """Shade the region between ``lo`` and ``hi`` on a given subplot row."""
    fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=lo, fill="tonexty", fillcolor=fill,
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), row=row, col=1)


# ---------------------------------------------------------------------------
# Reusable composite: SSA sample moments overlaid on the deterministic ODEs.
# ---------------------------------------------------------------------------
def composite_ssa_ode(ssa, t_ode, y_ode, fname, title=None,
                      show_analytic_sigma_R=True):
    """Overlay SSA sample moments and ODE moments on one 3-panel figure and save it.

    The SSA mean (green) is drawn with its empirical +/-sigma_G / +/-sigma_R
    shading; the deterministic ODE mean is drawn on top as a dashed blue line so
    the two descriptions are mapped directly onto each other. When
    ``show_analytic_sigma_R`` is set, the analytical spread ``mu_R +/- sigma_R``
    computed by the moment-ODE solver (``sigma_R = sqrt(sigma2_R)``) is overlaid
    as amber dotted lines on top of the stochastic band, giving a direct visual
    check that the SSA scatter matches the exact variance dynamics.

    Args:
        ssa (dict): output of ``compute_sample_moments`` (time, mu_G, mu_R,
            sigma_G, sigma_R, cov_RG).
        t_ode (np.ndarray): ODE time grid from ``solve_ode_moments``.
        y_ode (np.ndarray): ODE rows (mu_G, mu_R, sigma2_R, C_RG).
        fname (str): output PNG filename (written to ``paper/figures``).
        title (str, optional): overall figure title.
        show_analytic_sigma_R (bool): overlay the ODE mu_R +/- sigma_R lines.
    """
    mu_G_ode, mu_R_ode, sigma2_R_ode, c_rg_ode = y_ode
    sigma_R_ode = np.sqrt(np.clip(sigma2_R_ode, 0.0, None))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("<b>A</b>  Mean gene activity ⟨G⟩ ± σ_G",
                        "<b>B</b>  Mean mRNA count ⟨R⟩ ± σ_R",
                        "<b>C</b>  Gene–RNA covariance C_RG(t)"),
        vertical_spacing=0.07)

    # --- Panel A: gene activity ---
    band(fig, ssa["time"], ssa["mu_G"] - ssa["sigma_G"],
         ssa["mu_G"] + ssa["sigma_G"], 1)
    fig.add_trace(go.Scatter(x=ssa["time"], y=ssa["mu_G"],
                             line=dict(color=SSA_COLOR, width=2), name="SSA"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t_ode, y=mu_G_ode,
                             line=dict(color=ODE_COLOR, width=2, dash="dash"),
                             name="ODE"), row=1, col=1)

    # --- Panel B: mRNA count, with analytical sigma_R overlay ---
    band(fig, ssa["time"], ssa["mu_R"] - ssa["sigma_R"],
         ssa["mu_R"] + ssa["sigma_R"], 2)
    fig.add_trace(go.Scatter(x=ssa["time"], y=ssa["mu_R"],
                             line=dict(color=SSA_COLOR, width=2),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_ode, y=mu_R_ode,
                             line=dict(color=ODE_COLOR, width=2, dash="dash"),
                             showlegend=False), row=2, col=1)
    if show_analytic_sigma_R:
        fig.add_trace(go.Scatter(x=t_ode, y=mu_R_ode + sigma_R_ode,
                                 line=dict(color=SIGMA_COLOR, width=1.8, dash="dot"),
                                 name="ODE μ_R ± σ_R"), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_ode, y=mu_R_ode - sigma_R_ode,
                                 line=dict(color=SIGMA_COLOR, width=1.8, dash="dot"),
                                 showlegend=False), row=2, col=1)

    # --- Panel C: covariance ---
    fig.add_trace(go.Scatter(x=ssa["time"], y=ssa["cov_RG"],
                             line=dict(color=SSA_COLOR, width=2),
                             showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_ode, y=c_rg_ode,
                             line=dict(color=ODE_COLOR, width=2, dash="dash"),
                             showlegend=False), row=3, col=1)

    fig.update_layout(template="plotly_white", width=1000, height=820,
                      title=(f"<b>{title}</b>" if title else None),
                      margin=dict(l=80, r=30, t=(90 if title else 60), b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0,
                                  bgcolor="rgba(255,255,255,0.6)"))
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="⟨G⟩", row=1, col=1)
    fig.update_yaxes(title_text="⟨R⟩", row=2, col=1)
    fig.update_yaxes(title_text="Cov(R, G)", row=3, col=1)
    fig.write_image(os.path.join(OUT, fname), scale=2)
    print("  saved", fname)


# ---------------------------------------------------------------------------
# Single-cell trajectory: "the life of a single gene" over time.
# ---------------------------------------------------------------------------
def single_cell_trajectory(fname, seed, title="The Life of a Single Gene",
                           n_sim=400, **overrides):
    """Save one raw SSA realization: gene ON/OFF switching and mRNA count vs time.

    Draws a single trajectory (``n_rep=1``) and plots the gene state as a step
    line (0 = OFF, 1 = ON) above the mRNA count, so the reader can see how bursts
    of transcription follow the ON episodes and how mRNA then decays back down.
    """
    np.random.seed(seed)
    params = dict(**PARAMS, t0=0.0, g0=0, r0=0, n_sim=n_sim, n_rep=1)
    params.update(overrides)
    data = simulate_telegraph(**params)

    t = data[:, 0, 0]
    g = data[:, 0, 1]
    r = data[:, 0, 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("<b>A</b>  Gene state (OFF ↔ ON)",
                                        "<b>B</b>  mRNA molecule count R(t)"),
                        vertical_spacing=0.10, row_heights=[0.35, 0.65])
    fig.add_trace(go.Scatter(x=t, y=g, line=dict(color=GENE_COLOR, width=1.6, shape="hv"),
                             fill="tozeroy", fillcolor=GENE_FILL, showlegend=False,
                             hovertemplate="G = %{y:.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=r, line=dict(color=RNA_COLOR, width=1.6, shape="hv"),
                             showlegend=False,
                             hovertemplate="R = %{y:.0f}<extra></extra>"), row=2, col=1)

    fig.update_layout(template="plotly_white", width=1000, height=560,
                      title=f"<b>{title}</b>", margin=dict(l=80, r=30, t=70, b=50),
                      hovermode="x")
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Gene state", row=1, col=1,
                     tickmode="array", tickvals=[0, 1], ticktext=["OFF", "ON"],
                     range=[-0.15, 1.15])
    fig.update_yaxes(title_text="mRNA count R", row=2, col=1)
    fig.write_image(os.path.join(OUT, fname), scale=2)
    print("  saved", fname)


# ---------------------------------------------------------------------------
# Run the two dynamic approaches for the symmetric reference parameters.
# ---------------------------------------------------------------------------
print("Running SSA ...")
data = simulate_telegraph(**PARAMS, t0=0, g0=0, r0=0, n_sim=N_SIM, n_rep=N_REP)
ssa = compute_sample_moments(data, t_end=T_END)

print("Solving moment ODEs ...")
t_ode, y_ode = solve_ode_moments(**PARAMS, t0=0, g0=0, r0=0, t_end=T_END)

# ---------------------------------------------------------------------------
# SSA vs ODE overlay (composite), with analytical sigma_R line.
#
# The separate single-vertex figures (SSA moments alone, ODE moments alone) are
# deliberately not drawn: this overlay contains both, from the same `ssa`
# ensemble and the same `y_ode` solution computed above.
# ---------------------------------------------------------------------------
print("Figure: ssa_vs_ode.png")
composite_ssa_ode(ssa, t_ode, y_ode, "ssa_vs_ode.png")

# ---------------------------------------------------------------------------
# Figure: single-cell trajectory ("life of a single gene").
# ---------------------------------------------------------------------------
print("Figure: single_cell_trajectory.png")
single_cell_trajectory("single_cell_trajectory.png", seed=42, n_sim=400)

# ---------------------------------------------------------------------------
# Figure 4: Steady-state convergence
# ---------------------------------------------------------------------------
print("Figure 4: steady_state_convergence.png")
mu_G_ss = PARAMS["k_on"] / (PARAMS["k_on"] + PARAMS["k_off"])
mu_R_ss = (PARAMS["k_syn"] / PARAMS["k_deg"]) * mu_G_ss

n_reps = np.unique(np.round(np.logspace(1, 5, 25)).astype(int))
n_trials = 40
errs = []
for n in n_reps:
    devs = []
    for _ in range(n_trials):
        s = sample_steady_state(**PARAMS, n_rep=int(n))
        devs.append(abs(s["mu_R"] - mu_R_ss))
    errs.append(np.mean(devs))
errs = np.array(errs)

# Reference 1/sqrt(N) line scaled to the data
ref = errs[0] * np.sqrt(n_reps[0]) / np.sqrt(n_reps)

fig = go.Figure()
fig.add_trace(go.Scatter(x=n_reps, y=errs, mode="markers+lines",
                         marker=dict(color=SSA_COLOR, size=7),
                         line=dict(color=SSA_COLOR, width=2),
                         name="Mean abs. deviation of μ_R"))
fig.add_trace(go.Scatter(x=n_reps, y=ref, mode="lines",
                         line=dict(color=ODE_COLOR, width=2, dash="dash"),
                         name="∝ 1/√N_rep reference"))
fig.update_layout(template="plotly_white", width=850, height=560,
                  margin=dict(l=80, r=30, t=40, b=60),
                  xaxis=dict(title="Number of samples N_rep", type="log"),
                  yaxis=dict(title="Mean absolute deviation of μ_R", type="log"),
                  legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                              bgcolor="rgba(255,255,255,0.6)"))
fig.write_image(os.path.join(OUT, "steady_state_convergence.png"), scale=2)

# ---------------------------------------------------------------------------
# Parameter-exploration figures (Notebook: ssa_parameter_exploration)
# One 3-panel SSA sample-moment figure per experiment, matching the notebook's
# show_sample_moments styling. Base parameters mirror run_experiment().
# ---------------------------------------------------------------------------
PARAM_BASE = dict(k_on=0.1, k_off=0.1, k_syn=20.0, k_deg=1.0,
                  t0=0.0, g0=0, r0=0, n_sim=4000, n_rep=1000)


def sample_moments_fig(m, title):
    """3-panel SSA sample-moment figure (gene state, RNA count, covariance)."""
    t = m["time"]
    f = make_subplots(rows=3, cols=1, shared_xaxes=True,
                      subplot_titles=("<b>a</b> Gene State ⟨G⟩ ± σ<sub>G</sub>",
                                      "<b>b</b> RNA Count ⟨R⟩ ± σ<sub>R</sub>",
                                      "<b>c</b> Gene–RNA Covariance"),
                      vertical_spacing=0.06)
    # Panel a: gene state with ±sigma band
    f.add_trace(go.Scatter(x=t, y=m["mu_G"] + m["sigma_G"], line=dict(width=0),
                           showlegend=False, hoverinfo="skip"), row=1, col=1)
    f.add_trace(go.Scatter(x=t, y=m["mu_G"] - m["sigma_G"], fill="tonexty",
                           fillcolor=GENE_FILL, line=dict(width=0),
                           showlegend=False, hoverinfo="skip"), row=1, col=1)
    f.add_trace(go.Scatter(x=t, y=m["mu_G"], line=dict(color=GENE_COLOR, width=2),
                           showlegend=False), row=1, col=1)
    # Panel b: RNA count with ±sigma band
    f.add_trace(go.Scatter(x=t, y=m["mu_R"] + m["sigma_R"], line=dict(width=0),
                           showlegend=False, hoverinfo="skip"), row=2, col=1)
    f.add_trace(go.Scatter(x=t, y=m["mu_R"] - m["sigma_R"], fill="tonexty",
                           fillcolor=RNA_FILL, line=dict(width=0),
                           showlegend=False, hoverinfo="skip"), row=2, col=1)
    f.add_trace(go.Scatter(x=t, y=m["mu_R"], line=dict(color=RNA_COLOR, width=2),
                           showlegend=False), row=2, col=1)
    # Panel c: covariance
    f.add_trace(go.Scatter(x=t, y=m["cov_RG"], line=dict(color=COV_COLOR, width=2),
                           showlegend=False), row=3, col=1)
    f.update_layout(template="plotly_white", width=1100, height=850,
                    title=f"<b>{title}</b>", margin=dict(l=85, r=30, t=70, b=50))
    f.update_xaxes(title_text="Time", row=3, col=1)
    f.update_yaxes(title_text="Active Fraction ⟨G⟩", row=1, col=1)
    f.update_yaxes(title_text="Mean mRNA Count ⟨R⟩", row=2, col=1)
    f.update_yaxes(title_text="Cov(R, G)", row=3, col=1)
    return f


def save_param_experiment(fname, title, seed, **overrides):
    np.random.seed(seed)
    params = {**PARAM_BASE, **overrides}
    d = simulate_telegraph(**params)
    m = compute_sample_moments(d, n_grid=1000)
    sample_moments_fig(m, title).write_image(os.path.join(OUT, fname), scale=2)
    print("  saved", fname)


print("Parameter-exploration figures ...")
save_param_experiment("param_switching_slow.png", "Slow Gene Switching Regime",
                      seed=1, k_on=0.05, k_off=0.05, k_syn=20.0)
save_param_experiment("param_switching_fast.png", "Fast Gene Switching Regime",
                      seed=2, k_on=10.0, k_off=10.0, k_syn=20.0)
# The expression, initial-condition and numerical-resolution experiments are
# reported through the F^ss / F-hat columns of tab:param-exp rather than as
# figures of their own; their moment traces live in Notebook 1. Their runs are
# still executed below by FANO_RUNS, which re-seeds each configuration itself.

# ---------------------------------------------------------------------------
# SSA vs ODE in the slow-switching regime (Notebook: ssa_vs_ode, Experiment 1a)
# ---------------------------------------------------------------------------
print("Figure: ssaode_slow.png")
np.random.seed(11)
slow = {**PARAM_BASE, "k_on": 0.05, "k_off": 0.05, "k_syn": 20.0}
data = simulate_telegraph(**slow)
t_end_slow = float(data[-1, :, 0].min())
m_slow = compute_sample_moments(data, t_end=t_end_slow, n_grid=1000)
t_ode_s, y_ode_s = solve_ode_moments(
    k_on=slow["k_on"], k_off=slow["k_off"], k_syn=slow["k_syn"], k_deg=slow["k_deg"],
    t0=slow["t0"], g0=slow["g0"], r0=slow["r0"], t_end=t_end_slow)
composite_ssa_ode(m_slow, t_ode_s, y_ode_s, "ssaode_slow.png")

# ---------------------------------------------------------------------------
# Edge case k_off=0: constitutive gene -> Poisson(k_syn/k_deg) (Notebook: validation)
# ---------------------------------------------------------------------------
print("Figure: validation_poisson.png")
np.random.seed(0)
k_syn_v, k_deg_v = 10.0, 1.0
mu_theory = k_syn_v / k_deg_v
edge = sample_steady_state(k_on=0.5, k_off=0.0, k_syn=k_syn_v, k_deg=k_deg_v, n_rep=100_000)
R = edge["R_samples"]
k_grid = np.arange(R.min(), R.max() + 1)
pmf = stats.poisson.pmf(k_grid, mu_theory)
fig = go.Figure()
fig.add_trace(go.Histogram(x=R, histnorm="probability",
                           xbins=dict(start=R.min() - 0.5, end=R.max() + 0.5, size=1),
                           marker_color="rgba(239, 68, 68, 0.55)", name="sampler (k_off=0)"))
fig.add_trace(go.Scatter(x=k_grid, y=pmf, mode="lines+markers",
                         line=dict(color=RNA_COLOR, width=2, dash="dash"),
                         name=f"Poisson(μ={mu_theory:.0f})"))
fig.update_layout(template="plotly_white", width=800, height=450, bargap=0.05,
                  margin=dict(l=85, r=30, t=40, b=60),
                  xaxis_title="RNA count R", yaxis_title="probability",
                  legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                              bgcolor="rgba(255,255,255,0.6)"))
fig.write_image(os.path.join(OUT, "validation_poisson.png"), scale=2)

print("Done. Analytical steady state: mu_G_ss=%.4f mu_R_ss=%.4f" % (mu_G_ss, mu_R_ss))
print("Sampler final estimates: mu_G=%.4f mu_R=%.4f" % (
    sample_steady_state(**PARAMS, n_rep=200000)["mu_G"],
    sample_steady_state(**PARAMS, n_rep=200000)["mu_R"]))


# ===========================================================================
# APPEND-ONLY SECTION.
#
# Everything below is added after the original script body and every block
# re-seeds the global numpy generator itself. Nothing above this line consumes
# a different number of variates than before, so every figure produced above and
# the two sampler estimates printed just above stay bit-for-bit identical. New
# code must keep going *after* this section, never before it.
# ===========================================================================


def fano_ss(k_on, k_off, k_syn, k_deg):
    """Closed-form steady-state Fano factor F = sigma_R^2 / mu_R.

    Obtained by setting the right-hand sides of the moment ODEs for C_RG and
    sigma_R^2 to zero and eliminating C_RG; this is Eq. (eq:fano) in the paper.
    """
    return 1.0 + (k_syn * k_off) / ((k_on + k_off) * (k_on + k_off + k_deg))


def measured_fano(m, tail=0.2):
    """Fano factor of an SSA run, averaged over the last ``tail`` of the horizon.

    A single grid point is far too noisy to quote, so sigma_R^2 and mu_R are
    each averaged over the stationary tail of the run before the ratio is taken.
    """
    n = m["time"].size
    lo = int(round((1.0 - tail) * n))
    return float(np.mean(m["sigma_R"][lo:] ** 2) / np.mean(m["mu_R"][lo:]))


# ---------------------------------------------------------------------------
# Measured Fano factor for every row of tab:param-exp.
#
# The configurations and seeds are exactly those of the eight parameter-
# exploration runs, and every run below re-seeds before simulating, so these
# numbers are reproducible whether or not the corresponding figure is drawn.
# For the two runs that are still plotted (slow/fast switching) the measured F
# therefore describes the very picture the reader is looking at; for the other
# six it describes the run reported in Notebook 1.
# ---------------------------------------------------------------------------
print("Table: measured Fano factors for tab:param-exp ...")

FANO_RUNS = [
    ("Slow switching",  1, dict(k_on=0.05, k_off=0.05, k_syn=20.0)),
    ("Fast switching",  2, dict(k_on=10.0, k_off=10.0, k_syn=20.0)),
    ("High expression", 3, dict(k_on=0.5,  k_off=4,    k_syn=56)),
    ("Low expression",  4, dict(k_on=0.5,  k_off=10,   k_syn=30)),
    ("Initial (0,0)",   5, dict(g0=0, r0=0)),
    ("Initial (1,15)",  6, dict(g0=1, r0=15)),
    ("Numeric low",     7, dict(n_sim=1000)),
    ("Numeric full",    8, dict(n_rep=224, n_sim=5000)),
]

for label, seed, overrides in FANO_RUNS:
    np.random.seed(seed)
    p = {**PARAM_BASE, **overrides}
    d = simulate_telegraph(**p)
    m = compute_sample_moments(d, n_grid=1000)
    f_pred = fano_ss(p["k_on"], p["k_off"], p["k_syn"], p["k_deg"])
    print("  %-16s F_pred=%6.2f  F_meas=%6.2f  (horizon t=%.1f)"
          % (label, f_pred, measured_fano(m), m["time"][-1]))


# ---------------------------------------------------------------------------
# Edge 1, quantitative: does the SSA approach the ODE curves at the rate
# 1/sqrt(N_rep) claimed in Section 3.4?
#
# One pooled ensemble of independent trajectories is simulated once, then
# subsampled: subsets of an i.i.d. pool are themselves i.i.d. ensembles, so
# this measures the same thing as re-running the SSA at each size, far cheaper.
# The statistic is the RMS-over-time deviation of mu_R_hat(t) from the ODE
# curve — RMS rather than sup, because the maximum over a 1000-point grid
# carries an extreme-value correction and does not give a clean -1/2 slope.
# ---------------------------------------------------------------------------
print("Figure: edge1_convergence.png")
np.random.seed(20260719)

E1_POOL = 10000          # total trajectories in the pool
E1_CHUNK = 1000          # simulated per chunk, to bound peak memory
E1_NSIM = 800            # events per trajectory (>> what T_END=10 needs)

t_e1, y_e1 = solve_ode_moments(**PARAMS, t0=0, g0=0, r0=0, t_end=T_END)
mu_R_ref = y_e1[1]
grid_e1 = np.linspace(0.0, T_END, mu_R_ref.size)

R_pool = np.empty((grid_e1.size, E1_POOL), dtype=np.float32)
for c in range(E1_POOL // E1_CHUNK):
    d = simulate_telegraph(**PARAMS, t0=0, g0=0, r0=0,
                           n_sim=E1_NSIM, n_rep=E1_CHUNK)
    assert d[-1, :, 0].min() >= T_END, "n_sim too small: a trajectory ended before T_END"
    for j in range(E1_CHUNK):
        idx = np.clip(np.searchsorted(d[:, j, 0], grid_e1, side="right") - 1,
                      0, d.shape[0] - 1)
        R_pool[:, c * E1_CHUNK + j] = d[idx, j, 2]

e1_sizes = np.array([10, 32, 100, 316, 1000, 3162, 10000])
e1_rms = []
for n in e1_sizes:
    n_trials = 1 if n == E1_POOL else 40
    vals = []
    for _ in range(n_trials):
        sub = np.random.choice(E1_POOL, size=int(n), replace=False)
        mu_hat = R_pool[:, sub].mean(axis=1)
        vals.append(np.sqrt(np.mean((mu_hat - mu_R_ref) ** 2)))
    e1_rms.append(float(np.mean(vals)))
e1_rms = np.array(e1_rms)

e1_slope = float(np.polyfit(np.log10(e1_sizes), np.log10(e1_rms), 1)[0])
e1_ref = e1_rms[0] * np.sqrt(e1_sizes[0]) / np.sqrt(e1_sizes.astype(float))

fig = go.Figure()
fig.add_trace(go.Scatter(x=e1_sizes, y=e1_rms, mode="markers+lines",
                         marker=dict(color=SSA_COLOR, size=8),
                         line=dict(color=SSA_COLOR, width=2),
                         name="RMS deviation of μ̂_R(t) from ODE"))
fig.add_trace(go.Scatter(x=e1_sizes, y=e1_ref, mode="lines",
                         line=dict(color=ODE_COLOR, width=2, dash="dash"),
                         name="∝ 1/√N_rep reference"))
fig.update_layout(template="plotly_white", width=850, height=560,
                  margin=dict(l=80, r=30, t=40, b=60),
                  xaxis=dict(title="Number of trajectories N_rep", type="log"),
                  yaxis=dict(title="RMS deviation of μ̂_R(t) from the ODE curve",
                             type="log"),
                  legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                              bgcolor="rgba(255,255,255,0.6)"))
fig.write_image(os.path.join(OUT, "edge1_convergence.png"), scale=2)
print("  fitted log-log slope = %.3f" % e1_slope)


# ---------------------------------------------------------------------------
# Edge 3 as a *distributional* test: SSA histogram vs steady-state sampler.
#
# Both sides are empirical, which is exactly what Edge 3 asserts (SSA <-> the
# sampler). The SSA side is read off at a single late time across independent
# replicates — never pooled over a time window, whose samples are
# autocorrelated and would understate the spread. The discrepancy is measured
# in total variation, against a two-sample permutation null: TV needs no
# binning, is bounded in [0,1], and unlike chi^2 does not force an arbitrary
# merge of the near-empty bins in the trough of the bimodal slow-switching law.
# ---------------------------------------------------------------------------
print("Figure: edge3_distribution.png")
np.random.seed(20260720)

E3_SSA_REP = 20000
E3_SAMPLER_REP = 200000
E3_CHUNK = 2000
E3_BOOT = 200


def ssa_stationary_counts(params, n_sim, n_rep, chunk=E3_CHUNK):
    """mRNA counts at one late time, one draw per independent SSA replicate."""
    out = []
    t_star = None
    for _ in range(n_rep // chunk):
        d = simulate_telegraph(**params, t0=0, g0=0, r0=0,
                               n_sim=n_sim, n_rep=chunk)
        if t_star is None:
            t_star = 0.9 * float(d[-1, :, 0].min())
        assert d[-1, :, 0].min() >= t_star, "chunk finished before t_star"
        for j in range(chunk):
            i = np.searchsorted(d[:, j, 0], t_star, side="right") - 1
            out.append(d[max(i, 0), j, 2])
    return np.asarray(out), t_star


def pmf_on(counts, k_max):
    """Empirical PMF of integer counts on the common support 0..k_max."""
    return np.bincount(np.asarray(counts, dtype=int), minlength=k_max + 1)[:k_max + 1] \
        / len(counts)


def tv_distance(a, b):
    k_max = int(max(a.max(), b.max()))
    return 0.5 * float(np.abs(pmf_on(a, k_max) - pmf_on(b, k_max)).sum())


E3_CASES = [
    ("Symmetric switching (k_on = k_off = 0.5)",
     dict(k_on=0.5, k_off=0.5, k_syn=20.0, k_deg=1.0), 2000),
    ("Slow switching (k_on = k_off = 0.05)",
     dict(k_on=0.05, k_off=0.05, k_syn=20.0, k_deg=1.0), 4000),
]

e3 = []
for title, pars, n_sim in E3_CASES:
    ssa_R, t_star = ssa_stationary_counts(pars, n_sim=n_sim, n_rep=E3_SSA_REP)
    smp_R = sample_steady_state(**pars, n_rep=E3_SAMPLER_REP)["R_samples"]
    tv_obs = tv_distance(ssa_R, smp_R)
    # Exact null band by direct resimulation. Under H0 the stationary law the
    # SSA samples *is* the law the sampler draws from, so the null distribution
    # of TV can be simulated outright: draw a fresh pair at the two sample sizes
    # and recompute TV. This needs no bootstrap or permutation approximation,
    # and it is the null the test actually claims.
    #
    # Resampling each side from its own empirical measure -- the obvious first
    # guess -- is *not* this null: it adds sampling noise twice over and centres
    # the band above the TV expected under H0, so a passing test looks like a
    # suspiciously good one. A pooled permutation null does reproduce the band
    # correctly (checked: median within 2% of the exact null below), but the
    # direct simulation is both cheaper here and free of that subtlety.
    null = np.empty(E3_BOOT)
    for b in range(E3_BOOT):
        null[b] = tv_distance(
            sample_steady_state(**pars, n_rep=ssa_R.size)["R_samples"],
            sample_steady_state(**pars, n_rep=smp_R.size)["R_samples"])
    # One-sided: the test looks for a discrepancy *larger* than chance.
    p_val = float(np.mean(null >= tv_obs))
    e3.append(dict(title=title, ssa=ssa_R, smp=smp_R, t_star=t_star,
                   tv=tv_obs, med=float(np.median(null)),
                   p95=float(np.percentile(null, 95)), p=p_val))
    # var_exact = mu_R + (k_syn/k_deg) * C_RG at steady state; quoted in the text
    # as the check that the SSA sample is correctly dispersed, not under-dispersed.
    mu_G_e = pars["k_on"] / (pars["k_on"] + pars["k_off"])
    mu_R_e = pars["k_syn"] / pars["k_deg"] * mu_G_e
    c_rg_e = (pars["k_syn"] * mu_G_e * (1 - mu_G_e)
              / (pars["k_on"] + pars["k_off"] + pars["k_deg"]))
    var_e = mu_R_e + pars["k_syn"] / pars["k_deg"] * c_rg_e
    print("  %-42s t*=%6.1f  TV=%.4f  null median=%.4f  95th=%.4f  p=%.3f  P_ssa(0)=%.4f"
          % (title, t_star, tv_obs, e3[-1]["med"], e3[-1]["p95"], p_val,
             float(np.mean(ssa_R == 0))))
    print("      var: SSA=%.2f  sampler=%.2f  exact=%.2f  (SSA off by %.2f%%)"
          % (ssa_R.var(), smp_R.var(), var_e,
             100 * abs(ssa_R.var() - var_e) / var_e))

k_hi = int(max(max(c["ssa"].max(), c["smp"].max()) for c in e3))
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                    subplot_titles=tuple("<b>%s</b>  %s" % (ab, c["title"])
                                         for ab, c in zip("AB", e3)))
for col, c in enumerate(e3, start=1):
    k_max = int(max(c["ssa"].max(), c["smp"].max()))
    ks = np.arange(k_max + 1)
    fig.add_trace(go.Bar(x=ks, y=pmf_on(c["ssa"], k_max),
                         marker_color="rgba(16, 185, 129, 0.55)",
                         name="SSA (20 000 replicates)",
                         showlegend=(col == 1)), row=1, col=col)
    fig.add_trace(go.Scatter(x=ks, y=pmf_on(c["smp"], k_max), mode="lines+markers",
                             line=dict(color=ODE_COLOR, width=2, dash="dash"),
                             marker=dict(size=4),
                             name="steady-state sampler (200 000)",
                             showlegend=(col == 1)), row=1, col=col)
    fig.add_annotation(row=1, col=col, x=0.97, y=0.86, xref="x domain",
                       yref="y domain", showarrow=False, align="right",
                       text=("TV = %.4f<br>null median %.4f, 95th %.4f<br>p = %.2f"
                             % (c["tv"], c["med"], c["p95"], c["p"])),
                       bgcolor="rgba(255,255,255,0.75)")
    fig.update_xaxes(title_text="mRNA count R", row=1, col=col)
fig.update_yaxes(title_text="probability", row=1, col=1)
fig.update_layout(template="plotly_white", width=1100, height=480, bargap=0.05,
                  margin=dict(l=80, r=30, t=70, b=60),
                  legend=dict(orientation="h", yanchor="bottom", y=1.10,
                              xanchor="left", x=0,
                              bgcolor="rgba(255,255,255,0.6)"))
fig.write_image(os.path.join(OUT, "edge3_distribution.png"), scale=2)
print("Append-only section done.")


# ---------------------------------------------------------------------------
# Monte Carlo standard errors for the five-moment sampler table (tab:moments).
#
# All five moments in that table are computed from *one* array of Beta draws,
# so their errors are driven by a single shared fluctuation rather than being
# five independent confirmations. The SEs below are bootstrapped from a draw of
# the same size; at N_rep = 2e5 they are stable to the digits quoted, and the
# two that have a closed form are printed next to it as a check:
#   SE(mu_G) = sqrt(mu_G(1-mu_G)/N),  SE(mu_R) = sigma_R/sqrt(N).
# ---------------------------------------------------------------------------
print("Table: bootstrap standard errors for tab:moments ...")
np.random.seed(20260721)

SE_PARAMS = dict(k_on=0.5, k_off=1.0, k_syn=20.0, k_deg=1.0)
SE_N = 200000
SE_BOOT = 400

s = sample_steady_state(**SE_PARAMS, n_rep=SE_N)
G_s, R_s = s["G_samples"], s["R_samples"]

boot = np.empty((SE_BOOT, 5))
for b in range(SE_BOOT):
    idx = np.random.randint(0, SE_N, size=SE_N)
    g, r = G_s[idx], R_s[idx]
    boot[b] = (g.mean(), r.mean(), g.std(ddof=1), r.std(ddof=1),
               np.cov(g, r, ddof=1)[0, 1])
se = boot.std(axis=0, ddof=1)

mu_G_a = SE_PARAMS["k_on"] / (SE_PARAMS["k_on"] + SE_PARAMS["k_off"])
mu_R_a = (SE_PARAMS["k_syn"] / SE_PARAMS["k_deg"]) * mu_G_a
c_rg_a = (SE_PARAMS["k_syn"] * mu_G_a * (1 - mu_G_a)
          / (SE_PARAMS["k_on"] + SE_PARAMS["k_off"] + SE_PARAMS["k_deg"]))
sig_G_a = np.sqrt(mu_G_a * (1 - mu_G_a))
sig_R_a = np.sqrt(mu_R_a + (SE_PARAMS["k_syn"] / SE_PARAMS["k_deg"]) * c_rg_a)

for name, an, est in zip(("mu_G", "mu_R", "sigma_G", "sigma_R", "C_RG"),
                         (mu_G_a, mu_R_a, sig_G_a, sig_R_a, c_rg_a),
                         se):
    print("  SE(%-7s) = %.6f" % (name, est))
print("  closed-form check: SE(mu_G)=%.6f  SE(mu_R)=%.6f"
      % (np.sqrt(mu_G_a * (1 - mu_G_a) / SE_N), sig_R_a / np.sqrt(SE_N)))
print("All appended blocks done.")


# ===========================================================================
# APPEND-ONLY BLOCK 2 (revision).
#
# Supplies the four numbers the paper previously left as [INSERT_*] markers,
# plus the coefficient of variation behind the low-expression claim. Same
# contract as above: every block re-seeds numpy itself and nothing here is
# consumed by any earlier block, so all nine pre-existing figures and every
# number printed above stay bit-for-bit identical.
# ===========================================================================
import json  # noqa: E402  (kept here to honour the append-only contract)

RESULTS = {}


def measured_cv(m, tail=0.2):
    """Coefficient of variation sigma_R / mu_R over the stationary tail.

    Same tail-averaging convention as :func:`measured_fano`: a single grid
    point is far too noisy, so numerator and denominator are each averaged
    over the last ``tail`` of the horizon before the ratio is taken.
    """
    n = m["time"].size
    lo = int(round((1.0 - tail) * n))
    return float(np.mean(m["sigma_R"][lo:]) / np.mean(m["mu_R"][lo:]))


# ---------------------------------------------------------------------------
# Per-row uncertainty for tab:param-exp, by multi-seed replication.
#
# Each configuration is re-run under K independent seeds and we report the
# sample SD *across* those K Fano estimates. That is the standard error of a
# single run's estimate — which is what the +/- on a single tabulated number
# means. It is deliberately NOT SD/sqrt(K), which would be the SE of the mean
# of the K runs, a smaller number describing a quantity the table never quotes.
#
# Replication is used rather than a bootstrap because compute_sample_moments
# returns aggregated moments only: there is no per-trajectory array to
# resample, and resampling the 200 tail *grid points* instead would be wrong,
# since consecutive grid points of one ensemble are strongly autocorrelated
# and would understate the spread by a large factor.
# ---------------------------------------------------------------------------
print("Table: multi-seed Fano/CV standard errors for tab:param-exp ...")

FANO_SE_K = 10          # replicate seeds per configuration

# One configuration beyond the original eight. The paper's "numerical" comparison
# moved n_sim and n_rep together, which cannot isolate either; this run holds
# n_sim at the base 4000 and changes only the ensemble size, so that the pair
# (base, this) isolates n_rep and the pair (base, "Numeric low") isolates n_sim.
# Kept in its own list so the original FANO_RUNS block above is untouched.
FANO_RUNS_EXTRA = [
    ("Ensemble only",   9, dict(n_rep=224)),
]

for run_idx, (label, _orig_seed, overrides) in enumerate(FANO_RUNS + FANO_RUNS_EXTRA):
    p = {**PARAM_BASE, **overrides}
    f_pred = fano_ss(p["k_on"], p["k_off"], p["k_syn"], p["k_deg"])
    f_reps, cv_reps = [], []
    for k in range(FANO_SE_K):
        np.random.seed(30000 + 100 * run_idx + k)
        m = compute_sample_moments(simulate_telegraph(**p), n_grid=1000)
        f_reps.append(measured_fano(m))
        cv_reps.append(measured_cv(m))
    f_reps, cv_reps = np.array(f_reps), np.array(cv_reps)
    RESULTS[label] = dict(
        F_pred=f_pred,
        F_mean=float(f_reps.mean()), F_se=float(f_reps.std(ddof=1)),
        CV_mean=float(cv_reps.mean()), CV_se=float(cv_reps.std(ddof=1)),
        mu_R_ss=p["k_syn"] / p["k_deg"] * p["k_on"] / (p["k_on"] + p["k_off"]),
        n_rep=p["n_rep"], n_sim=p["n_sim"], K=FANO_SE_K)
    print("  %-16s F_pred=%6.2f  F=%6.2f +/- %.2f   CV=%5.3f +/- %.3f"
          % (label, f_pred, f_reps.mean(), f_reps.std(ddof=1),
             cv_reps.mean(), cv_reps.std(ddof=1)))


# ---------------------------------------------------------------------------
# Standard error on the Edge-1 log-log slope.
#
# e1_sizes / e1_rms are still in scope from the Edge-1 block above, so this
# only refits them; no simulation is repeated and the figure is untouched.
#
# The seven points are NOT homoscedastic: every size except the largest is an
# average over n_trials=40 subsamples, while N_rep=10000 is a single
# realisation (it exhausts the pool, so there is nothing to subsample). The
# plain fit therefore under-weights that extra noise. We report the plain fit
# with its covariance-based SE and, as a robustness check, the fit with the
# single-realisation point dropped; if the two agree the caveat is cosmetic.
# ---------------------------------------------------------------------------
print("Edge 1: slope standard error ...")

_c, _cov = np.polyfit(np.log10(e1_sizes), np.log10(e1_rms), 1, cov=True)
e1_slope_full, e1_slope_se = float(_c[0]), float(np.sqrt(_cov[0, 0]))
_c2, _cov2 = np.polyfit(np.log10(e1_sizes[:-1]), np.log10(e1_rms[:-1]), 1, cov=True)
e1_slope_drop, e1_slope_drop_se = float(_c2[0]), float(np.sqrt(_cov2[0, 0]))

RESULTS["edge1_slope"] = dict(
    slope=e1_slope_full, se=e1_slope_se,
    slope_drop_last=e1_slope_drop, se_drop_last=e1_slope_drop_se,
    sizes=[int(x) for x in e1_sizes], rms=[float(x) for x in e1_rms])
print("  slope = %.3f +/- %.3f   (dropping single-realisation point: %.3f +/- %.3f)"
      % (e1_slope_full, e1_slope_se, e1_slope_drop, e1_slope_drop_se))


# ---------------------------------------------------------------------------
# Same treatment for Edge 2, so the two convergence claims are stated to the
# same standard rather than one being fitted and the other eyeballed.
#
# n_reps / errs are in scope from the steady-state convergence figure. Unlike
# the Edge-1 sizes these are homoscedastic -- all 25 points average over the
# same n_trials=40 -- so the plain covariance-based SE needs no caveat.
# ---------------------------------------------------------------------------
print("Edge 2: slope standard error ...")

_c3, _cov3 = np.polyfit(np.log10(n_reps), np.log10(errs), 1, cov=True)
e2_slope, e2_slope_se = float(_c3[0]), float(np.sqrt(_cov3[0, 0]))
RESULTS["edge2_slope"] = dict(slope=e2_slope, se=e2_slope_se,
                              n_points=int(n_reps.size), n_trials=40)
print("  slope = %.3f +/- %.3f  (over %d sizes)" % (e2_slope, e2_slope_se, n_reps.size))


# ---------------------------------------------------------------------------
# k_off = 0 boundary: Fano factor of the constitutive gene, with a bootstrap SE.
#
# This number was previously computed only in notebooks/validation.ipynb and
# hand-copied; recomputing it here puts it under the same reproducible seed as
# validation_poisson.png, which is drawn from an identical sampler call.
# R_samples are i.i.d. draws, so the tab:moments bootstrap applies directly.
# ---------------------------------------------------------------------------
print("Boundary k_off=0: Fano factor with bootstrap SE ...")
np.random.seed(20260722)

POIS_N, POIS_BOOT = 100000, 400
_edge = sample_steady_state(k_on=0.5, k_off=0.0, k_syn=10.0, k_deg=1.0, n_rep=POIS_N)
_R = _edge["R_samples"]
pois_fano = float(_R.var(ddof=1) / _R.mean())
_bf = np.empty(POIS_BOOT)
for b in range(POIS_BOOT):
    _s = _R[np.random.randint(0, POIS_N, size=POIS_N)]
    _bf[b] = _s.var(ddof=1) / _s.mean()
pois_fano_se = float(_bf.std(ddof=1))

RESULTS["poisson_boundary"] = dict(fano=pois_fano, se=pois_fano_se,
                                   n_rep=POIS_N, n_boot=POIS_BOOT, theory=1.0)
print("  Fano(k_off=0) = %.4f +/- %.4f  (theory 1)" % (pois_fano, pois_fano_se))


# ---------------------------------------------------------------------------
# Low-expression moment traces (new figure for the CV claim in sec:res-explore).
#
# The comment beside the switching-speed figures above says the expression runs
# are reported through the table rather than as figures. That is no longer
# true for the low-expression run: the text now makes a quantitative claim
# about its coefficient of variation, so the run it describes is shown.
# ---------------------------------------------------------------------------
print("Figure: param_expression_low.png")
save_param_experiment("param_expression_low.png", "Low-Expression Regime",
                      seed=4, k_on=0.5, k_off=10, k_syn=30)


# ---------------------------------------------------------------------------
# Machine-readable dump, so the numbers in main.tex can be diffed against the
# run that produced them instead of being trusted to hand-transcription.
# ---------------------------------------------------------------------------
with open(os.path.join(os.path.dirname(OUT), "results.json"), "w") as fh:
    json.dump(RESULTS, fh, indent=2, sort_keys=True)
print("Wrote results.json")
print("Revision block done.")

"""Generate the PNG figures used in the paper.

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

SSA_COLOR = "#10B981"
SSA_FILL = "rgba(16, 185, 129, 0.12)"
ODE_COLOR = "#1E40AF"

# ---------------------------------------------------------------------------
# Run the three approaches
# ---------------------------------------------------------------------------
print("Running SSA ...")
data = simulate_telegraph(**PARAMS, t0=0, g0=0, r0=0, n_sim=N_SIM, n_rep=N_REP)
ssa = compute_sample_moments(data, t_end=T_END)

print("Solving moment ODEs ...")
t_ode, y_ode = solve_ode_moments(**PARAMS, t0=0, g0=0, r0=0, t_end=T_END)
mu_G_ode, mu_R_ode, sigma2_R_ode, c_rg_ode = y_ode


def band(fig, x, lo, hi, row):
    fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), row=row, col=1)
    fig.add_trace(go.Scatter(x=x, y=lo, fill="tonexty", fillcolor=SSA_FILL,
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), row=row, col=1)


# ---------------------------------------------------------------------------
# Figure 1: SSA moment dynamics
# ---------------------------------------------------------------------------
print("Figure 1: ssa_moments.png")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=("<b>A</b>  Mean gene activity μ_G(t) ± σ_G",
                                    "<b>B</b>  Mean mRNA count μ_R(t) ± σ_R",
                                    "<b>C</b>  Gene–RNA covariance C_RG(t)"),
                    vertical_spacing=0.07)
band(fig, ssa["time"], ssa["mu_G"] - ssa["sigma_G"], ssa["mu_G"] + ssa["sigma_G"], 1)
fig.add_trace(go.Scatter(x=ssa["time"], y=ssa["mu_G"], line=dict(color=SSA_COLOR, width=2),
                         showlegend=False), row=1, col=1)
band(fig, ssa["time"], ssa["mu_R"] - ssa["sigma_R"], ssa["mu_R"] + ssa["sigma_R"], 2)
fig.add_trace(go.Scatter(x=ssa["time"], y=ssa["mu_R"], line=dict(color=SSA_COLOR, width=2),
                         showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=ssa["time"], y=ssa["cov_RG"], line=dict(color=SSA_COLOR, width=2),
                         showlegend=False), row=3, col=1)
fig.update_layout(template="plotly_white", width=1000, height=820,
                  margin=dict(l=80, r=30, t=60, b=50))
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_yaxes(title_text="⟨G⟩", row=1, col=1)
fig.update_yaxes(title_text="⟨R⟩", row=2, col=1)
fig.update_yaxes(title_text="Cov(R, G)", row=3, col=1)
fig.write_image(os.path.join(OUT, "ssa_moments.png"), scale=2)

# ---------------------------------------------------------------------------
# Figure 2: ODE moment dynamics
# ---------------------------------------------------------------------------
print("Figure 2: ode_moments.png")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=("<b>A</b>  Mean gene activity μ_G(t)",
                                    "<b>B</b>  Mean mRNA count μ_R(t)",
                                    "<b>C</b>  Gene–RNA covariance C_RG(t)"),
                    vertical_spacing=0.07)
fig.add_trace(go.Scatter(x=t_ode, y=mu_G_ode, line=dict(color=ODE_COLOR, width=2),
                         showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=t_ode, y=mu_R_ode, line=dict(color=ODE_COLOR, width=2),
                         showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=t_ode, y=c_rg_ode, line=dict(color=ODE_COLOR, width=2),
                         showlegend=False), row=3, col=1)
fig.update_layout(template="plotly_white", width=1000, height=820,
                  margin=dict(l=80, r=30, t=60, b=50))
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_yaxes(title_text="μ_G", row=1, col=1)
fig.update_yaxes(title_text="μ_R", row=2, col=1)
fig.update_yaxes(title_text="C_RG", row=3, col=1)
fig.write_image(os.path.join(OUT, "ode_moments.png"), scale=2)

# ---------------------------------------------------------------------------
# Figure 3: SSA vs ODE overlay
# ---------------------------------------------------------------------------
print("Figure 3: ssa_vs_ode.png")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=("<b>A</b>  Mean gene activity ⟨G⟩ ± σ_G",
                                    "<b>B</b>  Mean mRNA count ⟨R⟩ ± σ_R",
                                    "<b>C</b>  Gene–RNA covariance"),
                    vertical_spacing=0.07)
band(fig, ssa["time"], ssa["mu_G"] - ssa["sigma_G"], ssa["mu_G"] + ssa["sigma_G"], 1)
fig.add_trace(go.Scatter(x=ssa["time"], y=ssa["mu_G"], line=dict(color=SSA_COLOR, width=2),
                         name="SSA"), row=1, col=1)
fig.add_trace(go.Scatter(x=t_ode, y=mu_G_ode, line=dict(color=ODE_COLOR, width=2, dash="dash"),
                         name="ODE"), row=1, col=1)
band(fig, ssa["time"], ssa["mu_R"] - ssa["sigma_R"], ssa["mu_R"] + ssa["sigma_R"], 2)
fig.add_trace(go.Scatter(x=ssa["time"], y=ssa["mu_R"], line=dict(color=SSA_COLOR, width=2),
                         showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=t_ode, y=mu_R_ode, line=dict(color=ODE_COLOR, width=2, dash="dash"),
                         showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=ssa["time"], y=ssa["cov_RG"], line=dict(color=SSA_COLOR, width=2),
                         showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=t_ode, y=c_rg_ode, line=dict(color=ODE_COLOR, width=2, dash="dash"),
                         showlegend=False), row=3, col=1)
fig.update_layout(template="plotly_white", width=1000, height=820,
                  margin=dict(l=80, r=30, t=60, b=50),
                  legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0))
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_yaxes(title_text="⟨G⟩", row=1, col=1)
fig.update_yaxes(title_text="⟨R⟩", row=2, col=1)
fig.update_yaxes(title_text="Cov(R, G)", row=3, col=1)
fig.write_image(os.path.join(OUT, "ssa_vs_ode.png"), scale=2)

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
GENE_COLOR = "#3B82F6"
GENE_FILL = "rgba(59, 130, 246, 0.07)"
RNA_COLOR = "#EF4444"
RNA_FILL = "rgba(239, 68, 68, 0.07)"
COV_COLOR = "#8B5CF6"

PARAM_BASE = dict(k_on=0.1, k_off=0.1, k_syn=10.0, k_deg=1.0,
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
save_param_experiment("param_expr_high.png", "High Expression",
                      seed=3, k_on=0.5, k_off=4, k_syn=56)
save_param_experiment("param_expr_low.png", "Low Expression",
                      seed=4, k_on=0.5, k_off=10, k_syn=30)
save_param_experiment("param_init_inactive.png", "Inactive Start (g0=0, r0=0)",
                      seed=5, g0=0, r0=0)
save_param_experiment("param_init_burst.png", "Active-Burst Start (g0=1, r0=15)",
                      seed=6, g0=1, r0=15)
save_param_experiment("param_numeric_low.png",
                      "Numerical Effects: Low Scale (n_sim=1000)",
                      seed=7, n_sim=1000)
save_param_experiment("param_numeric_full.png",
                      "Numerical Effects: Experiment Scale (n_rep=224, n_sim=5000)",
                      seed=8, n_rep=224, n_sim=5000)

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
mu_G_s, mu_R_s, _, c_rg_s = y_ode_s

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=("<b>a</b> Gene State ⟨G⟩ ± σ<sub>G</sub>",
                                    "<b>b</b> RNA Count ⟨R⟩ ± σ<sub>R</sub>",
                                    "<b>c</b> Gene–RNA Covariance"),
                    vertical_spacing=0.06)
band(fig, m_slow["time"], m_slow["mu_G"] - m_slow["sigma_G"],
     m_slow["mu_G"] + m_slow["sigma_G"], 1)
fig.add_trace(go.Scatter(x=m_slow["time"], y=m_slow["mu_G"],
                         line=dict(color=SSA_COLOR, width=2), name="SSA"), row=1, col=1)
fig.add_trace(go.Scatter(x=t_ode_s, y=mu_G_s,
                         line=dict(color=ODE_COLOR, width=2, dash="dash"), name="ODE"), row=1, col=1)
band(fig, m_slow["time"], m_slow["mu_R"] - m_slow["sigma_R"],
     m_slow["mu_R"] + m_slow["sigma_R"], 2)
fig.add_trace(go.Scatter(x=m_slow["time"], y=m_slow["mu_R"],
                         line=dict(color=SSA_COLOR, width=2), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=t_ode_s, y=mu_R_s,
                         line=dict(color=ODE_COLOR, width=2, dash="dash"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=m_slow["time"], y=m_slow["cov_RG"],
                         line=dict(color=SSA_COLOR, width=2), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=t_ode_s, y=c_rg_s,
                         line=dict(color=ODE_COLOR, width=2, dash="dash"), showlegend=False), row=3, col=1)
fig.update_layout(template="plotly_white", width=1000, height=820,
                  margin=dict(l=80, r=30, t=60, b=50),
                  legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0))
fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_yaxes(title_text="⟨G⟩", row=1, col=1)
fig.update_yaxes(title_text="⟨R⟩", row=2, col=1)
fig.update_yaxes(title_text="Cov(R, G)", row=3, col=1)
fig.write_image(os.path.join(OUT, "ssaode_slow.png"), scale=2)

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

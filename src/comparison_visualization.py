import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ODE in dark blue, SSA in green — flat palette consistent with the project.
ODE_COLOR = "#1E40AF"
SSA_COLOR = "#10B981"
SSA_FILL = "rgba(16, 185, 129, 0.10)"


def show_combined_moments(moments_ssa, t_ode, y_ode, title="SSA vs ODE Comparison"):
    """Overlay SSA sample moments and deterministic ODE moments on one 3-panel figure.

    All SSA traces (mean lines + the empirical ±σ band) share the legend group "ssa";
    all ODE mean lines share the group "ode". With legend ``groupclick="togglegroup"``,
    clicking the single "SSA" or "ODE" legend entry toggles every trace in that group,
    letting the reader switch between both / SSA-only / ODE-only interactively.

    Args:
        moments_ssa (dict): output of compute_sample_moments (time, mu_G, mu_R,
            sigma_G, sigma_R, cov_RG).
        t_ode (np.ndarray): ODE time grid from solve_ode_moments.
        y_ode (np.ndarray): ODE solution rows (mu_G, mu_R, sigma2_R, c_rg).
        title (str): figure title.
    """
    t_ssa = moments_ssa["time"]
    mu_G_ode, mu_R_ode, _, c_rg_ode = y_ode

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "<b>a</b> Gene State ⟨G⟩ ± σ<sub>G</sub>",
            "<b>b</b> RNA Count ⟨R⟩ ± σ<sub>R</sub>",
            "<b>c</b> Gene–RNA Covariance",
        ),
        vertical_spacing=0.06,
    )

    # --- Panel (a): Gene State G ---
    # SSA empirical ±σ band (drawn first, sits behind the lines)
    fig.add_trace(
        go.Scatter(
            x=t_ssa,
            y=moments_ssa["mu_G"] + moments_ssa["sigma_G"],
            line=dict(width=0),
            legendgroup="ssa",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ssa,
            y=moments_ssa["mu_G"] - moments_ssa["sigma_G"],
            fill="tonexty",
            fillcolor=SSA_FILL,
            line=dict(width=0),
            legendgroup="ssa",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    # SSA mean — carries the single "SSA" legend entry
    fig.add_trace(
        go.Scatter(
            x=t_ssa,
            y=moments_ssa["mu_G"],
            line=dict(color=SSA_COLOR, width=2),
            name="SSA",
            legendgroup="ssa",
            showlegend=True,
            hovertemplate="SSA ⟨G⟩ = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # ODE mean — carries the single "ODE" legend entry
    fig.add_trace(
        go.Scatter(
            x=t_ode,
            y=mu_G_ode,
            line=dict(color=ODE_COLOR, width=2),
            name="ODE",
            legendgroup="ode",
            showlegend=True,
            hovertemplate="ODE ⟨G⟩ = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # --- Panel (b): RNA Count R ---
    fig.add_trace(
        go.Scatter(
            x=t_ssa,
            y=moments_ssa["mu_R"] + moments_ssa["sigma_R"],
            line=dict(width=0),
            legendgroup="ssa",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ssa,
            y=moments_ssa["mu_R"] - moments_ssa["sigma_R"],
            fill="tonexty",
            fillcolor=SSA_FILL,
            line=dict(width=0),
            legendgroup="ssa",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ssa,
            y=moments_ssa["mu_R"],
            line=dict(color=SSA_COLOR, width=2),
            legendgroup="ssa",
            showlegend=False,
            hovertemplate="SSA ⟨R⟩ = %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ode,
            y=mu_R_ode,
            line=dict(color=ODE_COLOR, width=2),
            legendgroup="ode",
            showlegend=False,
            hovertemplate="ODE ⟨R⟩ = %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # --- Panel (c): Covariance Cov(R, G) (no band) ---
    fig.add_trace(
        go.Scatter(
            x=t_ssa,
            y=moments_ssa["cov_RG"],
            line=dict(color=SSA_COLOR, width=2),
            legendgroup="ssa",
            showlegend=False,
            hovertemplate="SSA Cov = %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_ode,
            y=c_rg_ode,
            line=dict(color=ODE_COLOR, width=2),
            legendgroup="ode",
            showlegend=False,
            hovertemplate="ODE Cov = %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Layout — matched to show_sample_moments, plus a grouped legend for toggling.
    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=850,
        title=f"<b>{title}</b>",
        margin=dict(l=85, r=30, t=100, b=50),
        hovermode="x",
        legend=dict(
            orientation="h",
            groupclick="togglegroup",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255, 255, 255, 0.5)",
        ),
    )

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Active Fraction ⟨G⟩", row=1, col=1)
    fig.update_yaxes(title_text="Mean mRNA Count ⟨R⟩", row=2, col=1)
    fig.update_yaxes(title_text="Cov(R, G)", row=3, col=1)

    fig.show()

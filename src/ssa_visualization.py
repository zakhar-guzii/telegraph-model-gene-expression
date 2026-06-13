import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_sample_moments(moments: dict, title: str = None) -> None:
    t = moments["time"]

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

    # Panel (a): Gene State G
    fig.add_trace(
        go.Scatter(
            x=t,
            y=moments["mu_G"] + moments["sigma_G"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=moments["mu_G"] - moments["sigma_G"],
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.07)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=moments["mu_G"],
            line=dict(color="#3B82F6", width=2),
            showlegend=False,
            hovertemplate="⟨G⟩ = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel (b): RNA Count R
    fig.add_trace(
        go.Scatter(
            x=t,
            y=moments["mu_R"] + moments["sigma_R"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=moments["mu_R"] - moments["sigma_R"],
            fill="tonexty",
            fillcolor="rgba(239, 68, 68, 0.07)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=moments["mu_R"],
            line=dict(color="#EF4444", width=2),
            showlegend=False,
            hovertemplate="⟨R⟩ = %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Panel (c): Covariance Cov(R, G)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=moments["cov_RG"],
            line=dict(color="#8B5CF6", width=2),
            showlegend=False,
            hovertemplate="Cov = %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Layout Adjustments
    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=850,
        title=f"<b>{title}</b>" if title else "<b>Evolution of Sample Moments of the Telegraph Model (SSA)</b>",        margin=dict(l=85, r=30, t=70, b=50),
        hovermode="x",
    )

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Active Fraction ⟨G⟩", row=1, col=1)
    fig.update_yaxes(title_text="Mean mRNA Count ⟨R⟩", row=2, col=1)
    fig.update_yaxes(title_text="Cov(R, G)", row=3, col=1)

    fig.show()

    return fig


def show_single_trajectory(
    raw_data, trajectory_index=0, title="Single Stochastic Realization"
):
    """Plot one raw SSA realization as a standalone 2-panel figure.

    Unlike show_sample_moments (ensemble mean ± band), each panel shows the absolute
    values of a single trajectory selected by ``trajectory_index`` — no averaging.

    Args:
        raw_data (np.ndarray): (n_sim+1, n_rep, 3) array from simulate_telegraph;
            column 0 = time, 1 = gene state, 2 = RNA count.
        trajectory_index (int): which trajectory (second axis) to plot.
        title (str): figure title.
    """
    t = raw_data[:, trajectory_index, 0]
    g = raw_data[:, trajectory_index, 1]
    r = raw_data[:, trajectory_index, 2]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "<b>a</b> Gene State (G) [0 or 1]",
            "<b>b</b> mRNA Molecule Count (R)",
        ),
        vertical_spacing=0.08,
    )

    # Panel (a): gene state — HV step line
    fig.add_trace(
        go.Scatter(
            x=t,
            y=g,
            line=dict(color="#3B82F6", width=1.5, shape="hv"),
            showlegend=False,
            hovertemplate="G = %{y:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # Panel (b): RNA count — HV step line
    fig.add_trace(
        go.Scatter(
            x=t,
            y=r,
            line=dict(color="#EF4444", width=1.5, shape="hv"),
            showlegend=False,
            hovertemplate="R = %{y:.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Layout — matched to show_sample_moments
    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=600,
        title=f"<b>{title}</b>",
        margin=dict(l=85, r=30, t=70, b=50),
        hovermode="x",
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Gene State (G) [0 or 1]", row=1, col=1)
    fig.update_yaxes(title_text="mRNA Molecule Count (R)", row=2, col=1)

    fig.show()

    return fig

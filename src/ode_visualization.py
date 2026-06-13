import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_ode_moments(t, y, title=None):
    """Plot the deterministic ODE moment trajectories, styled like show_sample_moments.

    The ODE is deterministic — each moment is a single-valued curve — so no noise
    band is drawn. Only the mean gene state, the mean mRNA count, and the gene–RNA
    covariance are shown.
    """
    mu_G, mu_R, _, c_rg = y

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "<b>a</b> Gene State ⟨G⟩",
            "<b>b</b> RNA Count ⟨R⟩",
            "<b>c</b> Gene–RNA Covariance",
        ),
        vertical_spacing=0.06,
    )

    # Panel (a): Gene State G (deterministic mean)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mu_G,
            line=dict(color="#3B82F6", width=2),
            showlegend=False,
            hovertemplate="⟨G⟩ = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel (b): RNA Count R (deterministic mean)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mu_R,
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
            y=c_rg,
            line=dict(color="#8B5CF6", width=2),
            showlegend=False,
            hovertemplate="Cov = %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Layout — matched to show_sample_moments
    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=850,
        title=f"<b>{title}</b>"
        if title
        else "<b>Evolution of ODE Moments of the Telegraph Model</b>",
        margin=dict(l=85, r=30, t=70, b=50),
        hovermode="x",
    )

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Active Fraction ⟨G⟩", row=1, col=1)
    fig.update_yaxes(title_text="Mean mRNA Count ⟨R⟩", row=2, col=1)
    fig.update_yaxes(title_text="Cov(R, G)", row=3, col=1)

    fig.show()

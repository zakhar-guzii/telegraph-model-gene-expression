import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_ode_moments(
    t: np.ndarray, y: np.ndarray, title: str = None, analytical: dict = None
) -> None:
    mu_G, mu_R, sigma2_R, c_rg = y
    sigma_G = np.sqrt(np.clip(mu_G * (1.0 - mu_G), 0, None))
    sigma_R = np.sqrt(np.clip(sigma2_R, 0, None))

    if analytical is not None:
        ss_G = analytical.get("mu_G")
        ss_R = analytical.get("mu_R")
        ss_cov = analytical.get("cov_RG")
    else:
        tail = max(1, len(mu_G) // 10)
        ss_G = float(mu_G[-tail:].mean())
        ss_R = float(mu_R[-tail:].mean())
        ss_cov = float(c_rg[-tail:].mean())

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

    # Panel (a)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mu_G + sigma_G,
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
            y=mu_G - sigma_G,
            fill="tonexty",
            fillcolor="rgba(30, 64, 175, 0.05)",
            line=dict(width=0),
            name="Fluctuation Band (±σ)",
            legendgroup="band",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mu_G,
            line=dict(color="#1E40AF", width=2.5),
            name="ODE Mean",
            legendgroup="mean",
            hovertemplate="⟨G⟩ = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    

    # Panel (b)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mu_R + sigma_R,
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
            y=mu_R - sigma_R,
            fill="tonexty",
            fillcolor="rgba(185, 28, 28, 0.05)",
            line=dict(width=0),
            name="Fluctuation Band (±σ)",
            legendgroup="band",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=mu_R,
            line=dict(color="#B91C1C", width=2.5),
            name="ODE Mean",
            legendgroup="mean",
            showlegend=False,
            hovertemplate="⟨R⟩ = %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    

    # Panel (c)
    fig.add_trace(
        go.Scatter(
            x=t,
            y=c_rg,
            line=dict(color="#5B21B6", width=2.5),
            name="Gene–RNA Covariance",
            legendgroup="cov",
            hovertemplate="Cov = %{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    plot_title = (
        f"<b>{title}</b>"
        if title
        else "<b>Evolution of ODE Moments of the Telegraph Model</b>"
    )

    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=850,
        title=dict(text=plot_title, x=0.05),
        margin=dict(l=85, r=30, t=110, b=50),
        hovermode="x",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.5)",
        ),
    )

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Active Fraction ⟨G⟩", row=1, col=1)
    fig.update_yaxes(title_text="Mean mRNA Count ⟨R⟩", row=2, col=1)
    fig.update_yaxes(title_text="Cov(R, G)", row=3, col=1)

    fig.show()

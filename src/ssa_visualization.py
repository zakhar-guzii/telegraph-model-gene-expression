import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_sample_moments(moments: dict) -> None:
    t = moments["time"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "<b>a</b> Gene State ⟨G⟩ ± σ<sub>G</sub>",
            "<b>b</b> RNA Count ⟨R⟩ ± σ<sub>R</sub>",
            "<b>c</b> Covariance Cov(R, G)",
        ),
        vertical_spacing=0.07,  
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
            fillcolor="rgba(59, 130, 246, 0.1)",
            line=dict(width=0),
            name="± σ_G",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=moments["mu_G"], line=dict(color="#3B82F6", width=2), name="⟨G⟩"
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
            fillcolor="rgba(239, 68, 68, 0.1)",
            line=dict(width=0),
            name="± σ_R",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=moments["mu_R"], line=dict(color="#EF4444", width=2), name="⟨R⟩"
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
            name="Cov(R, G)",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=800,  
        title="<b>Evolution of Sample Moments of the Telegraph Model (SSA)</b>",
        hovermode="x unified",  
    )

    fig.update_xaxes(title_text="Time (arbitrary units)", row=3, col=1)
    fig.update_yaxes(title_text="Mean Gene State ⟨G⟩ (Probability ON)", row=1, col=1)
    fig.update_yaxes(title_text="Mean RNA Count ⟨R⟩ (Molecules)", row=2, col=1)
    fig.update_yaxes(title_text="Cov(R, G)", row=3, col=1)

    fig.show()
import numpy as np
import plotly.graph_objects as go

def show_ode_moments(
    t: np.ndarray,
    y: np.ndarray,
    figsize: tuple[float, float] = (4.5, 2.8),
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Plot the time evolution of ODE-based moments as three interactive figures.

    * **Gene state** μ_G ± σ_G  (panel **a**)
    * **RNA count**  μ_R ± σ_R  (panel **b**)
    * **Gene–RNA covariance** C_RG  (panel **c**)

    Parameters
    ----------
    t : np.ndarray
        1-D time array returned by :func:`solve_ode_moments`.
    y : np.ndarray
        2-D array (4 × N) returned by :func:`solve_ode_moments`.
        Rows: μ_G, μ_R, σ²_R, C_RG.
    figsize : tuple[float, float], optional
        Width × height in inches.

    Returns
    -------
    tuple[go.Figure, go.Figure, go.Figure]
        ``(fig_gene, fig_rna, fig_cov)``
    """
    mu_G, mu_R, sigma2_R, c_rg = y

    # Derive std devs from the moments the ODE gives us directly
    sigma_G = np.sqrt(np.clip(mu_G * (1.0 - mu_G), 0, None))  # Bernoulli closure
    sigma_R = np.sqrt(np.clip(sigma2_R, 0, None))

    # ── Panel (a): Gene state μ_G ± σ_G
    fig_gene = go.Figure(
        layout=_base_layout(
            title="<b>a</b>  Gene state (ODE)",
            ylabel="μ<sub>G</sub>",
            figsize=figsize,
        )
    )
    _add_sigma_band(fig_gene, t, mu_G, sigma_G, _BLUE_LIGHT, "± σ<sub>G</sub>")
    fig_gene.add_trace(
        go.Scatter(
            x=t,
            y=mu_G,
            mode="lines",
            line=dict(color=_BLUE, width=2),
            name="μ<sub>G</sub>",
            hovertemplate="t = %{x:.1f}<br>μ_G = %{y:.3f}<extra></extra>",
        )
    )
    fig_gene.update_yaxes(range=[-0.05, 1.08])
    _add_ss_line(fig_gene, t, _steady_state_estimate(mu_G), _BLUE, fmt=".2f")

    # ── Panel (b): RNA count μ_R ± σ_R
    fig_rna = go.Figure(
        layout=_base_layout(
            title="<b>b</b>  RNA count (ODE)",
            ylabel="μ<sub>R</sub> (molecules)",
            figsize=figsize,
        )
    )
    _add_sigma_band(fig_rna, t, mu_R, sigma_R, _RED_LIGHT, "± σ<sub>R</sub>")
    fig_rna.add_trace(
        go.Scatter(
            x=t,
            y=mu_R,
            mode="lines",
            line=dict(color=_RED, width=2),
            name="μ<sub>R</sub>",
            hovertemplate="t = %{x:.1f}<br>μ_R = %{y:.1f}<extra></extra>",
        )
    )
    y_top = float(np.nanmax(mu_R + sigma_R))
    y_bot = max(-0.5, float(np.nanmin(mu_R - sigma_R)) - 0.5)
    fig_rna.update_yaxes(range=[y_bot, y_top * 1.12] if y_top > 0 else None)
    _add_ss_line(fig_rna, t, _steady_state_estimate(mu_R), _RED, fmt=".1f")

    # ── Panel (c): Gene–RNA covariance C_RG
    # ODE output is already smooth — no rolling mean needed
    fig_cov = go.Figure(
        layout=_base_layout(
            title="<b>c</b>  Gene–RNA covariance (ODE)",
            ylabel="C<sub>RG</sub>",
            figsize=figsize,
        )
    )
    fig_cov.add_trace(
        go.Scatter(
            x=t,
            y=c_rg,
            mode="lines",
            fill="tozeroy",
            fillcolor=_PURPLE_LIGHT,
            line=dict(color=_PURPLE, width=2.5),
            name="C<sub>RG</sub>",
            hovertemplate="t = %{x:.1f}<br>C_RG = %{y:.3f}<extra></extra>",
        )
    )
    fig_cov.add_hline(y=0, line=dict(color="#94a3b8", width=0.8))
    _add_ss_line(fig_cov, t, _steady_state_estimate(c_rg), _PURPLE, fmt=".3f")

    return fig_gene, fig_rna, fig_cov

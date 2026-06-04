import numpy as np
import plotly.graph_objects as go

from src.ssa_visualization import (
    _base_layout,
    _add_sigma_band,
    _add_ss_line,
    _steady_state_estimate,
    _rolling_mean,
    _BLUE,
    _BLUE_LIGHT,
    _RED,
    _RED_LIGHT,
    _PURPLE,
    _PURPLE_LIGHT,
    _FONT,
)

# ── ODE-only colours (darker tones to distinguish from SSA) ─────────────────
_ODE_BLUE = "#1E40AF"
_ODE_RED = "#B91C1C"
_ODE_PURPLE = "#5B21B6"


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


def show_ssa_vs_ode(
    moments_ssa: dict,
    t_ode: np.ndarray,
    y_ode: np.ndarray,
    title_suffix: str = "",
    figsize: tuple[float, float] = (5.5, 3.2),
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Overlay SSA sample moments and ODE population moments on the same axes.

    Produces three figures (Gene state, RNA count, Covariance) where the SSA
    data is shown as a noisy band + line and the ODE solution as a smooth
    dashed curve.

    Parameters
    ----------
    moments_ssa : dict
        Output of :func:`ssa_simulation.compute_sample_moments`.
    t_ode : np.ndarray
        1-D time array from :func:`ode_moments.solve_ode_moments`.
    y_ode : np.ndarray
        2-D array (4 × N) from :func:`ode_moments.solve_ode_moments`.
    title_suffix : str, optional
        Extra text appended to each panel title (e.g. "— Slow switching").
    figsize : tuple[float, float], optional
        Width × height in inches.

    Returns
    -------
    tuple[go.Figure, go.Figure, go.Figure]
        ``(fig_gene, fig_rna, fig_cov)``
    """
    # ── Unpack SSA ──────────────────────────────────────────────────────
    t_ssa = moments_ssa["time"]
    mu_G_ssa = moments_ssa["mu_G"]
    mu_R_ssa = moments_ssa["mu_R"]
    sigma_G_ssa = moments_ssa["sigma_G"]
    sigma_R_ssa = moments_ssa["sigma_R"]
    cov_RG_ssa = moments_ssa["cov_RG"]

    # ── Unpack ODE ──────────────────────────────────────────────────────
    mu_G_ode, mu_R_ode, sigma2_R_ode, c_rg_ode = y_ode
    sigma_G_ode = np.sqrt(np.clip(mu_G_ode * (1.0 - mu_G_ode), 0, None))
    sigma_R_ode = np.sqrt(np.clip(sigma2_R_ode, 0, None))

    suffix = f" — {title_suffix}" if title_suffix else ""

    # ════════════════════════════════════════════════════════════════════
    #  Panel (a): Gene state ⟨G⟩
    # ════════════════════════════════════════════════════════════════════
    fig_gene = go.Figure(layout=_base_layout(
        title=f"<b>a</b>  Gene state{suffix}",
        ylabel="⟨G⟩ / μ<sub>G</sub>",
        figsize=figsize,
    ))
    # SSA band + line
    _add_sigma_band(fig_gene, t_ssa, mu_G_ssa, sigma_G_ssa,
                    _BLUE_LIGHT, "SSA ± σ<sub>G</sub>")
    fig_gene.add_trace(go.Scatter(
        x=t_ssa, y=mu_G_ssa, mode="lines",
        line=dict(color=_BLUE, width=1.5),
        name="SSA ⟨G⟩",
        hovertemplate="t = %{x:.1f}<br>SSA ⟨G⟩ = %{y:.3f}<extra></extra>",
    ))
    # ODE line
    fig_gene.add_trace(go.Scatter(
        x=t_ode, y=mu_G_ode, mode="lines",
        line=dict(color=_ODE_BLUE, width=2.5, dash="dash"),
        name="ODE μ<sub>G</sub>",
        hovertemplate="t = %{x:.1f}<br>ODE μ_G = %{y:.3f}<extra></extra>",
    ))
    fig_gene.update_yaxes(range=[-0.05, 1.08])

    # ════════════════════════════════════════════════════════════════════
    #  Panel (b): RNA count ⟨R⟩
    # ════════════════════════════════════════════════════════════════════
    fig_rna = go.Figure(layout=_base_layout(
        title=f"<b>b</b>  RNA count{suffix}",
        ylabel="⟨R⟩ / μ<sub>R</sub> (molecules)",
        figsize=figsize,
    ))
    # SSA band + line
    _add_sigma_band(fig_rna, t_ssa, mu_R_ssa, sigma_R_ssa,
                    _RED_LIGHT, "SSA ± σ<sub>R</sub>")
    fig_rna.add_trace(go.Scatter(
        x=t_ssa, y=mu_R_ssa, mode="lines",
        line=dict(color=_RED, width=1.5),
        name="SSA ⟨R⟩",
        hovertemplate="t = %{x:.1f}<br>SSA ⟨R⟩ = %{y:.1f}<extra></extra>",
    ))
    # ODE line
    fig_rna.add_trace(go.Scatter(
        x=t_ode, y=mu_R_ode, mode="lines",
        line=dict(color=_ODE_RED, width=2.5, dash="dash"),
        name="ODE μ<sub>R</sub>",
        hovertemplate="t = %{x:.1f}<br>ODE μ_R = %{y:.1f}<extra></extra>",
    ))
    # ODE ±σ band (lighter, distinct)
    _add_sigma_band(fig_rna, t_ode, mu_R_ode, sigma_R_ode,
                    "rgba(185, 28, 28, 0.08)", "ODE ± σ<sub>R</sub>")

    y_top = max(float(np.nanmax(mu_R_ssa + sigma_R_ssa)),
                float(np.nanmax(mu_R_ode + sigma_R_ode)))
    y_bot = min(max(-0.5, float(np.nanmin(mu_R_ssa - sigma_R_ssa)) - 0.5),
                max(-0.5, float(np.nanmin(mu_R_ode - sigma_R_ode)) - 0.5))
    fig_rna.update_yaxes(range=[y_bot, y_top * 1.12] if y_top > 0 else None)

    # ════════════════════════════════════════════════════════════════════
    #  Panel (c): Covariance Cov(R, G)
    # ════════════════════════════════════════════════════════════════════
    fig_cov = go.Figure(layout=_base_layout(
        title=f"<b>c</b>  Gene–RNA covariance{suffix}",
        ylabel="Cov(R, G)",
        figsize=figsize,
    ))
    # SSA: raw (faint) + smoothed
    win = max(5, len(t_ssa) // 50)
    cov_smooth = _rolling_mean(cov_RG_ssa, win)

    fig_cov.add_trace(go.Scatter(
        x=t_ssa, y=cov_RG_ssa, mode="lines",
        line=dict(color=_PURPLE, width=0.5),
        opacity=0.15,
        name="SSA raw",
        hoverinfo="skip",
    ))
    fig_cov.add_trace(go.Scatter(
        x=t_ssa, y=cov_smooth, mode="lines",
        line=dict(color=_PURPLE, width=1.5),
        name="SSA smoothed",
        hovertemplate="t = %{x:.1f}<br>SSA Cov = %{y:.3f}<extra></extra>",
    ))
    # ODE line
    fig_cov.add_trace(go.Scatter(
        x=t_ode, y=c_rg_ode, mode="lines",
        line=dict(color=_ODE_PURPLE, width=2.5, dash="dash"),
        name="ODE C<sub>RG</sub>",
        hovertemplate="t = %{x:.1f}<br>ODE C_RG = %{y:.3f}<extra></extra>",
    ))
    fig_cov.add_hline(y=0, line=dict(color="#94a3b8", width=0.8))

    return fig_gene, fig_rna, fig_cov

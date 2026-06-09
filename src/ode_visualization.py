import numpy as np
import plotly.graph_objects as go

from src.ssa_visualization import (
    analytical_steady_state,
    _base_layout,
    _add_sigma_band,
    _add_ss_line,
    _rolling_mean,
    _trim_edge,
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
    analytical: dict | None = None,
    figsize: tuple[float, float] = (4.5, 2.8),
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Plot the time evolution of ODE-based moments as three interactive figures.

    Parameters
    ----------
    t : np.ndarray
        1-D time array returned by :func:`solve_ode_moments`.
    y : np.ndarray
        2-D array (4 × N). Rows: μ_G, μ_R, σ²_R, C_RG.
    analytical : dict | None, optional
        Exact steady-state values for reference lines.
    figsize : tuple[float, float], optional
        Width × height in inches.

    Returns
    -------
    tuple[go.Figure, go.Figure, go.Figure]
        ``(fig_gene, fig_rna, fig_cov)``
    """
    mu_G, mu_R, sigma2_R, c_rg = y
    sigma_G = np.sqrt(np.clip(mu_G * (1.0 - mu_G), 0, None))
    sigma_R = np.sqrt(np.clip(sigma2_R, 0, None))

    # Steady-state
    if analytical is not None:
        ss_G, ss_R, ss_cov = analytical["mu_G"], analytical["mu_R"], analytical["cov_RG"]
        ss_label = "analytical"
    else:
        tail = max(1, len(mu_G) // 10)
        ss_G = float(mu_G[-tail:].mean())
        ss_R = float(mu_R[-tail:].mean())
        ss_cov = float(c_rg[-tail:].mean())
        ss_label = "ss"

    # ── Panel (a): Gene state
    fig_gene = go.Figure(layout=_base_layout(
        title="<b>a</b>  Gene state (ODE)", ylabel="μ<sub>G</sub>", figsize=figsize,
    ))
    _add_sigma_band(fig_gene, t, mu_G, sigma_G, _BLUE_LIGHT,
                    "± σ<sub>G</sub>", clip_lo=0.0, clip_hi=1.0)
    fig_gene.add_trace(go.Scatter(
        x=t, y=mu_G, mode="lines",
        line=dict(color=_BLUE, width=2.2),
        name="μ<sub>G</sub>",
        hovertemplate="t = %{x:.1f}<br>μ_G = %{y:.3f}<extra></extra>",
    ))
    fig_gene.update_yaxes(range=[-0.05, 1.08])
    _add_ss_line(fig_gene, ss_G, _BLUE, fmt=".3f", label=ss_label)

    # ── Panel (b): RNA count
    fig_rna = go.Figure(layout=_base_layout(
        title="<b>b</b>  RNA count (ODE)", ylabel="μ<sub>R</sub> (molecules)", figsize=figsize,
    ))
    _add_sigma_band(fig_rna, t, mu_R, sigma_R, _RED_LIGHT,
                    "± σ<sub>R</sub>", clip_lo=0.0)
    fig_rna.add_trace(go.Scatter(
        x=t, y=mu_R, mode="lines",
        line=dict(color=_RED, width=2.2),
        name="μ<sub>R</sub>",
        hovertemplate="t = %{x:.1f}<br>μ_R = %{y:.1f}<extra></extra>",
    ))
    y_top = float(np.nanmax(mu_R + sigma_R))
    y_bot = max(-0.5, float(np.nanmin(np.clip(mu_R - sigma_R, 0, None))) - 0.5)
    fig_rna.update_yaxes(range=[y_bot, y_top * 1.12] if y_top > 0 else None)
    _add_ss_line(fig_rna, ss_R, _RED, fmt=".1f", label=ss_label)

    # ── Panel (c): Covariance
    fig_cov = go.Figure(layout=_base_layout(
        title="<b>c</b>  Gene–RNA covariance (ODE)", ylabel="C<sub>RG</sub>", figsize=figsize,
    ))
    fig_cov.add_trace(go.Scatter(
        x=t, y=c_rg, mode="lines", fill="tozeroy",
        fillcolor=_PURPLE_LIGHT,
        line=dict(color=_PURPLE, width=2.2),
        name="C<sub>RG</sub>",
        hovertemplate="t = %{x:.1f}<br>C_RG = %{y:.3f}<extra></extra>",
    ))
    fig_cov.add_hline(y=0, line=dict(color="#94a3b8", width=0.8))
    _add_ss_line(fig_cov, ss_cov, _PURPLE, fmt=".4f", label=ss_label)

    return fig_gene, fig_rna, fig_cov


def show_ssa_vs_ode(
    moments_ssa: dict,
    t_ode: np.ndarray,
    y_ode: np.ndarray,
    analytical: dict | None = None,
    title_suffix: str = "",
    figsize: tuple[float, float] = (5.5, 3.2),
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Overlay SSA sample moments and ODE population moments on the same axes.

    Parameters
    ----------
    moments_ssa : dict
        Output of :func:`ssa_simulation.compute_sample_moments`.
    t_ode : np.ndarray
        1-D time array from :func:`ode_moments.solve_ode_moments`.
    y_ode : np.ndarray
        2-D array (4 × N) from :func:`ode_moments.solve_ode_moments`.
    analytical : dict | None, optional
        Exact steady-state values for reference lines.
    title_suffix : str, optional
        Extra text appended to each panel title.
    figsize : tuple[float, float], optional
        Width × height in inches.

    Returns
    -------
    tuple[go.Figure, go.Figure, go.Figure]
        ``(fig_gene, fig_rna, fig_cov)``
    """
    # ── Unpack + trim SSA ───────────────────────────────────────────────
    t_ssa, mu_G_ssa, mu_R_ssa, sigma_G_ssa, sigma_R_ssa, cov_RG_ssa = _trim_edge(
        moments_ssa["time"],
        moments_ssa["mu_G"], moments_ssa["mu_R"],
        moments_ssa["sigma_G"], moments_ssa["sigma_R"],
        moments_ssa["cov_RG"],
    )

    # Consistent smoothing
    win = max(5, len(t_ssa) // 30)
    mu_G_sm = _rolling_mean(mu_G_ssa, win)
    mu_R_sm = _rolling_mean(mu_R_ssa, win)
    sigma_G_sm = _rolling_mean(sigma_G_ssa, win)
    sigma_R_sm = _rolling_mean(sigma_R_ssa, win)
    cov_sm = _rolling_mean(cov_RG_ssa, win)

    # ── Unpack ODE ──────────────────────────────────────────────────────
    mu_G_ode, mu_R_ode, sigma2_R_ode, c_rg_ode = y_ode
    sigma_G_ode = np.sqrt(np.clip(mu_G_ode * (1.0 - mu_G_ode), 0, None))
    sigma_R_ode = np.sqrt(np.clip(sigma2_R_ode, 0, None))

    suffix = f" — {title_suffix}" if title_suffix else ""

    # ═══════════════════════════════════════════════════════════════════
    #  Panel (a): Gene state
    # ═══════════════════════════════════════════════════════════════════
    fig_gene = go.Figure(layout=_base_layout(
        title=f"<b>a</b>  Gene state{suffix}",
        ylabel="⟨G⟩ / μ<sub>G</sub>",
        figsize=figsize,
    ))
    _add_sigma_band(fig_gene, t_ssa, mu_G_sm, sigma_G_sm,
                    _BLUE_LIGHT, "SSA ± σ<sub>G</sub>",
                    clip_lo=0.0, clip_hi=1.0)
    fig_gene.add_trace(go.Scatter(
        x=t_ssa, y=mu_G_sm, mode="lines",
        line=dict(color=_BLUE, width=1.8),
        name="SSA ⟨G⟩",
        hovertemplate="t = %{x:.1f}<br>SSA ⟨G⟩ = %{y:.3f}<extra></extra>",
    ))
    fig_gene.add_trace(go.Scatter(
        x=t_ode, y=mu_G_ode, mode="lines",
        line=dict(color=_ODE_BLUE, width=2.5, dash="dash"),
        name="ODE μ<sub>G</sub>",
        hovertemplate="t = %{x:.1f}<br>ODE μ_G = %{y:.3f}<extra></extra>",
    ))
    fig_gene.update_yaxes(range=[-0.05, 1.08])
    if analytical is not None:
        _add_ss_line(fig_gene, analytical["mu_G"], _BLUE, fmt=".3f", label="analytical")

    # ═══════════════════════════════════════════════════════════════════
    #  Panel (b): RNA count
    # ═══════════════════════════════════════════════════════════════════
    fig_rna = go.Figure(layout=_base_layout(
        title=f"<b>b</b>  RNA count{suffix}",
        ylabel="⟨R⟩ / μ<sub>R</sub> (molecules)",
        figsize=figsize,
    ))
    _add_sigma_band(fig_rna, t_ssa, mu_R_sm, sigma_R_sm,
                    _RED_LIGHT, "SSA ± σ<sub>R</sub>", clip_lo=0.0)
    fig_rna.add_trace(go.Scatter(
        x=t_ssa, y=mu_R_sm, mode="lines",
        line=dict(color=_RED, width=1.8),
        name="SSA ⟨R⟩",
        hovertemplate="t = %{x:.1f}<br>SSA ⟨R⟩ = %{y:.1f}<extra></extra>",
    ))
    fig_rna.add_trace(go.Scatter(
        x=t_ode, y=mu_R_ode, mode="lines",
        line=dict(color=_ODE_RED, width=2.5, dash="dash"),
        name="ODE μ<sub>R</sub>",
        hovertemplate="t = %{x:.1f}<br>ODE μ_R = %{y:.1f}<extra></extra>",
    ))
    _add_sigma_band(fig_rna, t_ode, mu_R_ode, sigma_R_ode,
                    "rgba(185, 28, 28, 0.06)", "ODE ± σ<sub>R</sub>", clip_lo=0.0)
    y_top = max(float(np.nanmax(mu_R_sm + sigma_R_sm)),
                float(np.nanmax(mu_R_ode + sigma_R_ode)))
    fig_rna.update_yaxes(range=[-0.5, y_top * 1.12] if y_top > 0 else None)
    if analytical is not None:
        _add_ss_line(fig_rna, analytical["mu_R"], _RED, fmt=".1f", label="analytical")

    # ═══════════════════════════════════════════════════════════════════
    #  Panel (c): Covariance
    # ═══════════════════════════════════════════════════════════════════
    fig_cov = go.Figure(layout=_base_layout(
        title=f"<b>c</b>  Gene–RNA covariance{suffix}",
        ylabel="Cov(R, G)",
        figsize=figsize,
    ))
    fig_cov.add_trace(go.Scatter(
        x=t_ssa, y=cov_sm, mode="lines",
        line=dict(color=_PURPLE, width=1.8),
        name="SSA Cov",
        hovertemplate="t = %{x:.1f}<br>SSA Cov = %{y:.3f}<extra></extra>",
    ))
    fig_cov.add_trace(go.Scatter(
        x=t_ode, y=c_rg_ode, mode="lines",
        line=dict(color=_ODE_PURPLE, width=2.5, dash="dash"),
        name="ODE C<sub>RG</sub>",
        hovertemplate="t = %{x:.1f}<br>ODE C_RG = %{y:.3f}<extra></extra>",
    ))
    fig_cov.add_hline(y=0, line=dict(color="#94a3b8", width=0.8))
    if analytical is not None:
        _add_ss_line(fig_cov, analytical["cov_RG"], _PURPLE, fmt=".4f", label="analytical")

    return fig_gene, fig_rna, fig_cov

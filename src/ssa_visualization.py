import numpy as np
import plotly.graph_objects as go

# ── colour palette ──────────────────────────────────────────────────────────
_BLUE = "#3B82F6"       # gene state G  (vivid blue)
_BLUE_LIGHT = "rgba(59, 130, 246, 0.12)"
_RED = "#EF4444"         # RNA count R   (vivid red)
_RED_LIGHT = "rgba(239, 68, 68, 0.12)"
_PURPLE = "#8B5CF6"      # covariance    (vivid purple)
_PURPLE_LIGHT = "rgba(139, 92, 246, 0.12)"
_SS_DASH = dict(dash="dot", width=1.5)
_FONT = "Inter, SF Pro Display, -apple-system, Helvetica Neue, Arial, sans-serif"


# ── shared layout ───────────────────────────────────────────────────────────
def _base_layout(
    title: str,
    ylabel: str,
    figsize: tuple[float, float],
) -> dict:
    """Return a Plotly layout dict with a clean, modern scientific look."""
    w_px = int(figsize[0] * 150)   # scale inches → pixels
    h_px = int(figsize[1] * 150)
    return dict(
        width=w_px,
        height=h_px,
        template="plotly_white",
        font=dict(family=_FONT, size=13, color="#1e293b"),
        title=dict(
            text=title,
            font=dict(size=15, color="#0f172a", family=_FONT),
            x=0.03, xanchor="left", y=0.97, yanchor="top",
        ),
        xaxis=dict(
            title=dict(text="Time (a.u.)", standoff=6),
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            gridwidth=1,
            zeroline=False,
            linecolor="#cbd5e1",
            linewidth=1,
            tickfont=dict(size=11),
            mirror=False,
            showline=True,
        ),
        yaxis=dict(
            title=dict(text=ylabel, standoff=6),
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            gridwidth=1,
            zeroline=False,
            linecolor="#cbd5e1",
            linewidth=1,
            tickfont=dict(size=11),
            mirror=False,
            showline=True,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=24, t=50, b=52),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(203,213,225,0.5)",
            borderwidth=1,
            font=dict(size=11),
            x=1, xanchor="right",
            y=1, yanchor="top",
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#94a3b8",
            font=dict(size=12, family=_FONT, color="#1e293b"),
        ),
    )


# ── helpers ─────────────────────────────────────────────────────────────────
def _steady_state_estimate(y: np.ndarray) -> float:
    """Average the last 10 % of a time-series as a steady-state proxy."""
    tail = max(1, len(y) // 10)
    return float(y[-tail:].mean())


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling average; edges are padded with the original values."""
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="same")
    # Fix edge artifacts from convolution
    half = window // 2
    smoothed[:half] = y[:half]
    smoothed[-half:] = y[-half:]
    return smoothed


def _add_sigma_band(
    fig: go.Figure,
    t: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    color_fill: str,
    name: str,
) -> None:
    """Add a shaded ±σ band (upper + lower boundary as a filled area)."""
    upper = mu + sigma
    lower = mu - sigma
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor=color_fill,
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=True,
        name=name,
        legendgroup=name,
    ))


def _add_ss_line(
    fig: go.Figure,
    t: np.ndarray,
    ss_val: float,
    color: str,
    fmt: str = ".2f",
) -> None:
    """Draw a dashed steady-state reference line with an annotation."""
    fig.add_hline(
        y=ss_val,
        line=dict(color=color, **_SS_DASH),
        opacity=0.55,
    )
    fig.add_annotation(
        x=t[-1],
        y=ss_val,
        text=f"<b>ss = {ss_val:{fmt}}</b>",
        showarrow=False,
        font=dict(size=10, color=color, family=_FONT),
        xanchor="right",
        yanchor="bottom",
        yshift=5,
    )





# ── public API ──────────────────────────────────────────────────────────────
def show_sample_moments(
    moments: dict,
    figsize: tuple[float, float] = (4.5, 2.8),
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Plot the time evolution of sample moments as three interactive Plotly figures.

    Each figure shows one statistic of the telegraph-model SSA ensemble:

    * **Gene state** ⟨G⟩ ± σ_G  (panel **a**)
    * **RNA count**  ⟨R⟩ ± σ_R  (panel **b**)
    * **Gene–RNA covariance** Cov(R, G)  (panel **c**)

    A dashed line marks the steady-state estimate (mean of the last 10 %
    of the time-series).  All plots are interactive: hover for values,
    zoom, pan, and export via the toolbar.

    Parameters
    ----------
    moments : dict
        Output of :func:`ssa_simulation.compute_sample_moments`.
        Required keys: ``time``, ``mu_G``, ``mu_R``, ``sigma_G``,
        ``sigma_R``, ``cov_RG``.
    figsize : tuple[float, float], optional
        Width × height in inches (converted to pixels at 150 dpi).

    Returns
    -------
    tuple[go.Figure, go.Figure, go.Figure]
        ``(fig_gene, fig_rna, fig_cov)``
    """
    t = moments["time"]
    mu_G = moments["mu_G"]
    mu_R = moments["mu_R"]
    sigma_G = moments["sigma_G"]
    sigma_R = moments["sigma_R"]
    cov_RG = moments["cov_RG"]

    # ── Panel (a): Gene state ⟨G⟩ ± σ ──────────────────────────────────
    fig_gene = go.Figure(layout=_base_layout(
        title="<b>a</b>  Gene state", ylabel="⟨G⟩", figsize=figsize,
    ))
    _add_sigma_band(fig_gene, t, mu_G, sigma_G, _BLUE_LIGHT, "± σ<sub>G</sub>")
    fig_gene.add_trace(go.Scatter(
        x=t, y=mu_G, mode="lines",
        line=dict(color=_BLUE, width=2),
        name="⟨G⟩",
        hovertemplate="t = %{x:.1f}<br>⟨G⟩ = %{y:.3f}<extra></extra>",
    ))
    fig_gene.update_yaxes(range=[-0.05, 1.08])
    ss_G = _steady_state_estimate(mu_G)
    _add_ss_line(fig_gene, t, ss_G, _BLUE, fmt=".2f")

    # ── Panel (b): RNA count ⟨R⟩ ± σ ───────────────────────────────────
    fig_rna = go.Figure(layout=_base_layout(
        title="<b>b</b>  RNA count", ylabel="⟨R⟩ (molecules)", figsize=figsize,
    ))
    _add_sigma_band(fig_rna, t, mu_R, sigma_R, _RED_LIGHT, "± σ<sub>R</sub>")
    fig_rna.add_trace(go.Scatter(
        x=t, y=mu_R, mode="lines",
        line=dict(color=_RED, width=2),
        name="⟨R⟩",
        hovertemplate="t = %{x:.1f}<br>⟨R⟩ = %{y:.1f}<extra></extra>",
    ))
    y_top = float(np.nanmax(mu_R + sigma_R))
    y_bot = max(-0.5, float(np.nanmin(mu_R - sigma_R)) - 0.5)
    fig_rna.update_yaxes(range=[y_bot, y_top * 1.12] if y_top > 0 else None)
    ss_R = _steady_state_estimate(mu_R)
    _add_ss_line(fig_rna, t, ss_R, _RED, fmt=".1f")

    # ── Panel (c): Covariance Cov(R, G) ────────────────────────────────
    fig_cov = go.Figure(layout=_base_layout(
        title="<b>c</b>  Gene–RNA covariance",
        ylabel="Cov(R, G)",
        figsize=figsize,
    ))
    # Smoothing window: ~2 % of data points, at least 5
    win = max(5, len(t) // 50)
    cov_smooth = _rolling_mean(cov_RG, win)

    # Raw data as a faint background trace
    fig_cov.add_trace(go.Scatter(
        x=t, y=cov_RG, mode="lines",
        line=dict(color=_PURPLE, width=0.5),
        opacity=0.2,
        name="raw",
        hoverinfo="skip",
        showlegend=True,
    ))
    # Smoothed line with filled area
    fig_cov.add_trace(go.Scatter(
        x=t, y=cov_smooth, mode="lines",
        fill="tozeroy",
        fillcolor=_PURPLE_LIGHT,
        line=dict(color=_PURPLE, width=2.5),
        name="smoothed",
        hovertemplate="t = %{x:.1f}<br>Cov = %{y:.3f}<extra></extra>",
    ))
    fig_cov.add_hline(y=0, line=dict(color="#94a3b8", width=0.8))


    return fig_gene, fig_rna, fig_cov

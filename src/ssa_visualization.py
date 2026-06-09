import numpy as np
import plotly.graph_objects as go

# ── colour palette ──────────────────────────────────────────────────────────
_BLUE = "#3B82F6"       # gene state G
_BLUE_LIGHT = "rgba(59, 130, 246, 0.08)"
_RED = "#EF4444"         # RNA count R
_RED_LIGHT = "rgba(239, 68, 68, 0.08)"
_PURPLE = "#8B5CF6"      # covariance
_PURPLE_LIGHT = "rgba(139, 92, 246, 0.08)"
_SS_DASH = dict(dash="dot", width=1.2)
_FONT = "Inter, SF Pro Display, -apple-system, Helvetica Neue, Arial, sans-serif"


# ── analytical steady state ─────────────────────────────────────────────────
def analytical_steady_state(
    k_on: float, k_off: float, k_syn: float, k_deg: float,
) -> dict:
    """Exact closed-form steady-state moments of the telegraph model.

    Returns
    -------
    dict
        Keys: ``mu_G``, ``mu_R``, ``sigma_G``, ``sigma_R``, ``cov_RG``.
    """
    mu_G = k_on / (k_on + k_off) if (k_on + k_off) > 0 else float(k_on > 0)
    mu_R = k_syn * mu_G / k_deg if k_deg > 0 else np.inf
    var_G = mu_G * (1.0 - mu_G)
    cov_RG = k_syn * var_G / (k_on + k_off + k_deg) if (k_on + k_off + k_deg) > 0 else 0.0
    var_R = mu_R + k_syn * cov_RG / k_deg if k_deg > 0 else np.inf
    return {
        "mu_G": mu_G,
        "mu_R": mu_R,
        "sigma_G": np.sqrt(var_G),
        "sigma_R": np.sqrt(max(var_R, 0.0)),
        "cov_RG": cov_RG,
    }


# ── shared layout ───────────────────────────────────────────────────────────
def _base_layout(
    title: str,
    ylabel: str,
    figsize: tuple[float, float],
) -> dict:
    """Return a Plotly layout dict with a clean, modern scientific look."""
    w_px = int(figsize[0] * 200)
    h_px = int(figsize[1] * 200)
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
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(203,213,225,0.4)",
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
def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling average; edges are padded with the original values."""
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="same")
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
    clip_lo: float | None = None,
    clip_hi: float | None = None,
) -> None:
    """Add a shaded ±σ band, optionally clipped to [clip_lo, clip_hi]."""
    upper = mu + sigma
    lower = mu - sigma
    if clip_lo is not None:
        lower = np.clip(lower, clip_lo, None)
    if clip_hi is not None:
        upper = np.clip(upper, None, clip_hi)
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
    ss_val: float,
    color: str,
    fmt: str = ".2f",
    label: str = "ss",
) -> None:
    """Draw a subtle dashed steady-state line + corner annotation."""
    if not np.isfinite(ss_val):
        return
    # Subtle dotted reference line
    fig.add_hline(
        y=ss_val,
        line=dict(color=color, **_SS_DASH),
        opacity=0.4,
    )
    # Annotation in the bottom-left corner, out of the data's way
    fig.add_annotation(
        text=f"<b>{label} = {ss_val:{fmt}}</b>",
        showarrow=False,
        font=dict(size=10, color=color, family=_FONT),
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        xanchor="left", yanchor="bottom",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=color,
        borderwidth=1,
        borderpad=3,
    )


def _trim_edge(t, *arrays, trim_frac=0.02):
    """Trim the last trim_frac of data to remove SSA edge artifacts."""
    n = len(t)
    cut = max(1, int(n * (1.0 - trim_frac)))
    return (t[:cut],) + tuple(a[:cut] for a in arrays)


# ── public API ──────────────────────────────────────────────────────────────
def show_sample_moments(
    moments: dict,
    analytical: dict | None = None,
    figsize: tuple[float, float] = (4.5, 2.8),
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Plot the time evolution of sample moments as three interactive figures.

    Each figure shows one statistic of the telegraph-model SSA ensemble:

    * **Gene state** ⟨G⟩ ± σ_G  (panel **a**)
    * **RNA count**  ⟨R⟩ ± σ_R  (panel **b**)
    * **Gene–RNA covariance** Cov(R, G)  (panel **c**)

    Parameters
    ----------
    moments : dict
        Output of :func:`ssa_simulation.compute_sample_moments`.
        Required keys: ``time``, ``mu_G``, ``mu_R``, ``sigma_G``,
        ``sigma_R``, ``cov_RG``.
    analytical : dict | None, optional
        Exact analytical steady-state values from
        :func:`analytical_steady_state`.  If provided, these are
        used for the reference lines.
    figsize : tuple[float, float], optional
        Width × height in inches (converted to pixels at 200 dpi).

    Returns
    -------
    tuple[go.Figure, go.Figure, go.Figure]
        ``(fig_gene, fig_rna, fig_cov)``
    """
    # Trim edge artifacts (last 2 % of SSA data)
    t, mu_G, mu_R, sigma_G, sigma_R, cov_RG = _trim_edge(
        moments["time"],
        moments["mu_G"], moments["mu_R"],
        moments["sigma_G"], moments["sigma_R"],
        moments["cov_RG"],
    )

    # ── Smoothing (consistent across all panels) ────────────────────────
    win = max(5, len(t) // 30)

    mu_G_s = _rolling_mean(mu_G, win)
    mu_R_s = _rolling_mean(mu_R, win)
    sigma_G_s = _rolling_mean(sigma_G, win)
    sigma_R_s = _rolling_mean(sigma_R, win)
    cov_s = _rolling_mean(cov_RG, win)

    # ── Steady-state values ─────────────────────────────────────────────
    if analytical is not None:
        ss_G = analytical["mu_G"]
        ss_R = analytical["mu_R"]
        ss_cov = analytical["cov_RG"]
        ss_label = "analytical"
    else:
        tail = max(1, len(mu_G) // 10)
        ss_G = float(mu_G[-tail:].mean())
        ss_R = float(mu_R[-tail:].mean())
        ss_cov = float(cov_RG[-tail:].mean())
        ss_label = "ss"

    # ═══════════════════════════════════════════════════════════════════
    #  Panel (a): Gene state ⟨G⟩ ± σ — clipped to [0, 1]
    # ═══════════════════════════════════════════════════════════════════
    fig_gene = go.Figure(layout=_base_layout(
        title="<b>a</b>  Gene state", ylabel="⟨G⟩", figsize=figsize,
    ))

    _add_sigma_band(fig_gene, t, mu_G_s, sigma_G_s,
                    _BLUE_LIGHT, "± σ<sub>G</sub>",
                    clip_lo=0.0, clip_hi=1.0)

    fig_gene.add_trace(go.Scatter(
        x=t, y=mu_G_s, mode="lines",
        line=dict(color=_BLUE, width=2.2),
        name="⟨G⟩",
        hovertemplate="t = %{x:.1f}<br>⟨G⟩ = %{y:.3f}<extra></extra>",
    ))

    fig_gene.update_yaxes(range=[-0.05, 1.08])
    _add_ss_line(fig_gene, ss_G, _BLUE, fmt=".3f", label=ss_label)

    # ═══════════════════════════════════════════════════════════════════
    #  Panel (b): RNA count ⟨R⟩ ± σ
    # ═══════════════════════════════════════════════════════════════════
    fig_rna = go.Figure(layout=_base_layout(
        title="<b>b</b>  RNA count", ylabel="⟨R⟩ (molecules)", figsize=figsize,
    ))

    _add_sigma_band(fig_rna, t, mu_R_s, sigma_R_s,
                    _RED_LIGHT, "± σ<sub>R</sub>",
                    clip_lo=0.0)

    fig_rna.add_trace(go.Scatter(
        x=t, y=mu_R_s, mode="lines",
        line=dict(color=_RED, width=2.2),
        name="⟨R⟩",
        hovertemplate="t = %{x:.1f}<br>⟨R⟩ = %{y:.1f}<extra></extra>",
    ))

    y_top = float(np.nanmax(mu_R_s + sigma_R_s))
    y_bot = max(-0.5, float(np.nanmin(np.clip(mu_R_s - sigma_R_s, 0, None))) - 0.5)
    fig_rna.update_yaxes(range=[y_bot, y_top * 1.12] if y_top > 0 else None)
    _add_ss_line(fig_rna, ss_R, _RED, fmt=".1f", label=ss_label)

    # ═══════════════════════════════════════════════════════════════════
    #  Panel (c): Covariance Cov(R, G)
    # ═══════════════════════════════════════════════════════════════════
    fig_cov = go.Figure(layout=_base_layout(
        title="<b>c</b>  Gene–RNA covariance",
        ylabel="Cov(R, G)",
        figsize=figsize,
    ))

    fig_cov.add_trace(go.Scatter(
        x=t, y=cov_s, mode="lines",
        fill="tozeroy",
        fillcolor=_PURPLE_LIGHT,
        line=dict(color=_PURPLE, width=2.2),
        name="Cov(R, G)",
        hovertemplate="t = %{x:.1f}<br>Cov = %{y:.3f}<extra></extra>",
    ))

    fig_cov.add_hline(y=0, line=dict(color="#94a3b8", width=0.8))
    _add_ss_line(fig_cov, ss_cov, _PURPLE, fmt=".4f", label=ss_label)

    return fig_gene, fig_rna, fig_cov

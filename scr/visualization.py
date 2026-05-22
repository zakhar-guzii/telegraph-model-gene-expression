import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec


_NATURE = {
    # Typography
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          7,          
    "axes.titlesize":     7,
    "axes.labelsize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    6,
    # Lines & ticks
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "lines.linewidth":    1.0,
    # Grid & background
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    # Saving
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
}

_BLUE   = "#2166AC"   # gene state G
_RED    = "#D6604D"   # RNA count  R
_PURPLE = "#762A83"   # covariance Cov(R, G)
_ALPHA  = 0.15        # confidence-band fill opacity


def show_sample_moments(
    moments: dict,
    save_path: str | None = None,
    figsize: tuple[float, float] = (3.35, 5.5),
) -> plt.Figure:
    """Plots the time evolution of the calculated sample moments.

    Creates a 3-panel figure showing:
    - Panel A: Mean gene state with +/- 1 SD band (clipped to [0, 1]).
    - Panel B: Mean RNA count with +/- 1 SD band (clipped to >= 0).
    - Panel C: Sample covariance between RNA count and gene state.

    Args:
        moments (dict): Dictionary from `compute_sample_moments` containing
            1-D arrays: 'time', 'mu_G', 'mu_R', 'sigma_G', 'sigma_R', 'cov_RG'.
        save_path (str | None): File path to save the generated plot.
            If None, the figure is only returned.
        figsize (tuple): Width and height of the figure in inches.

    Returns:
        matplotlib.figure.Figure: The rendered figure object.
    """
    t       = moments["time"]
    mu_G    = moments["mu_G"]
    mu_R    = moments["mu_R"]
    sigma_G = moments["sigma_G"]
    sigma_R = moments["sigma_R"]
    cov_RG  = moments["cov_RG"]

    # ------------------------------------------------------------------
    # Build figure with shared x-axis and tight vertical spacing
    # ------------------------------------------------------------------
    with plt.rc_context(_NATURE):
        fig = plt.figure(figsize=figsize)

        gs = GridSpec(
            nrows=3, ncols=1,
            figure=fig,
            hspace=0.45,          # breathing room between panels
            left=0.18, right=0.95,
            top=0.96, bottom=0.08,
        )

        ax_G   = fig.add_subplot(gs[0])
        ax_R   = fig.add_subplot(gs[1], sharex=ax_G)
        ax_cov = fig.add_subplot(gs[2], sharex=ax_G)

        # ------------------------------------------------------------------
        # Panel A — Gene state  E[G(t)]
        # ------------------------------------------------------------------
        ax_G.fill_between(
            t, mu_G - sigma_G, mu_G + sigma_G,
            color=_BLUE, alpha=_ALPHA, linewidth=0,
        )
        ax_G.plot(t, mu_G, color=_BLUE, linewidth=1.0)

        ax_G.set_ylabel(r"$\langle G \rangle$")
        ax_G.set_ylim(-0.05, 1.05)
        ax_G.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax_G.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax_G.set_title("Gene state", loc="left", pad=3)

        # Annotate steady-state level as a dashed reference line
        g_ss = float(mu_G[-int(len(mu_G) * 0.1):].mean())   # last 10 % mean
        ax_G.axhline(g_ss, color=_BLUE, linewidth=0.5,
                     linestyle="--", alpha=0.6)
        ax_G.text(
            t[-1], g_ss + 0.03, f"ss = {g_ss:.2f}",
            ha="right", va="bottom", color=_BLUE, fontsize=5,
        )

        # ------------------------------------------------------------------
        # Panel B — RNA count  E[R(t)]
        # ------------------------------------------------------------------
        ax_R.fill_between(
            t, mu_R - sigma_R, mu_R + sigma_R,
            color=_RED, alpha=_ALPHA, linewidth=0,
        )
        ax_R.plot(t, mu_R, color=_RED, linewidth=1.0)

        ax_R.set_ylabel(r"$\langle R \rangle$  (molecules)")
        ax_R.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))
        ax_R.set_title("RNA count", loc="left", pad=3)

        r_ss = float(mu_R[-int(len(mu_R) * 0.1):].mean())
        ax_R.axhline(r_ss, color=_RED, linewidth=0.5,
                     linestyle="--", alpha=0.6)
        ax_R.text(
            t[-1], r_ss * 1.02, f"ss = {r_ss:.1f}",
            ha="right", va="bottom", color=_RED, fontsize=5,
        )

        # ------------------------------------------------------------------
        # Panel C — Covariance  Cov(R, G)(t)
        # ------------------------------------------------------------------
        # Thin zero reference line first, so the data sits on top.
        ax_cov.axhline(0, color="black", linewidth=0.4, linestyle="-", alpha=0.4)
        ax_cov.plot(t, cov_RG, color=_PURPLE, linewidth=1.0)

        ax_cov.set_ylabel(r"$\mathrm{Cov}(R,\,G)$")
        ax_cov.set_xlabel("Time  (a.u.)")
        ax_cov.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax_cov.set_title("Gene–RNA covariance", loc="left", pad=3)

        # ------------------------------------------------------------------
        # Shared x-axis cosmetics
        # ------------------------------------------------------------------
        ax_G.tick_params(labelbottom=False)   # hide x labels on top panels
        ax_R.tick_params(labelbottom=False)

        for ax in (ax_G, ax_R, ax_cov):
            ax.spines["left"].set_position(("outward", 3))
            ax.spines["bottom"].set_position(("outward", 3))

        # ------------------------------------------------------------------
        # Panel labels  (a, b, c) — upper-left, bold, 8 pt
        # ------------------------------------------------------------------
        for ax, label in zip((ax_G, ax_R, ax_cov), "abc"):
            ax.text(
                -0.16, 1.08, label,
                transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top",
            )

        # ------------------------------------------------------------------
        # Optional: save to file (PDF / SVG recommended for vector output)
        # ------------------------------------------------------------------
        if save_path is not None:
            fig.savefig(save_path)

    return fig
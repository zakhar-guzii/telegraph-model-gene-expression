import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_BLUE = "#2166AC"  # gene state G
_RED = "#D6604D"  # RNA count  R
_PURPLE = "#762A83"  # covariance Cov(R, G)
_ALPHA = 0.15

_NATURE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def show_sample_moments(
    moments: dict,
    save_path: str | None = None,
    figsize: tuple[float, float] = (3.35, 2.0),
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Plot the time evolution of sample moments as three separate figures."""
    t = moments["time"]
    mu_G = moments["mu_G"]
    mu_R = moments["mu_R"]
    sigma_G = moments["sigma_G"]
    sigma_R = moments["sigma_R"]
    cov_RG = moments["cov_RG"]

    with plt.rc_context(_NATURE):

        # 1. Gene State Figure
        fig_gene, ax_G = plt.subplots(figsize=figsize)
        ax_G.fill_between(
            t, mu_G - sigma_G, mu_G + sigma_G, color=_BLUE, alpha=_ALPHA, linewidth=0
        )
        ax_G.plot(t, mu_G, color=_BLUE, linewidth=1.0)
        ax_G.set_ylabel(r"$\langle G \rangle$")
        ax_G.set_xlabel("Time  (a.u.)")
        ax_G.set_ylim(-0.05, 1.05)
        ax_G.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax_G.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax_G.set_title("Gene state", loc="left", pad=3)

        ss_G = float(mu_G[-max(1, len(mu_G) // 10) :].mean())
        ax_G.axhline(ss_G, color=_BLUE, linewidth=0.5, linestyle="--", alpha=0.6)
        ax_G.text(
            t[-1],
            ss_G + (0.03 if ss_G == 0 else ss_G * 0.04),
            f"ss = {ss_G:.2f}",
            ha="right",
            va="bottom",
            color=_BLUE,
            fontsize=5,
        )

        # 2. RNA Count Figure
        fig_rna, ax_R = plt.subplots(figsize=figsize)
        ax_R.fill_between(
            t, mu_R - sigma_R, mu_R + sigma_R, color=_RED, alpha=_ALPHA, linewidth=0
        )
        ax_R.plot(t, mu_R, color=_RED, linewidth=1.0)
        ax_R.set_ylabel(r"$\langle R \rangle$  (molecules)")
        ax_R.set_xlabel("Time  (a.u.)")
        ax_R.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))
        ax_R.set_title("RNA count", loc="left", pad=3)

        ss_R = float(mu_R[-max(1, len(mu_R) // 10) :].mean())
        ax_R.axhline(ss_R, color=_RED, linewidth=0.5, linestyle="--", alpha=0.6)
        ax_R.text(
            t[-1],
            ss_R + (0.03 if ss_R == 0 else ss_R * 0.04),
            f"ss = {ss_R:.1f}",
            ha="right",
            va="bottom",
            color=_RED,
            fontsize=5,
        )

        #  3. Covariance Figure
        fig_cov, ax_cov = plt.subplots(figsize=figsize)
        ax_cov.axhline(0, color="black", linewidth=0.4, linestyle="-", alpha=0.4)
        ax_cov.plot(t, cov_RG, color=_PURPLE, linewidth=1.0)
        ax_cov.set_ylabel(r"$\mathrm{Cov}(R,\,G)$")
        ax_cov.set_xlabel("Time  (a.u.)")
        ax_cov.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax_cov.set_title("Gene–RNA covariance", loc="left", pad=3)

        panels = [
            ("a", ax_G, fig_gene, "gene"),
            ("b", ax_R, fig_rna, "rna"),
            ("c", ax_cov, fig_cov, "cov"),
        ]

        for label, ax, fig, suffix in panels:
            ax.spines["left"].set_position(("outward", 3))
            ax.spines["bottom"].set_position(("outward", 3))

            ax.text(
                -0.16,
                1.08,
                label,
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                va="top",
            )

            if save_path is not None:
                dot = save_path.rfind(".")
                actual_path = (
                    f"{save_path}_{suffix}"
                    if dot == -1
                    else f"{save_path[:dot]}_{suffix}{save_path[dot:]}"
                )
                fig.savefig(actual_path)

    return fig_gene, fig_rna, fig_cov

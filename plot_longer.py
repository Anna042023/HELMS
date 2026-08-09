#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize MAE changes with horizons on PeMS-BAY.

Selected methods:
Traditional: ARIMA, SVR
Single-domain: STGCN, GCRN, GTS
Transformer/Attention: ST-MambaSync
Memory-augmented/Lifelong: D2MHyper
Ours: HELMS

Usage:
    python longer.py
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_matplotlib():
    """Use the same font style as tsne.py."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 450,
    })


def main():
    setup_matplotlib()

    horizons = ["15 min", "30 min", "60 min"]
    x = np.arange(len(horizons))

    mae_data = {
        # Traditional
        "ARIMA": [1.72, 2.48, 3.58],
        "SVR": [1.96, 2.63, 3.47],

        # Single-domain
        "STGCN": [1.44, 1.92, 2.64],
        "GCRN": [1.55, 2.00, 2.54],
        "GTS": [1.45, 1.83, 2.18],

        # Transformer/Attention
        "ST-MambaSync": [1.37, 1.70, 1.96],

        # Memory-augmented/Lifelong
        "D2MHyper": [1.42, 1.77, 2.03],

        # Ours
        "HELMS": [1.29, 1.59, 1.83],
    }

    # Re-designed vivid high-contrast paper-style palette.
    # The first seven baselines use more separable colors;
    # HELMS red is kept unchanged.
    colors = {
        "ARIMA": "#2F4858",          # deep slate blue
        "SVR": "#3366CC",            # royal blue
        "STGCN": "#00A676",          # emerald green
        "GCRN": "#7B2CBF",           # vivid purple
        "GTS": "#F9A620",            # warm amber
        "ST-MambaSync": "#FF6B35",   # coral orange
        "D2MHyper": "#4A4E69",       # elegant indigo gray
        "HELMS": "#D62728",          # unchanged highlight red
    }

    markers = {
        "ARIMA": "o",
        "SVR": "s",
        "STGCN": "^",
        "GCRN": "D",
        "GTS": "v",
        "ST-MambaSync": "P",
        "D2MHyper": "X",
        "HELMS": "*",
    }

    # Restore differentiated line styles.
    linestyles = {
        "ARIMA": "-",
        "SVR": (0, (5.5, 2.2)),
        "STGCN": (0, (4.5, 1.6, 1.3, 1.6)),
        "GCRN": "-",
        "GTS": (0, (3.0, 1.8)),
        "ST-MambaSync": (0, (6.0, 2.0)),
        "D2MHyper": (0, (5.0, 1.8, 1.2, 1.8)),
        "HELMS": "-",
    }

    fig, ax = plt.subplots(figsize=(8.9, 5.35))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Light red shadow band around HELMS.
    helms_y = np.asarray(mae_data["HELMS"], dtype=float)
    ax.fill_between(
        x,
        helms_y - 0.055,
        helms_y + 0.055,
        color=colors["HELMS"],
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )

    # Soft glow under the HELMS curve.
    ax.plot(
        x,
        helms_y,
        color=colors["HELMS"],
        linewidth=8.0,
        alpha=0.11,
        solid_capstyle="round",
        zorder=2,
    )

    for method, values in mae_data.items():
        is_ours = method == "HELMS"

        ax.plot(
            x,
            values,
            label=method,
            color=colors[method],
            marker=markers[method],
            linestyle=linestyles[method],
            linewidth=3.30 if is_ours else 2.50,
            markersize=18.0 if is_ours else 11.7,
            markeredgecolor="white",
            markeredgewidth=1.18,
            alpha=1.0 if is_ours else 0.94,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=8 if is_ours else 4,
        )

        if is_ours:
            for xi, yi in zip(x, values):
                ax.text(
                    xi,
                    yi - 0.060,
                    f"{yi:.2f}",
                    ha="center",
                    va="top",
                    fontsize=16.5,
                    fontweight="bold",
                    color=colors[method],
                    zorder=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=19)

    ax.set_xlabel("Prediction Horizon", fontsize=22, labelpad=8)
    ax.set_ylabel("MAE", fontsize=22, labelpad=8)

    ax.set_title(
        "MAE Variation Across Horizons on PeMS-BAY",
        fontsize=23,
        fontweight="bold",
        pad=15,
    )

    ax.set_ylim(1.08, 3.88)
    ax.set_yticks(np.arange(1.2, 3.9, 0.4))
    ax.set_xlim(-0.10, 2.10)

    ax.tick_params(axis="both", labelsize=18, length=4.2, width=1.0)

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=0.25,
    )
    ax.grid(
        True,
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.12,
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.1)
        ax.spines[spine].set_alpha(0.86)

    # Legend: 4 rows × 2 columns, no border, light transparent blue background.
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.035, 0.965),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.48,
        facecolor="#EAF6FF",
        edgecolor="none",
        fontsize=13.5,
        borderpad=0.58,
        labelspacing=0.45,
        handlelength=2.35,
        handletextpad=0.55,
        columnspacing=1.10,
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_facecolor("#EAF6FF")
    legend.get_frame().set_alpha(0.48)

    plt.tight_layout()

    save_dir = "./figures"
    os.makedirs(save_dir, exist_ok=True)

    png_path = os.path.join(save_dir, "pems_bay_mae_horizon.png")
    pdf_path = os.path.join(save_dir, "pems_bay_mae_horizon.pdf")

    plt.savefig(png_path, bbox_inches="tight", dpi=450)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved to: {png_path}")
    print(f"Saved to: {pdf_path}")


if __name__ == "__main__":
    main()
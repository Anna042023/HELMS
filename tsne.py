#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-style t-SNE visualization for HELMS memory prototypes on PeMS-BAY and PeMS04.

This version is designed to reproduce the cleaner PeMS-BAY style shown in the
user's target figure:
  - one dataset only by default: PeMS-BAY;
  - wide-but-not-squeezed layout, similar to the target figure;
  - no equal-aspect forcing, so the cloud will not become a narrow vertical plot;
  - soft but distinguishable category colors;
  - light translucent ellipses whose edges are included in axis limits;
  - title is only "PEMS-BAY";
  - axis labels default to "t-SNE Dimension 1/2" to match the target figure;
    use --dimension_only if you want "Dimension 1/2".

Expected input files:
    outputs/PEMS-BAY/H12/memory_bank_final.npz
    outputs/PEMS-BAY/H12/semantic_info.json

Typical usage:
    python tsne_fixed2.py --root_path ./outputs --pred_len 12

If you already have the old embedding that produced the target-style figure,
this script will try to reuse it first by default. This avoids generating a new
t-SNE layout with a different cluster shape. Disable with --rerun_tsne.

Outputs:
    outputs/PEMS-BAY/H12/memory_tsne_pems_bay_fixed2_bigger.pdf
    outputs/PEMS-BAY/H12/memory_tsne_pems_bay_fixed2_bigger.png
    outputs/PEMS-BAY/H12/memory_tsne_pems_bay_fixed2_bigger_embedding.npz
    outputs/PEMS-BAY/H12/memory_tsne_pems_bay_fixed2_bigger_embedding.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SCRIPT_VERSION = "tsne_fixed2_pems_bay_pems04_pems08_bigfont_bigmarker_20260611"

# Category order used in legend and plotting.
CATEGORY_ORDER = [
    "Early Peak",
    "Late Peak",
    "Rising Transition",
    "Falling Transition",
    "Event Spike",
    "Regular Pattern",
    "Off-Peak",
    "Congestion",
    "New Unseen Traffic Pattern",
    "Other",
]

# Soft but clearly distinguishable palette, close to the user's target figure.
# It is more elegant than hard red/blue/green/black, but still readable.
CATEGORY_COLORS = {
    "Early Peak": "#E76F51",                  # coral
    "Late Peak": "#F4A261",                   # warm orange
    "Rising Transition": "#2A9D8F",           # teal green
    "Falling Transition": "#118AB2",          # cyan-blue
    "Event Spike": "#9B5DE5",                 # violet
    "Regular Pattern": "#8D99AE",             # slate gray
    "Off-Peak": "#457B9D",                    # steel blue
    "Congestion": "#D62828",                  # red
    "New Unseen Traffic Pattern": "#7A7A7A",  # gray
    "Other": "#ADB5BD",
}


def canonical_dataset_name(name: str) -> str:
    key = str(name).strip().replace("_", "-").upper()
    aliases = {
        "PEMSBAY": "PEMS-BAY",
        "PEMS-BAY": "PEMS-BAY",
        "PEMS08": "PEMS08",
        "PEMS-08": "PEMS08",
        "PEMS04": "PEMS04",
        "PEMS-04": "PEMS04",
        "PEMS03": "PEMS03",
        "PEMS-03": "PEMS03",
        "PEMS07": "PEMS07",
        "PEMS-07": "PEMS07",
        "METRLA": "METR-LA",
        "METR-LA": "METR-LA",
    }
    if key in aliases:
        return aliases[key]
    return aliases.get(key.replace("-", ""), key)


def display_dataset_name(name: str) -> str:
    """Display name used in the figure title."""
    ds = canonical_dataset_name(name)
    display = {
        "PEMS-BAY": "PeMS-BAY",
        "PEMS04": "PeMS04",
        "PEMS03": "PeMS03",
        "PEMS07": "PeMS07",
        "PEMS08": "PeMS08",
        "METR-LA": "METR-LA",
    }
    return display.get(ds, ds)


def dataset_file_key(name: str) -> str:
    """Safe lowercase key used in output filenames."""
    return canonical_dataset_name(name).lower().replace("-", "_")


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def title_case_tag(tag: str) -> str:
    tag = str(tag or "Unknown").replace("_", "-").strip()
    tag = re.sub(r"\s+", " ", tag)
    if not tag:
        return "Unknown"
    small_words = {"a", "an", "the", "and", "or", "of", "to", "before", "after"}
    words: List[str] = []
    for i, w in enumerate(tag.split(" ")):
        if i > 0 and w.lower() in small_words:
            words.append(w.lower())
        else:
            words.append(w[:1].upper() + w[1:])
    return " ".join(words)


def semantic_category(tag: str, mode: str = "fine") -> str:
    """Map semantic_info.json tag to a visual category.

    Only the short tag is used. The long description is deliberately ignored,
    because HELMS descriptions contain generic template words such as
    "semantic regularization", which previously caused false matches.
    """
    t = normalize_text(tag)

    if mode == "raw":
        return title_case_tag(tag)

    if "early-window traffic peak" in t or "early window traffic peak" in t or "early peak" in t:
        fine = "Early Peak"
    elif "late-window traffic peak" in t or "late window traffic peak" in t or "late peak" in t:
        fine = "Late Peak"
    elif "rising transition before peak" in t or ("rising" in t and "transition" in t):
        fine = "Rising Transition"
    elif "falling transition after peak" in t or ("falling" in t and "transition" in t):
        fine = "Falling Transition"
    elif any(w in t for w in ["accident", "incident", "event", "spike", "burst", "anomaly", "abnormal", "shock", "sudden"]):
        fine = "Event Spike"
    elif any(w in t for w in ["new unseen", "unseen", "new traffic", "emerging", "novel"]):
        fine = "New Unseen Traffic Pattern"
    elif any(w in t for w in ["congestion", "congested", "jam", "heavy", "queue", "bottleneck", "slowdown"]):
        fine = "Congestion"
    elif "stable off-peak low-flow" in t or "off-peak" in t or "off peak" in t:
        fine = "Off-Peak"
    elif "regular recurrent traffic pattern" in t or "recurrent traffic pattern" in t or t == "regular pattern":
        fine = "Regular Pattern"
    elif any(w in t for w in ["regular", "normal", "recurrent"]):
        fine = "Regular Pattern"
    else:
        fine = title_case_tag(tag) if t and t != "unknown" else "Other"

    if mode == "coarse":
        if fine in {"Early Peak", "Late Peak"}:
            return "Peak Pattern"
        if fine in {"Rising Transition", "Falling Transition"}:
            return "Transition"
        if fine == "Regular Pattern":
            return "Off-Peak"
    return fine


def find_run_dir(root_path: str, dataset: str, pred_len: int) -> Path:
    root = Path(root_path)
    ds = canonical_dataset_name(dataset)
    candidates = [
        root / ds / f"H{pred_len}",
        root / ds.lower() / f"H{pred_len}",
        root / ds.replace("-", "") / f"H{pred_len}",
        root / ds.replace("-", "_") / f"H{pred_len}",
    ]
    for c in candidates:
        if (c / "memory_bank_final.npz").exists() and (c / "semantic_info.json").exists():
            return c

    target = f"H{pred_len}"
    ds_key = ds.upper().replace("-", "")
    matches = []
    for p in root.rglob("memory_bank_final.npz"):
        if p.parent.name != target:
            continue
        if not (p.parent / "semantic_info.json").exists():
            continue
        parent_key = p.parent.parent.name.upper().replace("_", "-").replace("-", "")
        if parent_key == ds_key:
            matches.append(p.parent)
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Cannot find {ds} H{pred_len} memory files under {root_path}.\n"
        f"Expected files like:\n"
        f"  {root_path}/{ds}/H{pred_len}/memory_bank_final.npz\n"
        f"  {root_path}/{ds}/H{pred_len}/semantic_info.json"
    )


def load_memory_and_semantics(run_dir: Path, category_mode: str = "fine") -> Dict[str, np.ndarray]:
    bank_path = run_dir / "memory_bank_final.npz"
    semantic_path = run_dir / "semantic_info.json"

    bank = np.load(bank_path, allow_pickle=True)
    if "active_prototypes" in bank:
        prototypes = np.asarray(bank["active_prototypes"], dtype=np.float32)
    elif "prototypes" in bank and "active_indices" in bank:
        active_indices_tmp = np.asarray(bank["active_indices"], dtype=np.int64)
        prototypes = np.asarray(bank["prototypes"], dtype=np.float32)[active_indices_tmp]
    else:
        raise KeyError(f"{bank_path} does not contain active_prototypes or prototypes+active_indices.")

    if prototypes.ndim != 2 or prototypes.shape[0] < 3:
        raise ValueError(f"Need at least 3 active prototypes for t-SNE, got shape {prototypes.shape}.")

    utilities = np.asarray(bank["active_utilities"], dtype=np.float32) if "active_utilities" in bank else np.ones(prototypes.shape[0], dtype=np.float32)
    core_mask = np.asarray(bank["active_core_mask"], dtype=bool) if "active_core_mask" in bank else np.zeros(prototypes.shape[0], dtype=bool)
    active_indices = np.asarray(bank["active_indices"], dtype=np.int64) if "active_indices" in bank else np.arange(prototypes.shape[0])

    with open(semantic_path, "r", encoding="utf-8") as f:
        semantic_rows = json.load(f)

    tags, descs, memory_ids = [], [], []
    for i in range(prototypes.shape[0]):
        row = semantic_rows[i] if i < len(semantic_rows) else {}
        tag = str(row.get("tag", "Unknown"))
        tags.append(tag)
        descs.append(str(row.get("description", "")))
        memory_ids.append(int(row.get("memory_id", active_indices[i] if i < len(active_indices) else i)))

    categories = [semantic_category(t, mode=category_mode) for t in tags]

    return {
        "prototypes": prototypes,
        "utilities": utilities,
        "core_mask": core_mask,
        "active_indices": active_indices,
        "memory_ids": np.asarray(memory_ids, dtype=np.int64),
        "tags": np.asarray(tags, dtype=object),
        "descriptions": np.asarray(descs, dtype=object),
        "categories": np.asarray(categories, dtype=object),
        "bank_path": str(bank_path),
        "semantic_path": str(semantic_path),
    }


def prepare_features(x: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = x / np.maximum(std, 1e-6)

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norm, 1e-6)

    n, d = x.shape
    out_dim = min(int(pca_dim), d, n - 1)
    if out_dim >= 2 and d > out_dim:
        x = PCA(n_components=out_dim, random_state=seed).fit_transform(x)
    return x.astype(np.float32)


def run_tsne(features: np.ndarray, perplexity: float, seed: int, n_iter: int, method: str) -> np.ndarray:
    n = features.shape[0]
    if perplexity <= 0:
        # Use 30 by default to stay close to the earlier target-style result.
        perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
    perplexity = float(min(perplexity, max(2.0, n - 1.0)))
    if perplexity >= n:
        perplexity = max(1.0, n - 1.0)

    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
        method=method,
        verbose=0,
    )
    sig = inspect.signature(TSNE.__init__).parameters
    if "n_jobs" in sig:
        kwargs["n_jobs"] = 1
    if "max_iter" in sig:
        kwargs["max_iter"] = int(n_iter)
    else:
        kwargs["n_iter"] = int(n_iter)

    y = TSNE(**kwargs).fit_transform(features)
    y = y - y.mean(axis=0, keepdims=True)

    # Use one scalar scale to preserve the t-SNE shape.
    scale = np.percentile(np.abs(y), 98)
    if scale > 1e-6:
        y = y / scale
    return y.astype(np.float32)


def find_existing_embedding(run_dir: Path, expected_n: int) -> Optional[Path]:
    """Find the old embedding that usually produced the target-style figure.

    We intentionally prioritize memory_tsne_embedding.npz because it was saved by
    the earlier multi-dataset script that generated the target-like PeMS-BAY
    layout. We do not prioritize the latest fixed embedding because that may be
    the vertically compact layout the user disliked.
    """
    candidates = [
        run_dir / "memory_tsne_embedding.npz",
        run_dir / "memory_tsne_pems_bay_embedding.npz",
        run_dir / "memory_tsne_fixed_embedding.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = np.load(path, allow_pickle=True)
            if "tsne" not in data:
                continue
            xy = np.asarray(data["tsne"])
            if xy.ndim == 2 and xy.shape[0] == expected_n and xy.shape[1] == 2:
                return path
        except Exception:
            continue
    return None


def load_or_compute_embedding(run_dir: Path, meta: Dict[str, np.ndarray], args: argparse.Namespace) -> Tuple[np.ndarray, str]:
    if not args.rerun_tsne:
        old_path = find_existing_embedding(run_dir, expected_n=meta["prototypes"].shape[0])
        if old_path is not None:
            data = np.load(old_path, allow_pickle=True)
            xy = np.asarray(data["tsne"], dtype=np.float32)
            return xy, f"loaded existing coordinates from {old_path}"

    features = prepare_features(meta["prototypes"], pca_dim=args.pca_dim, seed=args.seed)
    xy = run_tsne(features, perplexity=args.perplexity, seed=args.seed, n_iter=args.n_iter, method=args.tsne_method)
    return xy, "computed new t-SNE coordinates"


def ellipse_params(points: np.ndarray, n_std: float = 2.0) -> Tuple[np.ndarray, float, float, float]:
    if points.shape[0] < 6:
        raise ValueError("Need at least 6 points for ellipse.")
    cov = np.cov(points.T)
    if not np.all(np.isfinite(cov)):
        raise ValueError("Invalid covariance.")
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.maximum(vals, 1e-8)
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(vals)
    center = points.mean(axis=0)
    return center, float(width), float(height), angle


def rotated_ellipse_bounds(center: np.ndarray, width: float, height: float, angle_deg: float) -> Tuple[float, float, float, float]:
    a = width / 2.0
    b = height / 2.0
    theta = np.deg2rad(angle_deg)
    dx = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
    dy = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    return center[0] - dx, center[0] + dx, center[1] - dy, center[1] + dy


def add_confidence_ellipse(
    ax,
    points: np.ndarray,
    color: str,
    n_std: float,
    fill_alpha: float,
    edge_alpha: float,
    linewidth: float,
) -> Optional[Tuple[float, float, float, float]]:
    if points.shape[0] < 6:
        return None
    try:
        center, width, height, angle = ellipse_params(points, n_std=n_std)
    except ValueError:
        return None

    ell = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=to_rgba(color, fill_alpha),
        edgecolor=to_rgba(color, edge_alpha),
        linewidth=linewidth,
        linestyle="-",
        zorder=1,
        clip_on=False,
    )
    ax.add_patch(ell)
    return rotated_ellipse_bounds(center, width, height, angle)


def compute_plot_limits(xy: np.ndarray, ellipse_bounds: List[Tuple[float, float, float, float]], pad_ratio: float = 0.075) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xmin, xmax = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
    ymin, ymax = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))

    for bxmin, bxmax, bymin, bymax in ellipse_bounds:
        xmin = min(xmin, float(bxmin))
        xmax = max(xmax, float(bxmax))
        ymin = min(ymin, float(bymin))
        ymax = max(ymax, float(bymax))

    x_range = max(xmax - xmin, 1e-6)
    y_range = max(ymax - ymin, 1e-6)
    x_pad = max(0.05, pad_ratio * x_range)
    y_pad = max(0.05, pad_ratio * y_range)
    return (xmin - x_pad, xmax + x_pad), (ymin - y_pad, ymax + y_pad)


def category_list(categories: np.ndarray) -> List[str]:
    present = set(str(c) for c in categories.tolist())
    ordered = [c for c in CATEGORY_ORDER if c in present]
    extra = sorted([c for c in present if c not in CATEGORY_ORDER])
    return ordered + extra


def setup_matplotlib() -> None:
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


def prettify_axes(ax, dimension_only: bool) -> None:
    if dimension_only:
        ax.set_xlabel("Dimension 1", fontsize=25, labelpad=8)
        ax.set_ylabel("Dimension 2", fontsize=25, labelpad=8)
    else:
        ax.set_xlabel("Dimension 1", fontsize=22, labelpad=8)
        ax.set_ylabel("Dimension 2", fontsize=22, labelpad=8)
    ax.tick_params(axis="both", labelsize=18, length=4.0, width=0.9)
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.23)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_alpha(0.80)


def plot_pems_bay(
    xy: np.ndarray,
    meta: Dict[str, np.ndarray],
    out_dir: Path,
    dataset_title: str,
    dpi: int,
    output_name: str,
    draw_ellipses: bool,
    show_core_outline: bool,
    ellipse_n_std: float,
    ellipse_fill_alpha: float,
    ellipse_edge_alpha: float,
    ellipse_linewidth: float,
    dimension_only: bool,
) -> Tuple[Path, Path]:
    categories = meta["categories"]
    utilities = np.asarray(meta["utilities"], dtype=np.float32)
    core_mask = np.asarray(meta["core_mask"], dtype=bool)

    if np.max(utilities) - np.min(utilities) > 1e-8:
        u_norm = (utilities - np.min(utilities)) / (np.max(utilities) - np.min(utilities))
    else:
        u_norm = np.zeros_like(utilities)

    sizes = 36.0 + 66.0 * u_norm

    # Target-style ratio: wide enough to show the elongated t-SNE cloud, but not
    # as wide as the overly stretched version. No equal aspect is enforced.
    fig, ax = plt.subplots(figsize=(8.7, 5.75), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ellipse_bounds: List[Tuple[float, float, float, float]] = []
    if draw_ellipses:
        for cat in category_list(categories):
            pts = xy[categories == cat]
            color = CATEGORY_COLORS.get(cat, "#777777")
            bound = add_confidence_ellipse(
                ax,
                pts,
                color=color,
                n_std=ellipse_n_std,
                fill_alpha=ellipse_fill_alpha,
                edge_alpha=ellipse_edge_alpha,
                linewidth=ellipse_linewidth,
            )
            if bound is not None:
                ellipse_bounds.append(bound)

    for cat in category_list(categories):
        mask = categories == cat
        pts = xy[mask]
        color = CATEGORY_COLORS.get(cat, "#777777")
        if show_core_outline:
            edgecolors = np.where(core_mask[mask], "#111111", "white")
            linewidths = np.where(core_mask[mask], 0.95, 0.45)
        else:
            edgecolors = "white"
            linewidths = 0.45
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=sizes[mask],
            c=color,
            label=f"{cat} ({int(mask.sum())})",
            alpha=0.90,
            edgecolors=edgecolors,
            linewidths=linewidths,
            zorder=3,
        )

    xlim, ylim = compute_plot_limits(xy, ellipse_bounds, pad_ratio=0.050)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    prettify_axes(ax, dimension_only=dimension_only)
    ax.set_title(dataset_title, fontsize=24, pad=10, weight="bold")

    counts = Counter(str(c) for c in categories.tolist())
    handles, labels = [], []
    for cat in category_list(categories):
        color = CATEGORY_COLORS.get(cat, "#777777")
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=9.2,
            )
        )
        labels.append(cat)

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.034),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        fontsize=14.0,
        borderpad=0.58,
        labelspacing=0.48,
        handletextpad=0.50,
        columnspacing=1.15,
    )
    legend.get_frame().set_linewidth(0.45)

    # Target-style spacing: enough for legend, no right-side squeeze.
    fig.subplots_adjust(left=0.105, right=0.985, top=0.900, bottom=0.235)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{output_name}.pdf"
    png_path = out_dir / f"{output_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=dpi, pad_inches=0.04)
    plt.close(fig)
    return pdf_path, png_path



def plot_combined(
    results: List[Tuple[str, np.ndarray, Dict[str, np.ndarray]]],
    output_dir: Path,
    dpi: int,
    output_name: str,
    draw_ellipses: bool,
    show_core_outline: bool,
    ellipse_n_std: float,
    ellipse_fill_alpha: float,
    ellipse_edge_alpha: float,
    ellipse_linewidth: float,
    dimension_only: bool,
) -> Tuple[Path, Path]:
    """Draw a side-by-side combined t-SNE figure for multiple datasets.

    This function was missing in the previous generated script, which caused:
        NameError: name 'plot_combined' is not defined
    """
    n = len(results)
    if n < 2:
        raise ValueError("plot_combined requires at least two datasets.")

    fig, axes = plt.subplots(1, n, figsize=(8.2 * n, 5.75), dpi=dpi, squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("white")

    all_legend_cats: List[str] = []
    for dataset, _, meta in results:
        for cat in category_list(meta["categories"]):
            if cat not in all_legend_cats:
                all_legend_cats.append(cat)

    for ax, (dataset, xy, meta) in zip(axes, results):
        categories = meta["categories"]
        utilities = np.asarray(meta["utilities"], dtype=np.float32)
        core_mask = np.asarray(meta["core_mask"], dtype=bool)

        if np.max(utilities) - np.min(utilities) > 1e-8:
            u_norm = (utilities - np.min(utilities)) / (np.max(utilities) - np.min(utilities))
        else:
            u_norm = np.zeros_like(utilities)
        sizes = 32.0 + 58.0 * u_norm

        ax.set_facecolor("white")
        ellipse_bounds: List[Tuple[float, float, float, float]] = []

        if draw_ellipses:
            for cat in category_list(categories):
                pts = xy[categories == cat]
                color = CATEGORY_COLORS.get(cat, "#777777")
                bound = add_confidence_ellipse(
                    ax,
                    pts,
                    color=color,
                    n_std=ellipse_n_std,
                    fill_alpha=ellipse_fill_alpha,
                    edge_alpha=ellipse_edge_alpha,
                    linewidth=ellipse_linewidth,
                )
                if bound is not None:
                    ellipse_bounds.append(bound)

        for cat in category_list(categories):
            mask = categories == cat
            pts = xy[mask]
            color = CATEGORY_COLORS.get(cat, "#777777")
            if show_core_outline:
                edgecolors = np.where(core_mask[mask], "#111111", "white")
                linewidths = np.where(core_mask[mask], 0.95, 0.45)
            else:
                edgecolors = "white"
                linewidths = 0.45
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=sizes[mask],
                c=color,
                alpha=0.90,
                edgecolors=edgecolors,
                linewidths=linewidths,
                zorder=3,
            )

        xlim, ylim = compute_plot_limits(xy, ellipse_bounds, pad_ratio=0.050)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        prettify_axes(ax, dimension_only=dimension_only)
        ax.set_title(display_dataset_name(dataset), fontsize=24, pad=10, weight="bold")

    handles, labels = [], []
    for cat in all_legend_cats:
        color = CATEGORY_COLORS.get(cat, "#777777")
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=9.2,
            )
        )
        labels.append(cat)

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.030),
        ncol=min(4, max(1, len(labels))),
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        fontsize=14.0,
        borderpad=0.58,
        labelspacing=0.48,
        handletextpad=0.50,
        columnspacing=1.15,
    )
    legend.get_frame().set_linewidth(0.45)

    fig.subplots_adjust(left=0.065, right=0.990, top=0.900, bottom=0.235, wspace=0.22)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{output_name}.pdf"
    png_path = output_dir / f"{output_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=dpi, pad_inches=0.04)
    plt.close(fig)
    return pdf_path, png_path

def save_embedding(run_dir: Path, xy: np.ndarray, meta: Dict[str, np.ndarray], output_name: str) -> None:
    np.savez_compressed(
        run_dir / f"{output_name}_embedding.npz",
        tsne=xy,
        memory_ids=meta["memory_ids"],
        active_indices=meta["active_indices"],
        tags=meta["tags"],
        descriptions=meta["descriptions"],
        categories=meta["categories"],
        utilities=meta["utilities"],
        core_mask=meta["core_mask"],
        bank_path=np.asarray([meta["bank_path"]], dtype=object),
        semantic_path=np.asarray([meta["semantic_path"]], dtype=object),
    )
    with open(run_dir / f"{output_name}_embedding.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["memory_id", "active_index", "x", "y", "category", "tag", "utility", "is_core", "description"])
        for i in range(xy.shape[0]):
            writer.writerow([
                int(meta["memory_ids"][i]),
                int(meta["active_indices"][i]) if i < len(meta["active_indices"]) else i,
                float(xy[i, 0]),
                float(xy[i, 1]),
                str(meta["categories"][i]),
                str(meta["tags"][i]),
                float(meta["utilities"][i]),
                bool(meta["core_mask"][i]),
                str(meta["descriptions"][i]),
            ])


def print_summary(dataset: str, run_dir: Path, meta: Dict[str, np.ndarray], pdf_path: Path, png_path: Path, output_name: str, xy_source: str) -> None:
    category_counts = Counter(str(c) for c in meta["categories"].tolist())
    ordered_counts = {k: category_counts[k] for k in CATEGORY_ORDER if category_counts.get(k, 0) > 0}
    for k in sorted([k for k in category_counts if k not in ordered_counts]):
        ordered_counts[k] = category_counts[k]

    raw_counts = Counter(str(t) for t in meta["tags"].tolist())

    print(f"\n[{canonical_dataset_name(dataset)}]")
    print(f"  run_dir: {run_dir}")
    print(f"  active prototypes: {meta['prototypes'].shape[0]}")
    print(f"  coordinate source: {xy_source}")
    print(f"  raw tag counts: {dict(raw_counts.most_common())}")
    print(f"  semantic category counts: {ordered_counts}")
    print(f"  saved figure: {pdf_path}")
    print(f"  saved figure: {png_path}")
    print(f"  saved coordinates: {run_dir / (output_name + '_embedding.npz')}")
    print(f"  saved coordinates: {run_dir / (output_name + '_embedding.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw target-style t-SNE visualizations for HELMS memory prototypes on PeMS-BAY, PeMS04, and PeMS08.")
    parser.add_argument("--root_path", type=str, default="./outputs", help="Root output folder produced by training.")
    parser.add_argument("--datasets", nargs="+", default=["PEMS-BAY", "PEMS04", "PEMS08"], help="Datasets to draw. Default: PEMS-BAY PEMS04 PEMS08.")
    parser.add_argument("--pred_len", type=int, default=12, help="Prediction length folder, e.g., H12.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity. Default 30 keeps the target-like layout.")
    parser.add_argument("--pca_dim", type=int, default=30, help="PCA dimension before t-SNE.")
    parser.add_argument("--n_iter", type=int, default=1200, help="t-SNE optimization iterations.")
    parser.add_argument("--tsne_method", type=str, default="exact", choices=["exact", "barnes_hut"], help="t-SNE solver.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for PCA/t-SNE.")
    parser.add_argument("--dpi", type=int, default=450, help="PNG rendering dpi.")
    parser.add_argument("--category_mode", type=str, default="fine", choices=["fine", "coarse", "raw"], help="Semantic category mode.")
    parser.add_argument("--output_prefix", type=str, default="memory_tsne", help="Output filename prefix.")
    parser.add_argument("--rerun_tsne", action="store_true", help="Force recomputing t-SNE instead of reusing an existing target-style embedding.")
    parser.add_argument("--no_ellipses", action="store_true", help="Disable class ellipses.")
    parser.add_argument("--ellipse_n_std", type=float, default=2.00, help="Ellipse radius in standard deviations.")
    parser.add_argument("--ellipse_fill_alpha", type=float, default=0.055, help="Ellipse fill transparency.")
    parser.add_argument("--ellipse_edge_alpha", type=float, default=0.115, help="Ellipse edge transparency.")
    parser.add_argument("--ellipse_linewidth", type=float, default=1.15, help="Ellipse edge line width.")
    parser.add_argument("--hide_core_outline", action="store_true", help="Do not use black outline for core memories.")
    parser.add_argument("--dimension_only", action="store_true", help="Use Dimension 1/2 instead of t-SNE Dimension 1/2.")
    parser.add_argument("--no_combined", action="store_true", help="Do not draw the combined side-by-side figure.")
    parser.add_argument("--combined_dir", type=str, default=None, help="Output folder for combined figure. Default: <root_path>/figures.")
    args = parser.parse_args()

    setup_matplotlib()
    print(f"[Script] {SCRIPT_VERSION}")

    results: List[Tuple[str, np.ndarray, Dict[str, np.ndarray]]] = []
    for dataset in args.datasets:
        ds = canonical_dataset_name(dataset)
        run_dir = find_run_dir(args.root_path, ds, args.pred_len)
        meta = load_memory_and_semantics(run_dir, category_mode=args.category_mode)
        xy, xy_source = load_or_compute_embedding(run_dir, meta, args)

        output_name = f"{args.output_prefix}_{dataset_file_key(ds)}_fixed2_bigger"
        save_embedding(run_dir, xy, meta, output_name=output_name)

        pdf_path, png_path = plot_pems_bay(
            xy,
            meta,
            out_dir=run_dir,
            dataset_title=display_dataset_name(ds),
            dpi=args.dpi,
            output_name=output_name,
            draw_ellipses=not args.no_ellipses,
            show_core_outline=not args.hide_core_outline,
            ellipse_n_std=args.ellipse_n_std,
            ellipse_fill_alpha=args.ellipse_fill_alpha,
            ellipse_edge_alpha=args.ellipse_edge_alpha,
            ellipse_linewidth=args.ellipse_linewidth,
            dimension_only=args.dimension_only,
        )
        print_summary(ds, run_dir, meta, pdf_path, png_path, output_name=output_name, xy_source=xy_source)
        results.append((ds, xy, meta))

    if len(results) > 1 and not args.no_combined:
        combined_dir = Path(args.combined_dir) if args.combined_dir else Path(args.root_path) / "figures"
        combined_name = f"{args.output_prefix}_combined_" + "_".join(dataset_file_key(ds) for ds, _, _ in results) + "_fixed2_bigger"
        pdf_path, png_path = plot_combined(
            results,
            output_dir=combined_dir,
            dpi=args.dpi,
            output_name=combined_name,
            draw_ellipses=not args.no_ellipses,
            show_core_outline=not args.hide_core_outline,
            ellipse_n_std=args.ellipse_n_std,
            ellipse_fill_alpha=args.ellipse_fill_alpha,
            ellipse_edge_alpha=args.ellipse_edge_alpha,
            ellipse_linewidth=args.ellipse_linewidth,
            dimension_only=args.dimension_only,
        )
        print("\n[Combined]")
        print(f"  saved figure: {pdf_path}")
        print(f"  saved figure: {png_path}")


if __name__ == "__main__":
    main()

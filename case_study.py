#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw 24-hour case-study curves from saved HELMS prediction npz files.

Requested version:
  1) Use the same serif font style as tsne.py: Times New Roman/Times/DejaVu Serif.
  2) Select the best 20 nodes with the smallest error in one 24-hour window.
  3) One node per figure.
  4) No figure title.
  5) Figure size defaults to width=7.0 and height=4.4.
  6) Larger font size: default font_size=20.
  7) X-axis tick labels are 0:00, 4:00, ..., 24:00, not 0-24 and not "hour".
  8) Plot Ground Truth and Prediction with two nice colors.
  9) Use denser y-axis ticks to avoid overly sparse vertical labels.
 10) Put the legend inside the axes box, vertically stacked in the middle blank area.

Typical command:
python case_study_24h_inner_middle_legend.py \
  --save_dir ./outputs \
  --datasets PEMS08,PEMS-BAY \
  --pred_len 12 \
  --horizon 12 \
  --num_nodes 20 \
  --select_metric mae \
  --day_index 0 \
  --output_dir ./case_study_24h
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


FLOW_DATASETS = {"PEMS03", "PEMS04", "PEMS07", "PEMS08"}
SPEED_DATASETS = {"PEMS-BAY", "PEMS_BAY", "PEMSBAY", "METR-LA", "METR_LA", "METRLA"}


def parse_list(text: str) -> List[str]:
    if text is None or text.strip() == "":
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def canonical_dataset_name(name: str) -> str:
    n = name.strip()
    u = n.upper().replace("_", "-")
    if u in {"PEMSBAY", "PEMS-BAY"}:
        return "PEMS-BAY"
    if u in {"METRLA", "METR-LA"}:
        return "METR-LA"
    return u


def output_dataset_name(name: str) -> str:
    return canonical_dataset_name(name).replace("-", "_")


def dataset_aliases(name: str) -> List[str]:
    c = canonical_dataset_name(name)
    aliases = [name, c, c.lower(), c.replace("-", "_"), c.replace("-", "_").lower()]
    if c == "PEMS-BAY":
        aliases += ["PEMSBAY", "pemsbay", "PEMS_BAY", "pems_bay"]
    if c == "METR-LA":
        aliases += ["METRLA", "metrla", "METR_LA", "metr_la"]
    seen, out = set(), []
    for a in aliases:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out


def find_npz(save_dir: str, dataset: str, pred_len: int, npz_name: str) -> Path:
    root = Path(save_dir)
    candidates = []
    for ds in dataset_aliases(dataset):
        candidates.extend([
            root / ds / f"H{pred_len}" / npz_name,
            root / ds / f"h{pred_len}" / npz_name,
            root / ds / npz_name,
            root / ds / "test_predictions.npz",
            root / ds / "zero_shot_predictions.npz",
        ])
    for p in candidates:
        if p.exists():
            return p

    # Fallback: recursive search, useful when save_dir contains nested experiment folders.
    wanted_dataset_names = {a.lower() for a in dataset_aliases(dataset)}
    matches = []
    for p in root.rglob(npz_name):
        parent_text = str(p.parent).lower()
        if any(ds.lower() in parent_text for ds in wanted_dataset_names):
            matches.append(p)
    if matches:
        matches = sorted(matches, key=lambda x: len(str(x)))
        return matches[0]

    msg = [f"Cannot find {npz_name} for {dataset} under {save_dir}."]
    msg.append("Checked examples:")
    for p in candidates[:10]:
        msg.append(f"  {p}")
    raise FileNotFoundError("\n".join(msg))


def get_array(pack: Dict[str, np.ndarray], keys: List[str]) -> np.ndarray:
    for k in keys:
        if k in pack:
            return pack[k]
    raise KeyError(f"None of keys {keys} found. Available keys: {list(pack.keys())}")


def squeeze_to_bthn(arr: np.ndarray) -> np.ndarray:
    """Return prediction/target array as [B, T, N]."""
    arr = np.asarray(arr)
    if arr.ndim == 4:
        # Common format [B, T, N, C]
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[..., 0]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 2:
        # [B, N], treat as one horizon.
        arr = arr[:, None, :]
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}; expected [B,T,N,C], [B,T,N] or [B,N].")
    return arr.astype(np.float64)


def load_prediction_npz(npz_path: Path, prefer: str) -> Tuple[np.ndarray, np.ndarray]:
    pack = dict(np.load(npz_path, allow_pickle=True))
    true = get_array(pack, ["y_true", "true", "ground_truth", "target", "targets"])

    if prefer == "raw":
        pred_keys = ["y_pred_raw", "pred_raw", "raw_pred", "prediction_raw"]
    else:
        pred_keys = [
            "y_pred_calibrated", "pred_calibrated", "pred_final", "y_pred_final",
            "y_pred", "pred", "prediction", "y_hat",
        ]
    try:
        pred = get_array(pack, pred_keys)
    except KeyError:
        pred = get_array(pack, ["y_pred_raw", "pred_raw", "raw_pred", "prediction_raw"])

    true = squeeze_to_bthn(true)
    pred = squeeze_to_bthn(pred)
    if true.shape != pred.shape:
        min_b = min(true.shape[0], pred.shape[0])
        min_t = min(true.shape[1], pred.shape[1])
        min_n = min(true.shape[2], pred.shape[2])
        true = true[:min_b, :min_t, :min_n]
        pred = pred[:min_b, :min_t, :min_n]
    return true, pred


def one_day_series(true: np.ndarray, pred: np.ndarray, horizon: int, day_index: int,
                   start_index: int, interval_minutes: int) -> Tuple[np.ndarray, np.ndarray]:
    hidx = max(0, int(horizon) - 1)
    if hidx >= true.shape[1]:
        hidx = true.shape[1] - 1

    points_per_day = int(round(24 * 60 / interval_minutes))
    start = int(start_index) if start_index >= 0 else int(day_index) * points_per_day
    end = start + points_per_day
    if end > true.shape[0]:
        end = true.shape[0]
        start = max(0, end - points_per_day)
    if end <= start:
        raise ValueError(f"Invalid 24h window: start={start}, end={end}, total samples={true.shape[0]}")

    return true[start:end, hidx, :], pred[start:end, hidx, :]


def compute_node_metrics(true_24h: np.ndarray, pred_24h: np.ndarray) -> Dict[str, np.ndarray]:
    err = pred_24h - true_24h
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    denom = np.maximum(np.abs(true_24h), 1e-5)
    mape = np.mean(np.abs(err) / denom, axis=0) * 100.0
    return {"mae": mae, "rmse": rmse, "mape": mape}


def select_best_nodes(metrics: Dict[str, np.ndarray], metric: str, num_nodes: int) -> np.ndarray:
    metric = metric.lower()
    values = metrics[metric]
    order = np.argsort(values)
    return order[: min(int(num_nodes), len(order))]


def y_label_for_dataset(dataset: str) -> str:
    c = canonical_dataset_name(dataset)
    if c in SPEED_DATASETS:
        return "Speed"
    if c in FLOW_DATASETS:
        return "Flow"
    return "Traffic Value"


def setup_default_style(font_size: int) -> None:
    # Keep the same font family used in tsne.py.
    # Matplotlib will use Times New Roman when it is installed; otherwise it
    # falls back to Times and then DejaVu Serif, matching the t-SNE script.
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": max(font_size - 1, 1),
        "ytick.labelsize": max(font_size - 1, 1),
        "legend.fontsize": max(font_size - 1, 1),
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })




def apply_better_y_ticks(ax, true_node: np.ndarray, pred_node: np.ndarray) -> None:
    """Make y-axis ticks denser so the interval is smaller and the labels are more informative."""
    y_all = np.concatenate([np.asarray(true_node).ravel(), np.asarray(pred_node).ravel()])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return

    if np.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.05, 1.0)
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        span = y_max - y_min
        pad = max(span * 0.04, 1e-6)
        ax.set_ylim(y_min - pad, y_max + pad)

    # Ask Matplotlib for more major ticks than before.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5, steps=[1, 2, 2.5, 5, 10]))

def plot_one_node(true_node: np.ndarray, pred_node: np.ndarray, dataset: str, out_path: Path,
                  interval_minutes: int, figsize: Tuple[float, float], font_size: int,
                  truth_color: str, pred_color: str,
                  legend_x: float, legend_y: float) -> None:
    setup_default_style(font_size)

    n = len(true_node)
    x_hours = np.arange(n, dtype=np.float64) * interval_minutes / 60.0
    if n > 1:
        x_hours = x_hours * (24.0 / x_hours[-1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_hours, true_node, label="Ground Truth", color=truth_color, linewidth=2.2)
    ax.plot(x_hours, pred_node, label="Prediction", color=pred_color, linewidth=2.2)

    ax.set_xlabel("Time")
    ax.set_ylabel(y_label_for_dataset(dataset))
    apply_better_y_ticks(ax, true_node, pred_node)

    # 0:00 to 24:00. No "(hour)" anywhere.
    ticks = np.arange(0, 25, 4)
    ax.set_xlim(0, 24)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t)}:00" for t in ticks])

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    # Put the legend INSIDE the axes box and stack the two entries vertically.
    # For PEMS08, the central area of the U-shaped curve is mostly blank, so
    # this avoids covering the high curves on the left/right and the rising curve.
    if canonical_dataset_name(dataset) == "PEMS-BAY":
        ax.legend(
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.03, 0.05),
            ncol=1,
            handlelength=2.2,
            handletextpad=0.70,
            labelspacing=0.45,
            borderaxespad=0.0,
        )
    else:
        ax.legend(
            frameon=False,
            loc="center",
            bbox_to_anchor=(legend_x, legend_y),
            ncol=1,
            handlelength=2.2,
            handletextpad=0.70,
            labelspacing=0.45,
            borderaxespad=0.0,
        )

    # No title by design.
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metrics_csv(path: Path, selected_nodes: np.ndarray, metrics: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "node", "mae", "rmse", "mape"])
        writer.writeheader()
        for rank, node in enumerate(selected_nodes, start=1):
            writer.writerow({
                "rank": rank,
                "node": int(node),
                "mae": float(metrics["mae"][node]),
                "rmse": float(metrics["rmse"][node]),
                "mape": float(metrics["mape"][node]),
            })


def run_dataset(args, dataset: str) -> None:
    dataset = canonical_dataset_name(dataset)
    npz_path = find_npz(args.save_dir, dataset, args.pred_len, args.npz_name)
    print(f"[{dataset}] Load {npz_path}")

    true, pred = load_prediction_npz(npz_path, args.prefer)
    true_24h, pred_24h = one_day_series(
        true=true,
        pred=pred,
        horizon=args.horizon,
        day_index=args.day_index,
        start_index=args.start_index,
        interval_minutes=args.interval_minutes,
    )

    metrics = compute_node_metrics(true_24h, pred_24h)
    selected_nodes = select_best_nodes(metrics, args.select_metric, args.num_nodes)

    out_dir = Path(args.output_dir) / output_dataset_name(dataset) / f"H{args.horizon}_24h_best{len(selected_nodes)}_{args.select_metric.lower()}"
    save_metrics_csv(out_dir / "best_nodes_and_metrics.csv", selected_nodes, metrics)

    for rank, node in enumerate(selected_nodes, start=1):
        out_path = out_dir / f"rank{rank:02d}_node{int(node)}_{args.select_metric.lower()}_{metrics[args.select_metric.lower()][node]:.4f}.png"
        plot_one_node(
            true_node=true_24h[:, node],
            pred_node=pred_24h[:, node],
            dataset=dataset,
            out_path=out_path,
            interval_minutes=args.interval_minutes,
            figsize=(args.fig_width, args.fig_height),
            font_size=args.font_size,
            truth_color=args.truth_color,
            pred_color=args.pred_color,
            legend_x=args.legend_x,
            legend_y=args.legend_y,
        )

    print(f"[{dataset}] Saved {len(selected_nodes)} figures to {out_dir}")
    print(f"[{dataset}] Best-node metrics saved to {out_dir / 'best_nodes_and_metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw 24-hour case-study curves for best-error nodes.")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--datasets", type=str, default="PEMS08,PEMS-BAY")
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--select_metric", type=str, default="mae", choices=["mae", "rmse", "mape"])
    parser.add_argument("--day_index", type=int, default=0, help="Which 24h window to draw if start_index is not set.")
    parser.add_argument("--start_index", type=int, default=-1, help="Exact test-sample start index. If >=0, overrides day_index.")
    parser.add_argument("--interval_minutes", type=int, default=5)
    parser.add_argument("--prefer", type=str, default="calibrated", choices=["calibrated", "raw"])
    parser.add_argument("--npz_name", type=str, default="test_predictions.npz")
    parser.add_argument("--output_dir", type=str, default="./case_study_24h")
    parser.add_argument("--font_size", type=int, default=20)
    parser.add_argument("--fig_width", type=float, default=7.0)
    parser.add_argument("--fig_height", type=float, default=4.4)
    parser.add_argument("--truth_color", type=str, default="#018b8d")
    parser.add_argument("--pred_color", type=str, default="#e95d22")
    parser.add_argument("--legend_x", type=float, default=0.55,
                        help="Legend x-position inside axes, in axes fraction. Default places it in the middle blank area.")
    parser.add_argument("--legend_y", type=float, default=0.76,
                        help="Legend y-position inside axes, in axes fraction. Default places it in the middle blank area.")
    args = parser.parse_args()

    args.select_metric = args.select_metric.lower()
    for ds in parse_list(args.datasets):
        run_dataset(args, ds)


if __name__ == "__main__":
    main()

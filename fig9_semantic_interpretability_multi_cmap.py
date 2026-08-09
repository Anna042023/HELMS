#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw Fig.9-style HELMS semantic interpretability heatmaps only.

Compared with fig9_interpretability_v3.py, this version:
  1) Only generates the semantic activation heatmap.
  2) Produces multiple paper-friendly color versions in one run.
  3) Removes the figure title by default.
  4) Capitalizes the first letter of every word in labels.
  5) Increases all default font sizes by two points.

It does NOT require retraining. It uses the same 24-hour window rule as
case_study_24h.py:
  start = start_index if start_index >= 0 else day_index * (24*60/interval_minutes)
  end   = start + (24*60/interval_minutes)

Typical command:
python fig9_semantic_interpretability_multi_cmap.py \
  --save_dir ./outputs \
  --dataset PEMS08 \
  --pred_len 12 \
  --day_index 0 \
  --interval_minutes 5 \
  --time_bin 1 \
  --smooth_time 5 \
  --scale row_soft \
  --semantic_mode tag \
  --output_dir ./fig9_semantic_interpretability

To generate only selected color versions:
python fig9_semantic_interpretability_multi_cmap.py \
  --save_dir ./outputs \
  --dataset PEMS08 \
  --pred_len 12 \
  --cmaps paper_orange,paper_blue,paper_teal
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# -----------------------------
# Path helpers
# -----------------------------

def canonical_dataset_name(name: str) -> str:
    u = str(name).strip().upper().replace("_", "-")
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
    out, seen = [], set()
    for a in aliases:
        a = str(a)
        if a and a not in seen:
            out.append(a)
            seen.add(a)
    return out


def find_saved_file(save_dir: str, dataset: str, pred_len: int, filename: str, required: bool = True) -> Optional[Path]:
    root = Path(save_dir)
    candidates: List[Path] = [
        root / filename,
        root / f"H{pred_len}" / filename,
        root / f"h{pred_len}" / filename,
    ]
    for ds in dataset_aliases(dataset):
        ds_out = output_dataset_name(ds)
        candidates.extend([
            root / ds / f"H{pred_len}" / filename,
            root / ds / f"h{pred_len}" / filename,
            root / ds / filename,
            root / ds_out / f"H{pred_len}" / filename,
            root / ds_out / f"h{pred_len}" / filename,
            root / ds_out / filename,
        ])

    for p in candidates:
        if p.exists():
            return p

    wanted = {a.lower() for a in dataset_aliases(dataset)}
    matches: List[Path] = []
    if root.exists():
        for p in root.rglob(filename):
            text = str(p.parent).lower()
            if any(w in text for w in wanted):
                matches.append(p)
    if matches:
        htag = f"h{pred_len}"
        matches = sorted(matches, key=lambda x: (htag not in str(x).lower(), len(str(x))))
        return matches[0]

    if required:
        checked = "\n".join(f"  {p}" for p in candidates[:16])
        raise FileNotFoundError(
            f"Cannot find {filename} for {dataset} under {save_dir}.\n"
            f"Checked examples:\n{checked}"
        )
    return None


# -----------------------------
# Basic loading and parsing
# -----------------------------

def load_npz(path: Path) -> Dict[str, np.ndarray]:
    pack = np.load(path, allow_pickle=True)
    return {k: pack[k] for k in pack.files}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_existing_array(pack: Dict[str, np.ndarray], keys: Sequence[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in pack:
            return np.asarray(pack[k])
    return None


def clean_text(text: Any, max_len: int = 34) -> str:
    s = str(text if text is not None else "Unknown")
    s = s.replace("\n", " ").replace("\t", " ").strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        s = "Unknown"
    if len(s) > max_len:
        s = s[: max_len - 3].rstrip() + "..."
    return s


def title_case_label(text: Any) -> str:
    """Capitalize the first letter of every word in a figure label.

    Examples:
      late-window traffic peak -> Late-Window Traffic Peak
      stable off-peak low-flow -> Stable Off-Peak Low-Flow
      co-occurrence edge -> Co-Occurrence Edge

    Existing all-uppercase abbreviations such as LLM are preserved.
    """
    s = str(text if text is not None else "Unknown")
    s = s.replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)

    def cap_word(match: re.Match) -> str:
        word = match.group(0)
        if len(word) > 1 and word.isupper():
            return word
        return word[:1].upper() + word[1:].lower()

    return re.sub(r"[A-Za-z]+", cap_word, s)


def semantic_rows_to_maps(rows: Any) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Parse semantic_info.json into memory_id -> tag/description."""
    tag_map: Dict[int, str] = {}
    desc_map: Dict[int, str] = {}

    if isinstance(rows, dict) and "semantic_info" in rows:
        rows = rows["semantic_info"]

    if isinstance(rows, list):
        for i, item in enumerate(rows):
            if isinstance(item, dict):
                mid = int(item.get("memory_id", item.get("id", i)))
                tag = item.get("tag", item.get("label", item.get("category", "Unknown")))
                desc = item.get("description", item.get("desc", ""))
            else:
                mid = i
                tag, desc = item, ""
            tag_map[mid] = str(tag or "Unknown")
            desc_map[mid] = str(desc or "")
    elif isinstance(rows, dict):
        for k, v in rows.items():
            try:
                mid = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                tag_map[mid] = str(v.get("tag", v.get("label", v.get("category", "Unknown"))) or "Unknown")
                desc_map[mid] = str(v.get("description", v.get("desc", "")) or "")
            else:
                tag_map[mid] = str(v or "Unknown")
                desc_map[mid] = ""
    return tag_map, desc_map


def group_semantic_tag(tag: str) -> str:
    """Optional coarse paper-style grouping."""
    t = str(tag).lower()
    if any(w in t for w in ["incident", "accident", "anomaly", "abnormal", "spike", "event", "outlier", "sporadic"]):
        return "Incident / anomaly"
    if any(w in t for w in ["propagat", "spillback", "spread"]):
        return "Propagating congestion"
    if any(w in t for w in ["rising", "falling", "transition", "shift"]):
        return "Transition pattern"
    if any(w in t for w in ["peak", "rush", "congestion", "congested", "jam", "heavy"]):
        return "Recurrent congestion"
    if any(w in t for w in ["free", "smooth", "low", "light", "normal", "regular", "stable", "off-peak", "off peak"]):
        return "Regular / free-flow"
    if any(w in t for w in ["holiday", "weekend", "special"]):
        return "Holiday / special pattern"
    return clean_text(tag, max_len=30)


# -----------------------------
# Window and attention
# -----------------------------

def get_window_indices(day_index: int, start_index: int, interval_minutes: int, total_samples: int) -> Tuple[int, int]:
    points_per_day = int(round(24 * 60 / interval_minutes))
    start = int(start_index) if int(start_index) >= 0 else int(day_index) * points_per_day
    end = start + points_per_day
    if start < 0:
        start = 0
    if end > total_samples:
        raise ValueError(
            f"Requested 24h window [{start}:{end}] exceeds saved attention length {total_samples}. "
            f"Use a smaller --day_index/--start_index, or save more attention samples."
        )
    return start, end


def infer_memory_size(memory_pack: Optional[Dict[str, np.ndarray]], att_pack: Dict[str, np.ndarray]) -> int:
    if memory_pack is not None:
        for key in [
            "prototypes", "memory_prototypes", "memory_keys", "V", "values",
            "semantic_embeddings", "utilities", "active_mask"
        ]:
            arr = first_existing_array(memory_pack, [key])
            if arr is not None and arr.ndim >= 1 and arr.shape[0] > 0:
                return int(arr.shape[0])

    alpha = first_existing_array(att_pack, ["alpha", "attention", "memory_attention", "attn"])
    if alpha is not None:
        alpha = np.asarray(alpha)
        if alpha.ndim == 3 and alpha.shape[1] == 1:
            alpha = alpha[:, 0, :]
        if alpha.ndim == 2:
            return int(alpha.shape[1])

    ids = first_existing_array(att_pack, ["top_memory_ids", "topk_indices", "topk_ids", "memory_ids", "indices"])
    if ids is not None and ids.size > 0:
        return int(np.nanmax(ids)) + 1

    raise ValueError("Cannot infer memory size K from saved files.")


def squeeze_topk(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[1] == 1:
            arr = arr[:, 0, :]
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
    return arr


def build_attention_matrix(att_pack: Dict[str, np.ndarray], K: int, start: int, end: int) -> Tuple[np.ndarray, str]:
    """Return dense attention [T_window, K]. Supports full attention or saved top-k attention."""
    alpha = first_existing_array(att_pack, ["alpha", "attention", "memory_attention", "attn"])
    if alpha is not None:
        alpha = np.asarray(alpha)
        if alpha.ndim == 3 and alpha.shape[1] == 1:
            alpha = alpha[:, 0, :]
        if alpha.ndim == 2 and alpha.shape[0] >= end:
            K = max(K, alpha.shape[1])
            out = np.zeros((end - start, K), dtype=np.float32)
            out[:, : alpha.shape[1]] = alpha[start:end].astype(np.float32)
            return out, "full_attention"

    ids = first_existing_array(att_pack, ["top_memory_ids", "topk_indices", "topk_ids", "memory_ids", "indices"])
    weights = first_existing_array(att_pack, ["top_memory_weights", "topk_values", "topk_weights", "memory_weights", "values", "weights"])
    if ids is None or weights is None:
        raise KeyError(
            "Attention trace must contain either full attention or top-k ids/weights. "
            f"Available keys: {list(att_pack.keys())}"
        )

    ids = squeeze_topk(ids)
    weights = squeeze_topk(weights)
    if ids.ndim != 2 or weights.ndim != 2:
        raise ValueError(f"Top-k ids/weights should be [B, topk], got {ids.shape} and {weights.shape}.")
    if ids.shape[0] < end or weights.shape[0] < end:
        raise ValueError(f"Requested [{start}:{end}], but top-k attention has ids={ids.shape}, weights={weights.shape}.")

    ids_w = ids[start:end].astype(np.int64)
    w_w = weights[start:end].astype(np.float32)
    valid_all = ids_w >= 0
    if np.any(valid_all):
        K = max(K, int(ids_w[valid_all].max()) + 1)

    out = np.zeros((end - start, K), dtype=np.float32)
    for t in range(ids_w.shape[0]):
        valid = (ids_w[t] >= 0) & (ids_w[t] < K) & np.isfinite(w_w[t])
        if np.any(valid):
            np.add.at(out[t], ids_w[t, valid], w_w[t, valid])
    return out, "topk_reconstructed"


# -----------------------------
# Matrix transforms
# -----------------------------

def bin_time_matrix(mat_time_row: np.ndarray, bin_size: int, reducer: str = "mean") -> np.ndarray:
    """Aggregate [T, R] into [T_bin, R]."""
    mat = np.asarray(mat_time_row, dtype=np.float32)
    if int(bin_size) <= 1:
        return mat
    T, R = mat.shape
    trim = (T // int(bin_size)) * int(bin_size)
    if trim <= 0:
        return mat
    x = mat[:trim].reshape(trim // int(bin_size), int(bin_size), R)
    reducer = reducer.lower()
    if reducer == "mean":
        return x.mean(axis=1)
    if reducer == "sum":
        return x.sum(axis=1)
    if reducer == "median":
        return np.median(x, axis=1)
    return x.max(axis=1)


def smooth_time_axis(data: np.ndarray, window: int) -> np.ndarray:
    """Smooth each row along time with a moving average; keeps shape unchanged."""
    x = np.asarray(data, dtype=np.float32)
    w = int(window)
    if w <= 1 or x.size == 0:
        return x
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=np.float32) / float(w)
    pad = w // 2
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        row = np.pad(x[i], (pad, pad), mode="edge")
        out[i] = np.convolve(row, kernel, mode="valid")
    return out


def remove_flat_or_weak_rows(data: np.ndarray, labels: List[str], min_total: float, min_range: float) -> Tuple[np.ndarray, List[str], np.ndarray]:
    if data.size == 0:
        return data, labels, np.array([], dtype=np.int64)
    totals = np.nansum(data, axis=1)
    ranges = np.nanmax(data, axis=1) - np.nanmin(data, axis=1)
    keep = (totals > float(min_total)) & (ranges >= float(min_range))
    if not np.any(keep):
        keep = totals > 0
    idx = np.where(keep)[0]
    return data[idx], [labels[i] for i in idx], idx.astype(np.int64)


def select_and_sort_rows(data: np.ndarray, labels: List[str], max_rows: int, sort_by: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Select rows by total activation, then optionally sort by peak time or label name."""
    if data.size == 0:
        return data, labels, np.array([], dtype=np.int64)

    totals = np.nansum(data, axis=1)
    order = np.argsort(-totals)
    if int(max_rows) > 0:
        order = order[: int(max_rows)]

    sort_by = str(sort_by).lower()
    if sort_by == "peak":
        peaks = np.nanargmax(data[order], axis=1)
        order = order[np.argsort(peaks)]
    elif sort_by == "name":
        order = np.array(sorted(order, key=lambda i: labels[i]), dtype=np.int64)

    return data[order], [labels[i] for i in order], order.astype(np.int64)


def scale_heatmap(values: np.ndarray, mode: str = "row_soft", q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    mode = str(mode).lower()

    if mode in {"none", "raw"}:
        return arr
    if arr.size == 0:
        return arr
    if mode in {"global", "global_minmax"}:
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        return (arr - mn) / (mx - mn + 1e-12)
    if mode == "global_percentile":
        lo, hi = np.nanpercentile(arr, [q_low, q_high])
        arr = np.clip(arr, lo, hi)
        return (arr - lo) / (hi - lo + 1e-12)
    if mode in {"row", "row_minmax"}:
        mn = np.nanmin(arr, axis=1, keepdims=True)
        mx = np.nanmax(arr, axis=1, keepdims=True)
        return (arr - mn) / (mx - mn + 1e-12)
    if mode == "row_percentile":
        out = np.zeros_like(arr, dtype=np.float32)
        for i in range(arr.shape[0]):
            lo, hi = np.nanpercentile(arr[i], [q_low, q_high])
            row = np.clip(arr[i], lo, hi)
            out[i] = (row - lo) / (hi - lo + 1e-12)
        return out
    if mode == "row_soft":
        x = np.log1p(np.maximum(arr, 0))
        out = np.zeros_like(x, dtype=np.float32)
        for i in range(x.shape[0]):
            lo, hi = np.nanpercentile(x[i], [q_low, q_high])
            row = np.clip(x[i], lo, hi)
            out[i] = (row - lo) / (hi - lo + 1e-12)
        return np.power(out, 0.85)
    if mode == "global_soft":
        x = np.log1p(np.maximum(arr, 0))
        lo, hi = np.nanpercentile(x, [q_low, q_high])
        x = np.clip(x, lo, hi)
        out = (x - lo) / (hi - lo + 1e-12)
        return np.power(out, 0.85)
    if mode == "log_global":
        x = np.log1p(np.maximum(arr, 0))
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-12)

    raise ValueError(f"Unknown scale mode: {mode}")


# -----------------------------
# Semantic heatmap construction
# -----------------------------

def make_semantic_heatmap(att_24h: np.ndarray, tag_map: Dict[int, str], semantic_mode: str) -> Tuple[np.ndarray, List[str]]:
    K = att_24h.shape[1]
    label_to_ids: Dict[str, List[int]] = {}

    for mid in range(K):
        raw_tag = tag_map.get(mid, "Unknown")
        if semantic_mode == "group":
            label = group_semantic_tag(raw_tag)
        elif semantic_mode == "memory":
            label = f"M{mid}"
        else:
            label = clean_text(raw_tag, max_len=34)
        label_to_ids.setdefault(label, []).append(mid)

    labels = list(label_to_ids.keys())
    rows = []
    for label in labels:
        ids = [i for i in label_to_ids[label] if i < K]
        if ids:
            rows.append(att_24h[:, ids].sum(axis=1))
        else:
            rows.append(np.zeros(att_24h.shape[0], dtype=np.float32))

    data = np.stack(rows, axis=0).astype(np.float32) if rows else np.empty((0, att_24h.shape[0]), dtype=np.float32)
    return data, labels


# -----------------------------
# Paper-style colormaps
# -----------------------------

def make_paper_cmap(name: str):
    """Return a paper-friendly sequential colormap."""
    key = str(name).lower().strip()
    palettes = {
        # Original style
        "paper_orange": ["#FFF7EC", "#FDD49E", "#FDBB84", "#EF6548", "#7F0000"],

        # Elegant alternatives
        "paper_blue": ["#F7FBFF", "#DEEBF7", "#9ECAE1", "#4292C6", "#084594"],
        "paper_teal": ["#F7FCFD", "#CCECE6", "#66C2A4", "#238B8D", "#005824"],
        "paper_purple": ["#FCFBFD", "#E6E1EF", "#B2ABD2", "#8073AC", "#4D004B"],
        "paper_green": ["#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#00441B"],
        "paper_red": ["#FFF5F0", "#FEE0D2", "#FC9272", "#DE2D26", "#67000D"],
        "paper_grayblue": ["#F8FAFC", "#E2E8F0", "#94A3B8", "#475569", "#0F172A"],

        # More refined presentation palettes
        "soft_rose": ["#FFF7F6", "#FADADD", "#F2A7B5", "#D96C8B", "#7A174C"],
        "deep_indigo": ["#F8F7FF", "#DCD7FA", "#A79CE3", "#6750A4", "#2E1A47"],
        "ocean_mint": ["#F6FEFB", "#CDEFE7", "#7BD0C2", "#2E9E9B", "#075E67"],
        "warm_sunset": ["#FFF8ED", "#FDD8A5", "#F7A072", "#D95F59", "#7F1D1D"],
        "slate_gold": ["#FAFAF7", "#E7E0C5", "#C8B36B", "#8E7B36", "#3F3A2A"],
    }
    if key in palettes:
        return LinearSegmentedColormap.from_list(key, palettes[key], N=256)
    return plt.get_cmap(name)


def parse_cmaps(text: str) -> List[str]:
    all_cmaps = [
        "paper_orange",
        "paper_blue",
        "paper_teal",
        "paper_purple",
        "soft_rose",
        "deep_indigo",
        "ocean_mint",
        "warm_sunset",
        "paper_grayblue",
    ]
    if text is None or str(text).strip() == "":
        return all_cmaps
    text = str(text).strip()
    if text.lower() == "all":
        return all_cmaps

    out, seen = [], set()
    for item in text.split(","):
        name = item.strip()
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return out


# -----------------------------
# Plotting and saving
# -----------------------------

def setup_style(font_size: int) -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": max(font_size - 1, 1),
        "ytick.labelsize": max(font_size - 2, 1),
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_semantic_heatmap(
    raw_values: np.ndarray,
    row_labels: List[str],
    out_path: Path,
    cmap: str,
    scale: str,
    q_low: float,
    q_high: float,
    font_size: int,
    fig_width: float,
    fig_height: float,
    colorbar_label: str,
    interpolation: str = "nearest",
) -> None:
    if raw_values.size == 0:
        raise ValueError(f"No data to plot: {out_path}")

    values = scale_heatmap(raw_values, mode=scale, q_low=q_low, q_high=q_high)
    n_rows = values.shape[0]
    setup_style(font_size)

    row_labels = [title_case_label(x) for x in row_labels]
    colorbar_label = title_case_label(colorbar_label)

    height = max(float(fig_height), 0.32 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(float(fig_width), height))

    vmax = 1 if scale not in {"none", "raw"} else None
    im = ax.imshow(
        values,
        aspect="auto",
        interpolation=interpolation,
        cmap=make_paper_cmap(cmap),
        extent=[0, 24, n_rows, 0],
        vmin=0,
        vmax=vmax,
    )

    ax.set_xlabel("Time")
    ax.set_xlim(0, 24)
    xticks = np.arange(0, 25, 4)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(t)}:00" for t in xticks])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_labels)

    # No title by design.

    for y in range(1, n_rows):
        ax.axhline(y, color="white", linewidth=0.35, alpha=0.35)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_matrix_csv(path: Path, matrix: np.ndarray, labels: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_label"] + [f"t{i}" for i in range(matrix.shape[1])])
        for lab, row in zip(labels, matrix):
            writer.writerow([lab] + [float(x) for x in row])


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Draw semantic-only HELMS Fig.9 heatmaps with multiple color versions.")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--dataset", type=str, default="PEMS08")
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--day_index", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=-1)
    parser.add_argument("--interval_minutes", type=int, default=5)
    parser.add_argument("--time_bin", type=int, default=1, help="Number of 5-min samples per heatmap column. 3=15min, 6=30min, 1=raw 5min.")
    parser.add_argument("--bin_reducer", type=str, default="mean", choices=["max", "mean", "sum", "median"])
    parser.add_argument("--output_dir", type=str, default="./fig9_semantic_interpretability")
    parser.add_argument("--attention_name", type=str, default="attention_trace_test.npz")
    parser.add_argument("--fallback_attention_name", type=str, default="attention_trace.npz")

    parser.add_argument("--semantic_mode", type=str, default="tag", choices=["tag", "group", "memory"], help="tag=raw LLM tag; group=coarse paper-style category; memory=each memory row.")
    parser.add_argument("--max_semantic_rows", type=int, default=8)
    parser.add_argument("--sort_by", type=str, default="peak", choices=["peak", "strength", "name"])
    parser.add_argument("--min_total", type=float, default=1e-12)
    parser.add_argument("--min_range", type=float, default=1e-12)

    parser.add_argument("--scale", type=str, default="row_soft", choices=["none", "raw", "global_minmax", "global_percentile", "row_minmax", "row_percentile", "row_soft", "global_soft", "log_global"])
    parser.add_argument("--q_low", type=float, default=1.0)
    parser.add_argument("--q_high", type=float, default=99.0)

    # Increased by two points compared with v3 default 13.
    parser.add_argument("--font_size", type=int, default=15)
    parser.add_argument("--fig_width", type=float, default=8.8)
    parser.add_argument("--fig_height", type=float, default=4.4)

    parser.add_argument(
        "--cmaps",
        type=str,
        default="all",
        help=(
            "Comma-separated colormaps. Use all to generate several beautiful versions. "
            "Built-in options: paper_orange,paper_blue,paper_teal,paper_purple,paper_green,"
            "paper_red,paper_grayblue,soft_rose,deep_indigo,ocean_mint,warm_sunset,slate_gold."
        ),
    )
    parser.add_argument("--smooth_time", type=int, default=5, help="Moving-average smoothing window along time after binning. 1 disables smoothing.")
    parser.add_argument("--interpolation", type=str, default="nearest", choices=["nearest", "bilinear", "bicubic"], help="Image interpolation. Use nearest for strict values; bilinear for softer visual presentation.")
    parser.add_argument("--save_pdf", action="store_true")
    args = parser.parse_args()

    dataset = canonical_dataset_name(args.dataset)

    memory_path = find_saved_file(args.save_dir, dataset, args.pred_len, "memory_bank_final.npz", required=False)
    semantic_path = find_saved_file(args.save_dir, dataset, args.pred_len, "semantic_info.json", required=True)
    att_path = find_saved_file(args.save_dir, dataset, args.pred_len, args.attention_name, required=False)
    if att_path is None:
        att_path = find_saved_file(args.save_dir, dataset, args.pred_len, args.fallback_attention_name, required=True)

    memory_pack = load_npz(memory_path) if memory_path is not None else None
    att_pack = load_npz(att_path)
    semantic_rows = load_json(semantic_path)
    tag_map, desc_map = semantic_rows_to_maps(semantic_rows)

    alpha = first_existing_array(att_pack, ["alpha", "attention", "memory_attention", "attn"])
    if alpha is not None and np.asarray(alpha).ndim >= 2:
        total_samples = int(np.asarray(alpha).shape[0])
    else:
        ids = first_existing_array(att_pack, ["top_memory_ids", "topk_indices", "topk_ids", "memory_ids", "indices"])
        if ids is None:
            raise KeyError(f"Cannot infer attention length. Available keys: {list(att_pack.keys())}")
        total_samples = int(np.asarray(ids).shape[0])

    start, end = get_window_indices(args.day_index, args.start_index, args.interval_minutes, total_samples)
    K = infer_memory_size(memory_pack, att_pack)
    att_24h, att_mode = build_attention_matrix(att_pack, K, start, end)
    mean_mass = float(att_24h.sum(axis=1).mean()) if att_24h.size else 0.0

    sem_raw, sem_labels = make_semantic_heatmap(att_24h, tag_map, args.semantic_mode)
    sem_binned = bin_time_matrix(sem_raw.T, args.time_bin, reducer=args.bin_reducer).T
    sem_binned = smooth_time_axis(sem_binned, args.smooth_time)
    sem_binned, sem_labels, _ = remove_flat_or_weak_rows(sem_binned, sem_labels, args.min_total, args.min_range)
    sem_binned, sem_labels, sem_order = select_and_sort_rows(sem_binned, sem_labels, args.max_semantic_rows, args.sort_by)

    out_dir = Path(args.output_dir) / output_dataset_name(dataset) / f"H{args.pred_len}_day{args.day_index}_start{start}_end{end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmaps = parse_cmaps(args.cmaps)
    suffixes = ["png"] + (["pdf"] if args.save_pdf else [])

    print(f"[{dataset}] attention: {att_path}")
    print(f"[{dataset}] mode={att_mode}, 24h=[{start}:{end}], K={att_24h.shape[1]}, mean_saved_attention_mass={mean_mass:.4f}")
    print(f"[{dataset}] semantic rows={len(sem_labels)}, scale={args.scale}, semantic_mode={args.semantic_mode}, smooth_time={args.smooth_time}")
    print(f"[{dataset}] color versions: {', '.join(cmaps)}")

    for cmap in cmaps:
        for suffix in suffixes:
            out_name = f"fig9a_semantic_activation_heatmap_{cmap}.{suffix}"
            plot_semantic_heatmap(
                sem_binned,
                sem_labels,
                out_dir / out_name,
                cmap=cmap,
                scale=args.scale,
                q_low=args.q_low,
                q_high=args.q_high,
                font_size=args.font_size,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                colorbar_label="Relative Activation" if args.scale not in {"none", "raw"} else "Activation",
                interpolation=args.interpolation,
            )

    # Save a default copy using the first colormap for manuscript inclusion.
    if cmaps:
        default_png = out_dir / f"fig9a_semantic_activation_heatmap_{cmaps[0]}.png"
        if default_png.exists():
            import shutil
            shutil.copyfile(default_png, out_dir / "fig9a_semantic_activation_heatmap.png")

    np.savez_compressed(
        out_dir / "fig9_semantic_values.npz",
        start=np.array([start], dtype=np.int64),
        end=np.array([end], dtype=np.int64),
        day_index=np.array([args.day_index], dtype=np.int64),
        time_bin=np.array([args.time_bin], dtype=np.int64),
        attention_mode=np.asarray([att_mode], dtype=object),
        attention_24h=att_24h.astype(np.float32),
        semantic_heatmap=sem_binned.astype(np.float32),
        semantic_labels=np.asarray(sem_labels, dtype=object),
    )
    save_matrix_csv(out_dir / "semantic_activation_heatmap.csv", sem_binned, sem_labels)

    meta = {
        "dataset": dataset,
        "pred_len": int(args.pred_len),
        "day_index": int(args.day_index),
        "start": int(start),
        "end": int(end),
        "interval_minutes": int(args.interval_minutes),
        "time_bin": int(args.time_bin),
        "bin_reducer": args.bin_reducer,
        "smooth_time": int(args.smooth_time),
        "interpolation": args.interpolation,
        "attention_mode": att_mode,
        "mean_saved_attention_mass": mean_mass,
        "semantic_mode": args.semantic_mode,
        "scale": args.scale,
        "cmaps": cmaps,
        "font_size": int(args.font_size),
        "memory_path": str(memory_path) if memory_path is not None else None,
        "semantic_path": str(semantic_path),
        "attention_path": str(att_path),
        "note": (
            "This script only draws semantic activation heatmaps. "
            "When attention_mode is topk_reconstructed, heatmaps are based only on saved top-k attention. "
            "The colorbar represents row-normalized/relative activation when scale is row_soft or row_percentile."
        ),
    }
    with open(out_dir / "fig9_semantic_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[{dataset}] saved to: {out_dir}")
    print(f"  Default copy: {out_dir / 'fig9a_semantic_activation_heatmap.png'}")
    for cmap in cmaps:
        print(f"  {out_dir / f'fig9a_semantic_activation_heatmap_{cmap}.png'}")


if __name__ == "__main__":
    main()

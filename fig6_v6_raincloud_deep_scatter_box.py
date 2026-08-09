#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig6_v5.py

Alternative Fig. 6 structural interpretability analysis for HELMS: raincloud version.

This script keeps the original goal of Fig. 6: verifying whether HELMS learns
structurally coherent spatial dependencies through the hypergraph memory
database.  Different from the hop-distance decay curve, it draws two new
visualizations on PEMS03 and PEMS04:

1) Node influence network graph
   - node: traffic sensor
   - node color: detected community
   - edge width/alpha: gradient-based predictive sensitivity
   - only top-k strongest influence edges are drawn
   Expected pattern: denser intra-community influence edges and only a few
   meaningful cross-community edges.

2) Intra/Inter influence distribution
   - raincloud plot of predictive sensitivity values
   - compares intra-community and inter-community influence without using hop
     distance or MAE decay.
   Expected pattern: intra-community sensitivity has larger median/mean, while
   inter-community sensitivity is lower and more sparse.

The script uses intermediate files saved by train/train_helms.py/main.py:
    outputs/<DATASET>/H12/best_model.pth or best_helms.pt
    outputs/<DATASET>/H12/diagnostic_samples_test.npz
    outputs/<DATASET>/H12/graph_structure.npz
    outputs/<DATASET>/H12/memory_bank_final.npz
    outputs/<DATASET>/H12/hypergraph_final.npz

Typical usage:
    python fig6_v1.py --datasets PEMS03 PEMS04 --save_dir ./outputs --device cuda:0

Memory-saving usage:
    python fig6_v1.py --datasets PEMS03 PEMS04 --save_dir ./outputs \
        --device cuda:0 --batch_size 4 --max_samples 96 --max_target_nodes 64
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Matplotlib is imported after setting a non-interactive backend so the script
# works on servers without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None

try:
    from sklearn.cluster import SpectralClustering, KMeans
except Exception:
    SpectralClustering = None
    KMeans = None


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.helms import HELMS  # noqa: E402


DATASET_ALIASES = {
    "PeMS03": ["PEMS03", "PeMS03", "pems03"],
    "PeMS04": ["PEMS04", "PeMS04", "pems04"],
    "PeMS07": ["PEMS07", "PeMS07", "pems07"],
    "PeMS08": ["PEMS08", "PeMS08", "pems08"],
    "METR-LA": ["METR-LA", "metr-la", "METRLA", "metr_la"],
    "PEMS-BAY": ["PEMS-BAY", "PeMS-BAY", "pems-bay", "PEMSBAY", "pems_bay"],
}


def canonical_name(name: str) -> str:
    u = name.upper().replace("_", "-")
    if u in {"PeMS03", "PeMS04", "PeMS07", "PeMS08"}:
        return u
    if u in {"METR-LA", "METRLA"}:
        return "METR-LA"
    if u in {"PEMS-BAY", "PEMSBAY"}:
        return "PEMS-BAY"
    return name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. Please install pyyaml or use saved checkpoint/config_snapshot.json.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(dst: Dict, src: Dict) -> Dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def apply_dataset_preset(cfg: Dict, dataset_name: str) -> Dict:
    """Fallback used only when cfg is loaded from configs/config.yaml.

    The checkpoint usually already stores a dataset-specific cfg. This keeps the
    script usable even if only config.yaml is available.
    """
    import copy

    out = copy.deepcopy(cfg)
    presets = out.get("dataset_presets", {}) or {}
    key = canonical_name(dataset_name)
    if key in presets:
        force_disable_llm = (out.get("memory", {}) or {}).get("use_llm", None) is False
        deep_update(out, presets[key])
        if force_disable_llm:
            out.setdefault("memory", {})["use_llm"] = False
    return out


def find_run_dir(save_dir: Path, dataset: str, pred_len: int) -> Path:
    ds = canonical_name(dataset)
    names = [ds] + DATASET_ALIASES.get(ds, [])
    tried = []
    for name in dict.fromkeys(names):
        p = save_dir / name / f"H{pred_len}"
        tried.append(str(p))
        if p.is_dir():
            return p
    # Some users pass the run directory directly as --save_dir.
    if (save_dir / "diagnostic_samples_test.npz").exists():
        return save_dir
    raise FileNotFoundError(
        f"Cannot find run directory for {ds}, pred_len={pred_len}. Tried:\n  " + "\n  ".join(tried)
    )


def find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def load_run_cfg(run_dir: Path, checkpoint_payload: Optional[Dict], dataset: str, config_path: Optional[Path]) -> Dict:
    if isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("cfg"), dict):
        return checkpoint_payload["cfg"]

    snap = run_dir / "config_snapshot.json"
    if snap.exists():
        payload = load_json(snap)
        if isinstance(payload.get("cfg"), dict):
            return payload["cfg"]

    short = run_dir / "config.json"
    if short.exists():
        payload = load_json(short)
        if isinstance(payload.get("cfg"), dict):
            return payload["cfg"]
        if isinstance(payload, dict) and "model" in payload and "memory" in payload:
            return payload

    if config_path is not None and config_path.exists():
        return apply_dataset_preset(load_yaml(config_path), dataset)

    default_cfg = PROJECT_ROOT / "configs" / "config.yaml"
    if default_cfg.exists():
        return apply_dataset_preset(load_yaml(default_cfg), dataset)

    raise FileNotFoundError(
        f"No cfg found in checkpoint/config_snapshot.json/config.yaml for run directory {run_dir}"
    )


def build_memory_cfg(mem_cfg_raw: Dict) -> Dict:
    """Use the same HypergraphMemory constructor subset as HELMSTrainer."""
    return dict(
        init_memory_size=mem_cfg_raw.get("init_memory_size", 200),
        max_memory_size=mem_cfg_raw.get("max_memory_size", 400),
        semantic_dim=mem_cfg_raw.get("semantic_dim", 384),
        hyper_neighbors=mem_cfg_raw.get("hyper_neighbors", 5),
        transition_threshold=mem_cfg_raw.get("transition_threshold", 0.3),
        cooc_topk=mem_cfg_raw.get("cooc_topk", 5),
        cooc_threshold=mem_cfg_raw.get("cooc_threshold", 5),
        sr_pairs=mem_cfg_raw.get("sr_pairs", 1000),
        temperature=mem_cfg_raw.get("temperature", 1.0),
        retrieve_topk=mem_cfg_raw.get("retrieve_topk", 0),
        normalize_retrieval=mem_cfg_raw.get("normalize_retrieval", False),
        utility_retrieval_weight=mem_cfg_raw.get("utility_retrieval_weight", 0.15),
        residual_confidence_floor=mem_cfg_raw.get("residual_confidence_floor", 0.45),
        residual_confidence_power=mem_cfg_raw.get("residual_confidence_power", 1.0),
        residual_shrink_count=mem_cfg_raw.get("residual_shrink_count", 8.0),
        beta_init=mem_cfg_raw.get("beta_init", 0.05),
        memory_gate_init=mem_cfg_raw.get("memory_gate_init", -2.2),
        memory_residual_gate_init=mem_cfg_raw.get("memory_residual_gate_init", -0.8),
    )


def safe_load_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], device: torch.device) -> Tuple[int, int]:
    """Load only tensors whose names and shapes match.

    Dynamic buffers such as memory.incidence and memory.forecast_residuals can
    have run-specific shapes. They are restored separately from saved npz files.
    """
    current = model.state_dict()
    loadable = {}
    skipped = 0
    for k, v in state_dict.items():
        if k in current and tuple(current[k].shape) == tuple(v.shape):
            loadable[k] = v.to(device)
        else:
            skipped += 1
    model.load_state_dict(loadable, strict=False)
    return len(loadable), skipped


def copy_np_to_tensor_attr(module: torch.nn.Module, name: str, arr: np.ndarray, device: torch.device, dtype=None) -> bool:
    if not hasattr(module, name):
        return False
    tensor = torch.as_tensor(arr, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    old = getattr(module, name)
    if isinstance(old, torch.nn.Parameter):
        if tuple(old.shape) != tuple(tensor.shape):
            return False
        old.data.copy_(tensor.to(dtype=old.dtype))
        return True
    if torch.is_tensor(old) and tuple(old.shape) == tuple(tensor.shape):
        old.copy_(tensor.to(dtype=old.dtype))
        return True
    # For dynamic buffers with run-specific shape, replacing the buffer is OK.
    try:
        setattr(module, name, tensor)
        return True
    except Exception:
        return False


def restore_memory_artifacts(model: HELMS, run_dir: Path, device: torch.device) -> None:
    mem = model.memory
    bank_path = run_dir / "memory_bank_final.npz"
    if bank_path.exists():
        bank = np.load(bank_path, allow_pickle=True)
        for key in [
            "prototypes", "utility", "last_access", "access_count", "core_mask",
            "active_mask", "semantic_embeddings", "residual_counts"
        ]:
            if key in bank.files:
                old = getattr(mem, key, None)
                dtype = old.dtype if torch.is_tensor(old) else None
                copy_np_to_tensor_attr(mem, key, bank[key], device, dtype=dtype)
        if "forecast_residuals" in bank.files and bank["forecast_residuals"].size > 0:
            copy_np_to_tensor_attr(mem, "forecast_residuals", bank["forecast_residuals"], device, dtype=mem.prototypes.dtype)
        if "incidence" in bank.files and bank["incidence"].size > 0:
            copy_np_to_tensor_attr(mem, "incidence", bank["incidence"], device, dtype=mem.prototypes.dtype)

    hyper_path = run_dir / "hypergraph_final.npz"
    if hyper_path.exists():
        hg = np.load(hyper_path, allow_pickle=True)
        if "incidence" in hg.files and hg["incidence"].size > 0:
            copy_np_to_tensor_attr(mem, "incidence", hg["incidence"], device, dtype=mem.prototypes.dtype)
        if "active_mask" in hg.files:
            copy_np_to_tensor_attr(mem, "active_mask", hg["active_mask"], device, dtype=mem.active_mask.dtype)


def build_model_from_artifacts(
    run_dir: Path,
    dataset: str,
    x_shape: Tuple[int, ...],
    y_shape: Tuple[int, ...],
    device: torch.device,
    config_path: Optional[Path] = None,
) -> Tuple[HELMS, Dict]:
    ckpt_path = find_first_existing([run_dir / "best_model.pth", run_dir / "best_helms.pt"])
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Missing best_model.pth / best_helms.pt under {run_dir}. Please run main.py once so intermediate results are saved."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    cfg = load_run_cfg(run_dir, ckpt if isinstance(ckpt, dict) else None, dataset, config_path)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    mem_cfg = build_memory_cfg(cfg.get("memory", {}) or {})

    num_nodes = int(x_shape[2])
    input_dim = int(x_shape[3])
    output_dim = int(y_shape[-1]) if len(y_shape) >= 4 else int(data_cfg.get("output_dim", 1))
    seq_len = int(x_shape[1])
    pred_len = int(y_shape[1]) if len(y_shape) >= 4 else int(data_cfg.get("pred_len", 12))

    model = HELMS(
        num_nodes=num_nodes,
        input_dim=input_dim,
        output_dim=output_dim,
        pred_len=pred_len,
        hidden_dim=model_cfg.get("hidden_dim", 64),
        node_embed_dim=model_cfg.get("node_embed_dim", 16),
        time_embed_dim=model_cfg.get("time_embed_dim", 16),
        num_st_layers=model_cfg.get("num_st_layers", 3),
        gat_heads=model_cfg.get("gat_heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        dynamic_topk=model_cfg.get("dynamic_topk", 12),
        dynamic_alpha=model_cfg.get("dynamic_alpha", 0.5),
        static_alpha=model_cfg.get("static_alpha", 0.3),
        use_static_graph=model_cfg.get("use_static_graph", False),
        residual_forecast=model_cfg.get("residual_forecast", True),
        residual_clip=model_cfg.get("residual_clip", 3.0),
        trend_prior=model_cfg.get("trend_prior", True),
        trend_clip=model_cfg.get("trend_clip", 1.0),
        seq_len=seq_len,
        history_prior=model_cfg.get("history_prior", True),
        stid_prior=model_cfg.get("stid_prior", True),
        stid_hidden_dim=model_cfg.get("stid_hidden_dim", max(96, model_cfg.get("hidden_dim", 64) * 2)),
        stid_node_dim=model_cfg.get("stid_node_dim", 32),
        external_prior_fusion=model_cfg.get("external_prior_fusion", True),
        adaptive_prior_fusion=model_cfg.get("adaptive_prior_fusion", True),
        prior_bases=model_cfg.get("prior_bases", None),
        memory_cfg=mem_cfg,
    ).to(device)

    loaded, skipped = safe_load_state_dict(model, state, device)
    print(f"[{canonical_name(dataset)}] Loaded {loaded} checkpoint tensors, skipped {skipped} dynamic/mismatched tensors.")
    restore_memory_artifacts(model, run_dir, device)
    model.eval()
    return model, cfg


def load_diagnostic_artifacts(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diag_path = run_dir / "diagnostic_samples_test.npz"
    if not diag_path.exists():
        raise FileNotFoundError(
            f"Missing {diag_path}. The provided trainer saves it through save_diagnostic_samples(). "
            f"Please rerun training/evaluation first."
        )
    diag = np.load(diag_path, allow_pickle=True)
    x = np.asarray(diag["x"], dtype=np.float32)
    y = np.asarray(diag["y"], dtype=np.float32)
    tf = np.asarray(diag["time_features"], dtype=np.float32)
    starts = np.asarray(diag["starts"], dtype=np.int64) if "starts" in diag.files else np.arange(x.shape[0])
    if "adjacency" in diag.files:
        adj = np.asarray(diag["adjacency"], dtype=np.float32)
    else:
        graph_path = run_dir / "graph_structure.npz"
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing adjacency in diagnostic file and missing {graph_path}")
        graph = np.load(graph_path, allow_pickle=True)
        adj = np.asarray(graph["adjacency"], dtype=np.float32)
    return x, y, tf, starts, adj


def adjacency_to_binary(adj: np.ndarray, threshold: float = 1e-8, symmetrize: bool = True) -> np.ndarray:
    A = np.asarray(adj, dtype=np.float32)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape={A.shape}")
    A = (A > float(threshold))
    np.fill_diagonal(A, False)
    if symmetrize:
        A = np.logical_or(A, A.T)
    return A.astype(bool)


def hop_distance_matrix(A_bool: np.ndarray, max_hop: int) -> np.ndarray:
    """Compute shortest hop distances up to max_hop using boolean expansion."""
    N = A_bool.shape[0]
    dist = np.full((N, N), fill_value=np.inf, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    visited = np.eye(N, dtype=bool)
    frontier = A_bool.copy()
    A_u8 = A_bool.astype(np.uint8)
    for hop in range(1, int(max_hop) + 1):
        new = frontier & (~visited)
        dist[new] = float(hop)
        visited |= frontier
        frontier = ((frontier.astype(np.uint8) @ A_u8) > 0) & (~visited)
    return dist


def fallback_kmeans(X: np.ndarray, k: int, seed: int = 2026, n_iter: int = 50) -> np.ndarray:
    """Small deterministic KMeans fallback when sklearn is unavailable."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n <= k:
        return np.arange(n) % max(1, k)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(axis=0)
    return labels


def detect_communities(adj: np.ndarray, n_communities: int, seed: int, threshold: float) -> np.ndarray:
    """Community labels for intra-/inter-community aggregation."""
    A = adjacency_to_binary(adj, threshold=threshold, symmetrize=True).astype(np.float32)
    N = A.shape[0]
    k = int(n_communities)
    if k <= 0:
        k = max(4, min(16, int(round(math.sqrt(N / 2.0)))))
    k = max(2, min(k, N))

    # Prefer sklearn SpectralClustering on precomputed affinity.
    affinity = A.copy()
    np.fill_diagonal(affinity, 1.0)
    if SpectralClustering is not None:
        try:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=seed,
                n_init=10,
            )
            return sc.fit_predict(affinity).astype(np.int64)
        except Exception as e:
            print(f"[WARN] sklearn SpectralClustering failed: {e}. Use eigenvector fallback.")

    # Fallback: normalized Laplacian eigenvectors + kmeans.
    deg = affinity.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 1e-6, None))
    L = np.eye(N, dtype=np.float32) - deg_inv_sqrt[:, None] * affinity * deg_inv_sqrt[None, :]
    vals, vecs = np.linalg.eigh(L)
    X = vecs[:, np.argsort(vals)[:k]]
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
    if KMeans is not None:
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X).astype(np.int64)
    return fallback_kmeans(X.astype(np.float32), k, seed=seed)


def select_target_nodes(adj: np.ndarray, max_target_nodes: int, seed: int) -> np.ndarray:
    N = adj.shape[0]
    if max_target_nodes <= 0 or max_target_nodes >= N:
        return np.arange(N, dtype=np.int64)
    # Select a degree-stratified subset so both hubs and ordinary nodes are covered.
    A = adjacency_to_binary(adj, threshold=1e-8, symmetrize=True)
    degree = A.sum(axis=1)
    order = np.argsort(degree)
    # Even quantile sampling is deterministic and stable across runs.
    pos = np.linspace(0, N - 1, max_target_nodes).round().astype(np.int64)
    nodes = order[pos]
    return np.unique(nodes).astype(np.int64)


def evenly_select_samples(num_samples: int, max_samples: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= num_samples:
        return np.arange(num_samples, dtype=np.int64)
    return np.linspace(0, num_samples - 1, max_samples).round().astype(np.int64)


def compute_gradient_sensitivity(
    model: HELMS,
    x_np: np.ndarray,
    tf_np: np.ndarray,
    adj_np: np.ndarray,
    device: torch.device,
    horizon: int,
    batch_size: int,
    max_samples: int,
    max_target_nodes: int,
    seed: int,
    use_memory: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sensitivity matrix for selected target nodes.

    sensitivity[r, j] is the mean absolute gradient of target node target_nodes[r]
    with respect to source node j over selected test samples and historical time.
    """
    B, L, N, C = x_np.shape
    target_nodes = select_target_nodes(adj_np, max_target_nodes=max_target_nodes, seed=seed)
    sample_idx = evenly_select_samples(B, max_samples=max_samples)
    h_idx = max(0, min(int(horizon) - 1, int(model.pred_len) - 1))

    sens_sum = np.zeros((len(target_nodes), N), dtype=np.float64)
    batch_count = 0
    static_adj = torch.as_tensor(adj_np, dtype=torch.float32, device=device)

    print(
        f"Computing gradients: samples={len(sample_idx)}, batch_size={batch_size}, "
        f"target_nodes={len(target_nodes)}/{N}, horizon={h_idx + 1}, use_memory={use_memory}"
    )

    for b0 in range(0, len(sample_idx), batch_size):
        ids = sample_idx[b0:b0 + batch_size]
        xb = torch.as_tensor(x_np[ids], dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
        tfb = torch.as_tensor(tf_np[ids], dtype=torch.float32, device=device)

        pred, _ = model(
            xb,
            tfb,
            static_adj=static_adj,
            use_memory=use_memory,
            return_aux=True,
            external_priors=None,
        )
        # [B, N]
        pred_h = pred[:, h_idx, :, 0]

        for r, node in enumerate(target_nodes.tolist()):
            scalar = pred_h[:, node].mean()
            grad = torch.autograd.grad(
                scalar,
                xb,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]
            # Aggregate over batch, historical time and feature channels -> [N source nodes]
            src_sens = grad.detach().abs().mean(dim=(0, 1, 3)).float().cpu().numpy()
            sens_sum[r] += src_sens.astype(np.float64)

        batch_count += 1
        del xb, tfb, pred, pred_h
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  finished batch {batch_count}/{math.ceil(len(sample_idx) / batch_size)}")

    sens = sens_sum / max(1, batch_count)
    return sens.astype(np.float32), target_nodes





# -----------------------------------------------------------------------------
# Revised alternative Fig. 6 visualizations: community-clustered network graph
# + active intra/inter influence distribution.
# -----------------------------------------------------------------------------

from matplotlib.lines import Line2D


FIG_VERSION = "fig6_v6_raincloud_deep_scatter_box_fontplus5_equal_axis_labels_20260611"

COMMUNITY_PALETTE = [
    "#E76F51", "#F4A261", "#2A9D8F", "#118AB2", "#9B5DE5",
    "#8D99AE", "#457B9D", "#D62828", "#7A7A7A", "#ADB5BD",
    "#06D6A0", "#EF476F", "#073B4C", "#FFB703", "#6D597A", "#43AA8B",
]
# User-specified soft colors for the half-violin shadow.
INTRA_CLOUD_COLOR = "#B2A3DD"  # soft purple shadow
INTER_CLOUD_COLOR = "#ED949A"  # soft pink shadow

# Slightly deeper colors for scatter points and boxplot.
INTRA_COLOR = "#8F7CCB"  # deeper purple
INTER_COLOR = "#D96F78"  # deeper pink


def setup_matplotlib_tsne_style(base_font: int = 23) -> None:
    """Use the same font configuration as the uploaded tsne.py script."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 450,
        "font.size": base_font,
        "axes.labelsize": base_font + 2,
        "axes.titlesize": base_font + 6,
        "xtick.labelsize": base_font + 1,
        "ytick.labelsize": base_font + 1,
        "legend.fontsize": base_font,
        "axes.linewidth": 1.05,
    })


def finite_positive(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    v = v[v > 0]
    return v


def robust_scale(values: np.ndarray, percentile: float = 99.0) -> float:
    v = finite_positive(values)
    if v.size == 0:
        return 1.0
    s = float(np.percentile(v, percentile))
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.max(v))
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0
    return s


def row_relative_sensitivity(
    sensitivity: np.ndarray,
    target_nodes: np.ndarray,
    row_norm_percentile: float = 99.0,
    exclude_self: bool = True,
) -> np.ndarray:
    """Normalize each target row independently to avoid hub/star artifacts.

    sensitivity[r, j] is source-node j's influence on target node target_nodes[r].
    A row-wise robust normalization converts raw gradients into relative
    influence scores in [0, 1], which is more suitable for visualization.
    """
    rel = np.asarray(sensitivity, dtype=np.float64).copy()
    R, N = rel.shape
    for r in range(R):
        target = int(target_nodes[r])
        row = rel[r]
        row[~np.isfinite(row)] = 0.0
        row[row < 0] = 0.0
        if exclude_self and 0 <= target < N:
            row[target] = 0.0
        scale = robust_scale(row, percentile=row_norm_percentile)
        rel[r] = np.clip(row / scale, 0.0, 1.0)
    return rel.astype(np.float32)


def make_pair_records(
    rel_sens: np.ndarray,
    raw_sens: np.ndarray,
    target_nodes: np.ndarray,
    communities: np.ndarray,
    min_relative_weight: float = 0.0,
) -> List[Dict]:
    records: List[Dict] = []
    R, N = rel_sens.shape
    for r in range(R):
        target = int(target_nodes[r])
        tgt_comm = int(communities[target])
        for source in range(N):
            if source == target:
                continue
            w = float(rel_sens[r, source])
            if not np.isfinite(w) or w <= min_relative_weight:
                continue
            src_comm = int(communities[source])
            records.append({
                "source": int(source),
                "target": int(target),
                "weight": w,
                "raw_weight": float(raw_sens[r, source]),
                "type": "Intra-Community" if src_comm == tgt_comm else "Inter-Community",
                "source_community": src_comm,
                "target_community": tgt_comm,
            })
    return records


def select_network_edges(
    rel_sens: np.ndarray,
    raw_sens: np.ndarray,
    target_nodes: np.ndarray,
    communities: np.ndarray,
    intra_edges_per_community: int = 18,
    inter_edges_per_pair: int = 1,
    max_intra_edges: int = 150,
    max_inter_edges: int = 8,
    max_plot_nodes: int = 140,
    min_relative_weight: float = 0.03,
    min_inter_relative_weight: float = 0.30,
) -> List[Dict]:
    """Select a readable structural subgraph for the node influence network.

    The plot should communicate structural coherence rather than display every
    large gradient. Therefore, intra-community edges are selected more densely
    inside each community, while cross-community edges are treated as sparse
    bridges and selected much more strictly.
    """
    # Candidate edges for intra-community structure.
    records = make_pair_records(
        rel_sens=rel_sens,
        raw_sens=raw_sens,
        target_nodes=target_nodes,
        communities=communities,
        min_relative_weight=min_relative_weight,
    )
    if not records:
        records = make_pair_records(rel_sens, raw_sens, target_nodes, communities, min_relative_weight=0.0)

    intra_by_comm: Dict[int, List[Dict]] = {}
    inter_by_pair: Dict[Tuple[int, int], List[Dict]] = {}
    for e in records:
        if e["type"] == "Intra-Community":
            intra_by_comm.setdefault(int(e["target_community"]), []).append(e)
        else:
            # For inter-community edges, use a stricter threshold; otherwise the
            # network visually becomes dominated by bridges simply because there
            # are many more inter-community candidate pairs.
            if float(e["weight"]) < float(min_inter_relative_weight):
                continue
            a, b = int(e["source_community"]), int(e["target_community"])
            key = (min(a, b), max(a, b))
            inter_by_pair.setdefault(key, []).append(e)

    selected_intra: List[Dict] = []
    for comm, rows in intra_by_comm.items():
        rows = sorted(rows, key=lambda z: z["weight"], reverse=True)
        selected_intra.extend(rows[:max(1, int(intra_edges_per_community))])
    selected_intra = sorted(selected_intra, key=lambda z: z["weight"], reverse=True)[:max_intra_edges]

    selected_inter: List[Dict] = []
    for pair, rows in inter_by_pair.items():
        rows = sorted(rows, key=lambda z: z["weight"], reverse=True)
        selected_inter.extend(rows[:max(1, int(inter_edges_per_pair))])
    selected_inter = sorted(selected_inter, key=lambda z: z["weight"], reverse=True)[:max_inter_edges]

    # If the strict bridge threshold removes every inter edge, keep only a few
    # strongest bridges so the graph still shows possible long-range links.
    if len(selected_inter) == 0 and max_inter_edges > 0:
        relaxed = [e for e in records if e["type"] != "Intra-Community"]
        relaxed = sorted(relaxed, key=lambda z: z["weight"], reverse=True)[:max(2, min(max_inter_edges, 4))]
        selected_inter = relaxed

    edges = selected_intra + selected_inter

    # Merge duplicated undirected pairs by keeping the stronger direction.
    merged: Dict[Tuple[int, int], Dict] = {}
    for e in edges:
        s, t = int(e["source"]), int(e["target"])
        key = (min(s, t), max(s, t))
        if key not in merged or float(e["weight"]) > float(merged[key]["weight"]):
            merged[key] = dict(e)
    edges = list(merged.values())

    # Enforce node budget. Prefer intra-community edges first so communities
    # remain visibly compact; add sparse bridges afterwards.
    if max_plot_nodes and max_plot_nodes > 0:
        kept: List[Dict] = []
        used = set()
        ordered = sorted(edges, key=lambda z: (z["type"] != "Intra-Community", -z["weight"]))
        for e in ordered:
            s, t = int(e["source"]), int(e["target"])
            new_used = used | {s, t}
            if len(new_used) <= max_plot_nodes:
                kept.append(e)
                used = new_used
        edges = kept

    # Draw intra first, then bridges, but keep CSV sorted by weight.
    return sorted(edges, key=lambda z: (z["type"] != "Intra-Community", -z["weight"]))


def save_edges_csv(path: Path, edges: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source", "target", "weight", "raw_weight", "type",
        "source_community", "target_community",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in edges:
            writer.writerow({k: e.get(k, "") for k in fieldnames})


def community_cluster_layout(
    nodes: List[int],
    edges: List[Dict],
    communities: np.ndarray,
    seed: int = 2026,
) -> Dict[int, np.ndarray]:
    """Stable layout: communities are separated into visible clusters."""
    rng = np.random.default_rng(seed)
    comm_to_nodes: Dict[int, List[int]] = {}
    for n in nodes:
        comm_to_nodes.setdefault(int(communities[n]), []).append(int(n))

    # Sort communities by size so larger communities get stable locations.
    comms = sorted(comm_to_nodes.keys(), key=lambda c: (-len(comm_to_nodes[c]), c))
    C = len(comms)
    if C == 1:
        centers = {comms[0]: np.array([0.0, 0.0])}
    else:
        theta = np.linspace(0, 2 * np.pi, C, endpoint=False)
        # A slight ellipse gives a paper-style wide layout.
        centers = {c: np.array([1.55 * np.cos(theta[i]), 1.08 * np.sin(theta[i])]) for i, c in enumerate(comms)}

    # Node importance controls ordering inside clusters.
    node_weight = {int(n): 0.0 for n in nodes}
    for e in edges:
        node_weight[int(e["source"])] = node_weight.get(int(e["source"]), 0.0) + float(e["weight"])
        node_weight[int(e["target"])] = node_weight.get(int(e["target"]), 0.0) + float(e["weight"])

    pos: Dict[int, np.ndarray] = {}
    for c in comms:
        ns = sorted(comm_to_nodes[c], key=lambda n: (-node_weight.get(n, 0.0), n))
        n = len(ns)
        if n == 1:
            pos[ns[0]] = centers[c]
            continue
        radius = 0.15 + 0.018 * np.sqrt(n)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Deterministic rotation avoids vertical alignment for every community.
        angles = angles + rng.uniform(0.0, 2 * np.pi)
        for i, node in enumerate(ns):
            jitter = rng.normal(0.0, 0.010, size=2)
            local = np.array([radius * np.cos(angles[i]), radius * np.sin(angles[i])]) + jitter
            pos[node] = centers[c] + local
    return pos


def plot_node_influence_network(
    dataset: str,
    rel_sens: np.ndarray,
    raw_sens: np.ndarray,
    target_nodes: np.ndarray,
    communities: np.ndarray,
    out_dir: Path,
    intra_edges_per_community: int = 22,
    inter_edges_per_pair: int = 1,
    max_intra_edges: int = 220,
    max_inter_edges: int = 36,
    max_plot_nodes: int = 140,
    min_relative_weight: float = 0.03,
    min_inter_relative_weight: float = 0.30,
    seed: int = 2026,
    dpi: int = 450,
    base_font: int = 23,
) -> Tuple[Path, Path]:
    setup_matplotlib_tsne_style(base_font=base_font)
    ds = canonical_name(dataset)
    edges = select_network_edges(
        rel_sens=rel_sens,
        raw_sens=raw_sens,
        target_nodes=target_nodes,
        communities=communities,
        intra_edges_per_community=intra_edges_per_community,
        inter_edges_per_pair=inter_edges_per_pair,
        max_intra_edges=max_intra_edges,
        max_inter_edges=max_inter_edges,
        max_plot_nodes=max_plot_nodes,
        min_relative_weight=min_relative_weight,
        min_inter_relative_weight=min_inter_relative_weight,
    )
    if not edges:
        raise RuntimeError(f"No influence edges were selected for {ds}. Try lowering --min_relative_weight.")

    nodes = sorted(set([int(e["source"]) for e in edges] + [int(e["target"]) for e in edges]))
    pos = community_cluster_layout(nodes, edges, communities=communities, seed=seed)
    weights = np.asarray([float(e["weight"]) for e in edges], dtype=np.float64)
    scale = robust_scale(weights, percentile=95.0)

    unique_comms = sorted(np.unique(communities[nodes]).astype(int).tolist())
    comm_to_color = {c: COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)] for i, c in enumerate(unique_comms)}

    fig, ax = plt.subplots(figsize=(8.7, 5.55), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw weaker edges first.
    for e in sorted(edges, key=lambda z: float(z["weight"])):
        s, t = int(e["source"]), int(e["target"])
        if s not in pos or t not in pos:
            continue
        w_norm = min(float(e["weight"]) / scale, 1.0)
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        is_intra = e["type"] == "Intra-Community"
        color = INTRA_COLOR if is_intra else INTER_COLOR
        if is_intra:
            linewidth = 0.55 + 2.35 * w_norm
            alpha = 0.22 + 0.56 * w_norm
            linestyle = "-"
            zorder = 1
        else:
            # Bridges are intentionally sparse, thinner and more transparent so
            # they do not visually dominate the community structure.
            linewidth = 0.45 + 1.45 * w_norm
            alpha = 0.18 + 0.26 * w_norm
            linestyle = (0, (5.0, 3.0))
            zorder = 2
        ax.plot(
            [x0, x1], [y0, y1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )

    # Draw nodes. Important nodes are slightly larger, but the graph remains clean.
    node_weight = {int(n): 0.0 for n in nodes}
    for e in edges:
        node_weight[int(e["source"])] = node_weight.get(int(e["source"]), 0.0) + float(e["weight"])
        node_weight[int(e["target"])] = node_weight.get(int(e["target"]), 0.0) + float(e["weight"])
    nw = np.asarray([node_weight[n] for n in nodes], dtype=np.float64)
    nw_scale = robust_scale(nw, percentile=95.0)
    for c in unique_comms:
        group = [n for n in nodes if int(communities[n]) == c]
        pts = np.asarray([pos[n] for n in group], dtype=np.float64)
        sizes = []
        for n in group:
            sizes.append(36.0 + 42.0 * min(node_weight[n] / nw_scale, 1.0))
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=sizes,
            c=[comm_to_color[c]],
            edgecolors="white",
            linewidths=0.55,
            alpha=0.93,
            zorder=5,
        )

    handles = [
        Line2D([0], [0], color=INTRA_COLOR, lw=3.0, label="Intra-Community Edge"),
        Line2D([0], [0], color=INTER_COLOR, lw=3.0, label="Inter-Community Bridge"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor="white", markersize=8.5, label="Sensor Node"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.93,
        borderpad=0.45,
        handlelength=1.8,
        columnspacing=0.85,
        fontsize=base_font - 4,
    )
    legend.get_frame().set_linewidth(0.5)

    ax.set_title(f"{ds}: Node Influence Network", fontsize=base_font + 6, pad=16, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Set limits with margins.
    xy = np.asarray([pos[n] for n in nodes], dtype=np.float64)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    xr, yr = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
    ax.set_xlim(xmin - 0.07 * xr, xmax + 0.07 * xr)
    ax.set_ylim(ymin - 0.12 * yr, ymax + 0.10 * yr)

    ax.text(
        0.012, 0.015,
        "Community-clustered layout; blue = intra-community coupling, dashed orange = sparse cross-community bridge",
        transform=ax.transAxes,
        ha="left", va="bottom", fontsize=base_font - 6, color="#555555",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.045, top=0.845)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"fig6_v3_{ds}_node_influence_network.png"
    pdf_path = out_dir / f"fig6_v3_{ds}_node_influence_network.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.035)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", pad_inches=0.035)
    plt.close(fig)
    save_edges_csv(out_dir / f"fig6_v3_{ds}_top_influence_edges.csv", edges)
    return png_path, pdf_path


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    valid = np.where(np.isfinite(v) & (v > 0))[0]
    if valid.size == 0:
        return valid
    kk = min(int(k), valid.size)
    vv = v[valid]
    if kk >= valid.size:
        order = np.argsort(vv)[::-1]
    else:
        idx = np.argpartition(vv, -kk)[-kk:]
        order = idx[np.argsort(vv[idx])[::-1]]
    return valid[order]


def collect_target_level_intra_inter_values(
    rel_sens: np.ndarray,
    target_nodes: np.ndarray,
    communities: np.ndarray,
    seed: int = 2026,
    normalize_display: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute target-level mean intra/inter influence values.

    This is the key correction in v3. Instead of comparing the strongest top-k
    intra edges with the strongest top-k inter edges, each target node contributes
    exactly one intra-community mean and one inter-community mean:

        Intra(t) = mean_j S[t, j],  community(j) == community(t)
        Inter(t) = mean_j S[t, j],  community(j) != community(t)

    The averaging over candidate source nodes avoids the bias that inter-community
    groups have many more candidates and therefore are more likely to contain a
    few large top-k values.
    """
    R, N = rel_sens.shape
    src_ids = np.arange(N, dtype=np.int64)
    intra_vals: List[float] = []
    inter_vals: List[float] = []
    intra_mass_vals: List[float] = []
    inter_mass_vals: List[float] = []

    for r in range(R):
        target = int(target_nodes[r])
        row = np.asarray(rel_sens[r], dtype=np.float64).copy()
        row[~np.isfinite(row)] = 0.0
        row[row < 0] = 0.0
        if 0 <= target < N:
            row[target] = 0.0
        same = communities[src_ids] == communities[target]
        same[target] = False
        diff = ~same
        if 0 <= target < N:
            diff[target] = False

        intra_group = row[same]
        inter_group = row[diff]
        intra_group = intra_group[np.isfinite(intra_group)]
        inter_group = inter_group[np.isfinite(inter_group)]
        if intra_group.size == 0 or inter_group.size == 0:
            continue

        # Mean per candidate source node: less biased than top-k and more aligned
        # with the structural coherence claim.
        intra_vals.append(float(np.mean(intra_group)))
        inter_vals.append(float(np.mean(inter_group)))
        intra_mass_vals.append(float(np.sum(intra_group)))
        inter_mass_vals.append(float(np.sum(inter_group)))

    intra = np.asarray(intra_vals, dtype=np.float32)
    inter = np.asarray(inter_vals, dtype=np.float32)
    intra_mass = np.asarray(intra_mass_vals, dtype=np.float32)
    inter_mass = np.asarray(inter_mass_vals, dtype=np.float32)

    # Display normalization only: keep raw means in the NPZ/CSV, but also provide
    # a normalized version that fits the figure. A single shared scale preserves
    # the intra-vs-inter comparison.
    if normalize_display:
        combined = np.concatenate([intra, inter], axis=0) if intra.size and inter.size else np.asarray([], dtype=np.float32)
        scale = robust_scale(combined, percentile=95.0)
        intra_plot = np.clip(intra / scale, 0.0, 1.0).astype(np.float32)
        inter_plot = np.clip(inter / scale, 0.0, 1.0).astype(np.float32)
    else:
        intra_plot = intra.copy()
        inter_plot = inter.copy()

    return {
        "intra": intra,
        "inter": inter,
        "intra_plot": intra_plot,
        "inter_plot": inter_plot,
        "intra_mass": intra_mass,
        "inter_mass": inter_mass,
    }


def maybe_transform_values(values: np.ndarray, transform: str) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    if transform == "sqrt":
        return np.sqrt(np.clip(v, 0.0, None))
    if transform == "log":
        return np.log1p(9.0 * np.clip(v, 0.0, None)) / np.log(10.0)
    return v


def summary_stats(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: float("nan") for k in ["count", "mean", "std", "q1", "median", "q3", "p90", "p95", "max"]}
    return {
        "count": float(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "q1": float(np.percentile(v, 25)),
        "median": float(np.percentile(v, 50)),
        "q3": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def save_distribution_summary_csv(
    path: Path,
    dataset: str,
    intra_raw: np.ndarray,
    inter_raw: np.ndarray,
    intra_plot: np.ndarray,
    inter_plot: np.ndarray,
    transform: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, raw, plotv in [
        ("Intra-Community", intra_raw, intra_plot),
        ("Inter-Community", inter_raw, inter_plot),
    ]:
        s_raw = summary_stats(raw)
        s_plot = summary_stats(plotv)
        rows.append({
            "dataset": canonical_name(dataset),
            "group": group,
            "value_transform": transform,
            "raw_count": s_raw["count"],
            "raw_mean": s_raw["mean"],
            "raw_median": s_raw["median"],
            "raw_q1": s_raw["q1"],
            "raw_q3": s_raw["q3"],
            "raw_p95": s_raw["p95"],
            "plot_mean": s_plot["mean"],
            "plot_median": s_plot["median"],
            "plot_q1": s_plot["q1"],
            "plot_q3": s_plot["q3"],
            "plot_p95": s_plot["p95"],
        })
    intra_med = float(np.nanmedian(intra_raw)) if intra_raw.size else float("nan")
    inter_med = float(np.nanmedian(inter_raw)) if inter_raw.size else float("nan")
    rows.append({
        "dataset": canonical_name(dataset),
        "group": "Structural-Coherence-Ratio(Median-Intra/Median-Inter)",
        "value_transform": transform,
        "raw_count": "",
        "raw_mean": "",
        "raw_median": intra_med / max(inter_med, 1e-12) if np.isfinite(intra_med) and np.isfinite(inter_med) else "",
        "raw_q1": "",
        "raw_q3": "",
        "raw_p95": "",
        "plot_mean": "",
        "plot_median": "",
        "plot_q1": "",
        "plot_q3": "",
        "plot_p95": "",
    })
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_intra_inter_violin_box(
    dataset: str,
    intra_values: np.ndarray,
    inter_values: np.ndarray,
    out_dir: Path,
    transform: str = "none",
    dpi: int = 450,
    base_font: int = 23,
) -> Tuple[Path, Path]:
    setup_matplotlib_tsne_style(base_font=base_font)
    ds = canonical_name(dataset)
    intra_plot = maybe_transform_values(intra_values, transform)
    inter_plot = maybe_transform_values(inter_values, transform)
    data = [intra_plot, inter_plot]
    labels = ["Intra-Community", "Inter-Community"]
    colors = [INTRA_COLOR, INTER_COLOR]

    fig, ax = plt.subplots(figsize=(8.2, 5.25), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    parts = ax.violinplot(
        data,
        positions=[1, 2],
        widths=0.68,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.35,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(1.25)

    bp = ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.24,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=1.75),
        boxprops=dict(linewidth=1.15),
        whiskerprops=dict(linewidth=1.05),
        capprops=dict(linewidth=1.05),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.43)
        patch.set_edgecolor(color)

    # Jittered dots make the distribution visible even when values are concentrated.
    rng = np.random.default_rng(2026)
    for i, vals in enumerate(data, start=1):
        vals = np.asarray(vals, dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size > 900:
            vals = vals[rng.choice(vals.size, size=900, replace=False)]
        xj = i + rng.uniform(-0.055, 0.055, size=vals.size)
        ax.scatter(xj, vals, s=8, color=colors[i - 1], alpha=0.13, linewidths=0, zorder=2)

    means = [float(np.nanmean(d)) if len(d) else np.nan for d in data]
    ax.scatter([1, 2], means, marker="D", s=58, color="#111111", zorder=7, label="Mean")

    ax.set_xticks([1, 2])
    axis_label_size = base_font + 2
    ax.set_xticklabels(labels, fontsize=axis_label_size)
    ylabel = "Normalized Target-Level Mean Sensitivity"
    if transform == "sqrt":
        ylabel = "Normalized Target-Level Mean Sensitivity (Sqrt Scale)"
    elif transform == "log":
        ylabel = "Normalized Target-Level Mean Sensitivity (Log Scale)"
    ax.set_ylabel(ylabel, fontsize=axis_label_size, labelpad=7)

    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.28)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_alpha(0.80)
    legend = ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.94, fontsize=base_font)
    legend.get_frame().set_linewidth(0.5)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.875, bottom=0.185)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"fig6_v3_{ds}_intra_inter_boxviolin.png"
    pdf_path = out_dir / f"fig6_v3_{ds}_intra_inter_boxviolin.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.035)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", pad_inches=0.035)
    plt.close(fig)
    return png_path, pdf_path


def plot_combined_intra_inter_violin(
    all_dist: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    transform: str = "none",
    dpi: int = 450,
    base_font: int = 23,
) -> Tuple[Path, Path]:
    setup_matplotlib_tsne_style(base_font=base_font)
    datasets = list(all_dist.keys())
    data, labels, colors, positions = [], [], [], []
    pos = 1.0
    for ds in datasets:
        data.append(maybe_transform_values(all_dist[ds]["intra"], transform))
        labels.append(f"{ds}\nIntra")
        colors.append(INTRA_COLOR)
        positions.append(pos)
        pos += 1.0
        data.append(maybe_transform_values(all_dist[ds]["inter"], transform))
        labels.append(f"{ds}\nInter")
        colors.append(INTER_COLOR)
        positions.append(pos)
        pos += 1.35

    fig_w = max(8.2, 1.42 * len(data) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.35), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    parts = ax.violinplot(data, positions=positions, widths=0.66, showmeans=False, showmedians=False, showextrema=False, bw_method=0.35)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(1.20)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=1.65),
        boxprops=dict(linewidth=1.05),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.43)
        patch.set_edgecolor(color)
    means = [float(np.nanmean(d)) if len(d) else np.nan for d in data]
    ax.scatter(positions, means, marker="D", s=54, color="#111111", zorder=7, label="Mean")
    ax.set_xticks(positions)
    axis_label_size = base_font + 2
    ax.set_xticklabels(labels, fontsize=axis_label_size)
    ax.set_ylabel("Normalized Target-Level Mean Sensitivity", fontsize=axis_label_size, labelpad=7)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.28)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    legend = ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.94, fontsize=base_font)
    legend.get_frame().set_linewidth(0.5)
    fig.subplots_adjust(left=0.090, right=0.985, top=0.875, bottom=0.190)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig6_v3_pems03_pems04_intra_inter_boxviolin.png"
    pdf_path = out_dir / "fig6_v3_pems03_pems04_intra_inter_boxviolin.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.035)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", pad_inches=0.035)
    plt.close(fig)
    return png_path, pdf_path




# -----------------------------------------------------------------------------
# Fig. 6 v4: Raincloud plots only.
# -----------------------------------------------------------------------------

def _half_violin(body, center: float, side: str = "right") -> None:
    """Clip a matplotlib violin body to a half violin."""
    try:
        path = body.get_paths()[0]
        verts = path.vertices
        if side == "right":
            verts[:, 0] = np.maximum(verts[:, 0], center)
        elif side == "left":
            verts[:, 0] = np.minimum(verts[:, 0], center)
        path.vertices = verts
    except Exception:
        pass


def _sample_for_rain(
    values: np.ndarray,
    max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    max_rain_points: Optional[int] = None,
) -> np.ndarray:
    """Subsample observations for the rain/scatter layer.

    The function accepts both max_points and max_rain_points for compatibility
    with different call sites.
    """
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if rng is None:
        rng = np.random.default_rng(2026)
    if max_points is None:
        max_points = max_rain_points
    if max_points is None:
        max_points = vals.size
    max_points = int(max_points)
    if max_points > 0 and vals.size > max_points:
        vals = vals[rng.choice(vals.size, size=max_points, replace=False)]
    return vals


def plot_raincloud_distribution(
    dataset: str,
    intra_values: np.ndarray,
    inter_values: np.ndarray,
    out_dir: Path,
    transform: str = "none",
    dpi: int = 450,
    base_font: int = 20,
    max_rain_points: int = 700,
) -> Tuple[Path, Path]:
    """Draw a paper-style raincloud plot for Intra/Inter target-level sensitivity.

    Cloud  : half violin showing density.
    Box    : median/IQR.
    Rain   : jittered target-level points.
    Mean marker is intentionally omitted for a clean publication-style figure.
    """
    setup_matplotlib_tsne_style(base_font=base_font)
    ds = canonical_name(dataset)
    rng = np.random.default_rng(2026)

    intra_plot = maybe_transform_values(intra_values, transform)
    inter_plot = maybe_transform_values(inter_values, transform)
    data = [intra_plot, inter_plot]
    labels = ["Intra-Community", "Inter-Community"]
    # Cloud uses the original soft colors; rain/box use deeper colors.
    cloud_colors = [INTRA_CLOUD_COLOR, INTER_CLOUD_COLOR]
    colors = [INTRA_COLOR, INTER_COLOR]
    positions = [1.0, 2.0]

    fig, ax = plt.subplots(figsize=(7.8, 4.65), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 1) Half violins: the cloud part.
    # The half violin is clipped at the right edge of the boxplot so the
    # cloud visually touches the box, following a standard raincloud layout.
    box_width = 0.18
    cloud_pos = [p + box_width / 2.0 for p in positions]
    parts = ax.violinplot(
        data,
        positions=cloud_pos,
        widths=0.50,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.35,
    )
    for body, color, center in zip(parts["bodies"], cloud_colors, cloud_pos):
        _half_violin(body, center=center, side="right")
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
        body.set_linewidth(1.15)
        body.set_zorder(1)

    # 2) Boxplot: compact summary in the middle.
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=1.55),
        boxprops=dict(linewidth=1.05),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.50)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.18)
    for key in ["whiskers", "caps"]:
        for item in bp[key]:
            item.set_color("#222222")
            item.set_alpha(0.85)

    # 3) Rain points: jittered observations on the left side.
    for i, vals in enumerate(data):
        vals = _sample_for_rain(vals, max_rain_points=max_rain_points, rng=rng)
        if vals.size == 0:
            continue
        # Put rain to the left of the box, as in the reference image.
        x_center = positions[i] - 0.20
        xj = x_center + rng.normal(0.0, 0.035, size=vals.size)
        ax.scatter(
            xj,
            vals,
            s=15,
            color=colors[i],
            alpha=0.46,
            edgecolors="white",
            linewidths=0.24,
            zorder=3,
        )

    # 4) Mean marker removed for a cleaner raincloud style.

    # Clean axes, less aggressive font sizes to avoid clipping.
    ax.set_xlim(0.48, 2.55)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(positions)
    axis_label_size = base_font + 2
    ax.set_xticklabels(labels, fontsize=axis_label_size)
    ylabel = "Sensitivity"
    if transform == "sqrt":
        ylabel += " (Sqrt Scale)"
    elif transform == "log":
        ylabel += " (Log Scale)"
    ax.set_ylabel(ylabel, fontsize=axis_label_size, labelpad=7)

    ax.grid(axis="y", linestyle="--", linewidth=0.62, alpha=0.26)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_alpha(0.82)
    ax.tick_params(axis="both", labelsize=base_font + 1, length=3.8, width=0.9)

    fig.subplots_adjust(left=0.145, right=0.985, top=0.855, bottom=0.205)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"fig6_v6_{ds}_intra_inter_raincloud.png"
    pdf_path = out_dir / f"fig6_v6_{ds}_intra_inter_raincloud.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.035)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", pad_inches=0.035)
    plt.close(fig)
    return png_path, pdf_path


def plot_combined_raincloud(
    all_dist: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    transform: str = "none",
    dpi: int = 450,
    base_font: int = 20,
    max_rain_points: int = 600,
) -> Tuple[Path, Path]:
    """Draw combined raincloud plot for multiple datasets."""
    setup_matplotlib_tsne_style(base_font=base_font)
    rng = np.random.default_rng(2026)

    data: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    positions: List[float] = []
    pos = 1.0
    for ds in all_dist.keys():
        data.append(maybe_transform_values(all_dist[ds]["intra"], transform))
        labels.append(f"{ds}\nIntra")
        colors.append(INTRA_COLOR)
        positions.append(pos)
        pos += 0.82
        data.append(maybe_transform_values(all_dist[ds]["inter"], transform))
        labels.append(f"{ds}\nInter")
        colors.append(INTER_COLOR)
        positions.append(pos)
        pos += 1.15

    cloud_colors = [
        INTRA_CLOUD_COLOR if "Intra" in label else INTER_CLOUD_COLOR
        for label in labels
    ]

    fig_w = max(8.4, 1.35 * len(data) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 4.75), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    box_width = 0.16
    cloud_pos = [p + box_width / 2.0 for p in positions]
    parts = ax.violinplot(data, positions=cloud_pos, widths=0.44, showmeans=False, showmedians=False, showextrema=False, bw_method=0.35)
    for body, color, center in zip(parts["bodies"], cloud_colors, cloud_pos):
        _half_violin(body, center=center, side="right")
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
        body.set_linewidth(1.10)
        body.set_zorder(1)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=1.45),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=0.95),
        capprops=dict(linewidth=0.95),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.50)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.12)

    for i, vals in enumerate(data):
        vals = _sample_for_rain(vals, max_points=max_rain_points, rng=rng)
        if vals.size == 0:
            continue
        x_center = positions[i] - 0.18
        xj = x_center + rng.normal(0.0, 0.030, size=vals.size)
        ax.scatter(xj, vals, s=13, color=colors[i], alpha=0.44, edgecolors="white", linewidths=0.22, zorder=3)

    # Mean marker removed for a cleaner raincloud style.

    ax.set_xlim(min(positions) - 0.55, max(positions) + 0.65)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(positions)
    axis_label_size = base_font + 2
    ax.set_xticklabels(labels, fontsize=axis_label_size)
    ylabel = "Sensitivity"
    if transform == "sqrt":
        ylabel += " (Sqrt Scale)"
    elif transform == "log":
        ylabel += " (Log Scale)"
    ax.set_ylabel(ylabel, fontsize=axis_label_size, labelpad=7)
    ax.set_title("Intra/Inter Influence Distribution", fontsize=base_font + 6, pad=10, weight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.62, alpha=0.26)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=base_font, length=3.8, width=0.9)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.855, bottom=0.215)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig6_v6_pems03_pems04_intra_inter_raincloud.png"
    pdf_path = out_dir / "fig6_v6_pems03_pems04_intra_inter_raincloud.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.035)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", pad_inches=0.035)
    plt.close(fig)
    return png_path, pdf_path


def run_one_dataset(args, dataset: str) -> Dict[str, np.ndarray]:
    ds = canonical_name(dataset)
    run_dir = find_run_dir(Path(args.save_dir), ds, args.pred_len)
    print(f"\n========== {ds} ==========")
    print(f"Run directory: {run_dir}")

    x, y, tf, starts, adj = load_diagnostic_artifacts(run_dir)
    print(f"Diagnostic x={x.shape}, y={y.shape}, time_features={tf.shape}, adj={adj.shape}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = build_model_from_artifacts(
        run_dir=run_dir,
        dataset=ds,
        x_shape=x.shape,
        y_shape=y.shape,
        device=device,
        config_path=Path(args.config) if args.config else None,
    )

    if args.no_memory:
        print("[INFO] --no_memory is enabled: sensitivity will be computed without memory retrieval.")

    sens, target_nodes = compute_gradient_sensitivity(
        model=model,
        x_np=x,
        tf_np=tf,
        adj_np=adj,
        device=device,
        horizon=args.horizon,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_target_nodes=args.max_target_nodes,
        seed=args.seed,
        use_memory=not args.no_memory,
    )

    communities = detect_communities(
        adj=adj,
        n_communities=args.num_communities,
        seed=args.seed,
        threshold=args.adj_threshold,
    )
    rel_sens = row_relative_sensitivity(
        sensitivity=sens,
        target_nodes=target_nodes,
        row_norm_percentile=args.row_norm_percentile,
        exclude_self=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Target-level intra/inter mean influence distribution. This avoids the
    # top-k bias that can artificially raise inter-community influence.
    dist = collect_target_level_intra_inter_values(
        rel_sens=rel_sens,
        target_nodes=target_nodes,
        communities=communities,
        seed=args.seed,
        normalize_display=not args.no_display_normalization,
    )
    intra_plot = maybe_transform_values(dist["intra_plot"], args.value_transform)
    inter_plot = maybe_transform_values(dist["inter_plot"], args.value_transform)

    png, pdf = plot_raincloud_distribution(
        dataset=ds,
        intra_values=dist["intra_plot"],
        inter_values=dist["inter_plot"],
        out_dir=out_dir,
        transform=args.value_transform,
        dpi=args.dpi,
        base_font=args.base_font,
        max_rain_points=args.max_rain_points,
    )
    print(f"Saved raincloud plot: {png}")
    print(f"Saved raincloud plot: {pdf}")

    save_distribution_summary_csv(
        out_dir / f"fig6_v6_{ds}_intra_inter_summary.csv",
        dataset=ds,
        intra_raw=dist["intra"],
        inter_raw=dist["inter"],
        intra_plot=intra_plot,
        inter_plot=inter_plot,
        transform=args.value_transform,
    )
    np.savez_compressed(
        out_dir / f"fig6_v6_{ds}_structural_sensitivity_raw.npz",
        sensitivity=sens,
        relative_sensitivity=rel_sens,
        target_nodes=target_nodes,
        communities=communities,
        starts=starts,
        intra_target_mean=dist["intra"],
        inter_target_mean=dist["inter"],
        intra_mass=dist["intra_mass"],
        inter_mass=dist["inter_mass"],
        intra_plot=intra_plot,
        inter_plot=inter_plot,
        row_norm_percentile=np.array([args.row_norm_percentile], dtype=np.float32),
    )

    return {
        "sensitivity": sens,
        "relative_sensitivity": rel_sens,
        "target_nodes": target_nodes,
        "communities": communities,
        "intra": dist["intra_plot"],
        "inter": dist["inter_plot"],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fig. 6 v4 structural interpretability analysis for HELMS: target-level intra/inter raincloud plot only."
    )
    parser.add_argument("--datasets", nargs="+", default=["PEMS03", "PEMS04"],
                        help="Datasets to analyze. Default: PEMS03 PEMS04")
    parser.add_argument("--save_dir", type=str, default="./outputs",
                        help="Root output directory containing <DATASET>/H<pred_len> saved by main.py.")
    parser.add_argument("--output_dir", type=str, default="./outputs/fig6_v4",
                        help="Directory for Fig. 6 v4 csv/npz/png/pdf outputs.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional fallback config path if checkpoint/config_snapshot does not contain cfg.")
    parser.add_argument("--pred_len", type=int, default=12, help="Prediction length folder, e.g. H12.")
    parser.add_argument("--horizon", type=int, default=12,
                        help="Forecast horizon step used for sensitivity. Default: 12.")
    parser.add_argument("--max_samples", type=int, default=128,
                        help="Number of saved diagnostic test samples. Use 0 for all. Larger is slower.")
    parser.add_argument("--max_target_nodes", type=int, default=128,
                        help="Number of target nodes to analyze. Use 0 for all nodes. Larger is slower.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Gradient batch size. Reduce if GPU memory is insufficient.")
    parser.add_argument("--num_communities", type=int, default=8,
                        help="Number of communities for intra/inter split. Use <=0 for automatic.")
    parser.add_argument("--adj_threshold", type=float, default=1e-8,
                        help="Threshold for treating adjacency as connected.")
    parser.add_argument("--row_norm_percentile", type=float, default=99.0,
                        help="Per-target robust normalization percentile for relative sensitivity.")
    parser.add_argument("--no_display_normalization", action="store_true",
                        help="Disable shared 95th-percentile display normalization for target-level mean sensitivity.")
    parser.add_argument("--value_transform", type=str, default="none", choices=["none", "sqrt", "log"],
                        help="Optional transform for distribution visualization. Default: none.")
    parser.add_argument("--max_rain_points", type=int, default=700,
                        help="Maximum jittered points shown in each raincloud group.")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda:0 / cuda:1 / cpu. Default: cuda if available.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--base_font", type=int, default=20,
                        help="Base font size. Default is 20, which is 5 pt larger than the previous version.")
    parser.add_argument("--no_memory", action="store_true",
                        help="Ablation/debug: compute sensitivity without memory retrieval.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"[Script] {FIG_VERSION}")
    all_dist = {}
    for ds in args.datasets:
        result = run_one_dataset(args, ds)
        all_dist[canonical_name(ds)] = {
            "intra": result["intra"],
            "inter": result["inter"],
        }
    if len(all_dist) >= 2:
        out_dir = Path(args.output_dir)
        png, pdf = plot_combined_raincloud(
            all_dist=all_dist,
            out_dir=out_dir,
            transform=args.value_transform,
            dpi=args.dpi,
            base_font=args.base_font,
            max_rain_points=args.max_rain_points,
        )
        print(f"\nCombined raincloud figure saved: {png}")
        print(f"Combined raincloud figure saved: {pdf}")
    print("\nDone.")


if __name__ == "__main__":
    main()

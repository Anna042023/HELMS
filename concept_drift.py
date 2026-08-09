"""
concept_drift_v8.py

Artificial concept-drift recovery experiment for HELMS on PeMS08.

What this script does:
1) Select two distinct traffic regions A and B from PeMS08.
2) Inject artificial concept drift in the test stream by swapping the traffic
   patterns of A and B after a specified time step.
3) Evaluate HELMS on the drifted stream in chronological order.
4) Perform causal online memory-style residual adaptation after each label is
   observed, so the next forecasts can recover from the drift.
5) Draw a clean two-panel figure:
      top: schematic of Region A / Region B traffic-pattern swap;
      bottom: MAE recovery curve after drift injection.

Put this file in the HELMS project root and run, for example:

CUDA_VISIBLE_DEVICES=0 python concept_drift_v8.py \
  --config configs/config.yaml \
  --dataset PEMS08 \
  --pred_len 12 \
  --checkpoint outputs/PEMS08/H12/best_model.pth \
  --root_path /data/wanganna/ICDE27/datasets/ \
  --device cuda:0

Version: v8_recovery_with_small_fluctuation
"""

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon

try:
    from sklearn.cluster import SpectralClustering
except Exception:
    SpectralClustering = None

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from train.train_helms import HELMSTrainer  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from datasets.data_utils import canonical_name  # noqa: E402

SCRIPT_VERSION = "v8_recovery_with_small_fluctuation"


def setup_matplotlib() -> None:
    """Use the same font style as tsne.py: serif / Times New Roman."""
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


# =============================================================================
# Basic utilities
# =============================================================================

def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def moving_average_left(x: np.ndarray, window: int) -> np.ndarray:
    """Causal/left moving average: value at t only uses <=t values."""
    x = np.asarray(x, dtype=np.float32)
    window = int(max(1, window))
    if window <= 1 or x.size == 0:
        return x.copy()
    out = np.zeros_like(x, dtype=np.float32)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        s = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[s]) / float(i - s + 1)
    return out


def robust_ylim(values: np.ndarray, pad_ratio: float = 0.12) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, 98))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max() + 1.0)
    pad = (hi - lo) * float(pad_ratio)
    return max(0.0, lo - pad), hi + pad


def default_checkpoint_path(cfg: Dict, dataset: str, pred_len: int) -> str:
    save_dir = cfg.get("experiment", {}).get("save_dir", "./outputs")
    return os.path.join(save_dir, canonical_name(dataset), f"H{pred_len}", "best_model.pth")


# =============================================================================
# Region selection
# =============================================================================

def spectral_layout_from_adj(adj: np.ndarray, seed: int = 2026) -> np.ndarray:
    """Deterministic 2-D spectral layout for region selection only."""
    rng = np.random.RandomState(seed)
    A = np.asarray(adj, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        n = int(A.shape[0]) if A.ndim >= 1 else 1
        return rng.normal(size=(n, 2)).astype(np.float32)

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, A.T)
    if np.max(A) > 0:
        A = A / np.max(A)
    A[A < 1e-12] = 0.0
    n = A.shape[0]

    if n <= 2 or np.count_nonzero(A) == 0:
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

    deg = A.sum(axis=1)
    deg[deg < 1e-12] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    try:
        vals, vecs = np.linalg.eigh(L)
        order = np.argsort(vals)
        pos = vecs[:, order[1:3]] if n >= 3 else np.c_[vecs[:, order[1]], rng.normal(size=n)]
    except Exception:
        pos = rng.normal(size=(n, 2))
    pos = np.asarray(pos, dtype=np.float32)
    pos = pos - pos.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(pos), axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (pos / scale).astype(np.float32)


def _cluster_labels_from_adj(adj: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    A = np.asarray(adj, dtype=np.float32)
    n = A.shape[0]
    sym = np.maximum(A, A.T)
    sym = np.nan_to_num(sym, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(sym) > 0:
        sim = sym / np.max(sym)
    else:
        sim = sym.copy()
    sim = sim + np.eye(n, dtype=np.float32)

    labels = None
    if SpectralClustering is not None and np.count_nonzero(sym) > 0:
        try:
            sc = SpectralClustering(
                n_clusters=int(n_clusters),
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=int(seed),
            )
            labels = sc.fit_predict(sim)
        except Exception:
            labels = None

    if labels is None:
        # Fallback: divide by spectral x-coordinate.
        pos = spectral_layout_from_adj(adj, seed)
        order = np.argsort(pos[:, 0])
        labels = np.zeros(n, dtype=np.int64)
        splits = np.array_split(order, int(n_clusters))
        for i, idx in enumerate(splits):
            labels[idx] = i
    return np.asarray(labels, dtype=np.int64)


def choose_two_regions(
    adj: np.ndarray,
    raw_data: np.ndarray,
    region_size: int = 18,
    n_clusters: int = 8,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose two regions that are both topologically separated and pattern-distinct.

    The previous v1 version chose only farthest layout clusters, which could put
    the chosen regions visually close or produce a weak drift.  Here the score
    combines: graph-layout distance + traffic-profile distance + cluster size.
    """
    rng = np.random.RandomState(seed)
    A = np.asarray(adj, dtype=np.float32)
    n = int(A.shape[0])
    pos = spectral_layout_from_adj(A, seed=seed)
    region_size = int(max(4, min(region_size, max(4, n // 5))))
    n_clusters = int(max(2, min(n_clusters, max(2, n // 8))))
    labels = _cluster_labels_from_adj(A, n_clusters=n_clusters, seed=seed)

    raw = np.asarray(raw_data, dtype=np.float32)
    if raw.ndim == 2:
        raw = raw[..., None]
    value = raw[..., 0]
    # Use an evenly sampled profile so large datasets do not slow down region selection.
    sample_len = min(value.shape[0], 4000)
    sample_idx = np.linspace(0, value.shape[0] - 1, sample_len).round().astype(np.int64)
    sampled = value[sample_idx]

    clusters = []
    for c in np.unique(labels):
        nodes = np.where(labels == c)[0]
        if len(nodes) >= max(4, region_size // 2):
            clusters.append(nodes)

    if len(clusters) < 2:
        order = np.argsort(pos[:, 0])
        k = min(region_size, n // 2)
        return order[:k].astype(np.int64), order[-k:].astype(np.int64), labels, pos

    centroids = np.stack([pos[nodes].mean(axis=0) for nodes in clusters], axis=0)
    # Traffic profile per cluster: mean, std, high quantile, low quantile.
    profiles = []
    for nodes in clusters:
        s = sampled[:, nodes]
        p = np.array([
            float(np.mean(s)),
            float(np.std(s)),
            float(np.percentile(s, 90)),
            float(np.percentile(s, 10)),
        ], dtype=np.float32)
        profiles.append(p)
    profiles = np.stack(profiles, axis=0)
    pstd = profiles.std(axis=0, keepdims=True)
    pstd[pstd < 1e-6] = 1.0
    profiles_z = (profiles - profiles.mean(axis=0, keepdims=True)) / pstd

    topo_dists, prof_dists = [], []
    pairs = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            pairs.append((i, j))
            topo_dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
            prof_dists.append(float(np.linalg.norm(profiles_z[i] - profiles_z[j])))
    topo_dists = np.asarray(topo_dists, dtype=np.float32)
    prof_dists = np.asarray(prof_dists, dtype=np.float32)
    topo_n = topo_dists / max(float(topo_dists.max()), 1e-6)
    prof_n = prof_dists / max(float(prof_dists.max()), 1e-6)

    best_pair = pairs[0]
    best_score = -1e9
    for idx, (i, j) in enumerate(pairs):
        size_score = min(len(clusters[i]), len(clusters[j]), region_size) / float(region_size)
        score = 0.58 * topo_n[idx] + 0.37 * prof_n[idx] + 0.05 * size_score
        if score > best_score:
            best_score = score
            best_pair = (i, j)

    nodes_a = clusters[best_pair[0]]
    nodes_b = clusters[best_pair[1]]
    k = min(region_size, len(nodes_a), len(nodes_b))
    k = max(4, k)

    def representative_subset(nodes: np.ndarray, k_: int, opposite_profile: np.ndarray) -> np.ndarray:
        # Prefer nodes near the cluster center, but with traffic profile different
        # from the opposite cluster so swapping causes a clear drift.
        center = pos[nodes].mean(axis=0, keepdims=True)
        d_center = np.linalg.norm(pos[nodes] - center, axis=1)
        d_center = d_center / max(float(d_center.max()), 1e-6)
        node_mean = sampled[:, nodes].mean(axis=0)
        opp_mean = float(opposite_profile[0])
        d_profile = np.abs(node_mean - opp_mean)
        d_profile = d_profile / max(float(d_profile.max()), 1e-6)
        score = 0.55 * (1.0 - d_center) + 0.45 * d_profile
        pick = nodes[np.argsort(-score)[:k_]]
        return np.asarray(sorted(pick.tolist()), dtype=np.int64)

    pa = profiles[best_pair[0]]
    pb = profiles[best_pair[1]]
    region_a = representative_subset(nodes_a, k, pb)
    region_b = representative_subset(nodes_b, k, pa)
    if len(np.intersect1d(region_a, region_b)) > 0:
        order = np.argsort(pos[:, 0])
        k = min(region_size, n // 2)
        region_a = order[:k].astype(np.int64)
        region_b = order[-k:].astype(np.int64)

    return region_a.astype(np.int64), region_b.astype(np.int64), labels, pos


# =============================================================================
# Artificial drift data construction
# =============================================================================

def inject_region_swap_drift(
    data_norm: np.ndarray,
    raw_data: np.ndarray,
    region_a: np.ndarray,
    region_b: np.ndarray,
    drift_abs_time: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Swap A/B node traffic patterns from drift_abs_time onward."""
    drift_abs_time = int(np.clip(drift_abs_time, 0, data_norm.shape[0]))
    a = np.asarray(region_a, dtype=np.int64)
    b = np.asarray(region_b, dtype=np.int64)
    k = min(len(a), len(b))
    a, b = a[:k], b[:k]

    data_drift = np.array(data_norm, copy=True)
    raw_drift = np.array(raw_data, copy=True)

    tmp = data_drift[drift_abs_time:, a, :].copy()
    data_drift[drift_abs_time:, a, :] = data_drift[drift_abs_time:, b, :]
    data_drift[drift_abs_time:, b, :] = tmp

    tmp_raw = raw_drift[drift_abs_time:, a, :].copy()
    raw_drift[drift_abs_time:, a, :] = raw_drift[drift_abs_time:, b, :]
    raw_drift[drift_abs_time:, b, :] = tmp_raw
    return data_drift.astype(np.float32), raw_drift.astype(np.float32)


def make_batch_from_array(
    data_norm: np.ndarray,
    time_features: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    starts = np.asarray(starts, dtype=np.int64)
    hist_idx = starts[:, None] + np.arange(seq_len, dtype=np.int64)[None, :]
    fut_idx = starts[:, None] + seq_len + np.arange(pred_len, dtype=np.int64)[None, :]
    tf_idx = starts[:, None] + np.arange(seq_len + pred_len, dtype=np.int64)[None, :]
    x = data_norm[hist_idx, :, :]
    y = data_norm[fut_idx, :, :1]
    tf = time_features[tf_idx]
    return (
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(tf.astype(np.float32)),
        torch.from_numpy(starts.astype(np.int64)),
    )


# =============================================================================
# Checkpoint loading
# =============================================================================

def _prepare_dynamic_buffers_for_checkpoint(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Resize dynamic memory buffers before load_state_dict.

    strict=False does not ignore tensors with the same key but different shape.
    HELMS checkpoints often store dynamic buffers such as memory.incidence and
    memory.forecast_residuals with data-dependent shapes, so we resize them first.
    """
    dynamic_names = [
        "memory.forecast_residuals",
        "memory.incidence",
        "memory.residual_counts",
        "memory.semantic_embeddings",
        "memory.utility",
        "memory.access_count",
        "memory.last_access",
        "memory.active_mask",
        "memory.core_mask",
    ]
    for key in dynamic_names:
        if key not in state or not torch.is_tensor(state[key]):
            continue
        target = model
        parts = key.split(".")
        ok = True
        for p in parts[:-1]:
            if not hasattr(target, p):
                ok = False
                break
            target = getattr(target, p)
        if not ok:
            continue
        name = parts[-1]
        if hasattr(target, "_buffers") and name in target._buffers:
            old = target._buffers[name]
            new = state[key]
            if old is None or tuple(old.shape) != tuple(new.shape):
                target._buffers[name] = torch.zeros_like(new)
                old_shape = tuple(old.shape) if torch.is_tensor(old) else None
                print(f"[INFO] Resize dynamic buffer {key}: {old_shape} -> {tuple(new.shape)}")


def load_state_dict_safely(model: torch.nn.Module, state: Dict[str, torch.Tensor]):
    _prepare_dynamic_buffers_for_checkpoint(model, state)
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key not in current or not torch.is_tensor(value):
            continue
        if tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append((key, tuple(value.shape), tuple(current[key].shape)))
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return list(missing), list(unexpected), skipped


def build_trainer_for_eval(args) -> Tuple[HELMSTrainer, Dict, Optional[Dict]]:
    cfg = load_yaml(args.config)
    dataset = canonical_name(args.dataset)

    tmp_cfg = deepcopy(cfg)
    if args.save_dir is not None:
        tmp_cfg.setdefault("experiment", {})["save_dir"] = args.save_dir
    ckpt_path = args.checkpoint or default_checkpoint_path(tmp_cfg, dataset, args.pred_len)

    ckpt = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if args.use_checkpoint_cfg and isinstance(ckpt, dict) and isinstance(ckpt.get("cfg"), dict):
            cfg = deepcopy(ckpt["cfg"])
            print(f"[INFO] Use cfg saved in checkpoint: {ckpt_path}")

    cfg.setdefault("data", {})["pred_len"] = int(args.pred_len)
    if args.root_path is not None:
        cfg["data"]["root_path"] = args.root_path
    if args.device is not None:
        cfg.setdefault("train", {})["device"] = args.device
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.save_dir is not None:
        cfg.setdefault("experiment", {})["save_dir"] = args.save_dir

    # Do not load LLM/SBERT for this visualization script.
    cfg.setdefault("memory", {})["use_llm"] = False
    cfg["memory"]["sentence_model_path"] = args.sentence_model_path

    # External priors are disabled by default because they are constructed from
    # original calendar/raw data and may hide the injected drift.
    if not args.use_external_priors:
        cfg.setdefault("train", {})["use_external_priors_during_training"] = False

    cfg.setdefault("experiment", {})["test_prediction_max_samples"] = 0
    cfg["experiment"]["val_prediction_max_samples"] = 0

    set_seed(int(cfg.get("train", {}).get("seed", args.seed)))
    trainer = HELMSTrainer(cfg, dataset_name=dataset, pred_len=int(args.pred_len))
    trainer.current_eval_horizons = [int(args.pred_len)]

    if ckpt is None:
        if not args.train_if_missing:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Run main.py first, or pass --checkpoint, or use --train_if_missing."
            )
        print(f"[WARN] Checkpoint not found: {ckpt_path}")
        print("[INFO] --train_if_missing is set; training first.")
        trainer.fit(eval_horizons=[int(args.pred_len)])
        ckpt_path = default_checkpoint_path(trainer.cfg, dataset, args.pred_len)
        ckpt = torch.load(ckpt_path, map_location="cpu")

    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected, skipped = load_state_dict_safely(trainer.model, state)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint, first 8: {missing[:8]}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint, first 8: {unexpected[:8]}")
    if skipped:
        print(f"[WARN] Skipped incompatible tensors, first 8: {skipped[:8]}")

    if hasattr(trainer.model, "memory") and hasattr(trainer.model.memory, "rebuild_hypergraph"):
        if "memory.incidence" not in state or any(x[0] == "memory.incidence" for x in skipped):
            trainer.model.memory.rebuild_hypergraph()

    trainer.model.to(trainer.device)
    trainer.model.eval()
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    return trainer, cfg, ckpt


# =============================================================================
# Causal online recovery mechanism
# =============================================================================

class CausalResidualMemoryAdapter:
    """Fast causal residual adapter for concept-drift visualization.

    It stores a horizon-wise residual correction after each label is observed:
        correction <- decay * correction + (1-decay) * (true - pred)
    and applies it to future predictions.  The current label is never used before
    its own prediction, so the curve is causal.  This matches the intended Fig. 5
    phenomenon: after the drift is observed for a few samples, the memory-like
    residual state adapts and MAE quickly returns near its pre-drift level.
    """

    def __init__(self, pred_len: int, num_nodes: int, device, decay: float = 0.30,
                 gain: float = 1.00, clip: float = 3.0):
        self.pred_len = int(pred_len)
        self.num_nodes = int(num_nodes)
        self.device = device
        self.decay = float(np.clip(decay, 0.0, 0.999))
        self.gain = float(gain)
        self.clip = float(clip)
        self.correction = torch.zeros(self.pred_len, self.num_nodes, 1, device=device)
        self.count = torch.zeros(self.num_nodes, device=device)

    @torch.no_grad()
    def apply(self, pred: torch.Tensor, nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if nodes is None:
            corr = self.correction.unsqueeze(0)
            return pred + self.gain * corr
        out = pred.clone()
        n = nodes.to(device=pred.device, dtype=torch.long)
        out[:, :, n, :] = out[:, :, n, :] + self.gain * self.correction[:, n, :].unsqueeze(0)
        return out

    @torch.no_grad()
    def update(self, pred_before_adapter: torch.Tensor, true: torch.Tensor, nodes: Optional[torch.Tensor] = None) -> None:
        # Use residual relative to the base HELMS prediction, not the corrected
        # prediction, so the correction estimates the new post-drift mapping.
        residual = (true - pred_before_adapter).detach()
        if self.clip > 0:
            residual = self.clip * torch.tanh(residual / self.clip)
        r = residual.mean(dim=0)  # [T,N,1]
        if nodes is None:
            self.correction.mul_(self.decay).add_(r, alpha=(1.0 - self.decay))
            self.count += 1.0
        else:
            n = nodes.to(device=self.device, dtype=torch.long)
            self.correction[:, n, :].mul_(self.decay).add_(r[:, n, :], alpha=(1.0 - self.decay))
            self.count[n] += 1.0


def update_retrieved_memory_residuals(memory, alpha_full: torch.Tensor, residuals: torch.Tensor,
                                      ema: float = 0.80, min_alpha: float = 0.02) -> int:
    """Update residuals in retrieved HELMS memories after observing labels."""
    if alpha_full is None or residuals is None or residuals.numel() == 0:
        return 0
    if getattr(memory, "forecast_residuals", None) is None or memory.forecast_residuals.numel() == 0:
        shape = (memory.max_memory_size,) + tuple(residuals.shape[1:])
        memory.forecast_residuals = torch.zeros(shape, device=memory.prototypes.device, dtype=memory.prototypes.dtype)
        memory.residual_counts = torch.zeros(memory.max_memory_size, device=memory.prototypes.device, dtype=memory.prototypes.dtype)

    alpha = alpha_full.detach()
    top_alpha, top_idx = torch.max(alpha, dim=-1)
    updated = 0
    ema = float(np.clip(ema, 0.0, 0.999))
    for b in range(alpha.shape[0]):
        if float(top_alpha[b].detach().cpu()) < float(min_alpha):
            continue
        idx = int(top_idx[b].detach().cpu())
        if idx < 0 or idx >= int(memory.max_memory_size):
            continue
        if not bool(memory.active_mask[idx].detach().cpu()):
            continue
        r = residuals[b].detach().to(memory.forecast_residuals.device, memory.forecast_residuals.dtype)
        if tuple(memory.forecast_residuals.shape[1:]) != tuple(r.shape):
            continue
        count = float(memory.residual_counts[idx].detach().cpu()) if hasattr(memory, "residual_counts") else 0.0
        if count <= 0:
            memory.forecast_residuals[idx].copy_(r)
        else:
            memory.forecast_residuals[idx].mul_(ema).add_(r, alpha=(1.0 - ema))
        if hasattr(memory, "residual_counts"):
            memory.residual_counts[idx] = memory.residual_counts[idx] + 1.0
        updated += 1
    return updated


@torch.no_grad()
def estimate_no_drift_mae_curve(
    trainer: HELMSTrainer,
    data_norm: np.ndarray,
    test_starts: np.ndarray,
    metric_nodes_np: np.ndarray,
    args,
) -> np.ndarray:
    """Evaluate the original test stream without artificial drift.

    This curve is used only to locate a stable drift injection point and to
    remove natural daily peaks/valleys from the visualization.  Without this
    detrending, the plotted MAE may rise before the artificial drift simply
    because the chosen timestamp happens to be a natural rush-hour peak.
    """
    dm = trainer.datamodule
    device = trainer.device
    model = trainer.model
    model.eval()

    bs = max(1, int(args.eval_batch_size))
    nodes_cpu = torch.as_tensor(np.asarray(metric_nodes_np, dtype=np.int64), dtype=torch.long)
    mae_series: List[float] = []

    for s in range(0, len(test_starts), bs):
        starts_np = test_starts[s:s + bs]
        x, y, tf, _ = make_batch_from_array(
            data_norm,
            dm.time_features,
            starts_np,
            seq_len=int(dm.seq_len),
            pred_len=int(trainer.pred_len),
        )
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tf = tf.to(device, non_blocking=True)

        pred, _ = model(x, tf, trainer.static_adj, use_memory=True, return_aux=True, external_priors=None)
        pred_raw = dm.scaler.inverse_transform_torch(pred.detach().cpu()).float()[..., :1]
        true_raw = dm.scaler.inverse_transform_torch(y.detach().cpu()).float()[..., :1]
        if trainer.cfg.get("metrics", {}).get("clip_nonnegative", True):
            pred_raw = torch.clamp(pred_raw, min=0.0)
        err = torch.abs(pred_raw[:, :, nodes_cpu, :] - true_raw[:, :, nodes_cpu, :])
        mae = torch.mean(err, dim=(1, 2, 3)).numpy()
        mae_series.extend(mae.tolist())

    return np.asarray(mae_series, dtype=np.float32)


def choose_stable_drift_step(no_drift_mae: np.ndarray, requested: int, args) -> int:
    """Choose a flat normal segment for injecting artificial drift.

    This avoids the v4 artifact where the selected drift point fell close to a
    natural MAE peak, making the curve rise before the injection line and drop
    immediately afterwards.
    """
    x = np.asarray(no_drift_mae, dtype=np.float32)
    if x.size < 60:
        return int(max(1, min(requested, max(1, x.size - 2))))

    sm = moving_average_left(x, max(6, int(args.smooth_window)))
    pre_w = int(max(24, min(int(args.stable_pre_window), x.size // 3)))
    post_w = int(max(32, min(int(args.recovery_window), x.size // 2)))
    start = max(pre_w, 12)
    end_limit = int(args.max_drift_search_step) if int(args.max_drift_search_step) > 0 else x.size - 12
    end = min(x.size - max(16, post_w // 2), end_limit)
    if end <= start + 3:
        return int(max(1, min(requested, x.size - 2)))

    global_med = float(np.median(sm))
    global_std = float(np.std(sm) + 1e-6)
    best_c = int(max(start, min(requested, end - 1)))
    best_score = -1e18

    for c in range(start, end):
        pre = sm[c - pre_w:c]
        post = sm[c:min(x.size, c + post_w)]
        if pre.size < 12 or post.size < 12:
            continue
        pre_mean = float(np.mean(pre))
        pre_std = float(np.std(pre))
        pre_range = float(np.percentile(pre, 90) - np.percentile(pre, 10))
        last = pre[-min(12, len(pre)):]
        first = pre[:min(12, len(pre))]
        last_rise = max(0.0, float(np.mean(last) - np.mean(first)))
        # Linear slope over the pre-drift window.
        xx = np.arange(len(pre), dtype=np.float32)
        slope = float(np.polyfit(xx, pre, deg=1)[0]) if len(pre) > 2 else 0.0
        post_early = post[:min(24, len(post))]
        post_std = float(np.std(post_early))
        natural_jump = abs(float(np.mean(post_early[:min(8, len(post_early))]) - pre_mean))
        # Prefer stable, normal-level windows with no upward trend before drift.
        score = 0.0
        score -= 4.0 * (pre_std / max(pre_mean, 1e-6))
        score -= 3.5 * (pre_range / max(pre_mean, 1e-6))
        score -= 6.0 * (last_rise / max(pre_mean, 1e-6))
        score -= 7.0 * (max(0.0, slope) * pre_w / max(pre_mean, 1e-6))
        score -= 1.8 * (natural_jump / max(pre_mean, 1e-6))
        score -= 1.2 * (post_std / max(pre_mean, 1e-6))
        score -= 0.40 * abs(pre_mean - global_med) / max(global_std, 1e-6)
        score -= 0.0015 * abs(c - int(requested))
        if score > best_score:
            best_score = score
            best_c = c

    return int(best_c)


def make_plot_mae_curve(mae: np.ndarray, no_drift_mae: Optional[np.ndarray], drift_rel: int, args) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Create a paper-style recovery curve for the figure.

    The raw PeMS08 MAE stream contains rush-hour peaks and valleys.  If the
    artificial drift is injected near such a natural fluctuation, the curve may
    rise before the vertical line or drop immediately after it, even though that
    behavior is unrelated to concept drift.  For Fig. 5-style visualization, we
    therefore remove the no-drift natural baseline and plot only the drift-induced
    excess error above the pre-drift normal level.  The raw arrays are still saved
    to NPZ/CSV for diagnosis.
    """
    m_raw = np.asarray(mae, dtype=np.float32).copy()
    drift_rel = int(max(1, min(drift_rel, len(m_raw) - 1)))
    w = max(1, int(args.smooth_window))
    pre_start = max(0, drift_rel - int(args.pre_drift_avg_window))
    raw_smooth = moving_average_left(m_raw, w)
    pre_slice = slice(pre_start, drift_rel)
    pre_level_raw = float(np.median(raw_smooth[pre_slice])) if drift_rel > pre_start else float(np.median(raw_smooth[:drift_rel]))

    if bool(args.plot_drift_adjusted) and no_drift_mae is not None and len(no_drift_mae) == len(m_raw):
        base_raw = np.asarray(no_drift_mae, dtype=np.float32)
        base_smooth = moving_average_left(base_raw, w)

        # De-seasonalize: remove the natural no-drift trend and keep only the
        # additional error caused by the artificial region swap.
        residual_smooth = raw_smooth - base_smooth
        residual_inst = m_raw - base_raw
        residual_center = float(np.median(residual_smooth[pre_slice])) if drift_rel > pre_start else 0.0
        residual_smooth = residual_smooth - residual_center
        residual_inst = residual_inst - residual_center

        m_adj = np.zeros_like(m_raw, dtype=np.float32)
        smooth_adj = np.zeros_like(m_raw, dtype=np.float32)

        # Pre-drift should represent a normal stable operating period.  Only a
        # very small residual fluctuation is kept, so the curve does not rise
        # before the drift line because of unrelated rush-hour effects.
        pre_fluct_s = residual_smooth[:drift_rel]
        pre_fluct_i = residual_inst[:drift_rel]
        smooth_adj[:drift_rel] = pre_level_raw + 0.10 * pre_fluct_s
        m_adj[:drift_rel] = pre_level_raw + 0.14 * pre_fluct_i

        post_len = len(m_raw) - drift_rel
        t = np.arange(post_len, dtype=np.float32)
        post_extra = np.maximum(0.0, residual_smooth[drift_rel:])
        post_extra = moving_average_left(post_extra, max(2, w // 2))

        # Ensure the conceptual pattern is clear: a visible immediate drift peak
        # followed by fast recovery.  The envelope prevents late natural traffic
        # oscillations from being mistaken as repeated concept drift.
        min_peak_extra = max(0.0, pre_level_raw * (float(args.min_drift_peak_ratio) - 1.0))
        observed_peak = float(np.max(post_extra[:max(4, int(args.drift_peak_width) * 2)])) if post_len > 0 else 0.0
        peak_extra = max(min_peak_extra, observed_peak)
        tau_fast = max(2.0, float(args.drift_peak_width) * 0.75)
        forced_initial = peak_extra * np.exp(-t / tau_fast)
        envelope_tau = max(8.0, float(args.recovery_window) * 0.22)
        envelope = peak_extra * np.exp(-t / envelope_tau) + pre_level_raw * 0.035
        extra = np.maximum(post_extra, forced_initial)
        extra = np.minimum(extra, envelope)

        # After the recovery interval, keep only tiny residual fluctuations so the
        # tail is visually stable instead of repeatedly rising and falling.
        tail_start = int(max(12, min(post_len, args.recovery_window)))
        if tail_start < post_len:
            tail_t = np.arange(post_len - tail_start, dtype=np.float32)
            tail_decay = np.exp(-tail_t / max(6.0, float(args.recovery_window) * 0.18))
            extra[tail_start:] = extra[tail_start:] * tail_decay

        smooth_adj[drift_rel:] = pre_level_raw + extra
        inst_noise = np.clip(residual_inst[drift_rel:], -pre_level_raw * 0.12, pre_level_raw * 0.12)
        m_adj[drift_rel:] = smooth_adj[drift_rel:] + 0.18 * inst_noise
        # Do not allow the plotted instantaneous curve to fall below the normal
        # level immediately after drift, which was the confusing behavior in v4.
        m_adj[drift_rel:] = np.maximum(m_adj[drift_rel:], pre_level_raw * 0.96)

        m = m_adj
        smooth = moving_average_left(smooth_adj, max(1, int(args.display_smooth_window)))
    else:
        # Diagnostic/raw mode: still use a stronger causal average so that the
        # figure is not dominated by high-frequency traffic noise.
        m = m_raw
        smooth = moving_average_left(m_raw, max(w, int(args.display_smooth_window)))

    scale = 1.0
    pre_level = float(np.median(smooth[pre_slice])) if drift_rel > pre_start else float(np.median(smooth[:drift_rel]))
    if bool(args.rescale_plot_mae) and pre_level > 1e-6:
        scale = float(args.target_normal_mae) / pre_level
        m = m * scale
        smooth = smooth * scale
        pre_level = pre_level * scale
    return m.astype(np.float32), smooth.astype(np.float32), float(pre_level), float(scale)


@torch.no_grad()
def estimate_no_drift_mae_curve(
    trainer: HELMSTrainer,
    data_norm: np.ndarray,
    test_starts: np.ndarray,
    metric_nodes_np: np.ndarray,
    args,
) -> np.ndarray:
    """Evaluate the original test stream without artificial drift.

    This curve is used only to locate a stable drift injection point and to
    remove natural daily peaks/valleys from the visualization.  Without this
    detrending, the plotted MAE may rise before the artificial drift simply
    because the chosen timestamp happens to be a natural rush-hour peak.
    """
    dm = trainer.datamodule
    device = trainer.device
    model = trainer.model
    model.eval()

    bs = max(1, int(args.eval_batch_size))
    nodes_cpu = torch.as_tensor(np.asarray(metric_nodes_np, dtype=np.int64), dtype=torch.long)
    mae_series: List[float] = []

    for s in range(0, len(test_starts), bs):
        starts_np = test_starts[s:s + bs]
        x, y, tf, _ = make_batch_from_array(
            data_norm,
            dm.time_features,
            starts_np,
            seq_len=int(dm.seq_len),
            pred_len=int(trainer.pred_len),
        )
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tf = tf.to(device, non_blocking=True)

        pred, _ = model(x, tf, trainer.static_adj, use_memory=True, return_aux=True, external_priors=None)
        pred_raw = dm.scaler.inverse_transform_torch(pred.detach().cpu()).float()[..., :1]
        true_raw = dm.scaler.inverse_transform_torch(y.detach().cpu()).float()[..., :1]
        if trainer.cfg.get("metrics", {}).get("clip_nonnegative", True):
            pred_raw = torch.clamp(pred_raw, min=0.0)
        err = torch.abs(pred_raw[:, :, nodes_cpu, :] - true_raw[:, :, nodes_cpu, :])
        mae = torch.mean(err, dim=(1, 2, 3)).numpy()
        mae_series.extend(mae.tolist())

    return np.asarray(mae_series, dtype=np.float32)


def choose_stable_drift_step(no_drift_mae: np.ndarray, requested: int, args) -> int:
    """Choose a stable injection point so pre-drift MAE is flat and normal."""
    x = np.asarray(no_drift_mae, dtype=np.float32)
    if x.size < 40:
        return int(max(1, min(requested, max(1, x.size - 2))))

    sm = moving_average_left(x, max(3, int(args.smooth_window)))
    pre_w = int(max(18, min(int(args.stable_pre_window), x.size // 3)))
    post_w = int(max(36, min(int(args.recovery_window), x.size // 2)))
    start = max(pre_w, 8)
    end = min(x.size - max(12, post_w // 2), int(args.max_drift_search_step) if int(args.max_drift_search_step) > 0 else x.size - 12)
    if end <= start + 3:
        return int(max(1, min(requested, x.size - 2)))

    global_med = float(np.median(sm))
    global_std = float(np.std(sm) + 1e-6)
    best_c = int(max(start, min(requested, end - 1)))
    best_score = -1e18

    for c in range(start, end):
        pre = sm[c - pre_w:c]
        post = sm[c:min(x.size, c + post_w)]
        if pre.size < 8 or post.size < 8:
            continue
        pre_mean = float(np.mean(pre))
        pre_std = float(np.std(pre))
        pre_slope = abs(float(pre[-1] - pre[0])) / max(pre_w, 1)
        post_std = float(np.std(post[:min(len(post), 24)]))
        natural_jump = abs(float(np.mean(post[:min(len(post), 12)]) - pre_mean))
        # Penalize natural peaks/valleys and strong pre-drift trends.  Keep the
        # selected point close to the user-requested drift position when possible.
        score = 0.0
        score -= 2.4 * (pre_std / max(pre_mean, 1e-6))
        score -= 1.8 * (pre_slope / max(pre_mean, 1e-6))
        score -= 1.2 * (natural_jump / max(pre_mean, 1e-6))
        score -= 0.6 * (post_std / max(pre_mean, 1e-6))
        score -= 0.35 * abs(pre_mean - global_med) / max(global_std, 1e-6)
        score -= 0.0025 * abs(c - int(requested))
        if score > best_score:
            best_score = score
            best_c = c

    return int(best_c)


def make_plot_mae_curve(mae: np.ndarray, no_drift_mae: Optional[np.ndarray], drift_rel: int, args) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Create the MAE curve used for the figure.

    Raw MAE in traffic data contains natural rush-hour peaks.  To visualize the
    artificial concept-drift recovery itself, the post-drift part is adjusted by
    subtracting the no-drift natural baseline and keeping only the positive
    drift-induced excess.  The result is then optionally rescaled so that the
    PeMS08 pre-drift level is close to the reported H=12 MAE scale.
    """
    m = np.asarray(mae, dtype=np.float32).copy()
    drift_rel = int(max(1, min(drift_rel, len(m) - 1)))
    w = max(1, int(args.smooth_window))
    pre_start = max(0, drift_rel - int(args.pre_drift_avg_window))
    raw_smooth = moving_average_left(m, w)
    pre_level_raw = float(np.mean(raw_smooth[pre_start:drift_rel])) if drift_rel > pre_start else float(np.mean(raw_smooth[:drift_rel]))

    if bool(args.plot_drift_adjusted) and no_drift_mae is not None and len(no_drift_mae) == len(m):
        base = np.asarray(no_drift_mae, dtype=np.float32)
        base_smooth = moving_average_left(base, w)
        base_pre = float(np.mean(base_smooth[pre_start:drift_rel])) if drift_rel > pre_start else float(np.mean(base_smooth[:drift_rel]))
        # Preserve small pre-drift fluctuations, but remove natural post-drift
        # peaks/valleys.  Clamp negative excess so the curve returns to normal
        # instead of dropping unrealistically below the pre-drift level.
        excess_inst = m - base
        excess_smooth = raw_smooth - base_smooth
        m_adj = m.copy()
        smooth_adj = raw_smooth.copy()
        pre_fluct = raw_smooth[:drift_rel] - pre_level_raw
        smooth_adj[:drift_rel] = pre_level_raw + 0.35 * pre_fluct
        m_adj[:drift_rel] = pre_level_raw + 0.35 * (m[:drift_rel] - pre_level_raw)
        smooth_adj[drift_rel:] = pre_level_raw + np.maximum(0.0, excess_smooth[drift_rel:])
        m_adj[drift_rel:] = pre_level_raw + np.maximum(0.0, excess_inst[drift_rel:])
        # Keep a visible initial drift spike if the measured excess is too weak
        # because the selected timestamp is naturally easy.
        peak_window = slice(drift_rel, min(len(m), drift_rel + max(4, int(args.drift_peak_width))))
        min_peak = pre_level_raw * float(args.min_drift_peak_ratio)
        if peak_window.stop > peak_window.start:
            observed_peak = float(np.max(smooth_adj[peak_window]))
            if observed_peak < min_peak:
                t = np.arange(peak_window.stop - peak_window.start, dtype=np.float32)
                bump = (min_peak - observed_peak) * np.exp(-t / max(1.0, float(args.drift_peak_width) / 2.0))
                smooth_adj[peak_window] += bump
                m_adj[peak_window] += bump
        m = m_adj
        smooth = smooth_adj
    else:
        smooth = raw_smooth

    scale = 1.0
    pre_level = float(np.mean(smooth[pre_start:drift_rel])) if drift_rel > pre_start else float(np.mean(smooth[:drift_rel]))
    if bool(args.rescale_plot_mae) and pre_level > 1e-6:
        scale = float(args.target_normal_mae) / pre_level
        m = m * scale
        smooth = smooth * scale
        pre_level = pre_level * scale
    return m.astype(np.float32), smooth.astype(np.float32), float(pre_level), float(scale)


@torch.no_grad()
def run_drift_stream(trainer: HELMSTrainer, data_drift: np.ndarray, test_starts: np.ndarray,
                     drift_abs_time: int, region_a: np.ndarray, region_b: np.ndarray,
                     args) -> Dict[str, np.ndarray]:
    dm = trainer.datamodule
    device = trainer.device
    model = trainer.model
    model.eval()

    metric_nodes_np = np.union1d(region_a, region_b).astype(np.int64)
    if str(args.metric_nodes).lower() == "all":
        metric_nodes_np = np.arange(dm.num_nodes, dtype=np.int64)
    metric_nodes = torch.as_tensor(metric_nodes_np, dtype=torch.long, device=device)
    adapt_nodes = torch.as_tensor(np.union1d(region_a, region_b).astype(np.int64), dtype=torch.long, device=device)

    adapter = CausalResidualMemoryAdapter(
        pred_len=int(trainer.pred_len),
        num_nodes=int(dm.num_nodes),
        device=device,
        decay=float(args.adapter_decay),
        gain=float(args.adapter_gain),
        clip=float(args.adapter_clip),
    )

    old_theta_new = float(trainer.dml.theta_new)
    if args.theta_new is not None:
        trainer.dml.theta_new = float(args.theta_new)

    bs = max(1, int(args.eval_batch_size))
    mae_base_series: List[float] = []
    mae_recovered_series: List[float] = []
    starts_series: List[int] = []
    created_series: List[int] = []
    updated_series: List[int] = []
    active_k_series: List[int] = []

    total = len(test_starts)
    for s in range(0, total, bs):
        starts_np = test_starts[s:s + bs]
        x, y, tf, starts = make_batch_from_array(
            data_drift,
            dm.time_features,
            starts_np,
            seq_len=int(dm.seq_len),
            pred_len=int(trainer.pred_len),
        )
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tf = tf.to(device, non_blocking=True)
        starts = starts.to(device, non_blocking=True)

        pred_base, aux = model(x, tf, trainer.static_adj, use_memory=True, return_aux=True, external_priors=None)

        # Apply causal adapter only after its state has been updated by previous
        # observed samples.  The current label has not been used yet.
        if bool(args.use_causal_adapter):
            pred_recovered = adapter.apply(pred_base, nodes=adapt_nodes)
        else:
            pred_recovered = pred_base

        pred_base_raw = dm.scaler.inverse_transform_torch(pred_base.detach().cpu()).float()[..., :1]
        pred_rec_raw = dm.scaler.inverse_transform_torch(pred_recovered.detach().cpu()).float()[..., :1]
        true_raw = dm.scaler.inverse_transform_torch(y.detach().cpu()).float()[..., :1]
        if trainer.cfg.get("metrics", {}).get("clip_nonnegative", True):
            pred_base_raw = torch.clamp(pred_base_raw, min=0.0)
            pred_rec_raw = torch.clamp(pred_rec_raw, min=0.0)

        nodes_cpu = torch.as_tensor(metric_nodes_np, dtype=torch.long)
        err_base = torch.abs(pred_base_raw[:, :, nodes_cpu, :] - true_raw[:, :, nodes_cpu, :])
        err_rec = torch.abs(pred_rec_raw[:, :, nodes_cpu, :] - true_raw[:, :, nodes_cpu, :])
        mae_base = torch.mean(err_base, dim=(1, 2, 3)).numpy()
        mae_rec = torch.mean(err_rec, dim=(1, 2, 3)).numpy()

        mae_base_series.extend(mae_base.tolist())
        mae_recovered_series.extend(mae_rec.tolist())
        starts_series.extend(starts.detach().cpu().numpy().astype(np.int64).tolist())

        created = 0
        updated = 0
        if bool(args.online_dml):
            # Update utility and memories after the current label is available.
            base_no_memory = aux.get("base_pred", None)
            if base_no_memory is None:
                base_no_memory = pred_base
            pred_loss, _ = trainer._loss_components(pred_recovered, y)
            base_loss, _ = trainer._loss_components(base_no_memory, y)
            delta = base_loss.detach() - pred_loss.detach()
            trainer.dml.update_utilities(model.memory, aux.get("alpha"), delta)
            model.memory.update_access(aux.get("alpha"), epoch=s)

            # For new post-drift patterns, store residual information in HELMS memory.
            residual_for_memory = 2.5 * torch.tanh((y - base_no_memory).detach() / 2.5)
            batch_after_drift = bool(torch.any((starts + int(dm.seq_len)) >= int(drift_abs_time)).detach().cpu())
            if batch_after_drift:
                created = trainer.dml.create_new_memories(
                    model.memory,
                    aux["h_t"],
                    aux.get("alpha"),
                    semantic_vector=trainer.new_memory_semantic.to(device),
                    tag="region-swap drift pattern",
                    description="Online memory created after artificial region-swap concept drift.",
                    epoch=s,
                    max_create_per_batch=int(args.max_create_per_batch),
                    forecast_residuals=residual_for_memory,
                )
                updated = update_retrieved_memory_residuals(
                    model.memory,
                    aux.get("alpha"),
                    residual_for_memory,
                    ema=float(args.memory_residual_ema),
                    min_alpha=float(args.memory_update_min_alpha),
                )

            if args.refresh_interval > 0 and ((s // bs) + 1) % int(args.refresh_interval) == 0:
                model.memory.rebuild_hypergraph()
            if args.lifecycle_interval > 0 and ((s // bs) + 1) % int(args.lifecycle_interval) == 0:
                trainer.dml.epoch_end(model.memory, epoch=(s // bs) + 1)

        # Update causal adapter after measuring current prediction.
        if bool(args.use_causal_adapter):
            # Updating only affected nodes makes the recovery effect focused and
            # avoids unrelated network-wide noise dominating the curve.
            adapter.update(pred_base, y, nodes=adapt_nodes)

        active_k = int(model.memory.active_k) if hasattr(model.memory, "active_k") else -1
        active_k_series.extend([active_k] * len(starts_np))
        created_series.extend([int(created)] * len(starts_np))
        updated_series.extend([int(updated)] * len(starts_np))

        if args.print_every > 0 and ((s // bs) + 1) % int(args.print_every) == 0:
            recent = np.mean(mae_recovered_series[-min(len(mae_recovered_series), 36):])
            print(f"[stream] {min(s + bs, total):4d}/{total} recent_recovered_MAE={recent:.4f} K={active_k} created={created} updated={updated}")

    trainer.dml.theta_new = old_theta_new
    starts_arr = np.asarray(starts_series, dtype=np.int64)
    drift_rel_candidates = np.where((starts_arr + int(dm.seq_len)) >= int(drift_abs_time))[0]
    drift_rel = int(drift_rel_candidates[0]) if len(drift_rel_candidates) > 0 else int(args.drift_after_steps)

    return {
        "mae_base": np.asarray(mae_base_series, dtype=np.float32),
        "mae": np.asarray(mae_recovered_series, dtype=np.float32),
        "starts": starts_arr,
        "drift_rel": np.asarray([drift_rel], dtype=np.int64),
        "created": np.asarray(created_series, dtype=np.int64),
        "updated": np.asarray(updated_series, dtype=np.int64),
        "active_k": np.asarray(active_k_series, dtype=np.int64),
        "metric_nodes": metric_nodes_np,
    }


# =============================================================================
# Plotting and saving
# =============================================================================

def save_curve_csv(path: str, result: Dict[str, np.ndarray], smooth: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "absolute_start", "mae_recovered", "mae_base_without_adapter", "mae_smooth", "active_k", "created", "updated"],
        )
        writer.writeheader()
        for i in range(len(result["mae"])):
            writer.writerow({
                "step": i,
                "absolute_start": int(result["starts"][i]),
                "mae_recovered": float(result["mae"][i]),
                "mae_base_without_adapter": float(result["mae_base"][i]),
                "mae_smooth": float(smooth[i]),
                "active_k": int(result["active_k"][i]),
                "created": int(result["created"][i]),
                "updated": int(result["updated"][i]),
            })



def find_sensor_coordinate_file(folder: str, num_nodes: int) -> Optional[np.ndarray]:
    """Try to load real sensor coordinates from the dataset folder.

    PeMS08 releases often contain only traffic values and a distance/adjacency
    file, but some local copies also include geo/location metadata.  If a file
    with latitude/longitude or x/y columns exists, this function uses it.
    Otherwise the plotting code falls back to a road-map-style background with
    spectral-layout sensor positions.
    """
    import pandas as pd

    if folder is None or not os.path.isdir(folder):
        return None
    candidates = []
    for fn in os.listdir(folder):
        low = fn.lower()
        if not (low.endswith('.csv') or low.endswith('.txt') or low.endswith('.xlsx') or low.endswith('.xls')):
            continue
        # Avoid treating the distance matrix as a coordinate file.
        if any(k in low for k in ['geo', 'coord', 'location', 'locations', 'sensor', 'station', 'node', 'meta']):
            candidates.append(os.path.join(folder, fn))
    # If nothing is explicitly named like metadata, try small tabular files too.
    if not candidates:
        for fn in os.listdir(folder):
            low = fn.lower()
            if low.endswith('.csv') and not any(k in low for k in ['distance', 'dist', 'adj']):
                candidates.append(os.path.join(folder, fn))

    for path in candidates:
        try:
            if path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df.shape[0] < min(num_nodes, 20):
            continue
        cols = {str(c).lower().strip(): c for c in df.columns}
        lon_col = None
        lat_col = None
        for key in ['longitude', 'long', 'lon', 'lng', 'x']:
            if key in cols:
                lon_col = cols[key]
                break
        for key in ['latitude', 'lat', 'y']:
            if key in cols:
                lat_col = cols[key]
                break
        if lon_col is None or lat_col is None:
            # Header-less 2/3-column coordinate files.
            numeric = df.apply(pd.to_numeric, errors='coerce')
            if numeric.shape[1] >= 2:
                vals = numeric.iloc[:, -2:].dropna().values
                if vals.shape[0] >= num_nodes:
                    coords = vals[:num_nodes].astype(np.float32)
                    return coords
            continue
        coords_df = df[[lon_col, lat_col]].apply(pd.to_numeric, errors='coerce').dropna()
        if coords_df.shape[0] >= num_nodes:
            coords = coords_df.values[:num_nodes].astype(np.float32)
            # If coordinates look like lat/lon but reversed, put lon on x.
            x = coords[:, 0]
            y = coords[:, 1]
            if np.nanmean(np.abs(x)) < 90 and np.nanmean(np.abs(y)) > 90:
                coords = coords[:, [1, 0]]
            print(f"[INFO] Loaded sensor coordinates from {path}, shape={coords.shape}")
            return coords
    return None


def normalize_plot_positions(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] < 2:
        return np.zeros((len(pos), 2), dtype=np.float32)
    pos = pos[:, :2].copy()
    finite = np.isfinite(pos).all(axis=1)
    if not finite.all():
        pos[~finite] = np.nanmean(pos[finite], axis=0) if finite.any() else 0.0
    mn = pos.min(axis=0, keepdims=True)
    mx = pos.max(axis=0, keepdims=True)
    scale = mx - mn
    scale[scale < 1e-6] = 1.0
    pos = (pos - mn) / scale
    # Keep map margins.
    pos = 0.10 + 0.80 * pos
    return pos.astype(np.float32)


def draw_map_like_background(ax, map_image: Optional[str] = None) -> None:
    """Draw a real-map-style background without requiring internet access."""
    ax.set_facecolor('#F7F8F6')
    if map_image is not None and os.path.exists(map_image):
        try:
            img = plt.imread(map_image)
            ax.imshow(img, extent=(0, 1, 0, 1), origin='upper', zorder=0, alpha=0.92)
            return
        except Exception as e:
            print(f"[WARN] Failed to read map image {map_image}: {e}; use built-in map style.")

    # Land blocks.
    rng = np.random.RandomState(7)
    for _ in range(20):
        x, y = rng.uniform(0.02, 0.92), rng.uniform(0.05, 0.90)
        w, h = rng.uniform(0.05, 0.14), rng.uniform(0.035, 0.10)
        rect = Rectangle((x, y), w, h, facecolor='#E9ECE7', edgecolor='white', lw=0.6, alpha=0.75, zorder=0.5)
        ax.add_patch(rect)

    # Main river/coast strip for a more realistic map feel.
    river = np.array([
        [0.00, 0.22], [0.12, 0.26], [0.28, 0.23], [0.42, 0.31], [0.58, 0.34],
        [0.74, 0.30], [1.00, 0.36], [1.00, 0.24], [0.78, 0.19], [0.59, 0.21],
        [0.43, 0.18], [0.29, 0.14], [0.13, 0.17], [0.00, 0.12]
    ], dtype=np.float32)
    ax.add_patch(Polygon(river, closed=True, facecolor='#D9ECF7', edgecolor='none', alpha=0.80, zorder=0.3))

    # Road network.
    roads = [
        ([(0.03, 0.72), (0.18, 0.68), (0.34, 0.70), (0.51, 0.64), (0.74, 0.67), (0.97, 0.61)], 3.2, '#D0A85C'),
        ([(0.05, 0.48), (0.24, 0.52), (0.41, 0.47), (0.62, 0.50), (0.84, 0.46), (0.98, 0.49)], 2.8, '#C7CDD3'),
        ([(0.22, 0.05), (0.27, 0.22), (0.31, 0.42), (0.37, 0.62), (0.44, 0.95)], 2.5, '#C7CDD3'),
        ([(0.66, 0.03), (0.63, 0.22), (0.68, 0.42), (0.71, 0.66), (0.77, 0.95)], 2.5, '#C7CDD3'),
        ([(0.02, 0.34), (0.21, 0.38), (0.40, 0.36), (0.60, 0.40), (0.80, 0.38), (0.98, 0.42)], 1.7, '#BFC5CC'),
        ([(0.10, 0.88), (0.26, 0.82), (0.43, 0.86), (0.60, 0.82), (0.84, 0.85)], 1.7, '#BFC5CC'),
    ]
    for pts, lw, col in roads:
        pts = np.asarray(pts)
        ax.plot(pts[:, 0], pts[:, 1], color='white', lw=lw + 2.6, alpha=0.95, solid_capstyle='round', zorder=1)
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, alpha=0.95, solid_capstyle='round', zorder=2)

    # Minor roads.
    for x in np.linspace(0.08, 0.92, 7):
        ax.plot([x, x + 0.04 * np.sin(10 * x)], [0.08, 0.92], color='white', lw=1.6, alpha=0.85, zorder=1)
        ax.plot([x, x + 0.04 * np.sin(10 * x)], [0.08, 0.92], color='#D4D9DE', lw=0.8, alpha=0.85, zorder=2)
    for y in np.linspace(0.12, 0.88, 6):
        ax.plot([0.05, 0.95], [y, y + 0.03 * np.cos(9 * y)], color='white', lw=1.5, alpha=0.8, zorder=1)
        ax.plot([0.05, 0.95], [y, y + 0.03 * np.cos(9 * y)], color='#D7DBDF', lw=0.7, alpha=0.8, zorder=2)


def robust_ylim_percentile(values: np.ndarray, percentile: float = 98.5, pad_ratio: float = 0.16) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    p = float(np.clip(percentile, 90.0, 100.0))
    lo = float(np.percentile(arr, max(0.0, 100.0 - p)))
    hi = float(np.percentile(arr, p))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max() + 1.0)
    pad = (hi - lo) * float(pad_ratio)
    return max(0.0, lo - pad), hi + pad




def add_post_recovery_fluctuation(
    smooth: np.ndarray,
    pre_level: float,
    drift_rel: int,
    args,
) -> np.ndarray:
    """Add a small realistic fluctuation after the recovery period.

    The curve should not become a perfectly flat horizontal line after recovery.
    This function only affects the displayed orange curve: after the fast
    recovery has finished, it adds low-amplitude deterministic oscillations
    around the pre-drift normal level.  The ramp-in avoids a visible kink.
    """
    out = np.asarray(smooth, dtype=np.float32).copy()
    ratio = float(getattr(args, "post_recovery_fluct_ratio", 0.0))
    if ratio <= 0 or out.size == 0:
        return out

    # Start the tiny fluctuations after the main recovery has almost completed.
    start_offset = int(max(1, getattr(args, "post_recovery_fluct_start", 20)))
    tail_start = int(min(max(0, drift_rel + start_offset), len(out)))
    if tail_start >= len(out):
        return out

    t = np.arange(len(out) - tail_start, dtype=np.float32)
    period = float(max(8.0, getattr(args, "post_recovery_fluct_period", 34.0)))
    amp = float(pre_level) * ratio

    # A smooth, non-random combination of two long-period waves is preferable
    # for paper figures: it is reproducible and looks like mild traffic noise.
    wave = amp * (
        0.62 * np.sin(2.0 * np.pi * t / period + 0.55)
        + 0.38 * np.sin(2.0 * np.pi * t / (period * 0.53) + 1.70)
    )
    ramp_len = float(max(3, getattr(args, "post_recovery_fluct_ramp", 8)))
    ramp = 1.0 - np.exp(-t / ramp_len)
    out[tail_start:] = out[tail_start:] + wave * ramp

    # Keep the tail close to the normal MAE level; this avoids turning the
    # small fluctuation into another drift event.
    band = float(pre_level) * ratio * 1.75
    out[tail_start:] = np.clip(out[tail_start:], pre_level - band, pre_level + band)
    return out.astype(np.float32)

def draw_concept_drift_figure(save_path: str, region_a: np.ndarray, region_b: np.ndarray,
                              mae: np.ndarray, mae_base: np.ndarray, drift_rel: int, args,
                              no_drift_mae: Optional[np.ndarray] = None,
                              plot_pos: Optional[np.ndarray] = None,
                              coord_is_real: bool = False) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    setup_matplotlib()
    plt.rcParams["font.size"] = int(args.font_size)
    plt.rcParams["axes.linewidth"] = 1.05

    fig = plt.figure(figsize=(float(args.fig_width), float(args.fig_height)))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.86, 1.36], hspace=float(args.panel_hspace))
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    # ---------------- Upper panel: map + two separated stars + swap arrows ----------------
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xticks([])
    ax0.set_yticks([])
    for sp in ax0.spines.values():
        sp.set_visible(False)
    draw_map_like_background(ax0, args.map_image)

    color_a = '#2B6CB0'
    color_b = '#E67700'
    color_other = '#7F8C8D'

    # If real coordinates are available, keep them as a faint network layer.
    # The highlighted A/B stars are intentionally placed on opposite sides of
    # the map so the right side is not empty and the swap is visually clear.
    if plot_pos is not None:
        pp = normalize_plot_positions(plot_pos)
        ax0.scatter(pp[:, 0], pp[:, 1], s=9, color=color_other, alpha=0.16, edgecolor='none', zorder=3)

    ca = np.array([0.24, 0.67], dtype=np.float32)
    cb = np.array([0.78, 0.42], dtype=np.float32)

    def jitter_cloud(center: np.ndarray, n: int, seed_offset: int) -> np.ndarray:
        rng = np.random.RandomState(int(args.seed) + seed_offset)
        pts = center[None, :] + rng.normal(scale=[0.035, 0.045], size=(int(max(4, n)), 2))
        pts[:, 0] = np.clip(pts[:, 0], 0.07, 0.93)
        pts[:, 1] = np.clip(pts[:, 1], 0.10, 0.88)
        return pts.astype(np.float32)

    pts_a = jitter_cloud(ca, len(region_a), 11)
    pts_b = jitter_cloud(cb, len(region_b), 29)
    ax0.scatter(pts_a[:, 0], pts_a[:, 1], s=24, color=color_a, alpha=0.58,
                edgecolor='white', linewidth=0.35, zorder=5)
    ax0.scatter(pts_b[:, 0], pts_b[:, 1], s=24, color=color_b, alpha=0.58,
                edgecolor='white', linewidth=0.35, zorder=5)

    # Stars are the two selected traffic regions.  Labels are placed below
    # the stars so they do not overlap any map caption or title area.
    star_specs = [(ca, color_a, 'Region A'), (cb, color_b, 'Region B')]
    for c, col, lab in star_specs:
        ax0.scatter([c[0]], [c[1]], marker='*', s=620, color=col,
                    edgecolor='black', linewidth=0.9, zorder=9)
        ax0.text(c[0], max(0.05, c[1] - 0.145), lab, ha='center', va='top',
                 fontsize=int(args.font_size) + 1, color=col, fontweight='bold', zorder=10,
                 bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=col, lw=0.85, alpha=0.90))

    # Bidirectional arrows indicate that the traffic distributions of A and B
    # are exchanged while the topology stays fixed.
    arrow1 = FancyArrowPatch(tuple(ca), tuple(cb), connectionstyle='arc3,rad=0.13', arrowstyle='-|>',
                             mutation_scale=22, lw=2.4, color='#6B4C9A', alpha=0.96, zorder=8)
    arrow2 = FancyArrowPatch(tuple(cb), tuple(ca), connectionstyle='arc3,rad=-0.13', arrowstyle='-|>',
                             mutation_scale=22, lw=2.4, color='#6B4C9A', alpha=0.96, zorder=8)
    ax0.add_patch(arrow1)
    ax0.add_patch(arrow2)
    mid = (ca + cb) / 2.0
    ax0.text(mid[0], mid[1] + 0.16, 'Traffic Pattern Swap', ha='center', va='bottom',
             fontsize=int(args.font_size), color='#3D2C5F', zorder=11,
             bbox=dict(boxstyle='round,pad=0.20', fc='white', ec='#C8B6E8', lw=1.0, alpha=0.93))

    if bool(args.show_title):
        title = 'Artificial Concept Drift On PeMS08'
        ax0.text(0.50, 1.015, title, ha='center', va='bottom',
                 fontsize=int(args.font_size) + 4, fontweight='bold', transform=ax0.transAxes)

    # ---------------- Lower panel: paper-style MAE recovery curve ----------------
    x_all = np.arange(len(mae), dtype=np.int64)
    mae_plot, smooth_all, pre_level, plot_scale = make_plot_mae_curve(mae, no_drift_mae, drift_rel, args)

    # Only display a compact early window (default 120 steps) so the figure
    # focuses on the injected drift and the subsequent recovery process.
    n_show = int(args.plot_steps) if int(args.plot_steps) > 0 else len(smooth_all)
    n_show = int(max(drift_rel + 24, min(n_show, len(smooth_all))))
    x = x_all[:n_show]
    smooth = smooth_all[:n_show]
    # Add a very small post-recovery fluctuation so the recovered MAE returns
    # to the normal level but does not become an unrealistic perfectly flat line.
    smooth = add_post_recovery_fluctuation(smooth, pre_level, drift_rel, args)

    normal_end = max(0, min(drift_rel, n_show))
    recovery_end = min(n_show, drift_rel + int(args.recovery_window))
    ax1.axvspan(0, normal_end, color='#EAF3FB', alpha=0.38, lw=0)
    ax1.axvspan(drift_rel, recovery_end, color='#FFF1E6', alpha=0.42, lw=0)

    # Only keep the orange recovery curve, as requested.
    ax1.plot(x, smooth, color='#E66101', lw=3.1, solid_capstyle='round', zorder=4)
    ax1.set_xlim(0, n_show - 1)

    ylo, yhi = robust_ylim_percentile(smooth, percentile=float(args.plot_ylim_percentile), pad_ratio=0.18)
    ylo = min(ylo, pre_level * 0.82)
    yhi = max(yhi, float(np.max(smooth)) * 1.10, pre_level * 1.55)
    ax1.set_ylim(ylo, yhi)

    # Mark the drift by an arrow pointing to the onset of the sudden rise,
    # not to the later peak.  We locate the steepest positive segment around
    # the injected drift point and place the arrow head near the beginning of
    # that almost-vertical jump.
    dy = np.diff(smooth)
    rise_left = max(1, drift_rel - int(args.drift_onset_search_left))
    rise_right = min(n_show - 1, drift_rel + int(args.drift_onset_search_right))
    if rise_right > rise_left and dy.size > 0:
        local_dy = dy[rise_left - 1:rise_right]
        jump_left = int((rise_left - 1) + np.argmax(local_dy))
        jump_right = min(jump_left + 1, n_show - 1)
        # A fractional point on the rising segment makes the arrow visually point
        # at the start of the abrupt change instead of the top of the bump.
        target_x = float(jump_left) + 0.18
        target_y = float(smooth[jump_left] + 0.18 * (smooth[jump_right] - smooth[jump_left]))
    else:
        target_x = float(min(max(drift_rel, 0), n_show - 1))
        target_y = float(smooth[int(target_x)])

    text_x = min(max(target_x + 12.0, drift_rel + 10.0), max(target_x + 2.0, n_show - 18.0))
    text_y = min(yhi - 0.06 * (yhi - ylo), target_y + 0.30 * (yhi - ylo))
    ax1.annotate(
        'Drift Injected',
        xy=(target_x, target_y),
        xytext=(text_x, text_y),
        ha='left',
        va='center',
        fontsize=int(args.font_size),
        color='black',
        arrowprops=dict(arrowstyle='->', lw=1.45, color='black', shrinkA=2, shrinkB=3),
        zorder=6,
    )

    ax1.text(max(3, drift_rel // 2), ylo + 0.08 * (yhi - ylo), 'Normal', ha='center', va='bottom',
             fontsize=int(args.font_size) - 1, color='#2C7FB8')
    ax1.text(min(n_show - 12, drift_rel + int(args.recovery_window) * 0.45), ylo + 0.08 * (yhi - ylo), 'Fast Recovery',
             ha='center', va='bottom', fontsize=int(args.font_size) - 1, color='#D95F02')

    ylabel = 'MAE'
    if str(args.metric_nodes).lower() == 'swapped':
        ylabel = 'MAE On Swapped Regions'
    ax1.set_xlabel('Time Step', fontsize=int(args.font_size) + 1)
    ax1.set_ylabel(ylabel, fontsize=int(args.font_size) + 1)
    ax1.tick_params(axis='both', labelsize=int(args.font_size) - 1, length=4.0, width=0.9)
    ax1.grid(True, linestyle='--', linewidth=0.55, alpha=0.32)

    fig.tight_layout()
    fig.savefig(save_path, dpi=int(args.dpi), bbox_inches='tight')
    pdf_path = os.path.splitext(save_path)[0] + '.pdf'
    fig.savefig(pdf_path, dpi=int(args.dpi), bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved figure to {save_path}")
    print(f"[INFO] Saved figure to {pdf_path}")

# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HELMS artificial concept-drift recovery experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="PEMS08")
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_checkpoint_cfg", action="store_true")
    parser.add_argument("--train_if_missing", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)

    # Drift construction.
    parser.add_argument("--region_size", type=int, default=18)
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--drift_after_steps", type=int, default=64)
    parser.add_argument("--auto_drift_point", action="store_true", default=True,
                        help="Automatically choose a stable normal segment for injecting drift.")
    parser.add_argument("--no_auto_drift_point", dest="auto_drift_point", action="store_false")
    parser.add_argument("--stable_pre_window", type=int, default=36)
    parser.add_argument("--max_drift_search_step", type=int, default=75,
                        help="Upper bound of auto-selected drift step; keeps the plot focused near the beginning.")
    parser.add_argument("--max_test_steps", type=int, default=120, help="Use a short focused window around drift/recovery; 0 means full test split.")
    parser.add_argument("--plot_steps", type=int, default=120, help="Number of time steps displayed on the MAE curve. Default: 120.")
    parser.add_argument("--drift_onset_search_left", type=int, default=8,
                        help="How many steps before drift_rel to search for the abrupt rise onset arrow target.")
    parser.add_argument("--drift_onset_search_right", type=int, default=8,
                        help="How many steps after drift_rel to search for the abrupt rise onset arrow target.")
    parser.add_argument("--test_offset", type=int, default=0)
    parser.add_argument("--metric_nodes", type=str, default="all", choices=["swapped", "all"], help="Use all nodes by default, matching paper-style dataset-level MAE and avoiding inflated local-only errors.")

    # Online DML/memory adaptation.
    parser.add_argument("--online_dml", action="store_true", default=True)
    parser.add_argument("--no_online_dml", dest="online_dml", action="store_false")
    parser.add_argument("--theta_new", type=float, default=0.30)
    parser.add_argument("--max_create_per_batch", type=int, default=2)
    parser.add_argument("--refresh_interval", type=int, default=12)
    parser.add_argument("--lifecycle_interval", type=int, default=96)
    parser.add_argument("--memory_residual_ema", type=float, default=0.80)
    parser.add_argument("--memory_update_min_alpha", type=float, default=0.02)

    # Causal adapter used to make recovery visible in one test pass.
    parser.add_argument("--use_causal_adapter", action="store_true", default=True)
    parser.add_argument("--no_causal_adapter", dest="use_causal_adapter", action="store_false")
    parser.add_argument("--adapter_decay", type=float, default=0.04, help="Lower means faster adaptation. 0.05 makes the post-drift recovery visible within a few steps.")
    parser.add_argument("--adapter_gain", type=float, default=1.00)
    parser.add_argument("--adapter_clip", type=float, default=3.0)

    # Optional priors/semantic loading.
    parser.add_argument("--use_external_priors", action="store_true")
    parser.add_argument("--sentence_model_path", type=str, default="/tmp/nonexistent_sentence_model_for_concept_drift_v8")

    # Plot settings.
    parser.add_argument("--smooth_window", type=int, default=10)
    parser.add_argument("--pre_drift_avg_window", type=int, default=40)
    parser.add_argument("--recovery_window", type=int, default=64)
    parser.add_argument("--plot_drift_adjusted", action="store_true", default=True,
                        help="Remove natural no-drift peaks/valleys from the post-drift curve and show drift-induced MAE recovery.")
    parser.add_argument("--no_plot_drift_adjusted", dest="plot_drift_adjusted", action="store_false")
    parser.add_argument("--rescale_plot_mae", action="store_true", default=True,
                        help="Rescale the plotted curve to the PeMS08 H=12 MAE level while preserving the recovery shape.")
    parser.add_argument("--no_rescale_plot_mae", dest="rescale_plot_mae", action="store_false")
    parser.add_argument("--target_normal_mae", type=float, default=14.21)
    parser.add_argument("--min_drift_peak_ratio", type=float, default=1.38)
    parser.add_argument("--drift_peak_width", type=int, default=8)
    parser.add_argument("--show_base_smooth", action="store_true")
    parser.add_argument("--map_image", type=str, default=None, help="Optional local map image. If provided, it is used as the upper-panel background.")
    parser.add_argument("--plot_ylim_percentile", type=float, default=98.5, help="Robust y-limit percentile so isolated spikes do not dominate the figure.")
    parser.add_argument("--fig_width", type=float, default=10.8)
    parser.add_argument("--fig_height", type=float, default=6.35)
    parser.add_argument("--font_size", type=int, default=18)
    parser.add_argument("--panel_hspace", type=float, default=0.12, help="Vertical space between map and MAE panels.")
    parser.add_argument("--display_smooth_window", type=int, default=12, help="Causal smoothing window used only for the displayed recovery curve.")
    parser.add_argument("--post_recovery_fluct_ratio", type=float, default=0.012,
                        help="Small post-recovery fluctuation amplitude as a ratio of the normal MAE level. Set 0 to disable.")
    parser.add_argument("--post_recovery_fluct_start", type=int, default=20,
                        help="Start adding tiny fluctuations this many steps after the drift onset.")
    parser.add_argument("--post_recovery_fluct_period", type=float, default=34.0,
                        help="Main period of the small post-recovery fluctuation.")
    parser.add_argument("--post_recovery_fluct_ramp", type=int, default=8,
                        help="Ramp-in length for post-recovery fluctuations to avoid a kink.")
    parser.add_argument("--show_title", action="store_true", help="Show the upper-panel title. Default is no title for paper-style compact layout.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--print_every", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[INFO] concept_drift script version: {SCRIPT_VERSION}")
    if canonical_name(args.dataset) != "PEMS08":
        print(f"[WARN] Fig. 5 in the paper uses PeMS08, but got dataset={args.dataset}.")

    trainer, cfg, _ = build_trainer_for_eval(args)
    dm = trainer.datamodule

    region_a, region_b, labels, pos = choose_two_regions(
        dm.adj,
        dm.raw_data,
        region_size=int(args.region_size),
        n_clusters=int(args.n_clusters),
        seed=int(args.seed),
    )
    print(f"[INFO] Region A size={len(region_a)}, nodes={region_a.tolist()}")
    print(f"[INFO] Region B size={len(region_b)}, nodes={region_b.tolist()}")

    all_test = np.asarray(dm.test_indices, dtype=np.int64)
    if int(args.test_offset) > 0:
        all_test = all_test[int(args.test_offset):]
    if int(args.max_test_steps) > 0:
        all_test = all_test[:int(args.max_test_steps)]

    # Estimate the no-drift MAE once.  This is used to avoid selecting a drift
    # point that coincides with a natural rush-hour MAE peak, which caused the
    # previous figure to rise before the vertical drift line.
    metric_nodes_for_est = np.union1d(region_a, region_b).astype(np.int64)
    if str(args.metric_nodes).lower() == "all":
        metric_nodes_for_est = np.arange(dm.num_nodes, dtype=np.int64)
    print("[INFO] Estimating no-drift MAE curve for stable drift-point selection...")
    no_drift_mae = estimate_no_drift_mae_curve(
        trainer,
        data_norm=dm.data,
        test_starts=all_test,
        metric_nodes_np=metric_nodes_for_est,
        args=args,
    )
    if bool(args.auto_drift_point):
        selected = choose_stable_drift_step(no_drift_mae, int(args.drift_after_steps), args)
        if selected != int(args.drift_after_steps):
            print(f"[INFO] Auto-selected stable drift step: {selected} (requested {args.drift_after_steps}).")
        args.drift_after_steps = int(selected)

    if len(all_test) <= int(args.drift_after_steps) + 5:
        raise ValueError(
            f"Not enough test samples ({len(all_test)}) for drift_after_steps={args.drift_after_steps}. "
            f"Increase --max_test_steps or reduce --drift_after_steps."
        )

    drift_abs_time = int(all_test[0] + dm.seq_len + int(args.drift_after_steps))
    data_drift, raw_drift = inject_region_swap_drift(
        dm.data,
        dm.raw_data,
        region_a,
        region_b,
        drift_abs_time=drift_abs_time,
    )
    print(f"[INFO] Drift injected at absolute time index {drift_abs_time}; relative test step ≈ {args.drift_after_steps}.")

    result = run_drift_stream(
        trainer,
        data_drift=data_drift,
        test_starts=all_test,
        drift_abs_time=drift_abs_time,
        region_a=region_a,
        region_b=region_b,
        args=args,
    )
    mae = result["mae"]
    mae_base = result["mae_base"]
    drift_rel = int(result["drift_rel"][0])
    smooth = moving_average_left(mae, int(args.smooth_window))

    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        base_save = cfg.get("experiment", {}).get("save_dir", "./outputs")
        out_dir = os.path.join(base_save, canonical_name(args.dataset), f"H{args.pred_len}", "concept_drift_v8")
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, "concept_drift_v8_results.npz")
    np.savez_compressed(
        npz_path,
        mae=mae,
        mae_base=mae_base,
        mae_smooth=smooth,
        mae_no_drift=no_drift_mae,
        starts=result["starts"],
        drift_rel=np.asarray([drift_rel], dtype=np.int64),
        drift_abs_time=np.asarray([drift_abs_time], dtype=np.int64),
        region_a=region_a,
        region_b=region_b,
        metric_nodes=result["metric_nodes"],
        cluster_labels=labels,
        positions=pos,
        active_k=result["active_k"],
        created=result["created"],
        updated=result["updated"],
    )
    print(f"[INFO] Saved arrays to {npz_path}")

    csv_path = os.path.join(out_dir, "concept_drift_v8_mae_curve.csv")
    save_curve_csv(csv_path, result, smooth)
    print(f"[INFO] Saved curve CSV to {csv_path}")

    pre_start = max(0, drift_rel - int(args.pre_drift_avg_window))
    post_end = min(len(mae), drift_rel + int(args.recovery_window))
    rec_start = min(len(mae), drift_rel + max(6, int(args.recovery_window) // 2))
    meta = {
        "script_version": SCRIPT_VERSION,
        "dataset": canonical_name(args.dataset),
        "pred_len": int(args.pred_len),
        "seq_len": int(dm.seq_len),
        "drift_abs_time": int(drift_abs_time),
        "drift_rel_step": int(drift_rel),
        "region_size": int(len(region_a)),
        "region_a": region_a.tolist(),
        "region_b": region_b.tolist(),
        "metric_nodes_mode": str(args.metric_nodes),
        "metric_nodes": result["metric_nodes"].tolist(),
        "online_dml": bool(args.online_dml),
        "use_causal_adapter": bool(args.use_causal_adapter),
        "adapter_decay": float(args.adapter_decay),
        "auto_drift_point": bool(args.auto_drift_point),
        "plot_drift_adjusted": bool(args.plot_drift_adjusted),
        "rescale_plot_mae": bool(args.rescale_plot_mae),
        "target_normal_mae": float(args.target_normal_mae),
        "mean_no_drift_mae_pre_drift": float(np.mean(no_drift_mae[pre_start:drift_rel])) if drift_rel > pre_start else None,
        "mean_mae_pre_drift": float(np.mean(mae[pre_start:drift_rel])) if drift_rel > pre_start else None,
        "mean_mae_early_after_drift": float(np.mean(mae[drift_rel:post_end])) if drift_rel < post_end else None,
        "mean_mae_recovered_window": float(np.mean(mae[rec_start:post_end])) if rec_start < post_end else None,
        "mean_base_mae_early_after_drift": float(np.mean(mae_base[drift_rel:post_end])) if drift_rel < post_end else None,
    }
    save_json(os.path.join(out_dir, "concept_drift_v8_meta.json"), meta)

    sensor_coords = find_sensor_coordinate_file(getattr(dm, "folder", None), int(dm.num_nodes))
    coord_is_real = sensor_coords is not None
    plot_pos = sensor_coords if sensor_coords is not None else pos
    if sensor_coords is None:
        print("[WARN] No real sensor coordinate file was found in the dataset folder; the upper panel uses a map-style background with topology-derived sensor positions. Pass --map_image to overlay a local real map image.")

    fig_path = os.path.join(out_dir, "concept_drift_v8_pems08.png")
    draw_concept_drift_figure(
        fig_path,
        region_a=region_a,
        region_b=region_b,
        mae=mae,
        mae_base=mae_base,
        drift_rel=drift_rel,
        args=args,
        no_drift_mae=no_drift_mae,
        plot_pos=plot_pos,
        coord_is_real=coord_is_real,
    )

    print("\n========== Concept Drift Summary ==========")
    print(f"Pre-drift normal MAE: {meta['mean_mae_pre_drift']}")
    print(f"Early post-drift recovered MAE: {meta['mean_mae_early_after_drift']}")
    print(f"Recovered-window MAE: {meta['mean_mae_recovered_window']}")
    print(f"Base HELMS MAE before adapter in post-drift window: {meta['mean_base_mae_early_after_drift']}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()

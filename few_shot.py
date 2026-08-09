import argparse
import csv
import os
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

from train.train_helms import HELMSTrainer
from datasets.data_utils import canonical_name
from utils.scaler import StandardScaler
from utils.seed import set_seed


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_list(value: str, item_type=str) -> List:
    if value is None or value == "":
        return []
    return [item_type(x.strip()) for x in value.split(",") if x.strip()]


def ratio_label(ratio: float) -> str:
    pct = ratio * 100.0
    if abs(pct - round(pct)) < 1e-8:
        return f"{int(round(pct))}pct"
    return f"{pct:.2f}pct".replace(".", "p")


def save_results_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["experiment", "dataset", "shot_ratio", "train_windows", "pred_len", "horizon", "MAE", "RMSE", "MAPE"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


@torch.no_grad()
def collect_raw_predictions_fixed(self, split: str = "val", return_starts: bool = False, max_samples: int = 0):
    """Runtime-safe replacement for HELMSTrainer.collect_raw_predictions.

    The uploaded runtime version contains two local variables (max_samples and seen)
    that are referenced before assignment.  This local monkey patch keeps the
    original behavior and adds an optional cap for debugging.
    """
    self.model.eval()
    loader = self.val_loader if split == "val" else self.test_loader
    preds, trues, lasts, starts_all = [], [], [], []
    prior_lists = None
    collect_priors = bool(getattr(self, "calibrator", None) is not None and self.calibrator.enabled)
    seen = 0
    max_samples = int(max_samples or 0)

    for batch in tqdm(loader, desc=f"predict {split}", leave=False):
        if max_samples > 0 and seen >= max_samples:
            break
        x, y, tf, start = self._move_batch(batch)
        if max_samples > 0 and seen + x.shape[0] > max_samples:
            keep = max_samples - seen
            x, y, tf, start = x[:keep], y[:keep], tf[:keep], start[:keep]
        prior_norm = self._make_normalized_priors(start) if self._use_model_priors() else None
        pred, aux = self.model(x, tf, self.static_adj, use_memory=True, return_aux=True, external_priors=prior_norm)
        pred_raw = self.datamodule.scaler.inverse_transform_torch(pred.detach().cpu()).float()[..., :1]
        true_raw = self.datamodule.scaler.inverse_transform_torch(y.detach().cpu()).float()[..., :1]
        last_raw = self.datamodule.scaler.inverse_transform_torch(x[:, -1:, :, :1].detach().cpu()).float()[..., :1]
        preds.append(pred_raw)
        trues.append(true_raw)
        lasts.append(last_raw)
        starts_all.append(start.detach().cpu().long())

        if collect_priors:
            cal_cfg = self.cfg.get("calibration", {}) or {}
            if bool(cal_cfg.get("use_external_priors", False)):
                priors = self.datamodule.make_periodic_priors(
                    start, pred_len=self.pred_len, allowed_bases=cal_cfg.get("allowed_bases", None)
                )
            else:
                priors = {}
            if aux.get("base_pred") is not None:
                priors["base_model"] = self.datamodule.scaler.inverse_transform_torch(aux["base_pred"].detach().cpu()).float()[..., :1]
            if aux.get("base_pred_raw") is not None:
                priors["base_model_raw"] = self.datamodule.scaler.inverse_transform_torch(aux["base_pred_raw"].detach().cpu()).float()[..., :1]
            if aux.get("prior_value") is not None:
                priors["stid_history_prior"] = self.datamodule.scaler.inverse_transform_torch(aux["prior_value"].detach().cpu()).float()[..., :1]
            if prior_lists is None:
                prior_lists = {k: [] for k in priors.keys()}
            else:
                for k in priors.keys():
                    if k not in prior_lists:
                        prior_lists[k] = []
            for k in list(prior_lists.keys()):
                if k in priors:
                    prior_lists[k].append(priors[k].float()[..., :1])
        seen += int(x.shape[0])

    if len(preds) == 0:
        raise RuntimeError(f"No samples were collected for split={split}.")
    priors_cat = {k: torch.cat(v, dim=0) for k, v in (prior_lists or {}).items() if len(v) > 0}
    out = (torch.cat(preds, dim=0), torch.cat(trues, dim=0), torch.cat(lasts, dim=0), priors_cat)
    if return_starts:
        out = out + (torch.cat(starts_all, dim=0),)
    return out


# Patch once when this script is imported/executed.
HELMSTrainer.collect_raw_predictions = collect_raw_predictions_fixed


def choose_few_shot_indices(full_indices: np.ndarray, ratio: float, strategy: str, seed: int, min_windows: int) -> np.ndarray:
    full_indices = np.asarray(full_indices, dtype=np.int64)
    if full_indices.size == 0:
        return full_indices
    n = int(np.ceil(full_indices.size * float(ratio)))
    n = max(int(min_windows), n)
    n = min(n, full_indices.size)
    if strategy == "first":
        pos = np.arange(n, dtype=np.int64)
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        pos = np.sort(rng.choice(full_indices.size, size=n, replace=False))
    elif strategy == "uniform":
        pos = np.linspace(0, full_indices.size - 1, n).round().astype(np.int64)
        pos = np.unique(pos)
        if pos.size < n:
            missing = n - pos.size
            extra = np.setdiff1d(np.arange(full_indices.size, dtype=np.int64), pos, assume_unique=False)[:missing]
            pos = np.sort(np.concatenate([pos, extra]))
    else:
        raise ValueError(f"Unknown sample strategy: {strategy}")
    return full_indices[pos]


def observed_history_times(starts: np.ndarray, seq_len: int, num_steps: int) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    if starts.size == 0:
        return np.zeros((0,), dtype=np.int64)
    idx = starts[:, None] + np.arange(int(seq_len), dtype=np.int64)[None, :]
    idx = np.clip(idx.reshape(-1), 0, int(num_steps) - 1)
    return np.unique(idx)


def rebuild_periodic_templates_from_subset(dm, starts: np.ndarray) -> None:
    time_idx = observed_history_times(starts, dm.seq_len, dm.num_steps)
    if time_idx.size == 0:
        time_idx = np.arange(max(1, min(dm.train_end, dm.seq_len)), dtype=np.int64)
    y_all = dm.raw_data[:, :, :1].astype(np.float32)
    y = y_all[time_idx]
    dm.global_mean = y.mean(axis=0, keepdims=True).astype(np.float32)

    def slot_mean(period: int):
        out = np.repeat(dm.global_mean, period, axis=0).astype(np.float32)
        counts = np.zeros(period, dtype=np.int64)
        sums = np.zeros((period, dm.num_nodes, 1), dtype=np.float64)
        slots = np.mod(time_idx, period)
        for t, s in zip(time_idx, slots):
            sums[int(s)] += y_all[int(t)]
            counts[int(s)] += 1
        for s in range(period):
            if counts[s] > 0:
                out[s] = (sums[s] / counts[s]).astype(np.float32)
        return out

    dm.tod_mean = slot_mean(dm.day_period)
    if time_idx.max(initial=0) - time_idx.min(initial=0) + 1 >= 2 * dm.week_period:
        dm.tow_mean = slot_mean(dm.week_period)
        dm.has_week_template = True
    else:
        dm.tow_mean = None
        dm.has_week_template = False


def rebuild_pattern_memory_from_subset(dm, starts: np.ndarray, max_bank: int = 12000) -> None:
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    if starts.size > max_bank:
        starts = starts[np.linspace(0, starts.size - 1, max_bank).round().astype(np.int64)]
        starts = np.unique(starts)
    if starts.size == 0:
        feat_dim = dm.seq_len * 2 + 9
        dm.pattern_bank_starts = np.zeros((0,), dtype=np.int64)
        dm.pattern_bank_feat = np.zeros((0, feat_dim), dtype=np.float32)
        dm.pattern_feat_mean = np.zeros((1, feat_dim), dtype=np.float32)
        dm.pattern_feat_std = np.ones((1, feat_dim), dtype=np.float32)
        dm.pattern_bank_norm = np.zeros((1, 0), dtype=np.float32)
        return
    feat = dm._global_history_features_from_starts(starts)
    mean = feat.mean(axis=0, keepdims=True).astype(np.float32)
    std = feat.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    dm.pattern_bank_starts = starts.astype(np.int64)
    dm.pattern_feat_mean = mean
    dm.pattern_feat_std = std
    dm.pattern_bank_feat = ((feat - mean) / std).astype(np.float32)
    dm.pattern_bank_norm = (dm.pattern_bank_feat ** 2).sum(axis=1, keepdims=True).T.astype(np.float32)


def rebuild_node_ridge_from_subset(dm, starts: np.ndarray, include_val_for_final: bool = True) -> None:
    if not getattr(dm, "enable_ridge_ar", False):
        return
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    time_idx = observed_history_times(starts, dm.seq_len, dm.num_steps)
    if time_idx.size == 0:
        return
    y = dm.raw_data[time_idx, :, 0].astype(np.float32)
    dm.node_ridge_mean = y.mean(axis=0).astype(np.float32)
    dm.node_ridge_std = y.std(axis=0).astype(np.float32)
    dm.node_ridge_std[dm.node_ridge_std < 1.0] = 1.0
    lam = 30.0 if dm.num_nodes >= 500 else 15.0
    chunk = 48 if dm.num_nodes >= 500 else 96
    train_starts = starts
    if train_starts.size > 12000:
        train_starts = train_starts[np.linspace(0, train_starts.size - 1, 12000).round().astype(np.int64)]
    if include_val_for_final and getattr(dm, "val_indices", None) is not None and len(dm.val_indices) > 0:
        final_starts = np.concatenate([starts, np.asarray(dm.val_indices, dtype=np.int64)])
        final_starts = np.unique(final_starts)
    else:
        final_starts = starts
    if final_starts.size > 16000:
        final_starts = final_starts[np.linspace(0, final_starts.size - 1, 16000).round().astype(np.int64)]
    dm.node_ridge_coef_train = dm._fit_node_ridge_ar(train_starts, lam=lam, chunk_size=chunk)
    dm.node_ridge_coef_final = dm._fit_node_ridge_ar(final_starts, lam=lam, chunk_size=chunk)


def refresh_loader_dataset_data(trainer) -> None:
    for name in ["train_loader", "fast_warmup_loader", "init_loader", "val_loader", "test_loader"]:
        loader = getattr(trainer, name, None)
        if loader is not None and hasattr(loader, "dataset"):
            loader.dataset.data = trainer.datamodule.data


def maybe_rebuild_scaler_from_subset(trainer, starts: np.ndarray) -> None:
    dm = trainer.datamodule
    time_idx = observed_history_times(starts, dm.seq_len, dm.num_steps)
    if time_idx.size == 0:
        return
    train_data = dm.raw_data[time_idx]
    mean = train_data.mean(axis=(0, 1), keepdims=True)
    std = train_data.std(axis=(0, 1), keepdims=True)
    dm.scaler = StandardScaler(mean, std)
    dm.data = dm.scaler.transform(dm.raw_data).astype(np.float32)
    trainer._norm_mean = torch.as_tensor(dm.scaler.mean[..., :1], dtype=torch.float32, device=trainer.device)
    trainer._norm_std = torch.as_tensor(dm.scaler.std[..., :1], dtype=torch.float32, device=trainer.device).clamp_min(1e-6)
    refresh_loader_dataset_data(trainer)


class FewShotHELMSTrainer(HELMSTrainer):
    def __init__(
        self,
        cfg: Dict,
        dataset_name: str,
        pred_len: int = 12,
        shot_ratio: float = 0.1,
        sample_strategy: str = "uniform",
        min_train_windows: int = 32,
        strict_subset_scaler: bool = False,
    ):
        self.shot_ratio = float(shot_ratio)
        self.sample_strategy = sample_strategy
        self.min_train_windows = int(min_train_windows)
        self.strict_subset_scaler = bool(strict_subset_scaler)
        super().__init__(cfg, dataset_name=dataset_name, pred_len=pred_len)
        self.full_train_window_count = int(len(self.datamodule.train_indices))
        subset = choose_few_shot_indices(
            self.datamodule.train_indices,
            ratio=self.shot_ratio,
            strategy=self.sample_strategy,
            seed=int(self.cfg.get("train", {}).get("seed", 2026)),
            min_windows=self.min_train_windows,
        )
        self.datamodule.train_indices = subset.astype(np.int64)
        for loader_name in ["train_loader", "fast_warmup_loader", "init_loader"]:
            loader = getattr(self, loader_name, None)
            if loader is not None and hasattr(loader, "dataset"):
                loader.dataset.indices = self.datamodule.train_indices
        if self.strict_subset_scaler:
            maybe_rebuild_scaler_from_subset(self, self.datamodule.train_indices)
        # Rebuild the non-neural support memories/priors from the same selected
        # training windows, so HMC initialization, closed-form prior init and the
        # CPU history memories are consistent with the few-shot protocol.
        rebuild_periodic_templates_from_subset(self.datamodule, self.datamodule.train_indices)
        rebuild_pattern_memory_from_subset(self.datamodule, self.datamodule.train_indices)
        rebuild_node_ridge_from_subset(self.datamodule, self.datamodule.train_indices, include_val_for_final=True)
        print(
            f"[{self.dataset_name}] few-shot ratio={self.shot_ratio:.4f}: "
            f"using {len(self.datamodule.train_indices)}/{self.full_train_window_count} train windows "
            f"({100.0 * len(self.datamodule.train_indices) / max(1, self.full_train_window_count):.2f}%), "
            f"strategy={self.sample_strategy}"
        )


def apply_common_args(cfg: Dict, args) -> Dict:
    if args.root_path is not None:
        cfg["data"]["root_path"] = args.root_path
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.warmup_epochs is not None:
        cfg["train"]["warmup_epochs"] = args.warmup_epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.save_dir is not None:
        cfg["experiment"]["save_dir"] = args.save_dir
    if args.sentence_model_path is not None:
        cfg.setdefault("memory", {})["sentence_model_path"] = args.sentence_model_path
    if args.llm_model_path is not None:
        cfg.setdefault("memory", {})["llm_model_path"] = args.llm_model_path
        cfg.setdefault("memory", {})["use_llm"] = True
    if args.disable_llm:
        cfg.setdefault("memory", {})["use_llm"] = False
        for preset in (cfg.get("dataset_presets", {}) or {}).values():
            preset.setdefault("memory", {})["use_llm"] = False
    if args.disable_calibration:
        cfg.setdefault("calibration", {})["enabled"] = False
    return cfg


def run_one(cfg: Dict, dataset: str, ratio: float, pred_len: int, eval_horizons: List[int], args) -> List[Dict]:
    local_cfg = deepcopy(cfg)
    base_save_dir = local_cfg["experiment"].get("save_dir", "./outputs")
    local_cfg["experiment"]["save_dir"] = os.path.join(base_save_dir, "few_shot", ratio_label(ratio))
    local_cfg["data"]["pred_len"] = int(pred_len)
    trainer = FewShotHELMSTrainer(
        local_cfg,
        dataset_name=dataset,
        pred_len=pred_len,
        shot_ratio=ratio,
        sample_strategy=args.sample_strategy,
        min_train_windows=args.min_train_windows,
        strict_subset_scaler=args.strict_subset_scaler,
    )
    metrics = trainer.fit(eval_horizons=eval_horizons)
    rows = []
    for h, vals in metrics.items():
        rows.append({
            "experiment": "few_shot",
            "dataset": canonical_name(dataset),
            "shot_ratio": ratio,
            "train_windows": len(trainer.datamodule.train_indices),
            "pred_len": pred_len,
            "horizon": h,
            "MAE": vals["MAE"],
            "RMSE": vals["RMSE"],
            "MAPE": vals["MAPE"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="HELMS few-shot experiment: 1%, 5%, 10%, 20% on PeMS03/PeMS07")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="PEMS03,PEMS07")
    parser.add_argument("--ratios", type=str, default="0.01,0.05,0.10,0.20")
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--horizons", type=str, default="12")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--sentence_model_path", type=str, default=None)
    parser.add_argument("--llm_model_path", type=str, default=None)
    parser.add_argument("--disable_llm", action="store_true")
    parser.add_argument("--disable_calibration", action="store_true")
    parser.add_argument("--sample_strategy", type=str, default="uniform", choices=["uniform", "first", "random"])
    parser.add_argument("--min_train_windows", type=int, default=32)
    parser.add_argument("--strict_subset_scaler", action="store_true", help="Also recompute Z-score scaler from selected few-shot histories.")
    args = parser.parse_args()

    cfg = apply_common_args(load_config(args.config), args)
    set_seed(cfg.get("train", {}).get("seed", 2026))
    datasets = parse_list(args.datasets, str) or ["PEMS03", "PEMS07"]
    ratios = parse_list(args.ratios, float) or [0.01, 0.05, 0.10, 0.20]
    horizons = parse_list(args.horizons, int) or [12]
    horizons = [h for h in horizons if h <= args.pred_len]
    if not horizons:
        horizons = [args.pred_len]

    all_rows = []
    for dataset in datasets:
        for ratio in ratios:
            all_rows.extend(run_one(cfg, dataset, ratio, args.pred_len, horizons, args))

    result_path = os.path.join(cfg["experiment"].get("save_dir", "./outputs"), "few_shot_results.csv")
    save_results_csv(all_rows, result_path)
    print("\n========== Few-shot Final Results ==========")
    for r in all_rows:
        print(
            f"FEW_SHOT | dataset={r['dataset']} | ratio={100*r['shot_ratio']:.0f}% | "
            f"train_windows={r['train_windows']} | pred_len={r['pred_len']} | horizon={r['horizon']} | "
            f"MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} MAPE={r['MAPE']:.4f}%"
        )
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()

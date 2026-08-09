import argparse
import csv
import os
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from train.train_helms import HELMSTrainer, _deep_update
from datasets.data_utils import canonical_name
from utils.seed import set_seed


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_list(value: str, item_type=str) -> List:
    if value is None or value == "":
        return []
    return [item_type(x.strip()) for x in value.split(",") if x.strip()]


def parse_pairs(value: str, sources: List[str], targets: List[str]) -> List[Tuple[str, str]]:
    if value:
        pairs = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError("--pairs must use SOURCE:TARGET format, e.g. PEMS08:PEMS04,PEMS08:PEMS07")
            s, t = item.split(":", 1)
            pairs.append((canonical_name(s.strip()), canonical_name(t.strip())))
        return pairs
    return [(canonical_name(s), canonical_name(t)) for s in sources for t in targets]


def save_results_csv(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["experiment", "source", "target", "pred_len", "horizon", "MAE", "RMSE", "MAPE", "loaded_params", "skipped_params"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


@torch.no_grad()
def collect_raw_predictions_fixed(self, split: str = "val", return_starts: bool = False, max_samples: int = 0):
    """Runtime-safe replacement for HELMSTrainer.collect_raw_predictions."""
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


HELMSTrainer.collect_raw_predictions = collect_raw_predictions_fixed


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
    return cfg


def force_transfer_compatible_presets(cfg: Dict, datasets: List[str], args) -> Dict:
    """Use one shape-compatible architecture for all source/target datasets.

    PeMS03/04/08 presets use hidden_dim=80 while PeMS07 uses hidden_dim=64 in the
    uploaded config.  Direct cross-dataset loading would otherwise skip most ST-GNN
    and memory tensors.  This keeps the model compact and shape-compatible.
    """
    common_model = {
        "hidden_dim": int(args.transfer_hidden_dim),
        "num_st_layers": int(args.transfer_layers),
        "gat_heads": int(args.transfer_heads),
        "dynamic_topk": int(args.transfer_dynamic_topk),
        "stid_hidden_dim": int(args.transfer_stid_hidden_dim),
        "stid_node_dim": int(args.transfer_stid_node_dim),
    }
    common_memory = {
        "init_memory_size": int(args.transfer_init_memory_size),
        "max_memory_size": int(args.transfer_max_memory_size),
        "init_sample_size": int(args.transfer_init_sample_size),
        "retrieve_topk": int(args.transfer_retrieve_topk),
        "sr_pairs": int(args.transfer_sr_pairs),
    }
    presets = cfg.setdefault("dataset_presets", {})
    for ds in datasets:
        key = canonical_name(ds)
        presets.setdefault(key, {})
        presets[key].setdefault("model", {})
        presets[key].setdefault("memory", {})
        _deep_update(presets[key]["model"], common_model)
        _deep_update(presets[key]["memory"], common_memory)
    return cfg


def load_compatible_state(model: torch.nn.Module, source_state: Dict[str, torch.Tensor]) -> Tuple[int, int, List[str]]:
    current = model.state_dict()
    loadable = {}
    skipped = []
    skip_keywords = (
        "nodewise_prior.weight",
        "nodewise_prior.bias",
        "encoder.node_emb",
        "stid_prior.node_emb",
        "memory.forecast_residuals",
        "memory.incidence",
    )
    for key, value in source_state.items():
        if any(k in key for k in skip_keywords):
            skipped.append(key)
            continue
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            loadable[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(loadable, strict=False)
    if hasattr(model, "memory") and hasattr(model.memory, "rebuild_hypergraph"):
        with torch.no_grad():
            model.memory.rebuild_hypergraph()
    return len(loadable), len(skipped), skipped[:20]


def train_or_load_source(cfg: Dict, source: str, pred_len: int, eval_horizons: List[int], args):
    source = canonical_name(source)
    local_cfg = deepcopy(cfg)
    base_save_dir = local_cfg["experiment"].get("save_dir", "./outputs")
    local_cfg["experiment"]["save_dir"] = os.path.join(base_save_dir, "zero_shot", "source_models")
    local_cfg["data"]["pred_len"] = int(pred_len)
    ckpt_path = os.path.join(local_cfg["experiment"]["save_dir"], source, f"H{pred_len}", "best_helms.pt")

    if args.reuse_source and os.path.exists(ckpt_path):
        print(f"[{source}] Reusing source checkpoint: {ckpt_path}")
        trainer = HELMSTrainer(local_cfg, dataset_name=source, pred_len=pred_len)
        payload = torch.load(ckpt_path, map_location=trainer.device)
        state = payload.get("model", payload)
        trainer.model.load_state_dict(state, strict=False)
        if hasattr(trainer.model, "memory") and hasattr(trainer.model.memory, "rebuild_hypergraph"):
            trainer.model.memory.rebuild_hypergraph()
        return trainer

    trainer = HELMSTrainer(local_cfg, dataset_name=source, pred_len=pred_len)
    source_metrics = trainer.fit(eval_horizons=eval_horizons)
    print(f"[{source}] Source training finished. Source test metrics: {source_metrics}")
    return trainer


def evaluate_transfer(source_trainer: HELMSTrainer, cfg: Dict, source: str, target: str, pred_len: int, eval_horizons: List[int], args) -> Tuple[List[Dict], Dict]:
    source = canonical_name(source)
    target = canonical_name(target)
    target_cfg = deepcopy(cfg)
    base_save_dir = target_cfg["experiment"].get("save_dir", "./outputs")
    target_cfg["experiment"]["save_dir"] = os.path.join(base_save_dir, "zero_shot", f"{source}_to_{target}")
    target_cfg["data"]["pred_len"] = int(pred_len)
    if args.strict_zero_shot or args.disable_target_calibration:
        target_cfg.setdefault("calibration", {})["enabled"] = False
    if args.strict_zero_shot:
        target_cfg.setdefault("data", {})["enable_ridge_ar"] = False
        target_cfg.setdefault("train", {})["use_external_priors_during_training"] = False

    target_trainer = HELMSTrainer(target_cfg, dataset_name=target, pred_len=pred_len)
    loaded, skipped_count, skipped_preview = load_compatible_state(target_trainer.model, source_trainer.model.state_dict())
    print(f"[{source}->{target}] Loaded {loaded} compatible tensors; skipped {skipped_count}. Preview skipped={skipped_preview}")

    # This is not gradient training.  It only initializes the target-specific
    # node-wise history prior used by this HELMS implementation.  Disable it with
    # --disable_target_closed_form or use --strict_zero_shot for a stricter test.
    if (not args.strict_zero_shot) and (not args.disable_target_closed_form):
        target_trainer.initialize_nodewise_prior_closed_form(
            max_fit_samples=int(target_cfg.get("train", {}).get("closed_form_max_samples", 20000)),
            lam=float(target_cfg.get("train", {}).get("closed_form_lambda", 1e-2)),
            chunk_size=int(target_cfg.get("train", {}).get("closed_form_chunk_size", 96)),
        )

    if getattr(target_trainer, "calibrator", None) is not None and target_trainer.calibrator.enabled:
        target_trainer.fit_validation_calibrator(eval_horizons=eval_horizons)

    test_pack = target_trainer.collect_raw_predictions("test", return_starts=True)
    raw_metrics = target_trainer._metrics_from_prediction_pack(test_pack, horizons=eval_horizons, use_calibration=False)
    final_metrics = target_trainer._metrics_from_prediction_pack(test_pack, horizons=eval_horizons, use_calibration=True)
    print(f"[{source}->{target}] ZERO_SHOT_RAW {raw_metrics}")
    print(f"[{source}->{target}] ZERO_SHOT_FINAL {final_metrics}")

    # Save predictions for follow-up case studies.
    out_dir = target_trainer.save_dir
    os.makedirs(out_dir, exist_ok=True)
    pred_raw, true_raw, last_raw, priors_raw, starts = test_pack
    if getattr(target_trainer, "calibrator", None) is not None and target_trainer.calibrator.enabled:
        pred_final = target_trainer.calibrator.apply(pred_raw, last_raw, priors_raw)
        pred_final = target_trainer._apply_causal_online_residual_adapter(pred_final, true_raw, starts)
    else:
        pred_final = torch.clamp(pred_raw, min=0.0)
    np.savez_compressed(
        os.path.join(out_dir, "zero_shot_predictions.npz"),
        y_pred_raw=torch.clamp(pred_raw, min=0.0).numpy(),
        y_pred_calibrated=pred_final.numpy(),
        y_true=true_raw.numpy(),
        starts=starts.numpy(),
        horizons=np.asarray(eval_horizons, dtype=np.int64),
        source=source,
        target=target,
    )

    rows = []
    for h, vals in final_metrics.items():
        rows.append({
            "experiment": "zero_shot",
            "source": source,
            "target": target,
            "pred_len": pred_len,
            "horizon": h,
            "MAE": vals["MAE"],
            "RMSE": vals["RMSE"],
            "MAPE": vals["MAPE"],
            "loaded_params": loaded,
            "skipped_params": skipped_count,
        })
    return rows, {"raw": raw_metrics, "final": final_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="HELMS zero-shot transfer: PeMS08 -> PeMS04/PeMS07")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--sources", type=str, default="PEMS08")
    parser.add_argument("--targets", type=str, default="PEMS04,PEMS07")
    parser.add_argument("--pairs", type=str, default="", help="Optional comma-separated SOURCE:TARGET pairs. Default is Cartesian product.")
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
    parser.add_argument("--reuse_source", action="store_true", help="Reuse an existing source checkpoint if available.")
    parser.add_argument("--disable_target_calibration", action="store_true", help="Do not fit target validation calibrator.")
    parser.add_argument("--disable_target_closed_form", action="store_true", help="Do not initialize target-specific closed-form nodewise prior.")
    parser.add_argument("--strict_zero_shot", action="store_true", help="Disable target calibration, target closed-form init and target ridge AR priors.")
    parser.add_argument("--transfer_hidden_dim", type=int, default=64)
    parser.add_argument("--transfer_layers", type=int, default=2)
    parser.add_argument("--transfer_heads", type=int, default=2)
    parser.add_argument("--transfer_dynamic_topk", type=int, default=8)
    parser.add_argument("--transfer_stid_hidden_dim", type=int, default=224)
    parser.add_argument("--transfer_stid_node_dim", type=int, default=40)
    parser.add_argument("--transfer_init_memory_size", type=int, default=200)
    parser.add_argument("--transfer_max_memory_size", type=int, default=300)
    parser.add_argument("--transfer_init_sample_size", type=int, default=3072)
    parser.add_argument("--transfer_retrieve_topk", type=int, default=24)
    parser.add_argument("--transfer_sr_pairs", type=int, default=512)
    args = parser.parse_args()

    cfg = apply_common_args(load_config(args.config), args)
    set_seed(cfg.get("train", {}).get("seed", 2026))
    sources = parse_list(args.sources, str) or ["PEMS08"]
    targets = parse_list(args.targets, str) or ["PEMS04", "PEMS07"]
    pairs = parse_pairs(args.pairs, sources, targets)
    all_datasets = sorted(set([s for s, _ in pairs] + [t for _, t in pairs]))
    cfg = force_transfer_compatible_presets(cfg, all_datasets, args)
    horizons = parse_list(args.horizons, int) or [12]
    horizons = [h for h in horizons if h <= args.pred_len]
    if not horizons:
        horizons = [args.pred_len]

    source_cache = {}
    rows = []
    for source, target in pairs:
        if source not in source_cache:
            source_cache[source] = train_or_load_source(cfg, source, args.pred_len, horizons, args)
        new_rows, _ = evaluate_transfer(source_cache[source], cfg, source, target, args.pred_len, horizons, args)
        rows.extend(new_rows)

    result_path = os.path.join(cfg["experiment"].get("save_dir", "./outputs"), "zero_shot_results.csv")
    save_results_csv(rows, result_path)
    print("\n========== Zero-shot Final Results ==========")
    for r in rows:
        print(
            f"ZERO_SHOT | source={r['source']} -> target={r['target']} | pred_len={r['pred_len']} | "
            f"horizon={r['horizon']} | MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} MAPE={r['MAPE']:.4f}% | "
            f"loaded={r['loaded_params']} skipped={r['skipped_params']}"
        )
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()

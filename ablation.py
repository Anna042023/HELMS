"""
ablation.py

Run HELMS ablation experiments on PeMS04 and PeMS08.

Paper ablations:
  1) w/o HMC: remove Hypergraph Memory Construction and memory retrieval.
  2) w/o DML: keep the initialized HMC memory bank, but disable lifecycle
     utility update, memory creation, consolidation and forgetting.
  3) w/o SR: remove Semantic Regularization from the objective.

The script is designed to be placed in the same directory as main.py and reuses
all existing folders/modules without modifying them.
"""

import argparse
import csv
import os
import time
from copy import deepcopy
from typing import Dict, List, Optional

import torch
import yaml

from train.train_helms import HELMSTrainer, EarlyStopping
from utils.metrics import compute_all_metrics
from utils.seed import set_seed


ABLATION_MODES = ["wo_hmc", "wo_dml", "wo_sr"]
MODE_NAME = {
    "wo_hmc": "w/o HMC",
    "wo_dml": "w/o DML",
    "wo_sr": "w/o SR",
}


class StaticNoDML:
    """No-op replacement for DifferentiableMemoryLifecycle.

    This keeps the initial HMC memory database fixed during training and
    evaluation, which corresponds to removing DML while preserving HMC and SR.
    """

    def before_optimizer_step(self, memory):
        return None

    @torch.no_grad()
    def update_utilities(self, memory, alpha_full, delta_loss):
        return None

    @torch.no_grad()
    def create_new_memories(self, *args, **kwargs):
        return 0

    @torch.no_grad()
    def consolidate_core(self, memory):
        return None

    @torch.no_grad()
    def forget_obsolete(self, memory, epoch: int):
        return 0

    def epoch_end(self, memory, epoch: int):
        return 0


class AblationTrainer(HELMSTrainer):
    """HELMSTrainer with ablation switches.

    The original trainer is intentionally left unchanged.  Only this subclass
    changes the training/evaluation behavior for the ablation mode.
    """

    def __init__(self, cfg: Dict, dataset_name: str, pred_len: int = 12, ablation: str = "wo_hmc"):
        if ablation not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation mode: {ablation}")
        self.ablation = ablation

        # w/o HMC and w/o SR do not need LLM semantic annotation.  Set this
        # before HELMSTrainer applies dataset presets; apply_dataset_preset keeps
        # an explicit False and will not silently re-enable LLM loading.
        cfg = deepcopy(cfg)
        if ablation in {"wo_hmc", "wo_sr"}:
            cfg.setdefault("memory", {})["use_llm"] = False
        if ablation in {"wo_hmc", "wo_sr"}:
            cfg.setdefault("memory", {})["sr_weight"] = 0.0

        super().__init__(cfg, dataset_name=dataset_name, pred_len=pred_len)

        # Put different ablations into separated output folders.
        self.save_dir = os.path.join(
            self.cfg["experiment"].get("save_dir", "./outputs"),
            "ablation",
            self.ablation,
            self.dataset_name,
            f"H{pred_len}",
        )
        os.makedirs(self.save_dir, exist_ok=True)

        if self.ablation == "wo_dml":
            self.dml = StaticNoDML()
            if hasattr(self.model, "memory"):
                # With no utility update, utility-biased retrieval is also
                # disabled so that the memory bank remains a static HMC bank.
                self.model.memory.utility_retrieval_weight = 0.0
            self.cfg.setdefault("memory", {})["utility_retrieval_weight"] = 0.0

        if self.ablation in {"wo_hmc", "wo_sr"}:
            self.cfg.setdefault("memory", {})["sr_weight"] = 0.0
            if hasattr(self.model, "memory"):
                self.model.memory.sr_pairs = 0

    def _use_hmc_memory(self) -> bool:
        return self.ablation != "wo_hmc"

    def _use_dml(self) -> bool:
        return self.ablation not in {"wo_hmc", "wo_dml"}

    def _use_sr(self) -> bool:
        return self.ablation not in {"wo_hmc", "wo_sr"}

    def initialize_memory(self):
        if not self._use_hmc_memory():
            print(f"[{self.dataset_name}][{MODE_NAME[self.ablation]}] Skip HMC memory initialization.")
            return
        return super().initialize_memory()

    def train_one_epoch(self, epoch: int):
        if self._use_hmc_memory():
            # For w/o DML / w/o SR, reuse the original training loop.
            # w/o DML has already replaced self.dml with a no-op lifecycle.
            # w/o SR has already set sr_weight=0 and sr_pairs=0.
            return super().train_one_epoch(epoch)
        return self._train_one_epoch_without_hmc(epoch)

    def _train_one_epoch_without_hmc(self, epoch: int):
        """Train only the no-memory ST-GNN/temporal-prior branch.

        This removes HMC retrieval entirely.  It is stronger and cleaner than
        merely setting the memory gate to a small value, because the forward pass
        uses use_memory=False in both training and evaluation.
        """
        self.model.train()
        total_loss = total_pred = total_rel = total_prior = 0.0
        n = 0
        log_interval = self.cfg["train"].get("log_interval", 50)

        for step, batch in enumerate(self._progress(self.train_loader, desc=f"train epoch {epoch} {MODE_NAME[self.ablation]}"), 1):
            x, y, tf, start = self._move_batch(batch)
            train_bases = self.cfg.get("model", {}).get(
                "train_prior_bases", self.cfg.get("model", {}).get("prior_bases", None)
            )
            prior_norm = self._make_normalized_priors(start, allowed_bases=train_bases) if self._use_model_priors() else None
            with self._amp_context():
                pred, aux = self.model(
                    x, tf, self.static_adj, use_memory=False, return_aux=True,
                    external_priors=prior_norm,
                )
                pred_loss, comp = self._loss_components(pred, y)
                prior_aux = torch.zeros((), device=pred.device, dtype=pred.dtype)
                if aux.get("prior_value") is not None:
                    prior_aux = self._prediction_loss(aux["prior_value"], y)
                loss = pred_loss + float(self.cfg.get("loss", {}).get("prior_aux_weight", 0.15)) * prior_aux

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler_amp.scale(loss).backward()
            self.scaler_amp.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"].get("grad_clip", 5.0))
            self.scaler_amp.step(self.optimizer)
            self.scaler_amp.update()

            total_loss += float(loss.detach())
            total_pred += float(pred_loss.detach())
            total_rel += float(comp["relative"])
            total_prior += float(prior_aux.detach())
            n += 1

            if log_interval and step % log_interval == 0:
                print(
                    f"epoch {epoch} step {step}: loss={total_loss/n:.5f}, "
                    f"pred={total_pred/n:.5f}, rel={total_rel/n:.5f}, prior={total_prior/n:.5f}"
                )

        return {
            "loss": total_loss / max(1, n),
            "pred_loss": total_pred / max(1, n),
            "rel_loss": total_rel / max(1, n),
            "prior_loss": total_prior / max(1, n),
            "sem_loss": 0.0,
            "base_loss": total_pred / max(1, n),
            "created": 0,
            "removed": 0,
            "K": 0,
        }

    @torch.no_grad()
    def collect_raw_predictions(self, split="val", return_starts: bool = False, collect_priors: bool = False):
        if self._use_hmc_memory():
            return super().collect_raw_predictions(split=split, return_starts=return_starts, collect_priors=collect_priors)
        return self._collect_raw_predictions_without_hmc(split=split, return_starts=return_starts, collect_priors=collect_priors)

    @torch.no_grad()
    def _collect_raw_predictions_without_hmc(self, split="val", return_starts: bool = False, collect_priors: bool = False):
        self.model.eval()
        loader = self.val_loader if split == "val" else self.test_loader
        preds, trues, lasts, starts_all = [], [], [], []
        prior_lists = None
        collect_priors = bool(collect_priors and getattr(self, "calibrator", None) is not None and self.calibrator.enabled)

        exp_cfg = self.cfg.get("experiment", {}) or {}
        max_samples = int(exp_cfg.get(f"{split}_prediction_max_samples", 0) or 0)
        seen = 0

        for batch in self._progress(loader, desc=f"predict {split} {MODE_NAME[self.ablation]}"):
            if max_samples > 0 and seen >= max_samples:
                break
            x, y, tf, start = self._move_batch(batch)
            if max_samples > 0 and seen + x.shape[0] > max_samples:
                keep = max_samples - seen
                x, y, tf, start = x[:keep], y[:keep], tf[:keep], start[:keep]

            prior_norm = self._make_normalized_priors(start) if self._use_model_priors() else None
            pred, aux = self.model(
                x, tf, self.static_adj, use_memory=False, return_aux=True,
                external_priors=prior_norm,
            )
            pred_raw = self.datamodule.scaler.inverse_transform_torch(pred.detach().cpu()).float()[..., :1]
            true_raw = self.datamodule.scaler.inverse_transform_torch(y.detach().cpu()).float()[..., :1]
            last_raw = self.datamodule.scaler.inverse_transform_torch(x[:, -1:, :, :1].detach().cpu()).float()[..., :1]
            preds.append(pred_raw)
            trues.append(true_raw)
            lasts.append(last_raw)
            starts_all.append(start.detach().cpu().long())
            seen += int(pred_raw.shape[0])

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
                for k in prior_lists:
                    if k in priors:
                        prior_lists[k].append(priors[k].float()[..., :1])

        priors_cat = {k: torch.cat(v, dim=0) for k, v in (prior_lists or {}).items() if len(v) > 0}
        out = (torch.cat(preds, dim=0), torch.cat(trues, dim=0), torch.cat(lasts, dim=0), priors_cat)
        if return_starts:
            out = out + (torch.cat(starts_all, dim=0),)
        return out

    def fit(self, eval_horizons: Optional[List[int]] = None):
        """A lightweight fit loop for ablation experiments.

        The original main.py saves many visualization artifacts.  Ablation only
        needs MAE/RMSE/MAPE, so this method saves metrics only and avoids extra
        attention/memory/case-study exports.
        """
        self.current_eval_horizons = list(eval_horizons or [self.pred_len])
        init_cfg = self.cfg.get("train", {})

        if bool(init_cfg.get("closed_form_nodewise_init", True)):
            self.initialize_nodewise_prior_closed_form(
                max_fit_samples=int(init_cfg.get("closed_form_max_samples", 20000)),
                lam=float(init_cfg.get("closed_form_lambda", 1e-2)),
                chunk_size=int(init_cfg.get("closed_form_chunk_size", 96)),
            )

        self.warmup_without_memory(self.cfg["train"].get("warmup_epochs", 5))
        self.initialize_memory()

        stopper = EarlyStopping(patience=self.cfg["train"].get("patience", 20))
        for epoch in range(1, self.cfg["train"].get("epochs", 200) + 1):
            t0 = time.time()
            train_log = self.train_one_epoch(epoch)
            val_metrics = self.evaluate("val", horizons=eval_horizons)
            val_score = self._validation_score(val_metrics, eval_horizons)
            self.scheduler.step(val_score)
            print(
                f"[{self.dataset_name}][{MODE_NAME[self.ablation]}][H={self.pred_len}] "
                f"epoch {epoch} {time.time()-t0:.1f}s train={train_log} "
                f"val={val_metrics} val_score={val_score:.4f}"
            )
            if stopper.step(val_score, self.model):
                print(
                    f"[{self.dataset_name}][{MODE_NAME[self.ablation]}] "
                    f"Early stopping at epoch {epoch}. Best val score={stopper.best:.4f}"
                )
                break

        stopper.restore(self.model, self.device)
        self.fit_validation_calibrator(eval_horizons=eval_horizons)

        test_pack = self.collect_raw_predictions("test", return_starts=True, collect_priors=True)
        raw_test_metrics = self._metrics_from_prediction_pack(test_pack, horizons=eval_horizons, use_calibration=False)
        calibrated_test_metrics = self._metrics_from_prediction_pack(test_pack, horizons=eval_horizons, use_calibration=True)

        print(f"[{self.dataset_name}][{MODE_NAME[self.ablation]}][H={self.pred_len}] TEST_RAW {raw_test_metrics}")
        print(f"[{self.dataset_name}][{MODE_NAME[self.ablation]}][H={self.pred_len}] TEST_CALIBRATED {calibrated_test_metrics}")

        self.save_metrics_csv(
            raw_test_metrics=raw_test_metrics,
            calibrated_test_metrics=calibrated_test_metrics,
            eval_horizons=eval_horizons,
        )
        return calibrated_test_metrics if self.calibrator.enabled else raw_test_metrics


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_results_csv(rows: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["dataset", "variant", "pred_len", "horizon", "MAE", "RMSE", "MAPE"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_list(text: Optional[str], default: List[str]) -> List[str]:
    if text is None or str(text).strip() == "":
        return list(default)
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: Optional[str], default: List[int]) -> List[int]:
    if text is None or str(text).strip() == "":
        return list(default)
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def run_one(cfg: Dict, dataset: str, pred_len: int, horizons: List[int], ablation: str) -> List[Dict]:
    local_cfg = deepcopy(cfg)
    local_cfg["data"]["pred_len"] = pred_len
    trainer = AblationTrainer(local_cfg, dataset_name=dataset, pred_len=pred_len, ablation=ablation)
    metrics = trainer.fit(eval_horizons=horizons)
    rows = []
    for h, vals in metrics.items():
        rows.append({
            "dataset": trainer.dataset_name,
            "variant": MODE_NAME[ablation],
            "pred_len": int(pred_len),
            "horizon": int(h),
            "MAE": float(vals["MAE"]),
            "RMSE": float(vals["RMSE"]),
            "MAPE": float(vals["MAPE"]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run w/o HMC, w/o DML and w/o SR ablation experiments on PeMS04 and PeMS08")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="PEMS04,PEMS08")
    parser.add_argument("--modes", type=str, default="wo_hmc,wo_dml,wo_sr",
                        help="Comma-separated modes: wo_hmc,wo_dml,wo_sr")
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--horizons", type=str, default="12")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--sentence_model_path", type=str, default=None)
    parser.add_argument("--llm_model_path", type=str, default=None)
    parser.add_argument("--disable_llm", action="store_true")
    parser.add_argument("--disable_calibration", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.root_path is not None:
        cfg["data"]["root_path"] = args.root_path
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.warmup_epochs is not None:
        cfg["train"]["warmup_epochs"] = args.warmup_epochs
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience
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
    if args.disable_calibration:
        cfg.setdefault("calibration", {})["enabled"] = False
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)

    seed = int(cfg.get("train", {}).get("seed", 2026))
    set_seed(seed)

    datasets = parse_csv_list(args.datasets, ["PEMS04", "PEMS08"])
    modes = parse_csv_list(args.modes, ABLATION_MODES)
    for m in modes:
        if m not in ABLATION_MODES:
            raise ValueError(f"Unknown mode {m}. Valid modes: {ABLATION_MODES}")
    horizons = parse_int_list(args.horizons, [12])

    rows = []
    for dataset in datasets:
        for mode in modes:
            print("\n" + "=" * 90)
            print(f"Run ablation: dataset={dataset}, variant={MODE_NAME[mode]}, pred_len={args.pred_len}, horizons={horizons}")
            print("=" * 90)
            # Reset the seed before every run so differences mainly come from the ablated module.
            set_seed(seed)
            rows.extend(run_one(cfg, dataset, args.pred_len, horizons, mode))

    base_save = cfg["experiment"].get("save_dir", "./outputs")
    result_path = os.path.join(base_save, "ablation_results.csv")
    save_results_csv(rows, result_path)

    print("\n========== Final Ablation Results ==========")
    for r in rows:
        print(
            f"{r['dataset']} | {r['variant']} | pred_len={r['pred_len']} | horizon={r['horizon']} | "
            f"MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} MAPE={r['MAPE']:.4f}%"
        )
    print(f"Saved ablation results to {result_path}")


if __name__ == "__main__":
    main()

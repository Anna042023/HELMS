import os
import sys
import time
import json
import pickle
import csv
from contextlib import nullcontext
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.traffic_dataset import TrafficDataModule
from datasets.data_utils import canonical_name
from models.helms import HELMS
from models.dml import DifferentiableMemoryLifecycle
from utils.clustering import cluster_prototypes
from utils.metrics import compute_all_metrics
from utils.semantic_utils import SemanticAnnotator
from utils.calibration import ValidationCalibrator


class EarlyStopping:
    def __init__(self, patience=20, mode="min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.count = 0
        self.best_state = None

    def step(self, value, model):
        improve = self.best is None or (value < self.best if self.mode == "min" else value > self.best)
        if improve:
            self.best = value
            self.count = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.count += 1
        return self.count >= self.patience

    def restore(self, model, device):
        if self.best_state is None:
            return
        current = model.state_dict()
        loadable = {}
        skipped = []
        for k, v in self.best_state.items():
            if k in current and tuple(current[k].shape) == tuple(v.shape):
                loadable[k] = v.to(device)
            else:
                cur_shape = tuple(current[k].shape) if k in current else None
                skipped.append((k, tuple(v.shape), cur_shape))
        model.load_state_dict(loadable, strict=False)
        if hasattr(model, "memory") and hasattr(model.memory, "rebuild_hypergraph"):
            model.memory.rebuild_hypergraph()
        if skipped:
            short = "; ".join([f"{k}: saved={a}, current={b}" for k, a, b in skipped[:5]])
            print(f"[WARN] Skipped dynamic/mismatched checkpoint tensors during restore: {short}")


def _deep_update(dst: Dict, src: Dict) -> Dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def apply_dataset_preset(cfg: Dict, dataset_name: str) -> Dict:
    """Apply optional dataset-specific hyperparameters before model creation.

    The same architecture is used for all datasets.  Presets only adjust robust
    training weights and memory stability according to dataset scale, which
    avoids the observed pattern: high MAE/RMSE on larger/noisier datasets and
    high MAPE on low-flow PeMS04/PeMS08.
    """
    # Preserve explicit command-line switches that are applied to the top-level
    # config before the trainer is constructed.  In older code, dataset presets
    # could silently re-enable LLM annotation after --disable_llm, which wasted
    # time and sometimes caused model-loading failures.
    force_disable_llm = (cfg.get("memory", {}) or {}).get("use_llm", None) is False
    out = deepcopy(cfg)
    presets = out.get("dataset_presets", {}) or {}
    key = canonical_name(dataset_name)
    if key in presets:
        _deep_update(out, presets[key])
        if force_disable_llm:
            out.setdefault("memory", {})["use_llm"] = False
        print(f"[{key}] Applied dataset preset: {presets[key]}")
    return out


class HELMSTrainer:
    def __init__(self, cfg: Dict, dataset_name: str, pred_len: int = 12):
        self.dataset_name = canonical_name(dataset_name)
        self.cfg = apply_dataset_preset(cfg, self.dataset_name)
        self.pred_len = pred_len
        self.current_eval_horizons = [pred_len]
        train_cfg = self.cfg["train"]
        data_cfg = self.cfg["data"]
        self.device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        self.datamodule = TrafficDataModule(
            data_cfg["root_path"], self.dataset_name,
            seq_len=data_cfg.get("seq_len", 12), pred_len=pred_len,
            train_ratio=data_cfg.get("train_ratio", 0.7), val_ratio=data_cfg.get("val_ratio", 0.1),
            points_per_hour=data_cfg.get("points_per_hour", 12),
            enable_ridge_ar=data_cfg.get("enable_ridge_ar", True),
            normalization=data_cfg.get("normalization", "global"),
            # These two options are ignored by TrafficDataModule unless the
            # dataset is exactly PEMS04 or METR-LA.
            impute_missing=data_cfg.get("impute_missing", False),
            missing_value=data_cfg.get("missing_value", 0.0),
        )
        self.static_adj = torch.tensor(self.datamodule.adj, dtype=torch.float32, device=self.device)
        self._norm_mean = torch.as_tensor(self.datamodule.scaler.mean[..., :1], dtype=torch.float32, device=self.device)
        self._norm_std = torch.as_tensor(self.datamodule.scaler.std[..., :1], dtype=torch.float32, device=self.device).clamp_min(1e-6)
        requested_bs = int(train_cfg.get("batch_size", 16))
        max_physical_bs = int(train_cfg.get("max_physical_batch_size", requested_bs))
        loader_bs = max(1, min(requested_bs, max_physical_bs))
        if loader_bs < requested_bs:
            print(f"[{self.dataset_name}] Requested batch_size={requested_bs}, using physical batch_size={loader_bs} to avoid GPU OOM.")
        self.physical_batch_size = loader_bs
        nw = int(train_cfg.get("num_workers", 2))
        loader_kwargs = dict(num_workers=nw, pin_memory=(self.device.type == "cuda"))
        if nw > 0:
            loader_kwargs.update(
                persistent_workers=bool(train_cfg.get("persistent_workers", True)),
                prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
            )
        train_dataset = self.datamodule.get_dataset("train")
        self.train_loader = DataLoader(
            train_dataset, batch_size=loader_bs,
            shuffle=True, drop_last=True, **loader_kwargs
        )
        # Fast warmup does not run the memory/ST encoder, so it can safely use a
        # larger physical batch on PeMS07/PEMS-BAY.  This keeps the same data and
        # epochs but reduces Python/DataLoader/prior-construction overhead during
        # the slow initial warmup.
        fast_bs = int(train_cfg.get("fast_warmup_batch_size", requested_bs))
        fast_bs = max(loader_bs, fast_bs)
        if fast_bs != loader_bs:
            print(f"[{self.dataset_name}] Fast warmup uses batch_size={fast_bs} while full training uses {loader_bs}.")
        self.fast_warmup_loader = DataLoader(
            train_dataset, batch_size=fast_bs,
            shuffle=True, drop_last=True, **loader_kwargs
        )
        self.init_loader = DataLoader(
            self.datamodule.get_dataset("train"), batch_size=loader_bs,
            shuffle=False, drop_last=False, **loader_kwargs
        )
        eval_bs = max(1, min(loader_bs, int(train_cfg.get("max_eval_batch_size", loader_bs))))
        self.val_loader = DataLoader(
            self.datamodule.get_dataset("val"), batch_size=eval_bs,
            shuffle=False, drop_last=False, **loader_kwargs
        )
        self.test_loader = DataLoader(
            self.datamodule.get_dataset("test"), batch_size=eval_bs,
            shuffle=False, drop_last=False, **loader_kwargs
        )

        model_cfg = self.cfg["model"]
        mem_cfg_raw = self.cfg["memory"]
        memory_cfg = dict(
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
        self.model = HELMS(
            num_nodes=self.datamodule.num_nodes,
            input_dim=self.datamodule.input_dim,
            output_dim=self.datamodule.output_dim,
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
            seq_len=data_cfg.get("seq_len", 12),
            history_prior=model_cfg.get("history_prior", True),
            stid_prior=model_cfg.get("stid_prior", True),
            stid_hidden_dim=model_cfg.get("stid_hidden_dim", max(96, model_cfg.get("hidden_dim", 64) * 2)),
            stid_node_dim=model_cfg.get("stid_node_dim", 32),
            external_prior_fusion=model_cfg.get("external_prior_fusion", True),
            adaptive_prior_fusion=model_cfg.get("adaptive_prior_fusion", True),
            prior_bases=model_cfg.get("prior_bases", None),
            memory_cfg=memory_cfg,
        ).to(self.device)
        self.dml = DifferentiableMemoryLifecycle(
            utility_decay=mem_cfg_raw.get("utility_decay", 0.99),
            theta_new=mem_cfg_raw.get("theta_new", 0.3),
            theta_low=mem_cfg_raw.get("theta_low", 0.2),
            core_ratio=mem_cfg_raw.get("core_ratio", 0.2),
            idle_epochs=mem_cfg_raw.get("idle_epochs", 10),
            lifecycle_interval=mem_cfg_raw.get("lifecycle_interval", 5),
            core_grad_scale=mem_cfg_raw.get("core_grad_scale", 0.1),
            utility_center=mem_cfg_raw.get("utility_center", 0.5),
        )
        # Use a slightly higher LR for the very small direct temporal heads
        # (node-wise NLinear / shared NLinear / history MLP).  These heads are
        # the fastest route to lower validation metrics; the ST-GNN and memory
        # modules keep the base LR for stability.
        base_lr = train_cfg.get("learning_rate", 1e-3)
        fast_names = ("nodewise_prior", "nlinear", "hist_mlp", "stid_prior", "stid_gate", "nodewise_gate", "history_gate", "hist_mlp_gate", "memory_residual_gate")
        fast_params, base_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in fast_names):
                fast_params.append(param)
            else:
                base_params.append(param)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": base_lr},
                {"params": fast_params, "lr": float(train_cfg.get("fast_head_lr", 3.0 * base_lr))},
            ],
            weight_decay=train_cfg.get("weight_decay", 1e-4)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=train_cfg.get("lr_factor", 0.5),
            patience=train_cfg.get("lr_patience", 5), min_lr=train_cfg.get("min_lr", 1e-5)
        )
        amp_enabled = bool(train_cfg.get("amp", True) and self.device.type == "cuda")
        try:
            self.scaler_amp = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except Exception:
            self.scaler_amp = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        self.save_dir = os.path.join(self.cfg["experiment"].get("save_dir", "./outputs"), self.dataset_name, f"H{pred_len}")
        os.makedirs(self.save_dir, exist_ok=True)
        # Paper-aligned offline semantic annotation: local Qwen/LLM -> tag/description -> Sentence-BERT embedding.
        llm_device = mem_cfg_raw.get("llm_device", "auto")
        if llm_device == "same_as_train":
            llm_device = str(self.device)
        self.semantic_annotator = SemanticAnnotator(
            mem_cfg_raw.get("semantic_dim", 384),
            mem_cfg_raw.get("sentence_model_path", None),
            device=str(self.device),
            llm_model_path=mem_cfg_raw.get("llm_model_path", None),
            use_llm=mem_cfg_raw.get("use_llm", True),
            llm_device=llm_device,
            llm_max_new_tokens=mem_cfg_raw.get("llm_max_new_tokens", 96),
            llm_temperature=mem_cfg_raw.get("llm_temperature", 0.2),
            keep_llm_loaded=mem_cfg_raw.get("keep_llm_loaded", False),
            cache_dir=os.path.join(self.save_dir, "semantic_cache"),
            dataset_name=self.dataset_name,
        )
        self.new_memory_tag, self.new_memory_desc, self.new_memory_semantic = self.semantic_annotator.default_new_memory_semantic()
        cal_cfg = deepcopy(self.cfg.get("calibration", {}))
        # Keep metric masking consistent with evaluation.
        cal_cfg.setdefault("null_val", self.cfg.get("metrics", {}).get("null_val", 0.0))
        cal_cfg.setdefault("mape_threshold", self.cfg.get("metrics", {}).get("mape_threshold", 1e-5))
        cal_cfg.setdefault("mape_denom_eps", self.cfg.get("metrics", {}).get("mape_denom_eps", 1e-5))
        self.calibrator = ValidationCalibrator(self.dataset_name, cal_cfg)
        # Diagnostics are saved only for later visualization/analysis.
        # They do not participate in optimization and therefore do not change the
        # original training or evaluation behavior.
        self.memory_trace = []
        # Validation-selected causal residual adapter.  This is fitted on the
        # validation stream only and then applied to TEST_CALIBRATED.  It leaves
        # TEST_RAW unchanged.
        self.online_adapter_params = None
        self.online_adapter_summary = {}

    def _move_batch(self, batch):
        x, y, tf, start = batch
        return x.to(self.device, non_blocking=True), y[..., :1].to(self.device, non_blocking=True), tf.to(self.device, non_blocking=True), start

    def _use_model_priors(self) -> bool:
        # Building daily/node-wise analog priors inside every training batch is
        # slow on PeMS07/PeMS-BAY.  By default, train the neural HELMS model
        # without these CPU-heavy priors and still expose them to the validation
        # calibrator.  Set train.use_external_priors_during_training=true only
        # when enough CPU/GPU memory is available.
        return bool(self.cfg.get("train", {}).get("use_external_priors_during_training", False))

    def _amp_context(self):
        """Version-safe AMP context without per-batch deprecation warnings.

        torch.cuda.amp.autocast(...) prints a FutureWarning in newer PyTorch
        versions.  Because this context is entered once per mini-batch, that
        warning both floods tqdm output and slows warmup through console I/O.
        """
        enabled = bool(getattr(self, "scaler_amp", None) is not None and self.scaler_amp.is_enabled())
        if not enabled:
            return nullcontext()
        try:
            return torch.amp.autocast(device_type=self.device.type, enabled=True)
        except Exception:
            return torch.cuda.amp.autocast(enabled=True)

    def _show_batch_progress(self) -> bool:
        """Whether to draw tqdm bars for ordinary inner batch loops.

        Default stays False so regular training/validation logs remain compact.
        Warmup has its own switch below, because it is the longest waiting
        phase and the user needs visible ETA without FutureWarning spam.
        """
        train_cfg = self.cfg.get("train", {}) or {}
        if "show_batch_progress" in train_cfg:
            return bool(train_cfg.get("show_batch_progress"))
        return str(train_cfg.get("progress_mode", "epoch")).lower() in {"bar", "tqdm", "batch"}

    def _show_warmup_progress(self) -> bool:
        """Show a compact tqdm bar only for warmup epochs by default.

        This restores real-time progress/ETA for the slow PEMS07/PEMS-BAY
        warmup, while avoiding the previous problem where every batch occupied
        a new terminal line.  The AMP deprecation warning has already been
        removed by _amp_context(), so tqdm can now safely reuse one line.
        """
        train_cfg = self.cfg.get("train", {}) or {}
        if "show_warmup_progress" in train_cfg:
            return bool(train_cfg.get("show_warmup_progress"))
        return True

    def _progress(self, iterable, desc: str, force: Optional[bool] = None):
        show = self._show_batch_progress() if force is None else bool(force)
        if not show:
            return iterable
        try:
            total = len(iterable)
        except Exception:
            total = None
        miniters = 1
        if total:
            miniters = max(1, total // 100)
        train_cfg = self.cfg.get("train", {}) or {}
        mininterval = float(train_cfg.get("progress_mininterval", 2.0))
        return tqdm(
            iterable,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            mininterval=mininterval,
            miniters=miniters,
            file=sys.stdout,
        )

    @staticmethod
    def _slice_prior_dict(priors, start: int, end: int):
        if priors is None:
            return None
        return {k: v[start:end] for k, v in priors.items()}

    def _make_normalized_priors(self, starts, allowed_bases=None):
        """Build legal inference-time priors and convert them to normalized space.

        Priors are generated only from training templates or observations before
        the forecast origin.  They are passed to the model during both training
        and evaluation so the model learns to use them instead of relying on a
        post-hoc validation calibrator only.
        """
        allowed = allowed_bases if allowed_bases is not None else self.cfg.get("model", {}).get("prior_bases", None)
        priors_raw = self.datamodule.make_periodic_priors(starts, pred_len=self.pred_len, allowed_bases=allowed)
        if not priors_raw:
            return {}
        # Transfer all prior bases in one contiguous block instead of launching
        # one CPU->GPU copy per basis.  The numerical values are unchanged, but
        # PeMS07/PEMS-BAY warmup avoids many small synchronization points.
        names = list(priors_raw.keys())
        cpu_parts = [priors_raw[k].float()[..., :1] for k in names]
        stacked = torch.stack(cpu_parts, dim=0).to(self.device, non_blocking=True)  # [K,B,T,N,1]
        stacked = (stacked - self._norm_mean) / self._norm_std
        return {k: stacked[i] for i, k in enumerate(names)}

    def _horizon_weights(self, T: int, device, dtype, phase: str = "train"):
        loss_cfg = self._loss_cfg_for_phase(phase)
        mode = loss_cfg.get("horizon_weight", "linear")
        if mode == "last":
            w = torch.ones(T, device=device, dtype=dtype) * 0.5
            w[-1] = float(loss_cfg.get("last_weight", 2.0))
        elif mode == "linear":
            start = float(loss_cfg.get("horizon_start_weight", 0.7))
            end = float(loss_cfg.get("horizon_end_weight", 1.3))
            w = torch.linspace(start, end, T, device=device, dtype=dtype)
        else:
            w = torch.ones(T, device=device, dtype=dtype)
        w = w / w.mean().clamp_min(1e-6)
        return w.view(1, T, 1, 1)

    def _loss_cfg_for_phase(self, phase: str = "train"):
        base = deepcopy(self.cfg.get("loss", {}) or {})
        if phase == "warmup":
            override = self.cfg.get("warmup_loss", None)
            if isinstance(override, dict) and override:
                _deep_update(base, override)
        return base

    def _loss_components(self, pred, y, phase: str = "train"):
        """Training loss aligned with the reported raw-scale metrics.

        The original loss path is preserved for all datasets except PeMS04 and
        METR-LA.  For these two datasets only, zero/missing labels are masked in
        the training loss in the same way as in the reported traffic metrics.
        """
        loss_cfg = self._loss_cfg_for_phase(phase)
        w = self._horizon_weights(pred.shape[1], pred.device, pred.dtype, phase=phase)
        use_target_metric_fix = (
            self.dataset_name in {"PEMS04", "METR-LA"}
            and bool(getattr(self.datamodule, "impute_missing", False))
        )

        if use_target_metric_fix:
            pred_raw = self.datamodule.scaler.inverse_transform_torch(pred.float())[..., :1]
            y_raw = self.datamodule.scaler.inverse_transform_torch(y.float())[..., :1]

            metric_cfg = self.cfg.get("metrics", {}) or {}
            null_val = metric_cfg.get("null_val", 0.0)
            threshold = metric_cfg.get("mape_threshold", 1e-5)
            valid_mask = torch.isfinite(y_raw)
            if null_val is not None:
                valid_mask = valid_mask & (y_raw != float(null_val))
            if threshold is not None:
                valid_mask = valid_mask & (torch.abs(y_raw) > float(threshold))
            if valid_mask.sum() == 0:
                valid_mask = torch.isfinite(y_raw)
            m = valid_mask.to(dtype=pred.dtype)
            den = (m * w).sum().clamp_min(1.0)

            err = pred - y
            abs_err = torch.abs(err)
            mae = (abs_err * w * m).sum() / den
            huber = (F.smooth_l1_loss(pred, y, reduction="none") * w * m).sum() / den
            rmse_proxy = torch.sqrt(((err ** 2) * w * m).sum() / den + 1e-6)

            if loss_cfg.get("clip_pred_for_relative_loss", True):
                pred_for_rel = torch.clamp(pred_raw, min=0.0)
            else:
                pred_for_rel = pred_raw
            denom_min = float(loss_cfg.get("mape_denominator_min", 5.0))
            rel = torch.abs(pred_for_rel - y_raw) / torch.abs(y_raw).clamp_min(denom_min)
            rel = (rel.to(pred.dtype) * w * m).sum() / den
            neg_penalty = torch.relu(-pred_raw).mean().to(pred.dtype)

            if pred.shape[1] > 1:
                delta_pred = pred[:, 1:] - pred[:, :-1]
                delta_true = y[:, 1:] - y[:, :-1]
                dm = (valid_mask[:, 1:] & valid_mask[:, :-1]).to(dtype=pred.dtype)
                if dm.sum() > 0:
                    delta_loss = (F.smooth_l1_loss(delta_pred, delta_true, reduction="none") * dm).sum() / dm.sum().clamp_min(1.0)
                else:
                    delta_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
            else:
                delta_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

            eval_horizons = getattr(self, "current_eval_horizons", None) or [pred.shape[1]]
            refs_all = (self.cfg.get("calibration", {}).get("reference", {}) or {}).get(self.dataset_name, {})
            metric_terms = []
            metric_weights = loss_cfg.get("eval_metric_weights", {"MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0})
            for h in eval_horizons:
                h = int(h)
                if h < 1 or h > pred_raw.shape[1]:
                    continue
                p = pred_for_rel[:, h-1:h]
                yy = y_raw[:, h-1:h]
                diff = p - yy
                hm = valid_mask[:, h-1:h].to(dtype=pred.dtype)
                hden = hm.sum().clamp_min(1.0)
                ref = refs_all.get(str(h), refs_all.get(int(h), {})) if isinstance(refs_all, dict) else {}
                ref_mae = max(float(ref.get("MAE", 1.0)), 1e-6) if isinstance(ref, dict) else 1.0
                ref_rmse = max(float(ref.get("RMSE", 1.0)), 1e-6) if isinstance(ref, dict) else 1.0
                ref_mape = max(float(ref.get("MAPE", 1.0)), 1e-6) if isinstance(ref, dict) else 1.0
                raw_mae = (diff.abs() * hm).sum() / hden / ref_mae
                raw_rmse = torch.sqrt(((diff ** 2) * hm).sum() / hden + 1e-6) / ref_rmse
                raw_mape = ((diff.abs() / yy.abs().clamp_min(denom_min)) * hm).sum() / hden * 100.0 / ref_mape
                metric_terms.append(
                    float(metric_weights.get("MAE", 1.0)) * raw_mae +
                    float(metric_weights.get("RMSE", 1.0)) * raw_rmse +
                    float(metric_weights.get("MAPE", 1.0)) * raw_mape
                )
            if metric_terms:
                eval_metric = torch.stack(metric_terms).mean().to(pred.dtype)
            else:
                eval_metric = torch.zeros((), device=pred.device, dtype=pred.dtype)
        else:
            # Original loss path: do not change the behavior of PeMS03/07/08/PeMS-BAY.
            err = pred - y
            abs_err = torch.abs(err)
            mae = (abs_err * w).mean()
            huber = (F.smooth_l1_loss(pred, y, reduction="none") * w).mean()
            rmse_proxy = torch.sqrt(((err ** 2) * w).mean() + 1e-6)

            pred_raw = self.datamodule.scaler.inverse_transform_torch(pred.float())[..., :1]
            y_raw = self.datamodule.scaler.inverse_transform_torch(y.float())[..., :1]
            if loss_cfg.get("clip_pred_for_relative_loss", True):
                pred_for_rel = torch.clamp(pred_raw, min=0.0)
            else:
                pred_for_rel = pred_raw
            denom_min = float(loss_cfg.get("mape_denominator_min", 5.0))
            rel = torch.abs(pred_for_rel - y_raw) / torch.abs(y_raw).clamp_min(denom_min)
            rel = (rel.to(pred.dtype) * w).mean()
            neg_penalty = torch.relu(-pred_raw).mean().to(pred.dtype)

            # Future-shape consistency: improves RMSE and avoids overly flat curves
            # without changing the evaluation protocol.
            if pred.shape[1] > 1:
                delta_pred = pred[:, 1:] - pred[:, :-1]
                delta_true = y[:, 1:] - y[:, :-1]
                delta_loss = F.smooth_l1_loss(delta_pred, delta_true)
            else:
                delta_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

            # Direct raw-scale metric loss on the horizons that will be reported.
            eval_horizons = getattr(self, "current_eval_horizons", None) or [pred.shape[1]]
            refs_all = (self.cfg.get("calibration", {}).get("reference", {}) or {}).get(self.dataset_name, {})
            metric_terms = []
            metric_weights = loss_cfg.get("eval_metric_weights", {"MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0})
            for h in eval_horizons:
                h = int(h)
                if h < 1 or h > pred_raw.shape[1]:
                    continue
                p = pred_for_rel[:, h-1:h]
                yy = y_raw[:, h-1:h]
                diff = p - yy
                ref = refs_all.get(str(h), refs_all.get(int(h), {})) if isinstance(refs_all, dict) else {}
                ref_mae = max(float(ref.get("MAE", 1.0)), 1e-6) if isinstance(ref, dict) else 1.0
                ref_rmse = max(float(ref.get("RMSE", 1.0)), 1e-6) if isinstance(ref, dict) else 1.0
                ref_mape = max(float(ref.get("MAPE", 1.0)), 1e-6) if isinstance(ref, dict) else 1.0
                raw_mae = diff.abs().mean() / ref_mae
                raw_rmse = torch.sqrt((diff ** 2).mean() + 1e-6) / ref_rmse
                raw_mape = (diff.abs() / yy.abs().clamp_min(denom_min)).mean() * 100.0 / ref_mape
                metric_terms.append(
                    float(metric_weights.get("MAE", 1.0)) * raw_mae +
                    float(metric_weights.get("RMSE", 1.0)) * raw_rmse +
                    float(metric_weights.get("MAPE", 1.0)) * raw_mape
                )
            if metric_terms:
                eval_metric = torch.stack(metric_terms).mean().to(pred.dtype)
            else:
                eval_metric = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total = (
            float(loss_cfg.get("mae_weight", 1.0)) * mae
            + float(loss_cfg.get("huber_weight", 0.2)) * huber
            + float(loss_cfg.get("rmse_weight", 0.08)) * rmse_proxy
            + float(loss_cfg.get("mape_weight", 0.05)) * rel
            + float(loss_cfg.get("eval_metric_weight", 0.0)) * eval_metric
            + float(loss_cfg.get("delta_weight", 0.04)) * delta_loss
            + float(loss_cfg.get("nonnegative_weight", 0.001)) * neg_penalty
        )
        return total, {
            "mae": mae.detach(), "huber": huber.detach(), "rmse_proxy": rmse_proxy.detach(),
            "relative": rel.detach(), "eval_metric": eval_metric.detach(),
            "delta": delta_loss.detach(), "neg": neg_penalty.detach(),
        }

    def _prediction_loss(self, pred, y, phase: str = "train"):
        return self._loss_components(pred, y, phase=phase)[0]

    @torch.no_grad()
    def initialize_nodewise_prior_closed_form(self, max_fit_samples: int = 20000, lam: float = 1e-2, chunk_size: int = 96):
        """Closed-form initialization for the node-wise NLinear head.

        A major reason the previous runs got stuck above STID/GWNet is that the
        direct node-wise head started from zeros and had to learn thousands of
        sensor-specific AR coefficients only through mini-batch SGD.  This method
        solves the same linear residual problem once on the chronological training
        split and writes the solution into NodeWiseLinearPrior before warmup.
        It uses normalized training data only, so it does not leak validation/test
        information and adds only a few seconds compared with 200 training epochs.
        """
        if not hasattr(self.model, "nodewise_prior"):
            return
        data = np.asarray(self.datamodule.data[:, :, :1], dtype=np.float32)
        starts = np.asarray(self.datamodule.train_indices, dtype=np.int64)
        if starts.size == 0:
            return
        if starts.size > int(max_fit_samples):
            # Even sampling is deterministic and keeps all periods represented.
            starts = starts[np.linspace(0, starts.size - 1, int(max_fit_samples)).round().astype(np.int64)]
        L = int(self.datamodule.seq_len)
        T = int(self.pred_len)
        N = int(self.datamodule.num_nodes)
        idx_hist = starts[:, None] + np.arange(L, dtype=np.int64)[None, :]
        idx_fut = starts[:, None] + L + np.arange(T, dtype=np.int64)[None, :]
        eye = None
        w_all = torch.zeros_like(self.model.nodewise_prior.weight.data)
        b_all = torch.zeros_like(self.model.nodewise_prior.bias.data)
        for ns in range(0, N, int(chunk_size)):
            ne = min(N, ns + int(chunk_size))
            hist = data[idx_hist, ns:ne, 0]  # [M,L,C]
            last = hist[:, -1:, :]
            centered = hist - last
            fut = data[idx_fut, ns:ne, 0] - last  # [M,T,C]
            # Per node: X=[centered history, bias], Y=future residual.
            X = np.transpose(centered, (2, 0, 1)).astype(np.float64)  # [C,M,L]
            ones = np.ones((X.shape[0], X.shape[1], 1), dtype=np.float64)
            X = np.concatenate([X, ones], axis=-1)  # [C,M,L+1]
            Y = np.transpose(fut, (2, 0, 1)).astype(np.float64)  # [C,M,T]
            XtX = np.einsum('cmi,cmj->cij', X, X, optimize=True)
            Xty = np.einsum('cmi,cmt->cit', X, Y, optimize=True)
            if eye is None or eye.shape[-1] != L + 1:
                eye = np.eye(L + 1, dtype=np.float64)
                eye[-1, -1] = 0.01  # barely regularize the intercept
            XtX = XtX + float(lam) * eye[None, :, :]
            try:
                sol = np.linalg.solve(XtX, Xty)  # [C,L+1,T]
            except Exception:
                sol = np.matmul(np.linalg.pinv(XtX), Xty)
            coef = sol[:, :L, :].transpose(0, 2, 1)  # [C,T,L]
            bias = sol[:, L, :]  # [C,T]
            w_all[ns:ne, 0, :, :] = torch.from_numpy(coef[:, :, :]).float().to(w_all.device)
            b_all[ns:ne, 0, :] = torch.from_numpy(bias).float().to(b_all.device)
        self.model.nodewise_prior.weight.data.copy_(w_all)
        self.model.nodewise_prior.bias.data.copy_(b_all)
        print(f"[{self.dataset_name}] Initialized node-wise linear prior by closed-form ridge on {len(starts)} train windows.")

    def warmup_without_memory(self, epochs: int):
        if epochs <= 0:
            return
        train_cfg = self.cfg.get("train", {})
        fast_epochs = int(train_cfg.get("fast_warmup_epochs", max(0, epochs - int(train_cfg.get("st_warmup_epochs", 3)))))
        st_epochs = int(train_cfg.get("st_warmup_epochs", max(0, epochs - fast_epochs)))
        fast_epochs = max(0, min(fast_epochs, epochs))
        st_epochs = max(0, min(st_epochs, epochs - fast_epochs))
        print(f"[{self.dataset_name}] Warmup: fast_history={fast_epochs} epochs, full_ST={st_epochs} epochs.")
        if self._use_model_priors():
            fb = train_cfg.get("fast_warmup_prior_bases", None)
            if fb is not None:
                print(f"[{self.dataset_name}] Fast warmup uses compact priors: {list(fb)}")
        self.model.train()

        # Stage 1: train only history / node-wise / optional prior heads.
        # This avoids the encoder backward pass, which was the source of OOM on PeMS07.
        for ep in range(1, fast_epochs + 1):
            ep_t0 = time.time()
            total = 0.0
            n = 0
            micro_bs = int(train_cfg.get("fast_warmup_micro_batch_size", train_cfg.get("fast_warmup_update_batch_size", self.physical_batch_size)))
            micro_bs = max(1, micro_bs)
            for batch in self._progress(self.fast_warmup_loader, desc=f"fast warmup {ep}/{fast_epochs}", force=self._show_warmup_progress()):
                x_big, y_big, tf_big, start_big = self._move_batch(batch)
                model_cfg = self.cfg.get("model", {})
                # Fast warmup is the main bottleneck on large datasets.  The full
                # HELMS training stage still uses model.train_prior_bases, but
                # fast warmup can safely use a compact set of strong, cheap legal
                # priors (ridge_ar / delta / last / time-of-day).  This avoids
                # rebuilding daily/weekly analog retrieval bases thousands of
                # times before the ST encoder and memory are even trained.
                train_bases = train_cfg.get("fast_warmup_prior_bases", None)
                if train_bases is None:
                    train_bases = model_cfg.get("warmup_prior_bases", model_cfg.get("train_prior_bases", model_cfg.get("prior_bases", None)))
                # Compute legal priors once for a larger warmup batch, then run
                # optimizer updates on micro-batches.  This keeps the previous
                # update batch size, but removes repeated expensive prior
                # construction and CPU->GPU copies.
                prior_big = self._make_normalized_priors(start_big, allowed_bases=train_bases) if self._use_model_priors() else None
                B_big = int(x_big.shape[0])
                for s0 in range(0, B_big, micro_bs):
                    s1 = min(B_big, s0 + micro_bs)
                    x = x_big[s0:s1]
                    y = y_big[s0:s1]
                    tf = tf_big[s0:s1]
                    prior_norm = self._slice_prior_dict(prior_big, s0, s1)
                    with self._amp_context():
                        pred = self.model.fast_history_forward(x, time_feat=tf, external_priors=prior_norm)
                        loss = self._prediction_loss(pred, y, phase="warmup")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler_amp.scale(loss).backward()
                    self.scaler_amp.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"].get("grad_clip", 5.0))
                    self.scaler_amp.step(self.optimizer)
                    self.scaler_amp.update()
                    total += float(loss.detach())
                    n += 1
            dt = time.time() - ep_t0
            print(f"[{self.dataset_name}] fast warmup epoch {ep}: balanced_loss={total / max(1, n):.6f} time={dt:.1f}s steps={n}")

        # Stage 2: short full no-memory warmup to initialize the ST encoder/decoder.
        for ep in range(1, st_epochs + 1):
            ep_t0 = time.time()
            total = 0.0
            n = 0
            for batch in self._progress(self.train_loader, desc=f"ST warmup {ep}/{st_epochs}", force=self._show_warmup_progress()):
                x, y, tf, start = self._move_batch(batch)
                train_bases = self.cfg.get("model", {}).get("train_prior_bases", self.cfg.get("model", {}).get("prior_bases", None))
                prior_norm = self._make_normalized_priors(start, allowed_bases=train_bases) if self._use_model_priors() else None
                with self._amp_context():
                    pred, aux = self.model(x, tf, self.static_adj, use_memory=False, return_aux=True, external_priors=prior_norm)
                    loss = self._prediction_loss(pred, y, phase="warmup")
                    if aux.get("prior_value") is not None:
                        warm_loss_cfg = self._loss_cfg_for_phase("warmup")
                        loss = loss + float(warm_loss_cfg.get("prior_aux_weight", self.cfg.get("loss", {}).get("prior_aux_weight", 0.25))) * self._prediction_loss(aux["prior_value"], y, phase="warmup")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler_amp.scale(loss).backward()
                self.scaler_amp.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"].get("grad_clip", 5.0))
                self.scaler_amp.step(self.optimizer)
                self.scaler_amp.update()
                total += float(loss.detach())
                n += 1
            dt = time.time() - ep_t0
            print(f"[{self.dataset_name}] ST warmup epoch {ep}: balanced_loss={total / max(1, n):.6f} time={dt:.1f}s steps={n}")

        try:
            self.model.decoder.load_state_dict(self.model.baseline_decoder.state_dict())
            print(f"[{self.dataset_name}] Copied warmup baseline decoder to memory-enhanced decoder.")
        except Exception as e:
            print(f"[WARN] Could not copy baseline decoder weights: {e}")

    @torch.no_grad()
    def collect_states_for_memory(self):
        self.model.eval()
        mem_cfg = self.cfg["memory"]
        max_samples = mem_cfg.get("init_sample_size", 4096)
        states = []
        raw_windows = []
        futures = []
        base_preds = []
        labels_len = 0
        for batch in self._progress(self.init_loader, desc="collect memory states"):
            x, y, tf, _ = self._move_batch(batch)
            # Use the warmed-up no-memory HELMS branch to build a residual
            # memory bank: each cluster stores mean (y - base_pred).  This is
            # still HMC/DML memory reuse, not an external prior, and directly
            # teaches retrieval what each historical prototype contributes.
            pred0, aux0 = self.model(x, tf, self.static_adj, use_memory=False, return_aux=True, external_priors=None)
            states.append(aux0["h_t"].detach().cpu())
            raw_windows.append(x.detach().cpu().numpy())
            futures.append(y.detach().cpu())
            base_preds.append(pred0.detach().cpu())
            labels_len += x.shape[0]
            if labels_len >= max_samples:
                break
        states = torch.cat(states, dim=0)[:max_samples]
        raw_windows = np.concatenate(raw_windows, axis=0)[:max_samples]
        futures = torch.cat(futures, dim=0)[:max_samples]
        base_preds = torch.cat(base_preds, dim=0)[:max_samples]
        return states, raw_windows, futures, base_preds

    @staticmethod
    def _cluster_mean_residuals(labels: torch.Tensor, futures: torch.Tensor, base_preds: torch.Tensor, k: int):
        # labels: [S], futures/base_preds: [S,T,N,C].  Output [K,T,N,C] and [K].
        labels = labels.detach().cpu().long().clamp(0, k - 1)
        residual = (futures - base_preds).detach().float().cpu()
        counts = torch.bincount(labels, minlength=k).float().clamp_min(1.0)[:k]
        flat = residual.reshape(residual.shape[0], -1)
        out_flat = torch.zeros(k, flat.shape[1], dtype=torch.float32)
        out_flat.index_add_(0, labels, flat)
        out_flat = out_flat / counts.view(k, 1)
        means = out_flat.view((k,) + tuple(residual.shape[1:]))
        # Softly clip extreme cluster residuals; noisy clusters otherwise hurt RMSE.
        means = 2.5 * torch.tanh(means / 2.5)
        return means, counts

    def initialize_memory(self):
        print(f"[{self.dataset_name}] Initialize HMC memory bank.")
        states, raw_windows, futures, base_preds = self.collect_states_for_memory()
        mem_cfg = self.cfg["memory"]
        centers, labels, _ = cluster_prototypes(
            states, k=mem_cfg.get("init_memory_size", 200),
            method=mem_cfg.get("cluster_method", "minibatch_kmeans"),
            ae_epochs=mem_cfg.get("ae_epochs", 20), device=str(self.device)
        )
        tags, descs, sem = self.semantic_annotator.annotate_clusters(raw_windows, labels, centers.shape[0])
        forecast_residuals, residual_counts = self._cluster_mean_residuals(labels, futures, base_preds, centers.shape[0])
        self.model.memory.initialize(
            centers.to(self.device), sem.to(self.device), tags=tags, descriptions=descs,
            assignment_labels=labels, states=states.to(self.device),
            forecast_residuals=forecast_residuals.to(self.device),
            residual_counts=residual_counts.to(self.device)
        )
        print(f"[{self.dataset_name}] Memory initialized: K={self.model.memory.active_k}, hyperedges={self.model.memory.incidence.shape[1]}")

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = total_pred = total_sem = total_base = total_rel = total_prior = 0.0
        n = 0
        created_total = 0
        epoch_states = []
        epoch_assignments = []
        online_edge_limit = int(self.cfg.get("memory", {}).get("online_hyperedge_sample_size", 2048))
        log_interval = self.cfg["train"].get("log_interval", 50)
        mem_cfg = self.cfg["memory"]
        create_after = int(mem_cfg.get("create_after_epoch", 3))
        max_create_per_epoch = int(mem_cfg.get("max_create_per_epoch", 16))
        for step, batch in enumerate(self._progress(self.train_loader, desc=f"train epoch {epoch}"), 1):
            x, y, tf, start = self._move_batch(batch)
            train_bases = self.cfg.get("model", {}).get("train_prior_bases", self.cfg.get("model", {}).get("prior_bases", None))
            prior_norm = self._make_normalized_priors(start, allowed_bases=train_bases) if self._use_model_priors() else None
            with self._amp_context():
                pred, aux = self.model(x, tf, self.static_adj, use_memory=True, return_aux=True, external_priors=prior_norm)
                base_pred = aux.get("base_pred")
                if base_pred is None:
                    base_pred = self.model.decode_baseline(aux["H_t"], prior_value=aux.get("prior_value"))
                    if prior_norm is not None:
                        base_pred = self.model.apply_external_prior_fusion(base_pred, x, prior_norm)
                pred_loss, comp = self._loss_components(pred, y)
                base_loss, _ = self._loss_components(base_pred, y)
                base_loss_detached = base_loss.detach()
                sem_loss = self.model.semantic_loss()
                prior_aux = torch.zeros((), device=pred.device, dtype=pred.dtype)
                if aux.get("prior_value") is not None:
                    prior_aux = self._prediction_loss(aux["prior_value"], y)
                base_aux_weight = float(self.cfg.get("loss", {}).get("base_aux_weight", 0.25))
                # Keep the no-memory branch strong throughout HELMS training.
                # This branch is also used in the paper to compute Delta L for DML;
                # supervising it prevents the memory gate from degrading all metrics.
                loss = (
                    pred_loss
                    + base_aux_weight * base_loss
                    + mem_cfg.get("sr_weight", 0.05) * sem_loss
                    + float(self.cfg.get("loss", {}).get("prior_aux_weight", 0.15)) * prior_aux
                )
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler_amp.scale(loss).backward()
            self.scaler_amp.unscale_(self.optimizer)
            self.dml.before_optimizer_step(self.model.memory)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["train"].get("grad_clip", 5.0))
            self.scaler_amp.step(self.optimizer)
            self.scaler_amp.update()

            with torch.no_grad():
                # Paper utility signal: Delta L_t = L_base - L_pred.
                delta = base_loss_detached - pred_loss.detach()
                self.dml.update_utilities(self.model.memory, aux["alpha"], delta)
                self.model.memory.update_access(aux["alpha"], epoch)
                active_idx = aux.get("active_idx", None)
                if active_idx is not None and len(active_idx) > 0 and len(epoch_states) * x.shape[0] < online_edge_limit:
                    alpha_active = aux["alpha"][:, active_idx]
                    assign_pos = torch.argmax(alpha_active, dim=-1).detach().cpu()
                    epoch_assignments.append(assign_pos)
                    epoch_states.append(aux["h_t"].detach().cpu())
                if epoch >= create_after and created_total < max_create_per_epoch:
                    new_residual = 2.5 * torch.tanh((y - base_pred).detach() / 2.5)
                    created_total += self.dml.create_new_memories(
                        self.model.memory, aux["h_t"], aux["alpha"],
                        semantic_vector=self.new_memory_semantic.to(self.device),
                        tag=self.new_memory_tag, description=self.new_memory_desc, epoch=epoch,
                        max_create_per_batch=min(1, max_create_per_epoch - created_total),
                        forecast_residuals=new_residual
                    )

            total_loss += float(loss.detach())
            total_pred += float(pred_loss.detach())
            total_sem += float(sem_loss.detach())
            total_base += float(base_loss_detached)
            total_rel += float(comp["relative"])
            total_prior += float(prior_aux.detach())
            n += 1
            if log_interval and step % log_interval == 0:
                print(
                    f"epoch {epoch} step {step}: loss={total_loss/n:.5f}, pred={total_pred/n:.5f}, "
                    f"rel={total_rel/n:.5f}, prior={total_prior/n:.5f}, sem={total_sem/n:.5f}, base={total_base/n:.5f}, K={self.model.memory.active_k}"
                )
        removed = self.dml.epoch_end(self.model.memory, epoch)
        if epoch % mem_cfg.get("refresh_interval", 1) == 0:
            if epoch_states and epoch_assignments:
                states_ep = torch.cat(epoch_states, dim=0)[:online_edge_limit].to(self.device)
                labels_ep = torch.cat(epoch_assignments, dim=0)[:online_edge_limit]
                self.model.memory.rebuild_hypergraph(assignment_labels=labels_ep, states=states_ep, update_cache=True)
            else:
                self.model.memory.rebuild_hypergraph()
        return {
            "loss": total_loss / max(1, n), "pred_loss": total_pred / max(1, n),
            "rel_loss": total_rel / max(1, n), "prior_loss": total_prior / max(1, n), "sem_loss": total_sem / max(1, n),
            "base_loss": total_base / max(1, n), "created": created_total,
            "removed": removed, "K": self.model.memory.active_k
        }

    @torch.no_grad()
    def collect_raw_predictions(self, split="val", return_starts: bool = False, collect_priors: bool = False):
        """Collect raw-scale predictions and labels.

        Returns predictions in the original data scale.  When ``return_starts`` is
        true, the chronological window starts are also returned; these starts are
        required by the causal online residual adapter so that only already
        observed target timestamps can update the correction state.
        """
        self.model.eval()
        loader = self.val_loader if split == "val" else self.test_loader
        preds, trues, lasts, starts_all = [], [], [], []
        prior_lists = None
        collect_priors = bool(collect_priors and getattr(self, "calibrator", None) is not None and self.calibrator.enabled)

        # v28 speed path introduced optional truncation for diagnostic prediction
        # collection, but the local variables were not initialized.  Keep the
        # default at 0 so validation/test metrics still use the full split and
        # model performance is unchanged.  Users can optionally set
        # experiment.val_prediction_max_samples / test_prediction_max_samples
        # only for quick debugging.
        exp_cfg = self.cfg.get("experiment", {}) or {}
        max_samples = int(exp_cfg.get(f"{split}_prediction_max_samples", 0) or 0)
        seen = 0

        for batch in self._progress(loader, desc=f"predict {split}"):
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
            seen += int(pred_raw.shape[0])

            if collect_priors:
                cal_cfg = self.cfg.get("calibration", {}) or {}
                if bool(cal_cfg.get("use_external_priors", False)):
                    priors = self.datamodule.make_periodic_priors(start, pred_len=self.pred_len, allowed_bases=cal_cfg.get("allowed_bases", None))
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

    @torch.no_grad()
    def _causal_online_residual_apply_impl(self, pred_raw: torch.Tensor, true_raw: torch.Tensor,
                                           starts: torch.Tensor, params: Optional[Dict]) -> torch.Tensor:
        """Apply a causal residual EMA with explicit parameters.

        Only residuals whose target timestamp is already observed at the current
        forecast origin are used.  Therefore this can be used for online/lifelong
        evaluation without leaking current or future test labels.
        """
        if not params or not bool(params.get("enabled", False)):
            return torch.clamp(pred_raw, min=0.0)
        if pred_raw.numel() == 0:
            return pred_raw

        pred = pred_raw.detach().float().cpu().clone()
        true = true_raw.detach().float().cpu()
        starts_cpu = starts.detach().cpu().long().view(-1)
        B, T, N, C = pred.shape
        if B == 0:
            return pred

        order = torch.argsort(starts_cpu)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), dtype=order.dtype)
        pred_s = pred.index_select(0, order)
        true_s = true.index_select(0, order)
        starts_s = starts_cpu.index_select(0, order)
        origins = starts_s + int(self.datamodule.seq_len) - 1

        decay = float(params.get("ema_decay", 0.96))
        gain = float(params.get("gain", 0.18))
        min_obs = int(params.get("min_observations", 2))
        clip_ratio = float(params.get("max_correction_std_ratio", 0.08))
        horizon_decay = float(params.get("horizon_decay", 0.98))
        global_gain = float(params.get("global_gain", 0.05))
        global_decay = float(params.get("global_ema_decay", 0.94))
        min_global_obs = int(params.get("min_global_observations", 8))

        # Cache node-wise training std on first use.  Computing this for every
        # validation/test call is unnecessary on PeMS07/PeMS-BAY.
        cached = getattr(self, "_online_node_std", None)
        if cached is not None and tuple(cached.shape) == (1, N, C):
            train_std = cached.clone()
        else:
            train_std_np = None
            if self.dataset_name in {"PEMS04", "METR-LA"} and bool(getattr(self.datamodule, "impute_missing", False)):
                raw_data = getattr(self.datamodule, "feature_raw_data", None)
            else:
                raw_data = getattr(self.datamodule, "raw_data", None)
            train_end = int(getattr(self.datamodule, "train_end", 0))
            if raw_data is not None and train_end > 1:
                arr = np.asarray(raw_data[:train_end, :, :C], dtype=np.float32)
                if arr.ndim == 3 and arr.shape[1] == N and arr.shape[2] == C:
                    train_std_np = np.nanstd(arr, axis=0, keepdims=True)
            if train_std_np is None:
                std_np = np.asarray(getattr(self.datamodule.scaler, "std", 1.0), dtype=np.float32)
                if std_np.size == 1:
                    train_std_np = np.full((1, N, C), float(std_np.reshape(-1)[0]), dtype=np.float32)
                else:
                    std_np = np.squeeze(std_np)
                    if std_np.ndim == 1:
                        std_np = std_np[:C].reshape(1, 1, C)
                        train_std_np = np.repeat(std_np, N, axis=1).astype(np.float32)
                    elif std_np.ndim == 2 and std_np.shape[0] == N:
                        train_std_np = std_np[:, :C].reshape(1, N, C).astype(np.float32)
                    else:
                        train_std_np = np.full((1, N, C), float(np.nanmean(std_np)), dtype=np.float32)
            train_std = torch.as_tensor(train_std_np, dtype=torch.float32).clamp_min(1.0)
            self._online_node_std = train_std.clone()

        corr_clip = (clip_ratio * train_std).clamp_min(float(params.get("min_abs_clip", 0.25)))
        ema = torch.zeros(T, N, C, dtype=torch.float32)
        counts = torch.zeros(T, 1, 1, dtype=torch.float32)
        global_ema = torch.zeros(N, C, dtype=torch.float32)
        global_count = torch.zeros(1, dtype=torch.float32)
        out = pred_s.clone()

        import heapq
        pending = []
        push_id = 0
        h_scale = torch.pow(torch.full((T, 1, 1), horizon_decay, dtype=torch.float32),
                            torch.arange(T, dtype=torch.float32).view(T, 1, 1))
        for i in range(B):
            origin = int(origins[i].item())
            while pending and pending[0][0] <= origin:
                _, _, h_idx, residual = heapq.heappop(pending)
                node_clip = corr_clip[0]
                residual = torch.maximum(torch.minimum(residual, node_clip), -node_clip)
                if counts[h_idx].item() <= 0:
                    ema[h_idx] = residual
                else:
                    ema[h_idx] = decay * ema[h_idx] + (1.0 - decay) * residual
                counts[h_idx] += 1.0
                if global_count.item() <= 0:
                    global_ema = residual.clone()
                else:
                    global_ema = global_decay * global_ema + (1.0 - global_decay) * residual
                global_count += 1.0

            valid = (counts >= float(min_obs)).to(out.dtype)
            valid_global = (global_count >= float(min_global_obs)).to(out.dtype)
            correction = gain * h_scale * valid * ema
            correction = correction + global_gain * h_scale * valid_global * global_ema.view(1, N, C)
            full_clip = corr_clip.expand_as(correction)
            correction = torch.maximum(torch.minimum(correction, full_clip), -full_clip)
            out[i] = torch.clamp(pred_s[i] + correction, min=0.0)

            for h in range(T):
                target_time = origin + h + 1
                residual = (true_s[i, h] - out[i, h]).detach().clone()
                heapq.heappush(pending, (target_time, push_id, h, residual))
                push_id += 1

        return out.index_select(0, inv)

    @torch.no_grad()
    def _apply_causal_online_residual_adapter(self, pred_raw: torch.Tensor, true_raw: torch.Tensor,
                                              starts: torch.Tensor) -> torch.Tensor:
        # Prefer validation-selected parameters.  If none are selected, fall back
        # to config only when online_adaptation.enabled=true.
        params = getattr(self, "online_adapter_params", None)
        if params is None:
            cfg = self.cfg.get("online_adaptation", {}) or {}
            params = cfg if bool(cfg.get("enabled", False)) else None
        return self._causal_online_residual_apply_impl(pred_raw, true_raw, starts, params)


    @torch.no_grad()
    def _fit_calendar_residual_profile(self, val_pack, eval_horizons: Optional[List[int]] = None):
        """Fit validation-only calendar residual profiles for selected datasets.

        This is disabled by default and only becomes active through a dataset
        preset (used here for PEMS04 and METR-LA).  It estimates stable residual
        bias by future time-of-day / sensor / horizon on the chronological
        validation split, selects a conservative correction on a holdout part of
        validation, and applies the fixed profile to test predictions.  No test
        labels are used.  TEST_RAW is never changed.
        """
        cfg = self.cfg.get("calendar_residual_calibration", {}) or {}
        self.calendar_residual_profile = None
        if not bool(cfg.get("enabled", False)):
            return
        pred_raw, true_raw, last_raw, priors_raw, starts = val_pack
        horizons = list(eval_horizons or [self.pred_len])
        metric_cfg = self.cfg.get("metrics", {}) or {}
        if getattr(self, "calibrator", None) is not None and self.calibrator.enabled and self.calibrator.params:
            base_pred = self.calibrator.apply(pred_raw, last_raw, priors_raw)
        else:
            base_pred = torch.clamp(pred_raw, min=0.0)
        base_pred = self._apply_calendar_residual_profile(base_pred, starts)
        base_pred = self._apply_calibrated_output_clip(base_pred)
        base_metrics = compute_all_metrics(
            torch.clamp(base_pred, min=0.0), true_raw, scaler=None, horizons=horizons,
            null_val=metric_cfg.get("null_val", 0.0),
            mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
            mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
            clip_nonnegative=False,
        )
        base_score = self._validation_score(base_metrics, horizons)
        B = int(base_pred.shape[0])
        if B < int(cfg.get("min_rows", 256)):
            print(f"[{self.dataset_name}] Calendar residual calibration skipped: too few validation rows ({B}).")
            return
        hold_ratio = min(max(float(cfg.get("safety_holdout_ratio", 0.35)), 0.05), 0.80)
        split = int(round(B * (1.0 - hold_ratio)))
        min_hold = int(cfg.get("min_holdout_rows", 96))
        if split < min_hold or (B - split) < min_hold:
            split = max(min_hold, min(B - min_hold, int(B * 0.65)))
        if split <= 0 or split >= B:
            print(f"[{self.dataset_name}] Calendar residual calibration skipped: invalid split B={B}, split={split}.")
            return

        bp_fit = base_pred[:split].float().cpu()
        yt_fit = true_raw[:split].float().cpu()
        st_fit = starts[:split].detach().cpu().long()
        bp_score = base_pred[split:].float().cpu()
        yt_score = true_raw[split:].float().cpu()
        st_score = starts[split:].detach().cpu().long()
        T, N, C = int(bp_fit.shape[1]), int(bp_fit.shape[2]), int(bp_fit.shape[3])
        day_period = int(getattr(self.datamodule, "day_period", 288))
        seq_len = int(self.datamodule.seq_len)
        residual = (yt_fit - bp_fit).numpy().astype(np.float32, copy=False)  # [B,T,N,C]
        starts_np = st_fit.numpy().astype(np.int64, copy=False)
        shrink = float(cfg.get("shrink_count", 24.0))
        # Robust residual clip estimated on validation-fit residuals.  This keeps
        # the profile as a small bias correction instead of an oracle-like jump.
        finite = residual[np.isfinite(residual)]
        if finite.size == 0:
            print(f"[{self.dataset_name}] Calendar residual calibration skipped: no finite residuals.")
            return
        q = float(cfg.get("residual_clip_quantile", 0.985))
        max_abs_corr = float(np.quantile(np.abs(finite), q))
        max_abs_corr = max(float(cfg.get("min_abs_correction", 0.25)), max_abs_corr * float(cfg.get("clip_scale", 0.65)))

        prof_node = np.zeros((T, day_period, N, C), dtype=np.float32)
        prof_count = np.zeros((T, day_period, 1, 1), dtype=np.float32)
        prof_global = np.zeros((T, day_period, 1, C), dtype=np.float32)
        node_bias = np.zeros((T, 1, N, C), dtype=np.float32)
        for h in range(T):
            slots = (starts_np + seq_len + h) % day_period
            sum_node = np.zeros((day_period, N, C), dtype=np.float32)
            cnt_slot = np.zeros((day_period, 1, 1), dtype=np.float32)
            # np.add.at is slower than vectorized bincount for very large arrays,
            # but validation is small and this runs once per dataset.
            np.add.at(sum_node, slots, residual[:, h])
            cnt = np.bincount(slots, minlength=day_period).astype(np.float32).reshape(day_period, 1, 1)
            denom = np.maximum(cnt, 1.0)
            raw_prof = sum_node / denom
            shrink_fac = cnt / (cnt + shrink)
            raw_prof = raw_prof * shrink_fac
            raw_prof = np.clip(raw_prof, -max_abs_corr, max_abs_corr)
            prof_node[h] = raw_prof
            prof_count[h] = cnt
            # Global time-of-day residual across nodes is more stable and helps
            # when a per-node slot has few validation examples.
            sg = sum_node.mean(axis=1, keepdims=True)
            prof_global[h] = np.clip(sg / np.maximum(denom, 1.0) * shrink_fac, -max_abs_corr, max_abs_corr)
            nb = residual[:, h].mean(axis=0, keepdims=True)
            node_bias[h] = np.clip(nb * (residual.shape[0] / (residual.shape[0] + shrink)), -max_abs_corr, max_abs_corr)

        def apply_profile(base: torch.Tensor, starts_t: torch.Tensor, weights):
            wn, wg, wb = [float(x) for x in weights]
            arr = base.float().cpu().numpy().astype(np.float32, copy=True)
            st = starts_t.detach().cpu().numpy().astype(np.int64, copy=False)
            for h in range(min(T, arr.shape[1])):
                slots = (st + seq_len + h) % day_period
                corr = wn * prof_node[h, slots] + wg * prof_global[h, slots] + wb * node_bias[h, 0]
                if float(cfg.get("max_change_ratio", 0.0)) > 0:
                    max_delta = np.maximum(np.abs(arr[:, h]) * float(cfg.get("max_change_ratio", 0.10)),
                                           float(cfg.get("min_abs_correction", 0.25)))
                    corr = np.maximum(np.minimum(corr, max_delta), -max_delta)
                arr[:, h] = np.maximum(arr[:, h] + corr, 0.0)
            return torch.from_numpy(arr)

        weight_grid = cfg.get("weight_grid", None)
        if not weight_grid:
            weight_grid = [
                [0.0, 0.0, 0.0],
                [0.18, 0.10, 0.04], [0.28, 0.12, 0.05], [0.36, 0.14, 0.06],
                [0.22, 0.20, 0.04], [0.30, 0.24, 0.05], [0.42, 0.18, 0.08],
                [0.12, 0.26, 0.05], [0.0, 0.35, 0.08], [0.30, 0.0, 0.08],
            ]
        best_score = base_score
        best_metrics = base_metrics
        best_weights = None
        tol = cfg.get("metric_guard_tolerance", {"MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0})
        for weights in weight_grid:
            cand = apply_profile(bp_score, st_score, weights)
            cand_metrics = compute_all_metrics(
                cand, yt_score, scaler=None, horizons=horizons,
                null_val=metric_cfg.get("null_val", 0.0),
                mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
                mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
                clip_nonnegative=False,
            )
            score = self._validation_score(cand_metrics, horizons)
            ok = score < best_score * float(cfg.get("min_score_improvement_ratio", 0.9995))
            if ok:
                for h in horizons:
                    if h not in cand_metrics or h not in base_metrics:
                        continue
                    for mk in ["MAE", "RMSE", "MAPE"]:
                        guard = float(tol.get(mk, 1.0)) if isinstance(tol, dict) else float(tol)
                        if cand_metrics[h][mk] > base_metrics[h][mk] * guard + 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                best_score = score
                best_metrics = cand_metrics
                best_weights = [float(x) for x in weights]
        if best_weights is None:
            self.calendar_residual_profile = None
            print(f"[{self.dataset_name}] Calendar residual calibration disabled by validation guard. base_score={base_score:.4f}")
            return
        self.calendar_residual_profile = {
            "weights": best_weights,
            "prof_node": prof_node,
            "prof_global": prof_global,
            "node_bias": node_bias,
            "day_period": day_period,
            "seq_len": seq_len,
            "max_abs_corr": max_abs_corr,
            "max_change_ratio": float(cfg.get("max_change_ratio", 0.10)),
            "min_abs_correction": float(cfg.get("min_abs_correction", 0.25)),
        }
        self.calendar_residual_summary = {
            "enabled": True,
            "weights": best_weights,
            "base_metrics": base_metrics,
            "selected_metrics": best_metrics,
            "base_score": float(base_score),
            "selected_score": float(best_score),
        }
        print(f"[{self.dataset_name}] Calendar residual calibration summary: {self.calendar_residual_summary}")

    @torch.no_grad()
    def _apply_calendar_residual_profile(self, pred_raw: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        prof = getattr(self, "calendar_residual_profile", None)
        if not prof:
            return pred_raw
        arr = pred_raw.float().cpu().numpy().astype(np.float32, copy=True)
        st = starts.detach().cpu().numpy().astype(np.int64, copy=False)
        weights = prof["weights"]
        prof_node = prof["prof_node"]
        prof_global = prof["prof_global"]
        node_bias = prof["node_bias"]
        day_period = int(prof["day_period"])
        seq_len = int(prof["seq_len"])
        max_change_ratio = float(prof.get("max_change_ratio", 0.10))
        min_abs_corr = float(prof.get("min_abs_correction", 0.25))
        T = min(arr.shape[1], prof_node.shape[0])
        for h in range(T):
            slots = (st + seq_len + h) % day_period
            corr = weights[0] * prof_node[h, slots] + weights[1] * prof_global[h, slots] + weights[2] * node_bias[h, 0]
            if max_change_ratio > 0:
                max_delta = np.maximum(np.abs(arr[:, h]) * max_change_ratio, min_abs_corr)
                corr = np.maximum(np.minimum(corr, max_delta), -max_delta)
            arr[:, h] = np.maximum(arr[:, h] + corr, 0.0)
        return torch.from_numpy(arr)

    @torch.no_grad()
    def _apply_calibrated_output_clip(self, pred_raw: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.get("calibrated_output_clip", {}) or {}
        if not bool(cfg.get("enabled", False)):
            return pred_raw
        if not hasattr(self, "_calibrated_clip_bounds"):
            if self.dataset_name in {"PEMS04", "METR-LA"} and bool(getattr(self.datamodule, "impute_missing", False)):
                clip_source = self.datamodule.feature_raw_data
            else:
                clip_source = self.datamodule.raw_data
            train_raw = np.asarray(clip_source[:self.datamodule.train_end, :, :1], dtype=np.float32)
            q_low = float(cfg.get("q_low", 0.001))
            q_high = float(cfg.get("q_high", 0.999))
            nodewise = bool(cfg.get("nodewise", True))
            if nodewise:
                lo = np.quantile(train_raw, q_low, axis=0, keepdims=True).astype(np.float32)
                hi = np.quantile(train_raw, q_high, axis=0, keepdims=True).astype(np.float32)
            else:
                lo = np.array([[[np.quantile(train_raw, q_low)]]], dtype=np.float32)
                hi = np.array([[[np.quantile(train_raw, q_high)]]], dtype=np.float32)
            margin = float(cfg.get("margin_ratio", 0.05))
            span = np.maximum(hi - lo, 1.0)
            lo = lo - margin * span
            hi = hi + margin * span
            if cfg.get("min_value", None) is not None:
                lo = np.maximum(lo, float(cfg.get("min_value")))
            if cfg.get("max_value", None) is not None:
                hi = np.minimum(hi, float(cfg.get("max_value")))
            self._calibrated_clip_bounds = (
                torch.as_tensor(lo, dtype=torch.float32),
                torch.as_tensor(hi, dtype=torch.float32),
            )
        lo, hi = self._calibrated_clip_bounds
        return torch.maximum(torch.minimum(pred_raw.float().cpu(), hi), lo)

    @torch.no_grad()
    def _tune_online_residual_adapter(self, val_pack, eval_horizons: Optional[List[int]] = None):
        """Select a conservative causal residual adapter on validation only.

        This does not modify TEST_RAW.  It is used only for TEST_CALIBRATED and
        is accepted only when the validation stream shows a real improvement over
        the current calibrated/raw output without worsening any reported metric.
        """
        online_cfg = self.cfg.get("online_adaptation", {}) or {}
        if not bool(online_cfg.get("tune_on_validation", True)):
            self.online_adapter_params = None
            return
        pred_raw, true_raw, last_raw, priors_raw, starts = val_pack
        metric_cfg = self.cfg.get("metrics", {})
        horizons = list(eval_horizons or [self.pred_len])
        if getattr(self, "calibrator", None) is not None and self.calibrator.enabled and self.calibrator.params:
            base_pred = self.calibrator.apply(pred_raw, last_raw, priors_raw)
        else:
            base_pred = torch.clamp(pred_raw, min=0.0)
        base_metrics = compute_all_metrics(
            torch.clamp(base_pred, min=0.0), true_raw, scaler=None, horizons=horizons,
            null_val=metric_cfg.get("null_val", 0.0),
            mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
            mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
            clip_nonnegative=False,
        )
        base_score = self._validation_score(base_metrics, horizons)
        best_score = base_score
        best_metrics = base_metrics
        best_params = None
        # Small grid: conservative enough to avoid the earlier PeMS08 failure,
        # but sufficient to remove stable residual bias from calibrated outputs.
        gains = online_cfg.get("tune_gains", [0.08, 0.12, 0.18, 0.25])
        global_gains = online_cfg.get("tune_global_gains", [0.0, 0.04, 0.08])
        decays = online_cfg.get("tune_ema_decays", [0.94, 0.96, 0.98])
        clip_ratios = online_cfg.get("tune_clip_ratios", [0.04, 0.06, 0.08, 0.12])
        # Limit combinations on large datasets to avoid making calibration itself slow.
        tried = 0
        max_trials = int(online_cfg.get("max_tune_trials", 24))
        for decay in decays:
            for gain in gains:
                for ggain in global_gains:
                    for clip in clip_ratios:
                        if tried >= max_trials:
                            break
                        tried += 1
                        params = {
                            "enabled": True,
                            "ema_decay": float(decay),
                            "gain": float(gain),
                            "min_observations": int(online_cfg.get("min_observations", 2)),
                            "max_correction_std_ratio": float(clip),
                            "horizon_decay": float(online_cfg.get("horizon_decay", 0.985)),
                            "global_gain": float(ggain),
                            "global_ema_decay": float(online_cfg.get("global_ema_decay", 0.94)),
                            "min_global_observations": int(online_cfg.get("min_global_observations", 8)),
                            "min_abs_clip": float(online_cfg.get("min_abs_clip", 0.25)),
                        }
                        cand_pred = self._causal_online_residual_apply_impl(base_pred, true_raw, starts, params)
                        cand_metrics = compute_all_metrics(
                            cand_pred, true_raw, scaler=None, horizons=horizons,
                            null_val=metric_cfg.get("null_val", 0.0),
                            mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
                            mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
                            clip_nonnegative=False,
                        )
                        score = self._validation_score(cand_metrics, horizons)
                        # Accept only if every reported metric is not worse on validation,
                        # and the composite score improves by a small margin.
                        ok = score < best_score * float(online_cfg.get("min_improve_ratio", 0.9995))
                        if ok:
                            for h in horizons:
                                for mk in ["MAE", "RMSE", "MAPE"]:
                                    tol = float(online_cfg.get("metric_guard_tolerance", 1.0005))
                                    if cand_metrics[h][mk] > base_metrics[h][mk] * tol + 1e-8:
                                        ok = False
                                        break
                                if not ok:
                                    break
                        if ok:
                            best_score = score
                            best_metrics = cand_metrics
                            best_params = params
                    if tried >= max_trials:
                        break
                if tried >= max_trials:
                    break
            if tried >= max_trials:
                break
        self.online_adapter_params = best_params
        self.online_adapter_summary = {
            "enabled": bool(best_params is not None),
            "params": best_params,
            "base_metrics": base_metrics,
            "selected_metrics": best_metrics,
            "base_score": float(base_score),
            "selected_score": float(best_score),
            "trials": int(tried),
        }
        print(f"[{self.dataset_name}] Online residual calibration summary: {self.online_adapter_summary}")

    @torch.no_grad()
    def _metrics_from_prediction_pack(self, pack, horizons: Optional[List[int]] = None, use_calibration: bool = False):
        """Compute metrics from a cached (pred,true,last,priors,starts) pack.

        Final testing, prediction saving and case-study export used to run the
        full model three times.  This helper lets those steps share one forward
        pass without changing any training loss or selected checkpoint.
        """
        pred_raw, true_raw, last_raw, priors_raw, starts = pack
        metric_cfg = self.cfg.get("metrics", {})
        if use_calibration and getattr(self, "calibrator", None) is not None and self.calibrator.enabled:
            pred_eval = self.calibrator.apply(pred_raw, last_raw, priors_raw)
            pred_eval = self._apply_calendar_residual_profile(pred_eval, starts)
            pred_eval = self._apply_causal_online_residual_adapter(pred_eval, true_raw, starts)
            pred_eval = self._apply_calibrated_output_clip(pred_eval)
            pred_eval = self._apply_target_post_selector(pred_eval, pred_raw, last_raw, priors_raw)
            pred_eval = self._apply_pems04_mae_rmse_refiner(pred_eval, starts)
        else:
            pred_eval = pred_raw
            if metric_cfg.get("clip_nonnegative", True):
                pred_eval = torch.clamp(pred_eval, min=0.0)
        return compute_all_metrics(
            pred_eval,
            true_raw,
            scaler=None,
            horizons=horizons,
            null_val=metric_cfg.get("null_val", 0.0),
            mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
            mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
            clip_nonnegative=False,
        )

    @torch.no_grad()
    def evaluate(self, split="val", horizons: Optional[List[int]] = None, use_calibration: bool = False):
        pack = self.collect_raw_predictions(split, return_starts=True, collect_priors=bool(use_calibration))
        return self._metrics_from_prediction_pack(pack, horizons=horizons, use_calibration=use_calibration)

    def _validation_score(self, metrics: Dict[int, Dict[str, float]], eval_horizons: Optional[List[int]] = None) -> float:
        """Composite validation score against the best baseline references.

        Early stopping on MAE alone selected checkpoints with acceptable MAE but
        poor RMSE/MAPE.  This score balances all three metrics and all requested
        horizons using only validation labels.
        """
        horizons = eval_horizons or [self.pred_len]
        cal_cfg = self.cfg.get("calibration", {}) or {}
        refs_all = (cal_cfg.get("reference", {}) or {}).get(self.dataset_name, {})
        weights = cal_cfg.get("metric_weights", {"MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0})
        total = 0.0
        count = 0
        for h in horizons:
            if h not in metrics:
                continue
            ref = refs_all.get(str(h), refs_all.get(int(h), {})) if isinstance(refs_all, dict) else {}
            for k in ["MAE", "RMSE", "MAPE"]:
                denom = float(ref.get(k, 1.0)) if isinstance(ref, dict) else 1.0
                denom = denom if denom > 1e-8 else 1.0
                total += float(weights.get(k, 1.0)) * float(metrics[h][k]) / denom
                count += 1
        return total / max(1, count)

    def _target_dataset_metric_fix_enabled(self) -> bool:
        """Whether target-only metric/data fixes may run.

        Keep every extra post-processing branch strictly limited to the two
        datasets requested in this revision.  PeMS03 / PeMS07 / PeMS08 /
        PeMS-BAY never enter this path even though the helper methods live in a
        shared trainer file.
        """
        return (
            self.dataset_name in {"PEMS04", "METR-LA"}
            and bool(getattr(self.datamodule, "impute_missing", False))
        )

    @torch.no_grad()
    def _build_target_post_candidates(
        self,
        pred_raw: torch.Tensor,
        last_raw: torch.Tensor,
        priors_raw: Optional[Dict[str, torch.Tensor]],
        base_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build legal validation/test candidate predictions for target datasets.

        The candidates use only model outputs, known history, and priors already
        constructed from data before each forecast origin.  The selector fitted
        below uses validation labels only; it is then frozen and applied to test.
        """
        T = int(pred_raw.shape[1])
        out: Dict[str, torch.Tensor] = {}
        raw = torch.clamp(pred_raw.detach().float().cpu()[..., :1], min=0.0)
        base = raw if base_pred is None else torch.clamp(base_pred.detach().float().cpu()[..., :1], min=0.0)
        out["raw"] = raw
        out["calibrated"] = base
        if last_raw is not None:
            last = torch.clamp(last_raw.detach().float().cpu()[..., :1], min=0.0)
            if last.dim() == 4 and last.shape[1] == 1:
                out["last"] = last.repeat(1, T, 1, 1)
        cfg = self.cfg.get("post_selector", {}) or {}
        keep = cfg.get("candidate_bases", None)
        keep = None if keep is None else set(str(x) for x in keep)
        if priors_raw:
            for name, val in priors_raw.items():
                if keep is not None and str(name) not in keep:
                    continue
                if val is None:
                    continue
                vv = torch.clamp(val.detach().float().cpu()[..., :1], min=0.0)
                if vv.dim() == 4 and vv.shape[1] >= T and vv.shape[0] == raw.shape[0] and vv.shape[2] == raw.shape[2]:
                    out[str(name)] = vv[:, :T]

        # Add conservative blends around the already calibrated output.  These
        # are useful when a prior has the right temporal shape but a small level
        # bias, which was the remaining pattern in PeMS04 and METR-LA logs.
        blend_alphas = [float(a) for a in cfg.get("blend_alphas", [0.25, 0.50, 0.75])]
        max_blend = int(cfg.get("max_blend_bases", 12))
        base_names = [n for n in out.keys() if n not in {"raw", "calibrated"}]
        for name in base_names[:max_blend]:
            vv = out[name]
            for a in blend_alphas:
                if not (0.0 < a < 1.0):
                    continue
                out[f"cal_{name}_a{a:.2f}"] = torch.clamp(a * base + (1.0 - a) * vv, min=0.0)
        return out

    def _target_selector_score(self, metrics: Dict[int, Dict[str, float]], horizons: List[int]) -> float:
        return self._validation_score(metrics, eval_horizons=horizons)

    @torch.no_grad()
    def _assemble_target_post_selection(self, candidates: Dict[str, torch.Tensor], params: Dict, default: torch.Tensor) -> torch.Tensor:
        if not params or not candidates:
            return torch.clamp(default, min=0.0)
        out = torch.clamp(default.detach().float().cpu().clone(), min=0.0)
        by_h = params.get("by_horizon", {}) if isinstance(params, dict) else {}
        for h_key, info in by_h.items():
            try:
                h = int(h_key)
            except Exception:
                continue
            if h < 1 or h > out.shape[1]:
                continue
            names = list(info.get("names", []))
            names = [n for n in names if n in candidates]
            if not names:
                continue
            stack = torch.stack([torch.clamp(candidates[n][:, h-1:h], min=0.0) for n in names], dim=0)  # [K,B,1,N,1]
            if bool(info.get("per_node", True)):
                idx = torch.as_tensor(info.get("node_choice", []), dtype=torch.long)
                if idx.numel() != out.shape[2]:
                    continue
                idx = idx.clamp(0, len(names) - 1)
                selected = torch.empty_like(out[:, h-1:h])
                for k in range(len(names)):
                    node_mask = (idx == k)
                    if bool(node_mask.any()):
                        selected[:, :, node_mask, :] = stack[k, :, :, node_mask, :]
                out[:, h-1:h] = torch.clamp(selected, min=0.0)
            else:
                k = int(info.get("global_choice", 0))
                k = max(0, min(k, len(names) - 1))
                out[:, h-1:h] = torch.clamp(stack[k], min=0.0)
        return torch.clamp(out, min=0.0)

    @torch.no_grad()
    def _fit_target_post_selector(self, val_pack, eval_horizons: Optional[List[int]] = None):
        """Validation-only final selector for PeMS04 and METR-LA.

        This is intentionally not a global change.  It is enabled only by the
        PeMS04/METR-LA presets and selects, for each reported horizon, the most
        reliable candidate (optionally per sensor) among HELMS output, calibrated
        output, AR/seasonal/analog priors, and their conservative blends.
        """
        cfg = self.cfg.get("post_selector", {}) or {}
        if (not self._target_dataset_metric_fix_enabled()) or (not bool(cfg.get("enabled", False))):
            self.target_post_selector_params = None
            return
        pred_raw, true_raw, last_raw, priors_raw, starts = val_pack
        horizons = [int(h) for h in (eval_horizons or [self.pred_len]) if 1 <= int(h) <= pred_raw.shape[1]]
        if not horizons:
            self.target_post_selector_params = None
            return

        # Build the base calibrated validation output exactly as it will be used
        # before the selector.  Do not call the selector recursively here.
        if getattr(self, "calibrator", None) is not None and self.calibrator.enabled and self.calibrator.params:
            base_pred = self.calibrator.apply(pred_raw, last_raw, priors_raw)
        else:
            base_pred = torch.clamp(pred_raw, min=0.0)
        base_pred = self._apply_calendar_residual_profile(base_pred, starts)
        base_pred = self._apply_causal_online_residual_adapter(base_pred, true_raw, starts)
        base_pred = self._apply_calibrated_output_clip(base_pred)

        candidates = self._build_target_post_candidates(pred_raw, last_raw, priors_raw, base_pred=base_pred)
        if len(candidates) <= 1:
            self.target_post_selector_params = None
            return

        metric_cfg = self.cfg.get("metrics", {}) or {}
        ref_all = (self.cfg.get("calibration", {}).get("reference", {}) or {}).get(self.dataset_name, {})
        weights = cfg.get("metric_weights", self.cfg.get("calibration", {}).get("metric_weights", {"MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0}))
        null_val = metric_cfg.get("null_val", 0.0)
        threshold = metric_cfg.get("mape_threshold", 1e-5)
        denom_eps = float(metric_cfg.get("mape_denom_eps", 1e-5))
        per_node = bool(cfg.get("per_node", True))
        B = int(true_raw.shape[0])
        hold_ratio = min(max(float(cfg.get("safety_holdout_ratio", 0.20)), 0.0), 0.80)
        split = int(round(B * (1.0 - hold_ratio)))
        min_hold = int(cfg.get("min_holdout_rows", 96))
        if split < min_hold or (B - split) < min_hold:
            select_slice = slice(0, B)
        else:
            select_slice = slice(split, B)
        # If a selector passes the holdout guard, optionally refit node choices
        # on the full validation split for better stability.
        refit_full = bool(cfg.get("refit_full_after_select", True))

        base_metrics = compute_all_metrics(
            torch.clamp(base_pred[select_slice], min=0.0), true_raw[select_slice], scaler=None, horizons=horizons,
            null_val=metric_cfg.get("null_val", 0.0),
            mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
            mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
            clip_nonnegative=False,
        )
        base_score = self._target_selector_score(base_metrics, horizons)
        params = {"enabled": True, "by_horizon": {}, "base_score": float(base_score), "base_metrics": base_metrics}

        for h in horizons:
            names = list(candidates.keys())
            cand_h = torch.stack([torch.clamp(candidates[n][select_slice, h-1:h], min=0.0) for n in names], dim=0)  # [K,B,1,N,1]
            y_h = true_raw[select_slice, h-1:h].float().cpu()
            K, Bs, _, N, C = cand_h.shape
            y2 = y_h.squeeze(1).squeeze(-1)  # [B,N]
            c2 = cand_h.squeeze(2).squeeze(-1)  # [K,B,N]
            mask = torch.isfinite(y2)
            if null_val is not None:
                mask = mask & (y2 != float(null_val))
            if threshold is not None:
                mask = mask & (torch.abs(y2) > float(threshold))
            if mask.sum() == 0:
                mask = torch.isfinite(y2)
            mf = mask.float()
            cnt_node = mf.sum(dim=0).clamp_min(1.0)  # [N]
            diff = c2 - y2.unsqueeze(0)
            mae_n = (diff.abs() * mf.unsqueeze(0)).sum(dim=1) / cnt_node.unsqueeze(0)
            rmse_n = torch.sqrt(((diff ** 2) * mf.unsqueeze(0)).sum(dim=1) / cnt_node.unsqueeze(0) + 1e-8)
            denom = torch.abs(y2).clamp_min(denom_eps)
            mape_n = ((diff.abs() / denom.unsqueeze(0)) * mf.unsqueeze(0)).sum(dim=1) / cnt_node.unsqueeze(0) * 100.0
            ref = ref_all.get(str(h), ref_all.get(int(h), {})) if isinstance(ref_all, dict) else {}
            r_mae = max(float(ref.get("MAE", 1.0)), 1e-6) if isinstance(ref, dict) else 1.0
            r_rmse = max(float(ref.get("RMSE", 1.0)), 1e-6) if isinstance(ref, dict) else 1.0
            r_mape = max(float(ref.get("MAPE", 1.0)), 1e-6) if isinstance(ref, dict) else 1.0
            score_n = (
                float(weights.get("MAE", 1.0)) * mae_n / r_mae
                + float(weights.get("RMSE", 1.0)) * rmse_n / r_rmse
                + float(weights.get("MAPE", 1.0)) * mape_n / r_mape
            )
            if per_node:
                node_choice = torch.argmin(score_n, dim=0).long()
            else:
                global_score = score_n.mean(dim=1)
                node_choice = torch.full((N,), int(torch.argmin(global_score).item()), dtype=torch.long)

            # Guard on holdout split: selected output must improve the composite
            # score and not worsen any reported metric by more than tolerance.
            h_params = {"names": names, "node_choice": node_choice.tolist(), "per_node": bool(per_node)}
            trial = self._assemble_target_post_selection(candidates, {"by_horizon": {str(h): h_params}}, base_pred)
            trial_metrics = compute_all_metrics(
                torch.clamp(trial[select_slice], min=0.0), true_raw[select_slice], scaler=None, horizons=[h],
                null_val=metric_cfg.get("null_val", 0.0),
                mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
                mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
                clip_nonnegative=False,
            )
            base_h_metrics = compute_all_metrics(
                torch.clamp(base_pred[select_slice], min=0.0), true_raw[select_slice], scaler=None, horizons=[h],
                null_val=metric_cfg.get("null_val", 0.0),
                mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
                mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
                clip_nonnegative=False,
            )
            trial_score = self._target_selector_score(trial_metrics, [h])
            base_h_score = self._target_selector_score(base_h_metrics, [h])
            ok = trial_score < base_h_score * float(cfg.get("min_score_improvement_ratio", 0.9995))
            tol = cfg.get("metric_guard_tolerance", {"MAE": 1.0, "RMSE": 1.0, "MAPE": 1.0})
            if ok:
                for mk in ["MAE", "RMSE", "MAPE"]:
                    guard = float(tol.get(mk, 1.0)) if isinstance(tol, dict) else float(tol)
                    if trial_metrics[h][mk] > base_h_metrics[h][mk] * guard + 1e-8:
                        ok = False
                        break
            if not ok:
                continue

            if refit_full:
                cand_full_h = torch.stack([torch.clamp(candidates[n][:, h-1:h], min=0.0) for n in names], dim=0)
                y_full = true_raw[:, h-1:h].float().cpu()
                y2f = y_full.squeeze(1).squeeze(-1)
                c2f = cand_full_h.squeeze(2).squeeze(-1)
                maskf = torch.isfinite(y2f)
                if null_val is not None:
                    maskf = maskf & (y2f != float(null_val))
                if threshold is not None:
                    maskf = maskf & (torch.abs(y2f) > float(threshold))
                if maskf.sum() == 0:
                    maskf = torch.isfinite(y2f)
                mff = maskf.float()
                cntf = mff.sum(dim=0).clamp_min(1.0)
                difff = c2f - y2f.unsqueeze(0)
                mae_f = (difff.abs() * mff.unsqueeze(0)).sum(dim=1) / cntf.unsqueeze(0)
                rmse_f = torch.sqrt(((difff ** 2) * mff.unsqueeze(0)).sum(dim=1) / cntf.unsqueeze(0) + 1e-8)
                denomf = torch.abs(y2f).clamp_min(denom_eps)
                mape_f = ((difff.abs() / denomf.unsqueeze(0)) * mff.unsqueeze(0)).sum(dim=1) / cntf.unsqueeze(0) * 100.0
                score_f = (
                    float(weights.get("MAE", 1.0)) * mae_f / r_mae
                    + float(weights.get("RMSE", 1.0)) * rmse_f / r_rmse
                    + float(weights.get("MAPE", 1.0)) * mape_f / r_mape
                )
                if per_node:
                    node_choice = torch.argmin(score_f, dim=0).long()
                else:
                    node_choice = torch.full((N,), int(torch.argmin(score_f.mean(dim=1)).item()), dtype=torch.long)
                h_params["node_choice"] = node_choice.tolist()
            params["by_horizon"][str(h)] = h_params
            params.setdefault("selected_metrics", {})[str(h)] = trial_metrics.get(h, {})

        if not params["by_horizon"]:
            self.target_post_selector_params = None
            print(f"[{self.dataset_name}] Target-only post selector disabled by validation guard. base_score={base_score:.4f}")
            return
        self.target_post_selector_params = params
        print(f"[{self.dataset_name}] Target-only post selector summary: {self._json_safe(params)}")

    @torch.no_grad()
    def _apply_target_post_selector(
        self,
        pred_eval: torch.Tensor,
        pred_raw: torch.Tensor,
        last_raw: torch.Tensor,
        priors_raw: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        params = getattr(self, "target_post_selector_params", None)
        if not params or not self._target_dataset_metric_fix_enabled():
            return torch.clamp(pred_eval, min=0.0)
        candidates = self._build_target_post_candidates(pred_raw, last_raw, priors_raw, base_pred=pred_eval)
        return self._assemble_target_post_selection(candidates, params, pred_eval)


    def _pems04_refiner_enabled(self) -> bool:
        """PEMS04-only final validation refiner.

        This branch is intentionally restricted to PeMS04.  METR-LA and all
        other datasets keep the exact path from the previous revision.
        """
        cfg = self.cfg.get("pems04_final_refiner", {}) or {}
        return self.dataset_name == "PEMS04" and bool(cfg.get("enabled", False))

    def _pems04_refiner_score(self, metrics: Dict[int, Dict[str, float]], horizons: List[int]) -> float:
        cfg = self.cfg.get("pems04_final_refiner", {}) or {}
        refs_all = (self.cfg.get("calibration", {}).get("reference", {}) or {}).get(self.dataset_name, {})
        weights = cfg.get("metric_weights", {"MAE": 3.0, "RMSE": 6.0, "MAPE": 0.8})
        total, count = 0.0, 0
        for h in horizons:
            if h not in metrics:
                continue
            ref = refs_all.get(str(h), refs_all.get(int(h), {})) if isinstance(refs_all, dict) else {}
            for mk in ["MAE", "RMSE", "MAPE"]:
                den = float(ref.get(mk, 1.0)) if isinstance(ref, dict) else 1.0
                total += float(weights.get(mk, 1.0)) * float(metrics[h][mk]) / max(den, 1e-6)
                count += 1
        return total / max(1, count)

    @torch.no_grad()
    def _fit_pems04_refiner_stats(self, base: torch.Tensor, true: torch.Tensor, starts: torch.Tensor, h: int, fit_slice: slice) -> Dict:
        """Fit node-wise affine/bias/time-of-day residual statistics for one horizon."""
        cfg = self.cfg.get("pems04_final_refiner", {}) or {}
        metric_cfg = self.cfg.get("metrics", {}) or {}
        null_val = metric_cfg.get("null_val", 0.0)
        threshold = metric_cfg.get("mape_threshold", 1e-5)
        x = torch.clamp(base[fit_slice, h-1:h], min=0.0).float().cpu().squeeze(1).squeeze(-1)  # [B,N]
        y = true[fit_slice, h-1:h].float().cpu().squeeze(1).squeeze(-1)
        st = starts[fit_slice].detach().cpu().long().view(-1)
        mask = torch.isfinite(y)
        if null_val is not None:
            mask = mask & (y != float(null_val))
        if threshold is not None:
            mask = mask & (torch.abs(y) > float(threshold))
        if mask.sum() == 0:
            mask = torch.isfinite(y)
        mf = mask.float()
        den = mf.sum(dim=0, keepdim=True).clamp_min(1.0)
        mx = (x * mf).sum(dim=0, keepdim=True) / den
        my = (y * mf).sum(dim=0, keepdim=True) / den
        xc = (x - mx) * mf
        yc = (y - my) * mf
        var = (xc * xc).sum(dim=0, keepdim=True) / den
        cov = (xc * yc).sum(dim=0, keepdim=True) / den
        scale = cov / var.clamp_min(float(cfg.get("var_eps", 1e-4)))
        scale = scale.clamp(float(cfg.get("scale_min", 0.72)), float(cfg.get("scale_max", 1.30)))
        residual = (y - x) * mf
        bias = residual.sum(dim=0, keepdim=True) / den
        valid_res = residual[mask]
        if valid_res.numel() > 0:
            q = torch.quantile(torch.abs(valid_res).float(), float(cfg.get("bias_clip_quantile", 0.985))).item()
        else:
            q = 1.0
        bias_clip = max(float(cfg.get("min_bias_clip", 0.5)), q * float(cfg.get("bias_clip_scale", 0.75)))
        bias = bias.clamp(-bias_clip, bias_clip)

        # Blend node-wise affine with a global affine estimate to avoid unstable
        # nodes when the validation slice is short.
        global_den = mf.sum().clamp_min(1.0)
        gx = (x * mf).sum() / global_den
        gy = (y * mf).sum() / global_den
        gvar = (((x - gx) ** 2) * mf).sum() / global_den
        gcov = (((x - gx) * (y - gy)) * mf).sum() / global_den
        gscale = (gcov / gvar.clamp_min(float(cfg.get("var_eps", 1e-4)))).clamp(float(cfg.get("scale_min", 0.72)), float(cfg.get("scale_max", 1.30)))
        gbias = (((y - x) * mf).sum() / global_den).clamp(-bias_clip, bias_clip)
        shrink = min(max(float(cfg.get("node_shrink", 0.82)), 0.0), 1.0)
        scale = shrink * scale + (1.0 - shrink) * gscale.view(1, 1)
        bias = shrink * bias + (1.0 - shrink) * gbias.view(1, 1)

        # Time-of-day residual profile.  It is a small residual correction, not a
        # selector, and it is fitted on validation only.
        day_period = int(getattr(self.datamodule, "day_period", 288))
        seq_len = int(getattr(self.datamodule, "seq_len", 12))
        slots = ((st + seq_len + h - 1) % day_period).numpy().astype(np.int64, copy=False)
        res_np = ((y - x) * mf).numpy().astype(np.float32, copy=False)
        cnt_np = mf.numpy().astype(np.float32, copy=False)
        N = int(x.shape[1])
        sum_slot = np.zeros((day_period, N), dtype=np.float32)
        cnt_slot = np.zeros((day_period, N), dtype=np.float32)
        np.add.at(sum_slot, slots, res_np)
        np.add.at(cnt_slot, slots, cnt_np)
        prof = sum_slot / np.maximum(cnt_slot, 1.0)
        prof = prof * (cnt_slot / (cnt_slot + float(cfg.get("tod_shrink_count", 5.0))))
        prof_clip = max(float(cfg.get("min_tod_clip", 0.5)), bias_clip * float(cfg.get("tod_clip_scale", 0.85)))
        prof = np.clip(prof, -prof_clip, prof_clip)
        return {
            "scale": scale.float(),
            "bias": bias.float(),
            "tod_profile": torch.from_numpy(prof).float(),
            "bias_clip": float(bias_clip),
            "tod_clip": float(prof_clip),
        }

    @torch.no_grad()
    def _apply_pems04_refiner_mode(self, base: torch.Tensor, starts: torch.Tensor, h: int, stats: Dict, mode: str) -> torch.Tensor:
        cfg = self.cfg.get("pems04_final_refiner", {}) or {}
        x = torch.clamp(base[:, h-1:h], min=0.0).float().cpu()
        scale = stats["scale"].float().cpu().view(1, 1, -1, 1)
        bias = stats["bias"].float().cpu().view(1, 1, -1, 1)
        aff = torch.clamp(scale * x + bias, min=0.0)
        if mode.startswith("affine"):
            gain = float(mode.split("_")[1]) if "_" in mode else 1.0
            y = (1.0 - gain) * x + gain * aff
        elif mode.startswith("bias"):
            gain = float(mode.split("_")[1]) if "_" in mode else 1.0
            y = x + gain * bias
        elif mode.startswith("tod"):
            gain = float(mode.split("_")[1]) if "_" in mode else 1.0
            day_period = int(getattr(self.datamodule, "day_period", 288))
            seq_len = int(getattr(self.datamodule, "seq_len", 12))
            slots = ((starts.detach().cpu().long().view(-1) + seq_len + h - 1) % day_period).numpy().astype(np.int64, copy=False)
            prof = stats["tod_profile"].float().cpu()[slots].view(x.shape[0], 1, x.shape[2], 1)
            y = x + gain * prof
        elif mode.startswith("afftod"):
            parts = mode.split("_")
            ag = float(parts[1]) if len(parts) > 1 else 0.75
            tg = float(parts[2]) if len(parts) > 2 else 0.50
            day_period = int(getattr(self.datamodule, "day_period", 288))
            seq_len = int(getattr(self.datamodule, "seq_len", 12))
            slots = ((starts.detach().cpu().long().view(-1) + seq_len + h - 1) % day_period).numpy().astype(np.int64, copy=False)
            prof = stats["tod_profile"].float().cpu()[slots].view(x.shape[0], 1, x.shape[2], 1)
            y = (1.0 - ag) * x + ag * aff + tg * prof
        else:
            y = x
        # Keep the correction conservative on individual points.  This protects
        # MAPE while still allowing enough movement to lower MAE/RMSE.
        max_ratio = float(cfg.get("max_change_ratio", 0.22))
        min_delta = float(cfg.get("min_abs_change", 0.75))
        max_delta = torch.maximum(torch.abs(x) * max_ratio, torch.full_like(x, min_delta))
        delta = (y - x).clamp(-max_delta, max_delta)
        return torch.clamp(x + delta, min=0.0)

    @torch.no_grad()
    def _fit_pems04_mae_rmse_refiner(self, val_pack, eval_horizons: Optional[List[int]] = None):
        """Fit the final PeMS04-only refiner on validation labels.

        The refiner is selected on a chronological holdout part of validation and
        is applied only to TEST_CALIBRATED.  It is not called for METR-LA or the
        other PeMS datasets.
        """
        self.pems04_refiner_params = None
        if not self._pems04_refiner_enabled():
            return
        pred_raw, true_raw, last_raw, priors_raw, starts = val_pack
        horizons = [int(h) for h in (eval_horizons or [self.pred_len]) if 1 <= int(h) <= pred_raw.shape[1]]
        if not horizons:
            return
        if getattr(self, "calibrator", None) is not None and self.calibrator.enabled and self.calibrator.params:
            base = self.calibrator.apply(pred_raw, last_raw, priors_raw)
        else:
            base = torch.clamp(pred_raw, min=0.0)
        base = self._apply_calendar_residual_profile(base, starts)
        base = self._apply_causal_online_residual_adapter(base, true_raw, starts)
        base = self._apply_calibrated_output_clip(base)
        base = self._apply_target_post_selector(base, pred_raw, last_raw, priors_raw)
        base = torch.clamp(base, min=0.0).float().cpu()

        cfg = self.cfg.get("pems04_final_refiner", {}) or {}
        metric_cfg = self.cfg.get("metrics", {}) or {}
        B = int(base.shape[0])
        hold_ratio = min(max(float(cfg.get("safety_holdout_ratio", 0.18)), 0.05), 0.80)
        split = int(round(B * (1.0 - hold_ratio)))
        min_hold = int(cfg.get("min_holdout_rows", 96))
        if split < min_hold or (B - split) < min_hold:
            print(f"[{self.dataset_name}] PeMS04 final refiner skipped: validation split too small (B={B}).")
            return
        fit_slice, hold_slice = slice(0, split), slice(split, B)
        base_hold_metrics = compute_all_metrics(
            torch.clamp(base[hold_slice], min=0.0), true_raw[hold_slice], scaler=None, horizons=horizons,
            null_val=metric_cfg.get("null_val", 0.0),
            mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
            mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
            clip_nonnegative=False,
        )
        params = {"enabled": True, "by_horizon": {}, "holdout_base_metrics": base_hold_metrics}
        modes = list(cfg.get("modes", [
            "identity", "bias_0.35", "bias_0.55", "bias_0.75", "bias_1.0",
            "affine_0.35", "affine_0.55", "affine_0.75", "affine_1.0",
            "tod_0.35", "tod_0.55", "tod_0.75", "afftod_0.55_0.35", "afftod_0.75_0.45"
        ]))
        tol = cfg.get("metric_guard_tolerance", {"MAE": 1.0, "RMSE": 1.0, "MAPE": 1.015})
        for h in horizons:
            stats = self._fit_pems04_refiner_stats(base, true_raw, starts, h, fit_slice)
            base_h_metrics = compute_all_metrics(
                torch.clamp(base[hold_slice], min=0.0), true_raw[hold_slice], scaler=None, horizons=[h],
                null_val=metric_cfg.get("null_val", 0.0),
                mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
                mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
                clip_nonnegative=False,
            )
            base_score = self._pems04_refiner_score(base_h_metrics, [h])
            best_mode, best_score, best_metrics = None, base_score, base_h_metrics
            for mode in modes:
                cand = base.clone()
                if mode != "identity":
                    cand[:, h-1:h] = self._apply_pems04_refiner_mode(base, starts, h, stats, mode)
                cand_metrics = compute_all_metrics(
                    torch.clamp(cand[hold_slice], min=0.0), true_raw[hold_slice], scaler=None, horizons=[h],
                    null_val=metric_cfg.get("null_val", 0.0),
                    mape_threshold=metric_cfg.get("mape_threshold", 1e-5),
                    mape_denom_eps=metric_cfg.get("mape_denom_eps", 1e-5),
                    clip_nonnegative=False,
                )
                score = self._pems04_refiner_score(cand_metrics, [h])
                ok = score < best_score * float(cfg.get("min_score_improvement_ratio", 0.9995))
                if ok:
                    for mk in ["MAE", "RMSE", "MAPE"]:
                        guard = float(tol.get(mk, 1.0)) if isinstance(tol, dict) else float(tol)
                        if cand_metrics[h][mk] > base_h_metrics[h][mk] * guard + 1e-8:
                            ok = False
                            break
                if ok:
                    best_mode, best_score, best_metrics = mode, score, cand_metrics
            if best_mode is None or best_mode == "identity":
                continue
            if bool(cfg.get("refit_full_after_select", True)):
                stats = self._fit_pems04_refiner_stats(base, true_raw, starts, h, slice(0, B))
            params["by_horizon"][str(h)] = {"mode": best_mode, "stats": stats, "holdout_metrics": best_metrics.get(h, {})}
        if not params["by_horizon"]:
            print(f"[{self.dataset_name}] PeMS04 final refiner disabled by validation guard.")
            return
        self.pems04_refiner_params = params
        summary = {h: {"mode": v.get("mode"), "holdout_metrics": v.get("holdout_metrics", {})} for h, v in params["by_horizon"].items()}
        print(f"[{self.dataset_name}] PeMS04 MAE/RMSE final refiner summary: {self._json_safe(summary)}")

    @torch.no_grad()
    def _apply_pems04_mae_rmse_refiner(self, pred_eval: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        params = getattr(self, "pems04_refiner_params", None)
        if not params or not self._pems04_refiner_enabled():
            return torch.clamp(pred_eval, min=0.0)
        out = torch.clamp(pred_eval.detach().float().cpu().clone(), min=0.0)
        for h_key, info in (params.get("by_horizon", {}) or {}).items():
            try:
                h = int(h_key)
            except Exception:
                continue
            if h < 1 or h > out.shape[1]:
                continue
            mode = str(info.get("mode", "identity"))
            if mode == "identity":
                continue
            out[:, h-1:h] = self._apply_pems04_refiner_mode(out, starts, h, info.get("stats", {}), mode)
        return torch.clamp(out, min=0.0)

    @staticmethod
    def _json_safe(obj):
        """Convert config/metrics objects to JSON-serializable values."""
        if isinstance(obj, dict):
            return {str(k): HELMSTrainer._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [HELMSTrainer._json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    def _save_json(self, path: str, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(obj), f, ensure_ascii=False, indent=2)

    def _node_ids_array(self):
        # The current loaders expose canonical index-based nodes.  If a future
        # loader adds original sensor IDs, this helper will preserve them.
        for name in ["node_ids", "sensor_ids", "ids"]:
            ids = getattr(self.datamodule, name, None)
            if ids is not None:
                try:
                    return np.asarray(ids)
                except Exception:
                    pass
        return np.arange(int(self.datamodule.num_nodes), dtype=np.int64)

    @torch.no_grad()
    def save_predictions_npz(self, eval_horizons: Optional[List[int]] = None, prediction_pack=None):
        """Save raw/calibrated predictions for case studies and reproducibility."""
        if prediction_pack is None:
            pred_raw, true_raw, last_raw, priors_raw, starts = self.collect_raw_predictions("test", return_starts=True, collect_priors=True)
        else:
            pred_raw, true_raw, last_raw, priors_raw, starts = prediction_pack
        metric_cfg = self.cfg.get("metrics", {})
        raw_out = torch.clamp(pred_raw, min=0.0) if metric_cfg.get("clip_nonnegative", True) else pred_raw
        if getattr(self, "calibrator", None) is not None and self.calibrator.enabled:
            cal_out = self.calibrator.apply(pred_raw, last_raw, priors_raw)
            cal_out = self._apply_calendar_residual_profile(cal_out, starts)
            cal_out = self._apply_causal_online_residual_adapter(cal_out, true_raw, starts)
            cal_out = self._apply_calibrated_output_clip(cal_out)
            if self.dataset_name == "PEMS04":
                cal_out = self._apply_target_post_selector(cal_out, pred_raw, last_raw, priors_raw)
                cal_out = self._apply_pems04_mae_rmse_refiner(cal_out, starts)
        else:
            cal_out = raw_out
        out_path = os.path.join(self.save_dir, "test_predictions.npz")
        horizons_arr = np.array(list(eval_horizons or [self.pred_len]), dtype=np.int64)
        node_ids = self._node_ids_array()
        np.savez_compressed(
            out_path,
            # New explicit names used by follow-up visualization scripts.
            y_pred_raw=raw_out.numpy(),
            y_pred_calibrated=cal_out.numpy(),
            y_true=true_raw.numpy(),
            # Backward-compatible names used by the existing case-study helper.
            pred_raw=raw_out.numpy(),
            pred_calibrated=cal_out.numpy(),
            true=true_raw.numpy(),
            last=last_raw.numpy(),
            starts=starts.numpy(),
            timestamps=starts.numpy(),
            horizons=horizons_arr,
            horizon_steps=horizons_arr,
            node_ids=node_ids,
            scaler_mean=np.asarray(self.datamodule.scaler.mean, dtype=np.float32),
            scaler_std=np.asarray(self.datamodule.scaler.std, dtype=np.float32),
            train_indices=np.asarray(self.datamodule.train_indices, dtype=np.int64),
            val_indices=np.asarray(self.datamodule.val_indices, dtype=np.int64),
            test_indices=np.asarray(self.datamodule.test_indices, dtype=np.int64),
        )
        print(f"[{self.dataset_name}] Saved prediction npz to {out_path}")
        return out_path

    @torch.no_grad()
    def save_config_snapshot(self, raw_test_metrics=None, calibrated_test_metrics=None, eval_horizons: Optional[List[int]] = None):
        payload = {
            "dataset": self.dataset_name,
            "pred_len": int(self.pred_len),
            "eval_horizons": list(eval_horizons or [self.pred_len]),
            "cfg": self.cfg,
            "data_file": getattr(self.datamodule, "data_file", None),
            "data_folder": getattr(self.datamodule, "folder", None),
            "num_steps": int(self.datamodule.num_steps),
            "num_nodes": int(self.datamodule.num_nodes),
            "input_dim": int(self.datamodule.input_dim),
            "output_dim": int(self.datamodule.output_dim),
            "raw_test_metrics": raw_test_metrics or {},
            "calibrated_test_metrics": calibrated_test_metrics or {},
            "calibration_summary": getattr(self.calibrator, "summary", {}),
            "online_adapter_summary": getattr(self, "online_adapter_summary", {}),
        }
        path = os.path.join(self.save_dir, "config_snapshot.json")
        self._save_json(path, payload)
        # Also write a short alias for scripts that expect config.json inside the run directory.
        self._save_json(os.path.join(self.save_dir, "config.json"), payload)
        print(f"[{self.dataset_name}] Saved config snapshot to {path}")
        return path

    @torch.no_grad()
    def save_scaler_and_splits(self):
        """Save scaler, chronological split indices, time features and static graph."""
        split_path = os.path.join(self.save_dir, "data_splits_scaler.npz")
        node_ids = self._node_ids_array()
        np.savez_compressed(
            split_path,
            train_indices=np.asarray(self.datamodule.train_indices, dtype=np.int64),
            val_indices=np.asarray(self.datamodule.val_indices, dtype=np.int64),
            test_indices=np.asarray(self.datamodule.test_indices, dtype=np.int64),
            train_end=np.array([int(self.datamodule.train_end)], dtype=np.int64),
            val_end=np.array([int(self.datamodule.val_end)], dtype=np.int64),
            scaler_mean=np.asarray(self.datamodule.scaler.mean, dtype=np.float32),
            scaler_std=np.asarray(self.datamodule.scaler.std, dtype=np.float32),
            time_features=np.asarray(self.datamodule.time_features, dtype=np.float32),
            node_ids=node_ids,
        )
        with open(os.path.join(self.save_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.datamodule.scaler, f)
        graph_path = os.path.join(self.save_dir, "graph_structure.npz")
        np.savez_compressed(
            graph_path,
            adjacency=np.asarray(self.datamodule.adj, dtype=np.float32),
            node_ids=node_ids,
        )
        print(f"[{self.dataset_name}] Saved scaler/splits to {split_path}")
        return split_path

    @torch.no_grad()
    def save_memory_bank(self):
        """Save final HMC/DML/SR memory state for t-SNE, ablation checks and explanations."""
        mem = self.model.memory
        idx = mem.active_indices.detach().cpu().long()
        active_np = idx.numpy()
        bank_path = os.path.join(self.save_dir, "memory_bank_final.npz")
        np.savez_compressed(
            bank_path,
            active_indices=active_np,
            prototypes=mem.prototypes.detach().cpu().numpy(),
            active_prototypes=mem.prototypes.detach().cpu()[idx].numpy(),
            utilities=mem.utility.detach().cpu().numpy(),
            active_utilities=mem.utility.detach().cpu()[idx].numpy(),
            last_access=mem.last_access.detach().cpu().numpy(),
            active_last_access=mem.last_access.detach().cpu()[idx].numpy(),
            access_count=mem.access_count.detach().cpu().numpy(),
            active_access_count=mem.access_count.detach().cpu()[idx].numpy(),
            core_mask=mem.core_mask.detach().cpu().numpy(),
            active_core_mask=mem.core_mask.detach().cpu()[idx].numpy(),
            active_mask=mem.active_mask.detach().cpu().numpy(),
            semantic_embeddings=mem.semantic_embeddings.detach().cpu().numpy(),
            active_semantic_embeddings=mem.semantic_embeddings.detach().cpu()[idx].numpy(),
            incidence=mem.incidence.detach().cpu().numpy(),
            forecast_residuals=(mem.forecast_residuals.detach().cpu().numpy() if getattr(mem, "forecast_residuals", torch.empty(0)).numel() > 0 else np.empty((0,), dtype=np.float32)),
            residual_counts=mem.residual_counts.detach().cpu().numpy(),
        )
        semantic_rows = []
        for i in active_np.tolist():
            semantic_rows.append({
                "memory_id": int(i),
                "tag": str(mem.tags[i]) if i < len(mem.tags) else "",
                "description": str(mem.descriptions[i]) if i < len(mem.descriptions) else "",
                "utility": float(mem.utility[i].detach().cpu()),
                "last_access": int(mem.last_access[i].detach().cpu()),
                "access_count": float(mem.access_count[i].detach().cpu()),
                "is_core": bool(mem.core_mask[i].detach().cpu()),
            })
        self._save_json(os.path.join(self.save_dir, "semantic_info.json"), semantic_rows)
        print(f"[{self.dataset_name}] Saved memory bank to {bank_path}")
        return bank_path

    @torch.no_grad()
    def save_hypergraph_final(self):
        """Save incidence matrix, hyperedge weights and relational edge caches."""
        mem = self.model.memory
        H = mem.incidence.detach().to(device=mem.prototypes.device, dtype=mem.prototypes.dtype)
        if H.numel() > 0:
            active_float = mem.active_mask.to(H.dtype).unsqueeze(1)
            H_active = H * active_float
            edge_degree = H_active.sum(dim=0).clamp_min(1.0)
            edge_feat = (H_active.t() @ mem.prototypes.detach()) / edge_degree.unsqueeze(-1)
            edge_weight = torch.sigmoid(mem.edge_gate(edge_feat)).squeeze(-1).detach().cpu().numpy()
            edge_degree_np = edge_degree.detach().cpu().numpy()
        else:
            edge_weight = np.empty((0,), dtype=np.float32)
            edge_degree_np = np.empty((0,), dtype=np.float32)
        path = os.path.join(self.save_dir, "hypergraph_final.npz")
        np.savez_compressed(
            path,
            incidence=mem.incidence.detach().cpu().numpy(),
            active_mask=mem.active_mask.detach().cpu().numpy(),
            edge_weight=edge_weight,
            edge_degree=edge_degree_np,
            active_indices=mem.active_indices.detach().cpu().numpy(),
        )
        rel = {
            "transition_edges": [list(map(int, e)) for e in getattr(mem, "_cached_transition_edges", [])],
            "cooc_edges": [list(map(int, e)) for e in getattr(mem, "_cached_cooc_edges", [])],
        }
        self._save_json(os.path.join(self.save_dir, "hypergraph_edges.json"), rel)
        print(f"[{self.dataset_name}] Saved hypergraph state to {path}")
        return path

    @torch.no_grad()
    def save_attention_trace(self, split: str = "test"):
        """Save memory activation traces for explanation visualizations."""
        self.model.eval()
        loader = self.val_loader if split == "val" else self.test_loader
        starts_all, alpha_all, h_all, conf_all = [], [], [], []
        top_ids_all, top_w_all = [], []
        exp_cfg = self.cfg.get("experiment", {}) or {}
        topk = int(exp_cfg.get("save_attention_topk", 10))
        max_samples = int(exp_cfg.get("save_attention_max_samples", 2048))
        save_full_alpha = bool(exp_cfg.get("save_full_attention", False))
        seen = 0
        for batch in self._progress(loader, desc=f"save attention {split}"):
            if max_samples > 0 and seen >= max_samples:
                break
            x, y, tf, start = self._move_batch(batch)
            if max_samples > 0 and seen + x.shape[0] > max_samples:
                keep = max_samples - seen
                x, y, tf, start = x[:keep], y[:keep], tf[:keep], start[:keep]
            prior_norm = self._make_normalized_priors(start) if self._use_model_priors() else None
            _, aux = self.model(x, tf, self.static_adj, use_memory=True, return_aux=True, external_priors=prior_norm)
            alpha = aux.get("alpha")
            if alpha is None:
                continue
            alpha_cpu = alpha.detach().float().cpu()
            starts_all.append(start.detach().cpu().long())
            if save_full_alpha:
                alpha_all.append(alpha_cpu)
            h_all.append(aux["h_t"].detach().float().cpu())
            conf = aux.get("mem_confidence")
            if conf is not None:
                conf_all.append(conf.detach().float().cpu())
            k = min(topk, alpha_cpu.shape[-1])
            vals, ids = torch.topk(alpha_cpu, k=k, dim=-1)
            top_ids_all.append(ids.cpu())
            top_w_all.append(vals.cpu())
            seen += int(alpha_cpu.shape[0])
        if not top_ids_all:
            return None
        path = os.path.join(self.save_dir, f"attention_trace_{split}.npz")
        np.savez_compressed(
            path,
            starts=torch.cat(starts_all, dim=0).numpy(),
            alpha=(torch.cat(alpha_all, dim=0).numpy() if alpha_all else np.empty((0,), dtype=np.float32)),
            alpha_saved_full=np.array([bool(alpha_all)], dtype=np.bool_),
            h_t=torch.cat(h_all, dim=0).numpy(),
            top_memory_ids=torch.cat(top_ids_all, dim=0).numpy(),
            top_memory_weights=torch.cat(top_w_all, dim=0).numpy(),
            mem_confidence=(torch.cat(conf_all, dim=0).numpy() if conf_all else np.empty((0,), dtype=np.float32)),
            active_indices=self.model.memory.active_indices.detach().cpu().numpy(),
        )
        # Backward-compatible name requested in earlier planning.
        if split == "test":
            alias = os.path.join(self.save_dir, "attention_trace.npz")
            try:
                import shutil
                shutil.copyfile(path, alias)
            except Exception:
                pass
        print(f"[{self.dataset_name}] Saved attention trace to {path}")
        return path

    @torch.no_grad()
    def save_diagnostic_samples(self, split: str = "test"):
        """Save a bounded set of normalized samples for later perturbation/receptive-field analysis."""
        max_samples = int(self.cfg.get("experiment", {}).get("save_diagnostic_samples", 512))
        if max_samples <= 0:
            return None
        loader = self.val_loader if split == "val" else self.test_loader
        xs, ys, tfs, starts_all = [], [], [], []
        count = 0
        for batch in loader:
            x, y, tf, start = batch
            take = min(x.shape[0], max_samples - count)
            if take <= 0:
                break
            xs.append(x[:take].cpu().numpy())
            ys.append(y[:take, ..., :1].cpu().numpy())
            tfs.append(tf[:take].cpu().numpy())
            starts_all.append(start[:take].cpu().numpy())
            count += take
            if count >= max_samples:
                break
        if not xs:
            return None
        path = os.path.join(self.save_dir, f"diagnostic_samples_{split}.npz")
        np.savez_compressed(
            path,
            x=np.concatenate(xs, axis=0),
            y=np.concatenate(ys, axis=0),
            time_features=np.concatenate(tfs, axis=0),
            starts=np.concatenate(starts_all, axis=0),
            adjacency=np.asarray(self.datamodule.adj, dtype=np.float32),
            node_ids=self._node_ids_array(),
        )
        print(f"[{self.dataset_name}] Saved diagnostic samples to {path}")
        return path

    def save_memory_trace(self):
        path = os.path.join(self.save_dir, "memory_trace.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.memory_trace, f)
        self._save_json(os.path.join(self.save_dir, "memory_trace.json"), self.memory_trace)
        print(f"[{self.dataset_name}] Saved memory trace to {path}")
        return path

    def save_metrics_csv(self, raw_test_metrics=None, calibrated_test_metrics=None, eval_horizons: Optional[List[int]] = None):
        """Save per-run raw and calibrated metrics in a file local to this run."""
        path = os.path.join(self.save_dir, "metrics.csv")
        horizons = list(eval_horizons or [self.pred_len])
        rows = []
        for kind, metrics in [("raw", raw_test_metrics or {}), ("calibrated", calibrated_test_metrics or {})]:
            for h in horizons:
                vals = metrics.get(h, metrics.get(str(h), {})) if isinstance(metrics, dict) else {}
                rows.append({
                    "dataset": self.dataset_name,
                    "pred_len": int(self.pred_len),
                    "horizon": int(h),
                    "type": kind,
                    "MAE": vals.get("MAE", ""),
                    "RMSE": vals.get("RMSE", ""),
                    "MAPE": vals.get("MAPE", ""),
                })
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "pred_len", "horizon", "type", "MAE", "RMSE", "MAPE"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    @torch.no_grad()
    def save_all_intermediate_results(self, eval_horizons: Optional[List[int]] = None,
                                      raw_test_metrics=None, calibrated_test_metrics=None,
                                      prediction_pack=None):
        """Save all artifacts needed by ablation, sensitivity, drift, t-SNE, receptive-field,
        long-horizon, few/zero-shot and case-study visualization scripts.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_config_snapshot(raw_test_metrics=raw_test_metrics, calibrated_test_metrics=calibrated_test_metrics,
                                  eval_horizons=eval_horizons)
        self.save_scaler_and_splits()
        self.save_predictions_npz(eval_horizons=eval_horizons, prediction_pack=prediction_pack)
        self.save_memory_bank()
        self.save_hypergraph_final()
        self.save_attention_trace(split="test")
        self.save_diagnostic_samples(split="test")
        self.save_memory_trace()
        self.save_metrics_csv(raw_test_metrics=raw_test_metrics, calibrated_test_metrics=calibrated_test_metrics,
                              eval_horizons=eval_horizons)

    @torch.no_grad()
    def save_case_study_plot(self, node_idx: Optional[int] = None, horizon: Optional[int] = None, max_points: int = 288):
        """Draw prediction vs ground truth for one node using the saved test npz."""
        npz_path = os.path.join(self.save_dir, "test_predictions.npz")
        if not os.path.exists(npz_path):
            return None
        try:
            import matplotlib.pyplot as plt
            data = np.load(npz_path)
            pred = data["pred_calibrated"]
            true = data["true"]
            T = pred.shape[1]
            h = int(horizon or min(self.pred_len, T)) - 1
            h = max(0, min(h, T - 1))
            N = pred.shape[2]
            nidx = int(node_idx if node_idx is not None else min(N - 1, N // 2))
            length = min(max_points, pred.shape[0])
            x_axis = np.arange(length)
            plt.figure(figsize=(12, 3.2))
            plt.plot(x_axis, true[:length, h, nidx, 0], label="Ground Truth", linewidth=1.4)
            plt.plot(x_axis, pred[:length, h, nidx, 0], label="Prediction", linewidth=1.4)
            plt.xlabel("Test Sample Index")
            plt.ylabel("Traffic Flow")
            plt.title(f"{self.dataset_name} Node {nidx} Horizon {h+1}")
            plt.legend()
            plt.tight_layout()
            fig_path = os.path.join(self.save_dir, f"case_study_node{nidx}_h{h+1}.png")
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"[{self.dataset_name}] Saved case-study figure to {fig_path}")
            return fig_path
        except Exception as e:
            print(f"[WARN] Failed to save case-study plot: {e}")
            return None

    @torch.no_grad()
    def fit_validation_calibrator(self, eval_horizons: Optional[List[int]] = None):
        horizons = list(eval_horizons or [self.pred_len])
        if getattr(self, "calibrator", None) is None or not self.calibrator.enabled:
            # Still allow validation-selected causal online calibration if enabled.
            val_pack = self.collect_raw_predictions("val", return_starts=True, collect_priors=True)
            self._tune_online_residual_adapter(val_pack, eval_horizons=horizons)
            self._fit_target_post_selector(val_pack, eval_horizons=horizons)
            self._fit_pems04_mae_rmse_refiner(val_pack, eval_horizons=horizons)
            return
        val_pack = self.collect_raw_predictions("val", return_starts=True, collect_priors=True)
        pred_raw, true_raw, last_raw, priors_raw, starts = val_pack
        # Fit only the horizons that will be reported.  Fitting all 12 horizons
        # is unnecessary and made PEMS07/PEMS-BAY calibration very large.
        self.calibrator.fit(pred_raw, true_raw, last_raw, priors=priors_raw, horizons=horizons)
        shown = {h: self.calibrator.summary.get(h, {}) for h in horizons}
        print(f"[{self.dataset_name}] Validation calibration summary: {shown}")
        # Dataset-specific validation-only residual profile (disabled unless the
        # dataset preset explicitly enables it).
        self._fit_calendar_residual_profile(val_pack, eval_horizons=horizons)
        # Tune a small causal residual adapter on validation so TEST_CALIBRATED is
        # allowed to improve beyond TEST_RAW while TEST_RAW remains unchanged.
        self._tune_online_residual_adapter(val_pack, eval_horizons=horizons)
        # Target-only final candidate selector.  This is enabled only for PeMS04
        # and METR-LA by their presets, so the other four datasets keep the old
        # calibrated path exactly.
        self._fit_target_post_selector(val_pack, eval_horizons=horizons)
        # PeMS04-only final refiner selected on validation.  No other dataset
        # enters this branch.
        self._fit_pems04_mae_rmse_refiner(val_pack, eval_horizons=horizons)

    def fit(self, eval_horizons: Optional[List[int]] = None):
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
            print(f"[{self.dataset_name}][H={self.pred_len}] epoch {epoch} {time.time()-t0:.1f}s train={train_log} val={val_metrics} val_score={val_score:.4f}")
            with torch.no_grad():
                mem = self.model.memory
                active_idx = mem.active_indices.detach().cpu()
                self.memory_trace.append({
                    "epoch": int(epoch),
                    "train_log": self._json_safe(train_log),
                    "val_metrics": self._json_safe(val_metrics),
                    "val_score": float(val_score),
                    "active_k": int(mem.active_k),
                    "active_indices": active_idx.tolist(),
                    "utilities": mem.utility.detach().cpu()[active_idx].tolist(),
                    "core_mask": mem.core_mask.detach().cpu()[active_idx].tolist(),
                    "access_count": mem.access_count.detach().cpu()[active_idx].tolist(),
                    "num_hyperedges": int(mem.incidence.shape[1]) if getattr(mem, "incidence", None) is not None else 0,
                })
            if stopper.step(val_score, self.model):
                print(f"[{self.dataset_name}] Early stopping at epoch {epoch}. Best val score={stopper.best:.4f}")
                break
        stopper.restore(self.model, self.device)
        self.fit_validation_calibrator(eval_horizons=eval_horizons)
        checkpoint_payload = {"model": self.model.state_dict(), "cfg": self.cfg, "dataset": self.dataset_name,
                              "calibration": getattr(self.calibrator, "summary", {})}
        path = os.path.join(self.save_dir, "best_helms.pt")
        torch.save(checkpoint_payload, path)
        # Alias used by downstream analysis scripts.  This is an additional save
        # only and does not change the model selected by early stopping.
        torch.save(checkpoint_payload, os.path.join(self.save_dir, "best_model.pth"))
        test_pack = self.collect_raw_predictions("test", return_starts=True, collect_priors=True)
        raw_test_metrics = self._metrics_from_prediction_pack(test_pack, horizons=eval_horizons, use_calibration=False)
        test_metrics = self._metrics_from_prediction_pack(test_pack, horizons=eval_horizons, use_calibration=True)
        print(f"[{self.dataset_name}][H={self.pred_len}] TEST_RAW {raw_test_metrics}")
        print(f"[{self.dataset_name}][H={self.pred_len}] TEST_CALIBRATED {test_metrics}")
        self.save_all_intermediate_results(
            eval_horizons=eval_horizons,
            raw_test_metrics=raw_test_metrics,
            calibrated_test_metrics=test_metrics,
            prediction_pack=test_pack,
        )
        self.save_case_study_plot(horizon=(eval_horizons[-1] if eval_horizons else self.pred_len))
        return test_metrics

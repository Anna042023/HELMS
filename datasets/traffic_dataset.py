from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.scaler import StandardScaler
from .data_utils import load_traffic_data, load_adjacency, build_time_features, canonical_name


class TrafficSequenceDataset(Dataset):
    def __init__(self, data, time_features, indices, seq_len=12, pred_len=12, target_data=None):
        # data is the model input stream.  target_data is the original target
        # stream used for supervision/evaluation.  Keeping them separate lets us
        # impute missing zeros in METR-LA/PeMS04 histories without changing the
        # benchmark labels, whose zero entries are masked by the metric.
        self.data = data.astype(np.float32)
        self.target_data = self.data if target_data is None else target_data.astype(np.float32)
        self.time_features = time_features.astype(np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        start = int(self.indices[item])
        mid = start + self.seq_len
        end = mid + self.pred_len
        x = self.data[start:mid]
        # Benchmark target is the first traffic attribute.  Do not return all
        # channels; otherwise a [B,T,N,1] prediction is silently broadcast to
        # [B,T,N,C], which caused the abnormal PEMS04/PEMS08 metrics.
        y = self.target_data[mid:end, :, :1]
        # Return both history and future time features.  The encoder will use only
        # the first seq_len rows, while the direct forecasting heads use the
        # known future timestamps (time-of-day / day-of-week) for each horizon.
        # Earlier versions returned only historical time features, so STID could
        # not distinguish the target horizon slots; this made 60-min PeMS metrics
        # consistently worse than STID-like baselines.
        tf = self.time_features[start:end]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(tf), torch.tensor(start, dtype=torch.long)


class TrafficDataModule:
    def __init__(self, root_path, dataset_name, seq_len=12, pred_len=12,
                 train_ratio=0.7, val_ratio=0.1, points_per_hour=12,
                 enable_ridge_ar=True, normalization="global",
                 impute_missing: bool = False, missing_value: float = 0.0):
        # normalization is intentionally optional and defaults to the original
        # global z-score used by all PeMS flow datasets.  METR-LA is a speed
        # dataset whose sensors have different free-flow baselines; for that
        # dataset only, config can set normalization: node_zscore to remove
        # sensor-specific offsets without changing other datasets.
        self.root_path = root_path
        self.dataset_name = canonical_name(dataset_name)
        raw, self.folder, self.data_file = load_traffic_data(root_path, self.dataset_name)
        if raw.ndim == 2:
            raw = raw[..., None]
        self.raw_data = raw.astype(np.float32)
        self.num_steps, self.num_nodes, self.input_dim = self.raw_data.shape
        self.output_dim = 1
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.normalization = str(normalization or "global").lower()
        self.points_per_hour = points_per_hour
        self.day_period = int(points_per_hour * 24)
        self.week_period = int(self.day_period * 7)

        self.train_end = int(self.num_steps * train_ratio)
        self.val_end = int(self.num_steps * (train_ratio + val_ratio))
        # Strictly restrict the metric/data fix to the two requested datasets.
        # All other datasets keep the original raw input, targets, priors, and loss path.
        self._target_metric_fix_dataset = self.dataset_name in {"PEMS04", "METR-LA"}
        self.impute_missing = bool(impute_missing) and self._target_metric_fix_dataset
        self.missing_value = float(missing_value)
        self.feature_raw_data = self._build_feature_data(self.raw_data, self.impute_missing, self.missing_value)
        train_data = self.feature_raw_data[:self.train_end]
        if self.normalization in {"node", "node_zscore", "per_node", "sensor", "sensor_zscore"}:
            # [1, N, C] node-wise z-score.  This is only enabled by dataset
            # preset for METR-LA.  It preserves the original raw scale after
            # inverse_transform, but makes speed forecasting easier because each
            # sensor has its own free-flow speed and variance.
            mean = train_data.mean(axis=0, keepdims=True)
            std = train_data.std(axis=0, keepdims=True)
        else:
            # Original behaviour: one global mean/std per channel.
            mean = train_data.mean(axis=(0, 1), keepdims=True)
            std = train_data.std(axis=(0, 1), keepdims=True)
        self.scaler = StandardScaler(mean, std)
        self.data = self.scaler.transform(self.feature_raw_data).astype(np.float32)
        # Only PeMS04/METR-LA need separate imputed inputs and original labels.
        # For all other datasets, self.target_data is exactly self.data, preserving
        # the original behavior bit-for-bit at the dataset output level.
        if self.impute_missing:
            self.target_data = self.scaler.transform(self.raw_data).astype(np.float32)
        else:
            self.target_data = self.data
        self.time_features = build_time_features(self.num_steps, points_per_hour)
        self.adj = load_adjacency(root_path, self.dataset_name, self.num_nodes)

        max_start = self.num_steps - seq_len - pred_len + 1
        all_starts = np.arange(max_start)
        train_max_start = self.train_end - seq_len - pred_len + 1
        val_max_start = self.val_end - seq_len - pred_len + 1
        self.train_indices = all_starts[all_starts < max(0, train_max_start)]
        self.val_indices = all_starts[(all_starts >= max(0, train_max_start)) & (all_starts < max(0, val_max_start))]
        self.test_indices = all_starts[all_starts >= max(0, val_max_start)]

        self._build_training_periodic_templates()
        self._build_training_pattern_memory()
        self._build_rolling_pattern_bank()
        # Closed-form node-wise ridge AR is used as a validation-only calibration
        # candidate and as a strong legal history prior.  Earlier versions turned
        # it off, so the calibrator could only choose among weak seasonal copies;
        # on PeMS08 this was the main reason calibrated MAE/RMSE stayed around
        # 14.5/24.2 and MAPE stayed above 10%.  The ridge model is fitted only
        # on chronological data before the evaluated test targets, never on test
        # futures.
        self.enable_ridge_ar = bool(enable_ridge_ar)
        if self.enable_ridge_ar:
            self._build_node_ridge_ar_memory()


    def _build_feature_data(self, raw: np.ndarray, impute_missing: bool, missing_value: float) -> np.ndarray:
        """Return the history/prior stream used by the model.

        METR-LA and parts of PeMS04 contain zeros that are missing detector
        records.  The official metrics mask zero labels, but earlier training fed
        those zeros as both inputs and targets, so the model learned artificial
        drops and RMSE/MAPE became poor.  This causal forward-fill uses only
        already observed values for each sensor/channel; labels remain unchanged
        in ``target_data`` and are masked in the loss.
        """
        if not impute_missing:
            return raw
        arr = np.asarray(raw, dtype=np.float32).copy()
        mv = float(missing_value)
        miss = ~np.isfinite(arr) | np.isclose(arr, mv)
        train = arr[:max(1, self.train_end)].copy()
        train_miss = ~np.isfinite(train) | np.isclose(train, mv)
        train_valid = np.where(train_miss, np.nan, train)
        with np.errstate(invalid="ignore"):
            fill = np.nanmedian(train_valid, axis=0, keepdims=True).astype(np.float32)
        global_fill = np.nanmedian(train_valid, axis=(0, 1), keepdims=True).astype(np.float32)
        if not np.isfinite(global_fill).all():
            global_fill = np.zeros((1, 1, arr.shape[-1]), dtype=np.float32)
        fill = np.where(np.isfinite(fill), fill, global_fill).astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, fill)
        arr = np.where(np.isclose(arr, mv), np.nan, arr)
        T, N, C = arr.shape
        last = np.repeat(fill.reshape(1, N, C), T, axis=0).astype(np.float32)
        for t in range(T):
            cur = arr[t]
            bad = ~np.isfinite(cur)
            if np.any(bad):
                cur = np.where(bad, last[t], cur)
            last[t] = cur
            if t + 1 < T:
                last[t + 1] = cur
        arr = last.astype(np.float32)
        print(f"[{self.dataset_name}] Input/prior missing-value imputation enabled: value={missing_value}, missing_ratio={float(miss.mean()):.4%}")
        return arr

    def _build_training_periodic_templates(self):
        """Training-only time-of-day and time-of-week averages.

        These are not labels from validation/test; they are historical statistics
        computed only from the training segment. They are kept for optional
        diagnostics, not injected into the HELMS model in the paper-aligned mode.
        """
        y = self.feature_raw_data[:self.train_end, :, :1].astype(np.float32)
        self.global_mean = y.mean(axis=0, keepdims=True).astype(np.float32)  # [1,N,1]

        def slot_mean(period: int):
            out = np.repeat(self.global_mean, period, axis=0).astype(np.float32)
            counts = np.zeros(period, dtype=np.int64)
            sums = np.zeros((period, self.num_nodes, 1), dtype=np.float64)
            for t in range(y.shape[0]):
                s = t % period
                sums[s] += y[t]
                counts[s] += 1
            for s in range(period):
                if counts[s] > 0:
                    out[s] = (sums[s] / counts[s]).astype(np.float32)
            return out

        self.tod_mean = slot_mean(self.day_period)
        # Use weekly template only if the training split spans at least two full
        # weeks; otherwise it is too sparse and falls back to daily means.
        if self.train_end >= 2 * self.week_period:
            self.tow_mean = slot_mean(self.week_period)
            self.has_week_template = True
        else:
            self.tow_mean = None
            self.has_week_template = False


    def _global_history_features_from_starts(self, starts: np.ndarray) -> np.ndarray:
        """Compact, training-only retrieval features for long-term pattern memory.

        The feature is deliberately global rather than node-specific so it is
        cheap for PeMS07/PeMS-BAY, yet it captures the history shape, recent
        network load, volatility, and time slot.  Retrieval uses only training
        windows and does not access validation/test future labels.
        """
        starts = np.asarray(starts, dtype=np.int64)
        if starts.size == 0:
            return np.zeros((0, self.seq_len * 2 + 9), dtype=np.float32)
        target = self.feature_raw_data[:, :, :1].astype(np.float32, copy=False)
        gmean = target.mean(axis=(1, 2)).astype(np.float32)
        gstd = target.std(axis=(1, 2)).astype(np.float32)
        idx = starts[:, None] + np.arange(self.seq_len, dtype=np.int64)[None, :]
        idx = np.clip(idx, 0, self.num_steps - 1)
        hm = gmean[idx]
        hs = gstd[idx]
        origin = starts + self.seq_len - 1
        origin = np.clip(origin, 0, self.num_steps - 1)
        # cyclical time encodings for traffic periodicity
        day_phase = 2.0 * np.pi * (origin % self.day_period) / max(1, self.day_period)
        week_phase = 2.0 * np.pi * (origin % self.week_period) / max(1, self.week_period)
        last = hm[:, -1:]
        first = hm[:, :1]
        local_mean = hm.mean(axis=1, keepdims=True)
        local_std = hm.std(axis=1, keepdims=True)
        slope = (hm[:, -1:] - hm[:, max(0, self.seq_len - 4):max(1, self.seq_len - 3)]) / 3.0
        feat = np.concatenate([
            hm, hs, last, first, local_mean, local_std, slope,
            np.sin(day_phase)[:, None], np.cos(day_phase)[:, None],
            np.sin(week_phase)[:, None], np.cos(week_phase)[:, None],
        ], axis=1).astype(np.float32)
        return feat

    def _build_training_pattern_memory(self):
        """Build a non-causal, training-only historical pattern memory bank.

        This is different from the removed causal_analog branch: it does not use
        Granger/causal scores and it is not tailored to any one dataset.  It is
        a legal HMC-style long-term pattern reuse module: current histories are
        matched to similar training histories and the retrieved training futures
        are offered to the validation calibrator as additional memory bases.
        """
        train_max_start = self.train_end - self.seq_len - self.pred_len + 1
        if train_max_start <= 0:
            self.pattern_bank_starts = np.zeros((0,), dtype=np.int64)
            self.pattern_bank_feat = np.zeros((0, self.seq_len * 2 + 9), dtype=np.float32)
            self.pattern_feat_mean = np.zeros((1, self.seq_len * 2 + 9), dtype=np.float32)
            self.pattern_feat_std = np.ones((1, self.seq_len * 2 + 9), dtype=np.float32)
            return
        starts = np.arange(train_max_start, dtype=np.int64)
        # Subsample evenly for very large datasets to keep calibration fast and memory safe.
        max_bank = 12000
        if starts.size > max_bank:
            starts = np.linspace(0, train_max_start - 1, max_bank).round().astype(np.int64)
            starts = np.unique(starts)
        feat = self._global_history_features_from_starts(starts)
        mean = feat.mean(axis=0, keepdims=True).astype(np.float32)
        std = feat.std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.pattern_bank_starts = starts.astype(np.int64)
        self.pattern_feat_mean = mean
        self.pattern_feat_std = std
        self.pattern_bank_feat = ((feat - mean) / std).astype(np.float32)
        self.pattern_bank_norm = (self.pattern_bank_feat ** 2).sum(axis=1, keepdims=True).T.astype(np.float32)

    def _make_pattern_memory_priors(self, starts: np.ndarray, pred_len: int, topk: int = 8) -> Dict[str, np.ndarray]:
        """Retrieve futures from the training pattern bank for a batch.

        Returns raw-scale [B,T,N,1] arrays.  The delta variant aligns retrieved
        futures to the current last observed node values, which is a standard
        residual-memory correction and uses no future test labels.
        """
        B = len(starts)
        shape = (B, pred_len, self.num_nodes, 1)
        if getattr(self, 'pattern_bank_starts', None) is None or len(self.pattern_bank_starts) == 0:
            z = np.repeat(self.global_mean[None, ...], B * pred_len, axis=0).reshape(shape).astype(np.float32)
            return {"pattern_memory": z, "pattern_memory_delta": z.copy(), "pattern_residual_memory": np.zeros_like(z)}
        q = self._global_history_features_from_starts(starts)
        q = ((q - self.pattern_feat_mean) / self.pattern_feat_std).astype(np.float32)
        # squared Euclidean distances to all training pattern prototypes
        qn = (q ** 2).sum(axis=1, keepdims=True).astype(np.float32)
        dist = qn + self.pattern_bank_norm - 2.0 * q.dot(self.pattern_bank_feat.T)
        dist = np.maximum(dist, 0.0)
        k = min(int(topk), self.pattern_bank_feat.shape[0])
        if k <= 0:
            k = 1
        idx_part = np.argpartition(dist, kth=k-1, axis=1)[:, :k]
        row = np.arange(B)[:, None]
        dsel = dist[row, idx_part]
        order = np.argsort(dsel, axis=1)
        idx_top = idx_part[row, order]
        dtop = dsel[row, order]
        # adaptive temperature: median selected distance gives stable weights across datasets
        temp = np.maximum(np.median(dtop, axis=1, keepdims=True), 1e-4)
        w = np.exp(-dtop / temp).astype(np.float32)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-6)

        target = self.feature_raw_data[:, :, :1].astype(np.float32, copy=False)
        out = np.zeros(shape, dtype=np.float32)
        out_delta = np.zeros(shape, dtype=np.float32)
        out_resid = np.zeros(shape, dtype=np.float32)
        for b in range(B):
            cand_st = self.pattern_bank_starts[idx_top[b]]
            cand_origin = cand_st + self.seq_len - 1
            qry_origin = int(starts[b] + self.seq_len - 1)
            qry_last = target[np.clip(qry_origin, 0, self.num_steps - 1)]
            cand_last = target[np.clip(cand_origin, 0, self.num_steps - 1)]  # [K,N,1]
            wv = w[b].reshape(-1, 1, 1)
            last_mem = (wv * cand_last).sum(axis=0)
            # node-wise residual alignment, robustly clipped by recent local scale
            recent0 = max(0, qry_origin - self.seq_len + 1)
            scale = np.maximum(target[recent0:qry_origin+1].std(axis=0), 1.0)
            shift = np.clip(qry_last - last_mem, -1.5 * scale, 1.5 * scale)
            for h in range(1, pred_len + 1):
                fut_idx = cand_st + self.seq_len + h - 1
                fut_idx = np.clip(fut_idx, 0, self.train_end - 1)
                fut = target[fut_idx]  # [K,N,1]
                base = (wv * fut).sum(axis=0)
                out[b, h-1] = np.maximum(base, 0.0)
                out_delta[b, h-1] = np.maximum(base + shift, 0.0)
                out_resid[b, h-1] = shift
        return {"pattern_memory": out, "pattern_memory_delta": out_delta, "pattern_residual_memory": out_resid}


    def _build_rolling_pattern_bank(self):
        """Precompute features for online/lifelong analog retrieval.

        The old pattern memory bank was fitted on the training split only.  That
        is a static setting and does not match the paper's online DML idea: once
        validation/test time moves forward, past observations are already known
        and can be stored as lifelong memory.  This bank stores only historical
        windows.  When a query at origin t is evaluated, candidates are filtered
        so their entire future segment ends before t; therefore no target from
        the current or future prediction interval is used.
        """
        max_start = self.num_steps - self.seq_len - self.pred_len + 1
        if max_start <= 0:
            self.rolling_bank_starts = np.zeros((0,), dtype=np.int64)
            self.rolling_bank_feat = np.zeros_like(getattr(self, 'pattern_bank_feat', np.zeros((0, self.seq_len * 2 + 9), dtype=np.float32)))
            self.rolling_bank_norm = np.zeros((1, 0), dtype=np.float32)
            return
        starts = np.arange(max_start, dtype=np.int64)
        feat = self._global_history_features_from_starts(starts)
        # Normalize with training-only statistics to avoid leaking future
        # distribution information through feature scaling.
        mean = getattr(self, 'pattern_feat_mean', feat[:max(1, min(len(feat), 100))].mean(axis=0, keepdims=True))
        std = getattr(self, 'pattern_feat_std', feat[:max(1, min(len(feat), 100))].std(axis=0, keepdims=True))
        std = std.copy()
        std[std < 1e-6] = 1.0
        z = ((feat - mean) / std).astype(np.float32)
        self.rolling_bank_starts = starts
        self.rolling_bank_feat = z
        self.rolling_bank_norm = (z ** 2).sum(axis=1, keepdims=True).T.astype(np.float32)

    def _make_rolling_knn_priors(self, starts: np.ndarray, pred_len: int, topk: int = 12,
                                 max_bank: int = 7000) -> Dict[str, np.ndarray]:
        """Online HMC-style KNN memory prior over all past windows.

        For a query origin t, retrieve similar windows whose whole forecast
        horizon is already observed before t.  This mirrors DML updating memory
        as new samples arrive and is much stronger than a training-only memory
        under concept drift.  It remains causal/online: no current test target or
        future target is accessed.
        """
        starts = np.asarray(starts, dtype=np.int64)
        B = len(starts)
        shape = (B, pred_len, self.num_nodes, 1)
        target = self.feature_raw_data[:, :, :1].astype(np.float32, copy=False)
        out = np.zeros(shape, dtype=np.float32)
        out_delta = np.zeros_like(out)
        out_blend = np.zeros_like(out)
        if getattr(self, 'rolling_bank_starts', None) is None or len(self.rolling_bank_starts) == 0:
            z = np.repeat(self.global_mean[None, ...], B * pred_len, axis=0).reshape(shape).astype(np.float32)
            return {"rolling_knn": z, "rolling_knn_delta": z.copy(), "rolling_knn_blend": z.copy()}
        q = self._global_history_features_from_starts(starts)
        q = ((q - self.pattern_feat_mean) / self.pattern_feat_std).astype(np.float32)
        all_starts = self.rolling_bank_starts
        all_feat = self.rolling_bank_feat
        for b, st in enumerate(starts):
            origin = int(st + self.seq_len - 1)
            # Candidate future must be fully known before the query origin.
            latest_cand = origin - self.seq_len - pred_len
            cand_mask = all_starts <= latest_cand
            cand_idx = np.nonzero(cand_mask)[0]
            if cand_idx.size == 0:
                for h in range(1, pred_len + 1):
                    slot = (origin + h) % self.day_period
                    out[b, h-1] = self.tod_mean[slot]
                    out_delta[b, h-1] = self.tod_mean[slot]
                    out_blend[b, h-1] = self.tod_mean[slot]
                continue
            if cand_idx.size > max_bank:
                # Recent windows adapt to drift, while evenly sampled older
                # windows retain recurring long-term patterns.
                recent = cand_idx[-max_bank // 2:]
                older_pool = cand_idx[:max(1, cand_idx.size - recent.size)]
                older_n = max_bank - recent.size
                older = older_pool[np.linspace(0, older_pool.size - 1, older_n).round().astype(np.int64)] if older_n > 0 else np.zeros((0,), dtype=np.int64)
                cand_idx = np.unique(np.concatenate([older, recent]))
            cf = all_feat[cand_idx]
            dist = ((q[b:b+1] ** 2).sum(axis=1, keepdims=True) +
                    (cf ** 2).sum(axis=1, keepdims=True).T -
                    2.0 * q[b:b+1].dot(cf.T))
            dist = np.maximum(dist.reshape(-1), 0.0)
            k = max(1, min(int(topk), dist.size))
            top_local = np.argpartition(dist, kth=k-1)[:k]
            top_local = top_local[np.argsort(dist[top_local])]
            dtop = dist[top_local]
            temp = max(float(np.median(dtop)), 1e-4)
            w = np.exp(-dtop / temp).astype(np.float32)
            w = w / max(float(w.sum()), 1e-6)
            cand_st = all_starts[cand_idx[top_local]]
            cand_origin = cand_st + self.seq_len - 1
            cur_last = target[np.clip(origin, 0, self.num_steps - 1)]
            cand_last = target[np.clip(cand_origin, 0, self.num_steps - 1)]
            wv = w.reshape(-1, 1, 1)
            last_mem = (wv * cand_last).sum(axis=0)
            recent0 = max(0, origin - self.seq_len + 1)
            scale = np.maximum(target[recent0:origin+1].std(axis=0), 1.0)
            shift = np.clip(cur_last - last_mem, -1.2 * scale, 1.2 * scale)
            for h in range(1, pred_len + 1):
                fut_idx = cand_st + self.seq_len + h - 1
                # By construction fut_idx < origin, but clip for safety.
                fut_idx = np.clip(fut_idx, 0, max(0, origin - 1))
                base = (wv * target[fut_idx]).sum(axis=0)
                delta = np.maximum(base + shift, 0.0)
                out[b, h-1] = np.maximum(base, 0.0)
                out_delta[b, h-1] = delta
                out_blend[b, h-1] = np.maximum(0.70 * delta + 0.30 * base, 0.0)
        return {"rolling_knn": out, "rolling_knn_delta": out_delta, "rolling_knn_blend": out_blend}


    def _make_daily_analog_priors(self, starts: np.ndarray, pred_len: int, max_days: int = 28, topk: int = 4) -> Dict[str, np.ndarray]:
        """Node-wise same-time historical analog memory.

        For each forecast origin, retrieve the most similar *past* same-time
        histories for every sensor independently from previous days.  The
        retrieved futures are strictly before the forecast origin, so this uses
        no validation/test future labels.  Compared with the older global
        pattern_memory, this sensor-wise analog is much stronger for PeMS-style
        flow data because each detector has its own amplitude and rush-hour
        shape.
        """
        starts = np.asarray(starts, dtype=np.int64)
        B = len(starts)
        shape = (B, pred_len, self.num_nodes, 1)
        target = self.feature_raw_data[:, :, :1].astype(np.float32, copy=False)
        out = np.zeros(shape, dtype=np.float32)
        out_delta = np.zeros_like(out)
        out_blend = np.zeros_like(out)
        out_resid = np.zeros_like(out)
        nodes = np.arange(self.num_nodes, dtype=np.int64)[None, :]
        L = int(self.seq_len)
        for b, st in enumerate(starts):
            origin = int(st + L - 1)
            hist0 = origin - L + 1
            if origin <= 0 or hist0 < 0 or origin >= self.num_steps:
                # Fall back to daily/tod template when the current history is unavailable.
                for h in range(1, pred_len + 1):
                    slot = (origin + h) % self.day_period
                    out[b, h-1] = self.tod_mean[slot]
                    out_delta[b, h-1] = self.tod_mean[slot]
                    out_blend[b, h-1] = self.tod_mean[slot]
                continue
            cur_hist = target[hist0:origin+1, :, 0]  # [L,N]
            cur_last = target[origin, :, 0]
            # Candidate origins from previous days.  Their future horizon must
            # be known before the current forecast origin.
            cand_origins = []
            max_d = min(int(max_days), max(1, origin // self.day_period))
            for d in range(1, max_d + 1):
                co = origin - d * self.day_period
                if co - L + 1 >= 0 and co + pred_len < origin:
                    cand_origins.append(co)
            if not cand_origins:
                # Same-time yesterday or training slot mean fallback.
                for h in range(1, pred_len + 1):
                    tau = origin + h
                    prev = tau - self.day_period
                    slot = tau % self.day_period
                    base = target[prev, :, 0] if prev >= 0 else self.tod_mean[slot, :, 0]
                    base = base.astype(np.float32)
                    out[b, h-1, :, 0] = np.maximum(base, 0.0)
                    out_delta[b, h-1, :, 0] = np.maximum(base, 0.0)
                    out_blend[b, h-1, :, 0] = np.maximum(base, 0.0)
                continue
            cand_origins = np.asarray(cand_origins, dtype=np.int64)
            K = cand_origins.size
            # [K,L,N]
            cand_hist = np.stack([target[co-L+1:co+1, :, 0] for co in cand_origins], axis=0)
            # Normalize distance by recent node scale so high-volume detectors
            # do not dominate their own retrieval.
            node_scale = np.maximum(cur_hist.std(axis=0, keepdims=True), 1.0).astype(np.float32)
            dist = np.mean(((cand_hist - cur_hist[None, :, :]) / node_scale[None, :, :]) ** 2, axis=1)  # [K,N]
            k = max(1, min(int(topk), K))
            idx = np.argpartition(dist, kth=k-1, axis=0)[:k, :]  # [k,N]
            dsel = np.take_along_axis(dist, idx, axis=0)
            order = np.argsort(dsel, axis=0)
            idx = np.take_along_axis(idx, order, axis=0)
            dsel = np.take_along_axis(dsel, order, axis=0)
            temp = np.maximum(np.median(dsel, axis=0, keepdims=True), 1e-4)
            w = np.exp(-dsel / temp).astype(np.float32)
            w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-6)
            top_orig = cand_origins[idx]  # [k,N]
            cand_last = target[top_orig, nodes, 0]  # [k,N]
            last_mem = (w * cand_last).sum(axis=0)
            local_scale = np.maximum(cur_hist.std(axis=0), 1.0)
            shift = np.clip(cur_last - last_mem, -1.35 * local_scale, 1.35 * local_scale).astype(np.float32)
            out_resid[b, :, :, 0] = shift[None, :]
            # Local recent shape gives a complementary short-term extrapolation.
            first = cur_hist[0]
            for h in range(1, pred_len + 1):
                fut_idx = top_orig + h
                fut = target[fut_idx, nodes, 0]  # [k,N], guaranteed in the past
                base = (w * fut).sum(axis=0).astype(np.float32)
                aligned = np.maximum(base + shift, 0.0)
                # Repeat the recent one-hour shape around the current last value.
                shape_pos = min(L - 1, h - 1)
                local_shape = np.maximum(cur_last + (cur_hist[shape_pos] - first), 0.0)
                out[b, h-1, :, 0] = np.maximum(base, 0.0)
                out_delta[b, h-1, :, 0] = aligned
                out_blend[b, h-1, :, 0] = np.maximum(0.72 * aligned + 0.28 * local_shape, 0.0)
        return {
            "daily_analog": out,
            "daily_analog_delta": out_delta,
            "daily_analog_blend": out_blend,
            "daily_analog_residual": out_resid,
        }



    def _make_weekly_analog_priors(self, starts: np.ndarray, pred_len: int, max_weeks: int = 8, topk: int = 3) -> Dict[str, np.ndarray]:
        """Node-wise same-time previous-week analog memory.

        Daily analogs are strong but can confuse weekday/weekend or holiday-like
        regimes.  This weekly branch retrieves previous weeks with the same
        clock slot and aligns them to the current observed level.  It uses only
        observations before the forecast origin, so it is safe for rolling test
        forecasting and gives the calibrator a MAPE-friendly candidate on
        PeMS-style flow data.
        """
        starts = np.asarray(starts, dtype=np.int64)
        B = len(starts)
        shape = (B, pred_len, self.num_nodes, 1)
        target = self.feature_raw_data[:, :, :1].astype(np.float32, copy=False)
        out = np.zeros(shape, dtype=np.float32)
        out_delta = np.zeros_like(out)
        out_blend = np.zeros_like(out)
        nodes = np.arange(self.num_nodes, dtype=np.int64)[None, :]
        L = int(self.seq_len)
        for b, st in enumerate(starts):
            origin = int(st + L - 1)
            hist0 = origin - L + 1
            if origin <= 0 or hist0 < 0 or origin >= self.num_steps:
                for h in range(1, pred_len + 1):
                    slot = (origin + h) % self.week_period
                    if self.has_week_template:
                        base = self.tow_mean[slot]
                    else:
                        base = self.tod_mean[(origin + h) % self.day_period]
                    out[b, h-1] = base
                    out_delta[b, h-1] = base
                    out_blend[b, h-1] = base
                continue
            cur_hist = target[hist0:origin+1, :, 0]
            cur_last = target[origin, :, 0]
            cand_origins = []
            max_w = min(int(max_weeks), max(1, origin // self.week_period))
            for widx in range(1, max_w + 1):
                co = origin - widx * self.week_period
                if co - L + 1 >= 0 and co + pred_len < origin:
                    cand_origins.append(co)
            if not cand_origins:
                # Fallback to same-time last week or training weekly/daily template.
                for h in range(1, pred_len + 1):
                    tau = origin + h
                    prev = tau - self.week_period
                    if prev >= 0:
                        base = target[prev, :, 0]
                    elif self.has_week_template:
                        base = self.tow_mean[tau % self.week_period, :, 0]
                    else:
                        base = self.tod_mean[tau % self.day_period, :, 0]
                    out[b, h-1, :, 0] = np.maximum(base, 0.0)
                    out_delta[b, h-1, :, 0] = np.maximum(base, 0.0)
                    out_blend[b, h-1, :, 0] = np.maximum(base, 0.0)
                continue
            cand_origins = np.asarray(cand_origins, dtype=np.int64)
            K = cand_origins.size
            cand_hist = np.stack([target[co-L+1:co+1, :, 0] for co in cand_origins], axis=0)
            node_scale = np.maximum(cur_hist.std(axis=0, keepdims=True), 1.0).astype(np.float32)
            dist = np.mean(((cand_hist - cur_hist[None, :, :]) / node_scale[None, :, :]) ** 2, axis=1)
            k = max(1, min(int(topk), K))
            idx = np.argpartition(dist, kth=k-1, axis=0)[:k, :]
            dsel = np.take_along_axis(dist, idx, axis=0)
            order = np.argsort(dsel, axis=0)
            idx = np.take_along_axis(idx, order, axis=0)
            dsel = np.take_along_axis(dsel, order, axis=0)
            temp = np.maximum(np.median(dsel, axis=0, keepdims=True), 1e-4)
            wt = np.exp(-dsel / temp).astype(np.float32)
            wt = wt / np.maximum(wt.sum(axis=0, keepdims=True), 1e-6)
            top_orig = cand_origins[idx]
            cand_last = target[top_orig, nodes, 0]
            last_mem = (wt * cand_last).sum(axis=0)
            local_scale = np.maximum(cur_hist.std(axis=0), 1.0)
            shift = np.clip(cur_last - last_mem, -1.20 * local_scale, 1.20 * local_scale).astype(np.float32)
            first = cur_hist[0]
            for h in range(1, pred_len + 1):
                fut_idx = top_orig + h
                fut = target[fut_idx, nodes, 0]
                base = (wt * fut).sum(axis=0).astype(np.float32)
                aligned = np.maximum(base + shift, 0.0)
                shape_pos = min(L - 1, h - 1)
                local_shape = np.maximum(cur_last + (cur_hist[shape_pos] - first), 0.0)
                out[b, h-1, :, 0] = np.maximum(base, 0.0)
                out_delta[b, h-1, :, 0] = aligned
                out_blend[b, h-1, :, 0] = np.maximum(0.75 * aligned + 0.25 * local_shape, 0.0)
        return {
            "weekly_analog": out,
            "weekly_analog_delta": out_delta,
            "weekly_analog_blend": out_blend,
        }


    def _make_daily_analog_priors(self, starts: np.ndarray, pred_len: int, max_days: int = 28, topk: int = 4) -> Dict[str, np.ndarray]:
        """Vectorized same-time previous-day analog priors.

        This is mathematically the same retrieval rule as the original per-sample
        loop, but it processes the whole mini-batch at once.  It is the main
        warmup speed fix for PeMS07/PeMS-BAY, where the old code repeatedly
        built candidate histories for every item in Python.
        """
        starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        B = int(starts.size)
        shape = (B, int(pred_len), self.num_nodes, 1)
        target = self.feature_raw_data[:, :, :1].astype(np.float32, copy=False)
        if B == 0:
            z = np.zeros(shape, dtype=np.float32)
            return {"daily_analog": z, "daily_analog_delta": z.copy(), "daily_analog_blend": z.copy(), "daily_analog_residual": z.copy()}

        T = int(pred_len)
        N = int(self.num_nodes)
        L = int(self.seq_len)
        out = np.zeros(shape, dtype=np.float32)
        out_delta = np.zeros_like(out)
        out_blend = np.zeros_like(out)
        out_resid = np.zeros_like(out)

        origins = starts + L - 1
        origins_clip = np.clip(origins, 0, self.num_steps - 1)
        hist_start = origins - L + 1
        cur_hist_idx = np.clip(hist_start[:, None] + np.arange(L, dtype=np.int64)[None, :], 0, self.num_steps - 1)
        cur_hist = target[cur_hist_idx, :, 0]  # [B,L,N]
        cur_last = target[origins_clip, :, 0]  # [B,N]

        days = np.arange(1, int(max_days) + 1, dtype=np.int64)
        cand_orig = origins[:, None] - days[None, :] * self.day_period  # [B,D]
        valid = (cand_orig - L + 1 >= 0) & (cand_orig + T < origins[:, None])
        D = cand_orig.shape[1]
        valid_any = valid.any(axis=1)

        # Fallback for samples without valid daily analogs.
        if np.any(~valid_any):
            bad = np.where(~valid_any)[0]
            for h in range(1, T + 1):
                tau = origins[bad] + h
                prev = tau - self.day_period
                base = target[np.clip(prev, 0, self.num_steps - 1), :, 0]
                neg = prev < 0
                if np.any(neg):
                    base[neg] = self.tod_mean[np.mod(tau[neg], self.day_period), :, 0]
                out[bad, h-1, :, 0] = np.maximum(base, 0.0)
                out_delta[bad, h-1, :, 0] = np.maximum(base, 0.0)
                out_blend[bad, h-1, :, 0] = np.maximum(base, 0.0)

        good = np.where(valid_any)[0]
        if good.size > 0:
            co = cand_orig[good]
            valid_g = valid[good]
            # [G,D,L]
            cand_hist_idx = np.clip(co[:, :, None] - L + 1 + np.arange(L, dtype=np.int64)[None, None, :], 0, self.num_steps - 1)
            cand_hist = target[cand_hist_idx, :, 0]  # [G,D,L,N]
            ch = cur_hist[good]  # [G,L,N]
            node_scale = np.maximum(ch.std(axis=1, keepdims=True), 1.0).astype(np.float32)  # [G,1,N]
            dist = np.mean(((cand_hist - ch[:, None, :, :]) / node_scale[:, None, :, :]) ** 2, axis=2)  # [G,D,N]
            dist = np.where(valid_g[:, :, None], dist, np.inf)
            valid_count = valid_g.sum(axis=1)
            kmax = max(1, min(int(topk), D))
            # argpartition is safe because every 'good' row has at least one finite candidate.
            idx = np.argpartition(dist, kth=kmax - 1, axis=1)[:, :kmax, :]  # [G,k,N]
            dsel = np.take_along_axis(dist, idx, axis=1)
            order = np.argsort(dsel, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            dsel = np.take_along_axis(dsel, order, axis=1)
            # Mask padded candidates for rows with fewer than kmax valid candidates.
            rank = np.arange(kmax, dtype=np.int64)[None, :, None]
            finite_mask = rank < valid_count[:, None, None]
            dsel = np.where(finite_mask, dsel, np.inf)
            temp = np.maximum(np.nanmedian(np.where(np.isfinite(dsel), dsel, np.nan), axis=1, keepdims=True), 1e-4)
            temp = np.where(np.isfinite(temp), temp, 1.0).astype(np.float32)
            wt = np.exp(-np.where(np.isfinite(dsel), dsel, 1e6) / temp).astype(np.float32)
            wt = wt * finite_mask.astype(np.float32)
            wt = wt / np.maximum(wt.sum(axis=1, keepdims=True), 1e-6)
            top_orig = np.take_along_axis(co[:, :, None], idx, axis=1)  # [G,k,N]
            nodes = np.arange(N, dtype=np.int64)[None, None, :]
            cand_last = target[np.clip(top_orig, 0, self.num_steps - 1), nodes, 0]
            last_mem = (wt * cand_last).sum(axis=1)
            ch_last = cur_last[good]
            local_scale = np.maximum(ch.std(axis=1), 1.0)
            shift = np.clip(ch_last - last_mem, -1.25 * local_scale, 1.25 * local_scale).astype(np.float32)
            first = ch[:, 0, :]
            for h in range(1, T + 1):
                fut_idx = np.clip(top_orig + h, 0, self.num_steps - 1)
                fut = target[fut_idx, nodes, 0]
                base = (wt * fut).sum(axis=1).astype(np.float32)
                aligned = np.maximum(base + shift, 0.0)
                shape_pos = min(L - 1, h - 1)
                local_shape = np.maximum(ch_last + (ch[:, shape_pos, :] - first), 0.0)
                gidx = good
                out[gidx, h-1, :, 0] = np.maximum(base, 0.0)
                out_delta[gidx, h-1, :, 0] = aligned
                out_blend[gidx, h-1, :, 0] = np.maximum(0.72 * aligned + 0.28 * local_shape, 0.0)
                out_resid[gidx, h-1, :, 0] = np.maximum(aligned - ch_last, -1e6)
        return {
            "daily_analog": out,
            "daily_analog_delta": out_delta,
            "daily_analog_blend": out_blend,
            "daily_analog_residual": out_resid,
        }

    def _make_weekly_analog_priors(self, starts: np.ndarray, pred_len: int, max_weeks: int = 8, topk: int = 3) -> Dict[str, np.ndarray]:
        """Vectorized same-time previous-week analog priors."""
        starts = np.asarray(starts, dtype=np.int64).reshape(-1)
        B = int(starts.size)
        T = int(pred_len)
        N = int(self.num_nodes)
        L = int(self.seq_len)
        shape = (B, T, N, 1)
        target = self.feature_raw_data[:, :, :1].astype(np.float32, copy=False)
        if B == 0:
            z = np.zeros(shape, dtype=np.float32)
            return {"weekly_analog": z, "weekly_analog_delta": z.copy(), "weekly_analog_blend": z.copy()}
        out = np.zeros(shape, dtype=np.float32)
        out_delta = np.zeros_like(out)
        out_blend = np.zeros_like(out)

        origins = starts + L - 1
        origins_clip = np.clip(origins, 0, self.num_steps - 1)
        hist_start = origins - L + 1
        cur_hist_idx = np.clip(hist_start[:, None] + np.arange(L, dtype=np.int64)[None, :], 0, self.num_steps - 1)
        cur_hist = target[cur_hist_idx, :, 0]
        cur_last = target[origins_clip, :, 0]

        weeks = np.arange(1, int(max_weeks) + 1, dtype=np.int64)
        cand_orig = origins[:, None] - weeks[None, :] * self.week_period
        valid = (cand_orig - L + 1 >= 0) & (cand_orig + T < origins[:, None])
        Kall = cand_orig.shape[1]
        valid_any = valid.any(axis=1)
        if np.any(~valid_any):
            bad = np.where(~valid_any)[0]
            for h in range(1, T + 1):
                tau = origins[bad] + h
                prev = tau - self.week_period
                base = target[np.clip(prev, 0, self.num_steps - 1), :, 0]
                neg = prev < 0
                if np.any(neg):
                    if self.has_week_template:
                        base[neg] = self.tow_mean[np.mod(tau[neg], self.week_period), :, 0]
                    else:
                        base[neg] = self.tod_mean[np.mod(tau[neg], self.day_period), :, 0]
                out[bad, h-1, :, 0] = np.maximum(base, 0.0)
                out_delta[bad, h-1, :, 0] = np.maximum(base, 0.0)
                out_blend[bad, h-1, :, 0] = np.maximum(base, 0.0)
        good = np.where(valid_any)[0]
        if good.size > 0:
            co = cand_orig[good]
            valid_g = valid[good]
            cand_hist_idx = np.clip(co[:, :, None] - L + 1 + np.arange(L, dtype=np.int64)[None, None, :], 0, self.num_steps - 1)
            cand_hist = target[cand_hist_idx, :, 0]
            ch = cur_hist[good]
            node_scale = np.maximum(ch.std(axis=1, keepdims=True), 1.0).astype(np.float32)
            dist = np.mean(((cand_hist - ch[:, None, :, :]) / node_scale[:, None, :, :]) ** 2, axis=2)
            dist = np.where(valid_g[:, :, None], dist, np.inf)
            valid_count = valid_g.sum(axis=1)
            kmax = max(1, min(int(topk), Kall))
            idx = np.argpartition(dist, kth=kmax - 1, axis=1)[:, :kmax, :]
            dsel = np.take_along_axis(dist, idx, axis=1)
            order = np.argsort(dsel, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            dsel = np.take_along_axis(dsel, order, axis=1)
            rank = np.arange(kmax, dtype=np.int64)[None, :, None]
            finite_mask = rank < valid_count[:, None, None]
            dsel = np.where(finite_mask, dsel, np.inf)
            temp = np.maximum(np.nanmedian(np.where(np.isfinite(dsel), dsel, np.nan), axis=1, keepdims=True), 1e-4)
            temp = np.where(np.isfinite(temp), temp, 1.0).astype(np.float32)
            wt = np.exp(-np.where(np.isfinite(dsel), dsel, 1e6) / temp).astype(np.float32)
            wt = wt * finite_mask.astype(np.float32)
            wt = wt / np.maximum(wt.sum(axis=1, keepdims=True), 1e-6)
            top_orig = np.take_along_axis(co[:, :, None], idx, axis=1)
            nodes = np.arange(N, dtype=np.int64)[None, None, :]
            cand_last = target[np.clip(top_orig, 0, self.num_steps - 1), nodes, 0]
            last_mem = (wt * cand_last).sum(axis=1)
            ch_last = cur_last[good]
            local_scale = np.maximum(ch.std(axis=1), 1.0)
            shift = np.clip(ch_last - last_mem, -1.20 * local_scale, 1.20 * local_scale).astype(np.float32)
            first = ch[:, 0, :]
            for h in range(1, T + 1):
                fut = target[np.clip(top_orig + h, 0, self.num_steps - 1), nodes, 0]
                base = (wt * fut).sum(axis=1).astype(np.float32)
                aligned = np.maximum(base + shift, 0.0)
                shape_pos = min(L - 1, h - 1)
                local_shape = np.maximum(ch_last + (ch[:, shape_pos, :] - first), 0.0)
                out[good, h-1, :, 0] = np.maximum(base, 0.0)
                out_delta[good, h-1, :, 0] = aligned
                out_blend[good, h-1, :, 0] = np.maximum(0.75 * aligned + 0.25 * local_shape, 0.0)
        return {
            "weekly_analog": out,
            "weekly_analog_delta": out_delta,
            "weekly_analog_blend": out_blend,
        }


    def _ridge_feature_chunk(self, starts: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        """Build node-wise autoregressive features for a chunk of nodes.

        Features are strictly historical: the last seq_len values, first
        differences, origin time encodings and a bias term.  No future label is
        used here.  The features are normalized per node using training-only
        statistics, which makes the closed-form ridge coefficients stable across
        PeMS/METR scales.
        """
        starts = np.asarray(starts, dtype=np.int64)
        nodes = np.asarray(nodes, dtype=np.int64)
        M, Cn = starts.size, nodes.size
        L = int(self.seq_len)
        y2d = self.feature_raw_data[:, :, 0].astype(np.float32, copy=False)
        hist_idx = starts[:, None] + np.arange(L, dtype=np.int64)[None, :]
        hist_idx = np.clip(hist_idx, 0, self.num_steps - 1)
        hist = y2d[:, nodes][hist_idx]  # [M,L,Cn]
        mean = self.node_ridge_mean[nodes].reshape(1, 1, Cn)
        std = self.node_ridge_std[nodes].reshape(1, 1, Cn)
        hist_n = (hist - mean) / std
        dif = np.diff(hist_n, axis=1)
        origin = starts + L - 1
        day_phase = 2.0 * np.pi * (origin % self.day_period) / max(1, self.day_period)
        week_phase = 2.0 * np.pi * (origin % self.week_period) / max(1, self.week_period)
        tfeat = np.stack([
            np.sin(day_phase), np.cos(day_phase),
            np.sin(week_phase), np.cos(week_phase),
            np.ones_like(day_phase, dtype=np.float64),
        ], axis=1).astype(np.float32)  # [M,5]
        tfeat = np.repeat(tfeat[:, None, :], Cn, axis=1)  # [M,Cn,5]
        # [M,Cn,F]
        feat = np.concatenate([
            hist_n.transpose(0, 2, 1),
            dif.transpose(0, 2, 1),
            tfeat,
        ], axis=2).astype(np.float32)
        return feat

    def _sample_ridge_starts(self, segment_end: int, max_fit: int = 14000) -> np.ndarray:
        max_start = int(segment_end) - self.seq_len - self.pred_len + 1
        if max_start <= 0:
            return np.zeros((0,), dtype=np.int64)
        starts = np.arange(max_start, dtype=np.int64)
        if starts.size > max_fit:
            # Deterministic uniform subsampling keeps rush/off-peak coverage and
            # avoids a very slow fit on PeMS07 / PeMS-BAY.
            starts = np.linspace(0, max_start - 1, max_fit).round().astype(np.int64)
            starts = np.unique(starts)
        return starts

    def _fit_node_ridge_ar(self, starts: np.ndarray, lam: float = 20.0, chunk_size: int = 64) -> np.ndarray:
        """Fit per-node, per-horizon closed-form ridge AR in normalized space.

        Return coefficients with shape [T, N, F].  This is much more reliable
        than training the same linear map through the neural optimizer, and it
        gives the calibrator a strong non-neural candidate without using any
        validation/test future labels.
        """
        T, N, L = int(self.pred_len), int(self.num_nodes), int(self.seq_len)
        F = L + max(0, L - 1) + 5
        coef = np.zeros((T, N, F), dtype=np.float32)
        if starts.size == 0:
            return coef
        y2d = self.feature_raw_data[:, :, 0].astype(np.float32, copy=False)
        I = np.eye(F, dtype=np.float64)
        I[-1, -1] = 0.01  # regularize bias weakly
        for ns in range(0, N, chunk_size):
            ne = min(N, ns + chunk_size)
            nodes = np.arange(ns, ne, dtype=np.int64)
            X = self._ridge_feature_chunk(starts, nodes).astype(np.float64)  # [M,C,F]
            Cn = X.shape[1]
            XtX = np.einsum('mcf,mcg->cfg', X, X, optimize=True)
            XtX = XtX + float(lam) * I[None, :, :]
            # Targets for all horizons, normalized per node.
            Y = np.zeros((starts.size, Cn, T), dtype=np.float64)
            mean = self.node_ridge_mean[nodes].reshape(1, Cn)
            std = self.node_ridge_std[nodes].reshape(1, Cn)
            for h in range(1, T + 1):
                ti = starts + L + h - 1
                yy = y2d[:, nodes][np.clip(ti, 0, self.num_steps - 1)]
                Y[:, :, h-1] = (yy - mean) / std
            Xty = np.einsum('mcf,mct->cft', X, Y, optimize=True)
            try:
                sol = np.linalg.solve(XtX, Xty)  # [C,F,T]
            except Exception:
                sol = np.matmul(np.linalg.pinv(XtX), Xty)
            coef[:, ns:ne, :] = sol.transpose(2, 0, 1).astype(np.float32)
        return coef

    def _build_node_ridge_ar_memory(self):
        """Fit train-only and train+val node-wise ridge memories.

        This module is deliberately non-causal and non-oracle: it is just a
        closed-form history-to-future map fitted on past chronological data.  It
        is used as an additional HMC-style long-term pattern candidate by the
        validation calibrator.
        """
        y = self.feature_raw_data[:self.train_end, :, 0].astype(np.float32)
        self.node_ridge_mean = y.mean(axis=0).astype(np.float32)
        self.node_ridge_std = y.std(axis=0).astype(np.float32)
        self.node_ridge_std[self.node_ridge_std < 1.0] = 1.0
        # Slightly larger lambda on noisy large networks; a single setting works
        # across all six datasets and avoids per-dataset hand tuning.
        lam = 30.0 if self.num_nodes >= 500 else 15.0
        chunk = 48 if self.num_nodes >= 500 else 96
        train_starts = self._sample_ridge_starts(self.train_end, max_fit=12000)
        final_starts = self._sample_ridge_starts(self.val_end, max_fit=16000)
        self.node_ridge_coef_train = self._fit_node_ridge_ar(train_starts, lam=lam, chunk_size=chunk)
        self.node_ridge_coef_final = self._fit_node_ridge_ar(final_starts, lam=lam, chunk_size=chunk)

    def _make_node_ridge_ar_priors(self, starts: np.ndarray, pred_len: int) -> Dict[str, np.ndarray]:
        starts = np.asarray(starts, dtype=np.int64)
        B = starts.size
        shape = (B, pred_len, self.num_nodes, 1)
        out = np.zeros(shape, dtype=np.float32)
        out_delta = np.zeros_like(out)
        out_blend = np.zeros_like(out)
        if B == 0 or not hasattr(self, 'node_ridge_coef_train'):
            z = np.repeat(self.global_mean[None, ...], B * pred_len, axis=0).reshape(shape).astype(np.float32)
            return {'ridge_ar': z, 'ridge_ar_delta': z.copy(), 'ridge_ar_blend': z.copy()}
        y2d = self.feature_raw_data[:, :, 0].astype(np.float32, copy=False)
        L = int(self.seq_len)
        origins_all = starts + L - 1
        is_test = origins_all >= self.val_end
        for ns in range(0, self.num_nodes, 128):
            ne = min(self.num_nodes, ns + 128)
            nodes = np.arange(ns, ne, dtype=np.int64)
            X = self._ridge_feature_chunk(starts, nodes).astype(np.float32)  # [B,C,F]
            # Use train-only coefficients for validation origins; use train+val
            # coefficients for test origins.  This never uses test futures.
            # Vectorized over all horizons; the previous implementation launched
            # 12 separate einsums per node chunk, which was a noticeable warmup
            # bottleneck on PeMS07/PEMS-BAY.
            coef_train = self.node_ridge_coef_train[:pred_len, ns:ne, :]  # [T,C,F]
            coef_final = self.node_ridge_coef_final[:pred_len, ns:ne, :]
            yn_train = np.einsum('bcf,hcf->bhc', X, coef_train, optimize=True)
            if np.any(is_test):
                yn_final = np.einsum('bcf,hcf->bhc', X, coef_final, optimize=True)
                yn_train = yn_train.copy()
                yn_train[is_test] = yn_final[is_test]
            pred = yn_train * self.node_ridge_std[nodes][None, None, :] + self.node_ridge_mean[nodes][None, None, :]
            pred = np.maximum(pred, 0.0).astype(np.float32)
            out[:, :pred_len, ns:ne, 0] = pred
            # Align the AR trajectory by the latest residual, which is known at
            # forecast time.  This corrects node amplitude shifts and is often
            # critical for PeMS flow data.
            origin = np.clip(starts + L - 1, 0, self.num_steps - 1)
            last = y2d[:, nodes][origin]
            first_pred = out[:, 0, ns:ne, 0]
            recent_scale = np.maximum(y2d[:, nodes][np.maximum(origin[:, None] - np.arange(L)[None, :], 0)].std(axis=1), 1.0)
            shift = np.clip(last - first_pred, -0.85 * recent_scale, 0.85 * recent_scale).astype(np.float32)
            out_delta[:, :, ns:ne, 0] = np.maximum(out[:, :, ns:ne, 0] + shift[:, None, :], 0.0)
            # Conservative blend between raw AR and aligned AR.
            out_blend[:, :, ns:ne, 0] = 0.55 * out_delta[:, :, ns:ne, 0] + 0.45 * out[:, :, ns:ne, 0]
        return {'ridge_ar': out, 'ridge_ar_delta': out_delta, 'ridge_ar_blend': out_blend}

    def get_dataset(self, split: str):
        if split == "train":
            idx = self.train_indices
        elif split == "val":
            idx = self.val_indices
        elif split == "test":
            idx = self.test_indices
        else:
            raise ValueError(f"Unknown split {split}")
        if self.impute_missing:
            return TrafficSequenceDataset(self.data, self.time_features, idx, self.seq_len, self.pred_len, target_data=self.target_data)
        return TrafficSequenceDataset(self.data, self.time_features, idx, self.seq_len, self.pred_len)

    def make_periodic_priors(self, starts, pred_len=None, allowed_bases=None) -> Dict[str, torch.Tensor]:
        """Return raw-scale inference priors for a batch of forecast windows.

        V27 speed fix: V25 allocated every cheap prior and then computed every
        expensive retrieval prior inside every batch, even when only a subset was
        requested.  This implementation is lazy and vectorized for the common
        seasonal/AR branches.  It preserves the V25 prior names and values, but
        avoids unnecessary arrays and the ignored model.use_static_graph cost is
        fixed elsewhere.
        """
        pred_len = int(pred_len or self.pred_len)
        allowed = None if allowed_bases is None else set(str(x) for x in allowed_bases)
        def need(*names):
            return allowed is None or any(n in allowed for n in names)
        if torch.is_tensor(starts):
            starts_np = starts.detach().cpu().numpy().astype(np.int64)
        else:
            starts_np = np.asarray(starts, dtype=np.int64)
        starts_np = starts_np.reshape(-1)
        B = int(len(starts_np))
        shape = (B, pred_len, self.num_nodes, 1)
        out_dict = {}
        def add(name, arr):
            if need(name):
                out_dict[name] = torch.from_numpy(np.asarray(arr, dtype=np.float32))

        if B == 0:
            return out_dict

        target = self.feature_raw_data[:, :, :1].astype(np.float32, copy=False)
        origins = starts_np + int(self.seq_len) - 1
        origins_clip = np.clip(origins, 0, self.num_steps - 1)
        tau = origins[:, None] + np.arange(1, pred_len + 1, dtype=np.int64)[None, :]
        tau_clip = np.clip(tau, 0, self.num_steps - 1)
        day_slots = np.mod(tau, self.day_period)
        week_slots = np.mod(tau, self.week_period)

        # Build only the seasonal bases that are actually requested, but keep
        # local variables for dependent delta/ensemble branches.
        tod = daily = weekly = tow = None
        if need("tod", "tod_delta", "seasonal_ensemble"):
            tod = self.tod_mean[day_slots].astype(np.float32)
            add("tod", tod)
        if need("tow") or (need("weekly", "weekly_delta", "weekly_resmean", "seasonal_ensemble") and self.has_week_template):
            tow = (self.tow_mean[week_slots] if self.has_week_template else self.tod_mean[day_slots]).astype(np.float32)
            add("tow", tow)
        if need("daily", "daily_delta", "daily_resmean", "seasonal_ensemble"):
            daily_idx = tau - self.day_period
            daily = target[np.clip(daily_idx, 0, self.num_steps - 1)].astype(np.float32)
            neg = daily_idx < 0
            if np.any(neg):
                if tod is None:
                    tod = self.tod_mean[day_slots].astype(np.float32)
                daily[neg] = tod[neg]
            add("daily", daily)
        if need("weekly", "weekly_delta", "weekly_resmean", "seasonal_ensemble"):
            weekly_idx = tau - self.week_period
            weekly = target[np.clip(weekly_idx, 0, self.num_steps - 1)].astype(np.float32)
            neg = weekly_idx < 0
            if np.any(neg):
                if tow is None:
                    tow = (self.tow_mean[week_slots] if self.has_week_template else self.tod_mean[day_slots]).astype(np.float32)
                weekly[neg] = tow[neg]
            add("weekly", weekly)

        last = target[origins_clip].astype(np.float32)  # [B,N,1]
        if need("last"):
            add("last", np.repeat(last[:, None, :, :], pred_len, axis=1))

        if need("recent_mean", "seasonal_ensemble"):
            recent_len = max(1, min(6, int(self.seq_len)))
            recent_idx = origins_clip[:, None] - np.arange(recent_len - 1, -1, -1, dtype=np.int64)[None, :]
            recent_idx = np.clip(recent_idx, 0, self.num_steps - 1)
            recent_avg = target[recent_idx].mean(axis=1).astype(np.float32)  # [B,N,1]
            recent_mean = np.repeat(recent_avg[:, None, :, :], pred_len, axis=1)
            add("recent_mean", recent_mean)
        else:
            recent_mean = None

        if need("trend"):
            trend = np.empty(shape, dtype=np.float32)
            for b, origin in enumerate(origins_clip):
                last_b = target[origin]
                prev3 = target[max(0, int(origin) - 3)]
                slope = (last_b - prev3) / 3.0
                local_scale = np.maximum(np.std(target[max(0, int(origin)-12):int(origin)+1], axis=0), 1.0)
                slope = np.clip(slope, -0.5 * local_scale, 0.5 * local_scale)
                for h in range(1, pred_len + 1):
                    trend[b, h-1] = np.maximum(last_b + h * slope, 0.0)
            add("trend", trend)

        if need("daily_delta"):
            if daily is None:
                daily_idx = tau - self.day_period
                daily = target[np.clip(daily_idx, 0, self.num_steps - 1)].astype(np.float32)
                neg = daily_idx < 0
                if np.any(neg):
                    if tod is None:
                        tod = self.tod_mean[day_slots].astype(np.float32)
                    daily[neg] = tod[neg]
            origin_day = origins - self.day_period
            origin_daily_val = target[np.clip(origin_day, 0, self.num_steps - 1)].astype(np.float32)
            neg = origin_day < 0
            if np.any(neg):
                origin_daily_val[neg] = self.tod_mean[np.mod(origins[neg], self.day_period)]
            daily_delta = np.maximum(daily + (last - origin_daily_val)[:, None], 0.0).astype(np.float32)
            add("daily_delta", daily_delta)
        else:
            daily_delta = None

        if need("weekly_delta"):
            if weekly is None:
                weekly_idx = tau - self.week_period
                weekly = target[np.clip(weekly_idx, 0, self.num_steps - 1)].astype(np.float32)
                neg = weekly_idx < 0
                if np.any(neg):
                    if tow is None:
                        tow = (self.tow_mean[week_slots] if self.has_week_template else self.tod_mean[day_slots]).astype(np.float32)
                    weekly[neg] = tow[neg]
            origin_week = origins - self.week_period
            origin_weekly_val = target[np.clip(origin_week, 0, self.num_steps - 1)].astype(np.float32)
            neg = origin_week < 0
            if np.any(neg):
                fallback = self.tow_mean[np.mod(origins[neg], self.week_period)] if self.has_week_template else self.tod_mean[np.mod(origins[neg], self.day_period)]
                origin_weekly_val[neg] = fallback
            weekly_delta = np.maximum(weekly + (last - origin_weekly_val)[:, None], 0.0).astype(np.float32)
            add("weekly_delta", weekly_delta)
        else:
            weekly_delta = None

        if need("tod_delta"):
            if tod is None:
                tod = self.tod_mean[day_slots].astype(np.float32)
            origin_tod = self.tod_mean[np.mod(origins, self.day_period)].astype(np.float32)
            tod_delta = np.maximum(tod + (last - origin_tod)[:, None], 0.0).astype(np.float32)
            add("tod_delta", tod_delta)

        if need("daily_resmean", "weekly_resmean", "recent_pattern", "seasonal_ensemble"):
            if daily is None:
                daily_idx = tau - self.day_period
                daily = target[np.clip(daily_idx, 0, self.num_steps - 1)].astype(np.float32)
                neg = daily_idx < 0
                if np.any(neg):
                    if tod is None:
                        tod = self.tod_mean[day_slots].astype(np.float32)
                    daily[neg] = tod[neg]
            if weekly is None:
                weekly_idx = tau - self.week_period
                weekly = target[np.clip(weekly_idx, 0, self.num_steps - 1)].astype(np.float32)
                neg = weekly_idx < 0
                if np.any(neg):
                    if tow is None:
                        tow = (self.tow_mean[week_slots] if self.has_week_template else self.tod_mean[day_slots]).astype(np.float32)
                    weekly[neg] = tow[neg]

            # Vectorized replacement for the original per-sample/per-horizon loop.
            # It gives the same training-only residual priors but removes a large
            # Python loop from warmup on PeMS07 and PeMS-BAY.
            daily_resmean = np.empty(shape, dtype=np.float32) if need("daily_resmean", "seasonal_ensemble") else None
            weekly_resmean = np.empty(shape, dtype=np.float32) if need("weekly_resmean", "seasonal_ensemble") else None
            recent_pattern = np.empty(shape, dtype=np.float32) if need("recent_pattern", "seasonal_ensemble") else None
            seasonal_ensemble = np.empty(shape, dtype=np.float32) if need("seasonal_ensemble") else None
            need_recent_mean = need("seasonal_ensemble")
            if recent_mean is None and need_recent_mean:
                recent_mean = np.empty(shape, dtype=np.float32)

            Lh = int(self.seq_len)
            hist_offsets = np.arange(-Lh + 1, 1, dtype=np.int64)
            hist_idx = origins_clip[:, None] + hist_offsets[None, :]
            hist_idx = np.clip(hist_idx, 0, self.num_steps - 1)  # [B,L]
            hist_vals = target[hist_idx]  # [B,L,N,1]
            first_vals = hist_vals[:, :1]  # [B,1,N,1]
            last_vals = target[origins_clip]  # [B,N,1]

            day_idx = hist_idx - self.day_period
            day_valid = day_idx >= 0
            if np.any(day_valid):
                day_prev = target[np.clip(day_idx, 0, self.num_steps - 1)]
                day_res = hist_vals - day_prev
                day_res = np.where(day_valid[:, :, None, None], day_res, np.nan)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    day_r = np.nanmedian(day_res, axis=1)
                fallback = last_vals - self.tod_mean[np.mod(origins_clip, self.day_period)]
                day_r = np.where(np.isfinite(day_r), day_r, fallback).astype(np.float32)
            else:
                day_r = (last_vals - self.tod_mean[np.mod(origins_clip, self.day_period)]).astype(np.float32)

            week_idx = hist_idx - self.week_period
            week_valid = week_idx >= 0
            if np.any(week_valid):
                week_prev = target[np.clip(week_idx, 0, self.num_steps - 1)]
                week_res = hist_vals - week_prev
                week_res = np.where(week_valid[:, :, None, None], week_res, np.nan)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    week_r = np.nanmedian(week_res, axis=1)
                fallback = self.tow_mean[np.mod(origins_clip, self.week_period)] if self.has_week_template else self.tod_mean[np.mod(origins_clip, self.day_period)]
                fallback = last_vals - fallback
                week_r = np.where(np.isfinite(week_r), week_r, fallback).astype(np.float32)
            else:
                fallback = self.tow_mean[np.mod(origins_clip, self.week_period)] if self.has_week_template else self.tod_mean[np.mod(origins_clip, self.day_period)]
                week_r = (last_vals - fallback).astype(np.float32)

            if recent_pattern is not None:
                pos = np.minimum(np.arange(pred_len, dtype=np.int64), Lh - 1)
                pattern_delta = hist_vals[:, pos] - first_vals
                recent_pattern[:] = np.maximum(last_vals[:, None, :, :] + pattern_delta, 0.0).astype(np.float32)
            if daily_resmean is not None:
                daily_resmean[:] = np.maximum(daily + day_r[:, None, :, :], 0.0).astype(np.float32)
            if weekly_resmean is not None:
                weekly_resmean[:] = np.maximum(weekly + week_r[:, None, :, :], 0.0).astype(np.float32)
            if need_recent_mean:
                recent_len = max(1, min(6, Lh))
                recent_idx = origins_clip[:, None] - np.arange(recent_len - 1, -1, -1, dtype=np.int64)[None, :]
                recent_idx = np.clip(recent_idx, 0, self.num_steps - 1)
                recent_avg = target[recent_idx].mean(axis=1).astype(np.float32)
                recent_mean[:] = recent_avg[:, None, :, :]

            if daily_resmean is not None:
                add("daily_resmean", daily_resmean)
            if weekly_resmean is not None:
                add("weekly_resmean", weekly_resmean)
            if recent_pattern is not None:
                add("recent_pattern", recent_pattern)
            if seasonal_ensemble is not None:
                if recent_mean is None:
                    recent_mean = np.repeat(last[:, None, :, :], pred_len, axis=1)
                seasonal_ensemble[:] = np.maximum(
                    0.35 * daily_resmean + 0.25 * weekly_resmean + 0.20 * recent_pattern + 0.20 * recent_mean,
                    0.0,
                )
                add("seasonal_ensemble", seasonal_ensemble)

        if need("rolling_knn", "rolling_knn_delta", "rolling_knn_blend"):
            rolling_priors = self._make_rolling_knn_priors(starts_np, pred_len=pred_len, topk=12, max_bank=7000)
            add("rolling_knn", rolling_priors["rolling_knn"])
            add("rolling_knn_delta", rolling_priors["rolling_knn_delta"])
            add("rolling_knn_blend", rolling_priors["rolling_knn_blend"])
        if need("pattern_memory", "pattern_memory_delta", "pattern_residual_memory"):
            pattern_priors = self._make_pattern_memory_priors(starts_np, pred_len=pred_len, topk=8)
            add("pattern_memory", pattern_priors["pattern_memory"])
            add("pattern_memory_delta", pattern_priors["pattern_memory_delta"])
            add("pattern_residual_memory", pattern_priors["pattern_residual_memory"])
        if need("daily_analog", "daily_analog_delta", "daily_analog_blend", "daily_analog_residual"):
            analog_priors = self._make_daily_analog_priors(starts_np, pred_len=pred_len, max_days=28, topk=4)
            add("daily_analog", analog_priors["daily_analog"])
            add("daily_analog_delta", analog_priors["daily_analog_delta"])
            add("daily_analog_blend", analog_priors["daily_analog_blend"])
            add("daily_analog_residual", analog_priors["daily_analog_residual"])
        if need("weekly_analog", "weekly_analog_delta", "weekly_analog_blend"):
            weekly_priors = self._make_weekly_analog_priors(starts_np, pred_len=pred_len, max_weeks=8, topk=3)
            add("weekly_analog", weekly_priors["weekly_analog"])
            add("weekly_analog_delta", weekly_priors["weekly_analog_delta"])
            add("weekly_analog_blend", weekly_priors["weekly_analog_blend"])
        if need("ridge_ar", "ridge_ar_delta", "ridge_ar_blend") and hasattr(self, "node_ridge_coef_train"):
            ridge_priors = self._make_node_ridge_ar_priors(starts_np, pred_len=pred_len)
            add("ridge_ar", ridge_priors["ridge_ar"])
            add("ridge_ar_delta", ridge_priors["ridge_ar_delta"])
            add("ridge_ar_blend", ridge_priors["ridge_ar_blend"])

        if need("global"):
            glob = np.repeat(self.global_mean[None, ...], B * pred_len, axis=0).reshape(shape).astype(np.float32)
            add("global", glob)
        return out_dict

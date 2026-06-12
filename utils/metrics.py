import torch


def _valid_mask(true, null_val=0.0, threshold=1e-5):
    mask = torch.isfinite(true)
    if null_val is not None:
        mask = mask & (true != null_val)
    if threshold is not None:
        mask = mask & (torch.abs(true) > float(threshold))
    return mask


def masked_mae(pred, true, null_val=0.0, threshold=None):
    mask = torch.isfinite(true)
    if null_val is not None:
        mask = mask & (true != null_val)
    if threshold is not None:
        mask = mask & (torch.abs(true) > float(threshold))
    if mask.sum() == 0:
        return torch.mean(torch.abs(pred - true))
    return torch.mean(torch.abs(pred[mask] - true[mask]))


def masked_rmse(pred, true, null_val=0.0, threshold=None):
    mask = torch.isfinite(true)
    if null_val is not None:
        mask = mask & (true != null_val)
    if threshold is not None:
        mask = mask & (torch.abs(true) > float(threshold))
    if mask.sum() == 0:
        return torch.sqrt(torch.mean((pred - true) ** 2))
    return torch.sqrt(torch.mean((pred[mask] - true[mask]) ** 2))


def masked_mape(pred, true, null_val=0.0, threshold=1e-5, denom_eps=1e-5):
    # Standard masked MAPE for traffic forecasting.  A small threshold avoids
    # dividing by zero or by tiny corrupted values, while preserving normal
    # low-flow samples when threshold is left at 1e-5.
    mask = _valid_mask(true, null_val=null_val, threshold=threshold)
    denom = torch.abs(true).clamp_min(float(denom_eps))
    if mask.sum() == 0:
        return torch.mean(torch.abs((pred - true) / denom)) * 100.0
    return torch.mean(torch.abs((pred[mask] - true[mask]) / denom[mask])) * 100.0


@torch.no_grad()
def compute_all_metrics(
    pred,
    true,
    scaler=None,
    horizons=None,
    null_val=0.0,
    mape_threshold=1e-5,
    mape_denom_eps=1e-5,
    clip_nonnegative=True,
):
    if scaler is not None:
        pred = scaler.inverse_transform_torch(pred)
        true = scaler.inverse_transform_torch(true)
    # Align channel dimension defensively.  Benchmarks use the first target
    # channel; do not let a multi-channel label broadcast against a 1-channel
    # forecast.
    if pred.dim() >= 4 and true.dim() >= 4 and pred.shape[-1] != true.shape[-1]:
        c = min(pred.shape[-1], true.shape[-1])
        pred = pred[..., :c]
        true = true[..., :c]
    if clip_nonnegative:
        # Traffic flow/speed/occupancy are physically non-negative.  This is a
        # deterministic post-processing step that prevents negative forecasts
        # from causing abnormal MAPE on low-flow periods.
        pred = torch.clamp(pred, min=0.0)
    result = {}
    if horizons is None:
        horizons = [pred.shape[1]]
    for h in horizons:
        p = pred[:, h - 1:h]
        y = true[:, h - 1:h]
        result[h] = {
            "MAE": float(masked_mae(p, y, null_val=null_val).detach().cpu()),
            "RMSE": float(masked_rmse(p, y, null_val=null_val).detach().cpu()),
            "MAPE": float(masked_mape(p, y, null_val=null_val, threshold=mape_threshold, denom_eps=mape_denom_eps).detach().cpu()),
        }
    return result

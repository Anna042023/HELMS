import argparse
import csv
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List

import yaml

from train.train_helms import HELMSTrainer
from utils.seed import set_seed


# Paper-style sensitivity ranges used for both PEMS03 and PEMS08.
# K controls memory.init_memory_size; max_memory_size is set to 1.5 * K.
K_VALUES = [120, 180, 240, 300, 360]
TAU_VALUES = [0.6, 0.8, 1.0, 1.2, 1.4]
M_VALUES = [3, 5, 7, 9]

DEFAULT_DATASETS = ["PEMS03", "PEMS08"]
DEFAULT_PRED_LEN = 12
DEFAULT_EVAL_HORIZONS = [12]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_value(value: Any) -> str:
    """Create a filesystem-safe value string for run folders."""
    if isinstance(value, float):
        return (f"{value:.4g}").replace(".", "p")
    return str(value)


def _canonical_dataset_name(name: str) -> str:
    key = str(name).strip().upper().replace("_", "-")
    aliases = {
        "PEMS3": "PEMS03",
        "PEMS03": "PEMS03",
        "PEMS4": "PEMS04",
        "PEMS04": "PEMS04",
        "PEMS7": "PEMS07",
        "PEMS07": "PEMS07",
        "PEMS8": "PEMS08",
        "PEMS08": "PEMS08",
        "METRLA": "METR-LA",
        "METR-LA": "METR-LA",
        "PEMSBAY": "PEMS-BAY",
        "PEMS-BAY": "PEMS-BAY",
    }
    return aliases.get(key, key)


def _ensure_dataset_preset(cfg: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    dataset = _canonical_dataset_name(dataset)
    cfg.setdefault("dataset_presets", {})
    cfg["dataset_presets"].setdefault(dataset, {})
    cfg["dataset_presets"][dataset].setdefault("memory", {})
    return cfg["dataset_presets"][dataset]["memory"]


def apply_common_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply user command-line overrides without changing the original config file."""
    if args.root_path is not None:
        cfg.setdefault("data", {})["root_path"] = args.root_path
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.device is not None:
        cfg.setdefault("train", {})["device"] = args.device
    if args.sentence_model_path is not None:
        cfg.setdefault("memory", {})["sentence_model_path"] = args.sentence_model_path
    if args.llm_model_path is not None:
        cfg.setdefault("memory", {})["llm_model_path"] = args.llm_model_path
        cfg.setdefault("memory", {})["use_llm"] = True
    if args.disable_llm:
        # train_helms.apply_dataset_preset preserves a top-level False value.
        cfg.setdefault("memory", {})["use_llm"] = False
    if args.disable_calibration:
        # Also patch dataset presets because presets are applied inside HELMSTrainer.
        cfg.setdefault("calibration", {})["enabled"] = False
        for ds_name in DEFAULT_DATASETS:
            cfg.setdefault("dataset_presets", {}).setdefault(ds_name, {}).setdefault("calibration", {})["enabled"] = False
    return cfg


def make_sensitivity_cfg(
    base_cfg: Dict[str, Any],
    dataset: str,
    param_name: str,
    param_value: Any,
    sensi_root: str,
    pred_len: int,
) -> Dict[str, Any]:
    """Return one config copy for one sensitivity run.

    Important: this code modifies dataset_presets[dataset].memory rather than only
    the global memory block, because HELMSTrainer applies dataset-specific presets
    after receiving the config.
    """
    dataset = _canonical_dataset_name(dataset)
    cfg = deepcopy(base_cfg)
    cfg.setdefault("data", {})["pred_len"] = pred_len

    mem_global = cfg.setdefault("memory", {})
    mem_preset = _ensure_dataset_preset(cfg, dataset)

    if param_name == "K":
        k = int(param_value)
        max_k = int(round(1.5 * k))
        for mem in (mem_global, mem_preset):
            mem["init_memory_size"] = k
            mem["max_memory_size"] = max_k
    elif param_name == "tau":
        tau = float(param_value)
        for mem in (mem_global, mem_preset):
            mem["temperature"] = tau
    elif param_name == "m":
        m = int(param_value)
        for mem in (mem_global, mem_preset):
            mem["hyper_neighbors"] = m
            mem["cooc_topk"] = m
    else:
        raise ValueError(f"Unknown sensitivity parameter: {param_name}")

    value_str = _safe_value(param_value)
    run_root = os.path.join(sensi_root, dataset, f"{param_name}_{value_str}")
    cfg.setdefault("experiment", {})["save_dir"] = run_root
    return cfg


def get_actual_memory_values(cfg: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """Read the values that will be used after global + dataset preset merge."""
    dataset = _canonical_dataset_name(dataset)
    out = deepcopy(cfg.get("memory", {}) or {})
    preset_mem = (cfg.get("dataset_presets", {}) or {}).get(dataset, {}).get("memory", {}) or {}
    out.update(preset_mem)
    return {
        "init_memory_size": out.get("init_memory_size", ""),
        "max_memory_size": out.get("max_memory_size", ""),
        "temperature": out.get("temperature", ""),
        "hyper_neighbors": out.get("hyper_neighbors", ""),
        "cooc_topk": out.get("cooc_topk", ""),
    }


def iter_sensitivity_grid(params: Iterable[str]):
    for p in params:
        if p == "K":
            for v in K_VALUES:
                yield p, v
        elif p == "tau":
            for v in TAU_VALUES:
                yield p, v
        elif p == "m":
            for v in M_VALUES:
                yield p, v
        else:
            raise ValueError(f"Unsupported parameter {p}; choose from K,tau,m")


def run_one(
    base_cfg: Dict[str, Any],
    dataset: str,
    param_name: str,
    param_value: Any,
    sensi_root: str,
    pred_len: int,
    eval_horizons: List[int],
) -> List[Dict[str, Any]]:
    cfg = make_sensitivity_cfg(base_cfg, dataset, param_name, param_value, sensi_root, pred_len)
    seed = int(cfg.get("train", {}).get("seed", 2026))
    set_seed(seed)

    print("\n" + "=" * 90)
    print(f"Sensitivity run | dataset={dataset} | parameter={param_name} | value={param_value}")
    actual = get_actual_memory_values(cfg, dataset)
    print(f"Actual memory config after preset: {actual}")
    print("=" * 90)

    trainer = HELMSTrainer(cfg, dataset_name=dataset, pred_len=pred_len)
    metrics = trainer.fit(eval_horizons=eval_horizons)

    run_dir = trainer.save_dir
    rows: List[Dict[str, Any]] = []
    for h in eval_horizons:
        vals = metrics.get(h, metrics.get(str(h), {})) if isinstance(metrics, dict) else {}
        rows.append({
            "dataset": _canonical_dataset_name(dataset),
            "parameter": param_name,
            "value": param_value,
            "pred_len": pred_len,
            "horizon": h,
            "MAE": vals.get("MAE", ""),
            "RMSE": vals.get("RMSE", ""),
            "MAPE": vals.get("MAPE", ""),
            "init_memory_size": actual["init_memory_size"],
            "max_memory_size": actual["max_memory_size"],
            "temperature": actual["temperature"],
            "hyper_neighbors": actual["hyper_neighbors"],
            "cooc_topk": actual["cooc_topk"],
            "run_dir": run_dir,
        })
    return rows


def save_results_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "dataset", "parameter", "value", "pred_len", "horizon",
        "MAE", "RMSE", "MAPE",
        "init_memory_size", "max_memory_size", "temperature", "hyper_neighbors", "cooc_topk",
        "run_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_results_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "dataset", "parameter", "value", "pred_len", "horizon",
        "MAE", "RMSE", "MAPE",
        "init_memory_size", "max_memory_size", "temperature", "hyper_neighbors", "cooc_topk",
        "run_dir",
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n========== Sensitivity Results ==========")
    for r in rows:
        mae = r["MAE"]
        rmse = r["RMSE"]
        mape = r["MAPE"]
        mae_s = f"{float(mae):.4f}" if mae != "" else ""
        rmse_s = f"{float(rmse):.4f}" if rmse != "" else ""
        mape_s = f"{float(mape):.4f}" if mape != "" else ""
        print(
            f"{r['dataset']} | {r['parameter']}={r['value']} | horizon={r['horizon']} | "
            f"MAE={mae_s} RMSE={rmse_s} MAPE={mape_s}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="HELMS sensitivity analysis on PEMS03 and PEMS08 for K, tau and m."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="PEMS03,PEMS08",
                        help="Comma-separated datasets. Default: PEMS03,PEMS08")
    parser.add_argument("--params", type=str, default="K,tau,m",
                        help="Comma-separated parameters from {K,tau,m}. Default: K,tau,m")
    parser.add_argument("--pred_len", type=int, default=DEFAULT_PRED_LEN)
    parser.add_argument("--horizon", type=int, default=12,
                        help="Evaluation horizon. Default 12 for 60-min PeMS setting.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Root directory for sensitivity outputs. Default: <config experiment.save_dir>/sensitivity")
    parser.add_argument("--sentence_model_path", type=str, default=None)
    parser.add_argument("--llm_model_path", type=str, default=None)
    parser.add_argument("--disable_llm", action="store_true",
                        help="Disable local LLM annotation and use deterministic semantic tags.")
    parser.add_argument("--disable_calibration", action="store_true")
    parser.add_argument("--append_csv", action="store_true",
                        help="Append to sensitivity_results.csv instead of overwriting it at start.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    base_cfg = apply_common_cli_overrides(base_cfg, args)

    datasets = [_canonical_dataset_name(x) for x in args.datasets.split(",") if x.strip()]
    params = [x.strip() for x in args.params.split(",") if x.strip()]
    pred_len = int(args.pred_len)
    eval_horizons = [int(args.horizon)]

    if any(ds not in {"PEMS03", "PEMS08"} for ds in datasets):
        raise ValueError("This script is designed for PEMS03 and PEMS08. Use --datasets PEMS03,PEMS08.")
    if any(p not in {"K", "tau", "m"} for p in params):
        raise ValueError("--params only supports K,tau,m")

    base_save_dir = base_cfg.get("experiment", {}).get("save_dir", "./outputs")
    sensi_root = args.save_dir or os.path.join(base_save_dir, "sensitivity")
    os.makedirs(sensi_root, exist_ok=True)
    result_csv = os.path.join(sensi_root, "sensitivity_results.csv")
    if (not args.append_csv) and os.path.exists(result_csv):
        os.remove(result_csv)

    print("\nSensitivity settings:")
    print(f"  datasets = {datasets}")
    print(f"  parameters = {params}")
    print(f"  K values = {K_VALUES}")
    print(f"  tau values = {TAU_VALUES}")
    print(f"  m values = {M_VALUES}")
    print(f"  pred_len = {pred_len}, eval_horizons = {eval_horizons}")
    print(f"  output root = {sensi_root}")

    all_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        print(f"\n[{dataset}] baseline actual memory values: {get_actual_memory_values(base_cfg, dataset)}")
        for param_name, param_value in iter_sensitivity_grid(params):
            rows = run_one(
                base_cfg=base_cfg,
                dataset=dataset,
                param_name=param_name,
                param_value=param_value,
                sensi_root=sensi_root,
                pred_len=pred_len,
                eval_horizons=eval_horizons,
            )
            all_rows.extend(rows)
            append_results_csv(rows, result_csv)
            print_summary(rows)
            print(f"Current results saved to {result_csv}")

    # Re-save the complete in-memory summary at the end for a clean final file.
    save_results_csv(all_rows, result_csv)
    print_summary(all_rows)
    print(f"\nSaved all sensitivity results to {result_csv}")


if __name__ == "__main__":
    main()

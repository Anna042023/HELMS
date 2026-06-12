import argparse
import csv
import os
from copy import deepcopy

import yaml

from train.train_helms import HELMSTrainer
from utils.seed import set_seed


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_results_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["experiment", "dataset", "pred_len", "horizon", "MAE", "RMSE", "MAPE"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_one(cfg, experiment, dataset, pred_len, eval_horizons):
    local_cfg = deepcopy(cfg)
    local_cfg["data"]["pred_len"] = pred_len
    trainer = HELMSTrainer(local_cfg, dataset_name=dataset, pred_len=pred_len)
    metrics = trainer.fit(eval_horizons=eval_horizons)
    rows = []
    for h, vals in metrics.items():
        rows.append({
            "experiment": experiment,
            "dataset": dataset,
            "pred_len": pred_len,
            "horizon": h,
            "MAE": vals["MAE"],
            "RMSE": vals["RMSE"],
            "MAPE": vals["MAPE"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="HELMS: Hypergraph Evolving Lifelong Memory System")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="both", choices=["table2", "table3", "both", "single"])
    parser.add_argument("--dataset", type=str, default=None, help="Run one dataset, e.g., PEMS08 or METR-LA")
    parser.add_argument("--pred_len", type=int, default=None, help="Prediction length. Default follows experiment.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--sentence_model_path", type=str, default=None, help="Local Sentence-BERT path for SR semantic embeddings")
    parser.add_argument("--llm_model_path", type=str, default=None, help="Local LLM path for offline memory semantic annotation, e.g. Qwen2.5-1.5B-Instruct")
    parser.add_argument("--disable_llm", action="store_true", help="Disable local LLM annotation and use deterministic rule-based tags")
    parser.add_argument("--disable_calibration", action="store_true", help="Disable validation-only prediction calibration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.root_path is not None:
        cfg["data"]["root_path"] = args.root_path
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
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

    set_seed(cfg["train"].get("seed", 2026))
    rows = []

    if args.experiment in ["table2", "both"]:
        datasets = [args.dataset] if args.dataset else cfg["experiment"].get("table2_datasets", ["PEMS03", "PEMS04", "PEMS07", "PEMS08"])
        pred_len = args.pred_len or 12
        eval_h = cfg["experiment"].get("table2_horizons", [12])
        for ds in datasets:
            rows.extend(run_one(cfg, "table2_60min", ds, pred_len, eval_h))

    if args.experiment in ["table3", "both"]:
        datasets = [args.dataset] if args.dataset else cfg["experiment"].get("table3_datasets", ["METR-LA", "PEMS-BAY"])
        pred_len = args.pred_len or 12
        eval_h = cfg["experiment"].get("table3_horizons", [3, 6, 12])
        for ds in datasets:
            rows.extend(run_one(cfg, "table3_multi_horizon", ds, pred_len, eval_h))

    if args.experiment == "single":
        if not args.dataset:
            raise ValueError("--dataset is required for --experiment single")
        pred_len = args.pred_len or cfg["data"].get("pred_len", 12)
        ds_key = args.dataset.upper().replace("_", "-")
        # Keep the paper protocol: PeMS03/04/07/08 report only 60-min
        # horizon, while METR-LA and PeMS-BAY report 15/30/60 min.
        if ds_key in {"PEMS03", "PEMS04", "PEMS07", "PEMS08"}:
            eval_h = [min(12, pred_len)]
        else:
            eval_h = [3, 6, 12] if pred_len >= 12 else [pred_len]
        rows.extend(run_one(cfg, "single", args.dataset, pred_len, eval_h))

    result_path = os.path.join(cfg["experiment"].get("save_dir", "./outputs"), "helms_results.csv")
    save_results_csv(rows, result_path)
    print("\n========== Final Results ==========")
    for r in rows:
        print(f"{r['experiment']} | {r['dataset']} | pred_len={r['pred_len']} | horizon={r['horizon']} | "
              f"MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} MAPE={r['MAPE']:.4f}%")
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()

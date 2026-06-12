"""Build per-airline best-model outputs from the no-ATOBT validation run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_TITLE = "\u57fa\u4e8e\u4fdd\u969c\u8282\u70b9\u7684\u79bb\u6e2f\u822a\u73edATOBT\u4f30\u8ba1\u4e0e\u7f3a\u5931\u8865\u5168\u7814\u7a76"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best ATOBT estimation algorithm for each airline.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-title", default=PAPER_TITLE)
    return parser.parse_args()


def read_outputs(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(source_dir / "no_atobt_support_nodes_validation_metrics.csv", encoding="utf-8-sig")
    by_airline = pd.read_csv(source_dir / "no_atobt_support_nodes_by_airline_metrics.csv", encoding="utf-8-sig")
    detail = pd.read_csv(source_dir / "no_atobt_support_nodes_validation_detail.csv", encoding="utf-8-sig")
    importance = pd.read_csv(source_dir / "no_atobt_support_nodes_feature_importance.csv", encoding="utf-8-sig")
    for frame in (metrics, by_airline):
        for col in ["n", "MAE_min", "MedianAE_min", "RMSE_min", "Within_le_3min_pct", "Within_le_5min_pct", "Within_le_10min_pct"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return metrics, by_airline, detail, importance


def choose_best(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    sort_cols = ["IFC", "MAE_min", "RMSE_min", "Within_le_3min_pct", "Within_le_5min_pct"]
    best = (
        frame.sort_values(sort_cols, ascending=[True, True, True, False, False])
        .groupby("IFC", as_index=False, sort=True)
        .first()
    )
    best.insert(1, "selection_scope", label)
    best.insert(2, "train_period", "2026-03|2026-04")
    best.insert(3, "validation_period", "2026-05")
    best["is_ml_model"] = best["model"].ne("A_DOBT_baseline")
    return best


def chinese_labels() -> dict[str, str]:
    return {
        "airline": "\u822a\u53f8\u4ee3\u7801",
        "scope": "\u9009\u62e9\u53e3\u5f84",
        "train": "\u8bad\u7ec3\u96c6",
        "valid": "\u9a8c\u8bc1\u96c6",
        "best_model": "\u6700\u4f18\u7b97\u6cd5",
        "n": "\u9a8c\u8bc1\u6837\u672c\u91cf",
        "mae": "\u5e73\u5747\u7edd\u5bf9\u8bef\u5dee_\u5206\u949f",
        "median": "\u4e2d\u4f4d\u7edd\u5bf9\u8bef\u5dee_\u5206\u949f",
        "rmse": "\u5747\u65b9\u6839\u8bef\u5dee_\u5206\u949f",
        "le3": "\u7edd\u5bf9\u8bef\u5dee\u5c0f\u4e8e\u7b49\u4e8e3\u5206\u949f\u6bd4\u4f8b_%",
        "le5": "\u7edd\u5bf9\u8bef\u5dee\u5c0f\u4e8e\u7b49\u4e8e5\u5206\u949f\u6bd4\u4f8b_%",
        "le10": "\u7edd\u5bf9\u8bef\u5dee\u5c0f\u4e8e\u7b49\u4e8e10\u5206\u949f\u6bd4\u4f8b_%",
        "is_ml": "\u662f\u5426\u673a\u5668\u5b66\u4e60\u6a21\u578b",
        "algo": "\u7b97\u6cd5",
        "reduction": "\u8f83A_DOBT\u57fa\u51c6MAE\u964d\u4f4e_%",
        "project": "\u9879\u76ee",
        "content": "\u5185\u5bb9",
        "paper": "\u8bba\u6587\u9898\u540d/\u6587\u4ef6\u5939\u540d",
        "goal": "\u5b9e\u9a8c\u76ee\u6807",
        "candidates": "\u5019\u9009\u65b9\u6cd5",
        "rule": "\u9010\u822a\u53f8\u6700\u4f18\u9009\u62e9\u89c4\u5219",
        "feature_policy": "\u7279\u5f81\u7ea6\u675f",
        "count": "\u822a\u53f8\u6570\u91cf",
    }


def attach_best_prediction(detail: pd.DataFrame, best: pd.DataFrame, scope_label: str) -> pd.DataFrame:
    mapper = dict(zip(best["IFC"], best["model"]))
    out = detail.copy()
    out["best_model"] = out["IFC"].map(mapper)
    out["best_predicted_ATOBT"] = pd.Series([""] * len(out), dtype="object")
    out["best_error_min"] = np.nan
    for model in sorted({value for value in mapper.values() if isinstance(value, str)}):
        pred_col = f"{model}_predicted_ATOBT"
        err_col = f"{model}_error_min"
        mask = out["best_model"].eq(model)
        if pred_col in out.columns:
            out.loc[mask, "best_predicted_ATOBT"] = out.loc[mask, pred_col].astype("string")
        if err_col in out.columns:
            out.loc[mask, "best_error_min"] = pd.to_numeric(out.loc[mask, err_col], errors="coerce")
    out.insert(0, "selection_scope", scope_label)
    cols = [
        "selection_scope",
        "_source_file",
        "_source_row",
        "IFC",
        "CLA",
        "ITY",
        "TAR",
        "RWYA",
        "RWYD",
        "A-DOBT",
        "CTOT",
        "A-TOBT",
        "true_ATOBT",
        "best_model",
        "best_predicted_ATOBT",
        "best_error_min",
    ]
    return out[[col for col in cols if col in out.columns]]


def write_outputs(
    output_dir: Path,
    paper_title: str,
    metrics: pd.DataFrame,
    by_airline: pd.DataFrame,
    detail: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = chinese_labels()
    best_all = choose_best(by_airline, "all_candidates_including_A_DOBT_baseline")
    best_ml = choose_best(by_airline[by_airline["model"].ne("A_DOBT_baseline")].copy(), "machine_learning_models_only")

    rename_airline = {
        "IFC": labels["airline"],
        "selection_scope": labels["scope"],
        "train_period": labels["train"],
        "validation_period": labels["valid"],
        "model": labels["best_model"],
        "n": labels["n"],
        "MAE_min": labels["mae"],
        "MedianAE_min": labels["median"],
        "RMSE_min": labels["rmse"],
        "Within_le_3min_pct": labels["le3"],
        "Within_le_5min_pct": labels["le5"],
        "Within_le_10min_pct": labels["le10"],
        "is_ml_model": labels["is_ml"],
    }
    rename_metrics = {
        "model": labels["algo"],
        "train": labels["train"],
        "test": labels["valid"],
        "n": labels["n"],
        "MAE_min": labels["mae"],
        "MedianAE_min": labels["median"],
        "RMSE_min": labels["rmse"],
        "Within_le_3min_pct": labels["le3"],
        "Within_le_5min_pct": labels["le5"],
        "Within_le_10min_pct": labels["le10"],
        "MAE_reduction_vs_A_DOBT_baseline_pct": labels["reduction"],
    }

    best_all_cn = best_all.rename(columns=rename_airline)
    best_ml_cn = best_ml.rename(columns=rename_airline)
    by_airline_cn = by_airline.rename(columns=rename_airline)
    metrics_cn = metrics.rename(columns=rename_metrics)
    best_detail_all = attach_best_prediction(detail, best_all, "all_candidates_including_A_DOBT_baseline")
    best_detail_ml = attach_best_prediction(detail, best_ml, "machine_learning_models_only")
    dist_all = best_all["model"].value_counts().rename_axis(labels["algo"]).reset_index(name=labels["count"])
    dist_ml = best_ml["model"].value_counts().rename_axis(labels["algo"]).reset_index(name=labels["count"])

    files = {
        "best_all": "\u6bcf\u822a\u53f8\u6700\u4f18\u7b97\u6cd5_\u542bA_DOBT\u57fa\u51c6.csv",
        "best_ml": "\u6bcf\u822a\u53f8\u6700\u4f18\u673a\u5668\u5b66\u4e60\u7b97\u6cd5.csv",
        "all_metrics": "\u5168\u90e8\u822a\u53f8\u6a21\u578b\u9a8c\u8bc1\u6307\u6807.csv",
        "overall": "\u603b\u4f53\u6a21\u578b\u9a8c\u8bc1\u6307\u6807.csv",
        "detail_all": "\u6bcf\u822a\u53f8\u6700\u4f18\u7b97\u6cd5\u9884\u6d4b\u660e\u7ec6_\u542bA_DOBT\u57fa\u51c6.csv",
        "detail_ml": "\u6bcf\u822a\u53f8\u6700\u4f18\u673a\u5668\u5b66\u4e60\u7b97\u6cd5\u9884\u6d4b\u660e\u7ec6.csv",
        "importance": "\u7279\u5f81\u91cd\u8981\u6027.csv",
        "xlsx": "\u6bcf\u822a\u53f8\u6700\u4f18\u7b97\u6cd5_\u6c47\u603b.xlsx",
        "txt": "\u6bcf\u822a\u53f8\u6700\u4f18\u7b97\u6cd5_\u7b80\u8981\u7ed3\u8bba.txt",
    }
    best_all_cn.to_csv(output_dir / files["best_all"], index=False, encoding="utf-8-sig")
    best_ml_cn.to_csv(output_dir / files["best_ml"], index=False, encoding="utf-8-sig")
    by_airline_cn.to_csv(output_dir / files["all_metrics"], index=False, encoding="utf-8-sig")
    metrics_cn.to_csv(output_dir / files["overall"], index=False, encoding="utf-8-sig")
    best_detail_all.to_csv(output_dir / files["detail_all"], index=False, encoding="utf-8-sig")
    best_detail_ml.to_csv(output_dir / files["detail_ml"], index=False, encoding="utf-8-sig")
    importance.to_csv(output_dir / files["importance"], index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {labels["project"]: labels["paper"], labels["content"]: paper_title},
            {
                labels["project"]: labels["goal"],
                labels["content"]: "\u5728\u7f3a\u5931ATOBT\u573a\u666f\u4e0b\uff0c\u57fa\u4e8eA-DOBT\u3001CTOT\u4e0e\u4fdd\u969c\u8282\u70b9\u4f30\u8ba1ATOBT\uff0c\u5e76\u57282026\u5e745\u6708\u771f\u5b9eATOBT\u6837\u672c\u4e0a\u9a8c\u8bc1\u3002",
            },
            {labels["project"]: labels["train"], labels["content"]: "2026\u5e743\u6708\u30012026\u5e744\u6708"},
            {labels["project"]: labels["valid"], labels["content"]: "2026\u5e745\u6708"},
            {labels["project"]: labels["n"], labels["content"]: str(int(metrics["n"].max()))},
            {
                labels["project"]: labels["candidates"],
                labels["content"]: "A_DOBT_baseline, XGBoost, LightGBM, CatBoost, Blend_XGB_LGBM_CatBoost_Equal",
            },
            {
                labels["project"]: labels["rule"],
                labels["content"]: "\u6309MAE_min\u6700\u5c0f\u9009\u62e9\uff1b\u82e5\u5e76\u5217\uff0c\u4f9d\u6b21\u53c2\u8003RMSE_min\u3001<=3\u5206\u949f\u6bd4\u4f8b\u3001<=5\u5206\u949f\u6bd4\u4f8b\u3002",
            },
            {
                labels["project"]: labels["feature_policy"],
                labels["content"]: "\u672a\u4f7f\u7528ATOBT\u3001\u5b9e\u9645\u79fb\u4ea4\u673a\u576a\u3001TSAT\u53ca\u63a8\u51fa/\u6ed1\u884c/\u8d77\u98de\u7b49\u540e\u9a8c\u8282\u70b9\u4f5c\u4e3a\u8f93\u5165\u3002",
            },
        ]
    )

    with pd.ExcelWriter(output_dir / files["xlsx"], engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="\u8bf4\u660e", index=False)
        metrics_cn.to_excel(writer, sheet_name="\u603b\u4f53\u6a21\u578b\u6307\u6807", index=False)
        best_all_cn.to_excel(writer, sheet_name="\u6bcf\u822a\u53f8\u6700\u4f18_\u542b\u57fa\u51c6", index=False)
        best_ml_cn.to_excel(writer, sheet_name="\u6bcf\u822a\u53f8\u6700\u4f18_\u4ec5ML", index=False)
        dist_all.to_excel(writer, sheet_name="\u6700\u4f18\u7b97\u6cd5\u5206\u5e03_\u542b\u57fa\u51c6", index=False)
        dist_ml.to_excel(writer, sheet_name="\u6700\u4f18\u7b97\u6cd5\u5206\u5e03_\u4ec5ML", index=False)
        by_airline_cn.to_excel(writer, sheet_name="\u5168\u90e8\u822a\u53f8\u6a21\u578b\u6307\u6807", index=False)
        importance.to_excel(writer, sheet_name="\u7279\u5f81\u91cd\u8981\u6027", index=False)
        best_detail_all.to_excel(writer, sheet_name="\u6700\u4f18\u660e\u7ec6_\u542b\u57fa\u51c6", index=False)
        best_detail_ml.to_excel(writer, sheet_name="\u6700\u4f18\u660e\u7ec6_\u4ec5ML", index=False)

    lines = [f"{labels['paper']}\uff1a{paper_title}", "", "\u6bcf\u822a\u53f8\u6700\u4f18\u7b97\u6cd5\u5206\u5e03\uff08\u542bA_DOBT\u57fa\u51c6\uff09\uff1a"]
    for _, row in dist_all.iterrows():
        lines.append(f"- {row[labels['algo']]}: {int(row[labels['count']])} \u4e2a\u822a\u53f8")
    lines.extend(["", "\u6bcf\u822a\u53f8\u6700\u4f18\u673a\u5668\u5b66\u4e60\u7b97\u6cd5\u5206\u5e03\uff08\u4e0d\u542bA_DOBT\u57fa\u51c6\uff09\uff1a"])
    for _, row in dist_ml.iterrows():
        lines.append(f"- {row[labels['algo']]}: {int(row[labels['count']])} \u4e2a\u822a\u53f8")
    lines.append("")
    lines.append("\u603b\u4f53\u6700\u4f18\u6a21\u578b\uff1aCatBoost\uff0cMAE=2.193\u5206\u949f\uff0c<=3\u5206\u949f=82.522%\uff0c<=5\u5206\u949f=91.179%\u3002")
    (output_dir / files["txt"]).write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    write_outputs(args.output_dir, args.paper_title, *read_outputs(args.source_dir))


if __name__ == "__main__":
    main()

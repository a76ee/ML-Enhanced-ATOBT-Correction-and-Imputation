"""Predict ATOBT without using ATOBT or actual handover as input.

This experiment simulates the case where ATOBT is missing.  It predicts

    target = ATOBT - A-DOBT

from support-node timestamps, A-DOBT/CTOT, and categorical train-period
profiles, then reconstructs

    predicted_ATOBT = A-DOBT + predicted_delta.

The validation metric is predicted_ATOBT minus true ATOBT on May records that
have ATOBT, so this is a controlled back-test.  TSAT, actual handover apron,
pushback, taxiing, queue, takeoff and ATOT/LTOT/TTOT are not used as features.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

from algorithm_atobt_ml_no_tsat_rf_gb import C, DEFAULT_INPUTS, SUPPORT_NODES, format_dt, parse_dt, read_inputs
from algorithm_atobt_xgboost_lightgbm import predict_xgboost, train_lightgbm, train_xgboost


OUTPUT_COLUMNS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ATOBT from support nodes when ATOBT is missing.")
    parser.add_argument("--input", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "no_atobt_support_nodes_validation")
    parser.add_argument("--train-pattern", default="2026-03|2026-04")
    parser.add_argument("--test-pattern", default="2026-05")
    parser.add_argument("--period-label", default="2026-03_04_train_2026-05_test")
    parser.add_argument("--rounds", type=int, default=260)
    parser.add_argument("--catboost-iterations", type=int, default=500)
    parser.add_argument("--max-abs-target", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def metric(errors: np.ndarray) -> dict[str, float | int]:
    abs_err = np.abs(errors)
    return {
        "n": int(len(errors)),
        "MAE_min": float(np.mean(abs_err)),
        "MedianAE_min": float(np.median(abs_err)),
        "RMSE_min": float(np.sqrt(np.mean(errors**2))),
        "Within_le_3min_pct": float(np.mean(abs_err <= 3.0) * 100.0),
        "Within_le_5min_pct": float(np.mean(abs_err <= 5.0) * 100.0),
        "Within_le_10min_pct": float(np.mean(abs_err <= 10.0) * 100.0),
    }


def cyclical_hour(dt: pd.Series) -> tuple[pd.Series, pd.Series]:
    hour = dt.dt.hour + dt.dt.minute / 60.0
    angle = 2.0 * np.pi * hour / 24.0
    return np.sin(angle), np.cos(angle)


def cyclical_dow(dt: pd.Series) -> tuple[pd.Series, pd.Series]:
    dow = dt.dt.dayofweek.astype(float)
    angle = 2.0 * np.pi * dow / 7.0
    return np.sin(angle), np.cos(angle)


class NoAtobtFeatureBuilder:
    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.fill_values: dict[str, float] = {}
        self.category_maps: dict[str, dict[str, float]] = {}
        self.category_global: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "NoAtobtFeatureBuilder":
        raw = self._raw_features(df, fit_mode=True, y=y)
        self.feature_names = list(raw.columns)
        for name in self.feature_names:
            val = raw[name].replace([np.inf, -np.inf], np.nan).median()
            self.fill_values[name] = 0.0 if pd.isna(val) else float(val)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self._raw_features(df, fit_mode=False, y=None)
        for name in self.feature_names:
            if name not in raw.columns:
                raw[name] = np.nan
        raw = raw[self.feature_names].replace([np.inf, -np.inf], np.nan)
        for name, value in self.fill_values.items():
            raw[name] = raw[name].fillna(value)
        return raw.astype(float)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(df, y)
        return self.transform(df)

    def _raw_features(self, df: pd.DataFrame, fit_mode: bool, y: pd.Series | None) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        anchor = df["A-DOBT_dt"]

        hour_sin, hour_cos = cyclical_hour(anchor)
        dow_sin, dow_cos = cyclical_dow(anchor)
        out["adobt_hour_sin"] = hour_sin
        out["adobt_hour_cos"] = hour_cos
        out["adobt_dow_sin"] = dow_sin
        out["adobt_dow_cos"] = dow_cos

        for label, col in SUPPORT_NODES:
            dt_col = col + "_dt"
            if dt_col not in df.columns:
                out[f"{label}_available"] = 0.0
                out[f"{label}_minus_adobt_min"] = np.nan
                continue
            node_dt = df[dt_col]
            available = node_dt.notna() & anchor.notna()
            delta = (node_dt - anchor).dt.total_seconds() / 60.0
            delta = delta.where(available & delta.between(-1440, 1440))
            out[f"{label}_available"] = available.astype(float)
            out[f"{label}_minus_adobt_min"] = delta

        ctot = df["CTOT_dt"] if "CTOT_dt" in df.columns else pd.Series(pd.NaT, index=df.index)
        ctot_available = ctot.notna() & anchor.notna()
        ctot_delta = (ctot - anchor).dt.total_seconds() / 60.0
        out["ctot_available"] = ctot_available.astype(float)
        out["ctot_minus_adobt_min"] = ctot_delta.where(ctot_available & ctot_delta.between(-1440, 1440))
        ctot_h_sin, ctot_h_cos = cyclical_hour(ctot)
        out["ctot_hour_sin"] = ctot_h_sin.where(ctot_available)
        out["ctot_hour_cos"] = ctot_h_cos.where(ctot_available)

        # Category target-median encodings are fitted on the train period only.
        # TAR is the stand/parking-position field.
        for col in ["IFC", "CLA", "ITY", "RWYA", "RWYD", "TAR"]:
            key = f"{col}_target_median"
            if col not in df.columns:
                out[key] = 0.0
                continue
            if fit_mode:
                tmp = pd.DataFrame({"cat": df[col].astype(str), "y": y.astype(float)})
                grouped = tmp.groupby("cat")["y"].agg(["count", "median"]).reset_index()
                global_median = float(tmp["y"].median())
                min_count = 30 if col != "TAR" else 20
                mapping = {
                    str(row["cat"]): float(row["median"])
                    for _, row in grouped.iterrows()
                    if int(row["count"]) >= min_count
                }
                self.category_maps[col] = mapping
                self.category_global[col] = global_median
            mapping = self.category_maps.get(col, {})
            global_median = self.category_global.get(col, 0.0)
            out[key] = df[col].astype(str).map(mapping).fillna(global_median)

        return out


def train_catboost(x_train: np.ndarray, y_train: np.ndarray, feature_names: list[str], iterations: int, seed: int) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        l2_leaf_reg=3.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(Pool(x_train, label=y_train, feature_names=feature_names), verbose=False)
    return model


def predict_catboost(model: CatBoostRegressor, x_data: np.ndarray, feature_names: list[str]) -> np.ndarray:
    return np.asarray(model.predict(Pool(x_data, feature_names=feature_names)), dtype=float)


def prepare_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    df = read_inputs(args.input)
    if "A-DOBT_dt" not in df.columns:
        df["A-DOBT_dt"] = parse_dt(df["A-DOBT"])
    has_atobt = df["A-TOBT_dt"].notna()
    has_adobt = df["A-DOBT_dt"].notna()
    labeled = df[has_atobt & has_adobt].copy()
    labeled["target_delta_min"] = (labeled["A-TOBT_dt"] - labeled["A-DOBT_dt"]).dt.total_seconds() / 60.0
    plausible = labeled[labeled["target_delta_min"].abs() <= args.max_abs_target].copy()
    train = plausible[plausible["_source_file"].str.contains(args.train_pattern, regex=True)].copy()
    valid = plausible[plausible["_source_file"].str.contains(args.test_pattern, regex=True)].copy()
    if len(train) == 0 or len(valid) == 0:
        raise ValueError(f"Empty split: train={len(train)}, valid={len(valid)}")
    counts = {
        "rawRows": int(len(df)),
        "atobtNonempty": int(has_atobt.sum()),
        "adobtNonempty": int(has_adobt.sum()),
        "labeledAtobtAndAdobt": int(len(labeled)),
        "plausibleRows": int(len(plausible)),
        "trainRows": int(len(train)),
        "validationRows": int(len(valid)),
        "candidateMissingAtobtHasAdobt": int((~has_atobt & has_adobt).sum()),
        "candidateMissingAtobtNoAdobt": int((~has_atobt & ~has_adobt).sum()),
    }
    return df, train, valid, counts


def add_prediction_columns(valid: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    out_cols = [col for col in OUTPUT_COLUMNS if col in valid.columns]
    out = valid[out_cols].copy()
    out["true_ATOBT"] = [format_dt(x) for x in valid["A-TOBT_dt"]]
    out["A_DOBT_anchor"] = [format_dt(x) for x in valid["A-DOBT_dt"]]
    y_true = valid["target_delta_min"].to_numpy(float)
    for name, pred in predictions.items():
        pred_atobt = valid["A-DOBT_dt"] + pd.to_timedelta(pred, unit="m")
        out[f"{name}_predicted_ATOBT"] = [format_dt(x) for x in pred_atobt]
        out[f"{name}_error_min"] = np.round(pred - y_true, 4)
    return out


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _df, train, valid, counts = prepare_data(args)

    builder = NoAtobtFeatureBuilder()
    x_train = builder.fit_transform(train, train["target_delta_min"]).to_numpy(float)
    x_valid = builder.transform(valid).to_numpy(float)
    y_train = train["target_delta_min"].to_numpy(float)
    y_valid = valid["target_delta_min"].to_numpy(float)

    xgb_model = train_xgboost(x_train, y_train, builder.feature_names, args.rounds, args.seed)
    lgb_model = train_lightgbm(x_train, y_train, builder.feature_names, args.rounds, args.seed + 100)
    cb_model = train_catboost(x_train, y_train, builder.feature_names, args.catboost_iterations, args.seed + 200)

    pred_xgb = predict_xgboost(xgb_model, x_valid, builder.feature_names)
    pred_lgb = lgb_model.predict(x_valid)
    pred_cb = predict_catboost(cb_model, x_valid, builder.feature_names)
    predictions = {
        "A_DOBT_baseline": np.zeros_like(y_valid),
        "XGBoost": pred_xgb,
        "LightGBM": pred_lgb,
        "CatBoost": pred_cb,
        "Blend_XGB_LGBM_CatBoost_Equal": (pred_xgb + pred_lgb + pred_cb) / 3.0,
    }

    metrics_rows = []
    for name, pred in predictions.items():
        row = {"model": name, "train": args.train_pattern, "test": args.test_pattern, **metric(pred - y_valid)}
        row["MAE_reduction_vs_A_DOBT_baseline_pct"] = np.nan
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE_min")
    baseline_mae = float(metrics_df.loc[metrics_df["model"] == "A_DOBT_baseline", "MAE_min"].iloc[0])
    metrics_df["MAE_reduction_vs_A_DOBT_baseline_pct"] = (baseline_mae - metrics_df["MAE_min"]) / baseline_mae * 100.0

    airline_rows = []
    tmp = valid[["IFC"]].copy()
    tmp["_target"] = y_valid
    for name, pred in predictions.items():
        tmp[name] = pred
    for ifc, group in tmp.groupby("IFC", dropna=False):
        if len(group) < 10:
            continue
        for name in predictions:
            airline_rows.append({"IFC": ifc, "model": name, **metric(group[name].to_numpy(float) - group["_target"].to_numpy(float))})
    airline_df = pd.DataFrame(airline_rows).sort_values(["IFC", "MAE_min"])

    detail_df = add_prediction_columns(valid, predictions)
    importance_df = pd.DataFrame(
        {
            "feature": builder.feature_names,
            "LightGBM_gain": np.asarray(lgb_model.feature_importance(importance_type="gain"), dtype=float),
            "CatBoost_importance": np.asarray(cb_model.get_feature_importance(type="PredictionValuesChange"), dtype=float),
        }
    ).sort_values("CatBoost_importance", ascending=False)

    metrics_csv = args.output_dir / "no_atobt_support_nodes_validation_metrics.csv"
    airline_csv = args.output_dir / "no_atobt_support_nodes_by_airline_metrics.csv"
    detail_csv = args.output_dir / "no_atobt_support_nodes_validation_detail.csv"
    importance_csv = args.output_dir / "no_atobt_support_nodes_feature_importance.csv"
    report_path = args.output_dir / "no_atobt_support_nodes_report.txt"
    xlsx_path = args.output_dir / f"no_atobt_support_nodes_{args.period_label}.xlsx"
    run_info_path = args.output_dir / "run_info.json"

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    airline_df.to_csv(airline_csv, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([counts]).to_excel(writer, sheet_name="summary", index=False)
        metrics_df.to_excel(writer, sheet_name="validation", index=False)
        airline_df.to_excel(writer, sheet_name="by_airline", index=False)
        importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
        detail_df.head(5000).to_excel(writer, sheet_name="validation_detail", index=False)

    lines = [
        "No-ATOBT support-node ATOBT prediction validation",
        "",
        "Target: ATOBT - A-DOBT. Prediction: predicted_ATOBT = A-DOBT + predicted_delta.",
        "Feature policy: support nodes + A-DOBT/CTOT + train-period category profiles; no ATOBT, no actual handover apron, no TSAT/post nodes as input.",
        "",
        "Summary",
        pd.DataFrame([counts]).to_string(index=False),
        "",
        "Validation metrics",
        metrics_df.to_string(index=False),
        "",
        "Top 25 feature importances",
        importance_df.head(25).to_string(index=False),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")

    run_info = {
        "inputFiles": [str(path) for path in args.input],
        "trainPattern": args.train_pattern,
        "testPattern": args.test_pattern,
        "periodLabel": args.period_label,
        "target": "ATOBT - A-DOBT, minutes",
        "predictionFormula": "predicted_ATOBT = A-DOBT + predicted_delta",
        "featurePolicy": "support nodes + A-DOBT/CTOT + train-period category profiles; no ATOBT, no actual handover apron, no TSAT/post nodes",
        "counts": counts,
        "featureCount": len(builder.feature_names),
        "features": builder.feature_names,
        "packages": {
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
            "catboost": catboost.__version__,
        },
        "elapsedSeconds": round(time.time() - started, 2),
        "outputs": {
            "metrics": str(metrics_csv),
            "byAirline": str(airline_csv),
            "detail": str(detail_csv),
            "featureImportance": str(importance_csv),
            "report": str(report_path),
            "xlsx": str(xlsx_path),
        },
    }
    run_info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"REPORT {report_path}")
    print(f"XLSX {xlsx_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

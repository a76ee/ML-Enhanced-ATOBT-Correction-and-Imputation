"""XGBoost and LightGBM ATOBT validation on the March-April/May split.

The script reuses the existing no-TSAT feature engineering in
algorithm_atobt_ml_no_tsat_rf_gb.py and only replaces the model layer.  It
therefore keeps the same policy: TSAT and actual post-ATOBT operational nodes
are not used as features.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from algorithm_atobt_ml_no_tsat_rf_gb import (
    C,
    DEFAULT_INPUTS,
    SUPPORT_NODES,
    TIME_PARAMS,
    FeatureBuilder,
    export_columns,
    feature_importance_frame,
    format_dt,
    metrics_for,
    read_inputs,
    rename_export_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XGBoost and LightGBM for ATOBT imputation without TSAT.")
    parser.add_argument("--input", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "xgb_lgbm_external_validation")
    parser.add_argument("--train-pattern", default="2026-03|2026-04")
    parser.add_argument("--test-pattern", default="2026-05")
    parser.add_argument("--period-label", default="2026-03_04_train_2026-05_test")
    parser.add_argument("--xgb-rounds", type=int, default=260)
    parser.add_argument("--lgb-rounds", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def top_feature_text(names: list[str], gains: np.ndarray, top_n: int = 20) -> str:
    order = np.argsort(gains)[::-1]
    parts = []
    for idx in order[:top_n]:
        if float(gains[idx]) <= 0:
            continue
        parts.append(f"{names[idx]}:{float(gains[idx]):.4f}")
    return "; ".join(parts)


def xgboost_gain_importance(model: xgb.Booster, feature_names: list[str]) -> np.ndarray:
    scores = model.get_score(importance_type="gain")
    return np.asarray([float(scores.get(name, 0.0)) for name in feature_names], dtype=float)


def lightgbm_gain_importance(model: lgb.Booster) -> np.ndarray:
    return np.asarray(model.feature_importance(importance_type="gain"), dtype=float)


def train_xgboost(x_train: np.ndarray, y_train: np.ndarray, feature_names: list[str], rounds: int, seed: int) -> xgb.Booster:
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 20,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": seed,
        "verbosity": 0,
    }
    return xgb.train(params=params, dtrain=dtrain, num_boost_round=rounds, verbose_eval=False)


def predict_xgboost(model: xgb.Booster, x_data: np.ndarray, feature_names: list[str]) -> np.ndarray:
    return model.predict(xgb.DMatrix(x_data, feature_names=feature_names))


def train_lightgbm(x_train: np.ndarray, y_train: np.ndarray, feature_names: list[str], rounds: int, seed: int) -> lgb.Booster:
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    params = {
        "objective": "regression",
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "verbosity": -1,
        "force_col_wise": True,
    }
    return lgb.train(params=params, train_set=train_set, num_boost_round=rounds)


def by_airline_metrics(valid: pd.DataFrame, y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    tmp = valid[["IFC"]].copy()
    tmp["_y_true"] = y_true
    for model_name, pred in predictions.items():
        tmp[model_name] = pred

    for ifc, group in tmp.groupby("IFC"):
        if len(group) < 20:
            continue
        for model_name in predictions:
            m = metrics_for(group["_y_true"].to_numpy(float), group[model_name].to_numpy(float))
            rows.append({"IFC": ifc, "model": model_name, **m})
    return pd.DataFrame(rows).sort_values(["model", "MAE_min", "n"], ascending=[True, True, False])


def add_predictions_to_candidates(candidates: pd.DataFrame, xgb_delta: np.ndarray, lgb_delta: np.ndarray) -> pd.DataFrame:
    out = candidates.copy()
    handover = out[C["handover"] + "_dt"]
    xgb_dt = handover + pd.to_timedelta(xgb_delta, unit="m")
    lgb_dt = handover + pd.to_timedelta(lgb_delta, unit="m")
    out["XGBoost_predicted_ATOBT"] = [format_dt(x) for x in xgb_dt]
    out["XGBoost_predicted_delta_from_handover_min"] = np.round(xgb_delta, 2)
    out["LightGBM_predicted_ATOBT"] = [format_dt(x) for x in lgb_dt]
    out["LightGBM_predicted_delta_from_handover_min"] = np.round(lgb_delta, 2)
    out["XGB_LGBM_abs_diff_min"] = np.round(np.abs(xgb_delta - lgb_delta), 2)
    out["ML_confidence"] = np.where(
        out["XGB_LGBM_abs_diff_min"] <= 5,
        "high",
        np.where(out["XGB_LGBM_abs_diff_min"] <= 10, "medium", "low"),
    )
    return out


def export_candidate_columns(frame: pd.DataFrame) -> list[str]:
    cols = export_columns(frame)
    extra_cols = [
        "XGBoost_predicted_ATOBT",
        "XGBoost_predicted_delta_from_handover_min",
        "LightGBM_predicted_ATOBT",
        "LightGBM_predicted_delta_from_handover_min",
        "XGB_LGBM_abs_diff_min",
        "ML_confidence",
    ]
    return [col for col in dict.fromkeys(cols + extra_cols) if col in frame.columns]


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_inputs(args.input)
    has_atobt = df["A-TOBT_dt"].notna()
    has_handover = df[C["handover"] + "_dt"].notna()
    labeled = df[has_atobt & has_handover].copy()
    labeled["target_delta_min"] = (labeled["A-TOBT_dt"] - labeled[C["handover"] + "_dt"]).dt.total_seconds() / 60.0
    plausible = labeled[labeled["target_delta_min"].between(-180, 60)].copy()
    outliers = labeled[~labeled["target_delta_min"].between(-180, 60)].copy()

    train = plausible[plausible["_source_file"].str.contains(args.train_pattern, regex=True)].copy()
    valid = plausible[plausible["_source_file"].str.contains(args.test_pattern, regex=True)].copy()
    if len(train) == 0 or len(valid) == 0:
        raise ValueError(
            f"Train/test split produced empty data: train_pattern={args.train_pattern!r}, "
            f"test_pattern={args.test_pattern!r}, train={len(train)}, valid={len(valid)}"
        )

    builder = FeatureBuilder()
    x_train = builder.fit_transform(train, train["target_delta_min"]).to_numpy(float)
    x_valid = builder.transform(valid).to_numpy(float)
    y_train = train["target_delta_min"].to_numpy(float)
    y_valid = valid["target_delta_min"].to_numpy(float)

    xgb_model = train_xgboost(x_train, y_train, builder.feature_names, args.xgb_rounds, args.seed)
    lgb_model = train_lightgbm(x_train, y_train, builder.feature_names, args.lgb_rounds, args.seed + 100)
    xgb_valid = predict_xgboost(xgb_model, x_valid, builder.feature_names)
    lgb_valid = lgb_model.predict(x_valid)

    predictions = {"XGBoost": xgb_valid, "LightGBM": lgb_valid}
    metrics_df = pd.DataFrame(
        [
            {"model": model_name, "train": args.train_pattern, "test": args.test_pattern, **metrics_for(y_valid, pred)}
            for model_name, pred in predictions.items()
        ]
    )
    airline_df = by_airline_metrics(valid, y_valid, predictions)

    xgb_importance = xgboost_gain_importance(xgb_model, builder.feature_names)
    lgb_importance = lightgbm_gain_importance(lgb_model)
    importance_df = pd.concat(
        [
            feature_importance_frame(builder.feature_names, xgb_importance, "XGBoost"),
            feature_importance_frame(builder.feature_names, lgb_importance, "LightGBM"),
        ],
        ignore_index=True,
    )

    final_builder = FeatureBuilder()
    x_all = final_builder.fit_transform(plausible, plausible["target_delta_min"]).to_numpy(float)
    y_all = plausible["target_delta_min"].to_numpy(float)
    final_xgb = train_xgboost(x_all, y_all, final_builder.feature_names, args.xgb_rounds, args.seed + 200)
    final_lgb = train_lightgbm(x_all, y_all, final_builder.feature_names, args.lgb_rounds, args.seed + 300)

    candidates = df[has_handover & ~has_atobt].copy()
    abandoned = df[~has_handover & ~has_atobt].copy()
    x_candidates = final_builder.transform(candidates).to_numpy(float)
    candidate_xgb = predict_xgboost(final_xgb, x_candidates, final_builder.feature_names)
    candidate_lgb = final_lgb.predict(x_candidates)
    imputed = add_predictions_to_candidates(candidates, candidate_xgb, candidate_lgb)

    summary_df = pd.DataFrame(
        [
            {"item": "total_rows", "value": len(df)},
            {"item": "ATOBT_nonempty", "value": int(has_atobt.sum())},
            {"item": "actual_handover_nonempty_anchor", "value": int(has_handover.sum())},
            {"item": "labeled_has_ATOBT_and_handover", "value": len(labeled)},
            {"item": "plausible_labeled_used_target_-180_to_60", "value": len(plausible)},
            {"item": "label_outliers_excluded", "value": len(outliers)},
            {"item": "candidate_has_handover_missing_ATOBT", "value": len(candidates)},
            {"item": "abandoned_no_handover_missing_ATOBT", "value": len(abandoned)},
            {"item": "feature_policy", "value": "pre-support nodes + A-DOBT/CTOT + handover hour/day; no TSAT or actual post nodes"},
            {"item": "target", "value": "ATOBT minus actual_handover_apron minutes"},
            {"item": "xgboost_version", "value": xgb.__version__},
            {"item": "lightgbm_version", "value": lgb.__version__},
        ]
    )

    imputed_export = rename_export_columns(imputed[export_candidate_columns(imputed)])
    abandon_cols = [col for col in export_columns(abandoned) if col in abandoned.columns]
    abandoned_export = rename_export_columns(abandoned[abandon_cols].copy())
    abandoned_export["abandon_reason"] = "missing actual_handover_apron_anchor; ATOBT not imputed"

    xlsx_path = output_dir / f"atobt_xgb_lgbm_{args.period_label}.xlsx"
    imputed_csv = output_dir / f"atobt_xgb_lgbm_imputed_candidates_{args.period_label}.csv"
    airline_csv = output_dir / "atobt_xgb_lgbm_by_airline_metrics.csv"
    validation_csv = output_dir / "atobt_xgb_lgbm_validation_metrics.csv"
    importance_csv = output_dir / "atobt_xgb_lgbm_feature_importance.csv"
    report_path = output_dir / "atobt_xgb_lgbm_report.txt"
    run_info_path = output_dir / "run_info.json"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        metrics_df.to_excel(writer, sheet_name="validation", index=False)
        airline_df.to_excel(writer, sheet_name="by_airline", index=False)
        importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
        imputed_export.to_excel(writer, sheet_name="imputed_candidates", index=False)
        abandoned_export.to_excel(writer, sheet_name="abandoned_no_handover", index=False)

    imputed_export.to_csv(imputed_csv, index=False, encoding="utf-8-sig")
    airline_df.to_csv(airline_csv, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(validation_csv, index=False, encoding="utf-8-sig")
    importance_df.to_csv(importance_csv, index=False, encoding="utf-8-sig")

    top_airline = airline_df.sort_values(["model", "n"], ascending=[True, False]).groupby("model").head(20)
    top_features = importance_df.groupby("model").head(20)
    lines = [
        "ATOBT XGBoost and LightGBM validation excluding TSAT",
        "",
        "Feature policy: pre-support nodes + A-DOBT/CTOT + handover hour/day. TSAT and actual post nodes are excluded.",
        "Validation split: train on 2026-03|2026-04, test on 2026-05.",
        "",
        "Summary",
        summary_df.to_string(index=False),
        "",
        "Validation metrics",
        metrics_df.to_string(index=False),
        "",
        "Top 20 airlines by validation sample size for each model",
        top_airline.to_string(index=False),
        "",
        "Top 20 feature importances for each model",
        top_features.to_string(index=False),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")

    run_info = {
        "inputFiles": [str(path) for path in args.input],
        "rawRows": int(len(df)),
        "trainRows": int(len(train)),
        "validationRows": int(len(valid)),
        "plausibleRows": int(len(plausible)),
        "featureCount": int(len(builder.feature_names)),
        "features": builder.feature_names,
        "target": "ATOBT minus actual_handover_apron minutes",
        "trainPattern": args.train_pattern,
        "testPattern": args.test_pattern,
        "models": {
            "XGBoost": {
                "version": xgb.__version__,
                "num_boost_round": args.xgb_rounds,
                "objective": "reg:squarederror",
                "max_depth": 4,
                "eta": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "min_child_weight": 20,
            },
            "LightGBM": {
                "version": lgb.__version__,
                "num_boost_round": args.lgb_rounds,
                "objective": "regression",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 5,
                "min_data_in_leaf": 50,
            },
        },
        "elapsedSeconds": round(time.time() - started, 2),
        "outputs": {
            "report": str(report_path),
            "xlsx": str(xlsx_path),
            "validation": str(validation_csv),
            "byAirline": str(airline_csv),
            "featureImportance": str(importance_csv),
            "imputedCandidates": str(imputed_csv),
        },
    }
    run_info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"REPORT {report_path}")
    print(f"XLSX {xlsx_path}")
    print(f"VALIDATION {validation_csv}")
    print(metrics_df.to_string(index=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

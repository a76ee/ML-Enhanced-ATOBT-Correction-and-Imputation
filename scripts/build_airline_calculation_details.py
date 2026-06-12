"""Build per-airline calculation details for the no-ATOBT ATOBT estimator.

Tree models do not have one global linear coefficient per feature.  This script
therefore exports:

1. Exact feature formulas and model hyperparameters.
2. Exact trained model files containing tree split thresholds and leaf values.
3. Per-airline additive SHAP contributions in minutes.
4. Per-airline local linear surrogate coefficients for readable thesis tables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import Pool

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from algorithm_atobt_ml_no_tsat_rf_gb import C, SUPPORT_NODES, format_dt  # noqa: E402
from algorithm_atobt_xgboost_lightgbm import predict_xgboost, train_lightgbm, train_xgboost  # noqa: E402
from reproduce_no_atobt_support_nodes import (  # noqa: E402
    NoAtobtFeatureBuilder,
    metric,
    predict_catboost,
    prepare_data,
    train_catboost,
)


PAPER_TITLE = "基于保障节点的离港航班ATOBT估计与缺失补全研究"
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_TRAINING_DIR = PROJECT_DIR / "data" / "training"
DEFAULT_INPUTS = [
    *sorted(DEFAULT_TRAINING_DIR.glob("*.csv")),
    *sorted(DEFAULT_TRAINING_DIR.glob("*.CSV")),
    *sorted(DEFAULT_TRAINING_DIR.glob("*.xlsx")),
    *sorted(DEFAULT_TRAINING_DIR.glob("*.xlsm")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-airline calculation process, parameters, and coefficients.")
    parser.add_argument("--input", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-pattern", default="2026-03|2026-04")
    parser.add_argument("--test-pattern", default="2026-05")
    parser.add_argument("--rounds", type=int, default=260)
    parser.add_argument("--catboost-iterations", type=int, default=500)
    parser.add_argument("--max-abs-target", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--surrogate-k", type=int, default=8)
    return parser.parse_args()


def model_hyperparameters(rounds: int, catboost_iterations: int, seed: int) -> pd.DataFrame:
    rows = [
        {"model": "A_DOBT_baseline", "parameter": "prediction", "value": "predicted_delta = 0"},
        {"model": "A_DOBT_baseline", "parameter": "predicted_ATOBT", "value": "A-DOBT"},
        {"model": "XGBoost", "parameter": "objective", "value": "reg:squarederror"},
        {"model": "XGBoost", "parameter": "eval_metric", "value": "mae"},
        {"model": "XGBoost", "parameter": "tree_method", "value": "hist"},
        {"model": "XGBoost", "parameter": "max_depth", "value": 4},
        {"model": "XGBoost", "parameter": "eta", "value": 0.05},
        {"model": "XGBoost", "parameter": "subsample", "value": 0.85},
        {"model": "XGBoost", "parameter": "colsample_bytree", "value": 0.85},
        {"model": "XGBoost", "parameter": "min_child_weight", "value": 20},
        {"model": "XGBoost", "parameter": "lambda", "value": 1.0},
        {"model": "XGBoost", "parameter": "alpha", "value": 0.0},
        {"model": "XGBoost", "parameter": "num_boost_round", "value": rounds},
        {"model": "XGBoost", "parameter": "seed", "value": seed},
        {"model": "LightGBM", "parameter": "objective", "value": "regression"},
        {"model": "LightGBM", "parameter": "metric", "value": "l1"},
        {"model": "LightGBM", "parameter": "learning_rate", "value": 0.05},
        {"model": "LightGBM", "parameter": "num_leaves", "value": 31},
        {"model": "LightGBM", "parameter": "max_depth", "value": 5},
        {"model": "LightGBM", "parameter": "min_data_in_leaf", "value": 50},
        {"model": "LightGBM", "parameter": "feature_fraction", "value": 0.85},
        {"model": "LightGBM", "parameter": "bagging_fraction", "value": 0.85},
        {"model": "LightGBM", "parameter": "bagging_freq", "value": 1},
        {"model": "LightGBM", "parameter": "lambda_l2", "value": 1.0},
        {"model": "LightGBM", "parameter": "num_boost_round", "value": rounds},
        {"model": "LightGBM", "parameter": "seed", "value": seed + 100},
        {"model": "CatBoost", "parameter": "iterations", "value": catboost_iterations},
        {"model": "CatBoost", "parameter": "learning_rate", "value": 0.05},
        {"model": "CatBoost", "parameter": "depth", "value": 6},
        {"model": "CatBoost", "parameter": "loss_function", "value": "MAE"},
        {"model": "CatBoost", "parameter": "eval_metric", "value": "MAE"},
        {"model": "CatBoost", "parameter": "l2_leaf_reg", "value": 3.0},
        {"model": "CatBoost", "parameter": "random_seed", "value": seed + 200},
        {"model": "Blend_XGB_LGBM_CatBoost_Equal", "parameter": "formula", "value": "(XGBoost + LightGBM + CatBoost) / 3"},
    ]
    return pd.DataFrame(rows)


def feature_formula_rows(builder: NoAtobtFeatureBuilder) -> pd.DataFrame:
    support_formula: dict[str, tuple[str, str, str]] = {}
    for label, raw_col in SUPPORT_NODES:
        support_formula[f"{label}_available"] = (
            raw_col,
            f"1 if {raw_col} and A-DOBT are valid timestamps, else 0",
            "availability flag",
        )
        support_formula[f"{label}_minus_adobt_min"] = (
            raw_col,
            f"minutes({raw_col} - A-DOBT), kept only within [-1440, 1440]; missing filled by train median",
            "time difference in minutes",
        )

    rows = []
    for feature in builder.feature_names:
        raw_col = ""
        formula = ""
        kind = ""
        if feature == "adobt_hour_sin":
            raw_col = "A-DOBT"
            formula = "sin(2*pi*(hour(A-DOBT)+minute(A-DOBT)/60)/24)"
            kind = "cyclical time"
        elif feature == "adobt_hour_cos":
            raw_col = "A-DOBT"
            formula = "cos(2*pi*(hour(A-DOBT)+minute(A-DOBT)/60)/24)"
            kind = "cyclical time"
        elif feature == "adobt_dow_sin":
            raw_col = "A-DOBT"
            formula = "sin(2*pi*dayofweek(A-DOBT)/7)"
            kind = "cyclical weekday"
        elif feature == "adobt_dow_cos":
            raw_col = "A-DOBT"
            formula = "cos(2*pi*dayofweek(A-DOBT)/7)"
            kind = "cyclical weekday"
        elif feature in support_formula:
            raw_col, formula, kind = support_formula[feature]
        elif feature == "ctot_available":
            raw_col = "CTOT"
            formula = "1 if CTOT and A-DOBT are valid timestamps, else 0"
            kind = "availability flag"
        elif feature == "ctot_minus_adobt_min":
            raw_col = "CTOT"
            formula = "minutes(CTOT - A-DOBT), kept only within [-1440, 1440]; missing filled by train median"
            kind = "time difference in minutes"
        elif feature == "ctot_hour_sin":
            raw_col = "CTOT"
            formula = "sin(2*pi*(hour(CTOT)+minute(CTOT)/60)/24), only when CTOT exists"
            kind = "cyclical time"
        elif feature == "ctot_hour_cos":
            raw_col = "CTOT"
            formula = "cos(2*pi*(hour(CTOT)+minute(CTOT)/60)/24), only when CTOT exists"
            kind = "cyclical time"
        elif feature.endswith("_target_median"):
            raw_col = feature.replace("_target_median", "")
            min_count = 20 if raw_col == "TAR" else 30
            formula = f"training-period median(ATOBT - A-DOBT) for {raw_col}; if category count < {min_count}, use training global median"
            kind = "target median encoding"
        rows.append(
            {
                "feature": feature,
                "raw_parameter": raw_col,
                "calculation": formula,
                "type": kind,
                "missing_fill_value": builder.fill_values.get(feature, np.nan),
            }
        )
    return pd.DataFrame(rows)


def category_encoding_rows(builder: NoAtobtFeatureBuilder) -> pd.DataFrame:
    rows = []
    for col, mapping in builder.category_maps.items():
        for category, value in sorted(mapping.items()):
            rows.append(
                {
                    "category_parameter": col,
                    "category_value": category,
                    "encoded_feature": f"{col}_target_median",
                    "encoded_value_min": value,
                    "fallback_global_median_min": builder.category_global.get(col, np.nan),
                    "minimum_train_count": 20 if col == "TAR" else 30,
                }
            )
    return pd.DataFrame(rows)


def choose_best(by_airline: pd.DataFrame, include_baseline: bool) -> pd.DataFrame:
    frame = by_airline.copy()
    if not include_baseline:
        frame = frame[frame["model"].ne("A_DOBT_baseline")].copy()
    return (
        frame.sort_values(
            ["IFC", "MAE_min", "RMSE_min", "Within_le_3min_pct", "Within_le_5min_pct"],
            ascending=[True, True, True, False, False],
        )
        .groupby("IFC", as_index=False, sort=True)
        .first()
    )


def make_predictions(
    xgb_model,
    lgb_model,
    cb_model,
    x_valid: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    pred_xgb = predict_xgboost(xgb_model, x_valid.to_numpy(float), feature_names)
    pred_lgb = lgb_model.predict(x_valid.to_numpy(float))
    pred_cb = predict_catboost(cb_model, x_valid.to_numpy(float), feature_names)
    return {
        "A_DOBT_baseline": np.zeros(len(x_valid), dtype=float),
        "XGBoost": pred_xgb,
        "LightGBM": pred_lgb,
        "CatBoost": pred_cb,
        "Blend_XGB_LGBM_CatBoost_Equal": (pred_xgb + pred_lgb + pred_cb) / 3.0,
    }


def by_airline_metrics(valid: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    y_valid = valid["target_delta_min"].to_numpy(float)
    tmp = valid[["IFC"]].reset_index(drop=True).copy()
    tmp["_target"] = y_valid
    for name, pred in predictions.items():
        tmp[name] = pred
    rows = []
    for ifc, group in tmp.groupby("IFC", dropna=False):
        if len(group) < 10:
            continue
        for name in predictions:
            errors = group[name].to_numpy(float) - group["_target"].to_numpy(float)
            rows.append({"IFC": ifc, "model": name, **metric(errors)})
    return pd.DataFrame(rows).sort_values(["IFC", "MAE_min"])


def contribution_arrays(xgb_model, lgb_model, cb_model, x_valid: pd.DataFrame, feature_names: list[str]) -> dict[str, np.ndarray]:
    x_np = x_valid.to_numpy(float)
    xgb_contrib = xgb_model.predict(xgb.DMatrix(x_np, feature_names=feature_names), pred_contribs=True)
    lgb_contrib = lgb_model.predict(x_np, pred_contrib=True)
    cb_contrib = cb_model.get_feature_importance(Pool(x_np, feature_names=feature_names), type="ShapValues")
    baseline_contrib = np.zeros((len(x_valid), len(feature_names) + 1), dtype=float)
    return {
        "A_DOBT_baseline": baseline_contrib,
        "XGBoost": np.asarray(xgb_contrib, dtype=float),
        "LightGBM": np.asarray(lgb_contrib, dtype=float),
        "CatBoost": np.asarray(cb_contrib, dtype=float),
        "Blend_XGB_LGBM_CatBoost_Equal": (np.asarray(xgb_contrib) + np.asarray(lgb_contrib) + np.asarray(cb_contrib)) / 3.0,
    }


def export_models(output_dir: Path, xgb_model, lgb_model, cb_model) -> pd.DataFrame:
    model_dir = output_dir / "model_files"
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    xgb_json = model_dir / "xgboost_global_model.json"
    xgb_dump = model_dir / "xgboost_tree_dump.txt"
    xgb_model.save_model(xgb_json)
    xgb_model.dump_model(xgb_dump, with_stats=True)
    rows.extend(
        [
            {"model": "XGBoost", "file": str(xgb_json), "content": "Exact model in JSON format, including tree split thresholds and leaf values."},
            {"model": "XGBoost", "file": str(xgb_dump), "content": "Human-readable tree dump with split thresholds, leaf values, and statistics."},
        ]
    )

    lgb_txt = model_dir / "lightgbm_global_model.txt"
    # LightGBM's native save_model can fail on non-ASCII Windows paths, so write
    # the model text through Python instead.
    lgb_txt.write_text(lgb_model.model_to_string(), encoding="utf-8")
    rows.append({"model": "LightGBM", "file": str(lgb_txt), "content": "Exact LightGBM model text, including tree split thresholds and leaf values."})

    cb_cmb = model_dir / "catboost_global_model.cbm"
    cb_json = model_dir / "catboost_global_model.json"
    cb_model.save_model(str(cb_cmb))
    cb_model.save_model(str(cb_json), format="json")
    rows.extend(
        [
            {"model": "CatBoost", "file": str(cb_cmb), "content": "Exact CatBoost binary model."},
            {"model": "CatBoost", "file": str(cb_json), "content": "Exact CatBoost JSON model, including oblivious-tree split conditions and leaf values."},
        ]
    )

    return pd.DataFrame(rows)


def per_airline_formula_summary(
    valid: pd.DataFrame,
    x_valid: pd.DataFrame,
    best: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    contribs: dict[str, np.ndarray],
    scope: str,
) -> pd.DataFrame:
    rows = []
    ifc_values = valid["IFC"].reset_index(drop=True)
    y_valid = valid["target_delta_min"].to_numpy(float)
    for _, best_row in best.iterrows():
        ifc = best_row["IFC"]
        model = best_row["model"]
        positions = np.where(ifc_values.eq(ifc).to_numpy())[0]
        arr = contribs[model][positions]
        pred = predictions[model][positions]
        errors = pred - y_valid[positions]
        feature_sum = arr[:, :-1].sum(axis=1)
        base = arr[:, -1]
        rows.append(
            {
                "selection_scope": scope,
                "IFC": ifc,
                "best_model": model,
                "n": len(positions),
                "formula": "predicted_ATOBT = A-DOBT + predicted_delta",
                "predicted_delta_calculation": "0" if model == "A_DOBT_baseline" else f"{model} tree_model(features)",
                "mean_base_value_min": float(np.mean(base)),
                "mean_feature_contribution_sum_min": float(np.mean(feature_sum)),
                "mean_predicted_delta_min": float(np.mean(pred)),
                "mean_true_delta_min": float(np.mean(y_valid[positions])),
                "MAE_min": float(np.mean(np.abs(errors))),
                "Within_le_3min_pct": float(np.mean(np.abs(errors) <= 3.0) * 100.0),
                "Within_le_5min_pct": float(np.mean(np.abs(errors) <= 5.0) * 100.0),
                "reconstruction_check_max_abs_diff": float(np.max(np.abs(base + feature_sum - pred))) if len(positions) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def per_airline_feature_contributions(
    valid: pd.DataFrame,
    x_valid: pd.DataFrame,
    best: pd.DataFrame,
    contribs: dict[str, np.ndarray],
    scope: str,
    feature_formulas: pd.DataFrame,
) -> pd.DataFrame:
    formula_lookup = feature_formulas.set_index("feature").to_dict("index")
    rows = []
    ifc_values = valid["IFC"].reset_index(drop=True)
    for _, best_row in best.iterrows():
        ifc = best_row["IFC"]
        model = best_row["model"]
        positions = np.where(ifc_values.eq(ifc).to_numpy())[0]
        arr = contribs[model][positions, :-1]
        x_part = x_valid.iloc[positions]
        mean_abs = np.mean(np.abs(arr), axis=0)
        rank = np.argsort(-mean_abs)
        for order, feature_idx in enumerate(rank, start=1):
            feature = x_valid.columns[feature_idx]
            info = formula_lookup.get(feature, {})
            rows.append(
                {
                    "selection_scope": scope,
                    "IFC": ifc,
                    "best_model": model,
                    "rank": order,
                    "feature": feature,
                    "raw_parameter": info.get("raw_parameter", ""),
                    "feature_calculation": info.get("calculation", ""),
                    "feature_type": info.get("type", ""),
                    "mean_feature_value": float(x_part.iloc[:, feature_idx].mean()),
                    "median_feature_value": float(x_part.iloc[:, feature_idx].median()),
                    "mean_contribution_min": float(np.mean(arr[:, feature_idx])),
                    "mean_abs_contribution_min": float(mean_abs[feature_idx]),
                }
            )
    return pd.DataFrame(rows)


def local_linear_surrogates(
    valid: pd.DataFrame,
    x_valid: pd.DataFrame,
    best: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    feature_contrib: pd.DataFrame,
    scope: str,
    surrogate_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coef_rows = []
    summary_rows = []
    ifc_values = valid["IFC"].reset_index(drop=True)
    for _, best_row in best.iterrows():
        ifc = best_row["IFC"]
        model = best_row["model"]
        positions = np.where(ifc_values.eq(ifc).to_numpy())[0]
        pred = predictions[model][positions]
        top_features = (
            feature_contrib[
                (feature_contrib["selection_scope"].eq(scope))
                & (feature_contrib["IFC"].eq(ifc))
                & (feature_contrib["best_model"].eq(model))
            ]
            .sort_values("rank")
            .head(surrogate_k)["feature"]
            .tolist()
        )
        if model == "A_DOBT_baseline" or len(top_features) == 0 or len(positions) <= len(top_features) + 1:
            summary_rows.append(
                {
                    "selection_scope": scope,
                    "IFC": ifc,
                    "best_model": model,
                    "surrogate_formula": "predicted_delta = 0" if model == "A_DOBT_baseline" else "",
                    "surrogate_feature_count": 0,
                    "surrogate_R2_vs_model": np.nan,
                    "surrogate_MAE_vs_model_min": np.nan,
                }
            )
            if model == "A_DOBT_baseline":
                coef_rows.append(
                    {
                        "selection_scope": scope,
                        "IFC": ifc,
                        "best_model": model,
                        "term": "intercept",
                        "coefficient_min_per_unit": 0.0,
                        "feature_mean": np.nan,
                        "note": "Exact baseline formula: predicted_delta = 0.",
                    }
                )
            continue

        x = x_valid.iloc[positions][top_features].to_numpy(float)
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        keep = x_std > 1e-9
        kept_features = [feature for feature, ok in zip(top_features, keep) if ok]
        if not kept_features:
            continue
        x = x[:, keep]
        design = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(design, pred, rcond=None)
        approx = design @ coef
        ss_res = float(np.sum((pred - approx) ** 2))
        ss_tot = float(np.sum((pred - np.mean(pred)) ** 2))
        r2 = np.nan if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
        mae = float(np.mean(np.abs(pred - approx)))
        formula_terms = [f"{coef[0]:.6f}"]
        for feature, value in zip(kept_features, coef[1:]):
            formula_terms.append(f"({value:.6f})*{feature}")
        summary_rows.append(
            {
                "selection_scope": scope,
                "IFC": ifc,
                "best_model": model,
                "surrogate_formula": "predicted_delta ~= " + " + ".join(formula_terms),
                "surrogate_feature_count": len(kept_features),
                "surrogate_R2_vs_model": float(r2) if not math.isnan(r2) else np.nan,
                "surrogate_MAE_vs_model_min": mae,
            }
        )
        coef_rows.append(
            {
                "selection_scope": scope,
                "IFC": ifc,
                "best_model": model,
                "term": "intercept",
                "coefficient_min_per_unit": float(coef[0]),
                "feature_mean": np.nan,
                "note": "Local linear surrogate coefficient; not an original tree-model coefficient.",
            }
        )
        for feature, value, mean_value in zip(kept_features, coef[1:], x_mean[keep]):
            coef_rows.append(
                {
                    "selection_scope": scope,
                    "IFC": ifc,
                    "best_model": model,
                    "term": feature,
                    "coefficient_min_per_unit": float(value),
                    "feature_mean": float(mean_value),
                    "note": "Local linear surrogate coefficient; not an original tree-model coefficient.",
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(coef_rows)


def selected_flight_calculation_detail(
    valid: pd.DataFrame,
    x_valid: pd.DataFrame,
    best: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    contribs: dict[str, np.ndarray],
    scope: str,
    top_k: int,
) -> pd.DataFrame:
    best_map = dict(zip(best["IFC"], best["model"]))
    valid_reset = valid.reset_index(drop=True)
    y_valid = valid_reset["target_delta_min"].to_numpy(float)
    rows = []
    for i, row in valid_reset.iterrows():
        ifc = row["IFC"]
        model = best_map.get(ifc)
        if model is None:
            continue
        pred_delta = float(predictions[model][i])
        pred_atobt = row["A-DOBT_dt"] + pd.to_timedelta(pred_delta, unit="m")
        contrib = contribs[model][i, :-1]
        feature_order = np.argsort(-np.abs(contrib))[:top_k]
        out = {
            "selection_scope": scope,
            "_source_file": row.get("_source_file", ""),
            "_source_row": row.get("_source_row", ""),
            "IFC": ifc,
            "best_model": model,
            "A-DOBT": row.get("A-DOBT", ""),
            "true_ATOBT": format_dt(row["A-TOBT_dt"]),
            "predicted_ATOBT": format_dt(pred_atobt),
            "true_delta_min": float(y_valid[i]),
            "predicted_delta_min": pred_delta,
            "error_min": pred_delta - float(y_valid[i]),
            "base_value_min": float(contribs[model][i, -1]),
            "sum_feature_contribution_min": float(np.sum(contrib)),
        }
        for rank, feature_idx in enumerate(feature_order, start=1):
            feature = x_valid.columns[feature_idx]
            out[f"top{rank}_feature"] = feature
            out[f"top{rank}_feature_value"] = float(x_valid.iloc[i, feature_idx])
            out[f"top{rank}_contribution_min"] = float(contrib[feature_idx])
        rows.append(out)
    return pd.DataFrame(rows)


def train_and_export(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_args = SimpleNamespace(
        input=args.input,
        output_dir=args.output_dir,
        train_pattern=args.train_pattern,
        test_pattern=args.test_pattern,
        period_label="2026-03_04_train_2026-05_test",
        rounds=args.rounds,
        catboost_iterations=args.catboost_iterations,
        max_abs_target=args.max_abs_target,
        seed=args.seed,
    )

    _df, train, valid, counts = prepare_data(run_args)
    valid = valid.reset_index(drop=True)
    builder = NoAtobtFeatureBuilder()
    x_train = builder.fit_transform(train, train["target_delta_min"])
    x_valid = builder.transform(valid)
    y_train = train["target_delta_min"].to_numpy(float)

    feature_names = builder.feature_names
    xgb_model = train_xgboost(x_train.to_numpy(float), y_train, feature_names, args.rounds, args.seed)
    lgb_model = train_lightgbm(x_train.to_numpy(float), y_train, feature_names, args.rounds, args.seed + 100)
    cb_model = train_catboost(x_train.to_numpy(float), y_train, feature_names, args.catboost_iterations, args.seed + 200)

    predictions = make_predictions(xgb_model, lgb_model, cb_model, x_valid, feature_names)
    by_airline = by_airline_metrics(valid, predictions)
    best_all = choose_best(by_airline, include_baseline=True)
    best_ml = choose_best(by_airline, include_baseline=False)
    contribs = contribution_arrays(xgb_model, lgb_model, cb_model, x_valid, feature_names)

    model_files = export_models(args.output_dir, xgb_model, lgb_model, cb_model)
    hyper = model_hyperparameters(args.rounds, args.catboost_iterations, args.seed)
    feature_formulas = feature_formula_rows(builder)
    category_encodings = category_encoding_rows(builder)
    fill_values = pd.DataFrame(
        [{"feature": feature, "train_median_fill_value": builder.fill_values[feature]} for feature in feature_names]
    )

    formula_all = per_airline_formula_summary(valid, x_valid, best_all, predictions, contribs, "含A_DOBT基准",)
    formula_ml = per_airline_formula_summary(valid, x_valid, best_ml, predictions, contribs, "仅机器学习",)
    formula_summary = pd.concat([formula_all, formula_ml], ignore_index=True)

    contrib_all = per_airline_feature_contributions(valid, x_valid, best_all, contribs, "含A_DOBT基准", feature_formulas)
    contrib_ml = per_airline_feature_contributions(valid, x_valid, best_ml, contribs, "仅机器学习", feature_formulas)
    feature_contrib = pd.concat([contrib_all, contrib_ml], ignore_index=True)

    surrogate_summary_all, surrogate_coef_all = local_linear_surrogates(
        valid, x_valid, best_all, predictions, feature_contrib, "含A_DOBT基准", args.surrogate_k
    )
    surrogate_summary_ml, surrogate_coef_ml = local_linear_surrogates(
        valid, x_valid, best_ml, predictions, feature_contrib, "仅机器学习", args.surrogate_k
    )
    surrogate_summary = pd.concat([surrogate_summary_all, surrogate_summary_ml], ignore_index=True)
    surrogate_coef = pd.concat([surrogate_coef_all, surrogate_coef_ml], ignore_index=True)

    detail_all = selected_flight_calculation_detail(valid, x_valid, best_all, predictions, contribs, "含A_DOBT基准", args.top_k)
    detail_ml = selected_flight_calculation_detail(valid, x_valid, best_ml, predictions, contribs, "仅机器学习", args.top_k)

    explanation = pd.DataFrame(
        [
            {
                "item": "核心预测公式",
                "content": "predicted_ATOBT = A-DOBT + predicted_delta; predicted_delta is the estimated value of ATOBT - A-DOBT in minutes.",
            },
            {
                "item": "树模型系数说明",
                "content": "XGBoost/LightGBM/CatBoost do not have one fixed linear coefficient per feature. Their exact parameters are split thresholds and leaf values in model_files/.",
            },
            {
                "item": "SHAP贡献说明",
                "content": "For each flight, predicted_delta = base_value + sum(feature_contribution). Contributions are in minutes and exactly reconstruct the tree prediction up to numerical precision.",
            },
            {
                "item": "局部线性近似说明",
                "content": "The local linear surrogate coefficients are fitted per airline to approximate the selected tree model on validation records; they are readable explanatory coefficients, not the original prediction model.",
            },
            {
                "item": "训练与验证",
                "content": "Train on 2026-03 and 2026-04; validate on 2026-05. ATOBT, actual handover apron, TSAT, and post-pushback/taxi/takeoff nodes are not used as input features.",
            },
        ]
    )
    counts_df = pd.DataFrame([counts])

    csv_map = {
        "每航司计算公式汇总.csv": formula_summary,
        "每航司参数贡献_SHAP.csv": feature_contrib,
        "每航司局部线性近似系数.csv": surrogate_coef,
        "每航司局部线性近似公式.csv": surrogate_summary,
        "输入参数计算公式.csv": feature_formulas,
        "类别编码参数.csv": category_encodings,
        "缺失填充值参数.csv": fill_values,
        "模型超参数.csv": hyper,
        "模型文件索引.csv": model_files,
        "逐航班最佳模型计算明细_含A_DOBT基准.csv": detail_all,
        "逐航班最佳模型计算明细_仅机器学习.csv": detail_ml,
    }
    for name, frame in csv_map.items():
        frame.to_csv(args.output_dir / name, index=False, encoding="utf-8-sig")

    xlsx_path = args.output_dir / "每航司具体计算过程_参数_系数.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        explanation.to_excel(writer, sheet_name="说明", index=False)
        counts_df.to_excel(writer, sheet_name="样本统计", index=False)
        formula_summary.to_excel(writer, sheet_name="每航司计算公式", index=False)
        feature_contrib[feature_contrib["rank"].le(args.top_k)].to_excel(writer, sheet_name="每航司SHAP贡献Top参数", index=False)
        surrogate_summary.to_excel(writer, sheet_name="每航司近似公式", index=False)
        surrogate_coef.to_excel(writer, sheet_name="每航司近似系数", index=False)
        feature_formulas.to_excel(writer, sheet_name="输入参数计算公式", index=False)
        category_encodings.to_excel(writer, sheet_name="类别编码参数", index=False)
        fill_values.to_excel(writer, sheet_name="缺失填充值", index=False)
        hyper.to_excel(writer, sheet_name="模型超参数", index=False)
        by_airline.to_excel(writer, sheet_name="分航司模型指标", index=False)
        model_files.to_excel(writer, sheet_name="模型文件索引", index=False)
        detail_ml.head(5000).to_excel(writer, sheet_name="逐航班明细_仅ML前5000", index=False)

    run_info = {
        "paperTitle": PAPER_TITLE,
        "inputFiles": [str(path) for path in args.input],
        "trainPattern": args.train_pattern,
        "testPattern": args.test_pattern,
        "featureCount": len(feature_names),
        "counts": counts,
        "outputs": {name: str(args.output_dir / name) for name in csv_map},
        "xlsx": str(xlsx_path),
        "modelFiles": model_files.to_dict(orient="records"),
    }
    (args.output_dir / "calculation_details_run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"XLSX {xlsx_path}")
    print(f"MODEL_FILES {args.output_dir / 'model_files'}")


def main() -> None:
    train_and_export(parse_args())


if __name__ == "__main__":
    main()

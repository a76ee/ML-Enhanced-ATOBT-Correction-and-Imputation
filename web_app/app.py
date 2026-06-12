from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
STATIC_DIR = APP_DIR / "static"
OUTPUT_DIR = APP_DIR / "outputs"
for path in (SCRIPTS_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithm_atobt_ml_no_tsat_rf_gb import C, SUPPORT_NODES, TIME_PARAMS, format_dt, parse_dt, read_inputs  # noqa: E402
from algorithm_atobt_xgboost_lightgbm import predict_xgboost, train_lightgbm, train_xgboost  # noqa: E402
from reproduce_no_atobt_support_nodes import metric, predict_catboost, train_catboost  # noqa: E402


DEFAULT_TRAINING_DIR = PROJECT_DIR / "data" / "training"
TRAINING_INPUTS: list[Path] = []
HANDOVER_COL = C["handover"]
HANDOVER_DT = HANDOVER_COL + "_dt"


def discover_training_inputs(explicit_paths: list[Path] | None = None) -> list[Path]:
    if explicit_paths:
        return [Path(path) for path in explicit_paths]

    env_value = os.environ.get("ATOBT_TRAINING_INPUTS", "").strip()
    if env_value:
        return [Path(part.strip().strip('"')) for part in env_value.split(";") if part.strip()]

    paths: list[Path] = []
    for pattern in ("*.csv", "*.CSV", "*.xlsx", "*.xlsm"):
        paths.extend(DEFAULT_TRAINING_DIR.glob(pattern))
    return sorted(paths)


SUMMARY_COLUMN_LABELS = {
    "filename": "文件名",
    "scope": "算法口径",
    "target": "预测目标",
    "rows": "总行数",
    "predictedRows": "可预测行数",
    "historicalRows": "可用实际移交机坪对比行数",
    "MAE_min": "修正后平均绝对误差_分钟",
    "RMSE_min": "修正后均方根误差_分钟",
    "Within_le_3min_pct": "修正后绝对误差小于等于3分钟比例_%",
    "Within_le_5min_pct": "修正后绝对误差小于等于5分钟比例_%",
    "original_ATOBT_MAE_min": "原始A_TOBT平均绝对误差_分钟",
    "adobt_baseline_MAE_min": "A_DOBT基准平均绝对误差_分钟",
    "generatedAt": "生成时间",
}

AIRLINE_COLUMN_LABELS = {
    "IFC": "航司",
    "n": "对比样本量",
    "MAE_min": "修正后平均绝对误差_分钟",
    "MedianAE_min": "修正后中位绝对误差_分钟",
    "RMSE_min": "修正后均方根误差_分钟",
    "Within_le_3min_pct": "修正后绝对误差小于等于3分钟比例_%",
    "Within_le_5min_pct": "修正后绝对误差小于等于5分钟比例_%",
    "original_ATOBT_MAE_min": "原始A_TOBT平均绝对误差_分钟",
    "adobt_baseline_MAE_min": "A_DOBT基准平均绝对误差_分钟",
}

DETAIL_COLUMN_LABELS = {
    "_source_file": "来源文件",
    "_source_row": "来源行",
    "IFC": "航司",
    "CLA": "航班号",
    "TAR": "机位",
    "ITY": "机型",
    "RWYA": "到达跑道",
    "RWYD": "离港跑道",
    "A-TOBT": "原始A-TOBT",
    "A-DOBT": "A-DOBT",
    "CTOT": "CTOT",
    HANDOVER_COL: "实际移交机坪管制",
    "target_time": "历史实际移交机坪管制",
    "anchor_type": "锚点类型",
    "anchor_time": "锚点时间",
    "selected_model": "最优算法",
    "predicted_delta_min": "模型修正量_分钟",
    "predicted_handover": "预测实际移交机坪时间",
    "error_min": "预测误差_分钟",
    "abs_error_min": "绝对误差_分钟",
    "within_3min": "是否小于等于3分钟",
    "within_5min": "是否小于等于5分钟",
    "original_ATOBT_error_min": "原始A_TOBT误差_分钟",
    "adobt_baseline_error_min": "A_DOBT基准误差_分钟",
    "status": "状态",
}

STATUS_LABELS = {
    "ok": "已计算",
    "missing_anchor": "A-TOBT和A-DOBT均缺失或无法解析",
}


@dataclass
class AnchorModelStack:
    key: str
    label: str
    anchor_column: str
    anchor_dt_column: str
    feature_prefix: str
    builder: "AnchorFeatureBuilder"
    xgb_model: object
    lgb_model: object
    cb_model: object
    best_by_airline: dict[str, str]
    global_best: str
    validation_metrics: list[dict[str, object]]
    by_airline_metrics: pd.DataFrame


@dataclass
class ModelBundle:
    atobt_stack: AnchorModelStack
    adobt_stack: AnchorModelStack
    training_summary: dict[str, object]


MODEL: ModelBundle | None = None


def json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return format_dt(value)
    return value


def cyclical_hour(dt: pd.Series) -> tuple[pd.Series, pd.Series]:
    hour = dt.dt.hour + dt.dt.minute / 60.0
    angle = 2.0 * np.pi * hour / 24.0
    return np.sin(angle), np.cos(angle)


def cyclical_dow(dt: pd.Series) -> tuple[pd.Series, pd.Series]:
    dow = dt.dt.dayofweek.astype(float)
    angle = 2.0 * np.pi * dow / 7.0
    return np.sin(angle), np.cos(angle)


class AnchorFeatureBuilder:
    def __init__(self, anchor_dt_column: str, feature_prefix: str) -> None:
        self.anchor_dt_column = anchor_dt_column
        self.feature_prefix = feature_prefix
        self.feature_names: list[str] = []
        self.fill_values: dict[str, float] = {}
        self.category_maps: dict[str, dict[str, float]] = {}
        self.category_global: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "AnchorFeatureBuilder":
        raw = self._raw_features(df, fit_mode=True, y=y)
        self.feature_names = list(raw.columns)
        for name in self.feature_names:
            value = raw[name].replace([np.inf, -np.inf], np.nan).median()
            self.fill_values[name] = 0.0 if pd.isna(value) else float(value)
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
        anchor = df[self.anchor_dt_column] if self.anchor_dt_column in df.columns else pd.Series(pd.NaT, index=df.index)

        hour_sin, hour_cos = cyclical_hour(anchor)
        dow_sin, dow_cos = cyclical_dow(anchor)
        out[f"{self.feature_prefix}_hour_sin"] = hour_sin
        out[f"{self.feature_prefix}_hour_cos"] = hour_cos
        out[f"{self.feature_prefix}_dow_sin"] = dow_sin
        out[f"{self.feature_prefix}_dow_cos"] = dow_cos

        for label, col in SUPPORT_NODES:
            dt_col = col + "_dt"
            if dt_col not in df.columns:
                out[f"{label}_available"] = 0.0
                out[f"{label}_minus_{self.feature_prefix}_min"] = np.nan
                continue
            node_dt = df[dt_col]
            available = node_dt.notna() & anchor.notna()
            delta = (node_dt - anchor).dt.total_seconds() / 60.0
            out[f"{label}_available"] = available.astype(float)
            out[f"{label}_minus_{self.feature_prefix}_min"] = delta.where(available & delta.between(-1440, 1440))

        for other_col, other_dt, other_label in [
            ("A-DOBT", "A-DOBT_dt", "adobt"),
            ("A-TOBT", "A-TOBT_dt", "atobt"),
            ("CTOT", "CTOT_dt", "ctot"),
        ]:
            if other_dt == self.anchor_dt_column:
                continue
            dt = df[other_dt] if other_dt in df.columns else pd.Series(pd.NaT, index=df.index)
            available = dt.notna() & anchor.notna()
            delta = (dt - anchor).dt.total_seconds() / 60.0
            out[f"{other_label}_available"] = available.astype(float)
            out[f"{other_label}_minus_{self.feature_prefix}_min"] = delta.where(available & delta.between(-1440, 1440))
            if other_col == "CTOT":
                h_sin, h_cos = cyclical_hour(dt)
                out["ctot_hour_sin"] = h_sin.where(available)
                out["ctot_hour_cos"] = h_cos.where(available)

        # Category target-median encodings are fitted on the training period only.
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
                self.category_maps[col] = {
                    str(row["cat"]): float(row["median"])
                    for _, row in grouped.iterrows()
                    if int(row["count"]) >= min_count
                }
                self.category_global[col] = global_median
            mapping = self.category_maps.get(col, {})
            global_median = self.category_global.get(col, 0.0)
            out[key] = df[col].astype(str).map(mapping).fillna(global_median)
        return out


def bind_port(preferred: int) -> int:
    for port in range(preferred, preferred + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free local port near {preferred}")


def compute_metrics(errors: pd.Series | np.ndarray) -> dict[str, object]:
    err = pd.Series(errors, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(err) == 0:
        return {
            "n": 0,
            "MAE_min": None,
            "MedianAE_min": None,
            "RMSE_min": None,
            "Within_le_3min_pct": None,
            "Within_le_5min_pct": None,
        }
    abs_err = err.abs()
    return {
        "n": int(len(err)),
        "MAE_min": float(abs_err.mean()),
        "MedianAE_min": float(abs_err.median()),
        "RMSE_min": float(np.sqrt(np.mean(np.square(err.to_numpy(float))))),
        "Within_le_3min_pct": float((abs_err <= 3).mean() * 100),
        "Within_le_5min_pct": float((abs_err <= 5).mean() * 100),
    }


def predict_all(xgb_model, lgb_model, cb_model, x_data: pd.DataFrame, feature_names: list[str]) -> dict[str, np.ndarray]:
    x_np = x_data.to_numpy(float)
    pred_xgb = predict_xgboost(xgb_model, x_np, feature_names)
    pred_lgb = lgb_model.predict(x_np)
    pred_cb = predict_catboost(cb_model, x_np, feature_names)
    return {
        "XGBoost": np.asarray(pred_xgb, dtype=float),
        "LightGBM": np.asarray(pred_lgb, dtype=float),
        "CatBoost": np.asarray(pred_cb, dtype=float),
        "Blend_XGB_LGBM_CatBoost_Equal": (np.asarray(pred_xgb) + np.asarray(pred_lgb) + np.asarray(pred_cb)) / 3.0,
    }


def choose_best(by_airline: pd.DataFrame) -> pd.DataFrame:
    return (
        by_airline.sort_values(
            ["IFC", "MAE_min", "RMSE_min", "Within_le_3min_pct", "Within_le_5min_pct"],
            ascending=[True, True, True, False, False],
        )
        .groupby("IFC", as_index=False, sort=True)
        .first()
    )


def train_anchor_stack(
    df: pd.DataFrame,
    key: str,
    label: str,
    anchor_column: str,
    anchor_dt_column: str,
    feature_prefix: str,
    train_pattern: str,
    test_pattern: str,
    rounds: int,
    iterations: int,
    seed: int,
) -> AnchorModelStack:
    anchor_ok = df[anchor_dt_column].notna()
    target_ok = df[HANDOVER_DT].notna()
    labeled = df[anchor_ok & target_ok].copy()
    labeled["target_delta_min"] = (labeled[HANDOVER_DT] - labeled[anchor_dt_column]).dt.total_seconds() / 60.0
    plausible = labeled[labeled["target_delta_min"].abs() <= 180.0].copy()
    train = plausible[plausible["_source_file"].str.contains(train_pattern, regex=True)].copy()
    valid = plausible[plausible["_source_file"].str.contains(test_pattern, regex=True)].copy()
    if len(train) == 0 or len(valid) == 0:
        raise ValueError(f"{label} training split is empty: train={len(train)}, valid={len(valid)}")

    builder = AnchorFeatureBuilder(anchor_dt_column=anchor_dt_column, feature_prefix=feature_prefix)
    x_train = builder.fit_transform(train, train["target_delta_min"])
    x_valid = builder.transform(valid)
    y_train = train["target_delta_min"].to_numpy(float)
    y_valid = valid["target_delta_min"].to_numpy(float)

    xgb_model = train_xgboost(x_train.to_numpy(float), y_train, builder.feature_names, rounds, seed)
    lgb_model = train_lightgbm(x_train.to_numpy(float), y_train, builder.feature_names, rounds, seed + 100)
    cb_model = train_catboost(x_train.to_numpy(float), y_train, builder.feature_names, iterations, seed + 200)
    predictions = predict_all(xgb_model, lgb_model, cb_model, x_valid, builder.feature_names)

    overall_rows = [{"model": model, **compute_metrics(pred - y_valid)} for model, pred in predictions.items()]
    overall = pd.DataFrame(overall_rows).sort_values("MAE_min")
    global_best = str(overall.iloc[0]["model"])

    tmp = valid[["IFC"]].reset_index(drop=True).copy()
    tmp["_target"] = y_valid
    for model, pred in predictions.items():
        tmp[model] = pred
    airline_rows = []
    for ifc, group in tmp.groupby("IFC", dropna=False):
        if len(group) < 10:
            continue
        target = group["_target"].to_numpy(float)
        for model in predictions:
            airline_rows.append({"IFC": ifc, "model": model, **compute_metrics(group[model].to_numpy(float) - target)})
    by_airline = pd.DataFrame(airline_rows)
    best = choose_best(by_airline) if len(by_airline) else pd.DataFrame(columns=["IFC", "model"])

    return AnchorModelStack(
        key=key,
        label=label,
        anchor_column=anchor_column,
        anchor_dt_column=anchor_dt_column,
        feature_prefix=feature_prefix,
        builder=builder,
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        cb_model=cb_model,
        best_by_airline=dict(zip(best["IFC"].astype(str), best["model"].astype(str))),
        global_best=global_best,
        validation_metrics=overall.to_dict(orient="records"),
        by_airline_metrics=by_airline,
    )


def train_model_bundle() -> ModelBundle:
    print("Training ATOBT correction models. Target is actual apron handover time.", flush=True)
    inputs = TRAINING_INPUTS or discover_training_inputs()
    if not inputs:
        raise RuntimeError(
            "No training files found. Put local CSV/XLSX files under data/training "
            "or start the app with --train-input file1.csv file2.csv."
        )
    df = read_inputs(inputs)
    for col in ["A-TOBT", "A-DOBT"]:
        dt_col = col + "_dt"
        if dt_col not in df.columns and col in df.columns:
            df[dt_col] = parse_dt(df[col])
    if HANDOVER_DT not in df.columns:
        raise ValueError(f"Missing actual handover column: {HANDOVER_COL}")

    atobt_stack = train_anchor_stack(
        df=df,
        key="atobt",
        label="A-TOBT修正",
        anchor_column="A-TOBT",
        anchor_dt_column="A-TOBT_dt",
        feature_prefix="atobt",
        train_pattern="2026-03|2026-04",
        test_pattern="2026-05",
        rounds=260,
        iterations=500,
        seed=42,
    )
    adobt_stack = train_anchor_stack(
        df=df,
        key="adobt",
        label="A-DOBT兜底",
        anchor_column="A-DOBT",
        anchor_dt_column="A-DOBT_dt",
        feature_prefix="adobt",
        train_pattern="2026-03|2026-04",
        test_pattern="2026-05",
        rounds=260,
        iterations=500,
        seed=1042,
    )
    summary = {
        "target": "实际移交机坪管制时间",
        "primaryAnchor": "A-TOBT",
        "fallbackAnchor": "A-DOBT",
        "atobtGlobalBest": atobt_stack.global_best,
        "adobtGlobalBest": adobt_stack.global_best,
        "atobtValidation": atobt_stack.validation_metrics,
        "adobtValidation": adobt_stack.validation_metrics,
        "airlineCount": int(len(set(atobt_stack.best_by_airline) | set(adobt_stack.best_by_airline))),
        "featureCountAtobt": len(atobt_stack.builder.feature_names),
        "featureCountAdobt": len(adobt_stack.builder.feature_names),
    }
    print(
        f"Ready. Target=actual handover; A-TOBT best={atobt_stack.global_best}; A-DOBT fallback best={adobt_stack.global_best}.",
        flush=True,
    )
    return ModelBundle(atobt_stack=atobt_stack, adobt_stack=adobt_stack, training_summary=summary)


def read_uploaded_table(filename: str, content: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        df = pd.read_excel(io.BytesIO(content), dtype=str, keep_default_na=False)
    elif suffix == ".csv":
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "gb18030", "utf-8"):
            try:
                df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, encoding=encoding, low_memory=False)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise ValueError(f"CSV decoding failed: {last_error}")
    else:
        raise ValueError("只支持 .xlsx、.xlsm 或 .csv 文件")

    df = df.fillna("")
    df.columns = [str(col).strip().replace("\ufeff", "") for col in df.columns]
    df["_source_file"] = filename
    df["_source_row"] = np.arange(2, len(df) + 2)
    time_cols = sorted({"A-TOBT", "A-DOBT", HANDOVER_COL, *[col for _, col in SUPPORT_NODES], *[col for _, col in TIME_PARAMS]})
    for col in time_cols:
        if col in df.columns:
            df[col + "_dt"] = parse_dt(df[col])
    if "A-TOBT_dt" not in df.columns:
        df["A-TOBT_dt"] = pd.NaT
    if "A-DOBT_dt" not in df.columns:
        df["A-DOBT_dt"] = pd.NaT
    if HANDOVER_DT not in df.columns:
        df[HANDOVER_DT] = pd.NaT
    return df


def stack_predictions(stack: AnchorModelStack, work: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    x_data = stack.builder.transform(work)
    predictions = predict_all(stack.xgb_model, stack.lgb_model, stack.cb_model, x_data, stack.builder.feature_names)
    ifc_values = work["IFC"].astype(str) if "IFC" in work.columns else pd.Series([""] * len(work), index=work.index)
    selected_model = ifc_values.map(stack.best_by_airline).fillna(stack.global_best)
    selected_delta = np.zeros(len(work), dtype=float)
    for model, pred in predictions.items():
        mask = selected_model.eq(model).to_numpy()
        selected_delta[mask] = pred[mask]
    return selected_model, selected_delta


def localize_detail_for_export(result: pd.DataFrame) -> pd.DataFrame:
    localized = result.copy()
    if "status" in localized.columns:
        localized["status"] = localized["status"].map(STATUS_LABELS).fillna(localized["status"])
    for col in ["within_3min", "within_5min"]:
        if col in localized.columns:
            localized[col] = localized[col].map({True: "是", False: "否", "": ""}).fillna(localized[col])
    localized = localized.rename(columns=DETAIL_COLUMN_LABELS)
    priority = [
        "来源文件",
        "来源行",
        "航司",
        "航班号",
        "机位",
        "机型",
        "原始A-TOBT",
        "A-DOBT",
        "实际移交机坪管制",
        "历史实际移交机坪管制",
        "锚点类型",
        "锚点时间",
        "最优算法",
        "预测实际移交机坪时间",
        "模型修正量_分钟",
        "预测误差_分钟",
        "绝对误差_分钟",
        "是否小于等于3分钟",
        "是否小于等于5分钟",
        "原始A_TOBT误差_分钟",
        "A_DOBT基准误差_分钟",
        "状态",
    ]
    ordered = [col for col in priority if col in localized.columns]
    ordered.extend([col for col in localized.columns if col not in ordered])
    return localized[ordered]


def localize_airline_for_export(by_airline: pd.DataFrame) -> pd.DataFrame:
    return by_airline.rename(columns=AIRLINE_COLUMN_LABELS)


def localize_summary_for_export(summary: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([summary]).rename(columns=SUMMARY_COLUMN_LABELS)


def build_result_workbook(result: pd.DataFrame, summary: dict[str, object], by_airline: pd.DataFrame) -> tuple[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    xlsx_name = f"atobt_correction_to_handover_{stamp}.xlsx"
    csv_name = f"atobt_correction_to_handover_{stamp}.csv"
    xlsx_path = OUTPUT_DIR / xlsx_name
    csv_path = OUTPUT_DIR / csv_name

    result_export = localize_detail_for_export(result)
    summary_export = localize_summary_for_export(summary)
    by_airline_export = localize_airline_for_export(by_airline)
    result_export.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_export.to_excel(writer, sheet_name="汇总", index=False)
        by_airline_export.to_excel(writer, sheet_name="分航司对比", index=False)
        result_export.to_excel(writer, sheet_name="预测明细", index=False)
    return xlsx_name, csv_name


def predict_uploaded_file(filename: str, content: bytes, scope: str) -> dict[str, object]:
    if MODEL is None:
        raise RuntimeError("模型还没有准备好")
    df = read_uploaded_table(filename, content)
    original_cols = [col for col in df.columns if not col.endswith("_dt")]
    work = df.copy()

    atobt_mask = work["A-TOBT_dt"].notna()
    adobt_mask = ~atobt_mask & work["A-DOBT_dt"].notna()
    can_predict = atobt_mask | adobt_mask
    if not bool(can_predict.any()):
        raise ValueError("A-TOBT和A-DOBT均没有可解析时间，无法修正到实际移交机坪时间")

    atobt_model, atobt_delta = stack_predictions(MODEL.atobt_stack, work)
    adobt_model, adobt_delta = stack_predictions(MODEL.adobt_stack, work)

    selected_model = pd.Series("", index=work.index, dtype=object)
    selected_model.loc[atobt_mask] = atobt_model.loc[atobt_mask]
    selected_model.loc[adobt_mask] = adobt_model.loc[adobt_mask]
    selected_delta = pd.Series(np.nan, index=work.index, dtype=float)
    selected_delta.loc[atobt_mask] = atobt_delta[atobt_mask.to_numpy()]
    selected_delta.loc[adobt_mask] = adobt_delta[adobt_mask.to_numpy()]

    anchor_time = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
    anchor_time.loc[atobt_mask] = work.loc[atobt_mask, "A-TOBT_dt"]
    anchor_time.loc[adobt_mask] = work.loc[adobt_mask, "A-DOBT_dt"]
    anchor_type = pd.Series("", index=work.index, dtype=object)
    anchor_type.loc[atobt_mask] = "A-TOBT修正"
    anchor_type.loc[adobt_mask] = "A-DOBT兜底"

    predicted_handover = anchor_time + pd.to_timedelta(selected_delta, unit="m")
    target_available = work[HANDOVER_DT].notna()
    error = (predicted_handover - work[HANDOVER_DT]).dt.total_seconds() / 60.0
    original_atobt_error = (work["A-TOBT_dt"] - work[HANDOVER_DT]).dt.total_seconds() / 60.0
    adobt_baseline_error = (work["A-DOBT_dt"] - work[HANDOVER_DT]).dt.total_seconds() / 60.0

    result = work[original_cols].copy()
    result["target_time"] = [format_dt(value) for value in work[HANDOVER_DT]]
    result["anchor_type"] = anchor_type
    result["anchor_time"] = [format_dt(value) for value in anchor_time]
    result["selected_model"] = selected_model
    result["predicted_delta_min"] = np.round(selected_delta, 4)
    result["predicted_handover"] = [format_dt(value) if ok else "" for value, ok in zip(predicted_handover, can_predict)]
    result["error_min"] = np.round(error, 4)
    result["abs_error_min"] = np.round(error.abs(), 4)
    result["within_3min"] = np.where(error.notna(), error.abs() <= 3, "")
    result["within_5min"] = np.where(error.notna(), error.abs() <= 5, "")
    result["original_ATOBT_error_min"] = np.round(original_atobt_error, 4)
    result["adobt_baseline_error_min"] = np.round(adobt_baseline_error, 4)
    result["status"] = np.where(can_predict, "ok", "missing_anchor")

    predicted = result[result["status"].eq("ok")].copy()
    comparable = predicted[predicted["error_min"].notna()].copy()
    overall_metrics = compute_metrics(comparable["error_min"])
    original_metrics = compute_metrics(comparable["original_ATOBT_error_min"])
    adobt_metrics = compute_metrics(comparable["adobt_baseline_error_min"])

    by_airline_rows = []
    if "IFC" in comparable.columns and len(comparable) > 0:
        for ifc, group in comparable.groupby("IFC", dropna=False):
            row = {"IFC": ifc, **compute_metrics(group["error_min"])}
            row["original_ATOBT_MAE_min"] = compute_metrics(group["original_ATOBT_error_min"])["MAE_min"]
            row["adobt_baseline_MAE_min"] = compute_metrics(group["adobt_baseline_error_min"])["MAE_min"]
            by_airline_rows.append(row)
    by_airline = pd.DataFrame(by_airline_rows).sort_values("MAE_min") if by_airline_rows else pd.DataFrame()

    summary = {
        "filename": filename,
        "scope": "以实际移交机坪为真值：A-TOBT优先修正，A-DOBT缺失兜底",
        "target": "实际移交机坪管制时间",
        "rows": int(len(result)),
        "predictedRows": int(len(predicted)),
        "historicalRows": int(len(comparable)),
        "MAE_min": overall_metrics["MAE_min"],
        "RMSE_min": overall_metrics["RMSE_min"],
        "Within_le_3min_pct": overall_metrics["Within_le_3min_pct"],
        "Within_le_5min_pct": overall_metrics["Within_le_5min_pct"],
        "original_ATOBT_MAE_min": original_metrics["MAE_min"],
        "adobt_baseline_MAE_min": adobt_metrics["MAE_min"],
        "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    xlsx_name, csv_name = build_result_workbook(result, summary, by_airline)

    preview_cols = [
        "_source_row",
        "IFC",
        "A-TOBT",
        "A-DOBT",
        "target_time",
        "anchor_type",
        "anchor_time",
        "selected_model",
        "predicted_handover",
        "predicted_delta_min",
        "error_min",
        "abs_error_min",
        "within_3min",
        "within_5min",
        "original_ATOBT_error_min",
        "status",
    ]
    preview_cols = [col for col in preview_cols if col in result.columns]
    preview = result[preview_cols].head(200).replace({np.nan: None}).to_dict(orient="records")
    by_airline_preview = by_airline.head(50).replace({np.nan: None}).to_dict(orient="records") if len(by_airline) else []
    model_counts = result["selected_model"].replace("", np.nan).dropna().value_counts().rename_axis("model").reset_index(name="count").to_dict(orient="records")

    return {
        "summary": {key: json_ready(value) for key, value in summary.items()},
        "training": MODEL.training_summary,
        "modelCounts": model_counts,
        "byAirline": by_airline_preview,
        "preview": preview,
        "downloads": {"xlsx": f"/download/{xlsx_name}", "csv": f"/download/{csv_name}"},
    }


def parse_multipart(body: bytes, content_type: str) -> dict[str, dict[str, object]]:
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
    if not match:
        raise ValueError("Missing multipart boundary")
    boundary = match.group("boundary").strip('"').encode("utf-8")
    marker = b"--" + boundary
    fields: dict[str, dict[str, object]] = {}
    for raw_part in body.split(marker):
        part = raw_part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].rstrip(b"\r\n")
        header_blob, _, data = part.partition(b"\r\n\r\n")
        headers = header_blob.decode("utf-8", errors="replace").split("\r\n")
        disposition = next((line for line in headers if line.lower().startswith("content-disposition:")), "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        if not name_match:
            continue
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        name = name_match.group(1)
        fields[name] = {
            "filename": filename_match.group(1) if filename_match else None,
            "data": data,
        }
    return fields


class AppHandler(BaseHTTPRequestHandler):
    server_version = "ATOBTWeb/2.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_json(self, payload: dict[str, object], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=json_ready).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path, download_name: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self.send_file(STATIC_DIR / "index.html")
            return
        if path == "/api/status":
            self.send_json({"ready": MODEL is not None, "training": MODEL.training_summary if MODEL else None})
            return
        if path.startswith("/download/"):
            name = Path(path.removeprefix("/download/")).name
            self.send_file(OUTPUT_DIR / name, download_name=name)
            return
        if path.startswith("/static/"):
            relative = Path(path.removeprefix("/static/"))
            if ".." in relative.parts:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            self.send_file(STATIC_DIR / relative)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            content_type = self.headers.get("Content-Type", "")
            body = self.rfile.read(length)
            fields = parse_multipart(body, content_type)
            file_field = fields.get("file")
            if not file_field or not file_field.get("filename"):
                raise ValueError("请选择要上传的 Excel 或 CSV 文件")
            result = predict_uploaded_file(str(file_field["filename"]), bytes(file_field["data"]), "handover_target")
            self.send_json({"ok": True, **result})
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local ATOBT correction web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--train-input", nargs="*", type=Path, default=None)
    args = parser.parse_args()

    global MODEL, TRAINING_INPUTS
    TRAINING_INPUTS = discover_training_inputs(args.train_input)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL = train_model_bundle()
    port = bind_port(args.port)
    server = ThreadingHTTPServer((args.host, port), AppHandler)
    print(f"ATOBT correction web app ready: http://{args.host}:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

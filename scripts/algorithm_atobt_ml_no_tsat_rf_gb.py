"""RandomForest and GradientBoosting ATOBT imputation with time parameters.

This script intentionally avoids actual post-ATOBT operational outcomes as
features. It uses:
- actual apron handover as a gate and prediction anchor;
- pre-support/turnaround milestones;
- time parameters: A-DOBT and CTOT, plus handover hour/day cyclic terms.

It does not use startup request, startup permit, pushback, taxi, queue, takeoff,
ATOT, LTOT, TTOT or TSAT.

The execution environment used for this project does not include scikit-learn,
so the RandomForest and GradientBoosting regressors below are lightweight
NumPy implementations built on an approximate CART regression tree.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DIR = PROJECT_DIR / "data" / "training"
DEFAULT_INPUTS = [
    *sorted(DEFAULT_TRAINING_DIR.glob("*.csv")),
    *sorted(DEFAULT_TRAINING_DIR.glob("*.CSV")),
    *sorted(DEFAULT_TRAINING_DIR.glob("*.xlsx")),
    *sorted(DEFAULT_TRAINING_DIR.glob("*.xlsm")),
]

C = {
    "handover": "\u5b9e\u9645\u79fb\u4ea4\u673a\u576a\u7ba1\u5236",
    "arrival_handover": "\u8fdb\u8fd1\u7ba1\u5236\u79fb\u4ea4",
    "ready_landing": "\u51c6\u5907\u843d\u5730",
    "arrival_wait_cross": "\u8fdb\u6e2f\u7b49\u5f85\u7a7f\u8d8a",
    "arrival_taxi": "\u8fdb\u6e2f\u6ed1\u884c",
    "load_start": "\u88c5\u5378\u5f00\u59cb",
    "load_end": "\u88c5\u5378\u7ed3\u675f",
    "cargo_door_close": "\u5173\u8d27\u8231\u95e8",
    "tow_ready": "\u62d6\u8f66\u5230\u4f4d",
    "close_cabin": "\u5173\u5ba2\u8231\u95e8",
    "close_door": "\u5173\u8231\u95e8",
    "crew_ready": "\u673a\u7ec4\u5230\u4f4d",
    "maint": "\u673a\u52a1\u653e\u884c",
    "catering_start": "\u5f00\u59cb\u914d\u9910",
    "catering_end": "\u914d\u9910\u5b8c\u6210",
    "fuel_start": "\u4f9b\u6cb9\u5f00\u59cb",
    "fuel_end": "\u4f9b\u6cb9\u5b8c\u6210",
    "gate_open": "\u767b\u673a\u53e3\u5f00\u542f",
    "gate_close": "\u767b\u673a\u53e3\u5173\u95ed",
    "bridge_off": "\u79bb\u6865\u5b8c\u6210",
    "stair_leave": "\u79bb\u6e2f\u5ba2\u68af\u8f66\u64a4\u79bb",
}

SUPPORT_NODES = [
    ("arrival_handover", C["arrival_handover"]),
    ("ready_landing", C["ready_landing"]),
    ("arrival_wait_cross", C["arrival_wait_cross"]),
    ("arrival_taxi", C["arrival_taxi"]),
    ("load_start", C["load_start"]),
    ("load_end", C["load_end"]),
    ("cargo_door_close", C["cargo_door_close"]),
    ("tow_ready", C["tow_ready"]),
    ("close_cabin", C["close_cabin"]),
    ("close_door", C["close_door"]),
    ("crew_ready", C["crew_ready"]),
    ("maint", C["maint"]),
    ("catering_start", C["catering_start"]),
    ("catering_end", C["catering_end"]),
    ("fuel_start", C["fuel_start"]),
    ("fuel_end", C["fuel_end"]),
    ("gate_open", C["gate_open"]),
    ("gate_close", C["gate_close"]),
    ("bridge_off", C["bridge_off"]),
    ("stair_leave", C["stair_leave"]),
]

TIME_PARAMS = [("adobt", "A-DOBT"), ("ctot", "CTOT")]

FORBIDDEN_ACTUAL_POST_NODES = {
    "\u5b9e\u9645\u8bf7\u6c42\u5f00\u8f66",
    "\u8bb8\u53ef\u5f00\u8f66",
    "\u5b9e\u9645\u63a8\u51fa",
    "\u5b9e\u9645\u5f00\u59cb\u6ed1\u884c",
    "\u5b9e\u9645\u5f00\u59cb\u6392\u961f",
    "\u7ed3\u675f\u6392\u961f",
    "\u51c6\u5907\u8d77\u98de",
    "ATOT",
    "LTOT",
    "TTOT",
    "TSAT",
}


def parse_dt(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.mask(text.eq(""), pd.NA)
    return pd.to_datetime(text, errors="coerce")


def format_dt(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).round("min").strftime("%Y-%m-%d %H:%M:%S")


def read_inputs(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError(
            "No input files were provided. Put local training files under data/training "
            "or pass them with --input."
        )

    frames = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xlsm"}:
            df = pd.read_excel(path, dtype=str, keep_default_na=False)
        elif suffix == ".csv":
            df = pd.read_csv(path, dtype=str, encoding="gb18030", keep_default_na=False, low_memory=False)
        else:
            raise ValueError(f"Unsupported input file type: {path}")
        df.columns = [str(col).strip().replace("\ufeff", "") for col in df.columns]
        df["_source_file"] = path.name
        df["_source_row"] = np.arange(2, len(df) + 2)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    time_cols = sorted({"A-TOBT", C["handover"], *[col for _, col in SUPPORT_NODES], *[col for _, col in TIME_PARAMS]})
    for col in time_cols:
        if col in df.columns:
            df[col + "_dt"] = parse_dt(df[col])
    return df


def cyclical_hour(dt: pd.Series) -> tuple[pd.Series, pd.Series]:
    hour = dt.dt.hour + dt.dt.minute / 60.0
    angle = 2.0 * np.pi * hour / 24.0
    return np.sin(angle), np.cos(angle)


def cyclical_dow(dt: pd.Series) -> tuple[pd.Series, pd.Series]:
    dow = dt.dt.dayofweek.astype(float)
    angle = 2.0 * np.pi * dow / 7.0
    return np.sin(angle), np.cos(angle)


class FeatureBuilder:
    def __init__(self) -> None:
        self.feature_names: list[str] = []
        self.fill_values: dict[str, float] = {}
        self.category_maps: dict[str, dict[str, float]] = {}
        self.category_global: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "FeatureBuilder":
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
        handover = df[C["handover"] + "_dt"]

        hour_sin, hour_cos = cyclical_hour(handover)
        dow_sin, dow_cos = cyclical_dow(handover)
        out["handover_hour_sin"] = hour_sin
        out["handover_hour_cos"] = hour_cos
        out["handover_dow_sin"] = dow_sin
        out["handover_dow_cos"] = dow_cos

        for label, col in SUPPORT_NODES:
            dt_col = col + "_dt"
            if dt_col not in df.columns:
                out[f"{label}_available"] = 0.0
                out[f"{label}_minus_handover_min"] = np.nan
                continue
            node_dt = df[dt_col]
            available = node_dt.notna() & handover.notna() & (node_dt <= handover + pd.Timedelta(minutes=5))
            delta = (node_dt - handover).dt.total_seconds() / 60.0
            delta = delta.where(available & delta.between(-1440, 10))
            out[f"{label}_available"] = available.astype(float)
            out[f"{label}_minus_handover_min"] = delta

        for label, col in TIME_PARAMS:
            dt_col = col + "_dt"
            if dt_col not in df.columns:
                out[f"{label}_available"] = 0.0
                out[f"{label}_minus_handover_min"] = np.nan
                continue
            param_dt = df[dt_col]
            available = param_dt.notna() & handover.notna()
            delta = (param_dt - handover).dt.total_seconds() / 60.0
            delta = delta.where(available & delta.between(-1440, 1440))
            out[f"{label}_available"] = available.astype(float)
            out[f"{label}_minus_handover_min"] = delta
            h_sin, h_cos = cyclical_hour(param_dt)
            out[f"{label}_hour_sin"] = h_sin.where(available)
            out[f"{label}_hour_cos"] = h_cos.where(available)

        # Target-median encodings capture stable airline/type effects while
        # avoiding large one-hot matrices for this small script.
        for col in ["IFC", "CLA", "ITY", "RWYA", "RWYD"]:
            key = f"{col}_target_median"
            if fit_mode:
                tmp = pd.DataFrame({"cat": df[col].astype(str), "y": y.astype(float)})
                grouped = tmp.groupby("cat")["y"].agg(["count", "median"]).reset_index()
                global_median = float(tmp["y"].median())
                mapping = {
                    str(row["cat"]): float(row["median"])
                    for _, row in grouped.iterrows()
                    if int(row["count"]) >= 30
                }
                self.category_maps[col] = mapping
                self.category_global[col] = global_median
            mapping = self.category_maps.get(col, {})
            global_median = self.category_global.get(col, 0.0)
            out[key] = df[col].astype(str).map(mapping).fillna(global_median)

        return out


@dataclass
class TreeNode:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


class ApproxRegressionTree:
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 50,
        max_features: int | None = None,
        n_thresholds: int = 24,
        random_state: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.rng = np.random.default_rng(random_state)
        self.root: TreeNode | None = None
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ApproxRegressionTree":
        self.feature_importances_ = np.zeros(x.shape[1], dtype=float)
        self.root = self._build(x, y, np.arange(len(y)), depth=0)
        return self

    def _build(self, x: np.ndarray, y: np.ndarray, idx: np.ndarray, depth: int) -> TreeNode:
        y_node = y[idx]
        value = float(np.mean(y_node))
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf or np.var(y_node) < 1e-8:
            return TreeNode(value=value)

        feature, threshold, gain = self._best_split(x, y, idx)
        if feature is None or threshold is None or gain <= 1e-10:
            return TreeNode(value=value)

        mask = x[idx, feature] <= threshold
        left_idx = idx[mask]
        right_idx = idx[~mask]
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return TreeNode(value=value)

        if self.feature_importances_ is not None:
            self.feature_importances_[feature] += gain * len(idx)

        return TreeNode(
            value=value,
            feature=int(feature),
            threshold=float(threshold),
            left=self._build(x, y, left_idx, depth + 1),
            right=self._build(x, y, right_idx, depth + 1),
        )

    def _best_split(self, x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> tuple[int | None, float | None, float]:
        n_features = x.shape[1]
        if self.max_features is None or self.max_features >= n_features:
            features = np.arange(n_features)
        else:
            features = self.rng.choice(n_features, size=self.max_features, replace=False)

        y_node = y[idx]
        n = len(idx)
        parent_sse = float(np.sum((y_node - np.mean(y_node)) ** 2))
        best_feature: int | None = None
        best_threshold: float | None = None
        best_gain = 0.0

        qs = np.linspace(0.05, 0.95, self.n_thresholds)
        for feature in features:
            values = x[idx, feature]
            if np.nanmin(values) == np.nanmax(values):
                continue
            thresholds = np.unique(np.quantile(values, qs))
            for threshold in thresholds:
                left = values <= threshold
                n_left = int(np.sum(left))
                n_right = n - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                y_left = y_node[left]
                y_right = y_node[~left]
                sse_left = float(np.sum((y_left - np.mean(y_left)) ** 2))
                sse_right = float(np.sum((y_right - np.mean(y_right)) ** 2))
                gain = parent_sse - sse_left - sse_right
                if gain > best_gain:
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_gain = float(gain)
        return best_feature, best_threshold, best_gain

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree has not been fitted.")
        out = np.empty(x.shape[0], dtype=float)
        self._predict_node(self.root, x, np.arange(x.shape[0]), out)
        return out

    def _predict_node(self, node: TreeNode, x: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
        if node.feature is None or node.left is None or node.right is None:
            out[idx] = node.value
            return
        mask = x[idx, node.feature] <= node.threshold
        if np.any(mask):
            self._predict_node(node.left, x, idx[mask], out)
        if np.any(~mask):
            self._predict_node(node.right, x, idx[~mask], out)


class RandomForestRegressorLite:
    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 6,
        min_samples_leaf: int = 60,
        max_features: str | int = "sqrt",
        n_thresholds: int = 24,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.trees: list[ApproxRegressionTree] = []
        self.feature_importances_: np.ndarray | None = None

    def _max_features_count(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if isinstance(self.max_features, int):
            return max(1, min(n_features, self.max_features))
        return n_features

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestRegressorLite":
        rng = np.random.default_rng(self.random_state)
        max_features = self._max_features_count(x.shape[1])
        importances = np.zeros(x.shape[1], dtype=float)
        self.trees = []
        for i in range(self.n_estimators):
            sample_idx = rng.integers(0, len(y), size=len(y))
            tree = ApproxRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                n_thresholds=self.n_thresholds,
                random_state=self.random_state + i + 1,
            )
            tree.fit(x[sample_idx], y[sample_idx])
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_
            self.trees.append(tree)
        total = float(importances.sum())
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = np.vstack([tree.predict(x) for tree in self.trees])
        return np.mean(preds, axis=0)


class GradientBoostingRegressorLite:
    def __init__(
        self,
        n_estimators: int = 120,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 50,
        subsample: float = 0.85,
        n_thresholds: int = 28,
        random_state: int = 123,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.n_thresholds = n_thresholds
        self.random_state = random_state
        self.init_: float = 0.0
        self.trees: list[ApproxRegressionTree] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressorLite":
        rng = np.random.default_rng(self.random_state)
        self.init_ = float(np.mean(y))
        pred = np.full(len(y), self.init_, dtype=float)
        importances = np.zeros(x.shape[1], dtype=float)
        self.trees = []
        sample_size = max(2 * self.min_samples_leaf, int(len(y) * self.subsample))
        for i in range(self.n_estimators):
            residual = y - pred
            idx = rng.choice(len(y), size=sample_size, replace=False)
            tree = ApproxRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=None,
                n_thresholds=self.n_thresholds,
                random_state=self.random_state + i + 1,
            )
            tree.fit(x[idx], residual[idx])
            update = tree.predict(x)
            pred += self.learning_rate * update
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_
            self.trees.append(tree)
        total = float(importances.sum())
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = np.full(x.shape[0], self.init_, dtype=float)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(x)
        return pred


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "MAE_min": float(np.mean(np.abs(err))),
        "MedianAE_min": float(np.median(np.abs(err))),
        "RMSE_min": float(np.sqrt(np.mean(err**2))),
        "Within_5min": float(np.mean(np.abs(err) <= 5)),
        "Within_10min": float(np.mean(np.abs(err) <= 10)),
        "Within_15min": float(np.mean(np.abs(err) <= 15)),
    }


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


def feature_importance_frame(names: list[str], importances: np.ndarray, model: str) -> pd.DataFrame:
    out = pd.DataFrame({"model": model, "feature": names, "importance": importances})
    return out.sort_values("importance", ascending=False)


def add_predictions_to_candidates(
    candidates: pd.DataFrame,
    rf_delta: np.ndarray,
    gb_delta: np.ndarray,
) -> pd.DataFrame:
    out = candidates.copy()
    handover = out[C["handover"] + "_dt"]
    rf_dt = handover + pd.to_timedelta(rf_delta, unit="m")
    gb_dt = handover + pd.to_timedelta(gb_delta, unit="m")
    out["RF_predicted_ATOBT"] = [format_dt(x) for x in rf_dt]
    out["RF_predicted_delta_from_handover_min"] = np.round(rf_delta, 2)
    out["GB_predicted_ATOBT"] = [format_dt(x) for x in gb_dt]
    out["GB_predicted_delta_from_handover_min"] = np.round(gb_delta, 2)
    out["RF_GB_abs_diff_min"] = np.round(np.abs(rf_delta - gb_delta), 2)
    out["ML_confidence"] = np.where(
        out["RF_GB_abs_diff_min"] <= 5,
        "high",
        np.where(out["RF_GB_abs_diff_min"] <= 10, "medium", "low"),
    )
    return out


def export_columns(frame: pd.DataFrame) -> list[str]:
    cols = ["_source_file", "_source_row", "CLA", "TAR", "IFC", "ITY", "RWYA", "RWYD", "A-TOBT", C["handover"]]
    cols.extend(col for _, col in TIME_PARAMS)
    cols.extend(col for _, col in SUPPORT_NODES)
    cols.extend(["RF_predicted_ATOBT", "RF_predicted_delta_from_handover_min", "GB_predicted_ATOBT", "GB_predicted_delta_from_handover_min", "RF_GB_abs_diff_min", "ML_confidence"])
    return [col for col in dict.fromkeys(cols) if col in frame.columns]


def rename_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "_source_file": "source_file",
        "_source_row": "source_row",
        C["handover"]: "actual_handover_apron_anchor",
        C["arrival_handover"]: "arrival_approach_handover",
        C["ready_landing"]: "ready_landing",
        C["arrival_wait_cross"]: "arrival_wait_crossing",
        C["arrival_taxi"]: "arrival_taxi",
        C["load_start"]: "loading_start",
        C["load_end"]: "loading_finished",
        C["cargo_door_close"]: "cargo_door_closed",
        C["tow_ready"]: "tow_truck_ready",
        C["close_cabin"]: "cabin_door_closed",
        C["close_door"]: "door_closed",
        C["crew_ready"]: "crew_ready",
        C["maint"]: "maintenance_release",
        C["catering_start"]: "catering_start",
        C["catering_end"]: "catering_finished",
        C["fuel_start"]: "fuel_start",
        C["fuel_end"]: "fuel_finished",
        C["gate_open"]: "gate_opened",
        C["gate_close"]: "gate_closed",
        C["bridge_off"]: "bridge_off_finished",
        C["stair_leave"]: "departure_stair_removed",
    }
    return df.rename(columns=rename)


def run(
    inputs: list[Path],
    output_dir: Path,
    train_pattern: str = "2026-04",
    test_pattern: str = "2026-05",
    period_label: str = "2026-04_05",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_inputs(inputs)

    has_atobt = df["A-TOBT_dt"].notna()
    has_handover = df[C["handover"] + "_dt"].notna()
    labeled = df[has_atobt & has_handover].copy()
    labeled["target_delta_min"] = (labeled["A-TOBT_dt"] - labeled[C["handover"] + "_dt"]).dt.total_seconds() / 60.0
    plausible = labeled[labeled["target_delta_min"].between(-180, 60)].copy()
    outliers = labeled[~labeled["target_delta_min"].between(-180, 60)].copy()

    train = plausible[plausible["_source_file"].str.contains(train_pattern, regex=True)].copy()
    valid = plausible[plausible["_source_file"].str.contains(test_pattern, regex=True)].copy()
    if len(train) == 0 or len(valid) == 0:
        raise ValueError(
            f"Train/test split produced empty data: train_pattern={train_pattern!r}, "
            f"test_pattern={test_pattern!r}, train={len(train)}, valid={len(valid)}"
        )
    y_train = train["target_delta_min"].to_numpy(float)
    y_valid = valid["target_delta_min"].to_numpy(float)

    builder = FeatureBuilder()
    x_train = builder.fit_transform(train, train["target_delta_min"]).to_numpy(float)
    x_valid = builder.transform(valid).to_numpy(float)

    rf = RandomForestRegressorLite(n_estimators=80, max_depth=6, min_samples_leaf=60, random_state=2026)
    rf.fit(x_train, y_train)
    rf_valid = rf.predict(x_valid)

    gb = GradientBoostingRegressorLite(n_estimators=120, learning_rate=0.05, max_depth=3, min_samples_leaf=50, random_state=2027)
    gb.fit(x_train, y_train)
    gb_valid = gb.predict(x_valid)

    metrics_rows = []
    for model_name, pred in [("RandomForest", rf_valid), ("GradientBoosting", gb_valid)]:
        metrics_rows.append({"model": model_name, "train": train_pattern, "test": test_pattern, **metrics_for(y_valid, pred)})
    metrics_df = pd.DataFrame(metrics_rows)
    airline_df = by_airline_metrics(valid, y_valid, {"RandomForest": rf_valid, "GradientBoosting": gb_valid})
    importance_df = pd.concat(
        [
            feature_importance_frame(builder.feature_names, rf.feature_importances_, "RandomForest"),
            feature_importance_frame(builder.feature_names, gb.feature_importances_, "GradientBoosting"),
        ],
        ignore_index=True,
    )

    # Final models use all plausible labeled records for imputation.
    final_builder = FeatureBuilder()
    x_all = final_builder.fit_transform(plausible, plausible["target_delta_min"]).to_numpy(float)
    y_all = plausible["target_delta_min"].to_numpy(float)
    final_rf = RandomForestRegressorLite(n_estimators=100, max_depth=6, min_samples_leaf=60, random_state=3026)
    final_rf.fit(x_all, y_all)
    final_gb = GradientBoostingRegressorLite(n_estimators=140, learning_rate=0.05, max_depth=3, min_samples_leaf=50, random_state=3027)
    final_gb.fit(x_all, y_all)

    candidates = df[has_handover & ~has_atobt].copy()
    abandoned = df[~has_handover & ~has_atobt].copy()
    x_candidates = final_builder.transform(candidates).to_numpy(float)
    candidate_rf = final_rf.predict(x_candidates)
    candidate_gb = final_gb.predict(x_candidates)
    imputed = add_predictions_to_candidates(candidates, candidate_rf, candidate_gb)

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
        ]
    )

    imputed_export = rename_export_columns(imputed[export_columns(imputed)])
    abandon_cols = [col for col in export_columns(abandoned) if col in abandoned.columns]
    abandoned_export = rename_export_columns(abandoned[abandon_cols].copy())
    abandoned_export["abandon_reason"] = "missing actual_handover_apron_anchor; ATOBT not imputed"

    xlsx_path = output_dir / f"atobt_ml_no_tsat_rf_gb_{period_label}.xlsx"
    imputed_csv = output_dir / f"atobt_ml_no_tsat_imputed_candidates_{period_label}.csv"
    airline_csv = output_dir / "atobt_ml_no_tsat_by_airline_metrics.csv"
    report_path = output_dir / "atobt_ml_no_tsat_report.txt"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        metrics_df.to_excel(writer, sheet_name="validation", index=False)
        airline_df.to_excel(writer, sheet_name="by_airline", index=False)
        importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
        imputed_export.to_excel(writer, sheet_name="imputed_candidates", index=False)
        abandoned_export.to_excel(writer, sheet_name="abandoned_no_handover", index=False)

    imputed_export.to_csv(imputed_csv, index=False, encoding="utf-8-sig")
    airline_df.to_csv(airline_csv, index=False, encoding="utf-8-sig")

    top_airline = airline_df.sort_values(["model", "n"], ascending=[True, False]).groupby("model").head(20)
    top_features = importance_df.groupby("model").head(20)
    lines = [
        "ATOBT ML imputation with time parameters excluding TSAT",
        "",
        "Feature policy: pre-support nodes + A-DOBT/CTOT + handover hour/day. TSAT and actual post nodes are excluded.",
        "Excluded nodes: TSAT, startup request, startup permit, pushback, taxi start, queue, takeoff, ATOT, LTOT, TTOT.",
        "",
        "Summary",
        summary_df.to_string(index=False),
        "",
        f"Validation: train on {train_pattern}, test on {test_pattern}",
        metrics_df.to_string(index=False),
        "",
        "Top 20 airlines by validation sample size for each model",
        top_airline.to_string(index=False),
        "",
        "Top 20 feature importances for each model",
        top_features.to_string(index=False),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8-sig")

    print(f"REPORT {report_path}")
    print(f"XLSX {xlsx_path}")
    print(f"IMPUTED_CSV {imputed_csv}")
    print(f"AIRLINE_CSV {airline_csv}")
    print(metrics_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RandomForest and GradientBoosting for ATOBT imputation without TSAT.")
    parser.add_argument("--input", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "atobt_ml_no_tsat_outputs")
    parser.add_argument("--train-pattern", default="2026-04")
    parser.add_argument("--test-pattern", default="2026-05")
    parser.add_argument("--period-label", default="2026-04_05")
    args = parser.parse_args()
    run(args.input, args.output_dir, args.train_pattern, args.test_pattern, args.period_label)


if __name__ == "__main__":
    main()

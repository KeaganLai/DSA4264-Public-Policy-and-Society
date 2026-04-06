from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    DEFAULT_GOOD_SCHOOL_THRESHOLD,
    MODEL_META_PATH,
    MODEL_PATH,
    VALID_THRESHOLDS,
    model_meta_path_for_threshold,
    model_path_for_threshold,
)

try:
    import joblib  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("joblib is required to save artifacts. Install requirements first.") from exc


def _build_numeric_features(threshold: int) -> list[str]:
    return [
        "floor_area_sqm",
        "remaining_lease_years",
        "mature_estate",
        "dist_to_nearest_hawker_km",
        "dist_to_nearest_busstop_km",
        "dist_to_nearest_mall_km",
        "dist_to_nearest_mrt_km",
        "dist_to_cbd_km",
        "countall_0_1km",
        "countall_1_2km",
        f"d_0_1km_good{threshold}",
        f"d_1_2km_good{threshold}",
        f"count_0_1km_good{threshold}",
        f"count_1_2km_good{threshold}",
        f"dist_nearest_goodschool_{threshold}",
        "year",
        "quarter",
    ]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_price = float(np.mean(y_true))
    median_price = float(np.median(y_true))
    ape = np.abs((y_true - y_pred) / y_true)
    abs_err = np.abs(y_true - y_pred)
    return {
        "mean_price": mean_price,
        "median_price": median_price,
        "mae_price": mae,
        "rmse_price": rmse,
        "mae_pct_of_mean_price": float((mae / mean_price) * 100.0),
        "rmse_pct_of_mean_price": float((rmse / mean_price) * 100.0),
        "mae_pct_of_median_price": float((mae / median_price) * 100.0),
        "rmse_pct_of_median_price": float((rmse / median_price) * 100.0),
        "mape_pct": float(np.mean(ape) * 100.0),
        "abs_error_q80": float(np.quantile(abs_err, 0.80)),
        "abs_error_q90": float(np.quantile(abs_err, 0.90)),
        "abs_error_q95": float(np.quantile(abs_err, 0.95)),
        "ape_q80_pct": float(np.quantile(ape, 0.80) * 100.0),
        "ape_q90_pct": float(np.quantile(ape, 0.90) * 100.0),
        "ape_q95_pct": float(np.quantile(ape, 0.95) * 100.0),
    }


def train_for_threshold(threshold: int) -> dict:
    df = pd.read_csv(DATA_PATH)
    numeric_features = _build_numeric_features(threshold)
    categorical_features = ["town", "flat_type", "storey_relative_category"]
    target_col = "log_real_price"

    required_cols = numeric_features + categorical_features + [target_col, "Index"]
    model_df = df[required_cols].dropna().copy()

    # Time split to avoid leakage from future observations.
    train_df = model_df.loc[model_df["year"] <= 2023].copy()
    val_df = model_df.loc[model_df["year"] >= 2024].copy()
    if val_df.empty:
        val_df = model_df.sample(frac=0.2, random_state=42)
        train_df = model_df.drop(index=val_df.index)

    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target_col]
    X_val = val_df[numeric_features + categorical_features]
    y_val = val_df[target_col]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    pipeline = Pipeline(steps=[("preprocessor", pre), ("model", LinearRegression())])
    pipeline.fit(X_train, y_train)

    yhat_val = pipeline.predict(X_val)
    residual_std = float(np.std(y_val - yhat_val))

    # Real-price scale metrics (same target scale as model after exponentiating).
    y_val_real = np.exp(y_val.to_numpy())
    yhat_val_real = np.exp(yhat_val)
    real_metrics = _compute_metrics(y_val_real, yhat_val_real)

    # Nominal resale scale metrics (what end users compare against).
    index_factor = (val_df["Index"].to_numpy(dtype=float) / 100.0)
    y_val_nominal = y_val_real * index_factor
    yhat_val_nominal = yhat_val_real * index_factor
    nominal_metrics = _compute_metrics(y_val_nominal, yhat_val_nominal)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_path_for_threshold(threshold)
    meta_path = model_meta_path_for_threshold(threshold)
    joblib.dump(pipeline, model_path)

    meta = {
        "model_version": f"hedonic-linear-good{threshold}-{datetime.utcnow().strftime('%Y%m%d')}",
        "created_utc": datetime.utcnow().isoformat(),
        "good_school_threshold": threshold,
        "target_scale": "log_real_price",
        "prediction_scale": "real_price",
        "feature_columns": numeric_features + categorical_features,
        "train_n": int(train_df.shape[0]),
        "val_n": int(val_df.shape[0]),
        "validation_residual_std": residual_std,
        "validation_real": real_metrics,
        "validation_nominal": nominal_metrics,
        "interval_calibration": {
            "method": "validation_ape_quantile",
            "default_quantile_for_p10_p90": 0.90,
            "ape_q90_pct": nominal_metrics["ape_q90_pct"],
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Backward compatibility aliases for threshold 80.
    if threshold == DEFAULT_GOOD_SCHOOL_THRESHOLD:
        joblib.dump(pipeline, MODEL_PATH)
        MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[threshold={threshold}] Saved model to {model_path}")
    print(f"[threshold={threshold}] Saved metadata to {meta_path}")
    print(json.dumps(meta, indent=2))
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train threshold-specific hedonic baseline model artifacts.")
    parser.add_argument(
        "--threshold",
        type=int,
        choices=VALID_THRESHOLDS,
        default=DEFAULT_GOOD_SCHOOL_THRESHOLD,
        help="Single threshold to train.",
    )
    parser.add_argument(
        "--all-thresholds",
        action="store_true",
        help="Train all thresholds (75/80/85/90).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = list(VALID_THRESHOLDS) if args.all_thresholds else [args.threshold]
    for t in thresholds:
        train_for_threshold(t)


if __name__ == "__main__":
    main()


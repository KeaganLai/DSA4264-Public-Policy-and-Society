from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_GOOD_SCHOOL_THRESHOLD,
    MODEL_META_PATH,
    MODEL_PATH,
    VALID_THRESHOLDS,
    model_meta_path_for_threshold,
    model_path_for_threshold,
)
from .feature_engine import FeatureEngine
from .schemas import ComparableRecord, PredictRequest

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None


@dataclass
class _ArtifactBundle:
    model: Any
    meta: dict[str, Any]


@dataclass
class _IndexForecastModel:
    coeffs: np.ndarray
    train_rmse: float
    quarterly_log_growth: float
    n_points: int
    last_index: float


@dataclass
class _IndexResolution:
    value: float
    source: str
    horizon_quarters: int | None


@dataclass
class PredictionResult:
    predicted_price_nominal: float
    predicted_price_real: float
    p10_nominal: float
    p50_nominal: float
    p90_nominal: float
    p10_real: float
    p50_real: float
    p90_real: float
    valuation_price_index_used: float
    threshold_requested: int
    threshold_used: int
    model_mode: str
    model_version: str
    sample_size_used: int
    note: str
    comparables: list[ComparableRecord]


class ModelService:
    def __init__(self, feature_engine: FeatureEngine) -> None:
        self.feature_engine = feature_engine
        self.bundles: dict[int, _ArtifactBundle] = {}
        self.index_history = self._build_index_history(feature_engine.df)
        self.index_lookup = self._build_index_lookup(feature_engine.df)
        self.global_index_median = float(feature_engine.df["Index"].median())
        self.latest_observed_period = self._latest_observed_period(self.index_history)
        self.index_forecast_model = self._fit_index_forecast(self.index_history)
        self._load_artifact_models()

    @staticmethod
    def _build_index_lookup(df: pd.DataFrame) -> dict[tuple[int, int], float]:
        grouped = (
            df.groupby(["year", "quarter"], as_index=False)["Index"]
            .median()
            .rename(columns={"Index": "index_median"})
        )
        return {
            (int(row["year"]), int(row["quarter"])): float(row["index_median"])
            for _, row in grouped.iterrows()
        }

    @staticmethod
    def _build_index_history(df: pd.DataFrame) -> pd.DataFrame:
        grouped = (
            df.groupby(["year", "quarter"], as_index=False)["Index"]
            .median()
            .rename(columns={"Index": "index_median"})
        )
        grouped["year"] = grouped["year"].astype(int)
        grouped["quarter"] = grouped["quarter"].astype(int)
        grouped = grouped.sort_values(["year", "quarter"]).reset_index(drop=True)
        return grouped

    @staticmethod
    def _latest_observed_period(index_history: pd.DataFrame) -> tuple[int, int]:
        if index_history.empty:
            return (0, 0)
        last = index_history.iloc[-1]
        return int(last["year"]), int(last["quarter"])

    @staticmethod
    def _quarter_ordinal(year: int, quarter: int) -> int:
        return (int(year) * 4) + (int(quarter) - 1)

    def _quarter_distance(self, start: tuple[int, int], end: tuple[int, int]) -> int:
        return self._quarter_ordinal(end[0], end[1]) - self._quarter_ordinal(start[0], start[1])

    @staticmethod
    def _fit_index_forecast(index_history: pd.DataFrame) -> _IndexForecastModel | None:
        if index_history.shape[0] < 8:
            return None

        y = index_history["index_median"].to_numpy(dtype=float)
        t = np.arange(index_history.shape[0], dtype=float)
        q = index_history["quarter"].to_numpy(dtype=int)
        q2 = (q == 2).astype(float)
        q3 = (q == 3).astype(float)
        q4 = (q == 4).astype(float)
        x = np.column_stack([np.ones_like(t), t, t**2, q2, q3, q4])
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        yhat = x @ coeffs
        train_rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

        y_safe = np.maximum(y, 1e-6)
        q_log_growth = np.diff(np.log(y_safe))
        if q_log_growth.size == 0:
            growth = 0.0
        else:
            recent = q_log_growth[-8:] if q_log_growth.size >= 8 else q_log_growth
            growth = float(np.median(recent))
        growth = float(np.clip(growth, -0.03, 0.03))

        return _IndexForecastModel(
            coeffs=coeffs,
            train_rmse=train_rmse,
            quarterly_log_growth=growth,
            n_points=int(index_history.shape[0]),
            last_index=float(y[-1]),
        )

    def _forecast_index_for_period(self, target_year: int, target_quarter: int) -> _IndexResolution | None:
        model = self.index_forecast_model
        if model is None:
            return None

        horizon = self._quarter_distance(self.latest_observed_period, (target_year, target_quarter))
        if horizon <= 0:
            return None

        t_future = float((model.n_points - 1) + horizon)
        q2 = 1.0 if int(target_quarter) == 2 else 0.0
        q3 = 1.0 if int(target_quarter) == 3 else 0.0
        q4 = 1.0 if int(target_quarter) == 4 else 0.0
        x_future = np.array([1.0, t_future, t_future**2, q2, q3, q4], dtype=float)
        trend_pred = float(x_future @ model.coeffs)
        growth_pred = float(model.last_index * np.exp(model.quarterly_log_growth * horizon))

        # Blend parametric trend with recent-growth extrapolation for stability.
        blended = (0.65 * trend_pred) + (0.35 * growth_pred)
        cap_high = float(model.last_index * ((1.05) ** horizon))
        cap_low = float(max(40.0, model.last_index * ((0.97) ** horizon)))
        value = float(np.clip(blended, cap_low, cap_high))
        return _IndexResolution(
            value=max(40.0, value),
            source="forecast_market_index",
            horizon_quarters=horizon,
        )

    def _load_artifact_models(self) -> None:
        if joblib is None:
            return

        for threshold in VALID_THRESHOLDS:
            model_path = model_path_for_threshold(threshold)
            if not model_path.exists():
                continue

            model = joblib.load(model_path)
            meta_path = model_meta_path_for_threshold(threshold)
            meta: dict[str, Any] = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "good_school_threshold" not in meta:
                meta["good_school_threshold"] = threshold
            self.bundles[threshold] = _ArtifactBundle(model=model, meta=meta)

        # Backward compatibility with legacy single-artifact setup.
        if DEFAULT_GOOD_SCHOOL_THRESHOLD not in self.bundles and MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            meta: dict[str, Any] = {}
            if MODEL_META_PATH.exists():
                meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
            meta.setdefault("good_school_threshold", DEFAULT_GOOD_SCHOOL_THRESHOLD)
            self.bundles[DEFAULT_GOOD_SCHOOL_THRESHOLD] = _ArtifactBundle(model=model, meta=meta)

    def _select_bundle(self, requested_threshold: int) -> tuple[int, _ArtifactBundle, str] | None:
        if requested_threshold in self.bundles:
            return requested_threshold, self.bundles[requested_threshold], ""

        if DEFAULT_GOOD_SCHOOL_THRESHOLD in self.bundles:
            return (
                DEFAULT_GOOD_SCHOOL_THRESHOLD,
                self.bundles[DEFAULT_GOOD_SCHOOL_THRESHOLD],
                (
                    f"Threshold {requested_threshold} artifact not found. "
                    f"Used threshold {DEFAULT_GOOD_SCHOOL_THRESHOLD} artifact instead."
                ),
            )

        if self.bundles:
            nearest = sorted(self.bundles.keys(), key=lambda t: abs(t - requested_threshold))[0]
            return (
                nearest,
                self.bundles[nearest],
                f"Threshold {requested_threshold} artifact not found. Used nearest available threshold {nearest}.",
            )
        return None

    def _valuation_index(self, req: PredictRequest, anchor_row: pd.Series) -> _IndexResolution:
        key = (int(req.valuation_year), int(req.valuation_quarter))
        if key in self.index_lookup:
            return _IndexResolution(
                value=self.index_lookup[key],
                source="observed_quarter",
                horizon_quarters=0,
            )

        if key > self.latest_observed_period:
            forecast = self._forecast_index_for_period(int(req.valuation_year), int(req.valuation_quarter))
            if forecast is not None:
                return forecast

        year_keys = [k for k in self.index_lookup if k[0] == int(req.valuation_year)]
        if year_keys:
            return _IndexResolution(
                value=float(np.median([self.index_lookup[k] for k in year_keys])),
                source="observed_year_median",
                horizon_quarters=None,
            )
        if "Index" in anchor_row:
            return _IndexResolution(
                value=float(anchor_row["Index"]),
                source="anchor_row_fallback",
                horizon_quarters=None,
            )
        return _IndexResolution(
            value=self.global_index_median,
            source="global_median_fallback",
            horizon_quarters=None,
        )

    def predict(self, req: PredictRequest, model_features: dict[str, Any], anchor_row: pd.Series) -> PredictionResult:
        comparables_df = self.feature_engine.select_comparables(req, anchor_row, req.comparables_limit)
        comparable_records = [
            ComparableRecord(
                month=str(r["month"]),
                town=str(r["town"]),
                full_address=str(r["full_address"]),
                floor_area_sqm=float(r["floor_area_sqm"]),
                remaining_lease_years=float(r["remaining_lease_years"]),
                resale_price=float(r["resale_price"]),
                real_price=float(r["real_price"]),
                real_price_psf=float(r["real_price_psf"]),
                distance_score=float(r["distance_score"]),
            )
            for _, r in comparables_df.iterrows()
        ]

        index_resolution = self._valuation_index(req, anchor_row)
        index_used = index_resolution.value
        index_factor = index_used / 100.0

        selected = self._select_bundle(req.good_school_threshold)
        if selected is not None:
            threshold_used, bundle, threshold_note = selected
            model_features_for_bundle = model_features
            if threshold_used != req.good_school_threshold:
                req_for_bundle = req.model_copy(update={"good_school_threshold": threshold_used})
                model_features_for_bundle = self.feature_engine.build(req_for_bundle).model_features
            return self._predict_with_artifact(
                req=req,
                model_features=model_features_for_bundle,
                comparables=comparable_records,
                comparables_df=comparables_df,
                bundle=bundle,
                threshold_used=threshold_used,
                threshold_note=threshold_note,
                index_used=index_used,
                index_factor=index_factor,
                index_resolution=index_resolution,
            )

        return self._predict_with_comparables(
            req=req,
            comparables=comparable_records,
            comparables_df=comparables_df,
            index_used=index_used,
            index_factor=index_factor,
            note="No trained model artifact found. Used comparable-median fallback.",
            index_resolution=index_resolution,
        )

    @staticmethod
    def _interval_from_meta(predicted_real: float, meta: dict[str, Any]) -> tuple[float, float]:
        calibration = meta.get("interval_calibration", {})
        ape_q90_pct = calibration.get("ape_q90_pct")
        if ape_q90_pct is None:
            # Backward compatibility with earlier meta shape.
            ape_q90_pct = ((meta.get("validation_nominal", {}) or {}).get("ape_q90_pct"))
        if ape_q90_pct is not None:
            rel = float(ape_q90_pct) / 100.0
            lower = max(0.0, predicted_real * (1.0 - rel))
            upper = predicted_real * (1.0 + rel)
            return lower, upper

        residual_std = float(meta.get("validation_residual_std", 0.10))
        lower = float(np.exp(np.log(predicted_real) - 1.2816 * residual_std))
        upper = float(np.exp(np.log(predicted_real) + 1.2816 * residual_std))
        return max(0.0, lower), upper

    def _predict_with_artifact(
        self,
        req: PredictRequest,
        model_features: dict[str, Any],
        comparables: list[ComparableRecord],
        comparables_df: pd.DataFrame,
        bundle: _ArtifactBundle,
        threshold_used: int,
        threshold_note: str,
        index_used: float,
        index_factor: float,
        index_resolution: _IndexResolution,
    ) -> PredictionResult:
        feature_df = pd.DataFrame([model_features])
        try:
            yhat = float(bundle.model.predict(feature_df)[0])
        except Exception as exc:
            note = (
                f"Artifact prediction failed due to feature mismatch ({exc}). "
                "Used comparable-median fallback for this request."
            )
            return self._predict_with_comparables(
                req=req,
                comparables=comparables,
                comparables_df=comparables_df,
                index_used=index_used,
                index_factor=index_factor,
                note=note,
                index_resolution=index_resolution,
            )

        target_scale = str(bundle.meta.get("target_scale", "log_real_price")).lower()
        if target_scale in {"log_price", "log_real_price"}:
            predicted_real = float(np.exp(yhat))
        else:
            predicted_real = yhat

        p10_real, p90_real = self._interval_from_meta(predicted_real, bundle.meta)
        p50_real = predicted_real

        predicted_nominal = predicted_real * index_factor
        p10_nominal = p10_real * index_factor
        p50_nominal = p50_real * index_factor
        p90_nominal = p90_real * index_factor

        base_note = (
            "Prediction generated by threshold-specific artifact model with validation-calibrated uncertainty."
        )
        if threshold_note:
            base_note = f"{base_note} {threshold_note}"
        if target_scale in {"log_price", "log_real_price"}:
            base_note = (
                f"{base_note} Model predicts real-price scale and is converted to nominal using valuation index."
            )
        if index_resolution.source == "forecast_market_index":
            latest_year, latest_quarter = self.latest_observed_period
            horizon = index_resolution.horizon_quarters or 0
            base_note = (
                f"{base_note} Future valuation period detected ({req.valuation_year} Q{req.valuation_quarter}). "
                f"Nominal resale conversion used forecasted market index {index_used:.2f} "
                f"({horizon} quarter(s) ahead of latest observed {latest_year} Q{latest_quarter})."
            )
        elif index_resolution.source == "observed_year_median":
            base_note = (
                f"{base_note} Requested quarter index unavailable; used median index for {req.valuation_year}."
            )
        elif index_resolution.source in {"anchor_row_fallback", "global_median_fallback"}:
            base_note = (
                f"{base_note} Requested period index unavailable; used {index_resolution.source.replace('_', ' ')}."
            )

        return PredictionResult(
            predicted_price_nominal=predicted_nominal,
            predicted_price_real=predicted_real,
            p10_nominal=p10_nominal,
            p50_nominal=p50_nominal,
            p90_nominal=p90_nominal,
            p10_real=p10_real,
            p50_real=p50_real,
            p90_real=p90_real,
            valuation_price_index_used=index_used,
            threshold_requested=req.good_school_threshold,
            threshold_used=threshold_used,
            model_mode="artifact_model",
            model_version=str(bundle.meta.get("model_version", f"artifact-good{threshold_used}")),
            sample_size_used=int(comparables_df.shape[0]),
            note=base_note,
            comparables=comparables,
        )

    @staticmethod
    def _predict_with_comparables(
        req: PredictRequest,
        comparables: list[ComparableRecord],
        comparables_df: pd.DataFrame,
        index_used: float,
        index_factor: float,
        note: str,
        index_resolution: _IndexResolution,
    ) -> PredictionResult:
        values_real = comparables_df["real_price"].astype(float).to_numpy()
        p10_real = float(np.percentile(values_real, 10))
        p50_real = float(np.percentile(values_real, 50))
        p90_real = float(np.percentile(values_real, 90))

        if index_resolution.source == "forecast_market_index":
            horizon = index_resolution.horizon_quarters or 0
            note = (
                f"{note} Future valuation period used forecasted market index {index_used:.2f} "
                f"({horizon} quarter(s) ahead)."
            )

        return PredictionResult(
            predicted_price_nominal=p50_real * index_factor,
            predicted_price_real=p50_real,
            p10_nominal=p10_real * index_factor,
            p50_nominal=p50_real * index_factor,
            p90_nominal=p90_real * index_factor,
            p10_real=p10_real,
            p50_real=p50_real,
            p90_real=p90_real,
            valuation_price_index_used=index_used,
            threshold_requested=req.good_school_threshold,
            threshold_used=req.good_school_threshold,
            model_mode="comparable_baseline",
            model_version="comparable-median-v2",
            sample_size_used=int(comparables_df.shape[0]),
            note=note,
            comparables=comparables,
        )

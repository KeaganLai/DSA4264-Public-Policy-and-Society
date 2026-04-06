from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DATA_PATH
from .schemas import PredictRequest


@dataclass
class FeaturePayload:
    anchor_row: pd.Series
    model_features: dict[str, float | int | str]
    snapshot_features: dict[str, float | int | str]


class FeatureEngine:
    """Builds model inputs from user inputs and historical anchors."""

    def __init__(self, data_path=DATA_PATH) -> None:
        self.df = pd.read_csv(data_path)

    def build(self, req: PredictRequest) -> FeaturePayload:
        anchor = self._resolve_anchor_row(req)
        threshold = req.good_school_threshold

        remaining_lease_years = self._resolve_remaining_lease(req, anchor)
        mature_estate = int(anchor["mature_estate"] if req.mature_estate is None else req.mature_estate)

        dist_hawker = self._resolve_distance(req.dist_to_nearest_hawker_km, anchor["dist_to_nearest_hawker_km"])
        dist_bus = self._resolve_distance(req.dist_to_nearest_busstop_km, anchor["dist_to_nearest_busstop_km"])
        dist_mall = self._resolve_distance(req.dist_to_nearest_mall_km, anchor["dist_to_nearest_mall_km"])
        dist_mrt = self._resolve_distance(req.dist_to_nearest_mrt_km, anchor["dist_to_nearest_mrt_km"])
        dist_cbd = self._resolve_distance(req.dist_to_cbd_km, anchor["dist_to_cbd_km"])

        d_0_1_col = f"d_0_1km_good{threshold}"
        d_1_2_col = f"d_1_2km_good{threshold}"
        c_0_1_col = f"count_0_1km_good{threshold}"
        c_1_2_col = f"count_1_2km_good{threshold}"
        dist_school_col = f"dist_nearest_goodschool_{threshold}"
        school_name_col = f"sch_name_{threshold}"

        model_features: dict[str, float | int | str] = {
            "lat": float(anchor["lat"]),
            "lon": float(anchor["lon"]),
            "town": str(req.town or anchor["town"]),
            "flat_type": int(req.flat_type),
            "storey_relative_category": req.storey_relative_category,
            "mature_estate": mature_estate,
            "floor_area_sqm": float(req.floor_area_sqm),
            "remaining_lease_years": float(remaining_lease_years),
            "year": int(req.valuation_year),
            "quarter": int(req.valuation_quarter),
            "dist_to_nearest_hawker_km": dist_hawker,
            "dist_to_nearest_busstop_km": dist_bus,
            "dist_to_nearest_mall_km": dist_mall,
            "dist_to_nearest_mrt_km": dist_mrt,
            "dist_to_cbd_km": dist_cbd,
            "countall_0_1km": int(anchor["countall_0_1km"]),
            "countall_1_2km": int(anchor["countall_1_2km"]),
            d_0_1_col: int(anchor[d_0_1_col]),
            d_1_2_col: int(anchor[d_1_2_col]),
            c_0_1_col: int(anchor[c_0_1_col]),
            c_1_2_col: int(anchor[c_1_2_col]),
            dist_school_col: float(anchor[dist_school_col]),
            school_name_col: str(anchor[school_name_col]),
        }

        snapshot_features: dict[str, float | int | str] = {
            "lat": float(anchor["lat"]),
            "lon": float(anchor["lon"]),
            "town": str(req.town or anchor["town"]),
            "nearest_school_name": str(anchor[school_name_col]),
            "nearest_school_distance_km": float(anchor[dist_school_col]),
            "dist_to_nearest_mrt_km": dist_mrt,
            "dist_to_nearest_mall_km": dist_mall,
            "dist_to_nearest_hawker_km": dist_hawker,
            "dist_to_nearest_busstop_km": dist_bus,
            "dist_to_cbd_km": dist_cbd,
            "countall_0_1km": int(anchor["countall_0_1km"]),
            "countall_1_2km": int(anchor["countall_1_2km"]),
            "good_school_0_1km": int(anchor[c_0_1_col]),
            "good_school_1_2km": int(anchor[c_1_2_col]),
        }

        return FeaturePayload(anchor_row=anchor, model_features=model_features, snapshot_features=snapshot_features)

    def _resolve_anchor_row(self, req: PredictRequest) -> pd.Series:
        if req.latitude is not None and req.longitude is not None:
            return self._nearest_by_coordinate(req.latitude, req.longitude)
        if req.address:
            match = self.df.loc[self.df["full_address"].str.upper() == req.address.upper()]
            if not match.empty:
                return match.iloc[-1]
        if req.town:
            town_match = self.df.loc[self.df["town"].str.upper() == req.town.upper()]
            if not town_match.empty:
                # Closest to town median location.
                lat_mid = float(town_match["lat"].median())
                lon_mid = float(town_match["lon"].median())
                d = (town_match["lat"] - lat_mid) ** 2 + (town_match["lon"] - lon_mid) ** 2
                return town_match.loc[d.idxmin()]
        return self.df.iloc[-1]

    def _nearest_by_coordinate(self, latitude: float, longitude: float) -> pd.Series:
        d = (self.df["lat"] - latitude) ** 2 + (self.df["lon"] - longitude) ** 2
        return self.df.loc[d.idxmin()]

    @staticmethod
    def _resolve_distance(input_value: float | None, fallback: float) -> float:
        if input_value is not None:
            return float(input_value)
        return float(fallback)

    @staticmethod
    def _resolve_remaining_lease(req: PredictRequest, anchor: pd.Series) -> float:
        if req.remaining_lease_years is not None:
            return float(req.remaining_lease_years)
        if req.lease_commence_year is not None:
            elapsed = req.valuation_year - req.lease_commence_year + ((req.valuation_quarter - 1) / 4.0)
            return max(0.0, 99.0 - float(elapsed))
        return float(anchor["remaining_lease_years"])

    def select_comparables(self, req: PredictRequest, anchor: pd.Series, limit: int) -> pd.DataFrame:
        # Comparable selection is intentionally simple and transparent for policy users.
        mask = self.df["flat_type"] == req.flat_type
        if req.town:
            mask = mask & (self.df["town"].str.upper() == req.town.upper())

        comps = self.df.loc[mask].copy()
        if comps.empty:
            comps = self.df.copy()

        comps["distance_score"] = (
            ((comps["lat"] - float(anchor["lat"])) ** 2 + (comps["lon"] - float(anchor["lon"])) ** 2) ** 0.5 * 100
            + (comps["floor_area_sqm"] - req.floor_area_sqm).abs() / 10
            + (comps["remaining_lease_years"] - self._resolve_remaining_lease(req, anchor)).abs() / 5
        )
        comps = comps.sort_values("distance_score").head(limit)
        return comps

    @staticmethod
    def fallback_interval(values: pd.Series) -> tuple[float, float, float]:
        p10 = float(np.percentile(values, 10))
        p50 = float(np.percentile(values, 50))
        p90 = float(np.percentile(values, 90))
        return p10, p50, p90

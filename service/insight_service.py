from __future__ import annotations

from dataclasses import dataclass

from .config import RDD_SUMMARY_PATH


@dataclass(frozen=True)
class HedonicRow:
    premium_0_1km_pct: float
    premium_1_2km_pct: float
    adj_r2: float


HEDONIC_THRESHOLD_RESULTS: dict[int, HedonicRow] = {
    75: HedonicRow(0.4597, 0.8006, 0.8227),
    80: HedonicRow(1.1599, 0.4148, 0.8229),
    85: HedonicRow(1.2686, 0.3356, 0.8230),
    90: HedonicRow(0.6684, 0.1244, 0.8226),
}


class InsightService:
    def __init__(self) -> None:
        self.rdd_effect_pct = -0.53
        self.rdd_p_value = 0.328
        self._load_rdd_summary_if_available()

    def _load_rdd_summary_if_available(self) -> None:
        if not RDD_SUMMARY_PATH.exists():
            return
        text = RDD_SUMMARY_PATH.read_text(encoding="utf-8")
        # Parse the first adjusted 1km line in the summary for transparent reporting.
        # Expected format contains "pct=" and "p=".
        for line in text.splitlines():
            if "schoolfe_adjusted" in line and "cutoff=1.0 km" in line and "pct=" in line and "p=" in line:
                try:
                    pct_str = line.split("pct=")[1].split("%")[0].strip()
                    p_str = line.split("p=")[1].split("|")[0].strip()
                    self.rdd_effect_pct = float(pct_str)
                    self.rdd_p_value = float(p_str)
                except Exception:
                    pass
                break

    def hedonic_evidence(self, threshold: int) -> tuple[float, float, float]:
        row = HEDONIC_THRESHOLD_RESULTS[threshold]
        return row.premium_0_1km_pct, row.premium_1_2km_pct, row.adj_r2

    def rdd_evidence(self) -> tuple[float, float, str]:
        interpretation = (
            "RDD school-FE estimate at 1km is not statistically significant; use as a cautionary robustness signal, "
            "not as causal proof."
        )
        return self.rdd_effect_pct, self.rdd_p_value, interpretation

    def build_insights(
        self,
        threshold: int,
        nearest_school_distance_km: float,
        nearest_school_name: str,
        predicted_price: float,
        p10: float,
        p90: float,
    ) -> list[str]:
        premium_0_1, premium_1_2, _ = self.hedonic_evidence(threshold)
        rdd_effect_pct, rdd_p, _ = self.rdd_evidence()

        school_zone_text = "within 1 km" if nearest_school_distance_km <= 1.0 else "outside 1 km"

        return [
            (
                f"Predicted resale price is about SGD {predicted_price:,.0f}, with an uncertainty range of "
                f"SGD {p10:,.0f} to SGD {p90:,.0f}."
            ),
            (
                f"The nearest top-threshold school is {nearest_school_name} at {nearest_school_distance_km:.2f} km, "
                f"placing this unit {school_zone_text} of the priority zone."
            ),
            (
                f"Hedonic estimates at threshold {threshold} indicate an average premium of {premium_0_1:.2f}% "
                f"for 0-1 km and {premium_1_2:.2f}% for 1-2 km after controls."
            ),
            (
                f"RDD school-FE robustness around 1 km is {rdd_effect_pct:.2f}% (p={rdd_p:.3f}), so this should be "
                f"interpreted as supportive context rather than standalone causal evidence."
            ),
        ]


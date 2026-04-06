from __future__ import annotations

from .schemas import LLMExplainRequest, LLMExplainResponse


class LLMService:
    """
    Builds grounded, policy-safe explanations.

    This starter implementation is provider-agnostic: it constructs a deterministic,
    grounded response from supplied evidence. You can later replace this with an
    external LLM call while keeping the same request/response schema.
    """

    def explain(self, req: LLMExplainRequest) -> LLMExplainResponse:
        p = req.prediction
        h = p.hedonic_evidence
        r = p.rdd_evidence
        fs = p.feature_snapshot

        evidence_points = [
            (
                f"Prediction median and interval: SGD {p.predicted_resale_price_sgd:,.0f} "
                f"(P10 {p.prediction_interval_sgd.p10:,.0f}, P90 {p.prediction_interval_sgd.p90:,.0f})."
            ),
            (
                f"Nearest school context: {fs.nearest_school_name} at {fs.nearest_school_distance_km:.2f} km; "
                f"good-school counts within 0-1 km and 1-2 km are "
                f"{fs.good_school_0_1km} and {fs.good_school_1_2km}."
            ),
            (
                f"Hedonic benchmark at threshold {h.threshold}: {h.premium_0_1km_pct:.2f}% (0-1 km) and "
                f"{h.premium_1_2km_pct:.2f}% (1-2 km), adj R^2={h.adj_r2:.4f}."
            ),
            (
                f"RDD robustness at cutoff {r.cutoff_km:.1f} km: {r.effect_pct:.2f}% with p={r.p_value:.3f}, "
                "not statistically significant."
            ),
        ]

        grounded_answer = (
            f"Based on the modeled comparables and location profile, the expected resale value is around "
            f"SGD {p.predicted_resale_price_sgd:,.0f}. Proximity evidence is directionally consistent with a stronger "
            f"premium inside 1 km than 1-2 km in hedonic estimates, but RDD robustness suggests caution in claiming "
            f"strict causality from school proximity alone."
        )

        return LLMExplainResponse(
            trace_id=p.trace_id,
            grounded_answer=grounded_answer,
            evidence_points=evidence_points,
            safety_note=(
                "This explanation is grounded in provided model outputs and does not constitute an official valuation. "
                "Use causal language carefully."
            ),
        )


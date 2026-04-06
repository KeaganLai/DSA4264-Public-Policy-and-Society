from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


StoreyCategory = Literal["LOW_IN_ESTATE", "MID_IN_ESTATE", "HIGH_IN_ESTATE"]
GoodSchoolThreshold = Literal[75, 80, 85, 90]


def _default_year() -> int:
    return datetime.utcnow().year


def _default_quarter() -> int:
    return ((datetime.utcnow().month - 1) // 3) + 1


class PredictRequest(BaseModel):
    address: str | None = Field(
        default=None,
        description="Optional full address. If it matches a historical record, it is used as location anchor.",
    )
    latitude: float | None = Field(default=None, ge=1.15, le=1.48)
    longitude: float | None = Field(default=None, ge=103.55, le=104.10)
    town: str | None = Field(default=None, description="Optional town fallback when no coordinates are provided.")

    flat_type: int = Field(ge=1, le=7, description="Numeric flat type code used in your dataset.")
    floor_area_sqm: float = Field(gt=20, lt=300)
    remaining_lease_years: float | None = Field(default=None, ge=0, le=99)
    lease_commence_year: int | None = Field(default=None, ge=1960, le=2100)
    storey_relative_category: StoreyCategory = "MID_IN_ESTATE"
    mature_estate: bool | None = None

    valuation_year: int = Field(ge=2013, le=2100)
    valuation_quarter: Literal[1, 2, 3, 4] = 1
    good_school_threshold: GoodSchoolThreshold = 80

    dist_to_nearest_hawker_km: float | None = Field(default=None, ge=0, le=30)
    dist_to_nearest_busstop_km: float | None = Field(default=None, ge=0, le=30)
    dist_to_nearest_mall_km: float | None = Field(default=None, ge=0, le=30)
    dist_to_nearest_mrt_km: float | None = Field(default=None, ge=0, le=30)
    dist_to_cbd_km: float | None = Field(default=None, ge=0, le=50)

    include_comparables: bool = True
    comparables_limit: int = Field(default=12, ge=3, le=30)
    include_llm_insights: bool = True

    @model_validator(mode="after")
    def validate_location_inputs(self) -> "PredictRequest":
        has_lat = self.latitude is not None
        has_lon = self.longitude is not None
        if has_lat ^ has_lon:
            raise ValueError("`latitude` and `longitude` must be provided together.")
        if not self.address and not self.town and not (has_lat and has_lon):
            raise ValueError("Provide one of: (`latitude` + `longitude`), or `address`, or `town`.")
        return self


class ComparableRecord(BaseModel):
    month: str
    town: str
    full_address: str
    floor_area_sqm: float
    remaining_lease_years: float
    resale_price: float
    real_price: float
    real_price_psf: float
    distance_score: float


class FeatureSnapshot(BaseModel):
    lat: float
    lon: float
    town: str
    nearest_school_name: str
    nearest_school_distance_km: float
    dist_to_nearest_mrt_km: float
    dist_to_nearest_mall_km: float
    dist_to_nearest_hawker_km: float
    dist_to_nearest_busstop_km: float
    dist_to_cbd_km: float
    countall_0_1km: int
    countall_1_2km: int
    good_school_0_1km: int
    good_school_1_2km: int


class PredictionInterval(BaseModel):
    p10: float
    p50: float
    p90: float


class ModelDiagnostics(BaseModel):
    model_mode: Literal["artifact_model", "comparable_baseline"]
    model_version: str
    threshold_requested: int
    threshold_used: int
    valuation_price_index_used: float
    sample_size_used: int
    note: str


class HedonicEvidence(BaseModel):
    threshold: int
    premium_0_1km_pct: float
    premium_1_2km_pct: float
    adj_r2: float


class RddEvidence(BaseModel):
    cutoff_km: float
    effect_pct: float
    p_value: float
    interpretation: str


class PredictResponse(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    predicted_resale_price_sgd: float
    prediction_interval_sgd: PredictionInterval
    predicted_real_price_sgd: float
    prediction_interval_real_sgd: PredictionInterval
    feature_snapshot: FeatureSnapshot
    model_diagnostics: ModelDiagnostics
    comparables: list[ComparableRecord] = Field(default_factory=list)
    hedonic_evidence: HedonicEvidence
    rdd_evidence: RddEvidence
    insights: list[str] = Field(default_factory=list)


class LLMExplainRequest(BaseModel):
    question: str = Field(min_length=5)
    prediction: PredictResponse


class LLMExplainResponse(BaseModel):
    trace_id: str
    grounded_answer: str
    evidence_points: list[str]
    safety_note: str


class FriendlyPredictRequest(BaseModel):
    town: str
    street_name: str
    block: str
    number_of_rooms: int = Field(ge=1, le=7)
    floor_area_sqm: float = Field(gt=20, lt=300)
    lease_commence_year: int = Field(ge=1960, le=2100)
    storey_number: int | None = Field(
        default=None,
        ge=1,
        le=60,
        description="User-friendly floor level. Preferred over storey band.",
    )
    storey_range: str | None = Field(
        default=None,
        description='Optional range input like "10 TO 12".',
    )
    storey_relative_category: StoreyCategory | None = Field(
        default=None,
        description="Optional direct band input for advanced clients.",
    )
    mature_estate: bool | None = None
    valuation_year: int = Field(default_factory=_default_year, ge=2013, le=2100)
    valuation_quarter: Literal[1, 2, 3, 4] = Field(default_factory=_default_quarter)
    good_school_threshold: GoodSchoolThreshold = 80
    include_comparables: bool = True
    comparables_limit: int = Field(default=12, ge=3, le=30)
    include_llm_insights: bool = True

    @model_validator(mode="after")
    def validate_storey_inputs(self) -> "FriendlyPredictRequest":
        if self.storey_number is None and not self.storey_range and self.storey_relative_category is None:
            raise ValueError(
                "Provide one of: `storey_number`, `storey_range` (e.g. '10 TO 12'), "
                "or `storey_relative_category`."
            )
        return self


class ResolvedLocation(BaseModel):
    method: Literal["onemap", "local_fallback", "town_fallback"]
    normalized_town: str
    normalized_street_name: str
    normalized_block: str
    search_query: str
    matched_address: str | None = None
    latitude: float
    longitude: float


class FriendlyPredictResponse(BaseModel):
    resolved_location: ResolvedLocation
    prediction: PredictResponse


class ChatMessageRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)


class ChatMessageResponse(BaseModel):
    session_id: str
    intent: Literal["prediction", "analytics", "general"]
    reply: str
    requires_follow_up: bool
    missing_fields: list[str] = Field(default_factory=list)
    collected_slots: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    assumptions_used: list[str] = Field(default_factory=list)
    confidence_tier: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    prediction: FriendlyPredictResponse | None = None

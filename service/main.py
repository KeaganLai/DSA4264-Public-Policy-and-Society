from __future__ import annotations

from pathlib import Path
import re

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .chatbot_service import ChatbotService
from .config import (
    ALLOW_USER_THRESHOLD_SELECTION,
    FORCED_GOOD_SCHOOL_THRESHOLD,
    ROOT_DIR,
    VALID_THRESHOLDS,
)
from .feature_engine import FeatureEngine
from .geocoding import OneMapGeocoder
from .insight_service import InsightService
from .llm_service import LLMService
from .model_service import ModelService
from .policy_llm_service import PolicyLLMService
from .rag_service import RagService
from .schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    FeatureSnapshot,
    FriendlyPredictRequest,
    FriendlyPredictResponse,
    HedonicEvidence,
    LLMExplainRequest,
    LLMExplainResponse,
    ModelDiagnostics,
    PredictRequest,
    PredictResponse,
    PredictionInterval,
    RddEvidence,
    StoreyCategory,
)

app = FastAPI(
    title="HDB Resale Prediction + LLM Insights API",
    version="0.1.0",
    description=(
        "Starter API for natural-language policy exploration of HDB resale predictions, "
        "hedonic estimates, and RDD robustness context."
    ),
)

feature_engine = FeatureEngine()
geocoder = OneMapGeocoder(feature_engine.df)
model_service = ModelService(feature_engine)
insight_service = InsightService()
llm_service = LLMService()
rag_service = RagService()
policy_llm_service = PolicyLLMService()
web_root = ROOT_DIR / "service" / "web"


def _effective_threshold(requested_threshold: int) -> int:
    if not ALLOW_USER_THRESHOLD_SELECTION:
        return FORCED_GOOD_SCHOOL_THRESHOLD
    if requested_threshold in VALID_THRESHOLDS:
        return requested_threshold
    return FORCED_GOOD_SCHOOL_THRESHOLD


def _storey_band_from_number(storey_number: int) -> StoreyCategory:
    # User-facing approximation. Keep transparent and consistent.
    if storey_number <= 6:
        return "LOW_IN_ESTATE"
    if storey_number <= 12:
        return "MID_IN_ESTATE"
    return "HIGH_IN_ESTATE"


def _storey_band_from_range(storey_range: str) -> StoreyCategory | None:
    # Accept forms like "10 TO 12", "10-12", "10 12".
    nums = [int(x) for x in re.findall(r"\d+", storey_range or "")]
    if not nums:
        return None
    avg = round(sum(nums) / len(nums))
    return _storey_band_from_number(avg)


def _resolve_storey_band(req: FriendlyPredictRequest) -> StoreyCategory:
    if req.storey_number is not None:
        return _storey_band_from_number(req.storey_number)
    if req.storey_range:
        derived = _storey_band_from_range(req.storey_range)
        if derived is not None:
            return derived
    if req.storey_relative_category is not None:
        return req.storey_relative_category
    return "MID_IN_ESTATE"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(Path(web_root) / "dashboard.html")


@app.get("/predict-ui", include_in_schema=False)
def predict_ui() -> FileResponse:
    return FileResponse(Path(web_root) / "index.html")


@app.get("/chat-ui", include_in_schema=False)
def chat_ui() -> FileResponse:
    return FileResponse(Path(web_root) / "chat.html")


@app.get("/health")
def health() -> dict[str, str | bool | None]:
    llm_status = policy_llm_service.status(check_connection=False)
    return {
        "status": "ok",
        "llm_provider": str(llm_status.get("provider")),
        "llm_enabled": bool(llm_status.get("enabled")),
    }


@app.get("/llm/status")
def llm_status(check: bool = True) -> dict[str, str | bool | None]:
    """
    LLM diagnostics endpoint.
    - check=false: returns configured provider state only.
    - check=true: performs a lightweight provider connectivity probe.
    """
    return policy_llm_service.status(check_connection=check)


def _run_prediction(req: PredictRequest) -> PredictResponse:
    effective_threshold = _effective_threshold(req.good_school_threshold)
    req_effective = req.model_copy(update={"good_school_threshold": effective_threshold})

    payload = feature_engine.build(req_effective)
    pred = model_service.predict(req_effective, payload.model_features, payload.anchor_row)

    premium_0_1, premium_1_2, adj_r2 = insight_service.hedonic_evidence(req_effective.good_school_threshold)
    rdd_effect_pct, rdd_p, rdd_interpretation = insight_service.rdd_evidence()

    insights = []
    if req_effective.include_llm_insights:
        insights = insight_service.build_insights(
            threshold=req_effective.good_school_threshold,
            nearest_school_distance_km=float(payload.snapshot_features["nearest_school_distance_km"]),
            nearest_school_name=str(payload.snapshot_features["nearest_school_name"]),
            predicted_price=pred.predicted_price_nominal,
            p10=pred.p10_nominal,
            p90=pred.p90_nominal,
        )
    if req.good_school_threshold != req_effective.good_school_threshold:
        insights.insert(
            0,
            (
                f"Threshold selection is currently fixed to {req_effective.good_school_threshold} "
                f"by backend configuration; requested {req.good_school_threshold}."
            ),
        )

    return PredictResponse(
        predicted_resale_price_sgd=pred.predicted_price_nominal,
        prediction_interval_sgd=PredictionInterval(
            p10=pred.p10_nominal,
            p50=pred.p50_nominal,
            p90=pred.p90_nominal,
        ),
        predicted_real_price_sgd=pred.predicted_price_real,
        prediction_interval_real_sgd=PredictionInterval(
            p10=pred.p10_real,
            p50=pred.p50_real,
            p90=pred.p90_real,
        ),
        feature_snapshot=FeatureSnapshot(**payload.snapshot_features),
        model_diagnostics=ModelDiagnostics(
            model_mode=pred.model_mode,
            model_version=pred.model_version,
            threshold_requested=req.good_school_threshold,
            threshold_used=pred.threshold_used,
            valuation_price_index_used=pred.valuation_price_index_used,
            sample_size_used=pred.sample_size_used,
            note=pred.note,
        ),
        comparables=pred.comparables if req_effective.include_comparables else [],
        hedonic_evidence=HedonicEvidence(
            threshold=req_effective.good_school_threshold,
            premium_0_1km_pct=premium_0_1,
            premium_1_2km_pct=premium_1_2,
            adj_r2=adj_r2,
        ),
        rdd_evidence=RddEvidence(
            cutoff_km=1.0,
            effect_pct=rdd_effect_pct,
            p_value=rdd_p,
            interpretation=rdd_interpretation,
        ),
        insights=insights,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    return _run_prediction(req)


def _run_friendly_prediction(req: FriendlyPredictRequest) -> FriendlyPredictResponse:
    resolved = geocoder.resolve(req.town, req.street_name, req.block)
    storey_band = _resolve_storey_band(req)

    predict_req = PredictRequest(
        address=f"{resolved.normalized_block} {resolved.normalized_street_name}",
        latitude=resolved.latitude,
        longitude=resolved.longitude,
        town=resolved.normalized_town,
        flat_type=req.number_of_rooms,
        floor_area_sqm=req.floor_area_sqm,
        lease_commence_year=req.lease_commence_year,
        storey_relative_category=storey_band,
        mature_estate=req.mature_estate,
        valuation_year=req.valuation_year,
        valuation_quarter=req.valuation_quarter,
        good_school_threshold=req.good_school_threshold,
        include_comparables=req.include_comparables,
        comparables_limit=req.comparables_limit,
        include_llm_insights=req.include_llm_insights,
    )

    prediction = _run_prediction(predict_req)
    return FriendlyPredictResponse(
        resolved_location=resolved,
        prediction=prediction,
    )


chatbot_service = ChatbotService(
    df=feature_engine.df,
    predict_runner=_run_friendly_prediction,
    allow_user_threshold_selection=ALLOW_USER_THRESHOLD_SELECTION,
    forced_good_school_threshold=FORCED_GOOD_SCHOOL_THRESHOLD,
    rag_service=rag_service,
    policy_llm_service=policy_llm_service,
)


@app.post("/predict/friendly", response_model=FriendlyPredictResponse)
def predict_friendly(req: FriendlyPredictRequest) -> FriendlyPredictResponse:
    return _run_friendly_prediction(req)


@app.post("/chat", response_model=ChatMessageResponse)
def chat(req: ChatMessageRequest) -> ChatMessageResponse:
    try:
        return chatbot_service.handle_message(req)
    except Exception as exc:
        return ChatMessageResponse(
            session_id=req.session_id or "session-error",
            intent="general",
            reply=(
                "Chat request failed on the server. Please try again or start a new session. "
                f"Technical detail: {exc}"
            ),
            requires_follow_up=False,
            missing_fields=[],
            collected_slots={},
            assumptions_used=[],
            confidence_tier=None,
            prediction=None,
        )


@app.post("/llm/explain", response_model=LLMExplainResponse)
def llm_explain(req: LLMExplainRequest) -> LLMExplainResponse:
    return llm_service.explain(req)

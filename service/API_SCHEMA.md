# API Schema

## `POST /predict`

### Request schema

```json
{
  "address": "optional string",
  "latitude": 1.30,
  "longitude": 103.80,
  "town": "optional string",
  "flat_type": 4,
  "floor_area_sqm": 93.0,
  "remaining_lease_years": 72.5,
  "lease_commence_year": 2000,
  "storey_relative_category": "MID_IN_ESTATE",
  "mature_estate": true,
  "valuation_year": 2026,
  "valuation_quarter": 1,
  "good_school_threshold": 80,
  "dist_to_nearest_hawker_km": 0.4,
  "dist_to_nearest_busstop_km": 0.08,
  "dist_to_nearest_mall_km": 0.7,
  "dist_to_nearest_mrt_km": 0.5,
  "dist_to_cbd_km": 10.5,
  "include_comparables": true,
  "comparables_limit": 12,
  "include_llm_insights": true
}
```

Rules:
- Provide one of:
  - `latitude` + `longitude`
  - `address`
  - `town`
- `flat_type` follows dataset coding: `1..7`.
- `good_school_threshold`: `75 | 80 | 85 | 90` (or forced in backend config).

### Response schema

```json
{
  "trace_id": "uuid",
  "predicted_resale_price_sgd": 620000,
  "prediction_interval_sgd": { "p10": 570000, "p50": 620000, "p90": 690000 },
  "predicted_real_price_sgd": 422000,
  "prediction_interval_real_sgd": { "p10": 390000, "p50": 422000, "p90": 456000 },
  "feature_snapshot": {
    "lat": 1.3,
    "lon": 103.8,
    "town": "ANG MO KIO",
    "nearest_school_name": "Catholic High",
    "nearest_school_distance_km": 0.92,
    "dist_to_nearest_mrt_km": 0.45,
    "dist_to_nearest_mall_km": 0.75,
    "dist_to_nearest_hawker_km": 0.35,
    "dist_to_nearest_busstop_km": 0.07,
    "dist_to_cbd_km": 10.3,
    "countall_0_1km": 3,
    "countall_1_2km": 5,
    "good_school_0_1km": 1,
    "good_school_1_2km": 2
  },
  "model_diagnostics": {
    "model_mode": "artifact_model",
    "model_version": "hedonic-linear-good80-20260330",
    "threshold_requested": 75,
    "threshold_used": 75,
    "valuation_price_index_used": 146.9,
    "sample_size_used": 12,
    "note": "..."
  },
  "comparables": [
    {
      "month": "2025-12-01",
      "town": "ANG MO KIO",
      "full_address": "123 EXAMPLE AVE",
      "floor_area_sqm": 92,
      "remaining_lease_years": 71,
      "resale_price": 890000,
      "real_price": 610000,
      "real_price_psf": 610,
      "distance_score": 2.1
    }
  ],
  "hedonic_evidence": {
    "threshold": 80,
    "premium_0_1km_pct": 1.1599,
    "premium_1_2km_pct": 0.4148,
    "adj_r2": 0.8229
  },
  "rdd_evidence": {
    "cutoff_km": 1.0,
    "effect_pct": -0.53,
    "p_value": 0.328,
    "interpretation": "..."
  },
  "insights": ["..."]
}
```

## `POST /predict/friendly`

### Request schema

```json
{
  "town": "ANG MO KIO",
  "street_name": "AVE 8",
  "block": "510",
  "number_of_rooms": 4,
  "floor_area_sqm": 93,
  "lease_commence_year": 2000,
  "storey_number": 11,
  "storey_range": "10 TO 12",
  "valuation_year": 2026,
  "valuation_quarter": 1,
  "good_school_threshold": 80,
  "include_comparables": true,
  "comparables_limit": 12,
  "include_llm_insights": true
}
```

Notes:
- Provide at least one of `storey_number` or `storey_range`.
- Backend maps these into internal bands:
  - `1-6` => `LOW_IN_ESTATE`
  - `7-12` => `MID_IN_ESTATE`
  - `13+` => `HIGH_IN_ESTATE`
- `town + block` local matching is prioritized to avoid cross-town OneMap ambiguity.

### Response schema

```json
{
  "resolved_location": {
    "method": "local_fallback",
    "normalized_town": "ANG MO KIO",
    "normalized_street_name": "ANG MO KIO AVE 8",
    "normalized_block": "510",
    "search_query": "510 AVE 8, ANG MO KIO",
    "matched_address": "510 ANG MO KIO AVE 8",
    "latitude": 1.3734,
    "longitude": 103.8491
  },
  "prediction": { "... same payload as /predict response ..." }
}
```

## `POST /llm/explain`

### Request schema

```json
{
  "question": "What drove this estimate?",
  "prediction": { "... full /predict response payload ..." }
}
```

### Response schema

```json
{
  "trace_id": "uuid",
  "grounded_answer": "plain-language answer",
  "evidence_points": ["fact 1", "fact 2", "fact 3"],
  "safety_note": "not an official valuation"
}
```

## `POST /chat`

### Request schema

```json
{
  "session_id": "optional uuid-like string",
  "message": "average resale price in tampines in 2019"
}
```

### Response schema

```json
{
  "session_id": "chat-session-id",
  "intent": "analytics",
  "reply": "The average resale price in Tampines in 2019 is SGD 420,000 (median 405,000, n=8,120). Therefore: this is your baseline for future estimates. Report evidence: ...",
  "requires_follow_up": false,
  "missing_fields": [],
  "collected_slots": {
    "town": "TAMPINES"
  },
  "assumptions_used": [],
  "confidence_tier": null,
  "prediction": null
}
```

For prediction chats, `prediction` contains the same object returned by `POST /predict/friendly` once all required slots are collected.

When unknown critical fields are used in chat (e.g., `unknown floor area`), the response includes:
- `assumptions_used`: transparent imputation notes and source scope
- `confidence_tier`: `HIGH`, `MEDIUM`, or `LOW`
- wider prediction intervals when imputations are applied
- abstain message when data quality is insufficient for reliable estimation

For policy/general chats, the `reply` text includes:
- report-grounded retrieval snippets from local knowledge notes
- an explicit "Therefore" implication in plain language

Optional provider-backed generation:
- Set `CHAT_LLM_PROVIDER=openai` and `OPENAI_API_KEY` to let chat responses be composed by an API-hosted LLM.
- Or set `CHAT_LLM_PROVIDER=ollama` with `OLLAMA_MODEL` / `OLLAMA_BASE_URL` for a local pretrained model.
- Valuation itself remains model-computed by `/predict/friendly` pipeline.

## `GET /llm/status`

Use this to verify if chat is actually connected to an external/local LLM or running fallback mode.

Examples:
- `/llm/status?check=false` (configuration snapshot only)
- `/llm/status?check=true` (runs a lightweight connectivity probe)

### Response schema

```json
{
  "provider": "openai",
  "enabled": true,
  "openai_model": "gpt-5-mini",
  "openai_base_url": "https://api.openai.com/v1",
  "openai_api_key_configured": true,
  "ollama_model": null,
  "ollama_base_url": null,
  "reachable": false,
  "last_error": "OpenAI API returned 429 insufficient_quota. Add billing credits or use local ollama."
}
```

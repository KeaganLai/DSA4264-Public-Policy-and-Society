# HDB Resale Prediction + LLM Service (Starter)

This module gives you a production-style starter for your project objective:
- Predict resale prices from user inputs.
- Return transparent evidence and comparables.
- Provide grounded natural-language explanations for policy officers.

## What is included

- `main.py`: FastAPI app with:
  - `POST /predict`
  - `POST /predict/friendly`
  - `POST /chat`
  - `POST /llm/explain`
  - `GET /llm/status`
  - `GET /health`
  - `GET /` (tabbed policy dashboard)
  - `GET /predict-ui` (form-first prediction UI)
  - `GET /chat-ui` (chat-first UI for non-technical users)
- `schemas.py`: typed request/response contracts.
- `feature_engine.py`: converts user inputs into model-ready features.
- `geocoding.py`: OneMap geocoding + local/town fallback.
- `model_service.py`: loads saved model artifact if available; otherwise falls back to comparable median.
- `model_service.py`: loads threshold-specific artifacts (`good75/80/85/90`) and calibrates intervals from validation error quantiles.
- `insight_service.py`: injects verified hedonic and RDD context into outputs.
- `rag_service.py`: local retrieval layer over project knowledge notes for grounded policy Q&A in chat.
- `policy_llm_service.py`: optional provider-backed answer generation (OpenAI-compatible API or local Ollama).
- `llm_service.py`: deterministic grounded explanation layer (provider-agnostic).
- `train_baseline.py`: optional baseline trainer that saves a model artifact.
- `web/index.html`: non-technical form interface with popup results.
- `knowledge/report_findings.md`: retrieval corpus used by chat RAG.

## API contract (high level)

### `POST /predict`

Input includes:
- location (`latitude` + `longitude`, or `address`, or `town`)
- flat attributes (`flat_type`, `floor_area_sqm`, lease info, storey category)
- valuation period (`valuation_year`, `valuation_quarter`)
- school threshold (`75/80/85/90`)

Output includes:
- predicted nominal resale price and `P10/P50/P90` band
- predicted real price (inflation-adjusted) and corresponding interval
- computed feature snapshot (distances, school proximity, counts)
- comparables used
- hedonic evidence + RDD evidence
- plain-language insights

### `POST /predict/friendly`

Input includes simple fields for policy users:
- `town`, `street_name`, `block`
- `number_of_rooms`, `floor_area_sqm`, `lease_commence_year`
- `storey_number` (or `storey_range` like `10 TO 12`)
- valuation period and school threshold

Backend behavior:
- Normalizes text format (e.g., uppercase town/street/block)
- Accepts shorthand street input (e.g., `AVE 8`) and expands/matches town-specific full street names
- Uses strict local matching (`town + street + block`) first to prevent cross-town address ambiguity
- If no exact street match is found but `town + block` exists in local data, keeps resolution within that town
- Uses OneMap API as fallback, with town-aware result scoring
- Falls back to local historical matching if geocoding is unavailable
- Derives internal storey band from floor input:
  - `1-6` => `LOW_IN_ESTATE`
  - `7-12` => `MID_IN_ESTATE`
  - `13+` => `HIGH_IN_ESTATE`

Threshold note:
- Train one artifact per threshold to enable threshold-specific predictions:
  - `artifacts/resale_price_model_good75.joblib`
  - `artifacts/resale_price_model_good80.joblib`
  - `artifacts/resale_price_model_good85.joblib`
  - `artifacts/resale_price_model_good90.joblib`
- If a requested threshold artifact is unavailable, the service falls back to the nearest available artifact and reports this in diagnostics.
- You can force a hidden backend threshold with:
  - `ALLOW_USER_THRESHOLD_SELECTION=0`
  - `FORCED_GOOD_SCHOOL_THRESHOLD=80`

Price-scale note:
- Model target is `log_real_price` (inflation-adjusted).
- API returns both:
  - `predicted_real_price_sgd` (real-price scale)
  - `predicted_resale_price_sgd` (nominal scale using valuation period index)
- Runs the same prediction and evidence pipeline as `/predict`

### `POST /llm/explain`

Input:
- a user question
- full `predict` response payload

Output:
- grounded narrative answer
- evidence points
- safety note

### `POST /chat`

Input:
- `session_id` (optional; returned by API and reused by frontend)
- `message` (free-text user query)

Behavior:
- Detects intent (`analytics`, `prediction`, `general`)
- For analytics queries, computes historical stats (mean/median/count) from transaction data
- Adds local RAG retrieval for report-grounded policy context with a plain-language "Therefore" implication
- If `CHAT_LLM_PROVIDER=openai` and `OPENAI_API_KEY` is set, composes responses via OpenAI-compatible endpoints
- If `CHAT_LLM_PROVIDER=ollama`, composes responses with a local pretrained model over `http://127.0.0.1:11434`
- For prediction queries, performs slot-filling in chat and asks follow-up questions for missing fields
- Once required slots are complete, calls the same friendly prediction pipeline as `POST /predict/friendly`
- Uses explicit missing-data governance for policy safety:
  - Requires explicit confirmation before running (`CONFIRM`) and shows final inputs used
  - Accepts `unknown floor area` and `unknown lease commencement year`
  - Applies transparent imputations (reported back in `assumptions_used`)
  - Assigns confidence tiers (`HIGH` / `MEDIUM` / `LOW`) and widens intervals when imputations are used
  - Abstains when data quality is too weak and asks for minimum critical inputs
  - Abstains on address conflicts (for example, block exists in town but not on the provided street)
  - RAG retrieval is stateless per query (no silent carry-over of old questions into retrieval)

Output:
- assistant reply text
- whether more follow-up is needed
- missing fields (if any)
- collected slots
- assumptions used (if any)
- confidence tier
- optional full prediction payload once estimate is run

## Run locally

1. Install dependencies:

```powershell
pip install -r service/requirements.txt
```

2. Optional: train and save a baseline model artifact:

```powershell
python -m service.train_baseline --all-thresholds
```

Single-threshold training is also supported:

```powershell
python -m service.train_baseline --threshold 80
```

3. Start the API:

```powershell
python -m uvicorn --env-file .env service.main:app --reload
```

Optional LLM-backed chat generation:
Run this in terminal
$ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not (($userPath -split ';') -contains $ollamaPath)) {
  [Environment]::SetEnvironmentVariable("Path", "$userPath;$ollamaPath", "User")
}


```powershell
# Option A: OpenAI-compatible API (OpenAI/xAI-style endpoints)
$env:CHAT_LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_CHAT_MODEL="gpt-5-mini"
# Optional:
# $env:OPENAI_BASE_URL="https://api.openai.com/v1"

# Option B: Local pretrained model with Ollama (no API credits required)
# 1) Install Ollama and run: ollama serve
# 2) Pull a model once: ollama pull qwen2.5:7b-instruct
$env:CHAT_LLM_PROVIDER="ollama"
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
$env:OLLAMA_MODEL="qwen2.5:7b-instruct"

# Optional: directly ingest your report document for RAG
# $env:REPORT_DOCX_PATH="C:\Users\Keagan\Downloads\Report.docx"
# Or use a repo-local file for team portability:
# $env:REPORT_DOCX_PATH="service/knowledge/Report_latest.docx"
# Optional: enable auto-discovery of latest Report*.docx (disabled by default)
# $env:REPORT_DOCX_AUTO_DISCOVER="1"
```


4. Open Swagger UI:
- `http://127.0.0.1:8000/docs`

5. Open dashboard:
- `http://127.0.0.1:8000/`

6. Open form-first prediction UI:
- `http://127.0.0.1:8000/predict-ui`

7. Open chat-first UI:
- `http://127.0.0.1:8000/chat-ui`

8. Test `POST /predict` with:
- `service/sample_predict_request.json`

9. Verify LLM connectivity:
- `http://127.0.0.1:8000/llm/status?check=true`
  - If OpenAI credits are exhausted, this endpoint will explicitly show quota-related failure.
  - If using Ollama, it will confirm local-model reachability.

## Notes for your team

- This starter is intentionally conservative:
  - If no model artifact exists, it still returns usable outputs via comparables.
  - If no external LLM is configured, chat still works using local grounded fallback logic.
- For non-technical officers, pair this API with a simple Streamlit or React form.
- External LLM output is for explanation only; valuation is always computed by the trained artifact pipeline.

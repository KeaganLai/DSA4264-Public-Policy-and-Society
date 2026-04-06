from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "hdb_nearest_sch.csv"
RDD_SUMMARY_PATH = ROOT_DIR / "outputs" / "rdd_improved" / "rdd_schoolfe_summary.md"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "resale_price_model.joblib"
MODEL_META_PATH = ARTIFACTS_DIR / "resale_price_model_meta.json"

DEFAULT_GOOD_SCHOOL_THRESHOLD = 80
DEFAULT_COMPARABLES = 12
MAX_COMPARABLES = 30
VALID_THRESHOLDS = (75, 80, 85, 90)

# Toggle to support either user-selectable thresholds or hidden fixed threshold.
ALLOW_USER_THRESHOLD_SELECTION = os.getenv("ALLOW_USER_THRESHOLD_SELECTION", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
FORCED_GOOD_SCHOOL_THRESHOLD = int(
    os.getenv("FORCED_GOOD_SCHOOL_THRESHOLD", str(DEFAULT_GOOD_SCHOOL_THRESHOLD))
)

# Optional provider-backed policy chat generation.
# Keep "local" to use deterministic local composition.
CHAT_LLM_PROVIDER = os.getenv("CHAT_LLM_PROVIDER", "local").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini").strip()
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
# Backend-fixed conclusion label for policy phrasing.
POLICY_CONCLUSION_WORD = "Therefore"

# Local LLM settings (for zero-credit local inference via Ollama).
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct").strip()
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
OLLAMA_PROBE_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_PROBE_TIMEOUT_SECONDS", "8"))

# Optional: direct Report.docx ingestion path for RAG corpus augmentation.
REPORT_DOCX_PATH = os.getenv("REPORT_DOCX_PATH", "").strip()
REPORT_DOCX_AUTO_DISCOVER = os.getenv("REPORT_DOCX_AUTO_DISCOVER", "0").strip().lower() not in {
    "0",
    "false",
    "no",
}


def model_path_for_threshold(threshold: int) -> Path:
    return ARTIFACTS_DIR / f"resale_price_model_good{threshold}.joblib"


def model_meta_path_for_threshold(threshold: int) -> Path:
    return ARTIFACTS_DIR / f"resale_price_model_meta_good{threshold}.json"

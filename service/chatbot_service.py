from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from difflib import get_close_matches
import re
from typing import Callable
from uuid import uuid4

import pandas as pd

from .config import VALID_THRESHOLDS
from .policy_llm_service import PolicyLLMService
from .rag_service import RagService
from .schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    FriendlyPredictRequest,
    FriendlyPredictResponse,
)


_MONTH_TO_QUARTER = {
    "JAN": 1,
    "FEB": 1,
    "MAR": 1,
    "APR": 2,
    "MAY": 2,
    "JUN": 2,
    "JUL": 3,
    "AUG": 3,
    "SEP": 3,
    "OCT": 4,
    "NOV": 4,
    "DEC": 4,
}

_PREDICTION_REQUIRED = (
    "town",
    "street_name",
    "block",
    "number_of_rooms",
    "floor_area_sqm",
    "lease_commence_year",
)

_PREDICTION_FIELD_LABELS = {
    "town": "the town (for example, ANG MO KIO)",
    "street_name": "the street (for example, AVE 8 or ANG MO KIO AVE 8)",
    "block": "the block number",
    "number_of_rooms": "the number of rooms",
    "floor_area_sqm": "the floor area in sqm",
    "lease_commence_year": "the lease commencement year",
    "storey_input": "the floor number (or a storey range like 10 TO 12)",
}

_UNKNOWN_RE = re.compile(r"\b(DON'T KNOW|DONT KNOW|DO NOT KNOW|UNKNOWN|UNSURE|NOT SURE|NO IDEA)\b")
_CRITICAL_FIELDS = ("floor_area_sqm", "lease_commence_year")
_STREET_SUFFIX_PATTERN = (
    r"(?:AVE|AVENUE|ST|STREET|RD|ROAD|DR|DRIVE|CRES|CRESCENT|LOR|LORONG|JLN|WAY|LINK|TERRACE|TER|PLACE|PL)"
)
_STREET_NOISE_TOKENS = {
    "PLEASE",
    "PREDICT",
    "ESTIMATE",
    "PRICE",
    "VALUE",
    "IT",
    "IS",
    "ITS",
    "IT'S",
    "ROOM",
    "ROOMS",
    "FLAT",
    "RESALE",
    "AREA",
    "BLOCK",
    "BLK",
    "LEASE",
    "COMMENCE",
    "COMMENCEMENT",
    "YEAR",
    "FLOOR",
    "STOREY",
}


def _current_year_quarter() -> tuple[int, int]:
    now = datetime.utcnow()
    quarter = ((now.month - 1) // 3) + 1
    return now.year, quarter


def _canonical_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().upper())


def _to_title_words(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split())


@dataclass
class _ChatSession:
    intent: str | None = None
    slots: dict[str, str | int | float | bool | None] = field(default_factory=dict)
    expected_field: str | None = None
    awaiting_confirmation: bool = False


@dataclass
class _ImputationOutcome:
    field_name: str
    value: float | int
    method_code: str
    sample_size: int
    quality_score: float
    assumption_text: str


@dataclass
class _PreparedPrediction:
    request: FriendlyPredictRequest
    assumptions: list[str]
    confidence_tier: str
    interval_multiplier: float
    final_values: dict[str, str | int | float | bool | None]
    unknown_critical_count: int


class ChatbotService:
    """
    Rule-based chatbot for policy-user workflows with missing-data governance:
    - explicit unknown handling
    - transparent imputation rules
    - explicit confirmation before prediction
    """

    def __init__(
        self,
        df: pd.DataFrame,
        predict_runner: Callable[[FriendlyPredictRequest], FriendlyPredictResponse],
        allow_user_threshold_selection: bool,
        forced_good_school_threshold: int,
        rag_service: RagService | None = None,
        policy_llm_service: PolicyLLMService | None = None,
    ) -> None:
        self.predict_runner = predict_runner
        self.allow_user_threshold_selection = allow_user_threshold_selection
        self.forced_good_school_threshold = int(forced_good_school_threshold)
        self.rag_service = rag_service
        self.policy_llm_service = policy_llm_service
        self.sessions: dict[str, _ChatSession] = {}

        self.analytics_df = df[["town", "year", "quarter", "resale_price"]].copy()
        self.analytics_df["town_norm"] = self.analytics_df["town"].astype(str).map(_canonical_spaces)
        self.analytics_df["year"] = self.analytics_df["year"].astype(int)
        self.analytics_df["quarter"] = self.analytics_df["quarter"].astype(int)
        self.analytics_df["resale_price"] = self.analytics_df["resale_price"].astype(float)

        self.reference_df = df.copy()
        self.reference_df["town_norm"] = self.reference_df["town"].astype(str).map(_canonical_spaces)
        self.reference_df["street_norm"] = self.reference_df["street_name"].astype(str).map(_canonical_spaces)
        self.reference_df["block_norm"] = self.reference_df["block"].astype(str).map(_canonical_spaces)
        self.reference_df["flat_type"] = pd.to_numeric(self.reference_df["flat_type"], errors="coerce")
        self.reference_df["floor_area_sqm"] = pd.to_numeric(self.reference_df["floor_area_sqm"], errors="coerce")
        self.reference_df["lease_commence_date"] = pd.to_numeric(
            self.reference_df["lease_commence_date"],
            errors="coerce",
        )

        towns = sorted(self.analytics_df["town_norm"].dropna().unique().tolist(), key=len, reverse=True)
        self.towns = [t for t in towns if t]
        self.town_aliases = {
            "KALLANG": "KALLANG/WHAMPOA",
            "WHAMPOA": "KALLANG/WHAMPOA",
        }

    def handle_message(self, req: ChatMessageRequest) -> ChatMessageResponse:
        session_id = req.session_id or str(uuid4())
        state = self.sessions.get(session_id, _ChatSession())

        message = req.message.strip()
        message_up = _canonical_spaces(message)
        normalized_for_intent = message_up
        if self.rag_service is not None:
            normalized_for_intent = _canonical_spaces(self.rag_service.normalize_query(message))

        if self._is_reset_command(message_up):
            state = _ChatSession()
            self.sessions[session_id] = state
            return ChatMessageResponse(
                session_id=session_id,
                intent="general",
                reply=(
                    "Session cleared. You can ask for analytics like "
                    "'average resale price in Tampines in 2019' or start a prediction request."
                ),
                requires_follow_up=False,
                missing_fields=[],
                collected_slots={},
                assumptions_used=[],
                confidence_tier=None,
                prediction=None,
            )

        # Hard guard: report/insight questions should never be consumed as prediction slots.
        # This also allows policy users to ask side questions while in a prediction flow.
        if self._is_report_question(normalized_for_intent) and not self._has_strong_prediction_cues(message_up):
            state.intent = "general"
            response = self._handle_general(session_id, state, message)
            self.sessions[session_id] = state
            return response

        extracted = self._extract_slots(message_up, state, normalized_for_intent=normalized_for_intent)
        if self._starts_new_prediction_request(message_up, state, extracted):
            # Prevent silent carry-over across separate prediction requests.
            state = _ChatSession(intent="prediction")
            extracted = self._extract_slots(message_up, state, normalized_for_intent=normalized_for_intent)

        changed = self._apply_extracted_slots(state, extracted)
        if changed:
            state.awaiting_confirmation = False

        intent = self._detect_intent(message_up, state, normalized_for_intent=normalized_for_intent)
        state.intent = intent

        if intent == "analytics":
            response = self._handle_analytics(session_id, state, message_up, message)
        elif intent == "prediction":
            response = self._handle_prediction(session_id, state, message_up)
        else:
            response = self._handle_general(session_id, state, message)

        self.sessions[session_id] = state
        return response

    @staticmethod
    def _is_reset_command(message_up: str) -> bool:
        return bool(re.search(r"\b(RESET|START OVER|CLEAR SESSION|CLEAR)\b", message_up))

    def _starts_new_prediction_request(
        self,
        message_up: str,
        state: _ChatSession,
        extracted: dict[str, str | int | float | bool | None],
    ) -> bool:
        if not state.slots:
            return False

        id_fields = ("town", "street_name", "block", "number_of_rooms")
        provided_id_fields = [k for k in id_fields if extracted.get(k) not in (None, "")]
        if not provided_id_fields:
            return False

        start_phrase = bool(
            re.search(
                r"\b(PREDICT|ESTIMATE|VALUATION|VALUE|PRICE|COST|WORTH|ANOTHER|NEW REQUEST|NEW FLAT)\b",
                message_up,
            )
        )
        has_new_address_triplet = all(extracted.get(k) not in (None, "") for k in ("town", "street_name", "block"))

        conflicting_identity = False
        for key in id_fields:
            prev = state.slots.get(key)
            curr = extracted.get(key)
            if prev in (None, "") or curr in (None, ""):
                continue
            if _canonical_spaces(str(prev)) != _canonical_spaces(str(curr)):
                conflicting_identity = True
                break

        if state.awaiting_confirmation and (start_phrase or has_new_address_triplet):
            return True
        if conflicting_identity and has_new_address_triplet:
            return True
        if start_phrase and (conflicting_identity or has_new_address_triplet):
            return True
        return False

    def _detect_intent(
        self,
        message_up: str,
        state: _ChatSession,
        normalized_for_intent: str | None = None,
    ) -> str:
        report_text = normalized_for_intent or message_up
        if self._is_report_question(report_text):
            return "general"

        if state.intent == "prediction":
            if re.search(r"\b(AVERAGE|MEAN|MEDIAN|COUNT|HOW MANY)\b", message_up):
                return "analytics"
            if re.search(r"\b(POLICY|INSIGHT|FINDING|DRIVER|WHY|METHODOLOGY|REPORT)\b", report_text):
                return "general"
            return "prediction"

        if re.search(r"\b(AVERAGE|MEAN|MEDIAN|COUNT|HOW MANY)\b", message_up) and re.search(
            r"\b(RESALE|PRICE|FLAT|TRANSACTION)\b",
            message_up,
        ):
            return "analytics"

        if re.search(
            r"\b(PREDICT|ESTIMATE|VALUATION|HOW MUCH|COST|WORTH|BUY|BUYING|RESALE)\b",
            message_up,
        ):
            return "prediction"

        if re.search(
            r"\b(TOWN|STREET(?: NAME)?|BLOCK|BLK|NUMBER OF ROOMS?|FLOOR AREA|LEASE(?: COMMENCEMENT)?)\b\s*[:=]",
            message_up,
        ):
            return "prediction"

        if re.search(r"\b(BLK|BLOCK)\b", message_up) and re.search(
            r"\b(AVE|AVENUE|RD|ROAD|STREET|ST|ROOM|FLOOR AREA|LEASE)\b",
            message_up,
        ):
            return "prediction"

        return "general"

    @staticmethod
    def _is_report_question(message_up: str) -> bool:
        return bool(
            re.search(
                r"\b(PREMIUM|HEDONIC|RDD|SDI|GSI|GEP|SAP|TAKEAWAY|MAIN FINDING|FINDINGS|REPORT|METHODOLOGY|ROBUSTNESS|THRESHOLD|EFFECT)\b",
                message_up,
            )
        )

    def _extract_slots(
        self,
        message_up: str,
        state: _ChatSession,
        normalized_for_intent: str | None = None,
    ) -> dict[str, str | int | float | bool | None]:
        report_text = normalized_for_intent or message_up
        if self._is_report_question(report_text) and not self._has_strong_prediction_cues(message_up):
            return {}

        slots: dict[str, str | int | float | bool | None] = {}

        town = self._extract_town(message_up)
        if town:
            slots["town"] = town

        block = self._extract_block(message_up)
        if block:
            slots["block"] = block

        street = self._extract_street(message_up)
        if street:
            slots["street_name"] = street

        rooms = self._extract_number_of_rooms(message_up)
        if rooms is not None:
            slots["number_of_rooms"] = rooms

        area_unknown, lease_unknown = self._extract_unknown_markers(message_up, state.expected_field)
        if area_unknown:
            slots["floor_area_unknown"] = True
            slots["floor_area_sqm"] = None
        if lease_unknown:
            slots["lease_commence_year_unknown"] = True
            slots["lease_commence_year"] = None

        area = self._extract_floor_area(message_up)
        if area is not None:
            slots["floor_area_sqm"] = area
            slots["floor_area_unknown"] = False

        lease_year = self._extract_lease_commence_year(message_up)
        if lease_year is not None:
            slots["lease_commence_year"] = lease_year
            slots["lease_commence_year_unknown"] = False

        storey_number = self._extract_storey_number(message_up)
        if storey_number is not None:
            slots["storey_number"] = storey_number

        storey_range = self._extract_storey_range(message_up)
        if storey_range:
            slots["storey_range"] = storey_range

        quarter = self._extract_quarter(message_up)
        if quarter is not None:
            slots["valuation_quarter"] = quarter

        valuation_year = self._extract_valuation_year(message_up, lease_year)
        if valuation_year is not None:
            slots["valuation_year"] = valuation_year

        threshold = self._extract_threshold(message_up)
        if threshold is not None:
            slots["good_school_threshold"] = threshold

        expected = state.expected_field
        if expected and not self._is_report_question(message_up):
            self._fill_expected_field(expected, message_up, slots)

        missing = self._missing_prediction_fields(state, additional_slots=slots)
        if len(missing) == 1:
            one_missing = missing[0]
            raw = message_up.strip()
            if one_missing == "street_name" and raw and self._looks_like_street_candidate(raw):
                slots.setdefault("street_name", raw)
            if one_missing == "town" and raw and self._looks_like_town_candidate(raw):
                resolved_town = self._extract_town(raw) or raw
                slots.setdefault("town", resolved_town)
            if one_missing == "block":
                m = re.fullmatch(r"\d{1,4}[A-Z]?", raw)
                if m:
                    slots.setdefault("block", m.group(0))
            if one_missing == "number_of_rooms":
                m = re.fullmatch(r"[1-7]", raw)
                if m:
                    slots.setdefault("number_of_rooms", int(m.group(0)))
            if one_missing == "floor_area_sqm":
                m = re.fullmatch(r"\d{2,3}(?:\.\d+)?", raw)
                if m:
                    slots.setdefault("floor_area_sqm", float(m.group(0)))
                    slots.setdefault("floor_area_unknown", False)
            if one_missing == "lease_commence_year":
                m = re.fullmatch(r"(19\d{2}|20\d{2})", raw)
                if m:
                    slots.setdefault("lease_commence_year", int(m.group(0)))
                    slots.setdefault("lease_commence_year_unknown", False)

        return slots

    @staticmethod
    def _has_strong_prediction_cues(message_up: str) -> bool:
        return bool(
            re.search(
                r"\b(PREDICT|ESTIMATE|VALUATION|TOWN|STREET(?: NAME)?|BLOCK|BLK|NUMBER OF ROOMS?|ROOM|FLOOR AREA|LEASE(?: COMMENCEMENT)?|STOREY|LEVEL|SQM)\b",
                message_up,
            )
        )

    def _apply_extracted_slots(self, state: _ChatSession, extracted: dict[str, str | int | float | bool | None]) -> bool:
        changed = False
        for key, value in extracted.items():
            if key in {"floor_area_unknown", "lease_commence_year_unknown"}:
                current = state.slots.get(key)
                if current != value:
                    state.slots[key] = value
                    changed = True
                continue

            if key in _CRITICAL_FIELDS:
                current = state.slots.get(key)
                if current != value:
                    state.slots[key] = value
                    changed = True
                continue

            if value is None:
                continue
            current = state.slots.get(key)
            if current != value:
                state.slots[key] = value
                changed = True
        return changed

    def _fill_expected_field(
        self,
        expected: str,
        message_up: str,
        slots: dict[str, str | int | float | bool | None],
    ) -> None:
        raw = message_up.strip()
        if expected in slots and slots.get(expected) is not None:
            return
        if not raw:
            return

        if expected == "town" and self._looks_like_town_candidate(raw):
            slots["town"] = self._extract_town(raw) or raw
            return
        if expected == "street_name" and self._looks_like_street_candidate(raw):
            slots["street_name"] = raw
            return
        if expected == "block":
            m = re.fullmatch(r"\d{1,4}[A-Z]?", raw)
            if m:
                slots["block"] = m.group(0)
            return
        if expected == "number_of_rooms":
            m = re.fullmatch(r"[1-7]", raw)
            if m:
                slots["number_of_rooms"] = int(m.group(0))
            return
        if expected == "floor_area_sqm":
            if _UNKNOWN_RE.search(raw):
                slots["floor_area_unknown"] = True
                slots["floor_area_sqm"] = None
                return
            m = re.fullmatch(r"\d{2,3}(?:\.\d+)?", raw)
            if m:
                slots["floor_area_sqm"] = float(m.group(0))
                slots["floor_area_unknown"] = False
            return
        if expected == "lease_commence_year":
            if _UNKNOWN_RE.search(raw):
                slots["lease_commence_year_unknown"] = True
                slots["lease_commence_year"] = None
                return
            m = re.fullmatch(r"(19\d{2}|20\d{2})", raw)
            if m:
                slots["lease_commence_year"] = int(m.group(0))
                slots["lease_commence_year_unknown"] = False
            return
        if expected == "storey_input":
            m_floor = re.fullmatch(r"\d{1,2}", raw)
            if m_floor:
                slots["storey_number"] = int(m_floor.group(0))
                return
            m_range = re.fullmatch(r"\d{1,2}\s*(?:TO|-)\s*\d{1,2}", raw)
            if m_range:
                slots["storey_range"] = re.sub(r"\s*-\s*", " TO ", m_range.group(0))
            return

    def _looks_like_town_candidate(self, raw: str) -> bool:
        value = _canonical_spaces(raw)
        if self._extract_town(value):
            return True
        if self._is_report_question(value):
            return False
        if re.search(r"\b(WHAT|HOW|WHY|PLEASE|EXPLAIN|CAN YOU|TELL ME)\b", value):
            return False
        if not re.fullmatch(r"[A-Z /'-]{2,40}", value):
            return False
        words = [w for w in value.split() if w]
        return 1 <= len(words) <= 4

    def _looks_like_street_candidate(self, raw: str) -> bool:
        value = _canonical_spaces(raw)
        if self._is_report_question(value):
            return False
        if re.search(r"\b(WHAT|HOW|WHY|PLEASE|EXPLAIN|CAN YOU|TELL ME)\b", value):
            return False
        if not re.fullmatch(r"[A-Z0-9 /'-]{2,60}", value):
            return False
        has_street_token = bool(
            re.search(
                r"\b(AVE|AVENUE|ST|STREET|RD|ROAD|DR|DRIVE|CRES|CRESCENT|LOR|LORONG|JLN|WAY|LINK|TER|TERRACE|PL|PLACE)\b",
                value,
            )
        )
        return has_street_token

    @staticmethod
    def _extract_unknown_markers(message_up: str, expected_field: str | None) -> tuple[bool, bool]:
        if not _UNKNOWN_RE.search(message_up):
            return False, False

        mentions_area = bool(re.search(r"\b(FLOOR AREA|SQM|M2|SIZE)\b", message_up))
        mentions_lease = bool(re.search(r"\b(LEASE|COMMENCE|COMMENCEMENT)\b", message_up))

        area_unknown = mentions_area or expected_field == "floor_area_sqm"
        lease_unknown = mentions_lease or expected_field == "lease_commence_year"
        return area_unknown, lease_unknown

    def _extract_town(self, message_up: str) -> str | None:
        alias_tokens = re.findall(r"\b[A-Z][A-Z/]+\b", message_up)
        for token in alias_tokens:
            if token in self.town_aliases:
                return self.town_aliases[token]

        for town in self.towns:
            if re.search(rf"\b{re.escape(town)}\b", message_up):
                return town
        return None

    @staticmethod
    def _extract_block(message_up: str) -> str | None:
        m = re.search(r"\b(?:BLK|BLOCK)\s*([0-9]{1,4}[A-Z]?)\b", message_up)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _extract_street(message_up: str) -> str | None:
        explicit = re.search(
            r"\bSTREET(?: NAME)?\s*[:=]\s*([A-Z0-9 /'-]{2,50})",
            message_up,
        )
        if explicit:
            cleaned = ChatbotService._clean_street_candidate(explicit.group(1).strip())
            if cleaned:
                return cleaned

        preposition = re.search(
            rf"\b(?:ALONG|AT|ON|IN|NEAR)\s+([A-Z0-9 /'-]{{2,56}}\s{_STREET_SUFFIX_PATTERN}(?:\s+\d+[A-Z]?)?)\b",
            message_up,
        )
        if preposition:
            cleaned = ChatbotService._clean_street_candidate(preposition.group(1).strip())
            if cleaned:
                return cleaned

        shorthand = re.search(
            r"\b((?:AVE|AVENUE|ST|STREET|RD|ROAD|DR|DRIVE|CRES|CRESCENT|LOR|LORONG|JLN|WAY|LINK|TER|PL|PLACE)\s*\d+[A-Z]?)\b",
            message_up,
        )
        if shorthand:
            cleaned = ChatbotService._clean_street_candidate(shorthand.group(1).strip())
            if cleaned:
                return cleaned

        candidates = re.findall(
            rf"\b([A-Z0-9]+(?:\s+[A-Z0-9]+){{0,6}}\s{_STREET_SUFFIX_PATTERN}(?:\s+\d+[A-Z]?)?)\b",
            message_up,
        )
        if candidates:
            best_score = float("-inf")
            best = None
            bad_tokens = {"IT", "IS", "A", "AN", "ROOM", "FLAT", "PLEASE", "PREDICT", "PRICE", "ESTIMATE"}
            for cand in candidates:
                cleaned = ChatbotService._clean_street_candidate(cand)
                if not cleaned:
                    continue
                tokens = [t for t in cleaned.split() if t]
                score = float(len(tokens))
                if any(t in bad_tokens for t in tokens):
                    score -= 4.0
                if re.search(r"\b(?:ALONG|AT|ON|IN|NEAR)\b", cleaned):
                    score -= 2.0
                if score > best_score:
                    best_score = score
                    best = cleaned
            if best:
                return best
        return None

    @staticmethod
    def _clean_street_candidate(raw: str) -> str:
        text = _canonical_spaces(raw)
        text = re.sub(r"^\s*(?:IT IS|ITS|IT'S|A|AN)\s+", "", text)
        text = re.sub(r"\b\d+\s*ROOM\b", "", text)
        text = re.sub(r"\bROOM\b", "", text)
        text = re.sub(r"\bFLAT\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Prefer text after prepositions used in natural speech.
        parts = re.split(r"\b(?:ALONG|AT|ON|IN|NEAR)\b", text)
        if len(parts) > 1:
            text = parts[-1].strip()

        # Remove trailing non-street details.
        text = re.split(
            r"\b(?:BLOCK|BLK|FLOOR|STOREY|LEASE|COMMENCE|COMMENCEMENT|YEAR|QUARTER|VALUATION|ROOMS?)\b",
            text,
            maxsplit=1,
        )[0].strip()

        m = re.search(
            rf"([A-Z0-9]+(?:\s+[A-Z0-9]+){{0,6}}\s{_STREET_SUFFIX_PATTERN}(?:\s+\d+[A-Z]?)?)$",
            text,
        )
        if m:
            return m.group(1).strip()
        if re.search(rf"\b{_STREET_SUFFIX_PATTERN}\b", text):
            return text
        return ""

    @staticmethod
    def _extract_number_of_rooms(message_up: str) -> int | None:
        m = re.search(r"\b([1-7])\s*[- ]?ROOM\b", message_up)
        if not m:
            m = re.search(r"\bNUMBER OF ROOMS?\D*([1-7])\b", message_up)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _extract_floor_area(message_up: str) -> float | None:
        # If user explicitly says floor area is unknown, do not infer from unrelated numbers.
        if re.search(r"\b(?:UNKNOWN|DON'T KNOW|DONT KNOW|DO NOT KNOW)\b[^.]{0,24}\bFLOOR AREA\b", message_up) or re.search(
            r"\bFLOOR AREA\b[^.]{0,24}\b(?:UNKNOWN|DON'T KNOW|DONT KNOW|DO NOT KNOW)\b",
            message_up,
        ):
            return None

        # Prefer explicit unit-bearing forms.
        m = re.search(r"\b(\d{2,3}(?:\.\d+)?)\s*(?:SQM|SQ M|M2)\b", message_up)
        if not m:
            # Handles phrasing like "floor area in sqm is 123".
            m = re.search(
                r"\bFLOOR AREA\b[^.\n]{0,28}\b(?:SQM|SQ M|M2)\b[^0-9]{0,12}(\d{2,3}(?:\.\d+)?)\b",
                message_up,
            )
        if not m:
            # Only accept values directly attached to floor-area phrase (bounded context).
            m = re.search(
                r"\bFLOOR AREA(?:\s*(?:IS|=|:|ABOUT|AROUND))?\s*(\d{2,3}(?:\.\d+)?)\b(?:\s*(?:SQM|SQ M|M2))?",
                message_up,
            )
        if m:
            return float(m.group(1))
        return None

    @staticmethod
    def _extract_lease_commence_year(message_up: str) -> int | None:
        if re.search(
            r"\b(?:UNKNOWN|DON'T KNOW|DONT KNOW|DO NOT KNOW)\b[^.]{0,30}\bLEASE(?: COMMENCEMENT)?(?: YEAR)?\b",
            message_up,
        ) or re.search(
            r"\bLEASE(?: COMMENCEMENT)?(?: YEAR)?\b[^.]{0,30}\b(?:UNKNOWN|DON'T KNOW|DONT KNOW|DO NOT KNOW)\b",
            message_up,
        ):
            return None

        m = re.search(
            r"\bLEASE(?: COMMENCEMENT)?(?: YEAR)?(?:\s*(?:IS|=|:|ABOUT|AROUND))?\s*(19\d{2}|20\d{2})\b",
            message_up,
        )
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _extract_storey_number(message_up: str) -> int | None:
        m = re.search(r"\b(?:FLOOR|LEVEL|STOREY)\s*(?:NUMBER|NO|#)?\s*(?:IS|=|:)?\s*(\d{1,2})\b", message_up)
        if not m:
            m = re.search(r"\b(\d{1,2})\s*(?:TH\s+)?(?:FLOOR|LEVEL|STOREY)\b", message_up)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _extract_storey_range(message_up: str) -> str | None:
        m = re.search(r"\b(\d{1,2}\s*(?:TO|-)\s*\d{1,2})\b", message_up)
        if m:
            return re.sub(r"\s*-\s*", " TO ", m.group(1).strip())
        return None

    @staticmethod
    def _extract_quarter(message_up: str) -> int | None:
        m = re.search(r"\bQ([1-4])\b", message_up)
        if m:
            return int(m.group(1))

        for month_abbrev, quarter in _MONTH_TO_QUARTER.items():
            if re.search(rf"\b{month_abbrev}(?:[A-Z]*)\b", message_up):
                return quarter
        return None

    @staticmethod
    def _extract_valuation_year(message_up: str, lease_year: int | None) -> int | None:
        years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", message_up)]
        if not years:
            return None

        if lease_year is not None and lease_year in years:
            years = [y for y in years if y != lease_year]
            if not years:
                return None
        return years[0]

    @staticmethod
    def _extract_threshold(message_up: str) -> int | None:
        m = re.search(r"\b(?:THRESHOLD|GOOD SCHOOL)\D*(75|80|85|90)\b", message_up)
        if m:
            return int(m.group(1))
        return None

    def _missing_prediction_fields(
        self,
        state: _ChatSession,
        additional_slots: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[str]:
        merged = dict(state.slots)
        if additional_slots:
            merged.update(additional_slots)

        missing: list[str] = []
        for field_name in _PREDICTION_REQUIRED:
            value = merged.get(field_name)
            if field_name in _CRITICAL_FIELDS:
                if field_name == "floor_area_sqm":
                    unknown_flag = bool(merged.get("floor_area_unknown", False))
                else:
                    unknown_flag = bool(merged.get("lease_commence_year_unknown", False))
                if value in (None, "") and not unknown_flag:
                    missing.append(field_name)
            elif value in (None, ""):
                missing.append(field_name)

        has_storey = any(
            merged.get(name) not in (None, "")
            for name in ("storey_number", "storey_range", "storey_relative_category")
        )
        if not has_storey:
            missing.append("storey_input")
        return missing

    def _handle_prediction(self, session_id: str, state: _ChatSession, message_up: str) -> ChatMessageResponse:
        year_default, quarter_default = _current_year_quarter()
        state.slots.setdefault("valuation_year", year_default)
        state.slots.setdefault("valuation_quarter", quarter_default)
        state.slots.setdefault("good_school_threshold", self.forced_good_school_threshold)

        missing = self._missing_prediction_fields(state)
        if missing:
            state.expected_field = missing[0]
            state.awaiting_confirmation = False
            reply = self._build_prediction_follow_up(missing, state.slots)
            return ChatMessageResponse(
                session_id=session_id,
                intent="prediction",
                reply=reply,
                requires_follow_up=True,
                missing_fields=missing,
                collected_slots=dict(state.slots),
                assumptions_used=[],
                confidence_tier=None,
                prediction=None,
            )
        state.expected_field = None

        prepared, abstain_reason = self._prepare_prediction(state)
        if prepared is None:
            return ChatMessageResponse(
                session_id=session_id,
                intent="prediction",
                reply=abstain_reason or (
                    "Insufficient data for reliable estimate. Please provide floor area (sqm) or lease commencement year."
                ),
                requires_follow_up=True,
                missing_fields=[],
                collected_slots=dict(state.slots),
                assumptions_used=[],
                confidence_tier=None,
                prediction=None,
            )

        confirm_msg = bool(re.search(r"\b(CONFIRM|YES|PROCEED|RUN)\b", message_up))
        if not state.awaiting_confirmation:
            state.awaiting_confirmation = True
            return ChatMessageResponse(
                session_id=session_id,
                intent="prediction",
                reply=self._confirmation_prompt(prepared),
                requires_follow_up=True,
                missing_fields=[],
                collected_slots=dict(prepared.final_values),
                assumptions_used=prepared.assumptions,
                confidence_tier=prepared.confidence_tier,  # type: ignore[arg-type]
                prediction=None,
            )
        if state.awaiting_confirmation and not confirm_msg:
            return ChatMessageResponse(
                session_id=session_id,
                intent="prediction",
                reply=f"Pending confirmation. {self._confirmation_prompt(prepared)}",
                requires_follow_up=True,
                missing_fields=[],
                collected_slots=dict(prepared.final_values),
                assumptions_used=prepared.assumptions,
                confidence_tier=prepared.confidence_tier,  # type: ignore[arg-type]
                prediction=None,
            )
        state.awaiting_confirmation = False

        try:
            prediction = self.predict_runner(prepared.request)
        except Exception as exc:
            state.awaiting_confirmation = True
            return ChatMessageResponse(
                session_id=session_id,
                intent="prediction",
                reply=(
                    "I could not run the prediction with the current details. "
                    f"Please double-check town/street/block formatting and try again. Technical detail: {exc}"
                ),
                requires_follow_up=True,
                missing_fields=[],
                collected_slots=dict(prepared.final_values),
                assumptions_used=prepared.assumptions,
                confidence_tier=prepared.confidence_tier,  # type: ignore[arg-type]
                prediction=None,
            )

        self._apply_interval_penalty(prediction, prepared.interval_multiplier)
        pred = prediction.prediction
        assumptions_text = (
            "; ".join(prepared.assumptions) if prepared.assumptions else "None (all key inputs provided directly)."
        )

        summary = (
            f"Estimated resale price is SGD {pred.predicted_resale_price_sgd:,.0f} "
            f"(likely range P10 to P90: {pred.prediction_interval_sgd.p10:,.0f} to {pred.prediction_interval_sgd.p90:,.0f}). "
            f"Resolved address: {prediction.resolved_location.matched_address or prediction.resolved_location.search_query}. "
            f"Nearest good school: {pred.feature_snapshot.nearest_school_name} "
            f"at {pred.feature_snapshot.nearest_school_distance_km:.2f} km. "
            f"Confidence tier: {prepared.confidence_tier}. "
            f"Assumed values used: {assumptions_text}. "
            "Therefore: use this as a planning estimate, then cross-check against recent comparables before policy decisions."
        )
        if prepared.interval_multiplier > 1.0:
            summary = f"{summary} Interval widened by {prepared.interval_multiplier:.2f}x due to imputation uncertainty."
        if pred.insights:
            summary = f"{summary} {pred.insights[0]}"

        return ChatMessageResponse(
            session_id=session_id,
            intent="prediction",
            reply=summary,
            requires_follow_up=False,
            missing_fields=[],
            collected_slots=dict(prepared.final_values),
            assumptions_used=prepared.assumptions,
            confidence_tier=prepared.confidence_tier,  # type: ignore[arg-type]
            prediction=prediction,
        )

    def _prepare_prediction(self, state: _ChatSession) -> tuple[_PreparedPrediction | None, str | None]:
        slots = dict(state.slots)
        assumptions: list[str] = []
        quality_scores: list[float] = []
        unknown_critical_count = 0

        street_reason = self._street_plausibility_reason(slots)
        if street_reason:
            return None, street_reason

        address_conflict = self._address_conflict_reason(slots)
        if address_conflict:
            return None, address_conflict

        if slots.get("floor_area_sqm") in (None, "") and bool(slots.get("floor_area_unknown", False)):
            unknown_critical_count += 1
            outcome = self._impute_floor_area(slots)
            if outcome is None:
                return (
                    None,
                    "Insufficient data for reliable estimate. Please provide floor area (sqm), or provide both street and block with a known unit size.",
                )
            slots["floor_area_sqm"] = float(outcome.value)
            assumptions.append(outcome.assumption_text)
            quality_scores.append(outcome.quality_score)

        if slots.get("lease_commence_year") in (None, "") and bool(slots.get("lease_commence_year_unknown", False)):
            unknown_critical_count += 1
            outcome = self._impute_lease_commence_year(slots)
            if outcome is None:
                return (
                    None,
                    "Insufficient data for reliable estimate. Please provide lease commencement year, or provide a better-matched address context.",
                )
            slots["lease_commence_year"] = int(outcome.value)
            assumptions.append(outcome.assumption_text)
            quality_scores.append(outcome.quality_score)

        if unknown_critical_count >= 2 and (not quality_scores or min(quality_scores) < 2.0):
            return (
                None,
                (
                    "Insufficient data for reliable estimate. I can proceed once you provide at least one critical field "
                    "(floor area in sqm or lease commencement year)."
                ),
            )

        threshold = int(slots.get("good_school_threshold", self.forced_good_school_threshold))
        if not self.allow_user_threshold_selection or threshold not in VALID_THRESHOLDS:
            threshold = self.forced_good_school_threshold

        try:
            request = FriendlyPredictRequest(
                town=str(slots["town"]),
                street_name=str(slots["street_name"]),
                block=str(slots["block"]),
                number_of_rooms=int(slots["number_of_rooms"]),
                floor_area_sqm=float(slots["floor_area_sqm"]),
                lease_commence_year=int(slots["lease_commence_year"]),
                storey_number=int(slots["storey_number"]) if slots.get("storey_number") is not None else None,
                storey_range=str(slots["storey_range"]) if slots.get("storey_range") else None,
                valuation_year=int(slots["valuation_year"]),
                valuation_quarter=int(slots["valuation_quarter"]),
                good_school_threshold=threshold,
                include_comparables=True,
                comparables_limit=12,
                include_llm_insights=True,
            )
        except Exception as exc:
            err = str(exc).upper()
            if "FLOOR_AREA_SQM" in err:
                return (
                    None,
                    "Insufficient data for reliable estimate: floor area is missing or invalid. "
                    "Please provide floor area in sqm (for example, 93), or explicitly say 'unknown floor area'.",
                )
            if "LEASE_COMMENCE_YEAR" in err:
                return (
                    None,
                    "Insufficient data for reliable estimate: lease commencement year is missing or invalid. "
                    "Please provide a year like 1985, or explicitly say 'unknown lease commencement year'.",
                )
            return None, "Insufficient data for reliable estimate due to input validation. Please review the entered fields."

        confidence_tier, interval_multiplier = self._confidence_and_multiplier(assumptions, quality_scores)
        final_values = {
            "town": request.town,
            "street_name": request.street_name,
            "block": request.block,
            "number_of_rooms": request.number_of_rooms,
            "floor_area_sqm": request.floor_area_sqm,
            "lease_commence_year": request.lease_commence_year,
            "storey_number": request.storey_number,
            "storey_range": request.storey_range,
            "valuation_year": request.valuation_year,
            "valuation_quarter": request.valuation_quarter,
            "good_school_threshold": request.good_school_threshold,
        }
        prepared = _PreparedPrediction(
            request=request,
            assumptions=assumptions,
            confidence_tier=confidence_tier,
            interval_multiplier=interval_multiplier,
            final_values=final_values,
            unknown_critical_count=unknown_critical_count,
        )
        return prepared, None

    def _address_conflict_reason(self, slots: dict[str, str | int | float | bool | None]) -> str | None:
        town = _canonical_spaces(str(slots.get("town", "")))
        street_input = _canonical_spaces(str(slots.get("street_name", "")))
        block = _canonical_spaces(str(slots.get("block", "")))
        if not town or not street_input or not block:
            return None

        block_rows = self.reference_df.loc[
            (self.reference_df["town_norm"] == town) & (self.reference_df["block_norm"] == block)
        ]
        if block_rows.empty:
            return None

        block_streets = sorted(block_rows["street_norm"].dropna().astype(str).unique().tolist())
        if any(self._street_matches(street_input, known_street, town) for known_street in block_streets):
            return None

        known_display = ", ".join(block_streets[:3])
        return (
            "Insufficient data for reliable estimate due to address conflict: "
            f"in {town}, block {block} is recorded as {known_display}, not '{street_input}'. "
            "Please correct town/street/block or provide a verified full address."
        )

    def _street_plausibility_reason(self, slots: dict[str, str | int | float | bool | None]) -> str | None:
        town = _canonical_spaces(str(slots.get("town", "")))
        street_input = _canonical_spaces(str(slots.get("street_name", "")))
        if not town or not street_input:
            return None

        street_clean = self._clean_street_candidate(street_input)
        if not street_clean:
            return (
                "Insufficient data for reliable estimate: street looks invalid. "
                "Please provide a clear street like 'WHAMPOA RD' or 'AVE 8'."
            )
        if street_clean != street_input:
            # Keep slots canonical and avoid carrying conversational filler words into prediction.
            slots["street_name"] = street_clean
            street_input = street_clean

        tokens = [t for t in street_input.split() if t]
        if len(tokens) > 6:
            return (
                "Insufficient data for reliable estimate: street text is too long and may include non-address words. "
                "Please provide only the street name (for example, WHAMPOA RD)."
            )
        if any(tok in _STREET_NOISE_TOKENS for tok in tokens):
            return (
                "Insufficient data for reliable estimate: street field appears to contain non-address words. "
                "Please provide only the street name."
            )

        town_rows = self.reference_df.loc[self.reference_df["town_norm"] == town]
        if town_rows.empty:
            return None
        known_streets = sorted(town_rows["street_norm"].dropna().astype(str).unique().tolist())
        if any(self._street_matches(street_input, known, town) for known in known_streets):
            return None

        # Unknown streets can exist in future data, so only block when the text also looks suspicious.
        if re.search(r"\b(?:IT|IS|ROOM|FLAT|PLEASE|PREDICT|ESTIMATE)\b", street_input):
            suggestions = get_close_matches(street_input, known_streets, n=3, cutoff=0.72)
            suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            return (
                "Insufficient data for reliable estimate: street name does not look valid for the selected town."
                f"{suggestion_text}"
            )
        return None

    @staticmethod
    def _street_signature(value: str, town_norm: str) -> str:
        v = _canonical_spaces(value)
        replacements = {
            "AVENUE": "AVE",
            "STREET": "ST",
            "ROAD": "RD",
            "DRIVE": "DR",
            "CRESCENT": "CRES",
            "LORONG": "LOR",
            "TERRACE": "TER",
            "PLACE": "PL",
        }
        for src, dst in replacements.items():
            v = re.sub(rf"\b{src}\b", dst, v)
        if town_norm and v.startswith(f"{town_norm} "):
            v = v[len(town_norm) + 1 :]
        v = re.sub(r"\s+", " ", v).strip()
        return v

    def _street_matches(self, user_street: str, known_street: str, town_norm: str) -> bool:
        user_sig = self._street_signature(user_street, town_norm)
        known_sig = self._street_signature(known_street, town_norm)
        if not user_sig or not known_sig:
            return False
        if user_sig == known_sig:
            return True
        if known_sig.endswith(f" {user_sig}") or user_sig.endswith(f" {known_sig}"):
            return True

        user_tokens = [t for t in user_sig.split() if t]
        known_tokens = [t for t in known_sig.split() if t]
        if len(user_tokens) >= 2 and set(user_tokens).issubset(set(known_tokens)):
            return True
        return False

    def _impute_floor_area(self, slots: dict[str, str | int | float | bool | None]) -> _ImputationOutcome | None:
        town = _canonical_spaces(str(slots.get("town", "")))
        street = _canonical_spaces(str(slots.get("street_name", "")))
        block = _canonical_spaces(str(slots.get("block", "")))
        flat_type = int(slots.get("number_of_rooms")) if slots.get("number_of_rooms") is not None else None
        if flat_type is None:
            return None

        candidates = [
            (
                "block_street_flattype",
                (self.reference_df["town_norm"] == town)
                & (self.reference_df["street_norm"] == street)
                & (self.reference_df["block_norm"] == block)
                & (self.reference_df["flat_type"] == flat_type),
                5,
                3.0,
                "same town/street/block and flat type",
            ),
            (
                "street_flattype",
                (self.reference_df["town_norm"] == town)
                & (self.reference_df["street_norm"] == street)
                & (self.reference_df["flat_type"] == flat_type),
                10,
                2.6,
                "same town/street and flat type",
            ),
            (
                "town_flattype",
                (self.reference_df["town_norm"] == town) & (self.reference_df["flat_type"] == flat_type),
                20,
                2.0,
                "same town and flat type",
            ),
            (
                "flattype_global",
                self.reference_df["flat_type"] == flat_type,
                80,
                1.4,
                "same flat type (all towns)",
            ),
        ]
        for code, mask, min_n, score, label in candidates:
            series = self.reference_df.loc[mask, "floor_area_sqm"].dropna()
            n = int(series.shape[0])
            if n < min_n:
                continue
            value = round(float(series.median()), 1)
            return _ImputationOutcome(
                field_name="floor_area_sqm",
                value=value,
                method_code=code,
                sample_size=n,
                quality_score=score,
                assumption_text=f"floor_area_sqm imputed to {value:.1f} using median from {label} (n={n})",
            )
        return None

    def _impute_lease_commence_year(self, slots: dict[str, str | int | float | bool | None]) -> _ImputationOutcome | None:
        town = _canonical_spaces(str(slots.get("town", "")))
        street = _canonical_spaces(str(slots.get("street_name", "")))
        block = _canonical_spaces(str(slots.get("block", "")))
        flat_type = int(slots.get("number_of_rooms")) if slots.get("number_of_rooms") is not None else None
        if flat_type is None:
            return None

        lease = self.reference_df["lease_commence_date"]
        valid_lease_mask = lease.notna() & (lease >= 1960) & (lease <= 2100)

        candidates = [
            (
                "block_street_flattype",
                valid_lease_mask
                & (self.reference_df["town_norm"] == town)
                & (self.reference_df["street_norm"] == street)
                & (self.reference_df["block_norm"] == block)
                & (self.reference_df["flat_type"] == flat_type),
                3,
                3.0,
                "same town/street/block and flat type",
            ),
            (
                "block_street",
                valid_lease_mask
                & (self.reference_df["town_norm"] == town)
                & (self.reference_df["street_norm"] == street)
                & (self.reference_df["block_norm"] == block),
                5,
                2.7,
                "same town/street/block",
            ),
            (
                "town_flattype",
                valid_lease_mask
                & (self.reference_df["town_norm"] == town)
                & (self.reference_df["flat_type"] == flat_type),
                20,
                2.0,
                "same town and flat type",
            ),
            (
                "flattype_global",
                valid_lease_mask & (self.reference_df["flat_type"] == flat_type),
                80,
                1.4,
                "same flat type (all towns)",
            ),
        ]
        for code, mask, min_n, score, label in candidates:
            series = self.reference_df.loc[mask, "lease_commence_date"].dropna()
            n = int(series.shape[0])
            if n < min_n:
                continue
            value = self._mode_or_median_year(series)
            if value is None:
                continue
            return _ImputationOutcome(
                field_name="lease_commence_year",
                value=int(value),
                method_code=code,
                sample_size=n,
                quality_score=score,
                assumption_text=f"lease_commence_year imputed to {int(value)} using mode/median from {label} (n={n})",
            )
        return None

    @staticmethod
    def _mode_or_median_year(series: pd.Series) -> int | None:
        vals = series.dropna().astype(float)
        if vals.empty:
            return None
        mode_vals = vals.mode()
        if mode_vals.shape[0] == 1:
            return int(round(float(mode_vals.iloc[0])))
        return int(round(float(vals.median())))

    @staticmethod
    def _confidence_and_multiplier(assumptions: list[str], quality_scores: list[float]) -> tuple[str, float]:
        if not assumptions:
            return "HIGH", 1.0
        if len(assumptions) == 1:
            if quality_scores and min(quality_scores) < 2.0:
                return "LOW", 1.6
            return "MEDIUM", 1.35
        if quality_scores and min(quality_scores) < 2.0:
            return "LOW", 1.85
        return "LOW", 1.75

    @staticmethod
    def _apply_interval_penalty(prediction: FriendlyPredictResponse, multiplier: float) -> None:
        if multiplier <= 1.0:
            return
        nominal = prediction.prediction.prediction_interval_sgd
        real = prediction.prediction.prediction_interval_real_sgd
        for interval in (nominal, real):
            p50 = float(interval.p50)
            lower_half = max(0.0, p50 - float(interval.p10))
            upper_half = max(0.0, float(interval.p90) - p50)
            interval.p10 = max(0.0, p50 - lower_half * multiplier)
            interval.p90 = p50 + upper_half * multiplier

    def _confirmation_prompt(self, prepared: _PreparedPrediction) -> str:
        assumptions = "; ".join(prepared.assumptions) if prepared.assumptions else "None."
        return (
            f"I have all required fields. Final inputs used: {self._final_input_summary(prepared.final_values)}. "
            f"Assumed values used: {assumptions} "
            f"Confidence tier if run now: {prepared.confidence_tier}. "
            "Please type CONFIRM to run the prediction, or provide corrections."
        )

    @staticmethod
    def _final_input_summary(values: dict[str, str | int | float | bool | None]) -> str:
        storey = values.get("storey_range") or values.get("storey_number") or values.get("storey_relative_category")
        return (
            f"town={values.get('town')}, street={values.get('street_name')}, block={values.get('block')}, "
            f"rooms={values.get('number_of_rooms')}, floor_area_sqm={values.get('floor_area_sqm')}, "
            f"lease_commence_year={values.get('lease_commence_year')}, storey={storey}, "
            f"valuation_year={values.get('valuation_year')}, valuation_quarter=Q{values.get('valuation_quarter')}, "
            f"good_school_threshold={values.get('good_school_threshold')}"
        )

    @staticmethod
    def _build_prediction_follow_up(
        missing: list[str],
        collected: dict[str, str | int | float | bool | None],
    ) -> str:
        first = _PREDICTION_FIELD_LABELS[missing[0]]
        collected_summary = []
        for key in ("town", "street_name", "block", "number_of_rooms", "floor_area_sqm", "lease_commence_year"):
            if collected.get(key) not in (None, ""):
                collected_summary.append(f"{key}={collected[key]}")

        msg = f"I can run the estimate once I have {first}."
        if len(missing) > 1:
            remaining_labels = ", ".join(_PREDICTION_FIELD_LABELS[m] for m in missing[1:3])
            msg = f"{msg} I also still need {remaining_labels}."
        if "floor_area_sqm" in missing or "lease_commence_year" in missing:
            msg = f"{msg} If unknown, you can explicitly say 'unknown floor area' or 'unknown lease commencement year'."
        if collected_summary:
            msg = f"{msg} Collected so far: {', '.join(collected_summary)}."
        return msg

    def _handle_analytics(
        self,
        session_id: str,
        state: _ChatSession,
        message_up: str,
        original_message: str,
    ) -> ChatMessageResponse:
        state.expected_field = None
        town = self._extract_town(message_up) or (str(state.slots["town"]) if state.slots.get("town") else None)
        year = self._extract_valuation_year(message_up, lease_year=None)
        quarter = self._extract_quarter(message_up)
        metric = "mean"
        if re.search(r"\bMEDIAN\b", message_up):
            metric = "median"
        elif re.search(r"\bCOUNT|HOW MANY\b", message_up):
            metric = "count"

        if town is None:
            return ChatMessageResponse(
                session_id=session_id,
                intent="analytics",
                reply="Please specify a town so I can compute the historical statistics.",
                requires_follow_up=True,
                missing_fields=["town"],
                collected_slots=dict(state.slots),
                assumptions_used=[],
                confidence_tier=None,
                prediction=None,
            )

        filtered = self.analytics_df.loc[self.analytics_df["town_norm"] == town]
        if year is not None:
            filtered = filtered.loc[filtered["year"] == int(year)]
        if quarter is not None:
            filtered = filtered.loc[filtered["quarter"] == int(quarter)]

        if filtered.empty:
            period = self._period_text(year, quarter)
            return ChatMessageResponse(
                session_id=session_id,
                intent="analytics",
                reply=(
                    f"I could not find transactions for {_to_title_words(town)}{period}. "
                    "You can try a different year or remove quarter filtering."
                ),
                requires_follow_up=False,
                missing_fields=[],
                collected_slots=dict(state.slots),
                assumptions_used=[],
                confidence_tier=None,
                prediction=None,
            )

        avg = float(filtered["resale_price"].mean())
        med = float(filtered["resale_price"].median())
        n = int(filtered.shape[0])
        period = self._period_text(year, quarter)

        if metric == "median":
            text = f"The median resale price in {_to_title_words(town)}{period} is SGD {med:,.0f} (n={n:,})."
        elif metric == "count":
            text = f"There are {n:,} observed resale transactions in {_to_title_words(town)}{period}."
        else:
            text = (
                f"The average resale price in {_to_title_words(town)}{period} is SGD {avg:,.0f} "
                f"(median {med:,.0f}, n={n:,})."
            )

        rag_answer = self.rag_service.answer(original_message, top_k=2) if self.rag_service is not None else None
        computed_facts = [
            text,
            "Historical stat computed directly from transaction records in the loaded dataset.",
        ]
        fallback = (
            f"{text} Therefore: use this historical number to benchmark whether a future estimate looks plausible "
            "for that town and period. "
            "If you want a forward-looking valuation, I can run a prediction once you share town, street, "
            "block, rooms, floor area, lease year, and floor number."
        )
        text = self._compose_policy_response(
            question=original_message,
            mode="analytics",
            computed_facts=computed_facts,
            rag_answer=rag_answer,
            fallback_text=fallback,
        )

        return ChatMessageResponse(
            session_id=session_id,
            intent="analytics",
            reply=text,
            requires_follow_up=False,
            missing_fields=[],
            collected_slots=dict(state.slots),
            assumptions_used=[],
            confidence_tier=None,
            prediction=None,
        )

    @staticmethod
    def _period_text(year: int | None, quarter: int | None) -> str:
        if year is None and quarter is None:
            return " (all available periods)"
        if year is not None and quarter is None:
            return f" in {year}"
        if year is not None and quarter is not None:
            return f" in {year} Q{quarter}"
        return f" in Q{quarter}"

    def _compose_policy_response(
        self,
        *,
        question: str,
        mode: str,
        computed_facts: list[str],
        rag_answer,
        fallback_text: str,
    ) -> str:
        definition_facts = self._definition_facts_for_query(question)
        if definition_facts:
            computed_facts = [*definition_facts, *computed_facts]

        snippets: list[str] = []
        citations: list[str] = []
        if rag_answer is not None:
            snippets = [f"{h.section}: {h.snippet}" for h in rag_answer.hits]
            citations = list(rag_answer.citations)
            if rag_answer.answer:
                computed_facts = [*computed_facts, f"Report evidence summary: {rag_answer.answer}"]

        if self.policy_llm_service is not None:
            generated = self.policy_llm_service.compose(
                question=question,
                mode=mode,
                computed_facts=computed_facts,
                retrieved_snippets=snippets,
                citations=citations,
            )
            if generated:
                return generated

        text = fallback_text
        if rag_answer is not None and rag_answer.answer:
            text = f"{text}\n\nReport evidence: {rag_answer.answer} {rag_answer.so_what}"
        return text

    @staticmethod
    def _definition_facts_for_query(question: str) -> list[str]:
        q = _canonical_spaces(question)
        facts: list[str] = []
        if re.search(r"\bSDI\b", q):
            facts.append("Acronym definition: SDI means School Demand Index in this project.")
        if re.search(r"\bGSI\b", q):
            facts.append("Acronym definition: GSI means Good School Index in this project.")
        if re.search(r"\bGEP\b", q):
            facts.append("Acronym definition: GEP means Gifted Education Programme.")
        if re.search(r"\bSAP\b", q):
            facts.append("Acronym definition: SAP means Special Assistance Plan.")
        if re.search(r"\bRDD\b", q):
            facts.append("Acronym definition: RDD means Regression Discontinuity Design.")
        return facts

    def _handle_general(self, session_id: str, state: _ChatSession, original_message: str) -> ChatMessageResponse:
        state.expected_field = None
        message_up = _canonical_spaces(original_message)
        normalized_for_intent = message_up
        if self.rag_service is not None:
            normalized_for_intent = _canonical_spaces(self.rag_service.normalize_query(original_message))

        is_help_like = bool(
            re.search(
                r"\b(HELP|WHAT CAN YOU DO|HOW TO USE|HI|HELLO|THANKS|THANK YOU)\b",
                message_up,
            )
        )
        is_summary_request = bool(
            re.search(
                r"\b(SUMMARI[SZ]E|SUMMARY|KEY FINDINGS|MAIN FINDINGS|IN PLAIN ENGLISH)\b",
                normalized_for_intent,
            )
        )
        if is_help_like:
            reply = (
                "I can help in two practical ways. "
                "First, I answer analytics questions like 'average resale price in Tampines in 2019'. "
                "Second, I run unit-level estimates like 'estimate a 4-room flat in Ang Mo Kio for March 2027'. "
                "I will ask for missing inputs step by step and show a final-input confirmation before prediction."
            )
        else:
            if is_summary_request:
                rag_query = (
                    "hedonic preferred model 3 premium rdd boundary no stable jump "
                    "flat-type heterogeneity key findings policy implications"
                )
                rag_answer = self.rag_service.answer(rag_query, top_k=3) if self.rag_service is not None else None
                fallback = (
                    "High-level report summary: hedonic results show higher resale prices nearer desirable schools, "
                    "while RDD does not show a stable policy-induced jump exactly at the 1 km boundary."
                )
            else:
                rag_answer = self.rag_service.answer(original_message, top_k=3) if self.rag_service is not None else None
                if self._is_report_question(normalized_for_intent):
                    fallback = (
                        "Based on the report evidence, here is the clearest answer in plain language."
                    )
                else:
                    fallback = (
                        "I can help in two practical ways. "
                        "Ask analytics questions by town and year, or ask for a flat-level estimate."
                    )
            reply = self._compose_policy_response(
                question=original_message,
                mode="general",
                computed_facts=[
                    "The chatbot supports report clarification, historical analytics, and model-driven flat valuation.",
                    "Predictions are generated by trained model artifacts, not by free-text guessing.",
                    "Report focus: hedonic average effects, RDD boundary checks, and subgroup heterogeneity.",
                ],
                rag_answer=rag_answer,
                fallback_text=fallback,
            )
            if is_summary_request:
                reply = f"{reply}\n\nIf useful, I can next give a 5-bullet policymaker brief."
            else:
                reply = f"{reply}\n\nIf you want, I can also convert this into a concrete policy action checklist."
        return ChatMessageResponse(
            session_id=session_id,
            intent="general",
            reply=reply,
            requires_follow_up=False,
            missing_fields=[],
            collected_slots=dict(state.slots),
            assumptions_used=[],
            confidence_tier=None,
            prediction=None,
        )

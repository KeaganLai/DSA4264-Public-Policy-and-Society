from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import REPORT_DOCX_AUTO_DISCOVER, REPORT_DOCX_PATH, ROOT_DIR

_MOJIBAKE_REPLACEMENTS = {
    "Ã¢â‚¬â„¢": "'",
    "Ã¢â‚¬Ëœ": "'",
    "Ã¢â‚¬Å“": '"',
    "Ã¢â‚¬Â": '"',
    "Ã¢â‚¬â€œ": "-",
    "Ã¢â‚¬â€": "-",
    "Ã‚": "",
}

_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "so",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "us",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


@dataclass(frozen=True)
class RagChunk:
    source: str
    section: str
    text: str


@dataclass(frozen=True)
class RagHit:
    rank: int
    score: float
    source: str
    section: str
    snippet: str


@dataclass(frozen=True)
class RagSentence:
    source: str
    section: str
    text: str
    chunk_index: int


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    so_what: str
    citations: list[str]
    hits: list[RagHit]


class RagService:
    """
    Lightweight local RAG for policy-facing Q&A.

    Retrieval is stateless: only the current user message is embedded/retrieved.
    It does not silently carry over old questions into new retrieval calls.
    """

    def __init__(self, knowledge_paths: list[Path] | None = None) -> None:
        default_paths = [
            ROOT_DIR / "service" / "knowledge" / "report_findings.md",
            ROOT_DIR / "outputs" / "rdd_improved" / "rdd_schoolfe_summary.md",
        ]
        report_path = self._resolve_report_docx_path()
        if report_path is not None:
            default_paths.insert(0, report_path)
        paths = knowledge_paths or default_paths
        self.chunks = self._load_chunks(paths)
        if not self.chunks:
            self.chunks = [
                RagChunk(
                    source="fallback",
                    section="fallback",
                    text=(
                        "No local report knowledge was found. The assistant can still run "
                        "historical averages and prediction workflows."
                    ),
                )
            ]

        corpus = [c.text for c in self.chunks]
        self.word_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.word_matrix = self.word_vectorizer.fit_transform(corpus)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.char_matrix = self.char_vectorizer.fit_transform(corpus)
        self.domain_terms = self._build_domain_terms(corpus)

        self.sentences = self._build_sentence_records(self.chunks)
        sentence_corpus = [s.text for s in self.sentences] if self.sentences else [c.text for c in self.chunks]
        self.sent_word_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.sent_word_matrix = self.sent_word_vectorizer.fit_transform(sentence_corpus)
        self.sent_char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.sent_char_matrix = self.sent_char_vectorizer.fit_transform(sentence_corpus)

    def _resolve_report_docx_path(self) -> Path | None:
        if REPORT_DOCX_PATH:
            explicit = Path(REPORT_DOCX_PATH)
            return explicit if explicit.exists() else None
        if not REPORT_DOCX_AUTO_DISCOVER:
            return None

        # Prefer likely report locations and pick the most recently modified report-like docx.
        candidate_roots = [
            ROOT_DIR / "service" / "knowledge",
            ROOT_DIR,
            ROOT_DIR / "docs",
            ROOT_DIR / "outputs",
            Path.home() / "Downloads",
        ]
        candidates: list[Path] = []
        for root in candidate_roots:
            if not root.exists():
                continue
            try:
                for p in root.glob("*.docx"):
                    name = p.name.lower()
                    if "report" in name:
                        candidates.append(p)
            except OSError:
                continue

        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def retrieve(self, query: str, top_k: int = 3) -> list[RagHit]:
        clean_query = query.strip()
        if not clean_query:
            return []
        normalized_query = self.normalize_query(clean_query)
        query_profile = self._query_profile(normalized_query)
        word_query_vec = self.word_vectorizer.transform([normalized_query])
        char_query_vec = self.char_vectorizer.transform([normalized_query])
        word_scores = (self.word_matrix @ word_query_vec.T).toarray().reshape(-1)
        char_scores = (self.char_matrix @ char_query_vec.T).toarray().reshape(-1)
        if word_scores.size == 0:
            return []

        query_up = normalized_query.upper()
        chunk_scores = np.array(
            [
                ((0.8 * float(word_scores[i])) + (0.2 * float(char_scores[i])))
                * self._source_weight(self.chunks[i].source, query_up)
                * self._query_chunk_bonus(self.chunks[i], query_up)
                for i in range(word_scores.shape[0])
            ],
            dtype=float,
        )
        sentence_scores: np.ndarray | None = None
        if self.sentences:
            sent_word_query_vec = self.sent_word_vectorizer.transform([normalized_query])
            sent_char_query_vec = self.sent_char_vectorizer.transform([normalized_query])
            sent_word_scores = (self.sent_word_matrix @ sent_word_query_vec.T).toarray().reshape(-1)
            sent_char_scores = (self.sent_char_matrix @ sent_char_query_vec.T).toarray().reshape(-1)
            sentence_scores = np.array(
                [
                    ((0.75 * float(sent_word_scores[i])) + (0.25 * float(sent_char_scores[i])))
                    * self._source_weight(self.sentences[i].source, query_up)
                    * self._query_sentence_bonus(self.sentences[i], query_up)
                    for i in range(sent_word_scores.shape[0])
                ],
                dtype=float,
            )

        pool: list[tuple[float, RagHit]] = []
        chunk_ranked_idx = np.argsort(chunk_scores)[::-1]
        max_chunk_candidates = max(8, top_k * 6)
        for idx in chunk_ranked_idx[:max_chunk_candidates]:
            chunk = self.chunks[int(idx)]
            score = float(chunk_scores[idx])
            if score <= 0:
                continue
            if self._is_low_signal_chunk(chunk):
                continue
            snippet = self._query_focused_snippet(chunk.text, normalized_query, query_profile=query_profile)
            if not snippet:
                continue
            pool.append(
                (
                    score,
                    RagHit(
                        rank=0,
                        score=score,
                        source=chunk.source,
                        section=chunk.section,
                        snippet=snippet,
                    ),
                )
            )

        if sentence_scores is not None and self.sentences:
            sent_ranked_idx = np.argsort(sentence_scores)[::-1]
            max_sentence_candidates = max(12, top_k * 10)
            for idx in sent_ranked_idx[:max_sentence_candidates]:
                sent = self.sentences[int(idx)]
                score = float(sentence_scores[idx])
                if score <= 0:
                    continue
                if self._looks_low_signal_snippet(sent.text):
                    continue
                snippet = self._clean_snippet_text(sent.text)
                if not snippet:
                    continue
                # Slightly favor direct sentence hits for precise factual Q&A.
                eff = score * 1.08
                pool.append(
                    (
                        eff,
                        RagHit(
                            rank=0,
                            score=eff,
                            source=sent.source,
                            section=sent.section,
                            snippet=snippet,
                        ),
                    )
                )

        if not pool:
            return []

        # Deduplicate by normalized snippet text + section and keep best score.
        best_by_key: dict[str, tuple[float, RagHit]] = {}
        for score, hit in pool:
            key = f"{hit.source}|{hit.section}|{self._normalize_hit_key(hit.snippet)}"
            prev = best_by_key.get(key)
            if prev is None or score > prev[0]:
                best_by_key[key] = (score, hit)

        ranked = sorted(best_by_key.values(), key=lambda x: x[0], reverse=True)
        selected: list[RagHit] = []
        used_sections: set[str] = set()
        for score, hit in ranked:
            if len(selected) >= max(1, top_k):
                break
            section_key = f"{hit.source}|{hit.section}".lower()
            # Encourage section diversity for broad questions.
            if query_profile.get("summary") and section_key in used_sections and len(selected) < max(1, top_k - 1):
                continue
            selected.append(
                RagHit(
                    rank=len(selected) + 1,
                    score=score,
                    source=hit.source,
                    section=hit.section,
                    snippet=hit.snippet,
                )
            )
            used_sections.add(section_key)

        if len(selected) < max(1, top_k):
            for score, hit in ranked:
                if len(selected) >= max(1, top_k):
                    break
                if any(
                    self._normalize_hit_key(h.snippet) == self._normalize_hit_key(hit.snippet) and h.section == hit.section
                    for h in selected
                ):
                    continue
                selected.append(
                    RagHit(
                        rank=len(selected) + 1,
                        score=score,
                        source=hit.source,
                        section=hit.section,
                        snippet=hit.snippet,
                    )
                )

        return selected

    def answer(self, question: str, top_k: int = 3) -> RagAnswer:
        hits = self.retrieve(question, top_k=max(top_k, 6))
        if not hits:
            return RagAnswer(
                answer=(
                    "I could not find a close match in the local report notes. "
                    "I can still answer with historical transaction stats or run a flat-level prediction."
                ),
                so_what="Therefore: use this as a signal to ask a narrower town/year or unit-level question.",
                citations=[],
                hits=[],
            )

        answer_points = self._compose_answer_points(question, hits)
        answer_text = " ".join(answer_points) if answer_points else hits[0].snippet
        so_what = self._derive_so_what(" ".join(h.section + " " + h.snippet for h in hits))
        citations = [f"[{i + 1}] {h.source} > {h.section}" for i, h in enumerate(hits)]
        return RagAnswer(answer=answer_text, so_what=so_what, citations=citations, hits=hits)

    def normalize_query(self, text: str) -> str:
        return self._normalize_query(text)

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = re.sub(r"\s+", " ", item).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    @staticmethod
    def _normalize_hit_key(text: str) -> str:
        t = re.sub(r"https?://\S+", "", text)
        t = re.sub(r"\s+", " ", t).strip().lower()
        return t

    @staticmethod
    def _clean_snippet_text(text: str) -> str:
        t = re.sub(r"https?://\S+", "", text).strip()
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _looks_low_signal_snippet(text: str) -> bool:
        t = re.sub(r"\s+", " ", text).strip()
        up = t.upper()
        if len(t) < 28:
            return True
        if t.startswith(":"):
            return True
        alpha = re.findall(r"[A-Za-z]{2,}", t)
        if len(alpha) < 6 and not re.search(r"\d+(?:\.\d+)?%", t):
            return True
        if re.search(r"\b(FILES GENERATED|APPENDIX [A-Z0-9]+)\b", t.upper()):
            return True
        if re.search(r"\bMODEL SPECIFICATION\b", up):
            return True
        if up.count("MODEL") >= 2 and len(re.findall(r"\d+(?:\.\d+)?%", t)) >= 2 and t.count(".") <= 1:
            return True
        return False

    @staticmethod
    def _query_profile(normalized_query: str) -> dict[str, bool]:
        q_up = normalized_query.upper()
        premium = "PREMIUM" in q_up or "1KM" in q_up or "1 KM" in q_up or "GOOD SCHOOL" in q_up
        exact_numeric = premium and any(
            token in q_up for token in ["HOW MUCH", "WHAT IS", "WHAT'S", "VALUE", "PERCENT", "%", "FOUND"]
        )
        model = "MODEL" in q_up or "PREFERRED" in q_up or "SPECIFICATION" in q_up
        model_choice = any(token in q_up for token in ["WHICH MODEL", "MODEL DID WE USE", "PREFERRED MODEL"])
        model3 = "MODEL 3" in q_up
        rdd = "RDD" in q_up or "DISCONTINUITY" in q_up or "BOUNDARY" in q_up
        threshold = "THRESHOLD" in q_up or any(x in q_up for x in ["75", "80", "85", "90"])
        heterogeneity = "HETEROGENEITY" in q_up or "FLAT TYPE" in q_up or "ROOM" in q_up
        summary = any(
            token in q_up
            for token in [
                "SUMMARY",
                "MAIN TAKEAWAY",
                "KEY FINDINGS",
                "MAIN FINDINGS",
                "PLAIN ENGLISH",
                "TAKEAWAY",
            ]
        )
        sdi = "SDI" in q_up or "SCHOOL DESIRABILITY INDEX" in q_up
        return {
            "premium": premium,
            "exact_numeric": exact_numeric,
            "model": model,
            "model_choice": model_choice,
            "model3": model3,
            "rdd": rdd,
            "threshold": threshold,
            "heterogeneity": heterogeneity,
            "summary": summary,
            "sdi": sdi,
        }

    def _query_sentence_bonus(self, sentence: RagSentence, query_up: str) -> float:
        text_up = sentence.text.upper()
        section_up = sentence.section.upper()
        profile = self._query_profile(query_up)
        bonus = 1.0

        if profile["premium"] and not profile["rdd"]:
            if "QUICK FACTS" in section_up:
                bonus *= 2.0
            if re.search(r"\b(1\.16%|0\.41%)\b", text_up):
                bonus *= 1.8
            if re.search(r"\d+(?:\.\d+)?%", text_up):
                bonus *= 1.35
            if "RDD" in section_up and "HEDONIC" not in section_up:
                bonus *= 0.6

        if profile["model"]:
            if "MODEL 3" in text_up:
                bonus *= 1.45
            if "PREFERRED" in text_up and "SPECIFICATION" in text_up:
                bonus *= 1.55
            if "MODEL 1" in text_up and "MODEL 3" not in text_up:
                bonus *= 0.85

        if profile["sdi"]:
            if "SCHOOL DESIRABILITY INDEX" in text_up or "SDI" in text_up:
                bonus *= 1.3
            if re.search(r"\b(BALLOT|PERSISTENCE|DEMAND)\b", text_up):
                bonus *= 1.2

        if profile["summary"]:
            if any(
                token in section_up
                for token in ["QUICK FACTS", "HEDONIC", "EMPIRICAL RESULTS", "POLICY IMPLICATIONS", "RDD SUMMARY"]
            ):
                bonus *= 1.5
            if any(token in section_up for token in ["DEPENDENT VARIABLE", "METHODOLOGY", "APPENDIX"]):
                bonus *= 0.45
            if any(token in section_up for token in ["THRESHOLD ROBUSTNESS", "ALTERNATIVE WEIGHTS", "COVARIATE BALANCE"]):
                bonus *= 0.72

        if "HTTP://" in text_up or "HTTPS://" in text_up:
            bonus *= 0.5
        if self._looks_low_signal_snippet(sentence.text):
            bonus *= 0.55
        return max(0.2, min(5.0, bonus))

    def _compose_answer_points(self, question: str, hits: list[RagHit]) -> list[str]:
        profile = self._query_profile(self.normalize_query(question))
        snippets = [self._clean_snippet_text(h.snippet) for h in hits if h.snippet]
        snippets = [s for s in snippets if s and not self._looks_low_signal_snippet(s)]
        snippets = self._dedupe_preserve_order(snippets)
        if not snippets:
            return []

        def pick(matchers: list[re.Pattern[str]], exclude: set[int]) -> int | None:
            for idx, text in enumerate(snippets):
                if idx in exclude:
                    continue
                if any(p.search(text) for p in matchers):
                    return idx
            return None

        chosen_idx: list[int] = []
        used: set[int] = set()

        # Numeric premium question: lead with exact premium evidence.
        if profile["premium"]:
            idx = pick(
                [
                    re.compile(r"\b1\.16%\b.*\b0\.41%\b", re.IGNORECASE),
                    re.compile(r"\bwithin\s*0\s*to\s*1\s*km\b.*\bwithin\s*1\s*to\s*2\s*km\b", re.IGNORECASE),
                    re.compile(r"\bpremium\b.*\d+(?:\.\d+)?%", re.IGNORECASE),
                ],
                used,
            )
            if idx is not None:
                chosen_idx.append(idx)
                used.add(idx)

        if profile["model"]:
            idx = pick(
                [
                    re.compile(r"\bmodel\s*3\b.*\bpreferred\b", re.IGNORECASE),
                    re.compile(r"\bmodel\s*3\b.*\badopted\b", re.IGNORECASE),
                    re.compile(r"\bpreferred hedonic specification\b", re.IGNORECASE),
                ],
                used,
            )
            if idx is not None:
                chosen_idx.append(idx)
                used.add(idx)

        if profile["summary"]:
            idx = pick([re.compile(r"\brdd\b|\bboundary\b|\bdiscontinuity\b", re.IGNORECASE)], used)
            if idx is None:
                idx = pick(
                    [re.compile(r"\bpolicy\b|\bimplication\b|\binequality\b|\baffordability\b", re.IGNORECASE)],
                    used,
                )
            if idx is not None:
                chosen_idx.append(idx)
                used.add(idx)

        if profile["sdi"]:
            idx = pick([re.compile(r"\bconstruction\b.*\bschool desirability index\b", re.IGNORECASE)], used)
            if idx is None:
                idx = pick([re.compile(r"\bballot\b|\bpersistence\b|\bdemand\b", re.IGNORECASE)], used)
            if idx is None:
                idx = pick([re.compile(r"\bschool desirability index\b", re.IGNORECASE)], used)
            if idx is not None:
                chosen_idx.append(idx)
                used.add(idx)
            idx2 = pick([re.compile(r"\bballot\b|\bpersistence\b|\bdemand\b", re.IGNORECASE)], used)
            if idx2 is not None:
                chosen_idx.append(idx2)
                used.add(idx2)

        for idx in range(len(snippets)):
            if len(chosen_idx) >= 3:
                break
            if idx in used:
                continue
            if self._is_table_like_snippet(snippets[idx]):
                continue
            if profile["sdi"] and re.match(r"^\s*SDI\s*=", snippets[idx], flags=re.IGNORECASE):
                continue
            chosen_idx.append(idx)
            used.add(idx)

        points = [snippets[i] for i in chosen_idx[:3]]
        return self._dedupe_preserve_order(points)

    @staticmethod
    def _is_table_like_snippet(text: str) -> bool:
        up = text.upper()
        if "MODEL SPECIFICATION" in up:
            return True
        if up.count("MODEL") >= 2 and len(re.findall(r"\d+(?:\.\d+)?%", text)) >= 2 and text.count(".") <= 1:
            return True
        if re.search(r"\bTHRESHOLD\s+PREMIUM\b", up):
            return True
        if re.search(r"\bGSI WEIGHTING SCHEME\b", up):
            return True
        return False

    def _query_focused_snippet(
        self,
        text: str,
        normalized_query: str,
        max_len: int = 340,
        query_profile: dict[str, bool] | None = None,
    ) -> str:
        lines = [re.sub(r"^\s*[-*]\s+", "", ln.strip()) for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""

        sentence_candidates: list[str] = []
        for line in lines:
            parts = re.split(r"(?<=[.!?])\s+", line)
            for p in parts:
                p = re.sub(r"\s+", " ", p).strip()
                if p:
                    sentence_candidates.append(p)
        if not sentence_candidates:
            sentence_candidates = [re.sub(r"\s+", " ", " ".join(lines)).strip()]

        query_tokens = {
            tok
            for tok in re.findall(r"[a-z0-9]{2,}", normalized_query.lower())
            if tok not in _QUERY_STOPWORDS
        }
        if not query_tokens:
            query_tokens = {"report"}
        profile = query_profile or self._query_profile(normalized_query)
        query_has_percent = profile["premium"]
        query_asks_numeric = profile["exact_numeric"]
        query_mentions_model = profile["model"]
        query_mentions_model3 = profile["model3"]
        asks_model_choice = profile["model_choice"]
        query_mentions_rdd = profile["rdd"]

        def score_sentence(sent: str) -> float:
            sent_low = sent.lower()
            sent_tokens = {
                tok
                for tok in re.findall(r"[a-z0-9]{2,}", sent_low)
                if tok not in _QUERY_STOPWORDS
            }
            overlap = len(query_tokens & sent_tokens)
            score = float(overlap)
            if query_has_percent and ("%" in sent or re.search(r"\d+(?:\.\d+)?\*{0,2}", sent)):
                score += 1.2
            if query_asks_numeric and ("%" in sent or re.search(r"\d+(?:\.\d+)?\*{0,2}", sent)):
                score += 2.3
            if query_asks_numeric and "model 3" in sent_low and ("%" in sent or "1.16" in sent or "0.41" in sent):
                score += 1.8
            if query_asks_numeric and not re.search(r"\d", sent):
                score -= 0.9
            if query_mentions_model and "model" in sent_low:
                score += 1.0
            if query_mentions_model3 and "model 3" in sent_low:
                score += 1.4
            if asks_model_choice and ("preferred" in sent_low or "adopted" in sent_low):
                score += 1.3
            if asks_model_choice and "model 1 produces counterintuitive" in sent_low:
                score -= 1.8
            if query_asks_numeric and "counterintuitive" in sent_low:
                score -= 1.2
            if query_mentions_rdd and ("rdd" in sent_low or "boundary" in sent_low):
                score += 0.8
            if "http://" in sent_low or "https://" in sent_low:
                score -= 1.0
            if sent.strip().startswith(":"):
                score -= 1.1
            if re.search(r"\b(INDICATOR FOR FLATS|VECTOR OF CONTROL VARIABLES|ERROR TERM)\b", sent.upper()):
                score -= 1.0
            if len(sent) < 28:
                score -= 0.4
            return score

        best_idx = 0
        best_score = float("-inf")
        for idx, sent in enumerate(sentence_candidates):
            s = score_sentence(sent)
            if s > best_score:
                best_score = s
                best_idx = idx

        chosen_idx = best_idx
        chosen = sentence_candidates[best_idx]
        if query_asks_numeric and "%" not in chosen:
            numeric_best = ""
            numeric_best_score = float("-inf")
            numeric_idx = -1
            for idx, sent in enumerate(sentence_candidates):
                sent_score = score_sentence(sent)
                has_number = bool(re.search(r"\d", sent))
                has_percent = "%" in sent
                boost = 1.0 + (1.2 if has_percent else 0.0) + (0.4 if has_number else 0.0)
                effective = sent_score * boost
                if effective > numeric_best_score and has_number:
                    numeric_best_score = effective
                    numeric_best = sent
                    numeric_idx = idx
            if numeric_best and numeric_best_score >= best_score - 0.4:
                chosen = numeric_best
                if numeric_idx >= 0:
                    chosen_idx = numeric_idx
        # If the chosen sentence is short, append the next sentence for context.
        if len(chosen) < 120 and chosen_idx + 1 < len(sentence_candidates):
            nxt = sentence_candidates[chosen_idx + 1]
            if nxt and nxt.lower() != chosen.lower():
                chosen = f"{chosen} {nxt}"

        chosen = re.sub(r"\s+", " ", chosen).strip()
        chosen = self._clean_snippet_text(chosen)
        if chosen.startswith(":"):
            return ""
        if len(chosen) <= max_len:
            return chosen
        return f"{chosen[: max_len - 3].rstrip()}..."

    @staticmethod
    def _normalize_corpus_text(text: str) -> str:
        t = text.replace("\u200b", "").replace("\xa0", " ")
        for bad, good in _MOJIBAKE_REPLACEMENTS.items():
            t = t.replace(bad, good)
        t = re.sub(r"\s+\n", "\n", t)
        return t

    def _load_chunks(self, paths: list[Path]) -> list[RagChunk]:
        chunks: list[RagChunk] = []
        for path in paths:
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            if suffix in {".md", ".txt"}:
                text = self._normalize_corpus_text(path.read_text(encoding="utf-8", errors="ignore"))
                chunks.extend(self._chunk_markdown(path.name, text))
            elif suffix == ".docx":
                text = self._read_docx_text(path)
                chunks.extend(self._chunk_plain(path.name, text))
            else:
                text = self._normalize_corpus_text(path.read_text(encoding="utf-8", errors="ignore"))
                chunks.extend(self._chunk_plain(path.name, text))
        return chunks

    @staticmethod
    def _chunk_plain(source: str, text: str) -> list[RagChunk]:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        if not blocks:
            return []
        packed: list[str] = []
        buf: list[str] = []
        buf_len = 0
        max_chars = 1200
        for block in blocks:
            block_len = len(block)
            if buf and (buf_len + block_len + 1 > max_chars):
                packed.append("\n".join(buf))
                buf = [block]
                buf_len = block_len
            else:
                buf.append(block)
                buf_len += block_len + 1
        if buf:
            packed.append("\n".join(buf))
        return [RagChunk(source=source, section=f"Block {i + 1}", text=chunk) for i, chunk in enumerate(packed)]

    @staticmethod
    def _chunk_markdown(source: str, text: str) -> list[RagChunk]:
        lines = text.splitlines()
        chunks: list[RagChunk] = []
        current_section = "Overview"
        buffer: list[str] = []
        max_chars = 1100

        def flush() -> None:
            nonlocal buffer
            block = "\n".join(buffer).strip()
            if block:
                chunks.append(RagChunk(source=source, section=current_section, text=block))
            buffer = []

        for raw_line in lines:
            line = raw_line.rstrip()
            if re.match(r"^\s*Generated from\s+", line, flags=re.IGNORECASE):
                continue
            heading = re.match(r"^\s*#{1,6}\s+(.*)$", line)
            if heading:
                flush()
                current_section = heading.group(1).strip()
                continue
            if not line.strip():
                flush()
                continue
            buffer.append(line)
            if len("\n".join(buffer)) >= max_chars:
                flush()
        flush()
        return chunks

    def _build_sentence_records(self, chunks: list[RagChunk]) -> list[RagSentence]:
        records: list[RagSentence] = []
        for i, chunk in enumerate(chunks):
            lines = [re.sub(r"^\s*[-*]\s+", "", ln.strip()) for ln in chunk.text.splitlines() if ln.strip()]
            if not lines:
                continue
            text = " ".join(lines)
            text = re.sub(r"\s+", " ", text).strip()
            parts = re.split(r"(?<=[.!?])\s+", text)
            for part in parts:
                sent = self._clean_snippet_text(part)
                if self._looks_low_signal_snippet(sent):
                    continue
                if len(sent) > 420:
                    # Keep long evidence usable by slicing into smaller sentence-like spans.
                    start = 0
                    while start < len(sent):
                        sub = sent[start : start + 320].strip()
                        if sub and not self._looks_low_signal_snippet(sub):
                            records.append(
                                RagSentence(
                                    source=chunk.source,
                                    section=chunk.section,
                                    text=sub,
                                    chunk_index=i,
                                )
                            )
                        start += 300
                    continue
                records.append(
                    RagSentence(
                        source=chunk.source,
                        section=chunk.section,
                        text=sent,
                        chunk_index=i,
                    )
                )
        return records

    @staticmethod
    def _read_docx_text(path: Path) -> str:
        try:
            with zipfile.ZipFile(path, "r") as zf:
                xml_data = zf.read("word/document.xml")
        except Exception:
            return ""

        try:
            root = ET.fromstring(xml_data)
        except Exception:
            return ""

        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []
        for p in root.findall(".//w:p", ns):
            texts = [t.text or "" for t in p.findall(".//w:t", ns)]
            line = "".join(texts).strip()
            if line:
                paragraphs.append(line)
        return RagService._normalize_corpus_text("\n\n".join(paragraphs))

    @staticmethod
    def _build_domain_terms(corpus: list[str]) -> set[str]:
        terms: set[str] = set()
        for text in corpus:
            for token in re.findall(r"[A-Za-z][A-Za-z0-9/_-]{3,}", text):
                t = token.lower()
                if t.endswith(".csv") or t.endswith(".md"):
                    continue
                terms.add(t)
        # Strong policy/domain anchors for typo correction.
        terms.update(
            {
                "hedonic",
                "regression",
                "rdd",
                "premium",
                "threshold",
                "singapore",
                "resale",
                "flat",
                "school",
                "mnd",
                "policy",
                "tampines",
                "kallang",
                "whampoa",
                "ang",
                "mo",
                "kio",
                "bukit",
                "batok",
            }
        )
        return terms

    def _normalize_query(self, text: str) -> str:
        q = text.strip().lower()
        # Common misspellings and shorthand seen in user chats.
        replacements = {
            "hedonix": "hedonic",
            "hedonicc": "hedonic",
            "premuim": "premium",
            "premimum": "premium",
            "regresion": "regression",
            "schhol": "school",
            "shcool": "school",
            "singapre": "singapore",
            "resell": "resale",
            "sumarise": "summarise",
            "sumrise": "summarise",
            "summarizee": "summarize",
            "fromm": "from",
            "pleasse": "please",
            "resultz": "results",
            "ppl": "people",
            "eng": "english",
            "whampoa": "whampoa",
        }
        for bad, good in replacements.items():
            q = re.sub(rf"\b{re.escape(bad)}\b", good, q)

        tokens = re.findall(r"[a-z0-9/_-]+", q)
        q_norm = " ".join(tokens)
        profile = self._query_profile(q_norm)

        expansions: list[str] = []
        if profile["summary"]:
            expansions.extend(
                [
                    "hedonic preferred model 3",
                    "premium within 1 km and 1 to 2 km",
                    "rdd boundary no stable jump",
                    "policy implications affordability equity",
                ]
            )
        if profile["premium"]:
            expansions.extend(
                [
                    "hedonic empirical results",
                    "preferred model specification",
                    "within 1 km and 1 to 2 km premium",
                ]
            )
        if profile["model"] or profile["model_choice"] or profile["model3"]:
            expansions.extend(
                [
                    "model 3 preferred specification",
                    "hedonic regression empirical results",
                ]
            )
        if profile["sdi"]:
            expansions.extend(
                [
                    "construction of school desirability index",
                    "ballot intensity and persistence of demand",
                ]
            )

        if expansions:
            q_norm = f"{q_norm} " + " ".join(expansions)
        return re.sub(r"\s+", " ", q_norm).strip()

    @staticmethod
    def _source_weight(source: str, query_up: str) -> float:
        s = source.upper()
        asks_summary = any(token in query_up for token in ["SUMMAR", "REPORT", "TAKEAWAY", "KEY FINDING"])
        asks_rdd = any(token in query_up for token in ["RDD", "DISCONTINUITY", "BOUNDARY"])
        if "REPORT_FINDINGS" in s:
            if asks_summary:
                return 1.8
            return 1.4
        if "RDD_SCHOOLFE_SUMMARY" in s:
            if asks_rdd:
                return 1.8
            if asks_summary:
                return 1.3
            return 1.0
        return 1.0

    @staticmethod
    def _query_chunk_bonus(chunk: RagChunk, query_up: str) -> float:
        text_up = chunk.text.upper()
        section_up = chunk.section.upper()
        profile = RagService._query_profile(query_up)
        bonus = 1.0

        if profile["premium"] and not profile["rdd"]:
            if "QUICK FACTS" in section_up:
                bonus *= 2.2
            if any(token in section_up for token in ["HEDONIC", "EMPIRICAL RESULTS", "5.1.1", "5.1.4"]):
                bonus *= 1.6
            if "MODEL 3" in text_up:
                bonus *= 1.2
            pct_count = len(re.findall(r"\d+(?:\.\d+)?%\*{0,2}", chunk.text))
            if pct_count >= 2:
                bonus *= 1.35
            if profile["exact_numeric"] and re.search(r"\b1\.16%\b|\b0\.41%\b", text_up):
                bonus *= 1.8
            if "HETEROGENEITY" in section_up and not profile["heterogeneity"]:
                bonus *= 0.72
            if "THRESHOLD ROBUSTNESS" in section_up and not profile["threshold"]:
                bonus *= 0.78
            if "RDD" in section_up and "HEDONIC" not in section_up:
                bonus *= 0.55
            if "COUNTERINTUITIVE" in text_up and profile["exact_numeric"]:
                bonus *= 0.75

        if profile["model"] and not profile["rdd"]:
            if "MODEL 3" in text_up:
                bonus *= 1.45
            if "PREFERRED SPECIFICATION" in text_up or "MODEL 3 IS ADOPTED" in text_up:
                bonus *= 1.6
            if profile["model_choice"] and "QUICK FACTS" in section_up:
                bonus *= 1.9
            if "MODEL 1" in text_up and "MODEL 3" not in text_up:
                bonus *= 0.85
            if "RDD" in text_up and "HEDONIC" not in text_up:
                bonus *= 0.55

        if profile["sdi"]:
            if any(token in section_up for token in ["3.2", "SCHOOL DESIRABILITY INDEX", "DEFINITIONS", "ACRONYM"]):
                bonus *= 1.45
            if any(token in text_up for token in ["BALLOT", "PERSISTENCE", "DEMAND"]):
                bonus *= 1.2

        if profile["summary"]:
            if any(
                token in section_up
                for token in [
                    "QUICK FACTS",
                    "HEDONIC RESULTS OVERVIEW",
                    "EMPIRICAL RESULTS",
                    "POLICY INTERPRETATION",
                    "POLICY IMPLICATIONS",
                    "RDD SUMMARY",
                ]
            ):
                bonus *= 1.7
            if any(token in section_up for token in ["THRESHOLD ROBUSTNESS", "ALTERNATIVE WEIGHTS", "COVARIATE BALANCE"]):
                bonus *= 0.75
            if any(
                token in section_up
                for token in [
                    "DEPENDENT VARIABLE",
                    "METHODOLOGY",
                    "DATA PREPROCESSING",
                    "APPENDIX",
                ]
            ):
                bonus *= 0.4

        if "HTTP://" in text_up or "HTTPS://" in text_up:
            bonus *= 0.55
        return max(0.2, min(6.0, bonus))

    @staticmethod
    def _is_low_signal_chunk(chunk: RagChunk) -> bool:
        section_up = chunk.section.upper()
        text_up = chunk.text.upper()
        compact = re.sub(r"\s+", " ", chunk.text).strip()
        if len(compact) < 45:
            return True
        if "FILES GENERATED" in section_up:
            return True
        if "GENERATED FROM" in text_up and "RAG USAGE" in text_up:
            return True
        if compact.startswith(":"):
            return True
        if re.search(r"\b(WHERE:|INDICATOR FOR FLATS|VECTOR OF CONTROL VARIABLES|ERROR TERM)\b", text_up):
            return True
        file_like_lines = sum(
            1
            for line in chunk.text.splitlines()
            if line.strip().lower().endswith((".csv", ".md", ".json", ".txt"))
        )
        if file_like_lines >= 3:
            return True
        url_like = sum(
            1
            for line in chunk.text.splitlines()
            if "http://" in line.lower() or "https://" in line.lower()
        )
        if url_like >= 2 and len(re.findall(r"[A-Za-z]{3,}", compact)) < 20:
            return True
        alpha_words = re.findall(r"[A-Za-z]{2,}", compact)
        if len(alpha_words) < 7 and any(k in text_up for k in ["PREMIUM", "MODEL", "THRESHOLD"]):
            return True
        tokens = [t for t in re.split(r"\s+", text_up) if t]
        if tokens:
            long_token_ratio = sum(1 for t in tokens if "_" in t or t.endswith(".CSV") or t.endswith(".MD")) / len(tokens)
            if long_token_ratio > 0.45:
                return True
        return False

    @staticmethod
    def _derive_so_what(summary_text: str) -> str:
        up = summary_text.upper()
        if "4-ROOM" in up or "4 ROOM" in up or "FLAT-TYPE HETEROGENEITY" in up:
            return (
                "Therefore: monitor affordability pressure most closely for larger family flats, "
                "because that is where nearby-school premiums are strongest."
            )
        if "RDD" in up or "DISCONTINUITY" in up:
            return (
                "Therefore: avoid saying the 1 km boundary alone causes the full price difference; "
                "use broader neighborhood and demand levers in policy design."
            )
        if "THRESHOLD" in up:
            return (
                "Therefore: findings are directionally stable across reasonable thresholds, so policy discussion "
                "can focus on the broader demand pattern rather than one exact cutoff."
            )
        return (
            "Therefore: use this as directional evidence for policy planning, and pair it with affordability "
            "and equity checks before making implementation decisions."
        )

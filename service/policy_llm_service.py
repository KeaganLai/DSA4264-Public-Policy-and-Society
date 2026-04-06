from __future__ import annotations

import json
from http import HTTPStatus
import re
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import (
    CHAT_LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_PROBE_TIMEOUT_SECONDS,
    OLLAMA_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_CHAT_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    POLICY_CONCLUSION_WORD,
)


class PolicyLLMService:
    """
    Optional provider-backed answer composer for policy chat.

    Predictions remain tool-computed (model artifacts + feature pipeline).
    This service only generates natural-language explanations.
    """

    def __init__(self) -> None:
        self.provider = CHAT_LLM_PROVIDER
        self.conclusion_word = POLICY_CONCLUSION_WORD.rstrip(":")
        self.openai_api_key = OPENAI_API_KEY
        self.openai_model = OPENAI_CHAT_MODEL
        self.openai_base_url = OPENAI_BASE_URL.rstrip("/")
        self.timeout_seconds = OPENAI_TIMEOUT_SECONDS
        self.ollama_base_url = OLLAMA_BASE_URL.rstrip("/")
        self.ollama_model = OLLAMA_MODEL
        self.ollama_timeout_seconds = OLLAMA_TIMEOUT_SECONDS
        self.ollama_probe_timeout_seconds = OLLAMA_PROBE_TIMEOUT_SECONDS
        self.last_error: str | None = None

    @property
    def enabled(self) -> bool:
        if self.provider == "openai":
            return bool(self.openai_api_key)
        if self.provider == "ollama":
            return True
        return False

    def status(self, check_connection: bool = False) -> dict[str, str | bool | None]:
        payload: dict[str, str | bool | None] = {
            "provider": self.provider,
            "enabled": self.enabled,
            "openai_model": self.openai_model if self.provider == "openai" else None,
            "openai_base_url": self.openai_base_url if self.provider == "openai" else None,
            "openai_api_key_configured": bool(self.openai_api_key) if self.provider == "openai" else None,
            "ollama_model": self.ollama_model if self.provider == "ollama" else None,
            "ollama_base_url": self.ollama_base_url if self.provider == "ollama" else None,
            "reachable": None,
            "last_error": None,
        }
        if not check_connection:
            return payload

        ok, reason = self._connection_probe()
        payload["reachable"] = ok
        payload["last_error"] = reason
        return payload

    def compose(
        self,
        *,
        question: str,
        mode: str,
        computed_facts: Iterable[str],
        retrieved_snippets: Iterable[str],
        citations: Iterable[str],
    ) -> str | None:
        if not self.enabled:
            return None

        facts = [f.strip() for f in computed_facts if f and f.strip()]
        snippets = [s.strip() for s in retrieved_snippets if s and s.strip()]
        citation_text = " | ".join([c.strip() for c in citations if c and c.strip()])

        prompt = self._build_prompt(
            question=question,
            mode=mode,
            facts=facts,
            snippets=snippets,
            citation_text=citation_text,
        )
        text: str | None = None
        if self.provider == "openai":
            text = self._call_openai(prompt)
        elif self.provider == "ollama":
            text = self._call_ollama(prompt)
        if not text:
            return None

        normalized = text.strip()
        normalized = self._apply_grounding_guardrails(
            question=question,
            generated_text=normalized,
            facts=facts,
            snippets=snippets,
        )
        if normalized and f"{self.conclusion_word}:" not in normalized:
            normalized = f"{normalized}\n\n{self.conclusion_word}: This should be used with policy caution and supporting evidence."
        return normalized

    def _build_prompt(
        self,
        *,
        question: str,
        mode: str,
        facts: list[str],
        snippets: list[str],
        citation_text: str,
    ) -> str:
        if self.provider == "ollama":
            # Keep local-model prompts compact to reduce latency on CPU-only machines.
            compact_facts = facts[:8]
            compact_snippets = snippets[:4]
            facts_block = "\n".join(f"- {x[:360]}" for x in compact_facts) if compact_facts else "- No computed facts provided."
            snippets_block = (
                "\n".join(f"- {x[:360]}" for x in compact_snippets) if compact_snippets else "- No retrieved snippets."
            )
            return (
                "You are an assistant for non-technical policy officers in Singapore MND.\n"
                "Use plain language. Stay factual. Do not invent numbers.\n"
                "Do not invent acronym expansions. Only use acronym definitions explicitly provided in facts/snippets.\n"
                "If acronym meaning is not provided, say it is not specified in the current evidence.\n"
                "Do not output source lists, citation labels, or bracketed references.\n"
                "If asked about model choice or premium size, use exact values and model labels from facts/snippets.\n"
                "If preferred model evidence is present, prioritize that over intermediate model rows.\n"
                "If facts/snippets conflict, prefer 'preferred specification' and 'quick facts' evidence.\n"
                "If exact values are not present in facts/snippets, explicitly say the value is not available.\n"
                f"End with one line starting with '{self.conclusion_word}:'.\n\n"
                f"Question: {question}\n"
                f"Mode: {mode}\n"
                f"Facts:\n{facts_block}\n"
                f"Report snippets:\n{snippets_block}\n"
                "Write 2 short paragraphs, then the conclusion line."
            )

        facts_block = "\n".join(f"- {x}" for x in facts) if facts else "- No computed facts provided."
        snippets_block = "\n".join(f"- {x}" for x in snippets) if snippets else "- No retrieved snippets."
        citation_block = citation_text or "No citations available."
        return (
            "You are an assistant for non-technical policy officers in Singapore MND.\n"
            "Write in plain language, minimal jargon, and stay grounded in supplied facts.\n"
            "Do not invent numbers. Do not claim causal certainty unless facts explicitly support it.\n"
            "Do not invent acronym expansions. Only use acronym definitions explicitly provided in facts/snippets.\n"
            "If acronym meaning is not provided, say it is not specified in the current evidence.\n"
            "Do not output source lists, citation labels, or bracketed references.\n"
            "If asked about model choice or premium size, use exact values and model labels from facts/snippets.\n"
            "If facts/snippets conflict, prefer lines marked preferred specification and quick facts.\n"
            "If exact values are not present in facts/snippets, explicitly say the value is not available.\n"
            f"Use exactly one short conclusion line starting with '{self.conclusion_word}:'.\n\n"
            f"User question:\n{question}\n\n"
            f"Mode: {mode}\n\n"
            f"Computed facts:\n{facts_block}\n\n"
            f"Retrieved report snippets:\n{snippets_block}\n\n"
            f"Citations:\n{citation_block}\n\n"
            "Output style:\n"
            "1) 1 to 3 concise paragraphs.\n"
            f"2) End with one line that starts with '{self.conclusion_word}:'.\n"
            "3) Keep under 170 words.\n"
        )

    def _call_openai(self, prompt: str) -> str | None:
        response_payload = {
            "model": self.openai_model,
            "input": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
        raw = self._http_json_post(
            url=f"{self.openai_base_url}/responses",
            body=response_payload,
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_seconds,
        )
        if raw is None:
            # Fallback to chat/completions for compatible providers that do not expose /responses.
            return self._call_openai_chat_completions(prompt)

        output_text = raw.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = raw.get("output", [])
        for item in output:
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        return text
        return self._call_openai_chat_completions(prompt)

    def _call_openai_chat_completions(self, prompt: str) -> str | None:
        payload = {
            "model": self.openai_model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        raw = self._http_json_post(
            url=f"{self.openai_base_url}/chat/completions",
            body=payload,
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_seconds,
        )
        if raw is None:
            return None
        choices = raw.get("choices", [])
        if not choices:
            return None
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    txt = part.get("text")
                    if isinstance(txt, str) and txt.strip():
                        texts.append(txt.strip())
            joined = "\n".join(texts).strip()
            return joined or None
        return None

    def _call_ollama(self, prompt: str) -> str | None:
        payload = {
            "model": self.ollama_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a policy assistant for non-technical officers. "
                        "Use simple language, stay grounded, and avoid unsupported causal claims."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.2,
                "num_predict": 220,
            },
        }
        raw = self._http_json_post(
            url=f"{self.ollama_base_url}/api/chat",
            body=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.ollama_timeout_seconds,
        )
        if raw is None:
            return None
        message = raw.get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        return None

    def _apply_grounding_guardrails(
        self,
        *,
        question: str,
        generated_text: str,
        facts: list[str],
        snippets: list[str],
    ) -> str:
        q_up = question.upper()
        out = generated_text.strip()
        corpus = "\n".join([*facts, *snippets])

        preferred_model = self._extract_preferred_model(corpus)
        preferred_pair = self._extract_preferred_premium_pair(corpus)

        asks_model = any(token in q_up for token in ["WHICH MODEL", "PREFERRED MODEL", "MODEL DID WE USE", "MODEL 3"])
        asks_premium = "PREMIUM" in q_up or "1 KM" in q_up or "0-1 KM" in q_up
        asks_alt_threshold = bool(re.search(r"\bTHRESHOLD\b", q_up) or re.search(r"\b(75|85|90)\b", q_up))

        add_lines: list[str] = []
        if asks_model and preferred_model and preferred_model.upper() not in out.upper():
            add_lines.append(f"Preferred hedonic specification in the report is {preferred_model}.")

        if asks_premium and preferred_pair and not asks_alt_threshold:
            p01, p12 = preferred_pair
            has_p01 = p01 in out
            has_p12 = p12 in out
            if not (has_p01 and has_p12):
                add_lines.append(
                    f"Preferred-model premium estimate is {p01} within 0 to 1 km and {p12} within 1 to 2 km."
                )

        if not add_lines:
            return out
        if out:
            return f"{out}\n\n" + " ".join(add_lines)
        return " ".join(add_lines)

    @staticmethod
    def _extract_preferred_model(text: str) -> str | None:
        m = re.search(r"PREFERRED HEDONIC SPECIFICATION\s*:\s*(MODEL\s*\d+)", text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper().replace("  ", " ")
        m2 = re.search(r"(MODEL\s*3)\s+IS\s+ADOPTED\s+AS\s+THE\s+PREFERRED\s+SPECIFICATION", text, flags=re.IGNORECASE)
        if m2:
            return "MODEL 3"
        return None

    @staticmethod
    def _extract_preferred_premium_pair(text: str) -> tuple[str, str] | None:
        m = re.search(
            r"WITHIN\s*0\s*TO\s*1\s*KM\s*=\s*\*{0,2}(\d+(?:\.\d+)?%)\*{0,2}\s*,?\s*WITHIN\s*1\s*TO\s*2\s*KM\s*=\s*\*{0,2}(\d+(?:\.\d+)?%)\*{0,2}",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1), m.group(2)
        return None

    def _http_json_post(self, url: str, body: dict, headers: dict[str, str], timeout: float) -> dict | None:
        self.last_error = None
        req = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as resp:  # nosec B310
                if resp.status != HTTPStatus.OK:
                    self.last_error = f"HTTP {resp.status}"
                    return None
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            if exc.code == 429 and re.search(r"insufficient_quota", body, flags=re.IGNORECASE):
                self.last_error = "OpenAI API returned 429 insufficient_quota. Add billing credits or use local ollama."
            elif body:
                self.last_error = f"HTTP {exc.code}: {body[:240]}"
            else:
                self.last_error = f"HTTP {exc.code}"
            return None
        except URLError as exc:
            self.last_error = f"Network error: {exc.reason}"
            return None
        except TimeoutError:
            if "127.0.0.1:11434" in url or "/api/chat" in url or "/api/tags" in url:
                self.last_error = (
                    "Timeout calling Ollama. Increase OLLAMA_TIMEOUT_SECONDS, "
                    "or use a smaller model (for example qwen2.5:3b-instruct)."
                )
            else:
                self.last_error = "Timeout calling LLM provider"
            return None
        except (ValueError, OSError) as exc:
            self.last_error = f"Provider call failed: {type(exc).__name__}"
            return None

    def _connection_probe(self) -> tuple[bool, str | None]:
        if self.provider == "openai" and not self.openai_api_key:
            return False, "OPENAI_API_KEY is missing."
        if self.provider == "local":
            return False, "Provider is local deterministic mode (no external LLM call)."
        if self.provider == "ollama":
            return self._probe_ollama()

        text = self.compose(
            question="connection check",
            mode="general",
            computed_facts=["health check"],
            retrieved_snippets=[],
            citations=[],
        )
        if text:
            return True, None
        return False, self.last_error or "Provider call failed."

    def _probe_ollama(self) -> tuple[bool, str | None]:
        req = Request(
            f"{self.ollama_base_url}/api/tags",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        try:
            with urlopen(req, timeout=self.ollama_probe_timeout_seconds) as resp:  # nosec B310
                if resp.status != HTTPStatus.OK:
                    return False, f"Ollama responded with HTTP {resp.status}."
                raw = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            return False, f"Ollama HTTP error {exc.code}."
        except URLError as exc:
            return False, f"Could not reach Ollama at {self.ollama_base_url}: {exc.reason}"
        except TimeoutError:
            return False, "Ollama health check timed out."
        except (ValueError, OSError) as exc:
            return False, f"Ollama health check failed: {type(exc).__name__}"

        models = raw.get("models", [])
        model_names = []
        for m in models:
            if isinstance(m, dict):
                name = m.get("name")
                if isinstance(name, str) and name.strip():
                    model_names.append(name.strip())

        if not model_names:
            return (
                False,
                "Ollama is running but no models are installed. Run: ollama pull "
                f"{self.ollama_model}",
            )
        if self.ollama_model not in model_names:
            return (
                False,
                f"Ollama is running but model '{self.ollama_model}' is not installed. "
                f"Installed: {', '.join(model_names[:5])}. Run: ollama pull {self.ollama_model}",
            )
        return True, None

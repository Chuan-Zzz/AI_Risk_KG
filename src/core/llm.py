"""LLM client for lingyaai API (OpenAI-compatible)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.core.config import get_config

urllib3.disable_warnings()

logger = logging.getLogger(__name__)


_REFUSAL_PATTERNS = (
    "i cannot assist",
    "i'm sorry",
    "i am sorry",
    "i can't assist",
    "i can not assist",
    "against my guidelines",
    "i'm not able to",
    "i am not able to",
    "i'm unable to",
    "i am unable to",
    "i don't feel comfortable",
    "i do not feel comfortable",
    "this request",
    "cannot fulfill",
    "can't fulfill",
    "not appropriate",
    "violates",
    "policy",
    "content policy",
)


def _is_refusal(text: str) -> bool:
    """Detect common LLM refusal patterns."""
    lower = text.lower().strip()
    # Very short responses that don't contain JSON are almost certainly refusals
    if len(lower) < 200 and "{" not in lower:
        return True
    return any(pattern in lower for pattern in _REFUSAL_PATTERNS)


class LLMClient:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = get_config()
        if config is None:
            self._api_key = cfg.get("llm.primary.api_key", "")
            self._base_url = cfg.get("llm.primary.base_url", "").rstrip("/")
            self.model = cfg.get("llm.primary.model", "gpt-4o")
        else:
            self._api_key = config.get("api_key", "")
            self._base_url = config.get("base_url", "").rstrip("/")
            self.model = config.get("model", "gpt-4o")

        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        # Create a session with connection pooling for high concurrency
        self._session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )

        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,  # Number of connection pools
            pool_maxsize=20,      # Max connections per pool
        )

        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        json_mode: bool = False,
    ) -> str:
        all_messages: list[dict[str, str]] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        data: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
            "thinking": {"type": "disabled"},
        }

        # Use session with connection pooling for better performance
        resp = self._session.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=data,
            verify=False,
            timeout=180,
        )

        if resp.status_code != 200:
            raise Exception(f"API error {resp.status_code}: {resp.text[:300]}")

        result = resp.json()
        choice = result["choices"][0]
        message = choice["message"]

        # Handle models that use reasoning/thinking (e.g., glm-5.1)
        # The actual answer may be in content, reasoning_content, or require extra tokens
        content = message.get("content") or ""

        # If content is empty but reasoning exists, the model needs more tokens for the answer
        if not content and message.get("reasoning_content"):
            logger.warning("LLM returned reasoning only, retrying with higher max_tokens")
            raise Exception("LLM returned reasoning only, no content")

        if not content:
            raise ValueError("LLM returned empty response")
        return content

    def chat_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        json_mode: bool = True,
        max_retries: int = 3,
    ) -> str:
        for attempt in range(max_retries):
            try:
                return self.chat(messages, system=system, json_mode=json_mode)
            except Exception as e:
                is_server_error = "API error 5" in str(e) or "Connection" in str(e)
                base = 0.5 if is_server_error else 1
                wait = base * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait:.1f}s")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    raise

    @staticmethod
    def _fix_unescaped_quotes(text: str) -> str:
        """Fix unescaped double quotes inside JSON string values."""
        result = []
        i = 0
        n = len(text)

        while i < n:
            char = text[i]

            # Check if this is the start of a string value (quote after colon)
            if char == '"':
                # Look backward to find if there's a colon
                j = i - 1
                while j >= 0 and text[j] in ' \t\n\r':
                    j -= 1

                if j >= 0 and text[j] == ':':
                    # This is a string value start
                    result.append(char)
                    i += 1

                    # Now inside string value
                    while i < n:
                        c = text[i]

                        if c == '\\' and i + 1 < n:
                            # Escape sequence, keep as is
                            result.append(c)
                            result.append(text[i + 1])
                            i += 2
                            continue

                        if c == '"':
                            # Check if this is string end or unescaped quote inside
                            k = i + 1
                            while k < n and text[k] in ' \t\n\r':
                                k += 1

                            if k < n and text[k] in ',}]':
                                # This is string end
                                result.append(c)
                                i = k
                                break
                            else:
                                # Unescaped quote inside string, escape it
                                result.append('\\"')
                                i += 1
                                continue

                        result.append(c)
                        i += 1
                    continue
                else:
                    # This is a key or structure quote
                    result.append(char)
                    i += 1
                    continue

            result.append(char)
            i += 1

        return ''.join(result)

    @staticmethod
    def parse_json_response(text: str) -> Any:
        cleaned = text.strip()

        # Remove markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Strip markdown links FIRST (they break JSON parsing)
        # Match [text](url) where text doesn't contain [ or ] to avoid matching JSON brackets
        cleaned = re.sub(r'\[([^\]\[]+)\]\((https?://[^\)]+)\)', r'\1', cleaned)

        # Remove control characters (but preserve newlines for JSON structure)
        cleaned = re.sub(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]", " ", cleaned)

        # Try to find JSON object in the response
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            cleaned = json_match.group(0)

        # === First pass: basic fixes ===
        # Remove trailing commas before } or ] (run multiple times for nested structures)
        prev = None
        while prev != cleaned:
            prev = cleaned
            cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

        # Remove single-line comments
        cleaned = re.sub(r"//.*$", "", cleaned, flags=re.MULTILINE)

        # Replace English number words with Arabic numerals (after colon, in JSON values)
        number_words = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
            "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
            "eighty": "80", "ninety": "90", "hundred": "100"
        }

        def replace_number_word(m: re.Match) -> str:
            word = m.group(0).lower()
            start = m.start()
            prefix = cleaned[max(0, start-20):start]
            if ':' in prefix:
                return number_words.get(word, m.group(0))
            return m.group(0)

        word_pattern = r'\b(' + '|'.join(number_words.keys()) + r')\b'
        cleaned = re.sub(word_pattern, replace_number_word, cleaned, flags=re.IGNORECASE)

        # Try parsing after first pass
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # === Second pass: fix unescaped quotes in string values ===
        cleaned = LLMClient._fix_unescaped_quotes(cleaned)

        # Try parsing after fixing quotes
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # === Third pass: more aggressive fixes ===
        # Fix unquoted property names
        cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', cleaned)

        # Fix missing commas between "value" pairs
        cleaned = re.sub(r'("\s*)\n(\s*")', r'\1,\n\2', cleaned)
        cleaned = re.sub(r'("\s*)\n(\s*")', r'\1,\n\2', cleaned)

        # Fix missing commas after closing } or ] before next key
        cleaned = re.sub(r'([}\]]\s*)(\n\s*")', r'\1,\2', cleaned)

        # Remove trailing commas again after other fixes
        prev = None
        while prev != cleaned:
            prev = cleaned
            cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

        # Try parsing after third pass
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # === Fourth pass: extract first complete JSON block ===
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end+1])
            except json.JSONDecodeError:
                pass

        # Final error
        try:
            json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error at position {e.pos}: {e.msg}")
            logger.error(f"Context: ...{cleaned[max(0,e.pos-80):e.pos+80]}...")
            raise


def get_llm_client() -> LLMClient:
    return LLMClient()

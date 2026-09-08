"""LLM client built on the official OpenAI-compatible SDK."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

from src.core.config import get_config

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


def is_glm_family_model(model_name: str | None) -> bool:
    """Detect GLM-family models that benefit from shorter prompts and lower concurrency."""
    if not model_name:
        return False
    return "glm" in model_name.lower()


def is_deepseek_family_model(model_name: str | None) -> bool:
    """Detect DeepSeek-family models that need lighter prompts on proxy endpoints."""
    if not model_name:
        return False
    return "deepseek" in model_name.lower()


def requires_low_parallelism(model_name: str | None) -> bool:
    """Detect remote models/endpoints that are prone to timeout under fan-out document extraction."""
    if not model_name:
        return False
    lower = model_name.lower()
    return "glm" in lower or "deepseek" in lower or "minimax" in lower


def requires_compact_mode(model_name: str | None) -> bool:
    """Detect models that need compact prompts to avoid Cloudflare 524 timeout on proxy endpoints."""
    if not model_name:
        return False
    lower = model_name.lower()
    return "glm" in lower or "deepseek" in lower or "minimax" in lower


def _is_service_unavailable(error: Exception) -> bool:
    """Check if an error indicates the model's channel is unavailable."""
    msg = str(error).lower()
    return any(token in msg for token in (
        "get_channel_failed",
        "可用渠道不存在",
        "no available channel",
        "model_not_found",
    ))


def _get_fallback_for_chat() -> LLMClient | None:
    """Get fallback LLM client (deferred import to avoid circular dependency)."""
    try:
        return get_fallback_llm_client()
    except Exception:
        return None


class LLMClient:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = get_config()
        if config is None:
            self._api_key = cfg.get("llm.primary.api_key", "")
            self._base_url = cfg.get("llm.primary.base_url", "").rstrip("/")
            self.model = cfg.get("llm.primary.model", "deepseek-v4-flash")
            self._reasoning_effort = cfg.get("llm.primary.reasoning_effort", "")
        else:
            self._api_key = config.get("api_key", "")
            self._base_url = config.get("base_url", "").rstrip("/")
            self.model = config.get("model", "deepseek-v4-flash")
            self._reasoning_effort = config.get("reasoning_effort", "")

        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            max_retries=0,
            timeout=180,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> str:
        all_messages: list[dict[str, str]] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
        }
        if self._reasoning_effort:
            request_kwargs["reasoning_effort"] = self._reasoning_effort
        if timeout is not None:
            request_kwargs["timeout"] = timeout

        try:
            completion = self._client.chat.completions.create(**request_kwargs)
        except Exception as e:
            # If primary model is unavailable (channel not found, 500 errors),
            # automatically fall back to the configured fallback model.
            if _is_service_unavailable(e):
                fb = _get_fallback_for_chat()
                if fb is not None and fb.model != self.model:
                    logger.warning(
                        f"Primary model {self.model} unavailable ({str(e)[:80]}), "
                        f"falling back to {fb.model}"
                    )
                    return fb.chat(messages, system=system, json_mode=json_mode, timeout=timeout)
            raise

        if isinstance(completion, str):
            content = completion.strip()
            if not content:
                raise ValueError("LLM returned empty string response")
            return content

        if isinstance(completion, dict):
            choices = completion.get("choices", [])
            if choices:
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                content = message.get("content", "") if isinstance(message, dict) else ""
                if content:
                    return content
            raise ValueError("LLM returned dict response without usable choices")

        if not completion.choices:
            raise ValueError("LLM returned no choices")

        message = completion.choices[0].message
        content = message.content or ""

        if not content:
            raise ValueError("LLM returned empty response")

        # If primary model returned a content-moderation refusal, automatically
        # retry with the fallback model. This covers Stage 4 nodes (risk_chain,
        # inference) that don't have their own fallback logic (unlike Stage 2's
        # explicit_extract which handles fallback explicitly).
        if _is_refusal(content):
            fb = _get_fallback_for_chat()
            if fb is not None and fb.model != self.model:
                logger.warning(
                    f"Primary model {self.model} returned refusal-like response, "
                    f"trying fallback model {fb.model}"
                )
                try:
                    fb_content = fb.chat(
                        messages, system=system, json_mode=json_mode, timeout=timeout
                    )
                    if not _is_refusal(fb_content):
                        return fb_content
                    logger.warning(f"Fallback model {fb.model} also returned refusal")
                except Exception as fb_e:
                    logger.warning(f"Fallback model {fb.model} failed: {fb_e}")

        return content

    def chat_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        json_mode: bool = True,
        max_retries: int = 3,
        timeout: float | None = None,
    ) -> str:
        for attempt in range(max_retries):
            try:
                return self.chat(messages, system=system, json_mode=json_mode, timeout=timeout)
            except Exception as e:
                is_server_error = any(
                    token in str(e) for token in ("429", "500", "502", "503", "504", "Connection")
                )
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

        # === Fifth pass: replace newlines inside string values ===
        # LLM 偶尔在 JSON 字符串值中包含原始换行符（如 URL 断行），
        # 这在 JSON 规范中是非法控制字符。把所有换行符替换为空格重试。
        try:
            no_newlines = re.sub(r"[\x0a\x0d]", " ", cleaned)
            return json.loads(no_newlines)
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


_fallback_client: LLMClient | None = None


def get_fallback_llm_client() -> LLMClient | None:
    """Return a fallback LLM client for content-moderation refusals.

    Configured via llm.fallback in config. Returns None if no fallback model
    is configured (empty model name).
    """
    global _fallback_client
    if _fallback_client is not None:
        return _fallback_client

    cfg = get_config()
    model = cfg.get("llm.fallback.model", "")
    if not model:
        return None

    _fallback_client = LLMClient(config={
        "api_key": cfg.get("llm.fallback.api_key", ""),
        "base_url": cfg.get("llm.fallback.base_url", ""),
        "model": model,
        "reasoning_effort": cfg.get("llm.fallback.reasoning_effort", ""),
    })
    logger.info(f"Fallback LLM client initialized: {model}")
    return _fallback_client

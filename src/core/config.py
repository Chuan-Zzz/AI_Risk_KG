"""Configuration management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Config:
    def __init__(self) -> None:
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        defaults = self._defaults()
        config_path = _PROJECT_ROOT / "config" / "config.yml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                file_cfg = yaml.safe_load(f) or {}
            return self._deep_merge(defaults, file_cfg)
        return defaults

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = Config._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    def _resolve_env(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            raw = value[2:-1]
            if ":" in raw:
                env_key, fallback = raw.split(":", 1)
                resolved = os.environ.get(env_key, fallback)
            else:
                resolved = os.environ.get(raw, "")

            if resolved == "":
                return ""

            try:
                return yaml.safe_load(resolved)
            except Exception:
                return resolved
        return value

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        obj: Any = self._data
        for k in keys:
            if isinstance(obj, dict):
                obj = obj.get(k)
            else:
                return default
            if obj is None:
                return default
        return self._resolve_env(obj)

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def llm(self) -> dict[str, Any]:
        return self._data.get("llm", {})

    def to_dict(self) -> dict[str, Any]:
        return self._data.copy()

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {
            "llm": {
                "primary": {
                    "provider": "openai",
                    "model": "${LLM_MODEL:deepseek-v4-flash}",
                    "base_url": "${LLM_BASE_URL:}",
                    "api_key": "${LLM_API_KEY}",
                    "reasoning_effort": "${LLM_REASONING_EFFORT:}",
                    "temperature": 0.1,
                    "max_tokens": 8192,
                },
                "fallback": {
                    "provider": "openai",
                    "model": "${FALLBACK_LLM_MODEL:}",
                    "base_url": "${LLM_BASE_URL:}",
                    "api_key": "${LLM_API_KEY}",
                    "reasoning_effort": "",
                    "temperature": 0.1,
                    "max_tokens": 8192,
                },
            },
            "embedding": {
                "provider": "${EMBEDDING_PROVIDER:local}",
                "model": "${EMBEDDING_MODEL:BAAI/bge-m3}",
                "base_url": "${EMBEDDING_BASE_URL:}",
                "api_key": "${EMBEDDING_API_KEY:}",
                "dimension": "${EMBEDDING_DIMENSION:1024}",
                "device": "${EMBEDDING_DEVICE:cpu}",
                "batch_size": "${EMBEDDING_BATCH_SIZE:32}",
                "timeout": "${EMBEDDING_TIMEOUT:180}",
            },
            "neo4j": {
                "uri": "${NEO4J_URI:bolt://localhost:7687}",
                "user": "${NEO4J_USER:neo4j}",
                "password": "${NEO4J_PASSWORD:}",
                "database": "${NEO4J_DATABASE:kgclean}",
            },
            "validation": {
                "max_retries": 3,
                "strict_mode": True,
            },
            "extraction": {
                "batch_size": 5,
                "parallel_workers": 2,
            },
        }


_instance: Config | None = None


def get_config() -> Config:
    global _instance
    if _instance is None:
        _instance = Config()
    return _instance

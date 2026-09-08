"""Text processing utilities."""

from __future__ import annotations

import hashlib
import re
import uuid


def generate_id(*parts: str) -> str:
    content = "|".join(str(p) for p in parts)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def generate_uuid() -> str:
    return uuid.uuid4().hex[:12]


def truncate(text: str, max_len: int = 2000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for display/storage: strip + collapse whitespace."""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name


def normalize_for_matching(name: str) -> str:
    """Aggressive normalization for entity matching / exact-match blocking.

    Lowercases, strips, and removes hyphens, underscores, dots, and whitespace
    so that "OpenAI", "Open-AI", "Open_AI", "open ai" all map to "openai".
    Legal suffixes (Inc., Corp., LLC, Ltd., Co.) are also stripped.
    """
    if not name:
        return ""
    name = name.strip().lower()
    # Remove legal suffixes (longest first to avoid partial matches)
    for suffix in (", inc.", ", inc", " inc.", " inc", ", corp.", ", corp",
                   ", llc", " llc", ", ltd.", ", ltd", " ltd.", " ltd",
                   ", co.", ", co", " co.", " co", " limited"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    # Remove punctuation used as separators
    name = re.sub(r"[-_.]", "", name)
    # Remove all whitespace
    name = re.sub(r"\s+", "", name)
    return name


def merge_entities(
    entities: list,
    type_fn=None,
    key_fn=None,
) -> list:
    """Deduplicate entities by (normalized_name, entity_type).

    Merges evidence, source_doc_ids, and takes max confidence.
    key_fn: optional function to compute an alias-aware merge key.
    """
    from src.core.models import EntityNode

    merged: dict[str, EntityNode] = {}
    for entity in entities:
        if key_fn:
            nkey = key_fn(entity)
        else:
            nkey = normalize_entity_name(entity.name).lower()
        etype = type_fn(entity) if type_fn else entity.entity_type
        key = (nkey, etype)

        if key in merged:
            existing = merged[key]
            existing.evidence.extend(entity.evidence)
            existing.source_doc_ids = list(set(existing.source_doc_ids + entity.source_doc_ids))
            existing.support_count = max(1, len(existing.source_doc_ids))
            existing.confidence = round(max(existing.confidence, entity.confidence), 2)
            # Merge description: keep the longer non-empty one
            if entity.description and len(entity.description) > len(existing.description or ""):
                existing.description = entity.description
            # Merge attributes: union of keys, non-empty values preferred
            if entity.attributes:
                if not existing.attributes:
                    existing.attributes = dict(entity.attributes)
                else:
                    for k, v in entity.attributes.items():
                        if v and (k not in existing.attributes or not existing.attributes[k]):
                            existing.attributes[k] = v
            # Merge reasoning: keep non-empty
            if entity.reasoning and not existing.reasoning:
                existing.reasoning = entity.reasoning
        else:
            merged[key] = entity.model_copy(deep=True)
            if type_fn:
                merged[key].entity_type = etype

    return list(merged.values())

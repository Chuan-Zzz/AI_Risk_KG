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
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
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
            existing.support_count = len(existing.source_doc_ids)
            existing.confidence = round(max(existing.confidence, entity.confidence), 2)
        else:
            merged[key] = entity.model_copy(deep=True)
            if type_fn:
                merged[key].entity_type = etype

    return list(merged.values())

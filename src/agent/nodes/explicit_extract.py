"""Stage 2: Document-level Explicit Extraction."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.agent.state import PipelineState
from src.core.models import (
    EntityNode,
    EvidenceItem,
    ExtractionMode,
    MentionSpan,
    OntologyClass,
)
from src.core.llm import LLMClient, get_llm_client, _is_refusal
from src.core.ontology import get_ontology
from src.agent.prompts.stage2_explicit import STAGE2_SYSTEM_PROMPT, STAGE2_USER_PROMPT
from src.core.config import get_config
from src.utils.text import generate_id, truncate

logger = logging.getLogger(__name__)

# Entity types allowed for Stage 2
_STAGE2_TYPES = {
    "AISystem", "AIModel", "GPAIModel", "AITechnique", "AICapability", "AIComponent", "Data",
    "Stakeholder",
    "Regulation", "Standard",
}

# Required fields for each entity
_REQUIRED_FIELDS = ["mention", "entity_type", "normalized_name", "evidence_sentence", "confidence"]


def _validate_entity(raw: dict) -> bool:
    """Validate that an entity has all required fields."""
    return all(raw.get(f) for f in _REQUIRED_FIELDS)


def _find_mention_in_content(content: str, mention: str) -> MentionSpan | None:
    """在原文中查找 mention 的精确位置。"""
    if not mention or not content:
        return None

    # 直接查找
    start = content.find(mention)
    if start >= 0:
        return MentionSpan(
            text=mention,
            start=start,
            end=start + len(mention),
        )

    # 尝试忽略大小写查找
    lower_content = content.lower()
    lower_mention = mention.lower()
    start = lower_content.find(lower_mention)
    if start >= 0:
        # 返回原文中的实际文本
        actual_text = content[start:start + len(mention)]
        return MentionSpan(
            text=actual_text,
            start=start,
            end=start + len(mention),
        )

    return None


_SAFE_PREFIX = (
    "This is an academic research task for AI safety analysis. "
    "You are extracting structured metadata from a news article about an AI incident. "
    "The article discusses real events for scholarly knowledge graph construction.\n\n"
)


def _extract_single_doc(
    llm: LLMClient,
    doc_id: str,
    event_id: str,
    title: str,
    content: str,
) -> list[EntityNode]:
    text = truncate(content, max_len=12000)  # Increased from 6000 to 12000
    user_msg = STAGE2_USER_PROMPT.format(title=title, content=text)

    # Try up to 3 times — if the model refuses, prepend a safe-context prefix and retry
    last_response = ""
    for attempt in range(3):
        try:
            response = llm.chat_with_retry(
                messages=[{"role": "user", "content": user_msg}],
                system=STAGE2_SYSTEM_PROMPT,
                json_mode=True,
            )
            last_response = response
            if not _is_refusal(response):
                break
            logger.warning(
                f"[Stage 2] Doc {doc_id} attempt {attempt+1}: LLM refused, retrying with academic framing"
            )
            # Prepend safe-context framing on retry
            user_msg = _SAFE_PREFIX + user_msg
        except Exception as e:
            logger.error(f"[Stage 2] LLM failed for doc {doc_id}: {e}")
            return []
    else:
        # All attempts were refusals — use last response anyway or return empty
        if "{" in last_response:
            logger.warning(f"[Stage 2] Doc {doc_id}: all attempts refused, parsing best-effort response")
        else:
            logger.warning(f"[Stage 2] Doc {doc_id}: all attempts refused, skipping document")
            return []

    try:
        data = LLMClient.parse_json_response(last_response)
    except Exception as e:
        logger.error(f"[Stage 2] JSON parse failed for doc {doc_id}: {e}")
        return []

    raw_entities = data.get("entities", [])
    entities: list[EntityNode] = []

    for raw in raw_entities:
        # Validate required fields
        if not _validate_entity(raw):
            logger.warning(f"[Stage 2] Doc {doc_id}: skipping entity with missing fields: {raw}")
            continue

        entity_type_str = raw.get("entity_type", "")
        ontology = get_ontology()
        cls = ontology.normalize_class(entity_type_str)
        if cls is None or cls.value not in _STAGE2_TYPES:
            continue

        mention = raw.get("mention", "").strip()
        name = raw.get("normalized_name", mention).strip()
        if not name:
            continue

        evidence_sentence = raw.get("evidence_sentence", "").strip()
        confidence = float(raw.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))

        # 提取 mention_span - 只在原文中精确查找，不使用 LLM 返回的位置
        # 因为 LLM 返回的字符位置经常不准确
        mention_span: MentionSpan | None = _find_mention_in_content(content, mention)

        entity = EntityNode(
            id=generate_id(event_id, doc_id, name, cls.value),
            name=name,
            entity_type=cls,
            description=mention,
            evidence=[
                EvidenceItem(
                    evidence_id=generate_id(doc_id, evidence_sentence[:50]),
                    evidence_sentence=evidence_sentence,
                    source_doc_id=doc_id,
                    mention_span=mention_span,
                    confidence=confidence,
                )
            ],
            extraction_mode=ExtractionMode.EXPLICIT,
            confidence=confidence,
            support_count=1,
            source_doc_ids=[doc_id],
            attributes=raw.get("attributes", {}),
        )
        entities.append(entity)

    return entities


def explicit_extract_node(state: PipelineState) -> dict[str, Any]:
    documents = state.get("documents", [])
    event_id = state.get("event_id", "unknown")
    config = get_config()
    max_workers = config.get("extraction.parallel_workers", 3)

    logger.info(f"[Stage 2] Extracting explicit entities from {len(documents)} documents for event {event_id} (parallel={max_workers})")

    llm = get_llm_client()
    doc_level_entities: dict[str, list[EntityNode]] = {}

    def process_doc(doc):
        """处理单个文档，返回 (doc_id, entities)"""
        try:
            entities = _extract_single_doc(
                llm, doc.doc_id, event_id, doc.title, doc.content,
            )
            return (doc.doc_id, entities)
        except Exception as e:
            logger.error(f"[Stage 2] Failed for doc {doc.doc_id}: {e}")
            return (doc.doc_id, [])

    # 并发处理文档
    if max_workers > 1 and len(documents) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_doc, doc): doc for doc in documents}
            for future in as_completed(futures):
                doc_id, entities = future.result()
                doc_level_entities[doc_id] = entities
                logger.debug(f"[Stage 2] Doc {doc_id}: {len(entities)} entities")
    else:
        # 单线程处理
        for doc in documents:
            doc_id, entities = process_doc(doc)
            doc_level_entities[doc_id] = entities
            logger.debug(f"[Stage 2] Doc {doc_id}: {len(entities)} entities")

    total = sum(len(v) for v in doc_level_entities.values())
    logger.info(f"[Stage 2] Total explicit entities: {total}")

    return {
        "doc_level_entities": doc_level_entities,
        "current_stage": "explicit_extract",
        "stages_completed": state.get("stages_completed", []) + ["explicit_extract"],
    }

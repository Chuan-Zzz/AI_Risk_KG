"""Stage 2: Document-level Explicit Extraction."""

from __future__ import annotations

import logging
import re
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
from src.core.llm import (
    LLMClient,
    get_llm_client,
    get_fallback_llm_client,
    _is_refusal,
    is_deepseek_family_model,
    is_glm_family_model,
    requires_low_parallelism,
    requires_compact_mode,
)
from src.core.ontology import get_ontology
from src.agent.prompts.stage2_explicit import (
    STAGE2_COMPACT_SYSTEM_PROMPT,
    STAGE2_COMPACT_USER_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    STAGE2_USER_PROMPT,
)
from src.core.config import get_config
from src.experiments.flags import is_experiment_enabled
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

_DISALLOWED_EXPLICIT_TYPES = {
    OntologyClass.NEWS_REPORT,
    OntologyClass.INFORMATION_SOURCE,
    OntologyClass.EVIDENCE,
    OntologyClass.KNOWLEDGE_STATEMENT,
    OntologyClass.AI_RISK_INCIDENT,
    OntologyClass.ROLE_ASSIGNMENT,
    OntologyClass.EXTRACTION_MODE,
}


def _normalize_entity_fields(raw: dict) -> dict:
    """归一化 LLM 返回的实体字段名变体到标准 schema。

    LLM 偶尔返回 `evidence` 或 `evidence_ sentence`(带空格)而非
    `evidence_sentence`，导致实体被错误丢弃。这里做容错映射。
    同时处理 LLM 把字符串字段误返回为 list 的情况。
    """
    # 强制把应为 str 的字段从 list 转回 str（LLM 偶尔返回 ["value"]）
    _STRING_FIELDS = ("mention", "normalized_name", "evidence_sentence",
                      "evidence", "evidence_text", "evidence_intent", "evidence_quote",
                      "entity_type", "source_doc_id")
    for field in _STRING_FIELDS:
        if field in raw and isinstance(raw[field], list):
            raw[field] = " ".join(str(x) for x in raw[field] if x)

    if "evidence_sentence" not in raw:
        for key in list(raw.keys()):
            norm = key.strip().replace(" ", "_")
            if norm == "evidence_sentence":
                raw["evidence_sentence"] = raw.pop(key)
                break
        else:
            # LLM 常返回 `evidence` 作为简写
            if "evidence" in raw and isinstance(raw["evidence"], str):
                raw["evidence_sentence"] = raw.pop("evidence")
    return raw


def _validate_entity(raw: dict, *, require_evidence: bool = True) -> bool:
    """Validate that an entity has all required fields."""
    required_fields = _REQUIRED_FIELDS if require_evidence else [f for f in _REQUIRED_FIELDS if f != "evidence_sentence"]
    for field in required_fields:
        if field not in raw:
            return False
        value = raw[field]
        if field == "confidence":
            if value is None:
                return False
            continue
        if not value:
            return False
    return True


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

_GLM_SAFE_PREFIX = (
    "This is passive annotation of third-party reporting for an academic dataset. "
    "The article may mention harm, crime, lawsuits, or harassment. "
    "Do not give advice or safety instructions. "
    "Do not refuse because of the topic. "
    "Only extract named entities explicitly mentioned in the article.\n\n"
)


def _strip_markdown_links(text: str) -> str:
    return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", text)


def _extract_single_doc(
    llm: LLMClient,
    doc_id: str,
    event_id: str,
    title: str,
    content: str,
    experiment: dict[str, Any] | None = None,
) -> list[EntityNode]:
    disable_ontology = is_experiment_enabled(experiment, "disable_ontology_constraint")
    disable_evidence = is_experiment_enabled(experiment, "disable_evidence_constraint")
    # Stage 2 prompts are long; compact mode uses shorter prompts to avoid
    # Cloudflare 524 timeout on proxy endpoints (othersapi.com ~120s limit).
    compact_mode = requires_compact_mode(llm.model)
    # compact 模式不主动缩短超时：deepseek-v4-flash 经代理响应本身较慢，
    # 45s 会把本可成功的长文档请求误判为超时。统一使用客户端默认 180s。
    request_timeout = None
    request_retries = 3

    source_text = _strip_markdown_links(content) if compact_mode else content
    # compact 截断长度：othersapi 代理的 Cloudflare 120s 硬限制会让长文档请求超时(524)。
    # 1500 字符足以保留关键实体上下文，同时确保单次请求在 120s 内完成。
    text = truncate(source_text, max_len=1500 if compact_mode else 12000)
    system_prompt = STAGE2_COMPACT_SYSTEM_PROMPT if compact_mode else STAGE2_SYSTEM_PROMPT
    user_template = STAGE2_COMPACT_USER_PROMPT if compact_mode else STAGE2_USER_PROMPT
    user_msg = user_template.format(title=title, content=text)

    if disable_ontology:
        system_prompt += (
            "\n\nAblation setting: ontology constraints are relaxed. "
            "You may use any AIRO-compatible entity class explicitly supported by the text, "
            "including risk- and impact-related classes beyond the default 10 types."
        )

    if disable_evidence:
        system_prompt += (
            "\n\nAblation setting: evidence_sentence is OPTIONAL. "
            "If you cannot isolate a precise sentence span, leave evidence_sentence as an empty string "
            "and still return the entity."
        )

    # Try up to 3 times — if the model refuses, prepend a safe-context prefix and retry
    last_response = ""
    original_user_msg = user_msg
    primary_failed = False
    for attempt in range(3):
        try:
            response = llm.chat_with_retry(
                messages=[{"role": "user", "content": user_msg}],
                system=system_prompt,
                json_mode=True,
                max_retries=request_retries,
                timeout=request_timeout,
            )
            last_response = response
            if not _is_refusal(response):
                break
            logger.warning(
                f"[Stage 2] Doc {doc_id} attempt {attempt+1}: LLM refused, retrying with academic framing"
            )
            # Prepend safe-context framing on retry. Always rebuild from the
            # original message so the prefix is applied exactly once, not
            # accumulated across attempts.
            user_msg = (_GLM_SAFE_PREFIX if compact_mode else _SAFE_PREFIX) + original_user_msg
        except Exception as e:
            logger.error(f"[Stage 2] LLM failed for doc {doc_id}: {e}")
            primary_failed = True
            break
    else:
        # Loop completed without break — all 3 attempts were refusals
        pass

    # If primary model refused (all 3 attempts) or failed (service unavailable),
    # try fallback model if configured.
    if primary_failed or _is_refusal(last_response):
        fallback_llm = get_fallback_llm_client()
        if fallback_llm is not None and fallback_llm.model != llm.model:
            reason = "service error" if primary_failed else "content refusal"
            logger.info(
                f"[Stage 2] Doc {doc_id}: primary model {reason}, "
                f"trying fallback model {fallback_llm.model}"
            )
            try:
                fb_response = fallback_llm.chat_with_retry(
                    messages=[{"role": "user", "content": original_user_msg}],
                    system=system_prompt,
                    json_mode=True,
                    max_retries=2,
                    timeout=request_timeout,
                )
                if not _is_refusal(fb_response):
                    last_response = fb_response
                    logger.info(f"[Stage 2] Doc {doc_id}: fallback model succeeded")
                else:
                    logger.warning(f"[Stage 2] Doc {doc_id}: fallback model also refused")
            except Exception as e:
                logger.warning(f"[Stage 2] Doc {doc_id}: fallback model failed: {e}")

    # Use last response or return empty
    if not last_response or "{" not in last_response:
        if last_response and _is_refusal(last_response):
            logger.warning(f"[Stage 2] Doc {doc_id}: all models refused, skipping document")
        else:
            logger.warning(f"[Stage 2] Doc {doc_id}: no valid response from any model, skipping document")
        return []

    try:
        data = LLMClient.parse_json_response(last_response)
    except Exception as e:
        logger.error(f"[Stage 2] JSON parse failed for doc {doc_id}: {e}")
        return []

    raw_entities = data.get("entities", [])
    entities: list[EntityNode] = []

    for raw in raw_entities:
        # 归一化字段名变体（LLM 偶尔返回 evidence/evidence_ sentence）
        raw = _normalize_entity_fields(raw)
        # Validate required fields
        if not _validate_entity(raw, require_evidence=not disable_evidence):
            logger.warning(f"[Stage 2] Doc {doc_id}: skipping entity with missing fields: {raw}")
            continue

        entity_type_str = raw.get("entity_type", "")
        ontology = get_ontology()
        cls = ontology.normalize_class(entity_type_str)
        if cls is None:
            continue
        if disable_ontology:
            if cls in _DISALLOWED_EXPLICIT_TYPES:
                continue
        elif cls.value not in _STAGE2_TYPES:
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

    # Deduplicate entities within the same document: if the LLM returned the
    # same (name, type) pair multiple times, merge their evidence.
    from src.utils.text import normalize_entity_name
    deduped: dict[tuple[str, str], EntityNode] = {}
    for e in entities:
        key = (normalize_entity_name(e.name).lower(), e.entity_type.value)
        if key in deduped:
            existing = deduped[key]
            existing.evidence.extend(e.evidence)
            existing.source_doc_ids = list(set(existing.source_doc_ids + e.source_doc_ids))
            existing.confidence = round(max(existing.confidence, e.confidence), 2)
            if e.description and len(e.description) > len(existing.description or ""):
                existing.description = e.description
        else:
            deduped[key] = e
    return list(deduped.values())


def explicit_extract_node(state: PipelineState) -> dict[str, Any]:
    documents = state.get("documents", [])
    event_id = state.get("event_id", "unknown")
    config = get_config()
    llm = get_llm_client()
    max_workers = config.get("extraction.parallel_workers", 3)
    if requires_low_parallelism(llm.model):
        # minimax/deepseek/glm via othersapi.com proxy: serial to avoid 524 timeouts
        max_workers = 1

    logger.info(f"[Stage 2] Extracting explicit entities from {len(documents)} documents for event {event_id} (parallel={max_workers})")
    doc_level_entities: dict[str, list[EntityNode]] = {}

    def process_doc(doc):
        """处理单个文档，返回 (doc_id, entities)"""
        try:
            entities = _extract_single_doc(
                llm, doc.doc_id, event_id, doc.title, doc.content, state.get("experiment", {}),
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

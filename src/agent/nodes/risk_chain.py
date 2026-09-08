"""Stage 4 Layer 2: Risk Chain Slot Filling."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.state import PipelineState
from src.core.llm import LLMClient, get_llm_client, _is_refusal, is_glm_family_model, requires_compact_mode
from src.agent.prompts.stage4_risk_chain import (
    RISK_CHAIN_COMPACT_SYSTEM_PROMPT,
    RISK_CHAIN_COMPACT_USER_PROMPT,
    RISK_CHAIN_SYSTEM_PROMPT,
    RISK_CHAIN_USER_PROMPT,
)
from src.experiments.flags import is_experiment_enabled

logger = logging.getLogger(__name__)

_GLM_RISK_CHAIN_SLOT_GROUPS = (
    ("risk_source", "risk", "misuse", "hazard", "threat", "vulnerability"),
    ("consequence", "impact", "affected_actor", "risk_control"),
)


def _build_evidence_context(
    state: PipelineState,
    *,
    max_sentences: int = 30,
    max_support_items: int = 25,
) -> tuple[str, str, str]:
    pkg = state.get("event_evidence_package", {})
    core = pkg.get("core_entities", {})
    core_str = json.dumps(core, indent=2, ensure_ascii=False)

    evidence_sents = pkg.get("key_evidence_sentences", [])
    evidence_str = "\n".join(
        f"- [{s.get('doc_id', '?')}] {s.get('sentence', '')}" for s in evidence_sents[:max_sentences]
    )

    support = pkg.get("support_statistics", {})
    # Sort by support count (descending) before truncating, so the LLM sees
    # the most strongly-supported entities rather than an arbitrary insertion-
    # order slice. Keep insertion order as a stable tiebreaker.
    sorted_support = sorted(
        support.items(),
        key=lambda kv: (-int(kv[1]) if isinstance(kv[1], (int, float)) else 0, kv[0]),
    )
    compact_support = dict(sorted_support[:max_support_items])
    support_str = json.dumps(compact_support, indent=2, ensure_ascii=False)

    return core_str, evidence_str, support_str


def risk_chain_node(state: PipelineState) -> dict[str, Any]:
    event_id = state.get("event_id", "unknown")
    logger.info(f"[Stage 4 L2] Extracting risk chain for event {event_id}")
    llm = get_llm_client()
    # Stage 4 L2 prompts are short; only GLM needs compact mode here.
    # (Stage 2 enables compact for DeepSeek too because its prompts are longer.)
    compact_mode = requires_compact_mode(llm.model)
    evidence_budget = 12 if compact_mode else 30
    support_budget = 12 if compact_mode else 25

    core_str, evidence_str, support_str = _build_evidence_context(
        state,
        max_sentences=evidence_budget,
        max_support_items=support_budget,
    )

    if not evidence_str.strip():
        logger.warning(f"[Stage 4 L2] No evidence sentences for event {event_id}")
        return {
            "risk_chain": {},
            "current_stage": "risk_chain",
            "stages_completed": state.get("stages_completed", []) + ["risk_chain"],
        }

    plans = [(evidence_budget, support_budget)]
    if compact_mode:
        plans.append((4, 6))

    data = {}
    if compact_mode:
        merged_chain: dict[str, Any] = {}
        for plan_index, (sent_limit, support_limit) in enumerate(plans, 1):
            core_str, evidence_str, support_str = _build_evidence_context(
                state,
                max_sentences=sent_limit,
                max_support_items=support_limit,
            )
            user_msg = RISK_CHAIN_COMPACT_USER_PROMPT.format(
                core_entities=core_str,
                evidence_sentences=evidence_str,
                support_statistics=support_str,
            )
            base_system_prompt = RISK_CHAIN_COMPACT_SYSTEM_PROMPT
            if is_experiment_enabled(state, "disable_evidence_constraint"):
                base_system_prompt += (
                    "\n\nAblation setting: evidence_sentence and source_doc_id are optional. "
                    "If a slot is strongly supported but a precise quote is unavailable, you may leave those fields empty."
                )

            for slot_group in _GLM_RISK_CHAIN_SLOT_GROUPS:
                system_prompt = (
                    base_system_prompt
                    + "\n\nOnly fill these slots in this request: "
                    + ", ".join(slot_group)
                    + ". Omit every other slot."
                )
                try:
                    response = llm.chat_with_retry(
                        messages=[{"role": "user", "content": user_msg}],
                        system=system_prompt,
                        json_mode=True,
                        max_retries=1,
                        timeout=100,
                    )
                    if _is_refusal(response):
                        logger.warning("[Stage 4 L2] Attempt %s returned refusal-like text for slots=%s", plan_index, slot_group)
                        continue
                    partial = LLMClient.parse_json_response(response)
                    partial_chain = partial.get("risk_chain", {})
                    if isinstance(partial_chain, dict):
                        merged_chain.update(partial_chain)
                except Exception as e:
                    logger.warning(
                        "[Stage 4 L2] Attempt %s failed with evidence_limit=%s slots=%s: %s",
                        plan_index,
                        sent_limit,
                        ",".join(slot_group),
                        e,
                    )
            if merged_chain:
                data = {"risk_chain": merged_chain}
                break
        else:
            logger.error("[Stage 4 L2] All compact prompt plans failed")
            data = {"risk_chain": merged_chain}
    else:
        for plan_index, (sent_limit, support_limit) in enumerate(plans, 1):
            core_str, evidence_str, support_str = _build_evidence_context(
                state,
                max_sentences=sent_limit,
                max_support_items=support_limit,
            )
            user_msg = RISK_CHAIN_USER_PROMPT.format(
                core_entities=core_str,
                evidence_sentences=evidence_str,
                support_statistics=support_str,
            )
            system_prompt = RISK_CHAIN_SYSTEM_PROMPT

            if is_experiment_enabled(state, "disable_evidence_constraint"):
                system_prompt += (
                    "\n\nAblation setting: evidence_sentence and source_doc_id are optional. "
                    "If a slot is strongly supported but a precise quote is unavailable, you may leave those fields empty."
                )

            try:
                response = llm.chat_with_retry(
                    messages=[{"role": "user", "content": user_msg}],
                    system=system_prompt,
                    json_mode=True,
                )
                if _is_refusal(response):
                    logger.warning("[Stage 4 L2] Attempt %s returned refusal-like text", plan_index)
                    continue
                data = LLMClient.parse_json_response(response)
                break
            except Exception as e:
                logger.warning(
                    "[Stage 4 L2] Attempt %s failed with evidence_limit=%s support_limit=%s: %s",
                    plan_index,
                    sent_limit,
                    support_limit,
                    e,
                )
        else:
            logger.error("[Stage 4 L2] All prompt plans failed")

    raw_chain = data.get("risk_chain", {})
    risk_chain = raw_chain if isinstance(raw_chain, dict) else {}
    filled_slots = [k for k, v in risk_chain.items() if isinstance(v, dict) and v.get("value")]
    logger.info(f"[Stage 4 L2] Risk chain slots filled: {filled_slots}")

    return {
        "risk_chain": risk_chain,
        "current_stage": "risk_chain",
        "stages_completed": state.get("stages_completed", []) + ["risk_chain"],
    }

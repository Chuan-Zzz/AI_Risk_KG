"""Stage 4 Layer 3: Controlled Inference."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.state import PipelineState
from src.core.llm import LLMClient, get_llm_client
from src.agent.prompts.stage4_inference import INFERENCE_SYSTEM_PROMPT, INFERENCE_USER_PROMPT

logger = logging.getLogger(__name__)


def inference_node(state: PipelineState) -> dict[str, Any]:
    event_id = state.get("event_id", "unknown")
    logger.info(f"[Stage 4 L3] Running controlled inference for event {event_id}")

    pkg = state.get("event_evidence_package", {})
    core = pkg.get("core_entities", {})
    core_str = json.dumps(core, indent=2, ensure_ascii=False)

    evidence_sents = pkg.get("key_evidence_sentences", [])
    evidence_str = "\n".join(
        f"- [{s.get('doc_id', '?')}] {s.get('sentence', '')}" for s in evidence_sents[:20]
    )

    if not evidence_str.strip():
        logger.warning(f"[Stage 4 L3] No evidence for inference in event {event_id}")
        return {
            "inferences": {},
            "role_assignments": [],
            "current_stage": "inference",
            "stages_completed": state.get("stages_completed", []) + ["inference"],
        }

    user_msg = INFERENCE_USER_PROMPT.format(
        core_entities=core_str,
        evidence_sentences=evidence_str,
    )

    llm = get_llm_client()
    try:
        response = llm.chat_with_retry(
            messages=[{"role": "user", "content": user_msg}],
            system=INFERENCE_SYSTEM_PROMPT,
            json_mode=True,
        )
        data = LLMClient.parse_json_response(response)
    except Exception as e:
        logger.error(f"[Stage 4 L3] LLM failed: {e}")
        data = {}

    inferences = data.get("inferences", {})
    role_assignments = data.get("role_assignments", [])
    governance = data.get("governance", {})
    system_attributes = data.get("system_attributes", {})

    # Merge inferred attributes into the primary AISystem node's attributes
    existing_attrs = state.get("system_attributes", {})
    for attr_name, attr_data in system_attributes.items():
        if attr_data and attr_data.get("value") and attr_data.get("confidence", 0) >= 0.5:
            existing_attrs[attr_name] = {
                "value": attr_data["value"],
                "evidence": attr_data.get("evidence", ""),
                "source_doc_id": attr_data.get("source_doc_id", ""),
                "confidence": attr_data.get("confidence", 0.7),
            }

    inferred_fields = [k for k, v in inferences.items() if v.get("value")]
    logger.info(f"[Stage 4 L3] Inferred fields: {inferred_fields}")
    logger.info(f"[Stage 4 L3] Role assignments: {len(role_assignments)} stakeholders")
    gov_fields = [k for k, v in governance.items() if v.get("value")]
    logger.info(f"[Stage 4 L3] Governance fields: {gov_fields}")
    attr_fields = [k for k, v in system_attributes.items() if v.get("value")]
    logger.info(f"[Stage 4 L3] System attributes: {attr_fields}")

    return {
        "inferences": inferences,
        "role_assignments": role_assignments,
        "governance": governance,
        "system_attributes": existing_attrs,
        "current_stage": "inference",
        "stages_completed": state.get("stages_completed", []) + ["inference"],
    }

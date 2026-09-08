"""Stage 4 Layer 3: Controlled Inference."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.state import PipelineState
from src.core.llm import LLMClient, get_llm_client, _is_refusal, is_glm_family_model, is_deepseek_family_model, requires_compact_mode
from src.agent.prompts.stage4_inference import (
    COMPACT_FIELD_INFERENCE_SYSTEM_PROMPT,
    COMPACT_FIELD_INFERENCE_USER_PROMPT,
    COMPACT_ROLE_INFERENCE_SYSTEM_PROMPT,
    COMPACT_ROLE_INFERENCE_USER_PROMPT,
    FREE_INFERENCE_SYSTEM_PROMPT,
    FREE_INFERENCE_USER_PROMPT,
    INFERENCE_SYSTEM_PROMPT,
    INFERENCE_USER_PROMPT,
)
from src.experiments.flags import is_experiment_enabled

logger = logging.getLogger(__name__)


def _glm_core_entities_for_roles(core: dict[str, Any]) -> str:
    role_core = {
        "ai_systems": core.get("ai_systems", []),
        "ai_models": core.get("ai_models", []),
        "stakeholders": core.get("stakeholders", []),
        "regulations": core.get("regulations", []),
        "standards": core.get("standards", []),
    }
    return json.dumps(role_core, indent=2, ensure_ascii=False)


def _safe_section_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_role_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def inference_node(state: PipelineState) -> dict[str, Any]:
    event_id = state.get("event_id", "unknown")
    logger.info(f"[Stage 4 L3] Running controlled inference for event {event_id}")
    llm = get_llm_client()
    # Stage 4 L3: enable compact mode for both GLM and DeepSeek.
    # Non-compact system prompt is ~9800 chars; deepseek-v4-pro via upstream proxy
    # proxy easily exceeds Cloudflare's 120s hard limit on long prompts.
    # Compact mode uses ~600 char prompts + split into 2 shorter requests.
    compact_mode = requires_compact_mode(llm.model)

    pkg = state.get("event_evidence_package", {})
    core = pkg.get("core_entities", {})
    core_str = json.dumps(core, indent=2, ensure_ascii=False)

    evidence_sents = pkg.get("key_evidence_sentences", [])
    evidence_limit = 10 if compact_mode else 20
    evidence_str = "\n".join(
        f"- [{s.get('doc_id', '?')}] {s.get('sentence', '')}" for s in evidence_sents[:evidence_limit]
    )

    if not evidence_str.strip():
        logger.warning(f"[Stage 4 L3] No evidence for inference in event {event_id}")
        return {
            "inferences": {},
            "role_assignments": [],
            "current_stage": "inference",
            "stages_completed": state.get("stages_completed", []) + ["inference"],
        }

    disable_ontology = is_experiment_enabled(state, "disable_ontology_constraint")
    disable_controlled = is_experiment_enabled(state, "disable_controlled_inference")
    disable_evidence = is_experiment_enabled(state, "disable_evidence_constraint")

    if compact_mode:
        plans = [evidence_limit, 6]
        merged_data: dict[str, Any] = {"inferences": {}, "governance": {}, "system_attributes": {}, "role_assignments": []}

        for plan_index, sent_limit in enumerate(plans, 1):
            compact_evidence_str = "\n".join(
                f"- [{s.get('doc_id', '?')}] {s.get('sentence', '')}" for s in evidence_sents[:sent_limit]
            )
            if not compact_evidence_str.strip():
                continue

            field_user_msg = COMPACT_FIELD_INFERENCE_USER_PROMPT.format(
                core_entities=core_str,
                evidence_sentences=compact_evidence_str,
            )
            role_user_msg = COMPACT_ROLE_INFERENCE_USER_PROMPT.format(
                core_entities=_glm_core_entities_for_roles(core),
                evidence_sentences="\n".join(
                    f"- [{s.get('doc_id', '?')}] {s.get('sentence', '')}" for s in evidence_sents[: min(sent_limit, 4)]
                ),
            )

            if disable_evidence:
                field_system_prompt = COMPACT_FIELD_INFERENCE_SYSTEM_PROMPT + (
                    "\n\nAblation setting: evidence, source_doc_id, and reasoning are optional."
                )
                role_system_prompt = COMPACT_ROLE_INFERENCE_SYSTEM_PROMPT + (
                    "\n\nAblation setting: evidence, source_doc_id, and reasoning are optional."
                )
            else:
                field_system_prompt = COMPACT_FIELD_INFERENCE_SYSTEM_PROMPT
                role_system_prompt = COMPACT_ROLE_INFERENCE_SYSTEM_PROMPT

            # Ontology / controlled-inference ablation: relax the fixed candidate
            # lists (lifecycle phase, domain, role types) so inferred values may
            # be phrased freely as long as they stay grounded in the evidence.
            # This mirrors the non-compact FREE_INFERENCE_SYSTEM_PROMPT branch.
            if disable_ontology or disable_controlled:
                free_form_suffix = (
                    "\n\nAblation setting: ontology constraints are relaxed. "
                    "You are NOT restricted to fixed candidate lists for "
                    "lifecycle phase, impact domain, or role types. Phrase "
                    "inferred values freely when the evidence supports it."
                )
                field_system_prompt += free_form_suffix
                role_system_prompt += free_form_suffix

            try:
                field_response = llm.chat_with_retry(
                    messages=[{"role": "user", "content": field_user_msg}],
                    system=field_system_prompt,
                    json_mode=True,
                    max_retries=2,
                    timeout=100,
                )
                if not _is_refusal(field_response):
                    field_data = LLMClient.parse_json_response(field_response)
                    merged_data["inferences"].update(_safe_section_dict(field_data.get("inferences", {})))
                    merged_data["governance"].update(_safe_section_dict(field_data.get("governance", {})))
                    inferences_dict = _safe_section_dict(field_data.get("inferences", {}))
                    field_attrs = _safe_section_dict(field_data.get("system_attributes", {})) or _safe_section_dict(inferences_dict.get("system_attributes", {}))
                    merged_data["system_attributes"].update(field_attrs)
            except Exception as e:
                logger.warning("[Stage 4 L3] Field inference attempt %s failed with evidence_limit=%s: %s", plan_index, sent_limit, e)

            try:
                role_response = llm.chat_with_retry(
                    messages=[{"role": "user", "content": role_user_msg}],
                    system=role_system_prompt,
                    json_mode=True,
                    max_retries=1,
                    timeout=100,
                )
                if not _is_refusal(role_response):
                    role_data = LLMClient.parse_json_response(role_response)
                    role_assignments = _safe_role_list(role_data.get("role_assignments", []))
                    if role_assignments:
                        # Merge by (stakeholder_name, role) instead of overwriting,
                        # so later compact plans supplement (rather than replace)
                        # earlier role assignments.
                        existing_roles = {
                            f"{ra.get('stakeholder_name', '')}_{ra.get('role', '')}": ra
                            for ra in merged_data.get("role_assignments", [])
                        }
                        for ra in role_assignments:
                            key = f"{ra.get('stakeholder_name', '')}_{ra.get('role', '')}"
                            if key not in existing_roles:
                                existing_roles[key] = ra
                        merged_data["role_assignments"] = list(existing_roles.values())
            except Exception as e:
                logger.warning("[Stage 4 L3] Role inference attempt %s failed with evidence_limit=%s: %s", plan_index, sent_limit, e)

            if merged_data["inferences"] or merged_data["role_assignments"] or merged_data["governance"] or merged_data["system_attributes"]:
                data = merged_data
                break
        else:
            logger.error("[Stage 4 L3] All compact inference plans failed")
            data = merged_data
    elif disable_ontology or disable_controlled:
        system_prompt = FREE_INFERENCE_SYSTEM_PROMPT
        user_msg = FREE_INFERENCE_USER_PROMPT.format(
            core_entities=core_str,
            evidence_sentences=evidence_str,
        )
    else:
        system_prompt = INFERENCE_SYSTEM_PROMPT
        user_msg = INFERENCE_USER_PROMPT.format(
            core_entities=core_str,
            evidence_sentences=evidence_str,
        )

    if not compact_mode:
        if disable_evidence:
            system_prompt += (
                "\n\nAblation setting: evidence, source_doc_id, and reasoning are optional. "
                "When they are unavailable, leave them empty instead of omitting an otherwise supported inference."
            )

        data = {}
        try:
            response = llm.chat_with_retry(
                messages=[{"role": "user", "content": user_msg}],
                system=system_prompt,
                json_mode=True,
            )
            if not _is_refusal(response):
                data = LLMClient.parse_json_response(response)
            else:
                logger.warning("[Stage 4 L3] Inference returned refusal-like text")
        except Exception as e:
            logger.error(f"[Stage 4 L3] LLM failed: {e}")

    raw_inferences = data.get("inferences", {})
    inferences = raw_inferences if isinstance(raw_inferences, dict) else {}
    role_assignments = data.get("role_assignments", [])
    if not isinstance(role_assignments, list):
        role_assignments = []
    raw_governance = data.get("governance", {})
    governance = raw_governance if isinstance(raw_governance, dict) else {}
    raw_attrs = data.get("system_attributes", {})
    system_attributes = raw_attrs if isinstance(raw_attrs, dict) else inferences.pop("system_attributes", {})
    if not isinstance(system_attributes, dict):
        system_attributes = {}

    # Merge inferred attributes into the primary AISystem node's attributes
    existing_attrs = state.get("system_attributes", {})
    for attr_name, attr_data in system_attributes.items():
        if not isinstance(attr_data, dict):
            continue
        if attr_data and attr_data.get("value") and attr_data.get("confidence", 0) >= 0.5:
            existing_attrs[attr_name] = {
                "value": attr_data["value"],
                "evidence": attr_data.get("evidence", ""),
                "source_doc_id": attr_data.get("source_doc_id", ""),
                "confidence": attr_data.get("confidence", 0.7),
            }

    inferred_fields = [k for k, v in inferences.items() if isinstance(v, dict) and v.get("value")]
    logger.info(f"[Stage 4 L3] Inferred fields: {inferred_fields}")
    logger.info(f"[Stage 4 L3] Role assignments: {len(role_assignments)} stakeholders")
    gov_fields = [k for k, v in governance.items() if isinstance(v, dict) and v.get("value")]
    logger.info(f"[Stage 4 L3] Governance fields: {gov_fields}")
    attr_fields = [k for k, v in system_attributes.items() if isinstance(v, dict) and v.get("value")]
    logger.info(f"[Stage 4 L3] System attributes: {attr_fields}")

    return {
        "inferences": inferences,
        "role_assignments": role_assignments,
        "governance": governance,
        "system_attributes": existing_attrs,
        "current_stage": "inference",
        "stages_completed": state.get("stages_completed", []) + ["inference"],
    }

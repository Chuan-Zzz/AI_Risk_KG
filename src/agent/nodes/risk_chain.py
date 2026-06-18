"""Stage 4 Layer 2: Risk Chain Slot Filling."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.state import PipelineState
from src.core.llm import LLMClient, get_llm_client
from src.agent.prompts.stage4_risk_chain import RISK_CHAIN_SYSTEM_PROMPT, RISK_CHAIN_USER_PROMPT

logger = logging.getLogger(__name__)


def _build_evidence_context(state: PipelineState) -> tuple[str, str, str]:
    pkg = state.get("event_evidence_package", {})
    core = pkg.get("core_entities", {})
    core_str = json.dumps(core, indent=2, ensure_ascii=False)

    evidence_sents = pkg.get("key_evidence_sentences", [])
    evidence_str = "\n".join(
        f"- [{s.get('doc_id', '?')}] {s.get('sentence', '')}" for s in evidence_sents[:30]
    )

    support = pkg.get("support_statistics", {})
    support_str = json.dumps(support, indent=2, ensure_ascii=False)

    return core_str, evidence_str, support_str


def risk_chain_node(state: PipelineState) -> dict[str, Any]:
    event_id = state.get("event_id", "unknown")
    logger.info(f"[Stage 4 L2] Extracting risk chain for event {event_id}")

    core_str, evidence_str, support_str = _build_evidence_context(state)

    if not evidence_str.strip():
        logger.warning(f"[Stage 4 L2] No evidence sentences for event {event_id}")
        return {
            "risk_chain": {},
            "current_stage": "risk_chain",
            "stages_completed": state.get("stages_completed", []) + ["risk_chain"],
        }

    user_msg = RISK_CHAIN_USER_PROMPT.format(
        core_entities=core_str,
        evidence_sentences=evidence_str,
        support_statistics=support_str,
    )

    llm = get_llm_client()
    try:
        response = llm.chat_with_retry(
            messages=[{"role": "user", "content": user_msg}],
            system=RISK_CHAIN_SYSTEM_PROMPT,
            json_mode=True,
        )
        data = LLMClient.parse_json_response(response)
    except Exception as e:
        logger.error(f"[Stage 4 L2] LLM failed: {e}")
        data = {}

    risk_chain = data.get("risk_chain", {})
    filled_slots = [k for k, v in risk_chain.items() if v.get("value")]
    logger.info(f"[Stage 4 L2] Risk chain slots filled: {filled_slots}")

    return {
        "risk_chain": risk_chain,
        "current_stage": "risk_chain",
        "stages_completed": state.get("stages_completed", []) + ["risk_chain"],
    }

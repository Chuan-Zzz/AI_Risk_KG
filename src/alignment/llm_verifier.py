"""LLM-based verification for boundary entity pairs."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.core.llm import LLMClient, get_llm_client
from src.core.models import EntityNode

logger = logging.getLogger(__name__)

_VERIFY_PROMPT = """You are an expert in entity resolution for an AI risk knowledge graph.

Given two entities with their names, types, descriptions, and supporting evidence, \
determine if they refer to the SAME real-world entity.

Consider:
1. Whether the names refer to the same organization, product, model, or person
2. Whether the descriptions are consistent with being the same entity
3. Whether the evidence contexts overlap or contradict

Entity 1:
  Name: {name1}
  Type: {type1}
  Description: {desc1}
  Evidence: {evidence1}

Entity 2:
  Name: {name2}
  Type: {type2}
  Description: {desc2}
  Evidence: {evidence2}

Similarity score: {similarity:.3f}

Are they the same real-world entity? Return ONLY a JSON object:
{{"same": true|false, "reason": "brief explanation"}}"""


def _truncate(text: str, max_len: int = 300) -> str:
    if not text:
        return "N/A"
    return text[:max_len] + "..." if len(text) > max_len else text


class LLMVerifier:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or get_llm_client()

    def verify(
        self,
        entity1: EntityNode,
        entity2: EntityNode,
        similarity: float,
    ) -> bool:
        """Return True if the two entities should be merged."""
        prompt = _VERIFY_PROMPT.format(
            name1=entity1.name,
            type1=entity1.entity_type.value,
            desc1=_truncate(entity1.description),
            evidence1=_truncate("; ".join(ev.evidence_sentence for ev in entity1.evidence[:3] if ev.evidence_sentence) if entity1.evidence else ""),
            name2=entity2.name,
            type2=entity2.entity_type.value,
            desc2=_truncate(entity2.description),
            evidence2=_truncate("; ".join(ev.evidence_sentence for ev in entity2.evidence[:3] if ev.evidence_sentence) if entity2.evidence else ""),
            similarity=similarity,
        )

        try:
            resp = self._llm.chat_with_retry(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
                max_retries=2,
            )
            data = LLMClient.parse_json_response(resp)
            same = data.get("same")
            if isinstance(same, bool):
                return same
            if isinstance(same, str):
                return same.lower() in ("true", "yes", "1")
            logger.warning(
                f"LLM verify unexpected 'same' field for ({entity1.name}, {entity2.name}): {same}"
            )
            return False
        except json.JSONDecodeError as e:
            logger.warning(f"LLM verify JSON parse error for ({entity1.name}, {entity2.name}): {e}")
            return False
        except Exception as e:
            logger.warning(f"LLM verification failed for ({entity1.name}, {entity2.name}): {e}")
            return False

    def verify_batch(
        self,
        pairs: list[tuple[EntityNode, EntityNode, float]],
    ) -> list[bool]:
        """Verify multiple pairs. Returns list of booleans."""
        results: list[bool] = []
        for e1, e2, sim in pairs:
            results.append(self.verify(e1, e2, sim))
        return results


def verify_pairs(
    pairs: list[tuple[EntityNode, EntityNode, float]],
    llm: LLMClient | None = None,
) -> list[bool]:
    """Convenience function: verify a list of entity pairs."""
    verifier = LLMVerifier(llm=llm)
    return verifier.verify_batch(pairs)

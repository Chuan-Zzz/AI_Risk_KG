"""LLM-based verification for boundary entity pairs."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.core.llm import LLMClient, get_llm_client
from src.core.models import EntityNode

logger = logging.getLogger(__name__)

_VERIFY_PROMPT = """You are an expert in entity resolution.

Given two entity names and their types, determine if they refer to the SAME real-world entity.

IMPORTANT: Return ONLY a single-line JSON object with no markdown formatting:
{{"same": true, "reason": "explanation"}}
or
{{"same": false, "reason": "explanation"}}

Entity 1: {name1}
Type 1: {type1}

Entity 2: {name2}
Type 2: {type2}

Are they the same entity? Return JSON only."""


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
            name2=entity2.name,
            type2=entity2.entity_type.value,
        )

        try:
            resp = self._llm.chat_with_retry(
                messages=[{"role": "user", "content": prompt}],
                json_mode=False,
            )
            # Strip markdown code fences if present
            text = resp.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            # Find JSON object in response
            import re
            match = re.search(r'\{[^}]+\}', text)
            if match:
                data = json.loads(match.group())
                same = data.get("same")
                if isinstance(same, bool):
                    return same
                if isinstance(same, str):
                    return same.lower() in ("true", "yes", "1")
            logger.warning(f"LLM verify could not parse response for ({entity1.name}, {entity2.name}): {resp[:100]}")
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

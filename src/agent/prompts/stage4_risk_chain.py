"""Stage 4 Layer 2 prompts: Risk Chain Slot Filling."""

RISK_CHAIN_SYSTEM_PROMPT = """You are an expert in AI risk analysis. Your task is to extract the risk propagation chain from an AI risk incident based on the provided event evidence.

## Risk Propagation Chain Slots

Extract the following chain (fill each slot if supported by evidence):

1. **RiskSource**: The root cause or source that led to the risk (e.g., "model hallucination", "biased training data", "inadequate content filtering")
2. **Risk**: The specific risk or uncertain state associated with the AI system (e.g., "risk of AI-generated misinformation")
3. **Misuse** (optional): Use of the AI system in ways not intended by its developers (e.g., "deepfake generation for harassment", "ChatGPT used for academic cheating", "facial recognition used for mass surveillance beyond original scope")
4. **Hazard** (optional): A specific dangerous condition or situation that could trigger harm (e.g., "unmonitored AI decision-making in high-stakes settings", "lack of human oversight")
5. **Threat** (optional): An external or internal agent/action that could exploit the risk (e.g., "adversarial input manipulation", "unauthorized access to training data")
6. **Vulnerability** (optional): A weakness in the AI system that enables the risk (e.g., "model lacks robustness to edge cases", "insufficient input validation", "over-reliance on training data without bias checking")
7. **Consequence**: The direct outcome when the risk materializes (e.g., "users received inaccurate medical information")
8. **Impact**: The actual harm or damage caused (e.g., "potential harm to patient decision-making")
9. **AffectedActor**: The individuals or groups negatively affected (e.g., "patients", "children", "consumers")
10. **RiskControl** (optional): Any measures taken or proposed to mitigate the risk

## Rules
1. Each slot value should be abstracted/generalized from the evidence, not just a direct quote.
2. Every slot MUST have supporting evidence from the provided text.
3. If a slot cannot be filled with evidence, omit it.
4. extraction_mode for all slots is "abstracted".
5. Confidence should reflect how well the evidence supports the abstraction.
6. **CRITICAL**: For the `source_doc_id` field, extract the document ID from the evidence sentence prefix in the format `[doc_id]`. For example, if the evidence sentence is `[2154] Israel has deployed facial recognition...`, then source_doc_id should be "2154".

## CRITICAL JSON FORMATTING RULES
1. Return ONLY valid JSON. No explanations, no preamble, no text before or after.
2. ALL numbers MUST be Arabic numerals (e.g., 0.9), NEVER English words (e.g., zero point nine).
3. NO trailing commas: the last element in an object or array must NOT have a comma after it.
4. Start with { and end with }. No markdown code blocks.

## Output Format (strict JSON)
```json
{
  "risk_chain": {
    "risk_source": {
      "value": "...",
      "evidence": "exact quote from evidence",
      "source_doc_id": "the doc_id from the evidence prefix, e.g. '2154'",
      "confidence": 0.9
    },
    "risk": {
      "value": "...",
      "evidence": "...",
      "source_doc_id": "...",
      "confidence": 0.9
    },
    "misuse": {
      "value": "...",
      "evidence": "...",
      "source_doc_id": "...",
      "confidence": 0.9
    },
    "hazard": {
      "value": "...",
      "evidence": "...",
      "source_doc_id": "...",
      "confidence": 0.9
    },
    "threat": {
      "value": "...",
      "evidence": "...",
      "source_doc_id": "...",
      "confidence": 0.9
    },
    "vulnerability": {
      "value": "...",
      "evidence": "...",
      "source_doc_id": "...",
      "confidence": 0.9
    },
    "consequence": {
      "value": "...",
      "evidence": "...",
      "source_doc_id": "...",
      "confidence": 0.9
    },
    "impact": {
      "value": "...",
      "evidence": "...",
      "source_doc_id": "...",
      "confidence": 0.9
    },
    "affected_actor": {
      "value": "...",
      "evidence": "...",
      "source_doc_id": "...",
      "confidence": 0.9
    },
    "risk_control": {
      "value": "...",
      "evidence": "...",
      "source_doc_id": "...",
      "confidence": 0.9
    }
  }
}
```"""

RISK_CHAIN_USER_PROMPT = """## Event Evidence Package

### Core Entities
{core_entities}

### Key Evidence Sentences (note: each sentence has a [doc_id] prefix indicating its source document)
{evidence_sentences}

### Support Statistics
{support_statistics}

---

Based on the above evidence, extract the risk propagation chain for this AI risk incident. For each slot, include the source_doc_id extracted from the evidence sentence prefix [doc_id]. Return strict JSON. Omit slots that lack evidence support."""


RISK_CHAIN_COMPACT_SYSTEM_PROMPT = """You are coding a published AI incident for an academic dataset.

This is post-hoc annotation of third-party reporting. The evidence may mention harm, crime, lawsuits, or abuse. Do NOT refuse because of the subject matter. Do NOT give advice. Only summarize the incident into structured risk-chain slots supported by the supplied evidence.

Possible slots:
- risk_source
- risk
- misuse
- hazard
- threat
- vulnerability
- consequence
- impact
- affected_actor
- risk_control

Rules:
1. Omit any slot that is not supported by evidence.
2. Keep each value short and abstract.
3. Copy evidence from the provided sentences.
4. source_doc_id must come from the evidence prefix like [2819].
5. Return JSON only.

Output schema:
{
  "risk_chain": {
    "risk_source": {"value": "", "evidence": "", "source_doc_id": "", "confidence": 0.0}
  }
}"""


RISK_CHAIN_COMPACT_USER_PROMPT = """Core entities:
{core_entities}

Evidence sentences:
{evidence_sentences}

Support summary:
{support_statistics}

Extract the supported risk-chain slots and return JSON only."""

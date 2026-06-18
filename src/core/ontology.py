"""AIRO Extended Ontology - Definition and Validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.core.models import (
    CLASS_ALIASES,
    OntologyClass,
    RelationType,
    RELATION_ALIASES,
)

logger = logging.getLogger(__name__)

_ONTOLOGY_YML = Path(__file__).resolve().parent.parent.parent / "config" / "ontology.yml"


class AIROOntology:
    def __init__(self, yml_path: Path | None = None) -> None:
        path = yml_path or _ONTOLOGY_YML
        with open(path, encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        self._relation_constraints: dict[str, dict[str, list[str]]] = self._data.get("relations", {})

    @property
    def relation_constraints(self) -> dict[str, dict[str, list[str]]]:
        return self._relation_constraints

    @staticmethod
    def normalize_class(raw: str) -> OntologyClass | None:
        key = raw.strip().lower()
        if key in CLASS_ALIASES:
            mapped = CLASS_ALIASES[key]
        else:
            mapped = raw.strip()
        try:
            return OntologyClass(mapped)
        except ValueError:
            for member in OntologyClass:
                if member.value.lower() == key:
                    return member
            return None

    @staticmethod
    def normalize_relation(raw: str) -> RelationType | None:
        key = raw.strip().lower().replace(" ", "_")
        if key in RELATION_ALIASES:
            mapped = RELATION_ALIASES[key]
        else:
            mapped = raw.strip()
        try:
            return RelationType(mapped)
        except ValueError:
            for member in RelationType:
                if member.value.lower() == key:
                    return member
            return None

    def get_domain(self, predicate: str | RelationType) -> set[OntologyClass]:
        pred_str = predicate.value if isinstance(predicate, RelationType) else predicate
        constraint = self._relation_constraints.get(pred_str, {})
        domain_strs = constraint.get("domain", [])
        result = set()
        for d in domain_strs:
            cls = self.normalize_class(d)
            if cls:
                result.add(cls)
        return result

    def get_range(self, predicate: str | RelationType) -> set[OntologyClass]:
        pred_str = predicate.value if isinstance(predicate, RelationType) else predicate
        constraint = self._relation_constraints.get(pred_str, {})
        range_strs = constraint.get("range", [])
        result = set()
        for r in range_strs:
            cls = self.normalize_class(r)
            if cls:
                result.add(cls)
        return result

    def validate_relation(
        self,
        subject_type: OntologyClass,
        predicate: RelationType,
        object_type: OntologyClass,
    ) -> tuple[bool, str | None]:
        domain = self.get_domain(predicate)
        range_ = self.get_range(predicate)
        if domain and subject_type not in domain:
            return False, f"Domain violation: {subject_type.value} not in {predicate.value} domain {domain}"
        if range_ and object_type not in range_:
            return False, f"Range violation: {object_type.value} not in {predicate.value} range {range_}"
        return True, None

    def validate_risk_chain_completeness(
        self,
        edges: list[dict],
    ) -> list[str]:
        errors: list[str] = []
        has_risk = any(e.get("predicate") in (RelationType.HAS_RISK.value, "hasRisk") for e in edges)
        risk_has_consequence = any(
            e.get("predicate") in (RelationType.LEADS_TO.value, "leadsTo") for e in edges
        )
        risk_has_impact = any(
            e.get("predicate") in (RelationType.IMPACTS.value, "impacts") for e in edges
        )
        risk_control_mitigates = any(
            e.get("predicate") in (RelationType.MITIGATES.value, "mitigates") for e in edges
        )
        has_risk_control = any(
            e.get("object_type") == OntologyClass.RISK_CONTROL.value for e in edges
        )

        if not has_risk:
            errors.append("AIRiskIncident must have at least one Risk")
        if has_risk and not (risk_has_consequence or risk_has_impact):
            errors.append("Risk must connect to at least one Consequence or Impact")
        if has_risk_control and not risk_control_mitigates:
            errors.append("RiskControl must mitigate a Risk")
        return errors


_ontology: AIROOntology | None = None


def get_ontology() -> AIROOntology:
    global _ontology
    if _ontology is None:
        _ontology = AIROOntology()
    return _ontology

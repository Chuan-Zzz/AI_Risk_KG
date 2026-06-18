"""Tests for AIRO ontology validator (ontology.py).

Covers: normalize_class, normalize_relation, validate_relation,
validate_risk_chain_completeness.
"""

from __future__ import annotations

import pytest

from src.core.models import OntologyClass, RelationType
from src.core.ontology import AIROOntology, get_ontology


@pytest.fixture
def ontology():
    """Provide a fresh AIROOntology instance loaded from ontology.yml."""
    return AIROOntology()


# ============================================================
# normalize_class
# ============================================================


class TestNormalizeClass:
    def test_exact_standard_name(self, ontology):
        result = ontology.normalize_class("AISystem")
        assert result == OntologyClass.AI_SYSTEM

    def test_exact_standard_name_risk(self, ontology):
        result = ontology.normalize_class("Risk")
        assert result == OntologyClass.RISK

    def test_case_insensitive(self, ontology):
        result = ontology.normalize_class("aisystem")
        assert result == OntologyClass.AI_SYSTEM

    def test_case_insensitive_mixed(self, ontology):
        result = ontology.normalize_class("Risk")
        assert result == OntologyClass.RISK

    def test_alias_llm(self, ontology):
        result = ontology.normalize_class("LLM")
        assert result == OntologyClass.AI_MODEL

    def test_alias_model(self, ontology):
        result = ontology.normalize_class("model")
        assert result == OntologyClass.AI_MODEL

    def test_alias_system(self, ontology):
        result = ontology.normalize_class("system")
        assert result == OntologyClass.AI_SYSTEM

    def test_alias_foundation_model(self, ontology):
        result = ontology.normalize_class("foundation model")
        assert result == OntologyClass.AI_MODEL

    def test_alias_large_language_model(self, ontology):
        result = ontology.normalize_class("large language model")
        assert result == OntologyClass.AI_MODEL

    def test_alias_law(self, ontology):
        result = ontology.normalize_class("law")
        assert result == OntologyClass.REGULATION

    def test_alias_act(self, ontology):
        result = ontology.normalize_class("act")
        assert result == OntologyClass.REGULATION

    def test_alias_mitigation(self, ontology):
        result = ontology.normalize_class("mitigation")
        assert result == OntologyClass.RISK_CONTROL

    def test_alias_fine(self, ontology):
        result = ontology.normalize_class("fine")
        assert result == OntologyClass.ENFORCEMENT_ACTION

    def test_alias_victim(self, ontology):
        result = ontology.normalize_class("victim")
        assert result == OntologyClass.AFFECTED_ACTOR

    def test_alias_incident(self, ontology):
        result = ontology.normalize_class("incident")
        assert result == OntologyClass.AI_RISK_INCIDENT

    def test_alias_event(self, ontology):
        result = ontology.normalize_class("event")
        assert result == OntologyClass.AI_RISK_INCIDENT

    def test_alias_article(self, ontology):
        result = ontology.normalize_class("article")
        assert result == OntologyClass.NEWS_REPORT

    def test_alias_developer(self, ontology):
        result = ontology.normalize_class("developer")
        assert result == OntologyClass.AI_DEVELOPER

    def test_alias_provider(self, ontology):
        result = ontology.normalize_class("provider")
        assert result == OntologyClass.AI_PROVIDER

    def test_alias_deployer(self, ontology):
        result = ontology.normalize_class("deployer")
        assert result == OntologyClass.AI_DEPLOYER

    def test_alias_user(self, ontology):
        result = ontology.normalize_class("user")
        assert result == OntologyClass.AI_USER

    def test_alias_organization(self, ontology):
        result = ontology.normalize_class("organization")
        assert result == OntologyClass.STAKEHOLDER

    def test_alias_company(self, ontology):
        result = ontology.normalize_class("company")
        assert result == OntologyClass.STAKEHOLDER

    def test_whitespace_stripped(self, ontology):
        result = ontology.normalize_class("  AISystem  ")
        assert result == OntologyClass.AI_SYSTEM

    def test_chinese_alias_base_model(self, ontology):
        result = ontology.normalize_class("基础模型")
        assert result == OntologyClass.AI_MODEL

    def test_chinese_alias_large_model(self, ontology):
        result = ontology.normalize_class("大模型")
        assert result == OntologyClass.AI_MODEL

    def test_invalid_returns_none(self, ontology):
        result = ontology.normalize_class("NonExistentThing")
        assert result is None

    def test_empty_string_returns_none(self, ontology):
        result = ontology.normalize_class("")
        assert result is None

    def test_numeric_string_returns_none(self, ontology):
        result = ontology.normalize_class("12345")
        assert result is None

    def test_all_standard_enum_values_normalize(self, ontology):
        """Every OntologyClass member value should normalize back to itself."""
        for member in OntologyClass:
            result = ontology.normalize_class(member.value)
            assert result == member, f"Failed for {member.value}"


# ============================================================
# normalize_relation
# ============================================================


class TestNormalizeRelation:
    def test_exact_standard_name(self, ontology):
        result = ontology.normalize_relation("causes")
        assert result == RelationType.CAUSES

    def test_exact_standard_leads_to(self, ontology):
        result = ontology.normalize_relation("leadsTo")
        assert result == RelationType.LEADS_TO

    def test_alias_developed_by(self, ontology):
        result = ontology.normalize_relation("developed_by")
        assert result == RelationType.DEVELOPS

    def test_alias_developed_by_space(self, ontology):
        result = ontology.normalize_relation("developed by")
        assert result == RelationType.DEVELOPS

    def test_alias_provided_by(self, ontology):
        result = ontology.normalize_relation("provided_by")
        assert result == RelationType.PROVIDES

    def test_alias_deployed_by(self, ontology):
        result = ontology.normalize_relation("deployed_by")
        assert result == RelationType.DEPLOYS

    def test_alias_used_by(self, ontology):
        result = ontology.normalize_relation("used_by")
        assert result == RelationType.USES

    def test_alias_results_in(self, ontology):
        result = ontology.normalize_relation("results_in")
        assert result == RelationType.LEADS_TO

    def test_alias_regulates(self, ontology):
        result = ontology.normalize_relation("regulates")
        assert result == RelationType.GOVERNS

    def test_alias_governed_by(self, ontology):
        result = ontology.normalize_relation("governed_by")
        assert result == RelationType.GOVERNS

    def test_whitespace_stripped(self, ontology):
        result = ontology.normalize_relation("  causes  ")
        assert result == RelationType.CAUSES

    def test_case_insensitive(self, ontology):
        result = ontology.normalize_relation("Causes")
        assert result == RelationType.CAUSES

    def test_invalid_returns_none(self, ontology):
        result = ontology.normalize_relation("doesNotExist")
        assert result is None

    def test_empty_string_returns_none(self, ontology):
        result = ontology.normalize_relation("")
        assert result is None

    def test_all_standard_enum_values_normalize(self, ontology):
        """Every RelationType member value should normalize back to itself."""
        for member in RelationType:
            result = ontology.normalize_relation(member.value)
            assert result == member, f"Failed for {member.value}"


# ============================================================
# validate_relation
# ============================================================


class TestValidateRelation:
    def test_valid_involves_ai_system(self, ontology):
        valid, err = ontology.validate_relation(
            OntologyClass.AI_RISK_INCIDENT,
            RelationType.INVOLVES_AI_SYSTEM,
            OntologyClass.AI_SYSTEM,
        )
        assert valid is True
        assert err is None

    def test_valid_involves_stakeholder(self, ontology):
        valid, err = ontology.validate_relation(
            OntologyClass.AI_RISK_INCIDENT,
            RelationType.INVOLVES_STAKEHOLDER,
            OntologyClass.STAKEHOLDER,
        )
        assert valid is True
        assert err is None

    def test_valid_causes_chain(self, ontology):
        valid, err = ontology.validate_relation(
            OntologyClass.RISK_SOURCE,
            RelationType.CAUSES,
            OntologyClass.RISK,
        )
        assert valid is True
        assert err is None

    def test_valid_leads_to_chain(self, ontology):
        valid, err = ontology.validate_relation(
            OntologyClass.RISK,
            RelationType.LEADS_TO,
            OntologyClass.CONSEQUENCE,
        )
        assert valid is True
        assert err is None

    def test_valid_impacts_chain(self, ontology):
        valid, err = ontology.validate_relation(
            OntologyClass.CONSEQUENCE,
            RelationType.IMPACTS,
            OntologyClass.IMPACT,
        )
        assert valid is True
        assert err is None

    def test_valid_affects_chain(self, ontology):
        valid, err = ontology.validate_relation(
            OntologyClass.IMPACT,
            RelationType.AFFECTS,
            OntologyClass.AFFECTED_ACTOR,
        )
        assert valid is True
        assert err is None

    def test_valid_mitigates(self, ontology):
        valid, err = ontology.validate_relation(
            OntologyClass.RISK_CONTROL,
            RelationType.MITIGATES,
            OntologyClass.RISK,
        )
        assert valid is True
        assert err is None

    def test_valid_develops(self, ontology):
        valid, err = ontology.validate_relation(
            OntologyClass.AI_DEVELOPER,
            RelationType.DEVELOPS,
            OntologyClass.AI_MODEL,
        )
        assert valid is True
        assert err is None

    def test_valid_role_held_by(self, ontology):
        valid, err = ontology.validate_relation(
            OntologyClass.ROLE_ASSIGNMENT,
            RelationType.ROLE_HELD_BY,
            OntologyClass.STAKEHOLDER,
        )
        assert valid is True
        assert err is None

    def test_domain_violation_detected(self, ontology):
        """Stakeholder cannot be subject of 'causes' (domain is RiskSource)."""
        valid, err = ontology.validate_relation(
            OntologyClass.STAKEHOLDER,
            RelationType.CAUSES,
            OntologyClass.RISK,
        )
        assert valid is False
        assert "Domain violation" in err

    def test_range_violation_detected(self, ontology):
        """RiskSource cannot be object of 'deploys' (range is AISystem, not RiskSource)."""
        valid, err = ontology.validate_relation(
            OntologyClass.AI_DEPLOYER,
            RelationType.DEPLOYS,
            OntologyClass.RISK_SOURCE,
        )
        assert valid is False
        assert "Range violation" in err

    def test_stakeholder_in_deploys_range_violation(self, ontology):
        valid, err = ontology.validate_relation(
            OntologyClass.AI_DEPLOYER,
            RelationType.DEPLOYS,
            OntologyClass.STAKEHOLDER,
        )
        assert valid is False
        assert "Range violation" in err

    def test_unconstrained_predicate_passes(self, ontology):
        """Predicates not in ontology.yml have no domain/range constraints and pass any types."""
        # Use a predicate that is NOT in ontology.yml to test unconstrained behavior.
        # Most predicates ARE constrained, so use a string that doesn't appear.
        valid, err = ontology.validate_relation(
            OntologyClass.STAKEHOLDER,
            "totallyFakeRelation",
            OntologyClass.STAKEHOLDER,
        )
        assert valid is True
        assert err is None


# ============================================================
# validate_risk_chain_completeness
# ============================================================


class TestValidateRiskChainCompleteness:
    def test_complete_chain_passes(self, ontology):
        edges = [
            {"predicate": "hasRisk"},
            {"predicate": "leadsTo"},
            {"predicate": "impacts"},
        ]
        errors = ontology.validate_risk_chain_completeness(edges)
        assert errors == []

    def test_missing_risk_fails(self, ontology):
        edges = [
            {"predicate": "leadsTo"},
            {"predicate": "impacts"},
        ]
        errors = ontology.validate_risk_chain_completeness(edges)
        assert len(errors) >= 1
        assert any("at least one Risk" in e for e in errors)

    def test_missing_consequence_and_impact_fails(self, ontology):
        edges = [
            {"predicate": "hasRisk"},
        ]
        errors = ontology.validate_risk_chain_completeness(edges)
        assert len(errors) >= 1
        assert any("Consequence or Impact" in e for e in errors)

    def test_risk_with_consequence_passes(self, ontology):
        edges = [
            {"predicate": "hasRisk"},
            {"predicate": "leadsTo"},
        ]
        errors = ontology.validate_risk_chain_completeness(edges)
        assert errors == []

    def test_risk_with_impact_passes(self, ontology):
        edges = [
            {"predicate": "hasRisk"},
            {"predicate": "impacts"},
        ]
        errors = ontology.validate_risk_chain_completeness(edges)
        assert errors == []

    def test_risk_control_without_mitigates_fails(self, ontology):
        edges = [
            {"predicate": "hasRisk"},
            {"predicate": "leadsTo"},
            {"predicate": "impacts"},
            {"object_type": "RiskControl"},
        ]
        errors = ontology.validate_risk_chain_completeness(edges)
        assert any("mitigate" in e for e in errors)

    def test_risk_control_with_mitigates_passes(self, ontology):
        edges = [
            {"predicate": "hasRisk"},
            {"predicate": "leadsTo"},
            {"predicate": "impacts"},
            {"object_type": "RiskControl"},
            {"predicate": "mitigates"},
        ]
        errors = ontology.validate_risk_chain_completeness(edges)
        assert errors == []

    def test_empty_edges_fails(self, ontology):
        errors = ontology.validate_risk_chain_completeness([])
        assert any("at least one Risk" in e for e in errors)

    def test_uses_enum_values(self, ontology):
        """Should also work with RelationType enum values."""
        edges = [
            {"predicate": RelationType.HAS_RISK.value},
            {"predicate": RelationType.LEADS_TO.value},
            {"predicate": RelationType.IMPACTS.value},
        ]
        errors = ontology.validate_risk_chain_completeness(edges)
        assert errors == []


# ============================================================
# get_ontology singleton
# ============================================================


class TestGetOntology:
    def test_returns_instance(self):
        ont = get_ontology()
        assert isinstance(ont, AIROOntology)

    def test_returns_same_instance(self):
        ont1 = get_ontology()
        ont2 = get_ontology()
        assert ont1 is ont2

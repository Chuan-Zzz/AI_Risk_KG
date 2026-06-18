"""AI Risk Knowledge Graph - Core Data Models.

Pydantic models aligned with the AIRO Extended Ontology (airo_extended_en.ttl).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# Enumerations
# ============================================================

class ExtractionMode(str, Enum):
    EXPLICIT = "explicit"
    ABSTRACTED = "abstracted"
    INFERRED = "inferred"
    COMPLETED = "completed"


class OntologyClass(str, Enum):
    # Technical
    AI_SYSTEM = "AISystem"
    AI_MODEL = "AIModel"
    GPAI_MODEL = "GPAIModel"
    AI_TECHNIQUE = "AITechnique"
    AI_CAPABILITY = "AICapability"
    DATA = "Data"
    # Stakeholders
    STAKEHOLDER = "Stakeholder"
    AI_DEVELOPER = "AIDeveloper"
    AI_PROVIDER = "AIProvider"
    AI_DEPLOYER = "AIDeployer"
    AI_USER = "AIUser"
    REGULATOR = "Regulator"
    AFFECTED_ACTOR = "AffectedActor"
    # Risk
    RISK_SOURCE = "RiskSource"
    RISK = "Risk"
    MISUSE = "Misuse"
    HAZARD = "Hazard"
    THREAT = "Threat"
    VULNERABILITY = "Vulnerability"
    CONSEQUENCE = "Consequence"
    IMPACT = "Impact"
    RISK_CONTROL = "RiskControl"
    # Governance
    REGULATION = "Regulation"
    STANDARD = "Standard"
    OBLIGATION = "Obligation"
    COMPLIANCE_REQUIREMENT = "ComplianceRequirement"
    ENFORCEMENT_ACTION = "EnforcementAction"
    CODE_OF_CONDUCT = "CodeOfConduct"
    # Evidence
    NEWS_REPORT = "NewsReport"
    INFORMATION_SOURCE = "InformationSource"
    EVIDENCE = "Evidence"
    KNOWLEDGE_STATEMENT = "KnowledgeStatement"
    # Event
    AI_RISK_INCIDENT = "AIRiskIncident"
    ROLE_ASSIGNMENT = "RoleAssignment"
    # Auxiliary
    EXTRACTION_MODE = "ExtractionMode"
    TREND = "Trend"
    PURPOSE = "Purpose"
    AI_LIFECYCLE_PHASE = "AILifecyclePhase"
    DOMAIN = "Domain"
    DOCUMENTATION = "Documentation"
    TECHNICAL_DOCUMENTATION = "TechnicalDocumentation"
    VERIFICATION_TEST = "VerificationTest"


class RelationType(str, Enum):
    # Technical
    DEVELOPS = "develops"
    PROVIDES = "provides"
    DEPLOYS = "deploys"
    USES = "uses"
    HAS_MODEL = "hasModel"
    USES_TECHNIQUE = "usesTechnique"
    HAS_CAPABILITY = "hasCapability"
    # Event core
    INVOLVES_AI_SYSTEM = "involvesAISystem"
    INVOLVES_STAKEHOLDER = "involvesStakeholder"
    HAS_RISK = "hasRisk"
    HAS_CONSEQUENCE = "hasConsequence"
    HAS_IMPACT = "hasImpact"
    HAS_REPORT = "hasReport"
    # Role
    ROLE_HELD_BY = "roleHeldBy"
    HAS_ROLE_IN_INCIDENT = "hasRoleInIncident"
    ROLE_INVOLVES_SYSTEM = "roleInvolvesSystem"
    # Risk propagation chain
    CAUSES = "causes"
    LEADS_TO = "leadsTo"
    IMPACTS = "impacts"
    AFFECTS = "affects"
    MITIGATES = "mitigates"
    # Governance
    GOVERNS = "governs"
    IMPOSES_REQUIREMENT = "imposesRequirement"
    ENFORCES = "enforces"
    INVESTIGATES = "investigates"
    SPECIFIES = "specifies"
    RESPONDS_TO_INCIDENT = "respondsToIncident"
    # Evidence
    HAS_EVIDENCE = "hasEvidence"
    HAS_SOURCE_DOCUMENT = "hasSourceDocument"
    # Multi-document
    CONFLICTS_WITH = "conflictsWith"
    # Original AIRO
    IS_RISK_SOURCE_FOR = "isRiskSourceFor"
    HAS_HAZARD = "hasHazard"
    HAS_THREAT = "hasThreat"
    HAS_VULNERABILITY = "hasVulnerability"
    HAS_INFORMATION_SOURCE = "hasInformationSource"
    COMPLIES_WITH_REGULATION = "compliesWithRegulation"
    CONFORMS_TO_STANDARD = "conformsToStandard"


# ============================================================
# Class aliases for normalization
# ============================================================

CLASS_ALIASES: dict[str, str] = {
    "ai system": "AISystem",
    "ai_system": "AISystem",
    "system": "AISystem",
    "ai model": "AIModel",
    "ai_model": "AIModel",
    "model": "AIModel",
    "llm": "AIModel",
    "large language model": "AIModel",
    "foundation model": "AIModel",
    "基础模型": "AIModel",
    "大模型": "AIModel",
    "ai technique": "AITechnique",
    "ai_technique": "AITechnique",
    "technique": "AITechnique",
    "ai capability": "AICapability",
    "ai_capability": "AICapability",
    "capability": "AICapability",
    "stakeholder": "Stakeholder",
    "organization": "Stakeholder",
    "organisation": "Stakeholder",
    "company": "Stakeholder",
    "entity": "Stakeholder",
    "ai developer": "AIDeveloper",
    "ai_developer": "AIDeveloper",
    "developer": "AIDeveloper",
    "ai provider": "AIProvider",
    "ai_provider": "AIProvider",
    "provider": "AIProvider",
    "ai deployer": "AIDeployer",
    "ai_deployer": "AIDeployer",
    "deployer": "AIDeployer",
    "ai user": "AIUser",
    "ai_user": "AIUser",
    "user": "AIUser",
    "ai subject": "AffectedActor",
    "ai_subject": "AffectedActor",
    "regulator": "Regulator",
    "supervisory authority": "Regulator",
    "affected actor": "AffectedActor",
    "affected_actor": "AffectedActor",
    "affected party": "AffectedActor",
    "victim": "AffectedActor",
    "risk source": "RiskSource",
    "risk_source": "RiskSource",
    "risk": "Risk",
    "hazard": "Hazard",
    "threat": "Threat",
    "vulnerability": "Vulnerability",
    "consequence": "Consequence",
    "impact": "Impact",
    "risk control": "RiskControl",
    "risk_control": "RiskControl",
    "mitigation": "RiskControl",
    "control measure": "RiskControl",
    "regulation": "Regulation",
    "law": "Regulation",
    "act": "Regulation",
    "standard": "Standard",
    "obligation": "Obligation",
    "compliance requirement": "ComplianceRequirement",
    "enforcement action": "EnforcementAction",
    "enforcement": "EnforcementAction",
    "penalty": "EnforcementAction",
    "fine": "EnforcementAction",
    "news report": "NewsReport",
    "news_report": "NewsReport",
    "report": "NewsReport",
    "article": "NewsReport",
    "information source": "InformationSource",
    "evidence": "Evidence",
    "knowledge statement": "KnowledgeStatement",
    "ai risk incident": "AIRiskIncident",
    "incident": "AIRiskIncident",
    "event": "AIRiskIncident",
    "role assignment": "RoleAssignment",
    "trend": "Trend",
    "purpose": "Purpose",
    "lifecycle phase": "AILifecyclePhase",
    "lifecycle": "AILifecyclePhase",
    "domain": "Domain",
    "impact domain": "Domain",
    "area of impact": "Domain",
    "code of conduct": "CodeOfConduct",
    "documentation": "Documentation",
    "technical documentation": "TechnicalDocumentation",
    "verification test": "VerificationTest",
}

RELATION_ALIASES: dict[str, str] = {
    "developed_by": "develops",
    "developed by": "develops",
    "created_by": "develops",
    "built_by": "develops",
    "trained_by": "develops",
    "provided_by": "provides",
    "provided by": "provides",
    "deployed_by": "deploys",
    "deployed by": "deploys",
    "used_by": "uses",
    "used by": "uses",
    "has_risk_source": "isRiskSourceFor",
    "involves_ai_system": "involvesAISystem",
    "involves ai system": "involvesAISystem",
    "involves_stakeholder": "involvesStakeholder",
    "has_report": "hasReport",
    "role_held_by": "roleHeldBy",
    "has_role_in_incident": "hasRoleInIncident",
    "role_involves_system": "roleInvolvesSystem",
    "leads_to": "leadsTo",
    "leads to": "leadsTo",
    "results_in": "leadsTo",
    "causes": "causes",
    "impacts": "impacts",
    "affects": "affects",
    "mitigates": "mitigates",
    "governs": "governs",
    "governed_by": "governs",
    "governed by": "governs",
    "regulates": "governs",
    "imposes_requirement": "imposesRequirement",
    "enforces": "enforces",
    "investigates": "investigates",
    "specifies": "specifies",
    "responds_to_incident": "respondsToIncident",
    "has_evidence": "hasEvidence",
    "has_source_document": "hasSourceDocument",
    "conflicts_with": "conflictsWith",
    "has_hazard": "hasHazard",
    "has_threat": "hasThreat",
    "has_vulnerability": "hasVulnerability",
    "complies_with_regulation": "compliesWithRegulation",
    "conforms_to_standard": "conformsToStandard",
}


# ============================================================
# Data Models
# ============================================================

class NewsReport(BaseModel):
    doc_id: str
    event_id: str
    title: str
    content: str
    summary: str | None = None
    source_name: str | None = None
    url: str | None = None
    publish_time: str | None = None
    language: str = "en"


class MentionSpan(BaseModel):
    """精确的实体提及位置信息。"""
    text: str  # 原文中精确的文本
    start: int  # 字符起始位置（包含）
    end: int  # 字符结束位置（不包含）


class EvidenceItem(BaseModel):
    evidence_id: str
    evidence_sentence: str
    source_doc_id: str
    mention_span: MentionSpan | None = None  # 结构化的位置信息
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EntityNode(BaseModel):
    id: str
    name: str
    entity_type: OntologyClass
    description: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    extraction_mode: ExtractionMode = ExtractionMode.EXPLICIT
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    support_count: int = Field(default=1, ge=1)
    source_doc_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None


class RelationEdge(BaseModel):
    id: str
    subject_id: str
    subject_name: str = ""
    predicate: RelationType
    object_id: str
    object_name: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    extraction_mode: ExtractionMode = ExtractionMode.EXPLICIT
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    support_count: int = Field(default=1, ge=1)
    source_doc_ids: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class KnowledgeStatement(BaseModel):
    id: str
    subject_id: str
    predicate: RelationType
    object_id: str
    evidence: EvidenceItem
    extraction_mode: ExtractionMode
    confidence: float = Field(ge=0.0, le=1.0)
    support_count: int = Field(default=1, ge=1)


class EventKnowledgeSubgraph(BaseModel):
    event_id: str
    incident_node: EntityNode
    nodes: list[EntityNode] = Field(default_factory=list)
    edges: list[RelationEdge] = Field(default_factory=list)
    knowledge_statements: list[KnowledgeStatement] = Field(default_factory=list)
    support_statistics: dict[str, int] = Field(default_factory=dict)
    report_count: int = 0
    first_seen: str | None = None
    last_seen: str | None = None

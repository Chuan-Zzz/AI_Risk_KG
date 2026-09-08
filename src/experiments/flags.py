"""Experiment flag helpers for ablation and baseline runs."""

from __future__ import annotations

from typing import Any


ABLATION_VARIANTS = (
    "full",
    "no_ontology",
    "no_agg",
    "no_evidence",
    "no_inference",
)


VARIANT_FLAGS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_ontology": {"disable_ontology_constraint": True},
    "no_agg": {"disable_event_aggregation": True},
    "no_evidence": {"disable_evidence_constraint": True},
    "no_inference": {"disable_controlled_inference": True},
}


VARIANT_OUTPUT_DIRS: dict[str, str] = {
    "full": "full_ontorisk",
    "no_ontology": "ablation_no_ontology",
    "no_agg": "ablation_no_event_aggregation",
    "no_evidence": "ablation_no_evidence_constraint",
    "no_inference": "ablation_no_controlled_inference",
}


def build_experiment_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANT_FLAGS:
        raise ValueError(f"Unsupported experiment variant: {variant}")
    return {"variant": variant, **VARIANT_FLAGS[variant]}


def get_experiment_config(state_or_config: dict[str, Any] | None) -> dict[str, Any]:
    if not state_or_config:
        return {}
    if "experiment" in state_or_config and isinstance(state_or_config["experiment"], dict):
        return state_or_config["experiment"]
    return state_or_config


def is_experiment_enabled(state_or_config: dict[str, Any] | None, flag: str) -> bool:
    config = get_experiment_config(state_or_config)
    return bool(config.get(flag, False))

"""LangGraph StateGraph definition for the 5-stage extraction pipeline."""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from src.agent.state import PipelineState
from src.agent.nodes.input_node import input_node
from src.agent.nodes.explicit_extract import explicit_extract_node
from src.agent.nodes.aggregation import aggregation_node
from src.agent.nodes.entity_reuse import entity_reuse_node
from src.agent.nodes.risk_chain import risk_chain_node
from src.agent.nodes.inference import inference_node
from src.agent.nodes.graph_build import graph_build_node
from src.agent.nodes.validation import validation_node
from src.agent.nodes.store import store_node

logger = logging.getLogger(__name__)

MAX_VALIDATION_RETRIES = 3


def _route_after_validation(state: PipelineState) -> str:
    if state.get("validation_passed", False):
        return "store"
    retry = state.get("retry_count", 0)
    if retry >= MAX_VALIDATION_RETRIES:
        logger.warning(f"Max validation retries ({MAX_VALIDATION_RETRIES}) reached, storing anyway")
        return "store"
    return "graph_build"


def build_pipeline_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("input", input_node)
    graph.add_node("explicit_extract", explicit_extract_node)
    graph.add_node("aggregation", aggregation_node)
    graph.add_node("entity_reuse", entity_reuse_node)
    graph.add_node("risk_chain", risk_chain_node)
    graph.add_node("inference", inference_node)
    graph.add_node("graph_build", graph_build_node)
    graph.add_node("validation", validation_node)
    graph.add_node("store", store_node)

    # Linear edges: Stage 1 → 2 → 3 → 4(L1) → 4(L2) → 4(L3) → 5
    graph.add_edge("input", "explicit_extract")
    graph.add_edge("explicit_extract", "aggregation")
    graph.add_edge("aggregation", "entity_reuse")
    graph.add_edge("entity_reuse", "risk_chain")
    graph.add_edge("risk_chain", "inference")
    graph.add_edge("inference", "graph_build")
    graph.add_edge("graph_build", "validation")

    # Conditional: validation passed → store, failed → retry graph_build
    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {"store": "store", "graph_build": "graph_build"},
    )

    graph.add_edge("store", END)

    # Entry point
    graph.set_entry_point("input")

    return graph


def compile_pipeline():
    graph = build_pipeline_graph()
    return graph.compile()

"""Tests for Stage 1 - Input Node (input_node.py).

Covers: single document, multiple documents, empty documents,
first_seen/last_seen computation, deduplication of sources and languages.
"""

from __future__ import annotations

import pytest

from src.agent.nodes.input_node import input_node
from src.agent.state import PipelineState, create_initial_state
from src.core.models import NewsReport


def _make_report(
    doc_id: str,
    event_id: str = "evt1",
    title: str = "Test",
    content: str = "Content",
    source_name: str | None = None,
    language: str = "en",
    publish_time: str | None = None,
) -> NewsReport:
    return NewsReport(
        doc_id=doc_id,
        event_id=event_id,
        title=title,
        content=content,
        source_name=source_name,
        language=language,
        publish_time=publish_time,
    )


class TestInputNodeSingleDocument:
    def test_report_count_is_one(self):
        doc = _make_report("d1", source_name="BBC", language="en",
                           publish_time="2025-01-15T10:00:00Z")
        state = create_initial_state("evt1", [doc])
        result = input_node(state)
        assert result["report_count"] == 1

    def test_first_seen_equals_publish_time(self):
        doc = _make_report("d1", publish_time="2025-01-15T10:00:00Z")
        state = create_initial_state("evt1", [doc])
        result = input_node(state)
        assert result["first_seen"] == "2025-01-15T10:00:00Z"

    def test_last_seen_equals_publish_time(self):
        doc = _make_report("d1", publish_time="2025-01-15T10:00:00Z")
        state = create_initial_state("evt1", [doc])
        result = input_node(state)
        assert result["last_seen"] == "2025-01-15T10:00:00Z"

    def test_source_set_contains_source(self):
        doc = _make_report("d1", source_name="BBC")
        state = create_initial_state("evt1", [doc])
        result = input_node(state)
        assert result["source_set"] == ["BBC"]

    def test_language_set_contains_language(self):
        doc = _make_report("d1", language="en")
        state = create_initial_state("evt1", [doc])
        result = input_node(state)
        assert result["language_set"] == ["en"]

    def test_status_is_processing(self):
        doc = _make_report("d1")
        state = create_initial_state("evt1", [doc])
        result = input_node(state)
        assert result["status"] == "PROCESSING"

    def test_current_stage_is_input(self):
        doc = _make_report("d1")
        state = create_initial_state("evt1", [doc])
        result = input_node(state)
        assert result["current_stage"] == "input"

    def test_stages_completed_includes_input(self):
        doc = _make_report("d1")
        state = create_initial_state("evt1", [doc])
        result = input_node(state)
        assert "input" in result["stages_completed"]

    def test_no_publish_time_first_seen_is_none(self):
        doc = _make_report("d1")
        state = create_initial_state("evt1", [doc])
        result = input_node(state)
        assert result["first_seen"] is None
        assert result["last_seen"] is None

    def test_no_source_name_source_set_is_empty(self):
        doc = _make_report("d1")
        state = create_initial_state("evt1", [doc])
        result = input_node(state)
        assert result["source_set"] == []


class TestInputNodeMultipleDocuments:
    def test_report_count_is_three(self):
        docs = [
            _make_report("d1", source_name="BBC"),
            _make_report("d2", source_name="CNN"),
            _make_report("d3", source_name="Reuters"),
        ]
        state = create_initial_state("evt1", docs)
        result = input_node(state)
        assert result["report_count"] == 3

    def test_first_seen_is_earliest(self):
        docs = [
            _make_report("d1", publish_time="2025-01-20T10:00:00Z"),
            _make_report("d2", publish_time="2025-01-15T10:00:00Z"),
            _make_report("d3", publish_time="2025-01-18T10:00:00Z"),
        ]
        state = create_initial_state("evt1", docs)
        result = input_node(state)
        assert result["first_seen"] == "2025-01-15T10:00:00Z"

    def test_last_seen_is_latest(self):
        docs = [
            _make_report("d1", publish_time="2025-01-20T10:00:00Z"),
            _make_report("d2", publish_time="2025-01-15T10:00:00Z"),
            _make_report("d3", publish_time="2025-01-18T10:00:00Z"),
        ]
        state = create_initial_state("evt1", docs)
        result = input_node(state)
        assert result["last_seen"] == "2025-01-20T10:00:00Z"

    def test_deduplicated_sources(self):
        docs = [
            _make_report("d1", source_name="BBC"),
            _make_report("d2", source_name="BBC"),
            _make_report("d3", source_name="CNN"),
        ]
        state = create_initial_state("evt1", docs)
        result = input_node(state)
        assert result["source_set"] == ["BBC", "CNN"]

    def test_deduplicated_languages(self):
        docs = [
            _make_report("d1", language="en"),
            _make_report("d2", language="en"),
            _make_report("d3", language="zh"),
        ]
        state = create_initial_state("evt1", docs)
        result = input_node(state)
        assert result["language_set"] == ["en", "zh"]

    def test_sources_are_sorted(self):
        docs = [
            _make_report("d1", source_name="Reuters"),
            _make_report("d2", source_name="BBC"),
            _make_report("d3", source_name="Al Jazeera"),
        ]
        state = create_initial_state("evt1", docs)
        result = input_node(state)
        assert result["source_set"] == ["Al Jazeera", "BBC", "Reuters"]

    def test_languages_are_sorted(self):
        docs = [
            _make_report("d1", language="zh"),
            _make_report("d2", language="en"),
            _make_report("d3", language="fr"),
        ]
        state = create_initial_state("evt1", docs)
        result = input_node(state)
        assert result["language_set"] == ["en", "fr", "zh"]

    def test_mixed_none_and_present_publish_times(self):
        docs = [
            _make_report("d1", publish_time="2025-01-10T10:00:00Z"),
            _make_report("d2"),
            _make_report("d3", publish_time="2025-01-20T10:00:00Z"),
        ]
        state = create_initial_state("evt1", docs)
        result = input_node(state)
        assert result["first_seen"] == "2025-01-10T10:00:00Z"
        assert result["last_seen"] == "2025-01-20T10:00:00Z"


class TestInputNodeEmptyDocuments:
    def test_report_count_is_zero(self):
        state = create_initial_state("evt1", [])
        result = input_node(state)
        assert result["report_count"] == 0

    def test_first_seen_is_none(self):
        state = create_initial_state("evt1", [])
        result = input_node(state)
        assert result["first_seen"] is None

    def test_last_seen_is_none(self):
        state = create_initial_state("evt1", [])
        result = input_node(state)
        assert result["last_seen"] is None

    def test_source_set_is_empty(self):
        state = create_initial_state("evt1", [])
        result = input_node(state)
        assert result["source_set"] == []

    def test_language_set_is_empty(self):
        state = create_initial_state("evt1", [])
        result = input_node(state)
        assert result["language_set"] == []


class TestInputNodeEdgeCases:
    def test_preserves_existing_stages_completed(self):
        doc = _make_report("d1")
        state = create_initial_state("evt1", [doc])
        state["stages_completed"] = ["previous_stage"]
        result = input_node(state)
        assert "previous_stage" in result["stages_completed"]
        assert "input" in result["stages_completed"]

    def test_missing_documents_key_in_state(self):
        state = PipelineState(event_id="evt1")
        result = input_node(state)
        assert result["report_count"] == 0

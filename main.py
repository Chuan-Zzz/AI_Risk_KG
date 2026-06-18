"""AI Risk Knowledge Graph - Event-centric Extraction Pipeline.

Two-phase architecture:
  Phase 1: Per-event extraction (independent, no cross-event alignment)
  Phase 2: Global entity resolution + Neo4j assembly

Usage:
    python main.py single --event_id pred_0
    python main.py batch --max_events 10
    python main.py batch --all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import setup_logger
from src.core.config import get_config
from src.core.models import NewsReport, EventKnowledgeSubgraph
from src.agent.state import create_initial_state
from src.agent.graph import compile_pipeline

logger = setup_logger()

_EVENTS_FILE = Path(__file__).resolve().parent / "data" / "inferred_event_structure_6124.json"
_CASES_FILE = Path(__file__).resolve().parent / "data" / "eval_cases.jsonl"

_SAFE_ID_RE = re.compile(r"^[\w\-]+$")


def _sanitize_event_id(event_id: str) -> str:
    if not _SAFE_ID_RE.match(event_id):
        raise ValueError(f"Invalid event_id: {event_id!r}")
    return event_id


def load_cases() -> dict[str, dict]:
    cases: dict[str, dict] = {}
    with open(_CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            case_id = str(obj.get("id") or obj.get("original_id", ""))
            if case_id:
                cases[case_id] = obj
    return cases


def load_events() -> list[dict]:
    with open(_EVENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events", [])


def build_documents_for_event(event: dict, cases: dict[str, dict]) -> list[NewsReport]:
    documents: list[NewsReport] = []
    event_id = event.get("incident_id", "unknown")
    case_ids = event.get("ids", [])

    for cid in case_ids:
        case = cases.get(cid)
        if not case:
            continue

        content = case.get("text", "") or case.get("new_summary", "") or case.get("description", "") or ""
        title = case.get("title", "")
        summary = case.get("new_summary", "") or case.get("summary", "")

        report = NewsReport(
            doc_id=str(cid),
            event_id=event_id,
            title=title,
            content=content,
            summary=summary if summary else None,
            source_name=case.get("from_database") or case.get("source"),
            url=case.get("case_link") or case.get("from_url"),
            publish_time=case.get("release_date"),
            language="en" if case.get("from_database") == "aiid" else "zh",
        )
        documents.append(report)

    return documents


def run_single(event_id: str) -> None:
    event_id = _sanitize_event_id(event_id)
    cases = load_cases()
    events = load_events()

    event = None
    for e in events:
        if e.get("incident_id") == event_id:
            event = e
            break

    if event is None:
        logger.error(f"Event {event_id} not found")
        return

    documents = build_documents_for_event(event, cases)
    if not documents:
        logger.error(f"No documents found for event {event_id}")
        return

    logger.info(f"Running pipeline for event {event_id} ({len(documents)} documents)")
    start = time.time()

    pipeline = compile_pipeline()
    state = create_initial_state(event_id, documents)
    result = pipeline.invoke(state)

    elapsed = time.time() - start
    status = result.get("status", "unknown")
    logger.info(f"Event {event_id}: {status} in {elapsed:.1f}s")


def run_batch(max_events: int | None = None, all_events: bool = False) -> None:
    """Two-phase batch processing.

    Phase 1: Extract all event subgraphs independently (no cross-event alignment)
    Phase 2: Global entity resolution + Neo4j assembly
    """
    cases = load_cases()
    events = load_events()

    if not all_events and max_events:
        events = events[:max_events]

    logger.info(f"=== Phase 1: Per-event extraction ({len(events)} events) ===")

    pipeline = compile_pipeline()
    output_dir = get_config().project_root / get_config().get("output.dir", "output")

    # Phase 1: Extract all event subgraphs independently
    subgraphs: dict[str, EventKnowledgeSubgraph] = {}
    success, failed, skipped = 0, 0, 0

    for i, event in enumerate(events, 1):
        event_id = _sanitize_event_id(event.get("incident_id", f"event_{i}"))

        # Try loading existing subgraph from Phase 1 output
        event_output = output_dir / event_id / "event_subgraph.json"
        if event_output.exists() and not all_events:
            try:
                with open(event_output, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sg = EventKnowledgeSubgraph(**data)
                subgraphs[event_id] = sg
                skipped += 1
                continue
            except Exception as e:
                logger.warning(f"Could not load existing subgraph for {event_id}: {e}")

        documents = build_documents_for_event(event, cases)
        if not documents:
            logger.warning(f"[{i}/{len(events)}] No documents for {event_id}")
            failed += 1
            continue

        try:
            # No registry — Phase 1 is independent extraction
            state = create_initial_state(event_id, documents)
            result = pipeline.invoke(state)
            status = result.get("status", "unknown")

            sg = result.get("event_subgraph")
            if sg is not None and status == "COMPLETED":
                subgraphs[event_id] = sg
                success += 1

                # Save Phase 1 output
                event_output.parent.mkdir(parents=True, exist_ok=True)
                with open(event_output, "w", encoding="utf-8") as f:
                    json.dump(sg.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
            else:
                failed += 1

            logger.info(f"[{i}/{len(events)}] {event_id}: {status}")
        except Exception as e:
            logger.error(f"[{i}/{len(events)}] {event_id} FAILED: {e}")
            failed += 1

    logger.info(
        f"Phase 1 complete: {success} extracted, {skipped} cached, "
        f"{failed} failed, {len(subgraphs)} total subgraphs"
    )

    if not subgraphs:
        logger.error("No subgraphs to fuse")
        return

    # Phase 2: Global entity resolution + Neo4j assembly
    logger.info(f"=== Phase 2: Global entity resolution ({len(subgraphs)} subgraphs) ===")

    from src.alignment.kg_fusion import fuse_subgraphs, write_to_neo4j
    from src.core.config import get_config

    cfg = get_config()
    threshold = cfg.get("alignment.cross_event_threshold", 0.85)
    llm_top_k = cfg.get("alignment.llm_verify_top_k", 5)

    resolved = fuse_subgraphs(subgraphs, threshold=threshold, llm_verify_top_k=llm_top_k)

    # Save resolved subgraphs
    for event_id, sg in resolved.items():
        event_output = output_dir / event_id / "event_subgraph.json"
        event_output.parent.mkdir(parents=True, exist_ok=True)
        with open(event_output, "w", encoding="utf-8") as f:
            json.dump(sg.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    # Write to Neo4j
    write_to_neo4j(resolved)

    logger.info(f"=== Batch complete: {len(resolved)} events fused into global KG ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Risk Knowledge Graph Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    single_p = subparsers.add_parser("single", help="Process a single event")
    single_p.add_argument("--event_id", required=True, help="Event ID (e.g., pred_0)")

    batch_p = subparsers.add_parser("batch", help="Batch process events")
    batch_p.add_argument("--max_events", type=int, default=None, help="Max events to process")
    batch_p.add_argument("--all", action="store_true", help="Process all events (including already done)")

    args = parser.parse_args()

    if args.command == "single":
        run_single(args.event_id)
    elif args.command == "batch":
        run_batch(max_events=args.max_events, all_events=args.all)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

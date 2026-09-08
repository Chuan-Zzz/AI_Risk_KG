"""AI Risk Knowledge Graph - Event-centric Extraction Pipeline.

Two-phase architecture:
  Phase 1: Per-event extraction (independent, no cross-event alignment)
  Phase 2: Global entity resolution + Neo4j assembly

Usage:
    python main.py single --event_id pred_0
    python main.py batch --dataset all --max_events 10
    python main.py batch --dataset all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import setup_logger
from src.core.config import get_config
from src.core.datasets import build_documents_for_event, load_cases, load_events
from src.core.models import EventKnowledgeSubgraph
from src.core.runtime import event_id_for, output_root, sanitize_event_id
from src.agent.state import create_initial_state

logger = setup_logger()

def run_single(event_id: str) -> None:
    event_id = sanitize_event_id(event_id)
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

    from src.agent.graph import compile_pipeline

    pipeline = compile_pipeline()
    state = create_initial_state(event_id, documents)
    result = pipeline.invoke(state)

    elapsed = time.time() - start
    status = result.get("status", "unknown")
    logger.info(f"Event {event_id}: {status} in {elapsed:.1f}s")


def run_batch(
    max_events: int | None = None,
    force: bool = False,
    dataset: str = "all",
) -> None:
    """Two-phase batch processing.

    Phase 1: Extract all event subgraphs independently (no cross-event alignment)
    Phase 2: Global entity resolution + Neo4j assembly
    """
    if max_events is not None and max_events <= 0:
        raise ValueError("max_events must be a positive integer")

    cases = load_cases()
    events = load_events(dataset)
    if max_events is not None:
        events = events[:max_events]

    logger.info(f"=== Phase 1: Per-event extraction ({len(events)} events) ===")

    from src.agent.graph import compile_pipeline

    pipeline = compile_pipeline()
    output_dir = output_root()
    cfg = get_config()

    # Phase 1: Extract all event subgraphs independently
    subgraphs: dict[str, EventKnowledgeSubgraph] = {}
    success, failed, skipped = 0, 0, 0

    for i, event in enumerate(events, 1):
        event_id = event_id_for(event, default=f"event_{i}")

        documents = build_documents_for_event(event, cases)
        if not documents:
            logger.warning(f"[{i}/{len(events)}] No documents for {event_id}")
            failed += 1
            continue

        # Cached results are checked against the current validation policy.
        event_output = output_dir / event_id / "event_subgraph.json"
        if event_output.exists() and not force:
            try:
                with open(event_output, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sg = EventKnowledgeSubgraph(**data)
                from src.agent.nodes.validation import validation_node

                cached_state = create_initial_state(event_id, documents)
                cached_state["event_subgraph"] = sg
                validation = validation_node(cached_state)
                if validation["validation_passed"] or not cfg.get("validation.strict_mode", True):
                    subgraphs[event_id] = sg
                    skipped += 1
                    continue
                logger.warning("Cached subgraph for %s no longer passes validation; rebuilding", event_id)
            except Exception as e:
                logger.warning(f"Could not load existing subgraph for {event_id}: {e}")

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
    threshold = cfg.get("alignment.cross_event_threshold", 0.85)
    llm_top_k = cfg.get("alignment.llm_verify_top_k", 5)

    resolved = fuse_subgraphs(subgraphs, threshold=threshold, llm_verify_top_k=llm_top_k)

    # Save resolved subgraphs to a separate file so the original Phase 1 output
    # (event_subgraph.json) is preserved for re-running Phase 2 with different
    # alignment thresholds without re-extracting.
    for event_id, sg in resolved.items():
        event_output = output_dir / event_id / "event_subgraph_fused.json"
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
    batch_p.add_argument("--max_events", type=int, default=None, help="Maximum events to process")
    batch_p.add_argument("--dataset", choices=("all", "gold", "eval500"), default="all")
    batch_p.add_argument("--force", action="store_true", help="Reprocess events that already have output")
    batch_p.add_argument("--all", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    try:
        if args.command == "single":
            run_single(args.event_id)
        elif args.command == "batch":
            run_batch(max_events=args.max_events, force=args.force or args.all, dataset=args.dataset)
        else:
            parser.print_help()
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

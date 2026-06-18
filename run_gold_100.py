"""Run pipeline on the 100 gold standard events (force re-extract, no cache)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.logger import setup_logger
from src.core.config import get_config
from src.core.models import NewsReport
from src.agent.state import create_initial_state
from src.agent.graph import compile_pipeline

logger = setup_logger()

GOLD_FILE = Path(__file__).resolve().parent / "data" / "gold_standard_100_events.json"
CASES_FILE = Path(__file__).resolve().parent / "data" / "eval_cases.jsonl"
EVENTS_FILE = Path(__file__).resolve().parent / "data" / "inferred_event_structure_6124.json"


def main() -> None:
    # Load gold standard event IDs
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        gold = json.load(f)
    gold_ids = {e["event_id"] for e in gold.get("gold_standard", [])}
    logger.info(f"Gold standard: {len(gold_ids)} events")

    # Load cases
    cases: dict[str, dict] = {}
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = str(obj.get("id") or obj.get("original_id", ""))
            if cid:
                cases[cid] = obj
    logger.info(f"Loaded {len(cases)} cases")

    # Load all events, filter to gold standard
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        all_events = json.load(f).get("events", [])
    filtered = [e for e in all_events if e.get("incident_id") in gold_ids]
    logger.info(f"Filtered to {len(filtered)} events from gold standard")

    # Compile pipeline
    pipeline = compile_pipeline()
    output_dir = get_config().project_root / get_config().get("output.dir", "output")

    success, failed, skipped = 0, 0, 0
    results = []

    for i, event in enumerate(filtered, 1):
        event_id = event.get("incident_id", f"gold_{i}")

        # Skip if already exists
        event_output = output_dir / event_id / "event_subgraph.json"
        if event_output.exists():
            skipped += 1
            results.append({"event_id": event_id, "status": "SKIPPED"})
            if i % 10 == 0:
                logger.info(f"[{i}/{len(filtered)}] progress: {success} OK, {skipped} skipped, {failed} failed")
            continue

        # Build documents
        case_ids = event.get("ids", [])
        documents = []
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

        if not documents:
            failed += 1
            results.append({"event_id": event_id, "status": "NO_DOCS"})
            logger.warning(f"[{i}/{len(filtered)}] {event_id}: NO DOCS")
            continue

        try:
            start = time.time()
            state = create_initial_state(event_id, documents)
            result = pipeline.invoke(state)
            elapsed = time.time() - start
            status = result.get("status", "unknown")

            if status == "COMPLETED":
                success += 1
                results.append({"event_id": event_id, "status": "OK", "elapsed": round(elapsed, 1)})
            else:
                failed += 1
                results.append({"event_id": event_id, "status": f"FAILED:{status}"})

            logger.info(f"[{i}/{len(filtered)}] {event_id}: {status} ({elapsed:.1f}s)")
        except Exception as e:
            failed += 1
            results.append({"event_id": event_id, "status": f"ERROR:{e}"})
            logger.error(f"[{i}/{len(filtered)}] {event_id}: ERROR: {e}")

    logger.info(f"=== DONE: {success} success, {skipped} skipped, {failed} failed out of {len(filtered)} ===")


if __name__ == "__main__":
    main()

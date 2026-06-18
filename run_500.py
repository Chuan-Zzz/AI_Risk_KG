"""Batch process 500 events from eval_dataset_500.json with concurrent execution."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

_EVAL_DATASET = Path(__file__).resolve().parent / "data" / "gold_standard_100_events.json"
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


def load_eval_events() -> list[dict]:
    with open(_EVAL_DATASET, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("gold_standard", data.get("events", []))


def build_documents_for_event(event: dict, cases: dict[str, dict]) -> list[NewsReport]:
    documents: list[NewsReport] = []
    event_id = event.get("event_id") or event.get("incident_id", "unknown")
    case_ids = event.get("ids", event.get("labeling_cases", []))

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


def process_single_event(event: dict, cases: dict[str, dict], pipeline, output_dir: Path) -> dict:
    """Process a single event. Returns a status dict."""
    event_id = _sanitize_event_id(event.get("event_id") or event.get("incident_id", "unknown"))

    event_output = output_dir / event_id / "event_subgraph.json"
    if event_output.exists():
        return {"event_id": event_id, "status": "cached", "elapsed": 0}

    documents = build_documents_for_event(event, cases)
    if not documents:
        return {"event_id": event_id, "status": "no_documents", "elapsed": 0}

    start = time.time()
    try:
        state = create_initial_state(event_id, documents)
        result = pipeline.invoke(state)
        status = result.get("status", "unknown")

        sg = result.get("event_subgraph")
        if sg is not None and status == "COMPLETED":
            event_output.parent.mkdir(parents=True, exist_ok=True)
            with open(event_output, "w", encoding="utf-8") as f:
                json.dump(sg.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start
        return {"event_id": event_id, "status": status, "elapsed": round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - start
        return {"event_id": event_id, "status": "FAILED", "error": str(e), "elapsed": round(elapsed, 1)}


def main() -> None:
    cases = load_cases()
    events = load_eval_events()

    logger.info(f"Loaded {len(events)} events, {len(cases)} cases")

    cfg = get_config()
    output_dir = cfg.project_root / cfg.get("output.dir", "output")
    parallel_workers = cfg.get("extraction.parallel_workers", 5)

    # Run one event at a time to avoid connection pool exhaustion
    pipeline_workers = 1

    logger.info(f"Processing {len(events)} events with {pipeline_workers} concurrent pipelines")

    pipeline = compile_pipeline()

    success, cached, failed = 0, 0, 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=pipeline_workers) as executor:
        futures = {
            executor.submit(process_single_event, event, cases, pipeline, output_dir): event
            for event in events
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status = result["status"]
            eid = result["event_id"]
            elapsed = result["elapsed"]

            if status == "COMPLETED":
                success += 1
                tag = "OK"
            elif status == "cached":
                cached += 1
                tag = "CACHED"
            else:
                failed += 1
                tag = f"FAIL: {result.get('error', status)[:80]}"

            if i % 10 == 0 or status != "COMPLETED":
                total_elapsed = time.time() - start_time
                rate = i / total_elapsed * 60
                logger.info(f"[{i}/{len(events)}] {eid}: {tag} ({elapsed}s) | "
                            f"ok={success} cached={cached} fail={failed} | "
                            f"{rate:.1f} events/min")

    total_elapsed = time.time() - start_time
    logger.info(
        f"\n{'='*60}\n"
        f"BATCH COMPLETE in {total_elapsed:.1f}s\n"
        f"  Success: {success}\n"
        f"  Cached:  {cached}\n"
        f"  Failed:  {failed}\n"
        f"  Total:   {len(events)}\n"
        f"  Rate:    {len(events)/total_elapsed*60:.1f} events/min\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()

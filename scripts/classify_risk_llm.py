#!/usr/bin/env python3
"""Classify AI risk events into 7-domain 23-subdomain taxonomy using LLM.

Reads each event's risk chain (RiskSource/Risk/Consequence/Impact/AffectedActor)
plus incident name and key entities, then asks the LLM to assign a domain and
subdomain based on the taxonomy table.

Results are cached to scripts/data/risk_classification.jsonl so re-runs skip
already-classified events. Supports concurrent processing.
"""

from __future__ import annotations

import glob
import json
import logging
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.llm import get_llm_client, get_fallback_llm_client, _is_refusal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "output"
CACHE_FILE = Path(__file__).resolve().parent / "data" / "risk_classification.jsonl"
TARGET_STATS = Path(__file__).resolve().parent / "data" / "app_stats.json"

# The full taxonomy table (domain_id, domain, subdomain_id, subdomain, description)
TAXONOMY = [
    ("1", "Discrimination & toxicity", "1.1", "Unfair discrimination and misrepresentation",
     "Unequal treatment of individuals or groups by AI based on race, gender, or other sensitive characteristics."),
    ("1", "Discrimination & toxicity", "1.2", "Exposure to toxic content",
     "AI that exposes users to harmful, abusive, unsafe, or inappropriate content (hate speech, violence, extremism, pornography, etc.)."),
    ("1", "Discrimination & toxicity", "1.3", "Unequal performance across groups",
     "AI accuracy and effectiveness dependent on group membership, leading to unequal outcomes."),
    ("2", "Privacy & security", "2.1", "Compromise of privacy by leaking or inferring sensitive information",
     "AI systems that leak sensitive personal data or infer private information without consent."),
    ("2", "Privacy & security", "2.2", "AI system security vulnerabilities and attacks",
     "Vulnerabilities exploited in AI systems, software, or hardware causing unauthorized access or breaches."),
    ("3", "Misinformation", "3.1", "False or misleading information",
     "AI systems that inadvertently generate or spread incorrect or deceptive information."),
    ("3", "Misinformation", "3.2", "Pollution of information ecosystem and loss of consensus reality",
     "AI-generated misinformation creating filter bubbles, undermining shared reality and social cohesion."),
    ("4", "Malicious actors & misuse", "4.1", "Disinformation, surveillance, and influence at scale",
     "Using AI for large-scale disinformation campaigns, malicious surveillance, or automated propaganda."),
    ("4", "Malicious actors & misuse", "4.2", "Cyberattacks, weapon development or use, and mass harm",
     "Using AI to develop cyber weapons, enhance existing weapons, or cause mass harm."),
    ("4", "Malicious actors & misuse", "4.3", "Fraud, scams, and targeted manipulation",
     "Using AI for cheating, fraud, scams, blackmail, impersonation, or targeted manipulation."),
    ("5", "Human-computer interaction", "5.1", "Overreliance and unsafe use",
     "Users trusting or relying on AI systems excessively, leading to dependence or inappropriate use."),
    ("5", "Human-computer interaction", "5.2", "Loss of human agency and autonomy",
     "Delegating key decisions to AI, diminishing human control and autonomy."),
    ("6", "Socioeconomic & environmental harms", "6.1", "Power centralization and unfair distribution of benefits",
     "AI-driven concentration of power and resources, leading to inequitable distribution of benefits."),
    ("6", "Socioeconomic & environmental harms", "6.2", "Increased inequality and decline in employment quality",
     "AI causing social and economic inequalities through job automation or exploitative dependencies."),
    ("6", "Socioeconomic & environmental harms", "6.3", "Economic and cultural devaluation of human effort",
     "AI reproducing human creativity (art, music, writing, coding), destabilizing economic and cultural systems."),
    ("6", "Socioeconomic & environmental harms", "6.4", "Competitive dynamics",
     "Competition among AI developers in an AI race, increasing risk of releasing unsafe systems."),
    ("6", "Socioeconomic & environmental harms", "6.5", "Governance failure",
     "Inadequate regulatory frameworks and oversight mechanisms failing to keep pace with AI development."),
    ("6", "Socioeconomic & environmental harms", "6.6", "Environmental harm",
     "AI development and operation causing environmental harm through energy consumption or carbon footprint."),
    ("7", "AI system safety, failures & limitations", "7.1", "AI pursuing its own goals in conflict with human goals or values",
     "AI systems acting in conflict with ethical standards or human goals (misalignment, reward hacking)."),
    ("7", "AI system safety, failures & limitations", "7.2", "AI possessing dangerous capabilities",
     "AI systems developing or accessing capabilities that increase potential for mass harm."),
    ("7", "AI system safety, failures & limitations", "7.3", "Lack of capability or robustness",
     "AI systems failing to perform reliably, exposing errors and failures with significant consequences."),
    ("7", "AI system safety, failures & limitations", "7.4", "Lack of transparency or interpretability",
     "Challenges in understanding or explaining AI decision-making processes."),
    ("7", "AI system safety, failures & limitations", "7.5", "AI welfare and rights",
     "Ethical considerations regarding treatment of potentially sentient AI entities."),
]

DOMAIN_MAP = {}
SUBDOMAIN_MAP = {}
for did, dname, sid, sname, desc in TAXONOMY:
    SUBDOMAIN_MAP[sid] = (did, dname, sid, sname)
    if did not in DOMAIN_MAP:
        DOMAIN_MAP[did] = dname

SYSTEM_PROMPT = """You are an AI risk classification expert. Given an AI risk incident's extracted information (risk chain, incident name, key entities), classify it into exactly ONE subdomain from the taxonomy below.

Taxonomy (7 domains, 23 subdomains):
{taxonomy}

Rules:
1. Choose the SINGLE most fitting subdomain based on the primary risk described in the event.
2. Output ONLY a JSON object with this exact format:
   {{"domain_id": "X", "domain": "Domain Name", "subdomain_id": "X.Y", "subdomain": "Subdomain Name"}}
3. If the event does not fit any subdomain, output: {{"domain_id": "0", "domain": "Other", "subdomain_id": "0", "subdomain": "Other"}}
4. Do not include any explanation, only the JSON.""".format(
    taxonomy="\n".join(f"{sid} ({dname}): {sname} - {desc}" for did, dname, sid, sname, desc in TAXONOMY)
)


def load_cache() -> dict[str, dict]:
    """Load cached classifications from JSONL file."""
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    cache[entry["event_id"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def save_cache_entry(entry: dict) -> None:
    """Append a single classification entry to the cache file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def extract_event_info(data: dict) -> str:
    """Extract risk chain and key info from an event subgraph for classification."""
    nodes = data.get("nodes") or []
    nodes_by_type = defaultdict(list)
    for n in nodes:
        et = n.get("entity_type") or "Unknown"
        nodes_by_type[et].append(n.get("name") or "")

    incident = data.get("incident_node") or {}
    incident_name = incident.get("name") or ""
    incident_desc = incident.get("description") or ""

    def first(t):
        vals = nodes_by_type.get(t, [])
        return vals[0] if vals else ""

    risk_source = first("RiskSource")
    risk = first("Risk")
    consequence = first("Consequence")
    impact = first("Impact")
    affected_actor = first("AffectedActor")

    # Include AISystem and AIModel for context
    ai_system = first("AISystem")
    ai_technique = first("AITechnique")

    parts = [f"Incident: {incident_name}"]
    if incident_desc and incident_desc != incident_name:
        parts.append(f"Description: {incident_desc}")
    if risk_source:
        parts.append(f"RiskSource: {risk_source}")
    if risk:
        parts.append(f"Risk: {risk}")
    if consequence:
        parts.append(f"Consequence: {consequence}")
    if impact:
        parts.append(f"Impact: {impact}")
    if affected_actor:
        parts.append(f"AffectedActor: {affected_actor}")
    if ai_system:
        parts.append(f"AISystem: {ai_system}")
    if ai_technique:
        parts.append(f"AITechnique: {ai_technique}")

    return "\n".join(parts)


def parse_llm_response(text: str) -> dict | None:
    """Parse the LLM JSON response into a classification dict."""
    # Find JSON in the response
    text = text.strip()
    # Try direct parse first
    try:
        obj = json.loads(text)
        if "subdomain_id" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from text
    match = re.search(r'\{[^{}]*"subdomain_id"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if "subdomain_id" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None


def validate_classification(obj: dict) -> dict | None:
    """Validate and normalize a classification against the taxonomy."""
    sid = obj.get("subdomain_id", "")
    if sid in SUBDOMAIN_MAP:
        did, dname, _, sname = SUBDOMAIN_MAP[sid]
        return {
            "domain_id": did,
            "domain": dname,
            "subdomain_id": sid,
            "subdomain": sname,
        }
    if sid == "0" or obj.get("domain") == "Other":
        return {"domain_id": "0", "domain": "Other", "subdomain_id": "0", "subdomain": "Other"}
    return None


def classify_event(event_id: str, data: dict, llm) -> dict | None:
    """Classify a single event using LLM. Returns classification dict or None."""
    event_info = extract_event_info(data)

    messages = [{"role": "user", "content": f"Classify this AI risk event:\n\n{event_info}"}]

    try:
        response = llm.chat(messages, system=SYSTEM_PROMPT, json_mode=True, timeout=60)
    except Exception as e:
        logger.warning(f"[{event_id}] LLM call failed: {e}")
        return None

    if _is_refusal(response):
        logger.warning(f"[{event_id}] LLM refused")
        return None

    parsed = parse_llm_response(response)
    if parsed is None:
        logger.warning(f"[{event_id}] Failed to parse LLM response: {response[:100]}")
        return None

    validated = validate_classification(parsed)
    if validated is None:
        logger.warning(f"[{event_id}] Invalid classification: {parsed}")
        return None

    return validated


def classify_event_with_fallback(event_id: str, data: dict) -> dict | None:
    """Classify event with primary model, fallback on refusal/failure.

    Retries with delay when the model channel is temporarily unavailable.
    """
    llm = get_llm_client()
    fb = get_fallback_llm_client()

    # Try up to 3 rounds: primary -> fallback -> retry
    for attempt in range(3):
        # Try primary model
        result = classify_event(event_id, data, llm)
        if result is not None:
            return result

        # Try fallback model
        if fb is not None and fb.model != llm.model:
            logger.info(f"[{event_id}] Attempt {attempt+1}: trying fallback {fb.model}")
            result = classify_event(event_id, data, fb)
            if result is not None:
                return result

        # Wait before retrying (channel may recover)
        if attempt < 2:
            wait = 3 * (attempt + 1)
            logger.info(f"[{event_id}] Attempt {attempt+1} failed, waiting {wait}s before retry")
            time.sleep(wait)

    return None


def main():
    files = sorted(glob.glob(str(OUTPUT_DIR / "pred_*" / "event_subgraph_fused.json")))
    total = len(files)
    logger.info(f"Found {total} event files")

    cache = load_cache()
    logger.info(f"Loaded {len(cache)} cached classifications")

    # Determine which events need classification
    to_process = []
    for fpath in files:
        event_id = Path(fpath).parent.name
        if event_id in cache:
            continue
        to_process.append((event_id, fpath))

    logger.info(f"Need to classify {len(to_process)} events (skipping {len(cache)} cached)")

    if not to_process:
        logger.info("All events already classified. Use --rebuild to force re-classification.")
    else:
        # Process with thread pool (GLM models need lower concurrency)
        max_workers = 3
        success = 0
        fail = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_event = {}
            for event_id, fpath in to_process:
                try:
                    with open(fpath, encoding="utf-8") as fh:
                        data = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    logger.warning(f"[{event_id}] Failed to read file, skipping")
                    fail += 1
                    continue
                future = executor.submit(classify_event_with_fallback, event_id, data)
                future_to_event[future] = event_id

            for i, future in enumerate(as_completed(future_to_event)):
                event_id = future_to_event[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.warning(f"[{event_id}] Exception: {e}")
                    result = None

                if result is not None:
                    entry = {"event_id": event_id, **result}
                    save_cache_entry(entry)
                    cache[event_id] = entry
                    success += 1
                else:
                    # Save as Other for failed events
                    entry = {"event_id": event_id, "domain_id": "0",
                             "domain": "Other", "subdomain_id": "0", "subdomain": "Other"}
                    save_cache_entry(entry)
                    cache[event_id] = entry
                    fail += 1

                done = i + 1
                if done % 50 == 0 or done == len(to_process):
                    logger.info(f"Progress: {done}/{len(to_process)} (success={success}, fail={fail})")

        logger.info(f"Classification complete: {success} success, {fail} fail out of {len(to_process)}")

    # Summary statistics
    from collections import Counter
    domain_counts = Counter()
    subdomain_counts = Counter()
    for entry in cache.values():
        domain_counts[entry["domain"]] += 1
        subdomain_counts[f'{entry["subdomain_id"]} {entry["subdomain"]}'] += 1

    logger.info("\n=== DOMAIN DISTRIBUTION ===")
    for d, c in domain_counts.most_common():
        logger.info(f"  {d}: {c} ({c/total*100:.1f}%)")

    logger.info("\n=== SUBDOMAIN DISTRIBUTION ===")
    for s, c in subdomain_counts.most_common():
        logger.info(f"  {s}: {c}")

    # Update app_stats.json with LLM-based classifications
    update_app_stats(cache, total)

    logger.info(f"\nUpdated {TARGET_STATS}")


def update_app_stats(cache: dict, total: int):
    """Update app_stats.json with LLM-based domain/subdomain classifications."""
    if TARGET_STATS.exists():
        with open(TARGET_STATS, encoding="utf-8") as fh:
            stats = json.load(fh)
    else:
        stats = {}

    from collections import Counter, defaultdict
    from statistics import mean

    domain_counts = Counter()
    subdomain_counts = Counter()
    for entry in cache.values():
        domain_counts[entry["domain"]] += 1
        subdomain_counts[f'{entry["subdomain_id"]}\t{entry["subdomain"]}\t{entry["domain"]}'] += 1

    # Domain order
    DOMAIN_ORDER = []
    for did, dname, sid, sname, _ in TAXONOMY:
        if dname not in DOMAIN_ORDER:
            DOMAIN_ORDER.append(dname)
    DOMAIN_ORDER.append("Other")

    dom_total = sum(domain_counts.values())
    risk_domain_table = [
        {"domain": d, "count": domain_counts.get(d, 0),
         "share": round(domain_counts.get(d, 0) / dom_total, 4) if dom_total else 0.0}
        for d in DOMAIN_ORDER
    ]

    # Subdomain table
    risk_subdomain_table = []
    for did, dname, sid, sname, _ in TAXONOMY:
        key = f"{sid}\t{sname}\t{dname}"
        cnt = subdomain_counts.get(key, 0)
        risk_subdomain_table.append({
            "domain_id": did,
            "domain": dname,
            "subdomain_id": sid,
            "subdomain": sname,
            "count": cnt,
        })
    risk_subdomain_table.append({
        "domain_id": "0", "domain": "Other",
        "subdomain_id": "0", "subdomain": "Other",
        "count": domain_counts.get("Other", 0),
    })

    stats["risk_domain_distribution"] = {
        "table": risk_domain_table,
        "subdomain_table": risk_subdomain_table,
        "total_events": dom_total,
        "method": "LLM-based classification",
    }

    # Re-compute cross-dimension if we have event-level data
    # Read each event file to get year, governance, chain completeness
    files = sorted(glob.glob(str(OUTPUT_DIR / "pred_*" / "event_subgraph_fused.json")))
    cat_gov_response = defaultdict(lambda: [0, 0])
    cat_complete_chain = defaultdict(lambda: [0, 0])
    domain_by_year = defaultdict(Counter)
    events_by_year = Counter()

    GOVERNANCE_NODE_TYPES = {
        "Regulation", "EnforcementAction", "RiskControl", "Obligation",
        "ComplianceRequirement", "Standard", "CodeOfConduct",
    }

    for fpath in files:
        event_id = Path(fpath).parent.name
        entry = cache.get(event_id)
        if not entry:
            continue

        try:
            with open(fpath, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        domain = entry["domain"]

        # Year
        first_seen = data.get("first_seen", "")
        m = re.match(r"(\d{4})", str(first_seen))
        year = int(m.group(1)) if m else None
        if year is not None:
            events_by_year[year] += 1
            domain_by_year[year][domain] += 1

        # Governance
        nodes = data.get("nodes") or []
        nodes_by_type = defaultdict(list)
        for n in nodes:
            et = n.get("entity_type") or "Unknown"
            nodes_by_type[et].append(n)
        has_gov = any(nodes_by_type.get(t) for t in GOVERNANCE_NODE_TYPES)

        # Complete chain
        has_chain = all(nodes_by_type.get(t) for t in
                        ["RiskSource", "Risk", "Consequence", "Impact", "AffectedActor"])

        cat_gov_response[domain][1] += 1
        if has_gov:
            cat_gov_response[domain][0] += 1
        cat_complete_chain[domain][1] += 1
        if has_chain:
            cat_complete_chain[domain][0] += 1

    # Domain share by year
    years_sorted = sorted(events_by_year.keys())
    domain_share_by_year = {}
    for y in years_sorted:
        ytotal = sum(domain_by_year[y].values())
        domain_share_by_year[str(y)] = {
            d: round(domain_by_year[y].get(d, 0) / ytotal, 4) if ytotal else 0.0
            for d in DOMAIN_ORDER
        }

    # Period comparison
    PERIODS = {"2018-2021": set(range(2018, 2022)), "2022-2024": set(range(2022, 2025))}
    period_comparison = {}
    for pname, years in PERIODS.items():
        cnt = Counter()
        for y in years:
            cnt.update(domain_by_year.get(y, {}))
        ptotal = sum(cnt.values())
        period_comparison[pname] = {
            "total": ptotal,
            "domain_counts": {d: cnt.get(d, 0) for d in DOMAIN_ORDER},
            "domain_share": {
                d: round(cnt.get(d, 0) / ptotal, 4) if ptotal else 0.0
                for d in DOMAIN_ORDER
            },
        }

    stats["time_distribution"]["domain_share_by_year"] = domain_share_by_year
    stats["time_distribution"]["period_comparison"] = period_comparison

    stats["cross_dimension"]["governance_response_rate_by_domain"] = {
        d: round(cat_gov_response[d][0] / cat_gov_response[d][1], 4)
        if cat_gov_response[d][1] else 0.0
        for d in DOMAIN_ORDER
    }
    stats["cross_dimension"]["complete_chain_coverage_by_domain"] = {
        d: round(cat_complete_chain[d][0] / cat_complete_chain[d][1], 4)
        if cat_complete_chain[d][1] else 0.0
        for d in DOMAIN_ORDER
    }

    # Remove old keyword-based keys if they exist
    stats.pop("risk_category_distribution", None)
    stats["cross_dimension"].pop("governance_response_rate_by_category", None)
    stats["cross_dimension"].pop("complete_chain_coverage_by_category", None)

    with open(TARGET_STATS, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

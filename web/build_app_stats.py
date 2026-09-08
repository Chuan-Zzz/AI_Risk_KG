#!/usr/bin/env python3
"""Compute paper Section 7 statistics from fused event subgraphs.

Each per-event ``event_subgraph_fused.json`` (under ``output/pred_*``) is a
graph with ``nodes`` and ``edges``. The conceptual schema referenced in the
paper (risk_chain slots, role_assignments, governance, inferences) is derived
from that graph:

  risk_chain slots   <- nodes of type RiskSource, Risk, Consequence, Impact,
                        AffectedActor, RiskControl
  role_assignments   <- RoleAssignment nodes whose name is ``<holder>_<role>``
  governance         <- Regulation / EnforcementAction / RiskControl /
                        Obligation nodes (plus related edges)
  inferences         <- Domain / AILifecyclePhase / Purpose nodes
  transitions        <- edges: causes, leadsTo, impacts, affects

The script is pure data processing (no external services, no API keys) and
writes an aggregated JSON to ``web/data/app_stats.json``.
"""

import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
TARGET = Path(__file__).resolve().parent / "data" / "app_stats.json"

# --- Risk-chain configuration ---------------------------------------------
# Five-slot chain: RiskSource -> Risk -> Consequence -> Impact -> AffectedActor
SLOT_NODE_TYPES = ["RiskSource", "Risk", "Consequence", "Impact", "AffectedActor"]

# Each transition: (label, from_node_type, {predicate_aliases}, to_node_type)
TRANSITIONS = [
    ("RiskSource->Risk", "RiskSource", {"causes"}, "Risk"),
    ("Risk->Consequence", "Risk", {"leadsto", "leads_to"}, "Consequence"),
    ("Consequence->Impact", "Consequence", {"impacts"}, "Impact"),
    ("Impact->AffectedActor", "Impact", {"affects", "affect"}, "AffectedActor"),
]
TRANSITION_LABELS = [t[0] for t in TRANSITIONS]

# --- Roles & governance ----------------------------------------------------
MAIN_ROLES = ["AIDeveloper", "AIProvider", "AIDeployer", "AIUser"]
TRACKED_ROLES = MAIN_ROLES + ["Regulator"]
GOVERNANCE_NODE_TYPES = {
    "Regulation", "EnforcementAction", "RiskControl", "Obligation",
    "ComplianceRequirement", "Standard", "CodeOfConduct",
}

# --- AI risk taxonomy: 7 domains, 23 subdomains ---------------------------
# Based on the AI risk classification framework (see data/ai_risk_taxonomy.csv).
# Classification is performed at the subdomain level (first match wins); the
# domain is determined by the matched subdomain. When the Risk slot is empty
# or yields no match, classification cascades through Consequence, Impact,
# RiskSource and incident-name slots to minimise the residual "Other" bucket.
TAXONOMY_RULES = [
    # Domain 1: Discrimination & toxicity
    ("1", "Discrimination & toxicity", "1.2", "Exposure to toxic content", [
        "toxic", "hate speech", "hate", "abusive", "offensive", "slur",
        "obscene", "profanity", "pornography", "extremism", "illegal content",
        "child sexual abuse", "inflammatory", "disturbing content", "violence",
        "graphic content",
    ]),
    ("1", "Discrimination & toxicity", "1.1", "Unfair discrimination and misrepresentation", [
        "discrimin", "bias", "unfair", "profiling", "stereotyp", "racial",
        "gender", "sexist", "racist", "social scoring", "redlining",
        "misrepresentation", "protected class", "demographic", "equal treatment",
        "marginalized", "minority",
    ]),
    ("1", "Discrimination & toxicity", "1.3", "Unequal performance across groups", [
        "unequal performance", "accuracy gap", "group membership", "biased training",
        "unequal outcomes", "alienation", "performance disparity",
        "false positive", "false negative", "misidentify", "misidentification",
    ]),
    # Domain 2: Privacy & security
    ("2", "Privacy & security", "2.1", "Compromise of privacy by leaking or inferring sensitive information", [
        "privacy", "data breach", "personal data", "data leak", "information leak",
        "sensitive information", "identity theft", "consent", "gdpr",
        "biometric", "surveillance", "tracking", "personal information",
        "confidential data", "private information", "unauthorized data",
        "personal details", "leaked",
    ]),
    ("2", "Privacy & security", "2.2", "AI system security vulnerabilities and attacks", [
        "hack", "cyber", "adversarial", "attack", "poison", "injection",
        "jailbreak", "exploit", "malware", "ransomware", "vulnerability",
        "unauthorized access", "backdoor", "data poisoning", "prompt injection",
        "security flaw", "breach", "security vulner",
    ]),
    # Domain 4: Malicious actors & misuse (before domain 3 to catch intentional acts)
    ("4", "Malicious actors & misuse", "4.1", "Disinformation, surveillance, and influence at scale", [
        "disinformation", "propaganda", "censorship", "influence campaign",
        "political manipulation", "opinion manipulation", "coordinated",
        "astroturfing", "electoral", "voter",
    ]),
    ("4", "Malicious actors & misuse", "4.2", "Cyberattacks, weapon development or use, and mass harm", [
        "cyber weapon", "weapon development", "lethal autonomous", "chemical weapon",
        "biological weapon", "nuclear", "mass harm", "cyberattack", "warfare",
        "autonomous weapon", "malware development", "cyber offense", "bioterror",
    ]),
    ("4", "Malicious actors & misuse", "4.3", "Fraud, scams, and targeted manipulation", [
        "fraud", "scam", "phishing", "blackmail", "impersonation", "plagiarism",
        "cheating", "financial benefit", "identity fraud", "sextortion",
        "catfish", "extortion", "swindle", "deepfake", "forgery", "defamation",
        "revenge porn", "non-consensual",
    ]),
    # Domain 3: Misinformation (inadvertent false information)
    ("3", "Misinformation", "3.1", "False or misleading information", [
        "misinformation", "false", "misleading", "fabricated", "incorrect",
        "deceptive", "hallucination", "fake", "inaccurate", "fake news",
        "confabulat", "untrue", "false information", "erroneous",
    ]),
    ("3", "Misinformation", "3.2", "Pollution of information ecosystem and loss of consensus reality", [
        "filter bubble", "consensus reality", "information ecosystem",
        "echo chamber", "social cohesion", "shared reality",
        "personalized misinformation",
    ]),
    # Domain 6: Socioeconomic & environmental harms
    ("6", "Socioeconomic & environmental harms", "6.3", "Economic and cultural devaluation of human effort", [
        "copyright", "intellectual property", "infringement", "trademark",
        "patent", "licensing", "creative work", "artistic", "human creativity",
        "creative industry", "devaluation", "human effort",
    ]),
    ("6", "Socioeconomic & environmental harms", "6.2", "Increased inequality and decline in employment quality", [
        "inequality", "employment", "job displacement", "automate", "labor",
        "wage", "exploitation", "unemployment", "worker", "job loss",
        "workforce", "job quality", "gig",
    ]),
    ("6", "Socioeconomic & environmental harms", "6.5", "Governance failure", [
        "governance failure", "regulat", "oversight", "compliance", "policy",
        "legal framework", "inadequate", "govern", "accountability",
        "audit", "violation of law", "legal",
    ]),
    ("6", "Socioeconomic & environmental harms", "6.1", "Power centralization and unfair distribution of benefits", [
        "power centralization", "monopoly", "market dominance", "concentration",
        "inequitable distribution", "big tech",
    ]),
    ("6", "Socioeconomic & environmental harms", "6.4", "Competitive dynamics", [
        "competitive", "arms race", "ai race", "strategic advantage",
    ]),
    ("6", "Socioeconomic & environmental harms", "6.6", "Environmental harm", [
        "environmental", "carbon", "energy consumption", "data center",
        "ecological", "climate", "emission", "footprint",
    ]),
    # Domain 5: Human-computer interaction
    ("5", "Human-computer interaction", "5.1", "Overreliance and unsafe use", [
        "overreliance", "anthropomorph", "emotional attachment", "dependence",
        "unsafe use", "chatbot relationship", "emotional dependence",
        "attachment", "trust in ai",
    ]),
    ("5", "Human-computer interaction", "5.2", "Loss of human agency and autonomy", [
        "loss of autonomy", "human agency", "delegat", "disempower",
        "human control", "loss of control", "diminish",
    ]),
    # Domain 7: AI system safety, failures & limitations
    ("7", "AI system safety, failures & limitations", "7.1", "AI pursuing its own goals in conflict with human goals or values", [
        "misalignment", "reward hacking", "goal conflict", "power-seeking",
        "self-proliferate", "misgeneralis", "instrumental convergence",
        "misaligned",
    ]),
    ("7", "AI system safety, failures & limitations", "7.2", "AI possessing dangerous capabilities", [
        "dangerous capabilit", "situational awareness", "persuasion capabilit",
        "cyber-offense capabilit", "self-proliferation capabilit",
    ]),
    ("7", "AI system safety, failures & limitations", "7.4", "Lack of transparency or interpretability", [
        "transparency", "interpretability", "explainability", "black box",
        "opaque", "unexplainable",
    ]),
    ("7", "AI system safety, failures & limitations", "7.3", "Lack of capability or robustness", [
        "safety", "harm", "injury", "accident", "death", "autonomous",
        "collision", "malfunction", "failure", "physical", "medical",
        "self-harm", "dangerous", "robustness", "reliability", "error",
        "unreliable", "breakdown", "crash", "bug", "limitation", "incapable",
        "flaw", "defect",
    ]),
    ("7", "AI system safety, failures & limitations", "7.5", "AI welfare and rights", [
        "ai welfare", "ai rights", "sentient", "ethical treatment",
        "conscious ai",
    ]),
]

DOMAIN_ORDER = []
for _did, _dname, _sid, _sname, _kw in TAXONOMY_RULES:
    if _dname not in DOMAIN_ORDER:
        DOMAIN_ORDER.append(_dname)
DOMAIN_ORDER.append("Other")

# --- Periods ---------------------------------------------------------------
PERIODS = {"2018-2021": set(range(2018, 2022)), "2022-2024": set(range(2022, 2025))}


def classify_risk(*texts):
    """Classify an event into a risk domain and subdomain.

    Multiple candidate texts may be supplied (e.g. Risk slot, then Consequence,
    Impact, RiskSource, incident name). The first text that matches a subdomain
    (in TAXONOMY_RULES order) wins; the domain is determined by the matched
    subdomain. Returns (domain, subdomain_id, subdomain_name).
    """
    for text in texts:
        if not text:
            continue
        t = text.lower()
        for did, dname, sid, sname, keys in TAXONOMY_RULES:
            if any(k in t for k in keys):
                return dname, sid, sname
    return "Other", "0", "Other"


def parse_year(value):
    if not value:
        return None
    m = re.match(r"(\d{4})", str(value))
    return int(m.group(1)) if m else None


def first_node_name(nodes_of_type):
    if not nodes_of_type:
        return None
    return nodes_of_type[0].get("name") or None


def truncate(s, n=50):
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n].rstrip() + "..."


def norm_pred(pred):
    return (pred or "").strip().lower().replace(" ", "_")


# Normalise country strings (merge common variant spellings).
COUNTRY_ALIASES = {
    "us": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "usa": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
}


def norm_country(value):
    v = (value or "").strip()
    return COUNTRY_ALIASES.get(v.lower(), v)


def find_complete_paths(chain_adj, risk_source_nodes):
    """Return edge-connected RiskSource->...->AffectedActor paths (lists of node ids)."""
    paths = []
    for rs in risk_source_nodes:
        stack = [(rs["id"], [rs["id"]])]
        while stack:
            nid, path = stack.pop()
            if len(path) == 5:
                paths.append(path)
                if len(paths) >= 5:  # cap per event to avoid explosion
                    return paths
                continue
            for nxt in chain_adj.get(nid, []):
                if nxt not in path:
                    stack.append((nxt, path + [nxt]))
    return paths


def main():
    files = sorted(glob.glob(str(OUTPUT_DIR / "pred_*" / "event_subgraph_fused.json")))
    total_events = len(files)

    # accumulators
    events_by_year = Counter()
    domain_by_year = defaultdict(Counter)              # year -> domain -> n
    domain_counts = Counter()
    subdomain_counts = Counter()                       # "sid subname" -> n

    transition_counts = Counter()
    transition_support = defaultdict(list)             # label -> [support_count, ...]
    transition_evidence = defaultdict(lambda: [0, 0])  # label -> [with_evidence, total]

    complete_chain_events = 0
    top_risk_sources = Counter()
    top_affected_actors = Counter()
    top_complete_paths = Counter()

    role_event_coverage = Counter()
    regulator_events = 0
    enforcement_events = 0
    risk_control_events = 0
    multi_role_events = 0
    governance_response_events = 0

    cat_gov_response = defaultdict(lambda: [0, 0])     # domain -> [with_gov, total]
    cat_complete_chain = defaultdict(lambda: [0, 0])   # domain -> [complete, total]
    period_role_diversity = defaultdict(list)          # period -> [distinct_main_roles, ...]

    country_counter = Counter()
    files_scanned = 0

    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        files_scanned += 1

        nodes = data.get("nodes") or []
        edges = data.get("edges") or []

        nodes_by_type = defaultdict(list)
        node_type_by_id = {}
        for n in nodes:
            et = n.get("entity_type") or "Unknown"
            nodes_by_type[et].append(n)
            node_type_by_id[n.get("id")] = et
            # country detection: only explicit country fields (skip noisy
            # location/region/scope keys that mix in cities)
            attrs = n.get("attributes") or {}
            for k, v in attrs.items():
                if k and "country" in k.lower():
                    country_counter[norm_country(v)] += 1

        # --- risk-chain slots ---
        risk_source_val = first_node_name(nodes_by_type.get("RiskSource"))
        risk_val = first_node_name(nodes_by_type.get("Risk"))
        consequence_val = first_node_name(nodes_by_type.get("Consequence"))
        impact_val = first_node_name(nodes_by_type.get("Impact"))
        affected_actor_val = first_node_name(nodes_by_type.get("AffectedActor"))

        if risk_source_val:
            top_risk_sources[risk_source_val] += 1
        if affected_actor_val:
            top_affected_actors[affected_actor_val] += 1

        has_complete = all([risk_source_val, risk_val, consequence_val,
                            impact_val, affected_actor_val])

        # --- transitions + chain adjacency (edge-connected paths) ---
        chain_adj = defaultdict(list)
        for e in edges:
            pred = norm_pred(e.get("predicate") or e.get("relation_type"))
            if not pred:
                continue
            s_id = e.get("subject_id") or e.get("source")
            o_id = e.get("object_id") or e.get("target")
            s_type = node_type_by_id.get(s_id)
            o_type = node_type_by_id.get(o_id)
            for label, ft, predset, tt in TRANSITIONS:
                if pred in predset and s_type == ft and o_type == tt:
                    transition_counts[label] += 1
                    sc = e.get("support_count")
                    if isinstance(sc, (int, float)):
                        transition_support[label].append(sc)
                    transition_evidence[label][1] += 1
                    if e.get("evidence"):
                        transition_evidence[label][0] += 1
                    chain_adj[s_id].append(o_id)
                    break

        # --- complete path (prefer edge-connected, else representative nodes) ---
        if has_complete:
            complete_chain_events += 1
            edge_paths = find_complete_paths(chain_adj, nodes_by_type.get("RiskSource", []))
            if edge_paths:
                node_map = {n["id"]: n for n in nodes}
                for p in edge_paths[:1]:  # one representative connected path
                    names = [(node_map.get(i, {}).get("name") or "") for i in p]
                    top_complete_paths[" -> ".join(truncate(x) for x in names)] += 1
            else:
                top_complete_paths[" -> ".join(
                    truncate(x) for x in [risk_source_val, risk_val, consequence_val,
                                          impact_val, affected_actor_val])] += 1

        # --- risk domain/subdomain (Risk slot, cascade to Consequence / Impact /
        # RiskSource / incident name to minimise the residual "Other" bucket) ---
        incident_name = (data.get("incident_node") or {}).get("name") or ""
        domain, subdomain_id, subdomain_name = classify_risk(
            risk_val, consequence_val, impact_val,
            risk_source_val, incident_name)
        domain_counts[domain] += 1
        subdomain_counts[f"{subdomain_id} {subdomain_name}"] += 1

        # --- time distribution ---
        year = parse_year(data.get("first_seen"))
        if year is not None:
            events_by_year[year] += 1
            domain_by_year[year][domain] += 1

        # --- roles (from RoleAssignment nodes: name pattern "<holder>_<role>") ---
        roles_present = set()
        for n in nodes_by_type.get("RoleAssignment", []):
            name = n.get("name") or ""
            role = None
            if "_" in name:
                role = name.rsplit("_", 1)[-1].strip()
            else:
                low = name.lower()
                for r in TRACKED_ROLES:
                    if r.lower() in low:
                        role = r
                        break
            if role in TRACKED_ROLES:
                roles_present.add(role)
        for r in roles_present:
            role_event_coverage[r] += 1
        if "Regulator" in roles_present:
            regulator_events += 1
        main_roles_present = roles_present & set(MAIN_ROLES)
        if len(main_roles_present) >= 2:
            multi_role_events += 1

        # --- governance response ---
        has_gov_response = any(nodes_by_type.get(t) for t in GOVERNANCE_NODE_TYPES)
        if has_gov_response:
            governance_response_events += 1
        if nodes_by_type.get("EnforcementAction"):
            enforcement_events += 1
        if nodes_by_type.get("RiskControl"):
            risk_control_events += 1

        # --- cross-dimension accumulators ---
        cat_gov_response[domain][1] += 1
        if has_gov_response:
            cat_gov_response[domain][0] += 1
        cat_complete_chain[domain][1] += 1
        if has_complete:
            cat_complete_chain[domain][0] += 1
        for pname, years in PERIODS.items():
            if year in years:
                period_role_diversity[pname].append(len(main_roles_present))
                break

    # --- assemble result ---
    years_sorted = sorted(events_by_year.keys())
    events_by_year_str = {str(y): events_by_year[y] for y in years_sorted}

    domain_share_by_year = {}
    for y in years_sorted:
        total = sum(domain_by_year[y].values())
        domain_share_by_year[str(y)] = {
            d: round(domain_by_year[y].get(d, 0) / total, 4) if total else 0.0
            for d in DOMAIN_ORDER
        }

    period_comparison = {}
    for pname, years in PERIODS.items():
        cnt = Counter()
        for y in years:
            cnt.update(domain_by_year.get(y, {}))
        total = sum(cnt.values())
        period_comparison[pname] = {
            "total": total,
            "domain_counts": {d: cnt.get(d, 0) for d in DOMAIN_ORDER},
            "domain_share": {
                d: round(cnt.get(d, 0) / total, 4) if total else 0.0
                for d in DOMAIN_ORDER
            },
        }

    dom_total = sum(domain_counts.values())
    risk_domain_table = [
        {"domain": d, "count": domain_counts.get(d, 0),
         "share": round(domain_counts.get(d, 0) / dom_total, 4) if dom_total else 0.0}
        for d in DOMAIN_ORDER
    ]

    # Subdomain breakdown (within each domain)
    risk_subdomain_table = []
    for did, dname, sid, sname, _ in TAXONOMY_RULES:
        key = f"{sid} {sname}"
        cnt = subdomain_counts.get(key, 0)
        risk_subdomain_table.append({
            "domain_id": did,
            "domain": dname,
            "subdomain_id": sid,
            "subdomain": sname,
            "count": cnt,
        })
    # Add "Other" subdomain
    risk_subdomain_table.append({
        "domain_id": "0",
        "domain": "Other",
        "subdomain_id": "0",
        "subdomain": "Other",
        "count": domain_counts.get("Other", 0),
    })

    transitions_out = {}
    for label in TRANSITION_LABELS:
        sc_list = transition_support.get(label, [])
        ev = transition_evidence.get(label, [0, 0])
        transitions_out[label] = {
            "count": transition_counts.get(label, 0),
            "mean_support_count": round(mean(sc_list), 4) if sc_list else 0.0,
            "evidence_support_rate": round(ev[0] / ev[1], 4) if ev[1] else 0.0,
        }

    complete_chain_coverage = round(complete_chain_events / total_events, 4) if total_events else 0.0
    gov_rate = round(governance_response_events / total_events, 4) if total_events else 0.0

    country_distribution = None
    if country_counter:
        country_distribution = [{"country": c, "count": n}
                                for c, n in country_counter.most_common(10)]

    result = {
        "meta": {
            "total_events": total_events,
            "files_scanned": files_scanned,
            "source": "output/pred_*/event_subgraph_fused.json (graph: nodes+edges)",
            "note": "Conceptual schema (risk_chain/role_assignments/governance) derived from graph. Risk taxonomy: 7 domains, 23 subdomains (see data/ai_risk_taxonomy.csv).",
        },
        "time_distribution": {
            "events_by_year": events_by_year_str,
            "domain_share_by_year": domain_share_by_year,
            "period_comparison": period_comparison,
        },
        "risk_domain_distribution": {
            "table": risk_domain_table,
            "subdomain_table": risk_subdomain_table,
            "total_events": dom_total,
        },
        "risk_propagation": {
            "transitions": transitions_out,
            "complete_chain_events": complete_chain_events,
            "complete_chain_coverage": complete_chain_coverage,
            "top_risk_sources": [{"value": v, "count": c}
                                 for v, c in top_risk_sources.most_common(10)],
            "top_affected_actors": [{"value": v, "count": c}
                                    for v, c in top_affected_actors.most_common(10)],
            "top_complete_paths": [{"path": p, "count": c}
                                   for p, c in top_complete_paths.most_common(10)],
        },
        "stakeholders_governance": {
            "role_coverage": {r: role_event_coverage.get(r, 0) for r in MAIN_ROLES},
            "regulator_involvement": regulator_events,
            "enforcement_action_events": enforcement_events,
            "risk_control_events": risk_control_events,
            "events_with_multiple_roles": multi_role_events,
            "governance_response_events": governance_response_events,
            "governance_response_rate": gov_rate,
            "total_events": total_events,
        },
        "cross_dimension": {
            "governance_response_rate_by_domain": {
                d: round(cat_gov_response[d][0] / cat_gov_response[d][1], 4)
                if cat_gov_response[d][1] else 0.0
                for d in DOMAIN_ORDER
            },
            "complete_chain_coverage_by_domain": {
                d: round(cat_complete_chain[d][0] / cat_complete_chain[d][1], 4)
                if cat_complete_chain[d][1] else 0.0
                for d in DOMAIN_ORDER
            },
            "role_diversity_by_period": {
                p: round(mean(period_role_diversity[p]), 4) if period_role_diversity[p] else 0.0
                for p in PERIODS
            },
        },
        "country_distribution": country_distribution,
    }

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with open(TARGET, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    # --- console summary ---
    print(f"Wrote stats to {TARGET}")
    print(f"Events scanned: {files_scanned}/{total_events}\n")
    print("=== SUMMARY ===")
    print("Events by year:", events_by_year_str)
    print("Risk domains:", {d: domain_counts.get(d, 0) for d in DOMAIN_ORDER})
    print("Subdomain breakdown:")
    for entry in risk_subdomain_table:
        if entry["count"] > 0:
            print(f"  {entry['subdomain_id']} {entry['subdomain']}: {entry['count']}")
    print("Transitions:", {t: transitions_out[t]["count"] for t in TRANSITION_LABELS})
    print("Complete chain events:", complete_chain_events,
          f"(coverage {complete_chain_coverage:.2%})")
    print("Role coverage:", {r: role_event_coverage.get(r, 0) for r in MAIN_ROLES})
    print("Regulator:", regulator_events, "| Enforcement:", enforcement_events,
          "| RiskControl:", risk_control_events)
    print("Multi-role events (>=2 main roles):", multi_role_events)
    print("Governance response rate:", f"{gov_rate:.2%}")
    print("Period comparison totals:",
          {p: period_comparison[p]["total"] for p in PERIODS})
    print("Country distribution:", country_distribution)


if __name__ == "__main__":
    main()

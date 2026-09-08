"""Bucket risk propagation chain slots via LLM for the Section 7.3 Sankey.

Stages:
  - RiskSource    (9 buckets,  LLM)
  - Risk          (7 buckets,  reuse classification domain — no LLM)
  - Consequence   (13 buckets, LLM)
  - Impact        (10 buckets, LLM)
  - AffectedActor (9 buckets,  LLM)

Bucket definitions are grounded in:
  - NIST AI 600-1 Generative AI Profile (2024)
  - MIT AI Risk Repository causal taxonomy (Slattery et al., arXiv:2408.12622)
  - Huwyler AI System Threat Vector Taxonomy (arXiv:2511.21901)
  - Li et al. GenAI incident stakeholder framework (arXiv:2505.22073)
  - IEEE Decoding Real-World AI Incidents (2024)
  - IJHCI 2025 "Who is Responsible When AI Fails"
  - OECD Defining AI Incidents (May 2024)

Outputs (under web/data/):
  - bucketing_RiskSource.jsonl    (one record per event: prompt + LLM raw + parsed bucket)
  - bucketing_Consequence.jsonl
  - bucketing_Impact.jsonl
  - bucketing_AffectedActor.jsonl
  - bucketing_Risk.jsonl          (classification domain, no LLM call)
  - bucket_summary.csv            (per-bucket count + top-10 samples per stage)
  - bucket_transitions.csv        (stage-to-stage co-occurrence matrix)

Usage:
  export BUCKET_API_KEY="sk-..."
  export BUCKET_BASE_URL="https://api.yuk15n0w.asia/v1"
  python web/bucket_propagation_chain.py --stage all --concurrency 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
WEB_DATA = Path(__file__).parent / "data"
WEB_DATA.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_PATH = WEB_DATA / "risk_classification.jsonl"
SLOT_TYPES = ["RiskSource", "Risk", "Consequence", "Impact", "AffectedActor"]

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("bucket")
# Silence noisy httpx request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# --------------------------------------------------------------------------- #
# Bucket definitions (grounded in the references listed in the docstring)
# --------------------------------------------------------------------------- #
# Each bucket: (id, label, description for the LLM)
# Order matters: the LLM is told to pick the FIRST matching bucket.

RISKSOURCE_BUCKETS = [
    ("deepfake_ncii", "Deepfake & synthetic intimate/non-consensual content",
     "AI-generated deepfakes, face swaps, non-consensual intimate imagery (NCII), deepfake pornography, synthetic explicit content of real persons."),
    ("facial_recognition", "Facial recognition & biometric surveillance",
     "Facial recognition, face matching, biometric identification, mass surveillance via biometric systems."),
    ("bias_discrimination", "Algorithmic bias & discriminatory outcomes",
     "Algorithmic bias, unfair discrimination, disparate impact across demographic groups, biased training data leading to discriminatory outputs."),
    ("autonomous_system", "Autonomous vehicles & robotics safety",
     "Autonomous vehicles, self-driving cars, delivery robots, autonomous drones, robotics safety failures."),
    ("training_data", "Training data & data pipeline risks",
     "Data poisoning, training data leakage, toxic training data, dataset misuse, unauthorized data collection for training."),
    ("voice_clone", "Voice cloning & audio synthesis",
     "Voice cloning, speech synthesis, audio impersonation, voice fraud."),
    ("hallucination", "Hallucination & unfaithful output",
     "AI hallucination, confabulation, fabricated facts, fictional citations, unfaithful summaries, made-up information presented as factual."),
    ("fraud_abuse", "Fraud, scams & malicious use",
     "AI-enabled fraud, scams, phishing, social engineering, impersonation for financial gain, malicious use of AI for illegal purposes."),
    ("content_moderation", "Content moderation failure & toxic content",
     "Failure to filter toxic/hateful/violent content, inadequate content moderation, dissemination of dangerous content, CSAM generation."),
    ("other", "Other risk source",
     "Does not fit any of the above categories."),
]

CONSEQUENCE_BUCKETS = [
    ("privacy_violation", "Privacy violation",
     "Unauthorized collection, leakage, or exposure of personal/sensitive data; biometric privacy breach; surveillance without consent."),
    ("discrimination", "Discrimination & unfair treatment",
     "Unfair treatment of individuals or groups based on protected attributes; wrongful denial of services; biased decision outcomes."),
    ("misinformation_spread", "Misinformation & disinformation spread",
     "Spread of false, misleading, or fabricated information; disinformation campaigns; erosion of information ecosystem integrity."),
    ("psychological_harm", "Psychological & emotional harm",
     "Mental distress, emotional trauma, anxiety, reputational fear, psychological manipulation of victims."),
    ("physical_harm", "Physical harm & injury",
     "Bodily injury, death, property damage, physical safety incidents (e.g., autonomous vehicle collisions)."),
    ("financial_loss", "Financial loss & economic harm",
     "Direct monetary loss, fraud damages, identity theft costs, business revenue loss."),
    ("reputational_damage", "Reputational damage",
     "Harm to reputation of individuals, organizations, or institutions; defamation; brand damage."),
    ("legal_violation", "Legal & regulatory violation",
     "Violation of laws, regulations, or rights; unlawful arrests; IP infringement lawsuits; breach of compliance obligations."),
    ("security_breach", "Security incident & cyberattack",
     "Cyberattacks, data breaches, system intrusions, malware generation, unauthorized system access."),
    ("ip_infringement", "Intellectual property infringement",
     "Copyright infringement, unauthorized use of protected works, IP theft by AI systems."),
    ("democratic_harm", "Democratic & electoral harm",
     "Election interference, voter manipulation, undermining democratic processes, political destabilization."),
    ("social_division", "Social division & polarization",
     "Amplification of social polarization, racial/gender stereotyping, community harm, social unrest."),
    ("sector_harm", "Sector-specific harm (education/employment/healthcare)",
     "Harm in specific sectors: wrongful academic action, employment discrimination, healthcare misdiagnosis, child safety failures."),
    ("other", "Other consequence",
     "Does not fit any of the above categories."),
]

IMPACT_BUCKETS = [
    ("trust_erosion", "Erosion of trust in AI/technology",
     "Long-term decline in public trust toward AI systems, technology companies, or digital platforms."),
    ("regulatory_response", "Regulatory & policy response",
     "New regulations, legislation, policy changes, government investigations, or enforcement actions triggered by the incident."),
    ("public_discourse", "Shift in public discourse",
     "Changes in public debate, increased awareness, media attention shifts, societal conversation reframing."),
    ("market_industry", "Market & industry effect",
     "Changes in market dynamics, industry practices, business model shifts, competitive landscape effects, stock price impacts."),
    ("developmental_harm", "Long-term developmental harm",
     "Sustained harm to children's development, adolescents, or vulnerable populations over time; intergenerational effects."),
    ("social_norm_shift", "Social norm & behavior shift",
     "Changes in social norms, behavioral patterns, attitudes, or cultural practices."),
    ("democratic_erosion", "Democratic institutional erosion",
     "Weakening of democratic institutions, governance capacity, or civic engagement over time."),
    ("safety_security_risk", "Systemic safety & security risk",
     "Persistent safety or security risks to critical infrastructure, public safety systems, or national security."),
    ("individual_sustained", "Sustained individual harm",
     "Long-lasting harm to specific individuals: ongoing trauma, career damage, persistent surveillance, chronic stress."),
    ("ai_development", "Impact on AI development trajectory",
     "Changes in AI development, deployment, investment, or research direction; acceleration or restriction of AI capabilities."),
    ("other", "Other impact",
     "Does not fit any of the above categories."),
]

AFFECTEDACTOR_BUCKETS = [
    ("consumers_users", "Consumers & end users",
     "Individual consumers, end users of AI products, customers, subscribers, general product users."),
    ("general_public", "General public & society",
     "Society at large, the general public, citizens, communities as a whole."),
    ("children_students", "Children, students & minors",
     "Minors, children, students, adolescents, youth; educational settings."),
    ("women", "Women & girls",
     "Women, girls, female-identifying individuals; gender-based harm."),
    ("minorities", "Minorities & marginalized groups",
     "Racial/ethnic minorities, LGBTQ+ individuals, indigenous groups, religious minorities, marginalized communities."),
    ("patients_health", "Patients & healthcare recipients",
     "Patients, medical patients, healthcare recipients, individuals affected by health AI systems."),
    ("media", "Media & journalists",
     "Journalists, reporters, news organizations, media workers, content creators targeted by AI harms."),
    ("companies", "Companies & organizations",
     "Businesses, corporations, deploying organizations, institutions harmed operationally or financially."),
    ("government", "Government & public institutions",
     "Government agencies, public authorities, regulators, public institutions, democratic bodies."),
    ("other", "Other affected actor",
     "Does not fit any of the above categories."),
]

BUCKET_DEFS = {
    "RiskSource": RISKSOURCE_BUCKETS,
    "Consequence": CONSEQUENCE_BUCKETS,
    "Impact": IMPACT_BUCKETS,
    "AffectedActor": AFFECTEDACTOR_BUCKETS,
}

# Risk stage: 7 buckets mapped from classification domain (no LLM)
RISK_DOMAIN_TO_BUCKET = {
    "Malicious actors & misuse": "malicious_misuse",
    "AI system safety, failures & limitations": "ai_safety_failure",
    "Discrimination & toxicity": "discrimination_toxicity",
    "Misinformation": "misinformation",
    "Privacy & security": "privacy_security",
    "Socioeconomic & environmental harms": "socioeconomic_environmental",
    "Human-computer interaction": "hci",
    "Other": "other",
}
RISK_BUCKETS = [
    ("malicious_misuse", "Malicious actors & misuse",
     "Deliberate misuse of AI by malicious actors: fraud, scams, cyberattacks, weapons development."),
    ("ai_safety_failure", "AI system safety, failures & limitations",
     "AI system failures, lack of capability/robustness, hallucinations, unsafe outputs, system vulnerabilities."),
    ("discrimination_toxicity", "Discrimination & toxicity",
     "Unfair discrimination, toxic content, hateful outputs, misrepresentation of groups."),
    ("misinformation", "Misinformation",
     "False/misleading information, disinformation campaigns, pollution of information ecosystem."),
    ("privacy_security", "Privacy & security",
     "Privacy violations, data leakage, security breaches, biometric privacy, surveillance."),
    ("socioeconomic_environmental", "Socioeconomic & environmental harms",
     "Economic devaluation, employment harm, power centralization, environmental damage, inequality."),
    ("hci", "Human-computer interaction harms",
     "Overreliance, loss of agency, unsafe use, transparency failures, human-AI interaction risks."),
    ("other", "Other risk domain",
     "Does not fit the above categories."),
]

# --------------------------------------------------------------------------- #
# LLM client (standalone, reads env vars — never hardcode keys)
# --------------------------------------------------------------------------- #

class _RateLimiter:
    """Simple sliding-window rate limiter: at most `max_per_minute` calls/min.

    Thread-safe; blocks the calling thread if the limit would be exceeded.
    """

    def __init__(self, max_per_minute: int = 55) -> None:
        self.max_per_minute = max_per_minute
        self._lock = __import__("threading").Lock()
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        import time as _t
        with self._lock:
            now = _t.time()
            # Drop timestamps older than 60s
            cutoff = now - 60.0
            self._timestamps = [ts for ts in self._timestamps if ts > cutoff]
            if len(self._timestamps) >= self.max_per_minute:
                # Sleep until the oldest timestamp falls out of the window
                sleep_for = self._timestamps[0] + 60.0 - now + 0.1
                if sleep_for > 0:
                    _t.sleep(sleep_for)
                now = _t.time()
                cutoff = now - 60.0
                self._timestamps = [ts for ts in self._timestamps if ts > cutoff]
            self._timestamps.append(now)


class BucketLLM:
    """Standalone LLM client for bucketing. Reads credentials from env vars.

    Primary model (default deepseek-v4-flash) is tried first; on empty
    response / timeout / content-moderation refusal, automatically falls
    back to BUCKET_FALLBACK_MODEL (default minimax-m3) on a separate
    fallback endpoint (BUCKET_FALLBACK_BASE_URL / BUCKET_FALLBACK_API_KEY),
    falling back to the primary endpoint if not set.

    A built-in rate limiter caps requests to BUCKET_RPM (default 55) per
    minute to respect the proxy's 60-req/min limit.
    """

    def __init__(self) -> None:
        self.api_key = os.environ.get("BUCKET_API_KEY", "")
        self.base_url = os.environ.get("BUCKET_BASE_URL", "").rstrip("/")
        self.model = os.environ.get("BUCKET_MODEL", "deepseek-v4-flash")

        # Fallback (separate endpoint by default — minimax-m3 lives on
        # a different proxy than the primary deepseek-v4-flash endpoint)
        self.fallback_model = os.environ.get("BUCKET_FALLBACK_MODEL", "minimax-m3")
        self.fallback_base_url = os.environ.get(
            "BUCKET_FALLBACK_BASE_URL", ""
        ).rstrip("/") or self.base_url
        self.fallback_api_key = os.environ.get("BUCKET_FALLBACK_API_KEY", "") or self.api_key

        if not self.api_key or not self.base_url:
            raise RuntimeError(
                "Missing BUCKET_API_KEY or BUCKET_BASE_URL environment variable. "
                "Export them before running this script."
            )
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=0,
            timeout=120,
        )
        self._fallback_client = OpenAI(
            base_url=self.fallback_base_url,
            api_key=self.fallback_api_key,
            max_retries=0,
            timeout=120,
        )
        rpm = int(os.environ.get("BUCKET_RPM", "55"))
        self._limiter = _RateLimiter(max_per_minute=rpm)
        log.info(
            f"BucketLLM initialized: primary={self.model}@{self.base_url} "
            f"fallback={self.fallback_model}@{self.fallback_base_url} rpm={rpm}"
        )

    def _call(self, client: OpenAI, model: str, system: str, user: str) -> str:
        """Single LLM call. Raises on empty response. Rate-limited."""
        self._limiter.acquire()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=300,
            timeout=120,
        )
        raw = completion.choices[0].message.content or ""
        if not raw.strip():
            raise ValueError("Empty response from LLM")
        return raw

    def classify(self, slot_value: str, buckets: list[tuple[str, str, str]],
                 stage_label: str) -> tuple[str, str]:
        """Classify a single slot value into one of the buckets.

        Returns (bucket_id, raw_response). Retries primary model up to 2
        times; on persistent failure, falls back to fallback model once.
        """
        bucket_lines = []
        for i, (bid, label, desc) in enumerate(buckets, 1):
            bucket_lines.append(f"{i}. [{bid}] {label}: {desc}")
        bucket_text = "\n".join(bucket_lines)

        system = (
            "You are an AI risk taxonomy classifier. Your task is to assign a given "
            f"{stage_label} description to exactly ONE semantic bucket. "
            "Pick the FIRST bucket that best matches the description. "
            "Respond ONLY with a JSON object: {\"bucket_id\": \"...\", \"reason\": \"one short sentence\"}."
        )
        user = (
            f"Stage: {stage_label}\n\n"
            f"Description to classify:\n\"\"\"\n{slot_value}\n\"\"\"\n\n"
            f"Available buckets (pick the FIRST that matches):\n{bucket_text}\n\n"
            "Return JSON only."
        )

        last_err: Exception | None = None
        # Try primary model up to 2 times
        for attempt in range(2):
            try:
                raw = self._call(self._client, self.model, system, user)
                bucket_id = self._parse_bucket_id(raw, buckets)
                if bucket_id == "other" and "{" not in raw:
                    raise ValueError(f"No valid JSON in response: {raw[:80]}")
                return bucket_id, raw
            except Exception as e:
                last_err = e
                if attempt < 1:
                    time.sleep(1.0)
                    continue

        # All primary attempts failed → try fallback model once
        if self.fallback_model and self.fallback_model != self.model:
            log.info(
                f"Primary {self.model} failed ({str(last_err)[:60]}), "
                f"trying fallback {self.fallback_model}"
            )
            try:
                raw = self._call(self._fallback_client, self.fallback_model, system, user)
                bucket_id = self._parse_bucket_id(raw, buckets)
                return bucket_id, raw
            except Exception as fb_err:
                last_err = fb_err

        raise last_err if last_err else RuntimeError("classify failed after all retries")

    @staticmethod
    def _parse_bucket_id(raw: str, buckets: list[tuple[str, str, str]]) -> str:
        """Extract bucket_id from LLM JSON response, with fallbacks."""
        valid_ids = {bid for bid, _, _ in buckets}
        # Try JSON parse
        try:
            # Strip markdown fences
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            m = re.search(r'\{[\s\S]*\}', cleaned)
            if m:
                obj = json.loads(m.group(0))
                bid = obj.get("bucket_id", "").strip()
                if bid in valid_ids:
                    return bid
        except (json.JSONDecodeError, AttributeError):
            pass
        # Fallback: search for any valid bucket_id in the raw text
        for bid, _, _ in buckets:
            if bid in raw:
                return bid
        return "other"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_classifications() -> dict[str, str]:
    """event_id -> domain."""
    mapping = {}
    if CLASSIFICATION_PATH.exists():
        with open(CLASSIFICATION_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                mapping[entry["event_id"]] = entry.get("domain", "Other")
    return mapping


def load_chains() -> list[dict]:
    """Read every event_subgraph_fused.json and extract the 5-slot chain."""
    classifications = load_classifications()
    chains = []
    for json_path in sorted(OUTPUT_DIR.glob("pred_*/event_subgraph_fused.json")):
        event_id = json_path.parent.name
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        nodes_by_type: dict[str, list[str]] = {t: [] for t in SLOT_TYPES}
        for node in data.get("nodes", []):
            etype = node.get("entity_type", "")
            if etype in nodes_by_type:
                name = node.get("name", "").strip()
                if name:
                    nodes_by_type[etype].append(name)
        slots = {t: (nodes_by_type[t][0] if nodes_by_type[t] else None) for t in SLOT_TYPES}
        chains.append({
            "event_id": event_id,
            "domain": classifications.get(event_id, "Other"),
            "slots": slots,
        })
    return chains


def load_done(out_path: Path) -> set[str]:
    """Load event_ids already successfully processed (for resume).

    Only counts records with a valid (non-error) bucket_id. Records with
    bucket_id == 'error' are excluded so they get retried on the next run.
    """
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("bucket_id") not in ("error",):
                        done.add(rec["event_id"])
                except json.JSONDecodeError:
                    continue
    return done


# --------------------------------------------------------------------------- #
# Stage processing
# --------------------------------------------------------------------------- #

def process_llm_stage(
    stage: str,
    chains: list[dict],
    llm: BucketLLM,
    concurrency: int,
) -> None:
    """Run LLM bucketing for one stage (RiskSource/Consequence/Impact/AffectedActor)."""
    out_path = WEB_DATA / f"bucketing_{stage}.jsonl"
    buckets = BUCKET_DEFS[stage]
    done = load_done(out_path)
    log.info(f"[{stage}] {len(done)} events already done, {len(chains) - len(done)} remaining")

    # Build task list: only events with a non-empty slot value, not yet done
    tasks = []
    for c in chains:
        if c["event_id"] in done:
            continue
        val = c["slots"].get(stage)
        if not val or not val.strip():
            # Record empty slot as "missing" and skip LLM call
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event_id": c["event_id"],
                    "stage": stage,
                    "slot_value": "",
                    "bucket_id": "missing",
                    "raw_response": "",
                }, ensure_ascii=False) + "\n")
            continue
        tasks.append((c["event_id"], val.strip()))

    log.info(f"[{stage}] {len(tasks)} events to classify via LLM")

    # Write lock for jsonl (threads append)
    write_lock = __import__("threading").Lock()
    completed = 0
    failed = 0

    def classify_one(event_id: str, value: str) -> tuple[str, str, str, str]:
        """Returns (event_id, value, bucket_id, raw)."""
        try:
            bid, raw = llm.classify(value, buckets, stage)
            return event_id, value, bid, raw
        except Exception as e:
            log.warning(f"[{stage}] {event_id} failed: {e}")
            return event_id, value, "error", str(e)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(classify_one, eid, val): eid for eid, val in tasks}
        for fut in as_completed(futures):
            event_id, value, bid, raw = fut.result()
            rec = {
                "event_id": event_id,
                "stage": stage,
                "slot_value": value,
                "bucket_id": bid,
                "raw_response": raw,
            }
            with write_lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            completed += 1
            if bid == "error":
                failed += 1
            if completed % 50 == 0:
                log.info(f"[{stage}] progress: {completed}/{len(tasks)} (failed={failed})")

    log.info(f"[{stage}] DONE: {completed} classified, {failed} failed")


def process_risk_stage(chains: list[dict]) -> None:
    """Risk stage: map classification domain to bucket (no LLM)."""
    out_path = WEB_DATA / "bucketing_Risk.jsonl"
    log.info(f"[Risk] mapping classification domain to bucket for {len(chains)} events")
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chains:
            domain = c["domain"]
            bid = RISK_DOMAIN_TO_BUCKET.get(domain, "other")
            f.write(json.dumps({
                "event_id": c["event_id"],
                "stage": "Risk",
                "slot_value": c["slots"].get("Risk", ""),
                "domain": domain,
                "bucket_id": bid,
            }, ensure_ascii=False) + "\n")
    log.info(f"[Risk] DONE: wrote {out_path}")


# --------------------------------------------------------------------------- #
# Aggregation & summary
# --------------------------------------------------------------------------- #

def load_bucketing(stage: str) -> dict[str, str]:
    """event_id -> bucket_id for a stage."""
    out: dict[str, str] = {}
    path = WEB_DATA / f"bucketing_{stage}.jsonl"
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["event_id"]] = rec["bucket_id"]
    return out


def build_summary(chains: list[dict]) -> None:
    """Per-stage bucket counts + top-10 samples; cross-stage transitions."""
    log.info("Building summary...")
    stages = SLOT_TYPES
    bucketings = {s: load_bucketing(s) for s in stages}

    # Per-stage bucket counts + samples
    summary_path = WEB_DATA / "bucket_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "bucket_id", "count", "pct", "top_samples"])
        for stage in stages:
            bmap = bucketings[stage]
            counter: Counter = Counter()
            samples: dict[str, list[str]] = defaultdict(list)
            for c in chains:
                bid = bmap.get(c["event_id"], "missing")
                counter[bid] += 1
                val = c["slots"].get(stage, "")
                if val and len(samples[bid]) < 10:
                    samples[bid].append(val[:80])
            total = sum(counter.values())
            for bid, cnt in counter.most_common():
                pct = f"{cnt/total*100:.1f}%" if total else "0%"
                samp = " | ".join(samples[bid][:10])
                w.writerow([stage, bid, cnt, pct, samp])
    log.info(f"Wrote {summary_path}")

    # Cross-stage transitions (stage i -> stage i+1)
    trans_path = WEB_DATA / "bucket_transitions.csv"
    with open(trans_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["from_stage", "from_bucket", "to_stage", "to_bucket", "count"])
        for i in range(len(stages) - 1):
            s_from = stages[i]
            s_to = stages[i + 1]
            bmap_from = bucketings[s_from]
            bmap_to = bucketings[s_to]
            flow: Counter = Counter()
            for c in chains:
                a = bmap_from.get(c["event_id"], "missing")
                b = bmap_to.get(c["event_id"], "missing")
                if a != "missing" and b != "missing":
                    flow[(a, b)] += 1
            for (a, b), cnt in flow.most_common():
                w.writerow([s_from, a, s_to, b, cnt])
    log.info(f"Wrote {trans_path}")

    # Print quick distribution
    print("\n=== Bucket distribution by stage ===")
    for stage in stages:
        bmap = bucketings[stage]
        counter = Counter(bmap.get(c["event_id"], "missing") for c in chains)
        total = sum(counter.values())
        print(f"\n--- {stage} (n={total}) ---")
        for bid, cnt in counter.most_common():
            print(f"  {cnt:4d}  {cnt/total*100:5.1f}%  {bid}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "RiskSource", "Risk", "Consequence", "Impact", "AffectedActor", "summary"],
                    help="Which stage to process (default: all)")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="LLM concurrency (default: 4)")
    args = ap.parse_args()

    chains = load_chains()
    log.info(f"Loaded {len(chains)} chains")

    if args.stage == "summary":
        build_summary(chains)
        return

    if args.stage in ("RiskSource", "Consequence", "Impact", "AffectedActor"):
        llm = BucketLLM()
        if args.stage in BUCKET_DEFS:
            process_llm_stage(args.stage, chains, llm, args.concurrency)
        build_summary(chains)
        return

    if args.stage == "Risk":
        process_risk_stage(chains)
        build_summary(chains)
        return

    if args.stage == "all":
        # Risk first (no LLM, instant)
        process_risk_stage(chains)
        # Then LLM stages
        llm = BucketLLM()
        for stage in ["RiskSource", "Consequence", "Impact", "AffectedActor"]:
            process_llm_stage(stage, chains, llm, args.concurrency)
        build_summary(chains)
        return


if __name__ == "__main__":
    main()

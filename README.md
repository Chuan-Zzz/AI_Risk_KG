# AI Risk Knowledge Graph

> Event-centric, Evidence-constrained Extraction Pipeline based on LangGraph

## Overview

Automatically constructs event-centric AI risk knowledge graphs from multi-document incident reports. Single-event runs produce a validated event subgraph, while batch runs additionally perform cross-event entity fusion before writing the global graph to Neo4j.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run

```bash
# Single event
python main.py single --event_id pred_0

# Batch (first 10 events)
python main.py batch --dataset all --max_events 10

# All events
python main.py batch --dataset all

# Built-in evaluation subsets
python main.py batch --dataset gold
python main.py batch --dataset eval500
```

## Pipeline Architecture

```
input → explicit_extract → aggregation → entity_reuse → risk_chain → inference → graph_build → validation → store
```

Batch mode adds a second phase after per-event extraction:

```text
Phase 1: per-event extraction → output/<event_id>/event_subgraph.json
Phase 2: cross-event fusion   → Neo4j global KG
```

| Stage | Node | LLM? | Description |
|-------|------|------|-------------|
| 1 | `input_node` | No | Compute metadata: report count, time range, sources, languages |
| 2 | `explicit_extract` | Yes | Per-document ontology-aligned explicit entity extraction |
| 3 | `aggregation` | No | Cross-document entity merge with adaptive strategies |
| 4 L1 | `entity_reuse` | No | Normalize and deduplicate entities |
| 4 L2 | `risk_chain` | Yes | Risk propagation chain slot filling |
| 4 L3 | `inference` | Yes | Controlled inference of purpose, lifecycle, domain, roles, and governance hints |
| 5 | `graph_build` | No | Assemble EventKnowledgeSubgraph with all nodes, edges, statements |
| - | `validation` | No | Ontology constraint + evidence + risk chain completeness check |
| - | `store` | No | Output JSON + TTL for each event |

### Extraction Modes

| Mode | Stage | Meaning |
|------|-------|---------|
| `explicit` | 2 | Directly mentioned in text, with evidence sentence |
| `abstracted` | 4 L2 | Generalized from evidence (e.g., "model hallucination") |
| `inferred` | 4 L3 | Reasonably deduced with evidence + reasoning chain |
| `completed` | 5 | System-generated (e.g., AIRiskIncident node) |

### Stage 3 Aggregation Strategies

| Strategy | Report Count | Behavior |
|----------|-------------|----------|
| Small | 1-5 | Direct merge all entities |
| Medium | 6-30 | Select 5-8 representative reports, dedup evidence |
| Large | 30+ | Chunked processing (12 per chunk), global merge |

## Data

- **Events**: `data/inferred_event_structure_6124.json` — 2034 events mapping to case IDs
- **Cases**: `data/eval_cases.jsonl` (distributed via Releases, see below) or `data/translated_docs.json`

If `eval_cases.jsonl` is absent, the pipeline falls back to `translated_docs.json`.

## Data Availability

Full construction outputs and the source corpus are published as GitHub [Release assets](https://github.com/Chuan-Zzz/AI_Risk_KG/releases):

| Asset | Contents |
|-------|----------|
| `ontorisk_intermediate_graphs.zip` | Per-event `event_subgraph.json`, `event_subgraph_fused.json`, and `.ttl` for all 2,034 events |
| `ontorisk_ablation_outputs.zip` | Outputs of all ablation and baseline variants |
| `ontorisk_source_data.zip` | Source corpus (`eval_cases.jsonl`) and event structure needed to re-run the pipeline |

Human annotation results and computed metrics are tracked directly in this repository under `eval/annotations/` and `eval/results/`. Three sample events are included under `examples/` so the output format can be inspected without downloading the archives.

## Annotation Data

Three-level human evaluation was conducted with Label Studio (multiple annotators plus arbitration), and all results are public:

| Level | Files | Metrics |
|-------|-------|---------|
| 1 — Explicit entities | `eval/annotations/level1_entity_*.json` | Precision / Recall / F1 (exact + partial match) |
| 2 — Risk chain | `eval/annotations/level2_risk_chain_*.json` | Slot accuracy |
| 3 — Inference fields | `eval/annotations/level3_inference_*.json` | Field accuracy + evidence support rate |
| ESR validation | `eval/annotations/esr_annotation.json` | Evidence support re-annotation |

Annotation tasks, label configs, and arbitration assets live in `eval/label_studio/`; the protocol is documented in `docs/evaluation/annotation_guide.md` and `docs/evaluation/evaluation_plan.md`. All reported metrics can be recomputed offline:

```bash
python eval/scripts/compute_metrics.py               # pre-arbitration metrics
python eval/scripts/compute_metrics_arbitrated.py    # post-arbitration metrics
python eval/scripts/compute_iaa_v2.py                # inter-annotator agreement
```

## Project Structure

```
├── config/
│   ├── config.yml                    # Main configuration
│   └── ontology.yml                  # AIRO ontology constraints (domain/range)
├── data/                             # Input data files
│   └── AIRO_extended.ttl             # AI Risk Incident Ontology (extended AIRO, v2.0)
├── examples/                         # Sample per-event outputs (subgraph + fused graph)
├── src/
│   ├── alignment/
│   │   ├── kg_fusion.py              # Batch-mode cross-event fusion
│   │   ├── semantic_aligner.py       # Embedding + BM25 alignment
│   │   └── llm_verifier.py           # LLM boundary-case verification
│   ├── core/
│   │   ├── models.py                 # Pydantic models (43 classes, 37 relations)
│   │   ├── ontology.py               # Ontology validator + normalizer
│   │   ├── config.py                 # Config loader with env var interpolation
│   │   ├── datasets.py               # Event/case dataset loader
│   │   ├── runtime.py                # Shared event-ID and output-path helpers
│   │   └── llm.py                    # OpenAI-compatible LLM client
│   ├── agent/
│   │   ├── state.py                  # PipelineState TypedDict
│   │   ├── graph.py                  # LangGraph StateGraph definition
│   │   ├── nodes/                    # Pipeline stages (input → store)
│   │   └── prompts/                  # Stage 2 / Stage 4 prompts
│   ├── storage/
│   │   ├── ttl_store.py              # RDF/Turtle output (airo: namespace)
│   │   └── neo4j_store.py            # Neo4j graph database
│   ├── experiments/
│   │   └── ablation.py               # Ablation study runner (variant pipelines)
│   └── utils/
│       ├── text.py                   # Entity merge, ID generation, normalization
│       └── logger.py                 # Logging setup
├── eval/
│   ├── annotations/                  # Exported human annotation results (Levels 1-3, ESR)
│   ├── label_studio/                 # Annotation tasks, label configs, arbitration assets
│   ├── results/                      # Computed metrics and LLM judgments
│   └── scripts/                      # Evaluation and analysis commands
├── docs/
│   └── evaluation/                   # Evaluation plan, annotation guide, ablation report
├── scripts/                          # Graph statistics, risk classification, and paper figure generation
│   └── data/                         # Precomputed statistics and classification results
├── main.py                           # CLI entry point
├── LICENSE
├── CITATION.cff
└── requirements.txt
```

## Commands

<!-- AUTO-GENERATED -->
| Command | Description |
|---------|-------------|
| `python main.py single --event_id <ID>` | Process a single event by ID |
| `python main.py batch --dataset <all\|gold\|eval500> --max_events <N>` | Process the first N events in a dataset |
| `python main.py batch --dataset <all\|gold\|eval500>` | Process all events in a dataset (skips valid cached output) |
| `python main.py batch --dataset <...> --force` | Reprocess existing event output |
<!-- /AUTO-GENERATED -->

## Environment Variables

<!-- AUTO-GENERATED -->
| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `LLM_MODEL` | No | Override the default chat model | `DeepSeek-V4-Flash` |
| `LLM_BASE_URL` | No | Override the default OpenAI-compatible base URL | `https://api.ldwnb666.xyz/v1` |
| `LLM_API_KEY` | Yes | OpenAI-compatible API key | `sk-...` |
| `NEO4J_URI` | No | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | No | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | No | Neo4j password | `your_password` |
| `NEO4J_DATABASE` | No | Neo4j database name | `kgclean` |
| `EMBEDDING_PROVIDER` | No | Embedding backend: `local` or `openai` | `openai` |
| `EMBEDDING_MODEL` | No | Embedding model name or local path | `mlx-community/bge-m3-mlx-fp16` |
| `EMBEDDING_BASE_URL` | No | OpenAI-compatible embeddings endpoint | `http://127.0.0.1:8000/v1` |
| `EMBEDDING_API_KEY` | No | Embedding service API key | `your_local_key` |
| `EMBEDDING_DIMENSION` | No | Embedding vector dimension | `1024` |
| `EMBEDDING_DEVICE` | No | Embedding runtime device | `cpu` |
| `EMBEDDING_BATCH_SIZE` | No | Embedding encode batch size | `32` |
| `EMBEDDING_TIMEOUT` | No | Embedding request timeout in seconds | `180` |
| `TAVILY_API_KEY` | No | Tavily web search API key | `tvly-...` |
| `BING_API_KEY` | No | Bing search API key | `...` |
<!-- /AUTO-GENERATED -->

### Local oMLX Embeddings

If you run `bge-m3` through a local OpenAI-compatible service such as `oMLX`, set:

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=mlx-community/bge-m3-mlx-fp16
EMBEDDING_BASE_URL=http://127.0.0.1:8000/v1
EMBEDDING_API_KEY=your_local_omlx_key
```

The project will call `/v1/embeddings` directly and still normalize vectors before similarity scoring.

## Output

Each successful event run produces files under `output/<event_id>/`:

| File | Format | Description |
|------|--------|-------------|
| `event_subgraph.json` | JSON | Phase-1 EventKnowledgeSubgraph with nodes, edges, statements |
| `event_subgraph_fused.json` | JSON | Phase-2 subgraph after cross-event entity fusion |
| `event_subgraph.ttl` | Turtle/RDF | Semantic graph using `airo:` namespace (https://w3id.org/airo#) |

Neo4j output is written during batch-mode phase 2 after cross-event fusion, if Neo4j is configured and reachable. The full set of per-event outputs for all 2,034 events is distributed via GitHub Releases (see [Data Availability](#data-availability)); `examples/` contains three sample events.

### Risk Propagation Chain

```
RiskSource ──causes──→ Risk ──leadsTo──→ Consequence ──impacts──→ Impact ──affects──→ AffectedActor
```

### Entity Types

| Category | Types |
|----------|-------|
| Technical | AISystem, AIModel, AITechnique, AICapability |
| Stakeholder | Stakeholder, AIDeveloper, AIProvider, AIDeployer, AIUser, Regulator, AffectedActor |
| Risk | RiskSource, Risk, Hazard, Threat, Vulnerability, Consequence, Impact, RiskControl |
| Governance | Regulation, Standard, EnforcementAction |
| Evidence | Evidence, KnowledgeStatement, RoleAssignment |
| Event | AIRiskIncident |
| Auxiliary | Purpose, AILifecyclePhase, Domain |

## License

MIT

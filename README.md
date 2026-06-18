# AI Risk Knowledge Graph

> Event-centric, Evidence-constrained Extraction Pipeline based on LangGraph

## Overview

Automatically constructs knowledge graphs from 6124 AI risk incident reports across 2034 events. Each event is processed through a 5-stage LangGraph pipeline that extracts entities, aggregates multi-document evidence, fills risk chain slots, performs controlled inference, and outputs validated knowledge subgraphs.

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
python main.py batch --max_events 10

# All events
python main.py batch --all
```

## Pipeline Architecture

```
input → explicit_extract → aggregation → entity_reuse → risk_chain → inference → graph_build → validation → store
                                                                                           ↑               ↓
                                                                                           └── retry (≤3) ←┘
```

| Stage | Node | LLM? | Description |
|-------|------|------|-------------|
| 1 | `input_node` | No | Compute metadata: report count, time range, sources, languages |
| 2 | `explicit_extract` | Yes | Per-document extraction of 7 entity types |
| 3 | `aggregation` | No | Cross-document entity merge with adaptive strategies |
| 4 L1 | `entity_reuse` | No | Normalize and deduplicate entities |
| 4 L2 | `risk_chain` | Yes | Risk propagation chain slot filling |
| 4 L3 | `inference` | Yes | Controlled inference of Purpose, LifecyclePhase, Domain |
| 5 | `graph_build` | No | Assemble EventKnowledgeSubgraph with all nodes, edges, statements |
| - | `validation` | No | Ontology constraint + evidence + risk chain completeness check |
| - | `store` | No | Output JSON + TTL + Neo4j |

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
- **Cases**: `data/eval_cases.jsonl` — 6124 AI risk incident reports

Distribution: 88.5% small events (1-5 reports), 11.1% medium (6-30), 0.3% large (30+).

## Project Structure

```
├── config/
│   ├── config.yml                    # Main configuration
│   └── ontology.yml                  # AIRO ontology constraints (domain/range)
├── data/                             # Input data files
├── src/
│   ├── core/
│   │   ├── models.py                 # Pydantic models (41 classes, 37 relations)
│   │   ├── ontology.py               # Ontology validator + normalizer
│   │   ├── config.py                 # Config loader with env var interpolation
│   │   └── llm.py                    # OpenAI-compatible LLM client
│   ├── agent/
│   │   ├── state.py                  # PipelineState TypedDict
│   │   ├── graph.py                  # LangGraph StateGraph definition
│   │   ├── nodes/
│   │   │   ├── input_node.py         # Stage 1
│   │   │   ├── explicit_extract.py   # Stage 2
│   │   │   ├── aggregation.py        # Stage 3
│   │   │   ├── entity_reuse.py       # Stage 4 L1
│   │   │   ├── risk_chain.py         # Stage 4 L2
│   │   │   ├── inference.py          # Stage 4 L3
│   │   │   ├── graph_build.py        # Stage 5
│   │   │   ├── validation.py         # Constraint validation
│   │   │   └── store.py              # Output to JSON/TTL/Neo4j
│   │   └── prompts/
│   │       ├── stage2_explicit.py
│   │       ├── stage4_risk_chain.py
│   │       └── stage4_inference.py
│   ├── storage/
│   │   ├── ttl_store.py              # RDF/Turtle output (airo: namespace)
│   │   └── neo4j_store.py            # Neo4j graph database
│   └── utils/
│       ├── text.py                   # Entity merge, ID generation, normalization
│       └── logger.py                 # Logging setup
├── tests/                            # 302 test cases
├── main.py                           # CLI entry point
├── airo_extended_en.ttl              # Extended AIRO ontology
└── requirements.txt
```

## Commands

<!-- AUTO-GENERATED -->
| Command | Description |
|---------|-------------|
| `python main.py single --event_id <ID>` | Process a single event by ID |
| `python main.py batch --max_events <N>` | Batch process first N events |
| `python main.py batch --all` | Process all events (skips existing by default) |
| `python -m pytest tests/ -v` | Run test suite |
| `python -m pytest tests/ -q` | Run tests (quiet) |
<!-- /AUTO-GENERATED -->

## Environment Variables

<!-- AUTO-GENERATED -->
| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `LLM_API_KEY` | Yes | OpenAI-compatible API key | `sk-...` |
| `NEO4J_URI` | No | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | No | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | No | Neo4j password | `your_password` |
| `TAVILY_API_KEY` | No | Tavily web search API key | `tvly-...` |
| `BING_API_KEY` | No | Bing search API key | `...` |
<!-- /AUTO-GENERATED -->

## Output

Each event produces three outputs in `output/<event_id>/`:

| File | Format | Description |
|------|--------|-------------|
| `event_subgraph.json` | JSON | Full EventKnowledgeSubgraph with nodes, edges, statements |
| `event_subgraph.ttl` | Turtle/RDF | Semantic graph using `airo:` namespace (https://w3id.org/airo#) |
| Neo4j | Graph DB | Nodes with typed labels, relationships with evidence properties |

### Risk Propagation Chain

```
RiskSource ──causes──→ Risk ──leadsTo──→ Consequence ──impacts──→ Impact ──affects──→ AffectedActor
```

### Entity Types (41 total)

| Category | Types |
|----------|-------|
| Technical | AISystem, AIModel, AITechnique, AICapability |
| Stakeholder | Stakeholder, AIDeveloper, AIProvider, AIDeployer, AIUser, Regulator, AffectedActor |
| Risk | RiskSource, Risk, Hazard, Threat, Vulnerability, Consequence, Impact, RiskControl |
| Governance | Regulation, Standard, EnforcementAction |
| Evidence | Evidence, KnowledgeStatement, RoleAssignment |
| Event | AIRiskIncident |
| Auxiliary | Purpose, AILifecyclePhase, Domain |

## Testing

```bash
# Run all tests (302 test cases)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_stage5.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Test coverage: models, ontology normalization/validation, all 5 stages, validation logic.

## License

MIT

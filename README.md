# TariffIQ — US Import Tariff Intelligence Platform

**Course:** DAMG 7245 — Big Data and Intelligent Analytics (Team 3)
**Institution:** Northeastern University, College of Engineering — Spring 2026
---
> WE ATTEST THAT WE HAVEN'T USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK.

| Member | Contribution |
|---|---|
| Ayush Fulsundar | 33.3% |
| Ishaan Samel | 33.3% |
| Vaishnavi Srinivas | 33.3% |
---
## Live Application

| Component | Link |
|---|---|
| Demo Video |https://youtu.be/6uGhk5VQIAw |
| CodeLabs | [Codelab Preview](https://codelabs-preview.appspot.com/?file_id=1yQYEhEw4kgdSCLw9ahbBzbuS6SimOu1LME1asgVXb98#0) |
| Proposal Document | [Google Docs](https://docs.google.com/document/d/17Li9KOo8oT_stR5Ub6cVK2EzPJdUgTTmyNiLRfbTfdI/edit?usp=sharing) |
| Streamlit App | [http://34.45.251.65:8502](http://34.45.251.65:8502) |
| FastAPI Docs | [http://34.45.251.65:8001/docs](http://34.45.251.65:8001/docs) |
| Airflow DAGs | [http://34.45.251.65:9090/home](http://34.45.251.65:9090/home) |


---

## What is TariffIQ?

US import tariffs are split across three government systems that were never designed to work together. The USITC Harmonized Tariff Schedule has duty rates but no legal context. The Federal Register has the legal context behind every tariff action but no rates. The Census Bureau has actual import volumes but neither. Getting a complete answer for one product from one country means manually cross-referencing all three, which takes 20-30 minutes per query.

TariffIQ connects all three through a conversational multi-agent RAG pipeline. A procurement professional types a plain English question and gets a cited, accurate answer in seconds. Every duty rate comes from a Snowflake SQL lookup, every policy claim traces to a Federal Register document number, and every trade volume is pulled live from the Census Bureau API. The LLM synthesizes and cites but never generates numbers from memory.

**Sample query:** `"electric vehicles from China"`
- HTS code 8703.80.00 with 95% classification confidence
- 102.5% total effective duty (2.5% base MFN + 100% Section 301 adder via Chapter 99 code 9903.91.03)
- Full Section 301 policy history from 2018 to 2025 with Federal Register document numbers and dates
- Live Census Bureau import values and year-over-year trend
- Top 8 sourcing alternatives with comparative duty rates
- Tariff exposure score and sourcing recommendation

---

## System Architecture
<img width="1395" height="1230" alt="image" src="https://github.com/user-attachments/assets/3ce39a33-9edf-4caf-8d08-c5e8674b2c42" />


## The 7-Agent Pipeline

All agents share a `TariffState` TypedDict defined in `agents/state.py` that carries results forward through the pipeline. Each agent has one job.

### 1. Query Agent (`agents/query_agent.py`)

Parses the user query into `{product, country}` using GPT-4o-mini with a trade-aware system prompt. Handles spelling correction, product alias normalization (EV to electric vehicles, PV to solar panels), and country alias resolution (PRC to China, ROK to South Korea). Runs a SQL-based ambiguity check against `HTS_CODES` — if the product maps to 3+ distinct HTS headings, it returns clarification options instead of guessing. Detects query intent: standard rate lookup, rate change history, or country comparison.

Cache: exact match (24h) and semantic similarity cache (cosine above 0.92, 24h TTL).

### 2. Classification Agent (`agents/classification_agent.py`)

Three-layer HTS classification. First checks `PRODUCT_ALIASES` for previously confirmed mappings. Then runs ChromaDB semantic search against `hts_descriptions`. Then runs Snowflake keyword search against `HTS_CODES` with heading and chapter filtering. LLM (GPT-4o-mini) selects the best code from all candidates. If confidence falls below 0.80, the query goes to HITL — written to `HITL_RECORDS` and the user receives clarification suggestions built from real HTS subheadings. Human corrections write back to `PRODUCT_ALIASES` for future queries.

### 3. Base Rate Agent (`agents/base_rate_agent.py`)

SQL lookup on `HTS_CODES` for the confirmed HTS code. Parses the `special_rate` column for FTA program codes (CA/MX for USMCA, KR for KORUS FTA, AU for US-Australia FTA, A for GSP, etc.) and returns the preferential rate when the query country qualifies. Also parses the `footnotes` column for Chapter 99 cross-references (9903.xx.xx patterns) and passes them to the adder rate agent.

### 4. Adder Rate Agent (`agents/adder_rate_agent.py`)

Determines the Section 301, Section 232, or IEEPA adder through three steps.

Step 4 (Chapter 99 lookup) parses HTS footnotes for Chapter 99 surcharge codes, queries `HTS_CODES` for the adder rate, and filters China-specific codes (9903.88.xx, 9903.91.xx) for non-China queries at collection time. If footnotes are empty, scans all five `NOTICE_HTS_CODES` tables for Chapter 99 references in context snippets.

Step 4b (Global adder lookup) handles non-China countries using ChromaDB with HyDE and LLM extraction to catch Section 232 steel/aluminum and IEEPA reciprocal tariffs.

Step 5 (Notice table scan) queries all five notice-HTS linkage tables, collects context snippets, and passes them to the LLM with a structured JSON extraction prompt returning `{adder_rate, document_number, basis}`.

Priority: Chapter 99 footnote rate over notice LLM rate over 0.0. If FTA applied, adder = 0.0.

Cache: 1 hour keyed on hts_code and country.

### 5. Policy Agent (`agents/policy_agent.py`)

The RAG agent. Runs in four stages. First, HyDE enhancement (`services/retrieval/hyde.py`) generates a hypothetical Federal Register excerpt using the confirmed HTS subheading, country, and tariff action type. This is used as the search query because FR chunks use legal language, not plain English. Second, multi-agency hybrid retrieval (`services/retrieval/hybrid.py`) runs six parallel ChromaDB searches across CBP, USITC, ITA, EOP, and main collections using BM25 plus dense plus RRF fusion. Third, an exhaustive Snowflake HTS-linked load fetches all chunks for every document in `NOTICE_HTS_CODES` for this HTS and reranks with BM25. Fourth, a final BM25 filter with `min_score_ratio=0.15` drops low-relevance chunks before the LLM synthesizes.

Cache: 6 hours keyed on hts_code and query hash.

### 6. Trade Agent (`agents/trade_agent.py`)

Calls the Census Bureau API live for the specific HS6 code. Fetches the most recent available month, trailing 24-month trend for year-over-year comparison, and top 8 import partner countries with base MFN/FTA rates attached. HTTP 204 suppressed data is handled gracefully.

### 7. Synthesis Agent (`agents/synthesis_agent.py`)

Generates a structured 7-section Markdown response covering classification, all charges, policy notices, alternative sourcing, historical trail, Census snapshot, and top partners. Citation building applies a data-driven relevance filter using Snowflake `NOTICE_HTS_CODES` linkage as the primary gate. It drops antidumping/CVD trade remedy investigation docs and opioid/border enforcement docs that are structurally never relevant to product tariff rates. Every FR document number in the response is verified against Snowflake before returning. Hallucinated document numbers trigger HITL escalation.

---

## Data Sources and Ingestion

### USITC Harmonized Tariff Schedule

The complete HTS schedule with 35,733 product codes including Chapter 99 special tariff program codes for Section 301, Section 232, and IEEPA surcharges. Ingested via `airflow/dags/hts_ingest_dag.py` using `ingestion/usitc_client.py` and `ingestion/hts_idempotent_load.py` into `TARIFFIQ.RAW.HTS_CODES`. Idempotent MERGE so it is safe to re-run.

### Federal Register Notices (5 Agencies)

Full HTML body text of every tariff-related notice from 2018 to present across USTR, CBP, USITC, EOP, and ITA. Five independent Airflow DAGs under `airflow/dags/`. spaCy EntityRuler in `ingestion/hts_extractor.py` and the agency-specific extractors (eop, ita, itc) extract HTS codes from notice text. Extracted codes land in five `NOTICE_HTS_CODES` linkage tables with surrounding context snippets. All notice text is chunked via `ingestion/chunker.py`, embedded via `ingestion/embedder.py`, and stored in both Snowflake CHUNKS tables and ChromaDB `policy_notices`.

### Census Bureau Trade Data

Queried live at query time via `ingestion/census_client.py`. No bulk ingestion. The Trade Agent calls the Census timeseries API for the specific HS6 code on each request.

---

## Retrieval System

**HyDE** (`services/retrieval/hyde.py`) generates a realistic Federal Register paragraph before every ChromaDB search. The prompt includes the full HTS subheading, the country, and the expected tariff action type so the search vector matches the language of real FR chunks rather than plain English questions.

**Hybrid Retriever** (`services/retrieval/hybrid.py`) combines ChromaDB dense vector search with BM25Okapi sparse keyword search over ~32,000 chunks built in memory at startup. Fused via Reciprocal Rank Fusion with dense weight 0.6, sparse weight 0.4, rrf_k 60.

**LiteLLM Router** (`services/llm/router.py`) routes all LLM calls with per-task model selection, automatic provider fallback, daily budget enforcement, and per-call token and cost logging.

**Redis Cache** covers exact query cache, semantic similarity cache, adder rate cache (1h), and policy context cache (6h).

---

## Guardrails and HITL

Input guardrails check for prompt injection patterns, query length, and tariff signal coverage before the pipeline runs.

Output guardrails verify every Federal Register document number cited in the synthesis response against Snowflake before returning. Hallucinated document numbers go to HITL and never reach the user.

HITL triggers on classification confidence below 0.80 and on citation validation failure in synthesis. Both write to `HITL_RECORDS`. Human corrections to classification write back to `PRODUCT_ALIASES` with confidence 0.90 for future queries.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Cloud | Google Cloud Platform |
| Data Warehouse | Snowflake |
| Pipeline Orchestration | Apache Airflow |
| Agent Framework | LangGraph |
| Backend API | FastAPI |
| Vector Store | ChromaDB 0.4.24 |
| LLM Routing | LiteLLM |
| LLMs | Claude Haiku (policy + synthesis), GPT-4o-mini (classification + parsing) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Sparse Search | rank-bm25 |
| Cache | Redis 7 |
| NLP | spaCy EntityRuler |
| Frontend | Streamlit |
| Containers | Docker Compose |
| Airflow Metadata DB | PostgreSQL 15 |

---

## Prerequisites

- Docker Desktop with at least 8GB RAM
- Snowflake account with a warehouse and database provisioned
- OpenAI API key
- Anthropic API key
- AWS account (for S3 raw doc storage, optional)

---

## Setup

```bash
git clone https://github.com/BigDataIA-Spring26-Team-03/TarrifIQ.git
cd TarrifIQ
cp .env.example .env      # fill in credentials
docker compose up --build -d
```

ChromaDB rebuilds its vector index from Snowflake on every FastAPI container start. First startup takes 5-10 minutes.

```bash
docker compose logs -f fastapi
# Ready when you see: "Application startup complete"
```

Verify all services are up:

```bash
curl http://localhost:8001/health
```
---

## Environment Variables

Create a `.env` file at the project root with the following:

```bash
# Snowflake
SNOWFLAKE_USER=
SNOWFLAKE_PASSWORD=
SNOWFLAKE_ACCOUNT=
SNOWFLAKE_WAREHOUSE=
SNOWFLAKE_DATABASE=
SNOWFLAKE_SCHEMA=

# LLM
LLM_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# AWS (for S3 raw doc storage)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET=tariffiq-raw-docs

# Census Bureau
CENSUS_API_KEY=

# Redis (managed inside Docker, do not change)
REDIS_URL=redis://redis:6379

# ChromaDB (managed inside Docker, do not change)
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000

# Airflow
AIRFLOW_SECRET_KEY=

# FastAPI URL (used by Streamlit)
FASTAPI_URL=http://fastapi:8001
```

---

## Try It Out

Open http://34.45.251.65:8502 and type any of these into the chat:

```
electric vehicles from China
solar panels from China
semiconductors from China
steel wire from Germany
```

---

## Testing

```bash
docker compose exec fastapi pytest tests/ -v
```

---

## Project Structure

```
agents/
  query_agent.py              query parsing, intent detection, ambiguity check
  classification_agent.py     3-layer HTS classification with HITL
  base_rate_agent.py          MFN rate lookup, FTA detection, footnote extraction
  adder_rate_agent.py         Section 301/232/IEEPA adder determination
  policy_agent.py             HyDE + hybrid RAG + BM25 reranking
  trade_agent.py              live Census Bureau API
  synthesis_agent.py          response generation + citation relevance filter
  tools.py                    tool registry, all data access (16 tools)
  graph.py                    LangGraph pipeline definition and routing
  state.py                    TariffState TypedDict

services/
  llm/router.py               LiteLLM routing, budget enforcement, system prompts
  retrieval/hyde.py            HyDE query enhancement
  retrieval/hybrid.py          BM25 + ChromaDB dense + RRF fusion
  chromadb_init.py             ChromaDB collection builder (runs on startup)

api/
  main.py                     FastAPI app entry point
  schemas.py                  Pydantic request/response schemas
  tools/hts_search.py         HTS keyword search endpoint
  tools/resolve_hts_rate.py   rate resolution endpoint
  tools/search_policy_vector.py  policy vector search endpoint
  tools/search_hts_vector.py     HTS vector search endpoint

ingestion/
  usitc_client.py             USITC HTS API client
  federal_register_client.py  Federal Register API client
  census_client.py            Census Bureau API client
  cbp_client.py               CBP FR API client
  eop_client.py               EOP FR API client
  ita_client.py               ITA FR API client
  itc_client.py               ITC FR API client
  hts_extractor.py            spaCy HTS code extraction
  eop_hts_extractor.py        EOP-specific HTS extraction
  ita_hts_extractor.py        ITA-specific HTS extraction
  itc_hts_extractor.py        ITC-specific HTS extraction
  html_parser.py              FR HTML body extraction
  chunker.py                  text chunking
  embedder.py                 sentence-transformer embedding
  snowflake_writer.py         idempotent Snowflake writes
  cross_source_validator.py   cross-source HTS validation

airflow/dags/
  hts_ingest_dag.py           USITC HTS schedule ingestion
  federal_register_dag.py     USTR notice ingestion
  cbp_federal_register_dag.py CBP notice ingestion
  eop_federal_register_dag.py EOP proclamation ingestion
  ita_federal_register_dag.py ITA notice ingestion
  itc_federal_register_dag.py ITC document ingestion

snowflake/
  schema.sql                  full Snowflake schema DDL
  migrations/                 incremental schema migrations
  run_migrations.py           migration runner

validation/
  rate_reconciliation.py      cross-source rate validation

storage/
  chromadb_client.py          ChromaDB HTTP client wrapper

scripts/
  rebuild_chromadb.py                 force ChromaDB rebuild from Snowflake
  patch_chromadb_hts_chapter.py       backfill hts_chapter metadata
  patch_chromadb_publication_date.py  backfill publication_date metadata

tests/unit/
  test_hts_parser.py          HTS rate string parsing
  test_census_client.py       Census API client
  test_fr_pipeline.py         Federal Register ingestion pipeline
  test_fr_extractor.py        HTS code extraction from FR text
  tariffiq_test_suite.py      end-to-end pipeline test suite
  test_comprehensive.py       comprehensive agent tests

mcp_server.py                 MCP server exposing pipeline as tools
streamlit/app.py              Streamlit frontend
docker-compose.yaml           all services
Dockerfile.api                FastAPI container
Dockerfile.streamlit          Streamlit container
Dockerfile.airflow            Airflow container
requirements.txt              Python dependencies
requirements-api.txt          API-specific dependencies
```

---

## Team Contributions

**Ayush Fulsundar**
- Cloud infrastructure and GCP deployment
- Airflow DAG orchestration for all 5 data sources
- LiteLLM router and model configuration
- FastAPI service layer and tool registry

**Ishaan Samel**
- Streamlit frontend and all UI components
- LangGraph pipeline definition and agent routing
- Trade agent and Census Bureau integration
- Adder rate agent (Chapter 99 lookup, Section 232/301/IEEPA)
- Docker Compose setup and containerization

**Vaishnavi Srinivas**
- Query Agent, Classification agent and HTS classification pipeline
- Policy agent (HyDE, hybrid retrieval, BM25 reranking)
- Streamlit frontend and all UI components
- Synthesis agent and citation relevance filter
- HITL system and guardrails

---

**AI Usage Disclosure:** Claude (Anthropic) was used as a coding assistant for agent logic, retrieval pipeline design, and citation filtering. All architectural decisions, data modeling, and validation were made and verified by the team.

---

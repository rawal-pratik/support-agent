# Support Triage Agent

A Python 3.11+ AI-powered support triage agent that processes support tickets through a complete pipeline: ingestion, semantic retrieval, safety policy evaluation, LLM generation, and structured output.

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Corpus    │───▶│  Ingestion   │───▶│  Supabase    │    │              │
│  (Markdown) │    │  (chunks,    │    │  pgvector    │    │              │
└─────────────┘    │  embeddings) │    │  storage     │    │              │
                   └──────────────┘    └──────┬───────┘    │              │
                                              │            │              │
┌─────────────┐    ┌──────────────┐    ┌──────▼───────┐   │   Orchestrator│
│   Ticket    │───▶│    Policy    │───▶│  Retrieval   │───▶│  (pipeline)  │
│  (Issue,    │    │  (escalate,  │    │  (semantic)  │    │              │
│  Subject)   │    │   validate)  │    └──────────────┘    └──────┬───────┘
└─────────────┘    └──────────────┘                               │
                                                            ┌──────▼───────┐
                                                            │     LLM      │
                                                            │  (OpenRouter)│
                                                            └──────┬───────┘
                                                                   │
                                                            ┌──────▼───────┐
                                                            │  AgentResult │
                                                            │  (status,    │
                                                            │   product_area,│
                                                            │   response,  │
                                                            │   justification,│
                                                            │   request_type)│
                                                            └──────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| **CLI** | Terminal interface: `ingest`, `search`, `triage`, `evaluate` |
| **Ingestion** | Chunks markdown corpus, generates embeddings, stores in Supabase pgvector |
| **Policy** | Safety layer: prompt injection detection, empty ticket check, security/privacy escalation |
| **Retrieval** | Semantic vector search via Supabase pgvector |
| **LLM** | OpenRouter provider for structured generation |
| **Orchestrator** | Coordinates policy → retrieval → LLM → validation |
| **FastAPI** | HTTP API: `/health`, `/triage`, `/triage/batch` |
| **Docker/Render** | Production deployment packaging |

## Prerequisites

- Python 3.11+
- Supabase project with pgvector extension enabled
- Google Gemini API key (for embeddings)
- OpenRouter API key (for LLM)

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip and install project
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
# Edit .env with your actual values
```

### Required Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Supabase project URL (e.g., `https://xyz.supabase.co`) | Yes |
| `SUPABASE_KEY` | Supabase key with INSERT permission on document_chunks (service role key recommended for server-side) | Yes |
| `GEMINI_API_KEY` | Google AI Studio / Vertex AI key for embeddings | Yes |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM | Yes |
| `OPENROUTER_MODEL` | OpenRouter model identifier (e.g., `openai/gpt-4o-mini`) | Yes |
| `EMBEDDING_MODEL` | Embedding model name (default: `models/gemini-embedding-001`) | No |
| `EMBEDDING_BATCH_SIZE` | Batch size for embedding generation (default: `1`) | No |

**Never commit `.env` or real API keys to version control.**

## Supabase Setup

### 1. Enable pgvector Extension

In Supabase SQL editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Apply Migration

Run the migration to create the document chunks table:

```sql
-- See supabase/migrations/ for the full migration
-- This creates: document_chunks table with vector column
```

Or apply via Supabase CLI:

```bash
supabase db push
```

### 3. Verify Table Exists

```sql
SELECT * FROM document_chunks LIMIT 1;
```

## Usage

### 1. Ingest the Corpus

Loads markdown files from `public/corpus/`, chunks them, generates embeddings, and stores in Supabase:

```bash
python -m support_agent.cli ingest
```

Options:
```bash
python -m support_agent.cli ingest --help
# --index-path PATH     Path to corpus index.md (default: public/corpus/index.md)
# --chunk-size INT      Number of words per chunk (default: 400)
# --overlap INT         Number of overlapping words between chunks (default: 50)
# --clear / --no-clear  Clear existing chunks before ingest (default: --clear)
```

### 2. Search the Indexed Corpus

Semantic search against the ingested corpus:

```bash
python -m support_agent.cli search "How do I reset my password?" --top-k 5 --threshold 0.3
```

Options:
```bash
python -m support_agent.cli search --help
# QUERY                 Search query (required)
# --top-k INT           Number of results (default: 5)
# --threshold FLOAT     Similarity threshold 0-1 (default: 0.0)
```

Output includes: chunk ID, source file, title, category, similarity score.

### 3. Run Triage on Tickets

Process a CSV of support tickets through the full pipeline:

```bash
python -m support_agent.cli triage input.csv output.csv
```

**Input CSV format** (required columns):
```csv
Issue,Subject
"User cannot log in","Login issue"
"Billing question",""
```

**Output CSV format** (exact columns, in order):
```csv
status,product_area,response,justification,request_type
replied,screen,"Based on docs...","Doc 1 says...",product_issue
escalated,,Escalated to human,Security policy requires escalation,bug
```

Summary printed to stdout:
```
Processed 16 input tickets
Output: 16 tickets written to output.csv
Replied: 12
Escalated: 4
```

Options:
```bash
python -m support_agent.cli triage --help
# INPUT_CSV             Path to input CSV
# OUTPUT_CSV            Path to output CSV
# --min-chunks INT      Minimum retrieved chunks to call LLM (default: 1)
```

### 4. Run Evaluation

Evaluate the pipeline against golden sample data:

```bash
python -m support_agent.cli evaluate --golden public/samples/sample_support_tickets.csv
```

Output includes:
- Per-ticket comparison (status, product_area, request_type)
- Aggregate metrics: status accuracy, request_type accuracy, product_area accuracy, overall exact-match rate
- Lightweight response checks (replied→non-empty, escalated→mentions escalation)

## Running Tests

```bash
# Full test suite (all mocked, no real API calls)
python -m pytest

# With verbose output
python -m pytest -v

# Specific test modules
python -m pytest tests/test_cli.py
python -m pytest tests/test_api.py
python -m pytest tests/test_deployment.py
```

### Test Categories

| Module | Tests | External Calls |
|--------|-------|----------------|
| `test_scaffold.py` | Project structure | None |
| `test_models.py` | Pydantic models | None |
| `test_csv_io.py` | CSV reading/writing | None |
| `test_corpus.py` | Corpus indexing | None |
| `test_db.py` | Supabase repository | Mocked |
| `test_embeddings.py` | Gemini embeddings | Mocked |
| `test_retrieval.py` | Semantic retrieval | Mocked |
| `test_llm.py` | OpenRouter provider | Mocked |
| `test_policy.py` | Safety policy | None |
| `test_orchestrator.py` | Orchestration pipeline | Mocked |
| `test_ingestion.py` | Corpus ingestion | Mocked |
| `test_cli.py` | CLI commands | Mocked |
| `test_evaluation.py` | Evaluation framework | Mocked |
| `test_api.py` | FastAPI endpoints | Mocked |
| `test_deployment.py` | Docker/Render config | None |

**All 255+ tests use mocks — no real Supabase, Gemini, or OpenRouter calls.**

## FastAPI Server

Run the HTTP API locally:

```bash
# Development (auto-reload)
uvicorn support_agent.api:app --reload --host 0.0.0.0 --port 8000

# Production-like
uvicorn support_agent.api:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns `{"status": "healthy"}` |
| `/triage` | POST | Process single ticket — returns `AgentResult` |
| `/triage/batch` | POST | Process batch (max 100) — returns `List[AgentResult]` |

### Request/Response Examples

**POST /triage**
```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"issue": "Cannot log in", "subject": "Login problem"}'
```

Response:
```json
{
  "status": "replied",
  "product_area": "screen",
  "response": "Based on documentation...",
  "justification": "Doc 1 states...",
  "request_type": "product_issue"
}
```

**POST /triage/batch**
```bash
curl -X POST http://localhost:8000/triage/batch \
  -H "Content-Type: application/json" \
  -d '[{"issue": "Issue 1", "subject": "Sub 1"}, {"issue": "Issue 2", "subject": "Sub 2"}]'
```

Response:
```json
[
  {"status": "replied", "product_area": "screen", ...},
  {"status": "escalated", "product_area": "", ...}
]
```

OpenAPI docs available at: `http://localhost:8000/docs`

## Docker Usage

### Build Image

```bash
docker build -t support-triage-agent .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -e SUPABASE_URL="https://..." \
  -e SUPABASE_KEY="..." \
  -e GEMINI_API_KEY="..." \
  -e OPENROUTER_API_KEY="..." \
  -e OPENROUTER_MODEL="openai/gpt-4o-mini" \
  support-triage-agent
```

The container runs `uvicorn support_agent.api:app --host 0.0.0.0 --port ${PORT:-8000}`

### Environment Variables

All configuration via environment variables — no `.env` file copied into image.

## Render Deployment

### 1. Push to Git

```bash
git add .
git commit -m "Deploy"
git push origin main
```

### 2. Create Web Service on Render

- New → Web Service
- Connect repository
- Environment: Docker
- Dockerfile Path: `./Dockerfile`
- Docker Context: `.`

### 3. Configure Environment Variables

In Render dashboard → Environment, add as **Secret** values:

| Variable | Secret? |
|----------|---------|
| `SUPABASE_URL` | Yes |
| `SUPABASE_KEY` | Yes |
| `GEMINI_API_KEY` | Yes |
| `OPENROUTER_API_KEY` | Yes |
| `OPENROUTER_MODEL` | Yes |
| `EMBEDDING_MODEL` | No (default provided) |
| `EMBEDDING_BATCH_SIZE` | No (default provided) |

### 4. Health Check

Render automatically uses `/health` (configured in `render.yaml`).

### 5. Deploy

Click "Create Web Service" — Render builds from Dockerfile and deploys.

## Environment Variables as Secrets

The following **must** be configured as secrets in production (Render, Docker, CI/CD):

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `OPENROUTER_MODEL`

The following have defaults and are optional:

- `EMBEDDING_MODEL` (default: `models/gemini-embedding-001`)
- `EMBEDDING_BATCH_SIZE` (default: `1`)

## Real End-to-End Smoke Testing (NOT yet performed)

The automated test suite uses **mocks only**. To verify the complete pipeline against real services:

### Prerequisites for Real Test

1. Valid Supabase project with pgvector enabled and migration applied
2. Valid `GEMINI_API_KEY` with embedding model access
3. Valid `OPENROUTER_API_KEY` with model access
3. Ingested corpus (`python -m support_agent.cli ingest`)

### Smoke Test Steps

```bash
# 1. Verify configuration
python -c "from support_agent.cli import _get_embedding_provider, _get_repository, _get_llm_provider; print('Config OK')"

# 2. Ingest corpus (one-time)
python -m support_agent.cli ingest

# 3. Search verification
python -m support_agent.cli search "test query" --top-k 3

# 4. Triage verification (use sample tickets)
python -m support_agent.cli triage public/samples/sample_support_tickets.csv output.csv

# 5. Evaluate against golden data
python -m support_agent.cli evaluate --golden public/samples/sample_support_tickets.csv

# 6. API verification
uvicorn support_agent.api:app --host 0.0.0.0 --port 8000 &
curl http://localhost:8000/health
curl -X POST http://localhost:8000/triage -H "Content-Type: application/json" -d '{"issue": "Test", "subject": "Test"}'
```

**This project has NOT had a complete real end-to-end execution against actual external services.** The above steps document what remains to be performed.

## CSV Format Reference

### Input (triage)

| Column | Required | Description |
|--------|----------|-------------|
| `Issue` | Yes | Full ticket description |
| `Subject` | No | Ticket subject/title |

### Output (triage/evaluate)

| Column | Description |
|--------|-------------|
| `status` | `replied` or `escalated` |
| `product_area` | Product area (e.g., `screen`, `community`) or blank |
| `response` | Generated response text |
| `justification` | Reasoning from retrieved docs |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` |

### Golden/Evaluation (sample_support_tickets.csv)

| Column | Purpose |
|--------|---------|
| `Issue`, `Subject` | Input ticket |
| `Response` | Reference response (not compared exactly) |
| `Product Area` | Expected product_area |
| `Status` | Expected status |
| `Request Type` | Expected request_type |
| `Company` | Metadata (not evaluated) |

## Project Structure

```
support-agent/
├── .env.example          # Environment variable template
├── Dockerfile            # Production container
├── render.yaml           # Render deployment config
├── pyproject.toml        # Project metadata & dependencies
├── plan.md               # Implementation plan
├── README.md             # This file
├── public/
│   ├── corpus/           # Markdown documentation corpus
│   │   ├── index.md
│   │   └── <category>/   # Subdirectories by product area
│   └── samples/
│       ├── sample_support_tickets.csv  # 7 golden tickets
│       └── support_tickets.csv         # 16 target tickets
├── src/support_agent/
│   ├── __init__.py
│   ├── api.py            # FastAPI wrapper
│   ├── cli.py            # Terminal CLI (Typer)
│   ├── corpus.py         # Corpus indexing
│   ├── csv_io.py         # CSV read/write
│   ├── db.py             # Supabase repository
│   ├── embeddings.py     # Gemini embeddings provider
│   ├── evaluation.py     # Golden sample evaluation
│   ├── ingestion.py      # Corpus ingestion pipeline
│   ├── llm.py            # OpenRouter LLM provider
│   ├── models.py         # Pydantic models (Ticket, AgentResult)
│   ├── orchestrator.py   # Triage orchestration
│   ├── policy.py         # Safety/escalation policy
│   └── retrieval.py      # Semantic retrieval
├── supabase/
│   └── migrations/       # pgvector table schema
└── tests/
    ├── test_*.py         # 16 test modules, 255+ tests
    └── __init__.py
```

## Safety & Fail-Closed Behavior

The pipeline enforces escalation (never silently replies) for:

- Empty/short tickets (< 10 chars)
- Prompt injection attempts
- Security/privacy requests (account deletion, payment info, PII)
- Insufficient retrieval evidence (no relevant docs found)
- LLM failures or invalid structured output

## License

Internal challenge project — not for external distribution.
## Plan: Support Triage Agent

Build a challenge-sized Python application that reads support tickets from CSV, retrieves information only from the supplied documentation corpus, uses configurable embeddings with Supabase PostgreSQL/pgvector for semantic RAG, calls a configurable OpenRouter LLM, validates structured results with Pydantic, and writes the required CSV output. The terminal CLI is primary. FastAPI, Docker, and Render are added only after the core CLI works, as a thin deployment wrapper.

**Architecture and scope**

- Python 3.11+, pytest, Pydantic, and a small CLI framework such as Typer or `argparse`.
- Supabase PostgreSQL with the `vector` extension and one simple document-chunks table.
- OpenRouter as the LLM provider, with model and connection settings from environment variables.
- A small configurable embedding provider interface and one implementation.
- Semantic vector retrieval first. Hybrid keyword/vector retrieval and reranking are later improvements, not initial work.
- Application-level safety, risk assessment, escalation, grounding, and output validation.
- FastAPI, Docker, and Render only after the CLI workflow is complete.

**Implementation steps**

1. Inspect `sample_support_tickets.csv`, `support_tickets.csv`, and the documentation corpus. Derive the external CSV schema and expected-output mapping from the sample instead of guessing. Normalize internally to an identifier, subject, description, and optional metadata. Define the exact output fields: `status`, `product_area`, `response`, `justification`, and `request_type`.
2. Scaffold the Python package, configuration, source layout, pytest setup, and README basics. Configure Supabase URL/key, OpenRouter key/model, embedding provider/model, corpus path, retrieval settings, and output paths through environment variables.
3. Add a simple Supabase migration: enable `pgvector`, create a document-chunks table with source metadata, chunk text, optional product area, and `vector(n)` embedding, and add a parameterized similarity-search RPC. Do not add sophisticated migration/versioning systems, hashes, ingestion-run metadata, embedding compatibility systems, or vector-index optimization.
4. Implement corpus ingestion: load Markdown/text files, chunk them deterministically with modest overlap, embed them with the configurable embedding provider, and insert them into Supabase. The `ingest` command may clear and rebuild the table during development. Add `search` to inspect exact excerpts, source IDs, and similarity scores.
5. Implement semantic RAG: embed normalized ticket text, retrieve top relevant chunks from pgvector, apply a simple relevance threshold, and treat insufficient evidence as unsupported/escalated. Keep retrieval behind a small repository/retriever interface for testing.
6. Implement safety and policy in application code. Treat ticket text as untrusted input. Detect invalid/empty or irrelevant content, multiple requests, unsupported questions, sensitive/security/privacy/account issues, high-risk cases, and prompt injection. Use documented request-type precedence: invalid, bug, feature request, then product issue. Determine product area only from the known taxonomy and evidence; unknown areas escalate. Assess urgency/risk internally even though it is not a required output column.
7. Implement one OpenRouter client behind a small LLM interface. Send the ticket and bounded retrieved excerpts with clear separation between system instructions, untrusted ticket text, and corpus evidence. Require structured JSON and parse it with Pydantic. Enforce that factual claims come only from supplied documentation. Use a small retry limit and a deterministic escalation fallback for provider errors or invalid output.
8. Compose the shared pipeline: parse and normalize, apply hard-risk rules, classify, retrieve, assess supportability/risk, escalate or draft, validate, and write. Ensure hard-risk/unsupported cases cannot become normal replies, product areas use the taxonomy, and responses are concise and grounded. Process rows independently.
9. Build the terminal CLI with `ingest`, `search`, `triage`, and `evaluate` commands. Support UTF-8, BOMs, quoted/multiline fields, stable row order, clear malformed-row handling, and the exact required output columns.
10. Test and evaluate with pytest. Mock Supabase, embedding, and OpenRouter calls in normal tests. Cover CSV handling, chunking, policy, prompt injection, risk escalation, unsupported and multiple requests, retrieval thresholds, malformed model output, provider failures, and deterministic fallbacks. Convert sample expected outputs into regression tests; check status/request type exactly, product-area values, and grounding rather than exact generated wording.
11. Add final deployment support only after the CLI passes evaluation. Create a thin FastAPI layer that reuses the same triage service, with simple single-ticket or small-batch request/response models, a health endpoint, and basic errors. Add a straightforward Dockerfile and minimal Render configuration. Do not add authentication, queues, background jobs, request-management infrastructure, or production-grade container hardening.
12. Run the sample evaluator, rebuild the corpus index as needed, process `support_tickets.csv`, verify headers/order/encoding and safety behavior, and document setup, environment variables, migrations, indexing, CLI usage, tests, and deployment.

**Project structure**

- `src/support_agent/cli.py` — CLI commands
- `src/support_agent/config.py` — environment configuration
- `src/support_agent/models.py` — Pydantic models and enums
- `src/support_agent/csv_io.py` — CSV input/output
- `src/support_agent/corpus.py` — document loading and chunking
- `src/support_agent/embeddings.py` — configurable embedding provider
- `src/support_agent/db.py` — Supabase client and document repository
- `src/support_agent/retrieval.py` — query embedding and pgvector retrieval
- `src/support_agent/policy.py` — classification, risk, safety, and escalation
- `src/support_agent/llm.py` — OpenRouter client
- `src/support_agent/prompts.py` — constrained prompts
- `src/support_agent/orchestrator.py` — shared triage workflow
- `src/support_agent/validation.py` — output and grounding validation
- `src/support_agent/api.py` — thin FastAPI layer
- `supabase/migrations/` — simple schema and similarity-search function
- `tests/` — unit, integration, and evaluation tests
- `corpus/` — supplied documentation
- `Dockerfile`, `render.yaml`, `.env.example`, `README.md`

**Commit-by-commit roadmap**

Each feature is implemented, tested, committed, and merged before the next begins:

1. Project scaffold and configuration.
2. Pydantic contracts and CSV input/output.
3. Simple Supabase pgvector schema and similarity RPC.
4. Corpus loading, chunking, embeddings, ingestion, and search.
5. Semantic retrieval and threshold behavior.
6. Safety rules, classification, product areas, risk, and escalation.
7. OpenRouter structured generation and mocked failure handling.
8. End-to-end triage orchestration and output validation.
9. Sample-ticket evaluation and targeted tuning.
10. Target CSV processing and manual safety review.
11. Thin FastAPI wrapper.
12. Docker and Render deployment configuration.
13. Final pytest, static checks if configured, sample evaluation, and deployment verification.

**Required behavior**

- The documentation corpus is the only factual source; no browsing, external knowledge, or prior ticket answers.
- Prompt injection is treated as ticket content, not an instruction.
- Unsupported, unsafe, ambiguous, high-risk, invalid, or insufficiently grounded cases are escalated for Human Review.
- Every normal reply must be supported by retrieved documentation.
- Pydantic validates model and final outputs; application policy outranks model output.
- Provider, database, and parsing failures fail closed to escalation where possible.

**Deliberately excluded**

Content hashing and incremental synchronization; embedding/version compatibility systems; ingestion-run metadata; HNSW/IVFFlat tuning; hybrid retrieval and reranking in the initial version; elaborate provider/plugin frameworks; metrics, tracing, dashboards, and monitoring platforms; complex authentication; queues and background jobs; request-management infrastructure; production-grade Docker hardening; autonomous ticket updates; and external support-system integrations.

**Trade-offs**

The simple rebuildable index is easier to understand and sufficient for a challenge corpus, at the cost of re-embedding all documents when rebuilding. Semantic retrieval is transparent enough for the first version but may miss exact terminology, so hybrid retrieval is a future improvement. OpenRouter and hosted embeddings simplify model experimentation but require API keys, network access, and awareness of privacy/cost implications. Conservative escalation may over-escalate some tickets, but it demonstrates safer grounded behavior than unsupported replies.

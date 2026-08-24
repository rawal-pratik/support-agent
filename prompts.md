I want to build a terminal-based AI support triage agent from scratch.

Act as a senior software architect and create a practical implementation plan for the entire application. Do not write code yet.

The application should process support tickets from CSV files and use ONLY a provided support documentation corpus as its source of truth.

For each ticket, the agent must identify the request type, classify the product area, assess urgency/risk, decide whether to reply or escalate, retrieve relevant support documentation, generate a safe grounded response, and provide a concise justification.

The required output fields are status (replied or escalated), product_area, response, justification, and request_type (product_issue, feature_request, bug, or invalid).

The system must handle irrelevant or misleading content, multiple requests, sensitive or high-risk issues, unsupported questions, and prompt-injection attempts. It must not hallucinate or use knowledge outside the provided support corpus. Cases that cannot be safely answered should be escalated for Human Review.

We will have sample_support_tickets.csv containing example tickets and expected outputs, support_tickets.csv containing tickets that the finished application must process, and a support documentation corpus.

In your plan, determine the recommended architecture and technology stack, how the agent should process a ticket end-to-end, how the support corpus should be used for retrieval and grounding, how classification/risk assessment/escalation should work, how to handle prompt injection and unsupported questions, the proposed project structure, how CSV input/output should work, how we should test and evaluate the agent using the sample tickets, and a phased implementation plan from an empty repository to a completed application.

Also identify any important design decisions, trade-offs, or risks.

Prioritize simplicity, reliability, grounding, and maintainability. Do not over-engineer the solution.

Do not implement anything yet. Give me the plan first, including the recommended order in which we should build the application.













I reviewed your plan and want to make a few architectural changes and additions before we begin implementation.

Keep the overall architecture and phased approach, but make these changes:

Use Supabase PostgreSQL with pgvector instead of SQLite FTS5 or ChromaDB for persistent hosted vector storage.
Add an embedding provider abstraction with a configurable embedding model and a corpus indexing pipeline that stores embeddings in Supabase.
Use OpenRouter as the LLM provider through a provider abstraction, with the model configurable via environment variables so models can be swapped easily.
Add database migrations/schema setup for the Supabase database.
Keep semantic vector retrieval as the initial retrieval approach; add hybrid keyword + vector retrieval and reranking later as an improvement.
Keep the terminal CLI as the primary interface, but add a thin FastAPI layer later so the application can be deployed.
Add Docker and deployment configuration for Render, with Supabase providing persistent storage.
Add structured operational logging/observability and production error handling.
Keep the existing safety, escalation, validation, grounding, and evaluation approach from your plan.

Do not implement anything yet. Update your previous plan to incorporate these changes and additions, while keeping the solution simple and maintainable.
















I want to simplify the current plan before we start implementation.

Your revised plan is technically strong, but it has added several production-level features that are unnecessary for this challenge. We are building a functional challenge assignment, not a production-scale support platform.

Please revise the plan to align with our simpler implementation roadmap.

Keep these architectural decisions:

Python
Supabase PostgreSQL + pgvector
OpenRouter with a configurable LLM model
Configurable embeddings
RAG using the supplied documentation corpus
Application-level safety and escalation rules
Pydantic structured outputs
Terminal CLI as the primary interface
FastAPI + Docker + Render only for the final deployment
pytest for testing and evaluation

Remove or simplify unnecessary additions such as:

content hashing and incremental corpus synchronization
embedding/version compatibility systems
ingestion-run metadata
sophisticated database migration/versioning systems
HNSW/IVFFlat or vector-index optimization
extensive observability, metrics, and monitoring systems
complex API authentication, queues, background jobs, or request-management infrastructure
production-grade Docker hardening
elaborate provider abstractions beyond what is actually needed
hybrid retrieval and reranking from the initial implementation

For the database, a simple Supabase schema and pgvector similarity search are sufficient. For corpus ingestion, simply load, chunk, embed, and store the documents. If we need to rebuild the index during development, we can rebuild it.

For deployment, the FastAPI layer should remain very thin and simply expose the existing agent functionality. We only need enough Docker/Render configuration to deploy and demonstrate the application.

Keep the important functionality around grounding, classification, risk assessment, prompt-injection handling, escalation, structured outputs, retrieval, evaluation, and error handling.

The implementation should follow our original commit-by-commit approach, with each feature developed, tested, committed, and merged before moving to the next feature.

Do not add complexity just because it would be useful in a production system. If a feature is not necessary to satisfy the challenge or demonstrate strong AI engineering practices, leave it out.

I have created a local copy of plan.md
Use that copy from now on.
Update plan.md accordingly. Do not implement any code yet.













Create the initial project scaffold for this Python support AI agent.

The repository currently contains only plan.md. Read plan.md for context, but implement only the initial project foundation described below.

Create a clean Python 3.11+ project using a src layout:

src/support_agent/
tests/

Add:

pyproject.toml
.gitignore
.env.example
README.md
src/support_agent/init.py
tests/init.py

Configure pytest and basic project metadata in pyproject.toml. Add the minimal development dependencies we know we will need initially, including pytest, Pydantic, and Typer. Keep the dependency list minimal; do not add Supabase, OpenRouter, embedding, FastAPI, or other later-stage dependencies yet.

The project should be importable as a Python package and pytest should run successfully even though there are no application tests yet.

Do not implement any application functionality, CLI commands, database code, LLM code, retrieval code, or API code.

After creating the files, show me what you changed and how I can verify that the scaffold works. Do not modify plan.md.













We are now implementing the data contracts and CSV handling feature.

Read plan.md and inspect the current repository. Also inspect the supplied sample_support_tickets.csv and support_tickets.csv before making any assumptions about their schema.

Implement only this feature:

- Create Pydantic models for the internal Ticket and AgentResult.
- AgentResult must contain exactly the required fields: status, product_area, response, justification, and request_type.
- status must support replied and escalated.
- request_type must support product_issue, feature_request, bug, and invalid.
- Derive the actual ticket input fields from the supplied CSV files rather than guessing.
- Implement CSV input that converts the challenge CSV rows into our internal Ticket model.
- Preserve row order and support normal CSV quoting, multiline fields, and UTF-8/UTF-8-BOM.
- Implement CSV output with the required five output fields.

Keep this feature independent of all later functionality. Do not implement Supabase, pgvector, embeddings, OpenRouter, retrieval, safety logic, agent orchestration, CLI commands, FastAPI, or deployment.

Write pytest tests for everything you implement. The tests should verify actual behavior, including the real CSV schema, Pydantic validation, CSV parsing, row-order preservation, and required output columns.

Keep the implementation simple and avoid unnecessary abstractions or dependencies.

First inspect the existing files and propose a concise implementation plan. Do not modify plan.md. Then implement the feature and its tests.

After implementation, run the test suite and report:
1. Files created/modified
2. What was implemented
3. Tests added
4. Test results
5. Any assumptions or issues discovered











We are implementing the support corpus ingestion feature on branch feat/corpus-ingestion.

Read plan.md and inspect the existing project first. Then inspect the actual support documentation corpus and public/corpus/index.md. The corpus contains many Markdown files organized into directories, and index.md is the catalogue of the articles.

The corpus root is public/corpus and the index is public/corpus/index.md. Article paths listed in the index are relative to public/corpus. The sample CSV files under public/samples are unrelated to this feature.

Implement only the corpus loading and chunking layer.

The module should:
- Read public/corpus/index.md and discover the articles referenced by it.
- Resolve the referenced article paths relative to public/corpus.
- Load the referenced Markdown files.
- Preserve useful metadata such as source path, article title, and category/path information.
- Represent documents/chunks using a small structured model.
- Split articles into deterministic text chunks with modest overlap.
- Preserve source metadata on every chunk so the originating article can be identified later during retrieval.
- Detect missing files referenced by the index and fail clearly.
- Avoid ingesting unrelated Markdown files that are not referenced by the index.

Keep the implementation simple and challenge-focused.

Do not implement embeddings, Supabase, pgvector, retrieval, OpenRouter, RAG, safety policy, CLI commands, FastAPI, or deployment.

Do not add content hashing, incremental synchronization, document versioning, ingestion metadata, or other production-scale corpus management.

Write pytest tests for the functionality you implement. Use the actual supplied corpus/index where appropriate and small temporary fixtures for chunking and error cases.

Do not modify plan.md.

First inspect the corpus structure and briefly explain your implementation approach. Then implement the feature and its tests.

After implementation, run the full test suite and report:
1. Files created/modified
2. Corpus structure discovered
3. Chunking approach used
4. Tests added
5. Test results
6. Any assumptions or issues discovered













We are implementing the Supabase vector-store foundation on branch feat/supabase-vector-store.

Read plan.md and inspect the existing project and the corpus ingestion implementation first.

Implement only the persistent document-chunk storage layer using Supabase PostgreSQL and pgvector.

Requirements:

- Add a simple Supabase migration under supabase/migrations.
- Enable the pgvector extension.
- Create a document_chunks table containing an id, source_path, title, category, chunk_text, and vector embedding column.
- Add a simple parameterized similarity-search function/RPC that can later accept a query embedding and return the most similar chunks.
- Add a small Python database/repository module for interacting with Supabase.
- Keep Supabase configuration in environment variables and update .env.example with the required placeholders.
- Keep the repository interface small and focused on storing chunks and performing similarity search.
- Do not generate embeddings in this commit. The embedding provider will be implemented in the next feature.
- Do not implement retrieval logic, OpenRouter, RAG, safety policy, CLI, FastAPI, or deployment.

Keep the database schema and repository simple and challenge-focused.

Do not add authentication, ingestion tracking, content hashes, embedding version management, complex migration systems, vector-index tuning, queues, or other production infrastructure.

Write pytest tests for the database/repository behavior using mocks rather than requiring a live Supabase connection for normal tests.

The tests should verify:
- repository initialization/configuration
- chunk insertion behavior
- clearing/rebuilding behavior if implemented
- similarity-search request/response handling
- handling of database errors

First inspect the existing project and briefly explain the proposed schema and repository design. Then implement the feature and tests.

Do not modify plan.md.

Run the full test suite after implementation and report the files created/modified and test results.














We are implementing the embedding and corpus ingestion feature on branch feat/embeddings.

Read plan.md and inspect the existing corpus, models, corpus loader, Supabase repository, and database migration before making changes.

Implement only the embedding provider and support-corpus ingestion pipeline.

Requirements:

- Add a small embedding provider interface/module.
- Choose an embedding model/provider that produces exactly 1536-dimensional embeddings, matching the existing Supabase vector(1536) schema.
- Keep the embedding provider configurable through environment variables where appropriate.
- Implement batch embedding for document chunks.
- Implement corpus ingestion that loads the existing DocumentChunks, generates embeddings, and stores the chunks and embeddings in Supabase.
- Reuse the existing corpus loader and Supabase repository rather than duplicating their logic.
- Make ingestion deterministic and simple. It is acceptable for development to clear and rebuild the document_chunks table.
- Ensure failures from the embedding provider or database are surfaced clearly.
- Do not generate ticket embeddings yet.

Write pytest tests for the embedding provider and ingestion behavior. Mock external embedding and Supabase calls in normal tests; tests must not require real API calls.

Tests should verify:
- embedding output shape/dimension validation
- batching behavior
- successful corpus ingestion
- chunk/embedding pairing
- provider failures
- database failures
- empty corpus behavior

Do not implement semantic retrieval, OpenRouter, RAG, ticket classification, safety policy, CLI commands, FastAPI, or deployment.

Do not add embedding version management, incremental synchronization, content hashes, ingestion-run tracking, complex provider/plugin systems, or other production infrastructure.

Do not modify plan.md.

First inspect the existing implementation and briefly explain which embedding provider/model you will use and why it is compatible with vector(1536). Then implement the feature and its tests.

Run the full test suite and report:
1. Files created/modified
2. Embedding provider/model selected
3. Embedding dimension
4. Ingestion flow
5. Tests added
6. Test results
7. Any assumptions or issues discovered











Make only the following targeted corrections before finishing Commit 10.

1. Fix tests/test_cli.py::test_triage_invokes_orchestrator_once_per_ticket. The current test checks mock_llm.generate.call_count, which does not actually verify that TriageOrchestrator.process_ticket is called once per ticket. Mock TriageOrchestrator directly and assert process_ticket is called exactly once for each input ticket.

2. Remove obviously unused imports from cli.py if they are truly unused, such as sys and evaluate_policy. Do not otherwise refactor the CLI.

3. Run the full pytest suite.

Do not make any other functional changes and do not commit.












Make only the following targeted corrections before finishing Commit 10.

1. Fix tests/test_cli.py::test_triage_invokes_orchestrator_once_per_ticket. The current test checks mock_llm.generate.call_count, which does not actually verify that TriageOrchestrator.process_ticket is called once per ticket. Mock TriageOrchestrator directly and assert process_ticket is called exactly once for each input ticket.

2. Remove obviously unused imports from cli.py if they are truly unused, such as sys and evaluate_policy. Do not otherwise refactor the CLI.

3. Run the full pytest suite.

Do not make any other functional changes and do not commit.











The implementation looks good. Before committing, make one small cleanup:

Remove the unused --run / -r option from the evaluate CLI command and its parameter, since the evaluator intentionally runs the orchestrator directly against the golden CSV and does not support evaluating a pre-generated run CSV.

Do not add run CSV functionality.

After removing it:
1. Update any affected tests.
2. Run the full pytest suite.
3. Verify the evaluate command still works with:
   support-agent evaluate --golden public/samples/sample_support_tickets.csv
4. Do not make any other architectural changes.
5. Do not commit yet.

Report the final test count and git status.














Implement Commit 12: target CSV processing and manual safety review.

Read plan.md and inspect the existing project before making changes, especially csv_io.py, models.py, policy.py, retrieval.py, llm.py, orchestrator.py, cli.py, and evaluation.py.

The target dataset is public/samples/support_tickets.csv.

I have inspected this file previously. Its schema is exactly:

- Issue
- Subject

It contains 16 target support tickets.

Unlike sample_support_tickets.csv, this file contains NO expected output columns. Do not treat it as a golden/evaluation dataset and do not invent expected answers for it.

The goal of this commit is to process the actual target CSV through the completed triage pipeline and produce the required output CSV.

Required output columns, in exactly this order:

status,product_area,response,justification,request_type

Requirements:

1. Add or update the CLI workflow so the target CSV can be processed directly using the existing `triage` command.

2. Use the existing CSV reader/writer and TriageOrchestrator.
   Do not duplicate policy, retrieval, embedding, LLM, or orchestration logic.

3. Process all 16 target tickets independently and preserve their original row order.

4. Ensure every output row conforms to the existing AgentResult Pydantic contract.

5. Preserve the existing fail-closed behavior:
   - invalid/unsafe tickets escalate
   - mandatory policy escalations cannot become normal replies
   - insufficient retrieval evidence escalates
   - retrieval failures escalate
   - LLM failures/invalid outputs escalate

6. Do not add new policy rules in this commit unless required to fix an actual target-ticket behavior that is clearly mandated by the existing plan.

7. Do not modify the Supabase schema, embedding implementation, retrieval algorithm, or OpenRouter implementation.

8. Add robust handling for the target CSV:
   - UTF-8
   - UTF-8 BOM
   - quoted fields
   - multiline fields
   - blank subjects
   - malformed rows
   - missing required input headers
   - output encoding and exact output headers

9. The CLI should provide a clear summary after processing:
   - number of input tickets
   - number of output tickets
   - number replied
   - number escalated
   - output path

10. Do not print API keys, Supabase credentials, or other secrets.

Testing:

Add/update tests covering:
- processing all 16 target rows
- preserving row order
- exact required output headers and order
- blank Subject handling
- malformed target CSV behavior
- missing input headers
- each ticket being passed independently to the orchestrator
- one failed ticket not preventing other tickets from being processed, where the existing orchestration contract permits this
- output rows always conforming to AgentResult
- CLI summary output
- CLI failure behavior
- no external API calls in unit tests

Important:
- Do NOT add expected outputs for the 16 target tickets.
- Do NOT modify sample_support_tickets.csv.
- Do NOT modify support_tickets.csv.
- Do NOT create an LLM judge.
- Do NOT add evaluation metrics for the target tickets.
- Do NOT add FastAPI.
- Do NOT add Docker.
- Do NOT add Render configuration.
- Do NOT modify plan.md.

After implementation:

1. Run the complete pytest suite.
2. If the environment has the required real credentials/configuration, run the actual target dataset through the CLI.
3. If real credentials are unavailable, do NOT weaken tests or add fake credentials. Report that the real target run could not be performed.
4. If the real target run succeeds, inspect the generated output for:
   - exact headers
   - exactly 16 rows
   - row-order preservation
   - status distribution
   - empty/unknown product areas
   - escalation behavior
   - obviously unsupported responses
   - obvious prompt-injection attempts
   - sensitive/security/account requests
5. Do not commit.

Report:
- files created/modified
- target CSV schema
- number of target tickets processed
- output path if a real run was performed
- replied/escalated counts
- test count
- full pytest result
- manual safety observations
- any issues discovered
- any assumptions made










Implement Commit 13: thin FastAPI wrapper.

Read plan.md and inspect the existing project before making changes, especially models.py, csv_io.py, policy.py, retrieval.py, llm.py, orchestrator.py, cli.py, and evaluation.py.

The goal is to expose the existing triage functionality through a small FastAPI HTTP API.

Architecture:

HTTP request
    ↓
FastAPI
    ↓
existing TriageOrchestrator
    ↓
existing policy/retrieval/LLM pipeline

The FastAPI layer must remain thin. Do not duplicate business logic from the CLI, orchestrator, policy, retrieval, or LLM modules.

Add:

src/support_agent/api.py

Endpoints:

1. GET /health
   - Return a simple JSON response indicating the service is healthy.
   - Do not require external API calls.

2. POST /triage
   - Accept a single support ticket.
   - Request fields should correspond to the existing Ticket model:
     - issue
     - subject
   - Return the existing AgentResult structure.
   - Reuse the existing TriageOrchestrator.
   - Do not implement classification, retrieval, safety, prompting, or response generation inside the API layer.

3. POST /triage/batch
   - Accept a small list of support tickets.
   - Process each ticket through the existing TriageOrchestrator.
   - Preserve input order.
   - Return the corresponding AgentResult objects in the same order.
   - Keep this synchronous/simple; do not add queues or background jobs.

Configuration/dependency construction:

- Reuse the existing environment-based configuration.
- Reuse GeminiEmbeddingProvider, DocumentChunkRepository, SemanticRetriever, and OpenRouterProvider through the same patterns already used by the CLI.
- Keep dependency construction isolated in small helper functions if necessary.
- Do not duplicate configuration logic unnecessarily.
- Do not expose secrets in API responses or errors.

Error behavior:

- Invalid request bodies should produce normal FastAPI/Pydantic validation errors.
- Missing provider configuration should result in a clear server-side configuration error.
- Retrieval/provider/orchestration failures should fail closed according to the existing orchestrator behavior where possible.
- Do not invent a new safety or escalation policy in the API layer.
- Avoid exposing API keys, Supabase credentials, or internal secrets in errors.

Response models:

- Reuse the existing Ticket and AgentResult Pydantic models where appropriate.
- Do not create duplicate versions of these models.
- Ensure OpenAPI documentation exposes the request/response schemas correctly.

Testing:

Add pytest tests for the API using FastAPI's TestClient or an equivalent supported test client.

Tests must mock the underlying:
- TriageOrchestrator
- retrieval
- LLM
- embeddings
- Supabase

No real external API calls should occur in tests.

Cover at minimum:

1. GET /health returns 200 and the expected JSON structure.

2. POST /triage:
   - valid ticket returns 200
   - returned body conforms to AgentResult
   - orchestrator is called exactly once
   - the Ticket passed to the orchestrator contains the supplied issue and subject

3. POST /triage validation:
   - missing issue is rejected
   - malformed request body is rejected

4. POST /triage:
   - orchestrator/provider failure returns a clear appropriate HTTP error
   - secrets are not exposed

5. POST /triage/batch:
   - valid batch returns the correct number of results
   - orchestrator is called once per ticket
   - input/output order is preserved
   - empty batch is handled explicitly
   - one ticket's processing behavior does not silently reorder other tickets

6. API response schema:
   - fields match the existing AgentResult contract
   - no extra undocumented business fields are introduced

7. Ensure /docs or OpenAPI generation works without requiring external credentials.

Do not:
- modify the Supabase schema
- modify embeddings
- modify retrieval algorithms
- modify policy rules
- modify OpenRouter behavior
- add authentication
- add queues
- add background jobs
- add request-management infrastructure
- add streaming
- add WebSockets
- add rate limiting
- add production middleware
- add Docker
- add Render configuration
- modify plan.md

Keep the implementation challenge-sized and thin.

Run the complete pytest suite after implementation.

Do not commit.

Report:
1. files created/modified
2. endpoints added
3. request/response schemas
4. how the API reuses TriageOrchestrator
5. tests added
6. full pytest result
7. any assumptions or issues









Implement Docker and Render deployment configuration.

Read plan.md and inspect the existing project before making changes, especially api.py, cli.py, orchestrator.py, pyproject.toml, and the existing environment configuration.

The application already has a working thin FastAPI API. This commit should only package the existing application for deployment.

Add:

1. Dockerfile
2. render.yaml
3. Any minimal deployment-related configuration required to run the existing FastAPI application in production.

Docker requirements:

- Use a suitable lightweight Python base image compatible with the project's requires-python >=3.11.
- Install the project and its dependencies from pyproject.toml.
- Do not copy local .venv into the image.
- Do not include pytest/dev dependencies unnecessarily.
- Copy only the application files and required runtime assets.
- The supplied corpus is required by the ingestion/retrieval workflow where applicable, so ensure the runtime image contains whatever corpus files the existing application actually requires.
- Do not copy secrets or .env files into the image.
- Do not hardcode API keys.
- Expose the appropriate HTTP port.
- Run the existing FastAPI app with Uvicorn.
- Make the container listen on 0.0.0.0.
- The application must obtain configuration from environment variables at runtime.

Render requirements:

- Add a minimal render.yaml describing one web service.
- Use the Dockerfile for deployment.
- Configure the service to use Render's PORT environment variable appropriately.
- Do not hardcode a specific production port if Render expects PORT to be provided dynamically.
- Do not put API keys or secret values in render.yaml.
- Document required environment variables as Render configuration/secrets where appropriate, but do not provide actual values.
- Use the existing /health endpoint for the Render health check if supported by the chosen Render configuration.

Important architecture constraints:

- Do not modify the FastAPI API behavior.
- Do not modify the orchestrator.
- Do not modify policy.
- Do not modify retrieval.
- Do not modify embeddings.
- Do not modify the Supabase schema.
- Do not modify OpenRouter behavior.
- Do not add authentication.
- Do not add background workers.
- Do not add queues.
- Do not add Redis.
- Do not add Nginx.
- Do not add monitoring/tracing infrastructure.
- Do not add production-grade container hardening beyond what is reasonable for this challenge.
- Do not modify plan.md.

Testing:

Add deployment-focused tests only where they provide real value. Do not create fake tests that merely assert that strings exist.

At minimum verify:

1. Dockerfile exists.
2. Dockerfile starts the FastAPI application through Uvicorn.
3. Uvicorn binds to 0.0.0.0.
4. Docker configuration does not contain obvious API keys/secrets.
5. render.yaml exists and describes a web service.
6. Render configuration references the Dockerfile.
7. Render configuration does not contain secrets.
8. Existing /health endpoint remains usable.
9. The existing FastAPI tests still pass.

If Docker is available in the environment, perform a Docker build using the actual Dockerfile.

If Docker is unavailable, do not install or work around it. Report that the build could not be executed.

Do not perform a real Render deployment.

Run the complete pytest suite.

Do not commit.

Report:
1. files created/modified
2. Docker runtime command
3. Render service configuration
4. required environment variables
5. whether a Docker build was actually performed
6. full pytest result
7. any deployment assumptions or issues
8. git status








Implement Commit 15: final documentation, cleanup, and verification.

Read plan.md and inspect the entire repository before making changes. This is the final project commit. Do not add new architecture or features.

The project should now contain:
- CLI commands: ingest, search, triage, evaluate
- Supabase pgvector storage
- Gemini embeddings
- OpenRouter LLM
- safety/policy layer
- semantic retrieval
- triage orchestration
- FastAPI API
- Dockerfile
- Render configuration
- tests

Your job is to finalize the repository.

1. Review README.md and make it a complete setup and usage guide.

Document:
- what the project does
- architecture at a high level
- prerequisites
- Python version
- installation
- virtual environment setup
- environment variables
- how to configure Supabase
- how to apply the Supabase migration
- how to ingest the corpus
- how to search the indexed corpus
- how to run triage
- how to run evaluation
- expected CSV input/output format
- how to run pytest
- how to run the FastAPI server locally
- Docker usage
- Render deployment steps
- which environment variables must be configured as secrets
- that real API calls are NOT made by the normal pytest suite
- how to perform the real end-to-end smoke test

Do not put any real API keys or secrets in README.md.

2. Review .env.example and make sure it documents every environment variable actually required by the current implementation. Keep it secret-free.

3. Review the repository for accidental files, debug code, stale comments, TODOs, unused imports, generated artifacts, local environment files, or unrelated changes.

Do NOT delete anything merely because it looks unfamiliar. Only remove something if it is clearly an accidental/generated/local artifact and is not required by the project.

4. Review git status and make sure no real secrets are present in tracked or staged files.

5. Run the complete pytest suite.

6. Run relevant CLI help commands to verify:
   python -m support_agent.cli --help
   python -m support_agent.cli ingest --help
   python -m support_agent.cli search --help
   python -m support_agent.cli triage --help
   python -m support_agent.cli evaluate --help

7. Do NOT perform real API calls during pytest.

8. Do NOT deploy to Render.

9. Do NOT modify the core architecture, policy behavior, retrieval behavior, embeddings, LLM client, orchestrator, API behavior, or database schema unless you find an actual correctness bug that prevents the documented workflow from functioning. If you find such a bug, report it before making a larger change.

10. Do not commit.

11. At the end, report:
- files changed
- README sections added/updated
- environment variables documented
- full pytest result
- CLI help verification
- any issues found
- git status

Important:
The project has NOT yet had a complete real end-to-end execution against the actual external services. Do not claim that it has.

The final documentation must clearly distinguish:
A. automated mocked/unit tests
B. real end-to-end smoke testing that still needs to be performed











Before Commit 15, make these final documentation checks only:

1. Inspect db.py and verify what level/type of Supabase key the application actually requires. Update README.md accordingly. Do not recommend a service-role key unless the implementation genuinely requires it.
2. Verify the README terminology for chunk-size/overlap matches the actual CLI implementation (words vs tokens).
3. Verify all documented environment variable names exactly match the implementation.
4. Do not change application code or architecture.
5. Run the full pytest suite again.
6. Do not commit.
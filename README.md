# Support Triage Agent

A Python 3.11+ terminal-based AI support triage agent.

The repository is currently at the initial project-scaffold stage. Application functionality will be added incrementally according to [plan.md](plan.md).

## Setup

Create and activate a virtual environment, then install the project with its development dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Verify the scaffold

Run the test suite:

```powershell
python -m pytest
```

There are no application tests yet, so pytest should complete successfully with no tests collected.

## Project layout

```text
src/support_agent/  Python package
tests/               Test package
plan.md             Implementation plan
```

Future configuration values are listed in `.env.example`. Later stages will add Supabase, OpenRouter, embeddings, retrieval, and deployment dependencies.

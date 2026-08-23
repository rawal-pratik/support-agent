"""FastAPI wrapper for the support triage agent."""

from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from .models import AgentResult, Ticket
from .orchestrator import TriageOrchestrator
from .retrieval import SemanticRetriever
from .embeddings import GeminiEmbeddingProvider
from .db import DocumentChunkRepository
from .llm import OpenRouterProvider

load_dotenv()

# Global orchestrator instance (lazily initialized)
_orchestrator: TriageOrchestrator | None = None


def _build_orchestrator() -> TriageOrchestrator:
    """Build the TriageOrchestrator with environment-based dependencies."""
    retriever = SemanticRetriever(
        embedding_provider=GeminiEmbeddingProvider.from_env(),
        repository=DocumentChunkRepository.from_env(),
        top_k=5,
        similarity_threshold=0.0,
    )
    llm_provider = OpenRouterProvider.from_env()
    return TriageOrchestrator(
        retriever=retriever,
        llm_provider=llm_provider,
        min_retrieved_chunks=1,
    )


def _get_orchestrator() -> TriageOrchestrator:
    """Get the orchestrator, initializing lazily on first request."""
    global _orchestrator
    if _orchestrator is None:
        try:
            _orchestrator = _build_orchestrator()
        except ValueError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Service unavailable: configuration error - {e}"
            ) from e
    return _orchestrator


app = FastAPI(
    title="Support Triage Agent API",
    description="HTTP API for processing support tickets through the triage pipeline",
    version="0.1.0",
)


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint - returns simple JSON status."""
    return {"status": "healthy"}


@app.post("/triage", response_model=AgentResult, tags=["triage"])
async def triage_single(ticket: Ticket):
    """
    Process a single support ticket through the triage pipeline.

    Args:
        ticket: Support ticket with issue and optional subject

    Returns:
        AgentResult with status, product_area, response, justification, request_type
    """
    orchestrator = _get_orchestrator()
    try:
        result = orchestrator.process_ticket(ticket)
        return result
    except HTTPException:
        raise
    except Exception as e:
        # Fail closed - don't expose internal details
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing ticket"
        ) from e


@app.post("/triage/batch", response_model=List[AgentResult], tags=["triage"])
async def triage_batch(tickets: List[Ticket]):
    """
    Process a batch of support tickets through the triage pipeline.

    Args:
        tickets: List of support tickets (max 100)

    Returns:
        List of AgentResult objects in the same order as input
    """
    if not tickets:
        raise HTTPException(status_code=400, detail="Batch cannot be empty")

    if len(tickets) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 100")

    orchestrator = _get_orchestrator()
    results: List[AgentResult] = []

    try:
        for ticket in tickets:
            result = orchestrator.process_ticket(ticket)
            results.append(result)
        return results
    except HTTPException:
        raise
    except Exception as e:
        # Fail closed - don't expose internal details
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing batch"
        ) from e
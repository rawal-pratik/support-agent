"""Typer CLI for ingestion, search, triage, and evaluation."""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import typer

from .csv_io import read_tickets, write_results
from .db import DocumentChunkRepository
from .embeddings import GeminiEmbeddingProvider
from .evaluation import format_report, run_evaluation
from .ingestion import ingest_corpus
from .llm import OpenRouterProvider
from .models import Ticket
from .orchestrator import TriageOrchestrator
from .retrieval import SemanticRetriever

load_dotenv()
app = typer.Typer(name="support-agent", help="Support triage agent CLI", no_args_is_help=True, add_completion=False)


def _services(top_k=5, candidate_k=30, threshold=0.0, max_chunks_per_source=2):
    if candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")
    return (
        SemanticRetriever(
            embedding_provider=GeminiEmbeddingProvider.from_env(),
            repository=DocumentChunkRepository.from_env(),
            top_k=top_k,
            candidate_k=candidate_k,
            similarity_threshold=threshold,
            max_chunks_per_source=max_chunks_per_source,
        ),
        OpenRouterProvider.from_env(),
    )


@app.command()
def ingest(
    index_path: Path = typer.Option(
        "public/corpus/index.md",
        "--index-path",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    chunk_size: int = typer.Option(
        400,
        "--chunk-size",
        min=1,
    ),
    overlap: int = typer.Option(
        50,
        "--overlap",
        min=0,
    ),
    clear: bool = typer.Option(
        True,
        "--clear/--no-clear",
    ),
):
    try:
        result = ingest_corpus(
            index_path=index_path,
            repository=DocumentChunkRepository.from_env(),
            embedding_provider=GeminiEmbeddingProvider.from_env(),
            chunk_size=chunk_size,
            overlap=overlap,
            clear_existing=clear,
        )

        typer.echo(f"Documents processed: {result.document_count}")
        typer.echo(f"Chunks created: {result.chunk_count}")
        typer.echo(f"Chunks inserted: {result.inserted_count}")
        typer.echo(f"Chunks skipped: {result.skipped_count}")

    except Exception as exc:
        typer.echo(f"Ingestion failed: {exc}", err=True)
        raise typer.Exit(1)

@app.command()
def search(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1),
    candidate_k: int = typer.Option(30, "--candidate-k", min=1),
    threshold: float = typer.Option(0.0, "--threshold", "-t", min=0.0, max=1.0),
    max_chunks_per_source: int = typer.Option(2, "--max-chunks-per-source", min=1),
):
    if not query.strip():
        raise typer.BadParameter("query cannot be empty")
    try:
        retriever = SemanticRetriever(
            GeminiEmbeddingProvider.from_env(),
            DocumentChunkRepository.from_env(),
            top_k=top_k,
            candidate_k=candidate_k,
            similarity_threshold=threshold,
            max_chunks_per_source=max_chunks_per_source,
        )
        chunks = retriever.retrieve(Ticket(subject="", issue=query))
        if not chunks:
            typer.echo("No results found")
            return
        for i, c in enumerate(chunks, 1):
            typer.echo(
                f"\n--- Result {i} ---\n"
                f"Chunk ID: {c.chunk_id}\n"
                f"Source: {c.source_path}\n"
                f"Title: {c.title}\n"
                f"Category: {c.category}\n"
                f"Similarity: {c.similarity:.4f}\n"
                f"Text: {c.text[:500]}{'...' if len(c.text) > 500 else ''}"
            )
    except Exception as exc:
        typer.echo(f"Search failed: {exc}", err=True)
        raise typer.Exit(1)


@app.command()
def triage(
    input_csv: Path = typer.Argument(..., exists=True),
    output_csv: Path = typer.Argument(...),
    min_chunks: int = typer.Option(1, "--min-chunks", min=1),
    top_k: int = typer.Option(5, "--top-k", min=1),
    candidate_k: int = typer.Option(30, "--candidate-k", min=1),
    threshold: float = typer.Option(0.0, "--threshold", min=0.0, max=1.0),
    max_chunks_per_source: int = typer.Option(2, "--max-chunks-per-source", min=1),
):
    if candidate_k < top_k:
        raise typer.BadParameter("candidate_k must be greater than or equal to top_k")

    try:
        tickets = read_tickets(input_csv)
        retriever, llm = _services(top_k, candidate_k, threshold, max_chunks_per_source)
        agent = TriageOrchestrator(retriever, llm, min_chunks)
        results = [agent.process_ticket(ticket) for ticket in tickets]
        write_results(output_csv, results)

        replied = sum(r.status == "replied" for r in results)
        typer.echo(f"Processed {len(results)} input tickets")
        typer.echo(f"Output: {len(results)} tickets written to {output_csv}")
        typer.echo(f"Replied: {replied}")
        typer.echo(f"Escalated: {len(results) - replied}")
    except Exception as exc:
        typer.echo(f"Triage failed: {exc}", err=True)
        raise typer.Exit(1)


@app.command()
def evaluate(
    golden_csv: Path = typer.Option(..., "--golden", "-g", exists=True),
    top_k: int = typer.Option(5, "--top-k", min=1),
    candidate_k: int = typer.Option(30, "--candidate-k", min=1),
    threshold: float = typer.Option(0.0, "--threshold", min=0.0, max=1.0),
    max_chunks_per_source: int = typer.Option(2, "--max-chunks-per-source", min=1),
):
    try:
        retriever, llm = _services(top_k, candidate_k, threshold, max_chunks_per_source)
        report = run_evaluation(golden_csv, TriageOrchestrator(retriever, llm))
        typer.echo(format_report(report))
    except Exception as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(1)


@app.callback()
def main(version: bool = typer.Option(False, "--version", "-v")):
    if version:
        typer.echo("support-agent 0.1.0")
        raise typer.Exit()


if __name__ == "__main__":
    app()
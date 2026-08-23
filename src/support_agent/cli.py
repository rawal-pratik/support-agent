"""Terminal CLI for the support triage agent."""

from pathlib import Path
from typing import Optional

import typer
from typer import Context

from .models import AgentResult, Ticket
from .orchestrator import TriageOrchestrator
from .retrieval import Retriever, RetrievedChunk, SemanticRetriever
from .csv_io import read_tickets, write_results
from .ingestion import ingest_corpus, IngestionResult
from .embeddings import EmbeddingProvider, GeminiEmbeddingProvider
from .db import DocumentChunkRepository
from .llm import LLMProvider, OpenRouterProvider

app = typer.Typer(
    name="support-agent",
    help="Support triage agent CLI",
    no_args_is_help=True,
    add_completion=False,
)


def _get_embedding_provider() -> EmbeddingProvider:
    """Get embedding provider from environment."""
    return GeminiEmbeddingProvider.from_env()


def _get_repository() -> DocumentChunkRepository:
    """Get Supabase repository from environment."""
    return DocumentChunkRepository.from_env()


def _get_llm_provider() -> LLMProvider:
    """Get LLM provider from environment."""
    return OpenRouterProvider.from_env()


@app.command()
def ingest(
    ctx: Context,
    index_path: Path = typer.Option(
        "public/corpus/index.md",
        "--index-path",
        "-i",
        help="Path to corpus index.md file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    chunk_size: int = typer.Option(
        400, "--chunk-size", help="Number of words per chunk"
    ),
    overlap: int = typer.Option(
        50, "--overlap", help="Number of overlapping words between chunks"
    ),
    clear: bool = typer.Option(
        True, "--clear/--no-clear", help="Clear existing chunks before ingestion"
    ),
) -> None:
    """
    Load, chunk, embed, and store the documentation corpus in Supabase.
    """
    try:
        embedding_provider = _get_embedding_provider()
        repository = _get_repository()

        result: IngestionResult = ingest_corpus(
            index_path=index_path,
            repository=repository,
            embedding_provider=embedding_provider,
            chunk_size=chunk_size,
            overlap=overlap,
            clear_existing=clear,
        )

        typer.echo(f"Documents processed: {result.document_count}")
        typer.echo(f"Chunks created: {result.chunk_count}")
        typer.echo(f"Chunks inserted: {result.inserted_count}")

    except ValueError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Ingestion failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def search(
    ctx: Context,
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Number of results to return", min=1
    ),
    threshold: float = typer.Option(
        0.0, "--threshold", "-t", help="Minimum similarity threshold", min=0.0, max=1.0
    ),
) -> None:
    """
    Search the documentation corpus for relevant chunks.
    """
    if not query.strip():
        typer.echo("Query cannot be empty", err=True)
        raise typer.Exit(code=1)

    try:
        embedding_provider = _get_embedding_provider()
        repository = _get_repository()

        retriever: Retriever = SemanticRetriever(
            embedding_provider=embedding_provider,
            repository=repository,
            top_k=top_k,
            similarity_threshold=threshold,
        )

        # Create a dummy ticket for the query
        ticket = Ticket(issue=query, subject="")
        chunks = retriever.retrieve(ticket)

        if not chunks:
            typer.echo("No results found")
            return

        for i, chunk in enumerate(chunks, 1):
            typer.echo(f"\n--- Result {i} ---")
            typer.echo(f"Chunk ID: {chunk.chunk_id}")
            typer.echo(f"Source: {chunk.source_path}")
            typer.echo(f"Title: {chunk.title}")
            typer.echo(f"Category: {chunk.category}")
            typer.echo(f"Similarity: {chunk.similarity:.4f}")
            typer.echo(f"Text: {chunk.text[:500]}")
            if len(chunk.text) > 500:
                typer.echo("... (truncated)")

    except ValueError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Search failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def triage(
    ctx: Context,
    input_csv: Path = typer.Argument(
        ..., help="Path to input CSV file with tickets", exists=True, file_okay=True
    ),
    output_csv: Path = typer.Argument(
        ..., help="Path to output CSV file for results"
    ),
    min_chunks: int = typer.Option(
        1, "--min-chunks", help="Minimum retrieved chunks required to call LLM", min=1
    ),
) -> None:
    """
    Process support tickets from CSV and write triage results.
    """
    if not input_csv.exists():
        typer.echo(f"Input file not found: {input_csv}", err=True)
        raise typer.Exit(code=1)

    try:
        tickets = read_tickets(input_csv)
    except ValueError as e:
        typer.echo(f"Invalid input CSV: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Failed to read input CSV: {e}", err=True)
        raise typer.Exit(code=1)

    if not tickets:
        typer.echo("No tickets found in input CSV", err=True)
        raise typer.Exit(code=1)

    try:
        retriever: Retriever = SemanticRetriever(
            embedding_provider=_get_embedding_provider(),
            repository=_get_repository(),
            top_k=5,
            similarity_threshold=0.0,
        )
        llm_provider: LLMProvider = _get_llm_provider()
    except ValueError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Failed to initialize services: {e}", err=True)
        raise typer.Exit(code=1)

    orchestrator = TriageOrchestrator(
        retriever=retriever,
        llm_provider=llm_provider,
        min_retrieved_chunks=min_chunks,
    )

    results: list[AgentResult] = []
    for i, ticket in enumerate(tickets, 1):
        try:
            result = orchestrator.process_ticket(ticket)
            results.append(result)
        except Exception as e:
            typer.echo(f"Error processing ticket {i}: {e}", err=True)
            raise typer.Exit(code=1)

    try:
        write_results(output_csv, results)
        typer.echo(f"Processed {len(results)} tickets. Results written to {output_csv}")
    except Exception as e:
        typer.echo(f"Failed to write output CSV: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    ctx: Context,
    golden_csv: Optional[Path] = typer.Option(
        None, "--golden", "-g", help="Path to golden/evaluation CSV"
    ),
    run_csv: Optional[Path] = typer.Option(
        None, "--run", "-r", help="Path to run results CSV"
    ),
) -> None:
    """
    Evaluate triage results against golden data.

    [Not yet implemented — evaluation framework coming in Commit 11]
    """
    typer.echo(
        "Evaluation command is not yet implemented. "
        "The evaluation framework will be added in a future commit.",
        err=True,
    )
    raise typer.Exit(code=0)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo("support-agent 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    ctx: Context,
    version: bool = typer.Option(
        False, "--version", "-v", callback=_version_callback, is_eager=True, help="Show version"
    ),
) -> None:
    """
    Support triage agent — terminal CLI for processing support tickets.
    """
    pass


if __name__ == "__main__":
    app()
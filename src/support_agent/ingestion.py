"""Corpus ingestion: load, chunk, embed, and store documents."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .corpus import chunk_documents, load_documents
from .db import DocumentChunkRepository
from .embeddings import EmbeddingProvider


class IngestionResult(BaseModel):
    """Result of a corpus ingestion run."""

    document_count: int
    chunk_count: int
    inserted_count: int
    skipped_count: int


def ingest_corpus(
    index_path: str | Path,
    repository: DocumentChunkRepository,
    embedding_provider: EmbeddingProvider,
    *,
    chunk_size: int = 600,
    overlap: int = 75,
    clear_existing: bool = True,
    batch_size: int = 5,
    max_chunks: int | None = None,
) -> IngestionResult:
    """Load corpus, embed incrementally, and store in Supabase.

    Args:
        index_path: Path to the corpus index.md file.
        repository: Supabase document chunk repository.
        embedding_provider: Provider for generating embeddings.
        chunk_size: Number of words per chunk.
        overlap: Number of overlapping words between chunks.
        clear_existing: Whether to clear existing chunks before ingestion.
        batch_size: Maximum number of chunks processed per embedding request.
        max_chunks: Optional maximum number of chunks to process.

    Returns:
        IngestionResult with document, chunk, inserted, and skipped counts.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive")

    documents = load_documents(index_path)

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # Limit the corpus before any database operation or embedding request.
    # This is useful for safe development/testing runs.
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    if clear_existing:
        repository.clear_chunks()

    if not chunks:
        return IngestionResult(
            document_count=len(documents),
            chunk_count=0,
            inserted_count=0,
            skipped_count=0,
        )

    # Find chunks that were successfully persisted by an earlier ingestion
    # run. These must be skipped before embedding so we do not waste Gemini
    # quota regenerating vectors that already exist.
    existing_ids = repository.get_existing_chunk_ids(
        [chunk.chunk_id for chunk in chunks],
        batch_size=100,
    )

    pending_chunks = [
        chunk
        for chunk in chunks
        if chunk.chunk_id not in existing_ids
    ]

    skipped_count = len(chunks) - len(pending_chunks)
    inserted_count = 0

    # Process and persist one batch at a time.
    #
    # This is intentionally incremental:
    #
    #     chunks -> Gemini -> Supabase
    #                          |
    #                          +--> next batch
    #
    # If a later batch fails, all earlier batches remain persisted.
    for start in range(0, len(pending_chunks), batch_size):
        batch = pending_chunks[start : start + batch_size]

        texts = [chunk.text for chunk in batch]

        embeddings = embedding_provider.embed_documents(texts)

        if len(embeddings) != len(batch):
            raise ValueError(
                f"Embedding response length ({len(embeddings)}) does not "
                f"match batch size ({len(batch)})"
            )

        stored = repository.insert_chunks_with_embeddings(
            batch,
            embeddings,
        )

        inserted_count += len(stored)

    return IngestionResult(
        document_count=len(documents),
        chunk_count=len(chunks),
        inserted_count=inserted_count,
        skipped_count=skipped_count,
    )
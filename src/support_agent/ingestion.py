"""Corpus ingestion: load, chunk, embed, and store documents."""

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


def ingest_corpus(
    index_path: str | Path,
    repository: DocumentChunkRepository,
    embedding_provider: EmbeddingProvider,
    *,
    chunk_size: int = 400,
    overlap: int = 50,
    clear_existing: bool = True,
) -> IngestionResult:
    """Load corpus, chunk documents, embed chunks, and store in Supabase.

    Args:
        index_path: Path to the corpus index.md file
        repository: Supabase document chunk repository
        embedding_provider: Provider for generating embeddings
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        clear_existing: Whether to clear existing chunks before ingestion

    Returns:
        IngestionResult with counts of documents, chunks, and inserted rows
    """
    documents = load_documents(index_path)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

    if clear_existing:
        repository.clear_chunks()

    if not chunks:
        return IngestionResult(
            document_count=len(documents), chunk_count=0, inserted_count=0
        )

    embeddings = embedding_provider.embed_texts([chunk.text for chunk in chunks])
    stored = repository.insert_chunks_with_embeddings(chunks, embeddings)

    return IngestionResult(
        document_count=len(documents),
        chunk_count=len(chunks),
        inserted_count=len(stored),
    )

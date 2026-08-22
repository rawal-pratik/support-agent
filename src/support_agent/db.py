import os
from collections.abc import Sequence
from typing import Any

from supabase import Client, create_client

from .corpus import DocumentChunk

TABLE_NAME = "document_chunks"
SEARCH_FUNCTION = "match_document_chunks"


class DocumentChunkRepository:
    """Small Supabase repository for storing and searching document chunks."""

    def __init__(self, client: Client):
        self._client = client

    @classmethod
    def from_env(cls) -> "DocumentChunkRepository":
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured")
        return cls(create_client(url, key))

    def insert_chunks(self, chunks: Sequence[DocumentChunk]) -> list[dict[str, Any]]:
        """Insert chunks and return the rows returned by Supabase."""

        rows = [
            {
                "id": chunk.chunk_id,
                "source_path": chunk.source_path,
                "title": chunk.title,
                "category": chunk.category,
                "chunk_text": chunk.text,
            }
            for chunk in chunks
        ]
        if not rows:
            return []
        response = self._client.table(TABLE_NAME).insert(rows).execute()
        return response.data

    def clear_chunks(self) -> None:
        """Remove all stored chunks for a development rebuild."""

        self._client.table(TABLE_NAME).delete().neq("id", "").execute()

    def similarity_search(
        self, query_embedding: Sequence[float], match_count: int = 5
    ) -> list[dict[str, Any]]:
        """Return closest chunks for an already-created query embedding."""

        if match_count <= 0:
            raise ValueError("match_count must be positive")
        response = self._client.rpc(
            SEARCH_FUNCTION,
            {
                "query_embedding": list(query_embedding),
                "match_count": match_count,
            },
        ).execute()
        return response.data

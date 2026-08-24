-- Adds the article source URL to document_chunks.

alter table public.document_chunks
    add column if not exists url text not null default '';

-- The return type changed because url was added.
drop function if exists public.match_document_chunks(vector, integer);

create function public.match_document_chunks(
    query_embedding vector(1536),
    match_count integer default 5
)
returns table (
    id text,
    source_path text,
    title text,
    category text,
    url text,
    chunk_text text,
    similarity real
)
language sql
stable
as $$
    select
        document_chunks.id,
        document_chunks.source_path,
        document_chunks.title,
        document_chunks.category,
        document_chunks.url,
        document_chunks.chunk_text,
        1 - (document_chunks.embedding <=> query_embedding) as similarity
    from public.document_chunks
    where document_chunks.embedding is not null
    order by document_chunks.embedding <=> query_embedding
    limit match_count;
$$;
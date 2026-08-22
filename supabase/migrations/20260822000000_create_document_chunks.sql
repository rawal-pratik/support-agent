create extension if not exists vector;

create table if not exists public.document_chunks (
    id text primary key,
    source_path text not null,
    title text not null,
    category text not null default '',
    chunk_text text not null,
    embedding vector(1536)
);

create or replace function public.match_document_chunks(
    query_embedding vector(1536),
    match_count integer default 5
)
returns table (
    id text,
    source_path text,
    title text,
    category text,
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
        document_chunks.chunk_text,
        1 - (document_chunks.embedding <=> query_embedding) as similarity
    from public.document_chunks
    where document_chunks.embedding is not null
    order by document_chunks.embedding <=> query_embedding
    limit match_count;
$$;

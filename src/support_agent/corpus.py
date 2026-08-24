import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field

MARKDOWN_LINK_PATTERN = re.compile(r"^- \[(.*)\]\(([^)]+)\)\s*$")
CATEGORY_PATTERN = re.compile(r"^##\s+(.+?)\s*$")

# Articles are exported with a YAML frontmatter block, e.g.:
#   ---
#   title: "Integration Logs"
#   source_url: "https://support.hackerrank.com/articles/7263906600-integration-logs"
#   ...
#   ---
# We only need `source_url` out of it, so we avoid a full YAML parser (the
# `breadcrumbs` list wouldn't round-trip cleanly through a simple parser
# anyway) and instead pull just that one key with a couple of small regexes.
FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n(.*?\n)---[ \t]*\r?\n?", re.S)
SOURCE_URL_PATTERN = re.compile(r'^source_url:\s*"?([^"\n]+?)"?\s*$', re.M)


class Document(BaseModel):
    """An article referenced by the corpus catalogue."""

    model_config = ConfigDict(frozen=True)

    source_path: str
    title: str = Field(min_length=1)
    category: str = ""
    url: str = ""
    text: str = Field(min_length=1)


class DocumentChunk(BaseModel):
    """A deterministic section of a document, retaining its source metadata."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    source_path: str
    title: str
    category: str = ""
    url: str = ""
    text: str = Field(min_length=1)


def load_documents(index_path: str | Path) -> list[Document]:
    """Load exactly the Markdown articles referenced by an index file."""

    index_file = Path(index_path)
    corpus_root = index_file.parent.resolve()
    current_category = ""
    documents: list[Document] = []

    for line_number, line in enumerate(
        index_file.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        category_match = CATEGORY_PATTERN.match(line)
        if category_match:
            current_category = category_match.group(1).strip()
            continue

        link_match = MARKDOWN_LINK_PATTERN.match(line)
        if not link_match:
            continue

        title, link = link_match.groups()
        relative_path = _article_path(link)
        article_path, resolved_relative_path = _resolve_article_path(
            corpus_root, relative_path
        )
        if not _is_within(article_path, corpus_root):
            raise ValueError(
                f"Article link on index line {line_number} escapes the corpus: {link}"
            )
        if not article_path.is_file():
            raise FileNotFoundError(
                f"Article referenced on index line {line_number} was not found: "
                f"{relative_path}"
            )

        url, body = _split_frontmatter(
            article_path.read_text(encoding="utf-8-sig")
        )

        documents.append(
            Document(
                source_path=resolved_relative_path,
                title=title.strip(),
                category=current_category,
                url=url,
                text=body.strip(),
            )
        )

    return documents


def _split_frontmatter(raw_text: str) -> tuple[str, str]:
    """Pull the article's source_url out of its YAML frontmatter, if present.

    Returns (url, body) where body has the frontmatter block removed so it
    doesn't leak into chunked text. Articles without frontmatter (or without
    a source_url key) simply get an empty url — this is a best-effort
    enrichment, not a requirement.
    """

    match = FRONTMATTER_PATTERN.match(raw_text)
    if not match:
        return "", raw_text

    frontmatter, body = match.group(1), raw_text[match.end():]
    url_match = SOURCE_URL_PATTERN.search(frontmatter)
    url = url_match.group(1).strip() if url_match else ""
    return url, body


def chunk_documents(
    documents: list[Document], chunk_size: int = 400, overlap: int = 50
) -> list[DocumentChunk]:
    """Split documents into word-based chunks with deterministic overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap
    for document in documents:
        words = document.text.split()
        for chunk_number, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.source_path}#chunk-{chunk_number}",
                    source_path=document.source_path,
                    title=document.title,
                    category=document.category,
                    url=document.url,
                    text=" ".join(chunk_words),
                )
            )
            if start + chunk_size >= len(words):
                break

    return chunks


def _article_path(link: str) -> Path:
    path = urlsplit(link).path
    if not path or not path.lower().endswith(".md"):
        raise ValueError(f"Index link is not a Markdown article: {link}")
    return Path(unquote(path))


def _resolve_article_path(root: Path, relative_path: Path) -> tuple[Path, str]:
    """Resolve an indexed path, including the corpus export's filename escaping."""

    article_path = (root / Path(*relative_path.parts)).resolve()
    if article_path.is_file():
        return article_path, relative_path.as_posix()

    filename_match = re.match(r"^(\d+-).*\.md$", relative_path.name)
    if filename_match:
        candidates = sorted(
            (root / Path(*relative_path.parent.parts)).glob(
                f"{filename_match.group(1)}*.md"
            )
        )
        if len(candidates) == 1:
            resolved = candidates[0].resolve()
            return resolved, resolved.relative_to(root).as_posix()

    return article_path, relative_path.as_posix()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
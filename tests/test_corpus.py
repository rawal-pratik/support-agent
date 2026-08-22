from pathlib import Path

import pytest

from support_agent.corpus import Document, chunk_documents, load_documents

CORPUS_ROOT = Path(__file__).parents[1] / "public" / "corpus"


def test_loads_all_and_only_articles_listed_in_real_index():
    documents = load_documents(CORPUS_ROOT / "index.md")

    assert len(documents) == 394
    assert documents[0].source_path == (
        "uncategorized/1231590424-assessing-candidates-on-prompt-engineering-skills.md"
    )
    assert documents[0].title == "Assessing Candidates on Prompt Engineering Skills"
    assert documents[0].category == "Uncategorized"
    assert all(document.source_path != "index.md" for document in documents)
    assert all((CORPUS_ROOT / document.source_path).is_file() for document in documents)


def test_preserves_nested_category_and_article_metadata():
    documents = load_documents(CORPUS_ROOT / "index.md")

    document = next(
        document
        for document in documents
        if document.source_path.endswith("contact-hackerrank-support.md")
    )

    assert document.title == "Contact HackerRank Support"
    assert document.category == "General Help / Contact Us"
    assert "Contact" in document.text


def test_chunks_are_deterministic_and_retain_source_metadata():
    document = Document(
        source_path="category/article.md",
        title="Article",
        category="Category",
        text="one two three four five six seven eight nine ten",
    )

    chunks = chunk_documents([document], chunk_size=4, overlap=1)

    assert [chunk.chunk_id for chunk in chunks] == [
        "category/article.md#chunk-0",
        "category/article.md#chunk-1",
        "category/article.md#chunk-2",
    ]
    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]
    assert all(chunk.source_path == document.source_path for chunk in chunks)
    assert all(chunk.title == document.title for chunk in chunks)
    assert all(chunk.category == document.category for chunk in chunks)


def test_chunking_validates_size_and_overlap():
    document = Document(source_path="article.md", title="Article", text="one two")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_documents([document], chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        chunk_documents([document], chunk_size=2, overlap=2)


def test_missing_index_reference_fails_clearly(tmp_path: Path):
    index = tmp_path / "index.md"
    index.write_text("# Catalogue\n\n## Category\n- [Missing](missing.md)\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing.md"):
        load_documents(index)


def test_does_not_load_unreferenced_markdown_files(tmp_path: Path):
    (tmp_path / "listed.md").write_text("Listed article", encoding="utf-8")
    (tmp_path / "unlisted.md").write_text("Unlisted article", encoding="utf-8")
    (tmp_path / "index.md").write_text(
        "# Catalogue\n\n## Category\n- [Listed](listed.md)\n", encoding="utf-8"
    )

    documents = load_documents(tmp_path / "index.md")

    assert [document.source_path for document in documents] == ["listed.md"]

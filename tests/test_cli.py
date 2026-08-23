"""Tests for the terminal CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from support_agent.cli import app
from support_agent.models import AgentResult, Ticket
from support_agent.retrieval import RetrievedChunk
from support_agent.ingestion import IngestionResult
from support_agent.policy import PolicyDecision, RequestType, RiskLevel


runner = CliRunner()


def make_ticket(subject: str = "Test", issue: str = "This is a test issue") -> Ticket:
    return Ticket(subject=subject, issue=issue)


def make_policy_decision(
    should_escalate: bool = False,
    request_type: RequestType = RequestType.BUG,
    risk_level: RiskLevel = RiskLevel.LOW,
    escalation_reason=None,
) -> PolicyDecision:
    return PolicyDecision(
        should_escalate=should_escalate,
        request_type=request_type,
        risk_level=risk_level,
        escalation_reason=escalation_reason,
    )


def make_retrieved_chunk(
    chunk_id: str = "doc.md#chunk-0",
    title: str = "Test Doc",
    category: str = "General",
    text: str = "How to do the thing.",
    similarity: float = 0.85,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_path="doc.md",
        title=title,
        category=category,
        text=text,
        similarity=similarity,
    )


def make_agent_result(
    status: str = "replied",
    product_area: str = "Screen",
    response: str = "Based on the docs, do X.",
    justification: str = "Doc 1 says to do X.",
    request_type: str = "bug",
) -> AgentResult:
    return AgentResult(
        status=status,
        product_area=product_area,
        response=response,
        justification=justification,
        request_type=request_type,
    )


def make_mock_retriever():
    """Create a mock retriever."""
    mock = MagicMock()
    mock.retrieve = MagicMock()
    return mock


def make_mock_llm():
    """Create a mock LLM provider."""
    mock = MagicMock()
    mock.generate = MagicMock()
    return mock


class TestRootHelp:
    """Tests for root command help."""

    def test_root_help(self):
        """Root command shows help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "support-agent" in result.output.lower()

    def test_version_flag(self):
        """--version shows version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestCommandHelp:
    """Tests for each command's help."""

    def test_ingest_help(self):
        """ingest --help works."""
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0

    def test_search_help(self):
        """search --help works."""
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0

    def test_triage_help(self):
        """triage --help works."""
        result = runner.invoke(app, ["triage", "--help"])
        assert result.exit_code == 0

    def test_evaluate_help(self):
        """evaluate --help works."""
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0


class TestIngestCommand:
    """Tests for ingest command."""

    def test_ingest_invokes_ingestion_service(self):
        """ingest invokes ingest_corpus correctly."""
        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls, \
             patch("support_agent.cli.DocumentChunkRepository") as mock_repo_cls, \
             patch("support_agent.cli.ingest_corpus") as mock_ingest:
            mock_embed_cls.from_env.return_value = MagicMock()
            mock_repo_cls.from_env.return_value = MagicMock()
            mock_ingest.return_value = IngestionResult(
                document_count=10, chunk_count=50, inserted_count=50
            )

            result = runner.invoke(app, ["ingest", "--index-path", "public/corpus/index.md"])

            assert result.exit_code == 0
            mock_ingest.assert_called_once()
            assert "Chunks inserted: 50" in result.output

    def test_ingest_missing_config_exits_nonzero(self):
        """ingest with missing config exits non-zero."""
        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls:
            mock_embed_cls.from_env.side_effect = ValueError("GEMINI_API_KEY must be configured")

            result = runner.invoke(app, ["ingest"])

            assert result.exit_code == 1
            assert "Configuration error" in result.output


class TestSearchCommand:
    """Tests for search command."""

    def test_search_invokes_retrieval(self):
        """search invokes retrieval with query and options."""
        mock_retriever = make_mock_retriever()
        mock_retriever.retrieve.return_value = [
            make_retrieved_chunk(title="How to create a test", category="Screen")
        ]

        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls, \
             patch("support_agent.cli.DocumentChunkRepository") as mock_repo_cls, \
             patch("support_agent.cli.SemanticRetriever", return_value=mock_retriever) as mock_retriever_cls:

            mock_embed_cls.from_env.return_value = MagicMock()
            mock_repo_cls.from_env.return_value = MagicMock()

            result = runner.invoke(
                app, ["search", "how do I create a test", "--top-k", "5", "--threshold", "0.3"]
            )

            assert result.exit_code == 0
            # Verify retriever was constructed with options
            mock_retriever_cls.assert_called_once()
            _, kwargs = mock_retriever_cls.call_args
            assert kwargs["top_k"] == 5
            assert kwargs["similarity_threshold"] == 0.3

            # Verify retrieve was called with a Ticket containing the query
            mock_retriever.retrieve.assert_called_once()
            ticket_arg = mock_retriever.retrieve.call_args[0][0]
            assert ticket_arg.issue == "how do I create a test"

    def test_search_displays_result_info(self):
        """search displays source, title, category, similarity."""
        mock_retriever = make_mock_retriever()
        mock_retriever.retrieve.return_value = [
            make_retrieved_chunk(
                chunk_id="doc.md#chunk-0",
                title="Test Title",
                category="Test Category",
                text="Some doc text",
                similarity=0.9123,
            )
        ]

        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls, \
             patch("support_agent.cli.DocumentChunkRepository") as mock_repo_cls, \
             patch("support_agent.cli.SemanticRetriever", return_value=mock_retriever):

            mock_embed_cls.from_env.return_value = MagicMock()
            mock_repo_cls.from_env.return_value = MagicMock()

            result = runner.invoke(app, ["search", "query text"])

            assert result.exit_code == 0
            assert "doc.md#chunk-0" in result.output  # chunk ID
            assert "doc.md" in result.output  # source
            assert "Test Title" in result.output  # title
            assert "Test Category" in result.output  # category
            assert "0.9123" in result.output  # similarity

    def test_search_empty_query_fails(self):
        """search with empty query fails."""
        result = runner.invoke(app, ["search", ""])
        assert result.exit_code == 1

    def test_search_failure_exits_nonzero(self):
        """search with provider failure exits non-zero."""
        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls:
            mock_embed_cls.from_env.side_effect = ValueError("GEMINI_API_KEY must be configured")

            result = runner.invoke(app, ["search", "some query"])

            assert result.exit_code == 1


class TestTriageCommand:
    """Tests for triage command."""

    def test_triage_reads_csv_writes_results(self, tmp_path):
        """triage reads input CSV and writes output columns."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            "Issue,Subject\nCannot log in to my account,Login problem\n", encoding="utf-8"
        )

        output_csv = tmp_path / "output.csv"

        mock_retriever = make_mock_retriever()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm = make_mock_llm()
        mock_llm.generate.return_value = make_agent_result()

        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls, \
             patch("support_agent.cli.DocumentChunkRepository") as mock_repo_cls, \
             patch("support_agent.cli.OpenRouterProvider") as mock_llm_cls, \
             patch("support_agent.cli.SemanticRetriever", return_value=mock_retriever):

            mock_embed_cls.from_env.return_value = MagicMock()
            mock_repo_cls.from_env.return_value = MagicMock()
            mock_llm_cls.from_env.return_value = mock_llm

            result = runner.invoke(app, ["triage", str(input_csv), str(output_csv)])

            assert result.exit_code == 0
            assert output_csv.exists()

            # Verify output columns
            content = output_csv.read_text(encoding="utf-8")
            assert "status" in content
            assert "product_area" in content
            assert "response" in content
            assert "justification" in content
            assert "request_type" in content

    def test_triage_preserves_row_order(self, tmp_path):
        """triage preserves input row order."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            "Issue,Subject\n"
            "Issue A with more text here,Subject A\n"
            "Issue B with more text here,Subject B\n"
            "Issue C with more text here,Subject C\n",
            encoding="utf-8"
        )
        output_csv = tmp_path / "output.csv"

        mock_retriever = make_mock_retriever()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm = make_mock_llm()
        mock_llm.generate.return_value = make_agent_result()

        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls, \
             patch("support_agent.cli.DocumentChunkRepository") as mock_repo_cls, \
             patch("support_agent.cli.OpenRouterProvider") as mock_llm_cls, \
             patch("support_agent.cli.SemanticRetriever", return_value=mock_retriever):

            mock_embed_cls.from_env.return_value = MagicMock()
            mock_repo_cls.from_env.return_value = MagicMock()
            mock_llm_cls.from_env.return_value = mock_llm

            result = runner.invoke(app, ["triage", str(input_csv), str(output_csv)])

            assert result.exit_code == 0

            # Orchestrator should have been called 3 times (once per ticket)
            assert mock_llm.generate.call_count == 3

    def test_triage_invokes_orchestrator_once_per_ticket(self, tmp_path):
        """triage invokes orchestrator.process_ticket once per ticket."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            "Issue,Subject\n"
            "Issue one with enough text,A\n"
            "Issue two with enough text,B\n"
            "Issue three with enough text,C\n"
            "Issue four with enough text,D\n",
            encoding="utf-8"
        )
        output_csv = tmp_path / "output.csv"

        mock_orchestrator = MagicMock()
        mock_orchestrator.process_ticket.return_value = make_agent_result()

        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls, \
             patch("support_agent.cli.DocumentChunkRepository") as mock_repo_cls, \
             patch("support_agent.cli.OpenRouterProvider") as mock_llm_cls, \
             patch("support_agent.cli.SemanticRetriever") as mock_retriever_cls, \
             patch("support_agent.cli.TriageOrchestrator", return_value=mock_orchestrator):

            mock_embed_cls.from_env.return_value = MagicMock()
            mock_repo_cls.from_env.return_value = MagicMock()
            mock_llm_cls.from_env.return_value = make_mock_llm()
            mock_retriever_cls.return_value = make_mock_retriever()

            result = runner.invoke(app, ["triage", str(input_csv), str(output_csv)])

            assert result.exit_code == 0
            # TriageOrchestrator.process_ticket should be called exactly once per ticket
            assert mock_orchestrator.process_ticket.call_count == 4

    def test_triage_missing_input_file_fails(self):
        """triage with missing input file exits non-zero."""
        result = runner.invoke(app, ["triage", "nonexistent.csv", "output.csv"])
        # Typer returns exit code 2 for missing file argument
        assert result.exit_code in (1, 2)
        assert "Input file not found" in result.output or "does not exist" in result.output

    def test_triage_invalid_csv_fails(self, tmp_path):
        """triage with invalid CSV exits non-zero."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("wrong_column\nvalue\n", encoding="utf-8")

        output_csv = tmp_path / "output.csv"

        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls, \
             patch("support_agent.cli.DocumentChunkRepository") as mock_repo_cls, \
             patch("support_agent.cli.OpenRouterProvider") as mock_llm_cls, \
             patch("support_agent.cli.SemanticRetriever") as mock_retriever_cls:

            mock_embed_cls.from_env.return_value = MagicMock()
            mock_repo_cls.from_env.return_value = MagicMock()
            mock_llm_cls.from_env.return_value = MagicMock()
            mock_retriever_cls.return_value = make_mock_retriever()

            result = runner.invoke(app, ["triage", str(bad_csv), str(output_csv)])

            assert result.exit_code == 1
            assert "Invalid input CSV" in result.output

    def test_triage_provider_failure_exits_nonzero(self, tmp_path):
        """triage with provider config failure exits non-zero."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            "Issue,Subject\nA valid issue with enough text,1\n", encoding="utf-8"
        )
        output_csv = tmp_path / "output.csv"

        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls:
            mock_embed_cls.from_env.side_effect = ValueError("GEMINI_API_KEY must be configured")

            result = runner.invoke(app, ["triage", str(input_csv), str(output_csv)])

            assert result.exit_code == 1


class TestEvaluateCommand:
    """Tests for evaluate command interface."""

    def test_evaluate_has_command_interface(self):
        """evaluate command is available."""
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0

    def test_evaluate_not_implemented(self):
        """evaluate reports not implemented."""
        result = runner.invoke(app, ["evaluate"])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output.lower()


class TestSecretsNotPrinted:
    """Tests that secrets are not printed."""

    def test_ingest_failure_does_not_print_secret(self, tmp_path):
        """ingest failure should not print API keys."""
        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls:
            mock_embed_cls.from_env.side_effect = ValueError("GEMINI_API_KEY must be configured")

            result = runner.invoke(app, ["ingest"])

            assert result.exit_code == 1
            assert "GEMINI_API_KEY" in result.output
            # Verify no actual key value printed (would be in env)
            assert "AIza" not in result.output  # typical Gemini key prefix
            assert "sk-" not in result.output  # typical OpenAI/Supabase prefix

    def test_search_failure_does_not_print_secret(self):
        """search failure should not print API keys."""
        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed_cls:
            mock_embed_cls.from_env.side_effect = ValueError("GEMINI_API_KEY must be configured")

            result = runner.invoke(app, ["search", "some query"])

            assert result.exit_code == 1
            assert "GEMINI_API_KEY" in result.output
            assert "AIza" not in result.output

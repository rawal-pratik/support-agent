"""Tests for the evaluation module."""

import csv
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from support_agent.evaluation import (
    load_sample_tickets,
    evaluate_sample_tickets,
    format_report,
    SampleTicket,
    EvaluationReport,
    EvaluationResult,
    run_evaluation,
)
from support_agent.models import AgentResult, Ticket
from support_agent.orchestrator import TriageOrchestrator
from support_agent.policy import RequestType


def make_sample_csv_content() -> str:
    """Create sample CSV content matching the real schema."""
    return """Issue,Subject,Company,Response,Product Area,Status,Request Type
Test issue 1,Subject 1,Company A,Response 1,screen,Replied,product_issue
Test issue 2,,Company B,Response 2,,Escalated,bug
Test issue 3,Subject 3,None,Response 3,community,Replied,product_issue
Test issue 4,Subject 4,None,Response 4,conversation_management,Replied,invalid
Test issue 5,,None,Response 5,,Replied,invalid
Test issue 6,Subject 6,None,Response 6,screen,Escalated,bug
Test issue 7,Subject 7,HackerRank,Response 7,screen,Replied,product_issue
"""


def make_mock_orchestrator():
    """Create a mock orchestrator with controllable results."""
    mock = MagicMock(spec=TriageOrchestrator)
    return mock


def make_agent_result(
    status: str = "replied",
    product_area: str = "screen",
    request_type: str = "product_issue",
    response: str = "Based on docs...",
) -> AgentResult:
    # AgentResult uses lowercase status/request_type literals
    status_lc = status.lower()
    request_type_lc = request_type.lower()
    return AgentResult(
        status=status_lc,
        product_area=product_area,
        response=response,
        justification="Test justification",
        request_type=request_type_lc,
    )


class TestLoadSampleTickets:
    """Tests for loading sample tickets from CSV."""

    def test_loads_exactly_seven_tickets(self, tmp_path):
        """Should discover exactly 7 sample tickets."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        tickets = load_sample_tickets(csv_path)

        assert len(tickets) == 7

    def test_expected_columns_detected(self, tmp_path):
        """Expected reference columns are detected."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        tickets = load_sample_tickets(csv_path)

        # Check all tickets have expected fields populated
        assert all(hasattr(t, "expected_status") for t in tickets)
        assert all(hasattr(t, "expected_product_area") for t in tickets)
        assert all(hasattr(t, "expected_request_type") for t in tickets)

    def test_blank_subject_handled_correctly(self, tmp_path):
        """Blank Subject cells are handled correctly."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        tickets = load_sample_tickets(csv_path)

        # Tickets 2 and 5 have blank subjects
        assert tickets[1].subject == ""
        assert tickets[4].subject == ""

    def test_blank_product_area_handled_correctly(self, tmp_path):
        """Blank Product Area cells are handled correctly (not 'nan')."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        tickets = load_sample_tickets(csv_path)

        # Tickets 2 and 5 have blank product areas
        assert tickets[1].expected_product_area == ""
        assert tickets[4].expected_product_area == ""

    def test_blank_company_not_required(self, tmp_path):
        """Company column is not required for evaluation."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        tickets = load_sample_tickets(csv_path)

        # Should not raise; Company is simply ignored
        assert len(tickets) == 7

    def test_normalizes_status_values(self, tmp_path):
        """Status values are normalized (lowercase)."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        tickets = load_sample_tickets(csv_path)

        assert tickets[0].expected_status == "replied"
        assert tickets[1].expected_status == "escalated"

    def test_normalizes_request_type_values(self, tmp_path):
        """Request Type values are normalized (lowercased)."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        tickets = load_sample_tickets(csv_path)

        assert tickets[0].expected_request_type == "product_issue"
        assert tickets[1].expected_request_type == "bug"
        assert tickets[3].expected_request_type == "invalid"

    def test_normalizes_product_area_values(self, tmp_path):
        """Product Area values are normalized (lowercased)."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        tickets = load_sample_tickets(csv_path)

        assert tickets[0].expected_product_area == "screen"
        assert tickets[2].expected_product_area == "community"
        assert tickets[3].expected_product_area == "conversation_management"

    def test_malformed_missing_columns_raises(self, tmp_path):
        """Missing required columns raises clear error."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("Issue,Subject\nTest,Subject\n")

        with pytest.raises(ValueError, match="Missing required columns"):
            load_sample_tickets(bad_csv)

    def test_empty_evaluation_input(self, tmp_path):
        """Empty CSV (only header) returns empty list."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("Issue,Subject,Product Area,Status,Request Type\n")

        tickets = load_sample_tickets(empty_csv)

        assert tickets == []


class TestEvaluationMetrics:
    """Tests for evaluation metric calculations."""

    def test_status_exact_match_calculation(self):
        """Status exact-match calculation is correct."""
        mock_orch = make_mock_orchestrator()
        # 2 correct, 1 wrong
        mock_orch.process_ticket.side_effect = [
            make_agent_result(status="Replied"),
            make_agent_result(status="Escalated"),
            make_agent_result(status="Replied"),  # wrong, expected Escalated
        ]

        samples = [
            SampleTicket("i1", "", "replied", "", "product_issue"),
            SampleTicket("i2", "", "escalated", "", "bug"),
            SampleTicket("i3", "", "escalated", "", "bug"),  # expected escalated, got replied
        ]

        report = evaluate_sample_tickets(mock_orch, samples)

        assert report.total_tickets == 3
        assert report.status_accuracy == 2 / 3

    def test_request_type_exact_match_calculation(self):
        """Request type exact-match calculation is correct."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.side_effect = [
            make_agent_result(request_type="product_issue"),
            make_agent_result(request_type="bug"),
            make_agent_result(request_type="invalid"),
        ]

        samples = [
            SampleTicket("i1", "", "replied", "", "product_issue"),
            SampleTicket("i2", "", "escalated", "", "bug"),
            SampleTicket("i3", "", "replied", "", "product_issue"),  # expected product_issue, got invalid
        ]

        report = evaluate_sample_tickets(mock_orch, samples)

        assert report.request_type_accuracy == 2 / 3

    def test_product_area_exact_match_calculation(self):
        """Product area exact-match calculation is correct."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.side_effect = [
            make_agent_result(product_area="screen"),
            make_agent_result(product_area=""),  # blank
            make_agent_result(product_area="screen"),  # wrong, expected community
        ]

        samples = [
            SampleTicket("i1", "", "replied", "screen", "product_issue"),
            SampleTicket("i2", "", "escalated", "", "bug"),
            SampleTicket("i3", "", "replied", "community", "product_issue"),
        ]

        report = evaluate_sample_tickets(mock_orch, samples)

        assert report.product_area_accuracy == 2 / 3

    def test_blank_expected_product_area_handled_correctly(self):
        """Blank expected product area matches blank actual."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.side_effect = [
            make_agent_result(product_area=""),  # blank matches blank
            make_agent_result(product_area="screen"),  # wrong, expected blank
        ]

        samples = [
            SampleTicket("i1", "", "escalated", "", "bug"),
            SampleTicket("i2", "", "escalated", "", "bug"),  # expected blank
        ]

        report = evaluate_sample_tickets(mock_orch, samples)

        assert report.product_area_accuracy == 1 / 2
        # First should match (both blank)
        assert report.results[0].product_area_match is True
        # Second should not match (actual=screen, expected=blank)
        assert report.results[1].product_area_match is False

    def test_overall_exact_match_rate(self):
        """Overall exact-match rate across three fields."""
        mock_orch = make_mock_orchestrator()
        # Ticket 1: all match
        # Ticket 2: only status matches
        # Ticket 3: none match
        mock_orch.process_ticket.side_effect = [
            make_agent_result(status="Replied", product_area="screen", request_type="product_issue"),
            make_agent_result(status="Escalated", product_area="screen", request_type="bug"),
            make_agent_result(status="Replied", product_area="community", request_type="invalid"),
        ]

        samples = [
            SampleTicket("i1", "", "replied", "screen", "product_issue"),
            SampleTicket("i2", "", "escalated", "", "bug"),  # only status matches
            SampleTicket("i3", "", "escalated", "", "bug"),  # none match
        ]

        report = evaluate_sample_tickets(mock_orch, samples)

        assert report.overall_exact_match_rate == 1 / 3  # only 1 of 3 all_match

    def test_per_ticket_mismatch_reporting(self):
        """Per-ticket mismatches are reported with expected vs actual."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.return_value = make_agent_result(
            status="Replied",
            product_area="screen",
            request_type="product_issue",
        )

        samples = [
            SampleTicket("i1", "", "escalated", "community", "bug"),  # all different
        ]

        report = evaluate_sample_tickets(mock_orch, samples)

        result = report.results[0]
        assert result.all_match is False
        assert result.status_match is False
        assert result.product_area_match is False
        assert result.request_type_match is False
        assert result.expected_status == "escalated"
        assert result.actual_status == "replied"
        assert result.expected_product_area == "community"
        assert result.actual_product_area == "screen"
        assert result.expected_request_type == "bug"
        assert result.actual_request_type == "product_issue"

    def test_deterministic_evaluation(self):
        """Same inputs produce same evaluation results."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.return_value = make_agent_result()

        samples = [
            SampleTicket("i1", "", "replied", "screen", "product_issue"),
        ]

        report1 = evaluate_sample_tickets(mock_orch, samples)
        report2 = evaluate_sample_tickets(mock_orch, samples)

        assert report1.status_accuracy == report2.status_accuracy
        assert report1.results[0].status_match == report2.results[0].status_match


class TestResponseChecks:
    """Tests for lightweight response quality checks."""

    def test_replied_result_non_empty_response_check(self):
        """Replied results should have non-empty response."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.return_value = make_agent_result(
            status="replied",
            response="   ",  # whitespace only
        )

        samples = [SampleTicket("i1", "", "replied", "screen", "product_issue")]
        report = evaluate_sample_tickets(mock_orch, samples)

        assert report.results[0].response_checks["non_empty_response"] is False

    def test_replied_result_with_content_passes(self):
        """Replied result with content passes check."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.return_value = make_agent_result(
            status="replied",
            response="Here is the answer.",
        )

        samples = [SampleTicket("i1", "", "replied", "screen", "product_issue")]
        report = evaluate_sample_tickets(mock_orch, samples)

        assert report.results[0].response_checks["non_empty_response"] is True

    def test_escalated_result_indicates_escalation(self):
        """Escalated results should indicate escalation/human review."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.return_value = make_agent_result(
            status="escalated",
            response="This has been escalated to human review.",
        )

        samples = [SampleTicket("i1", "", "escalated", "", "bug")]
        report = evaluate_sample_tickets(mock_orch, samples)

        assert report.results[0].response_checks["indicates_escalation"] is True

    def test_escalated_without_indication_fails(self):
        """Escalated result without escalation indication fails check."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.return_value = make_agent_result(
            status="escalated",
            response="This is just a normal reply.",
        )

        samples = [SampleTicket("i1", "", "escalated", "", "bug")]
        report = evaluate_sample_tickets(mock_orch, samples)

        assert report.results[0].response_checks["indicates_escalation"] is False

    def test_response_text_not_compared_exact(self):
        """Response text is NOT compared using exact string equality."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.return_value = make_agent_result(
            response="Completely different text from reference",
        )

        samples = [SampleTicket("i1", "", "replied", "screen", "product_issue")]
        report = evaluate_sample_tickets(mock_orch, samples)

        # Response text difference should NOT affect the three-field match
        # The three-field match only cares about status, product_area, request_type
        # The reference Response column is loaded but not used for exact matching
        # (it's only used for the lightweight quality checks)
        assert "Response" not in dir(report.results[0]) or \
               not hasattr(report.results[0], "expected_response")


class TestFormatReport:
    """Tests for report formatting."""

    def test_format_report_contains_metrics(self):
        """Formatted report contains aggregate metrics."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.return_value = make_agent_result()

        samples = [SampleTicket("i1", "", "replied", "screen", "product_issue")]
        report = evaluate_sample_tickets(mock_orch, samples)

        formatted = format_report(report)

        assert "EVALUATION REPORT" in formatted
        assert "Total tickets evaluated: 1" in formatted
        assert "Status accuracy" in formatted
        assert "Request type accuracy" in formatted
        assert "Product area accuracy" in formatted
        assert "Overall exact-match rate" in formatted

    def test_format_report_shows_mismatches(self):
        """Formatted report shows per-ticket mismatches."""
        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.return_value = make_agent_result(
            status="Replied", product_area="screen", request_type="product_issue"
        )

        samples = [SampleTicket("i1", "", "escalated", "community", "bug")]
        report = evaluate_sample_tickets(mock_orch, samples)

        formatted = format_report(report)

        # Should show mismatch details
        assert "escalated" in formatted
        assert "replied" in formatted
        assert "community" in formatted
        assert "screen" in formatted
        assert "bug" in formatted
        assert "product_issue" in formatted


class TestRunEvaluation:
    """Tests for the main run_evaluation entry point."""

    def test_run_evaluation_loads_and_evaluates(self, tmp_path):
        """run_evaluation loads CSV and runs evaluation."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        mock_orch = make_mock_orchestrator()
        mock_orch.process_ticket.return_value = make_agent_result()

        report = run_evaluation(csv_path, mock_orch)

        assert report.total_tickets == 7
        assert mock_orch.process_ticket.call_count == 7


class TestCLIEvaluateCommand:
    """Tests for the CLI evaluate command."""

    def test_evaluate_command_invokes_evaluator(self, tmp_path, monkeypatch):
        """CLI evaluate invokes the evaluator with mocked providers."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())
        output_csv = tmp_path / "output.csv"

        # Mock all external providers
        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed, \
             patch("support_agent.cli.DocumentChunkRepository") as mock_repo, \
             patch("support_agent.cli.OpenRouterProvider") as mock_llm, \
             patch("support_agent.cli.SemanticRetriever") as mock_retriever_cls, \
             patch("support_agent.cli.TriageOrchestrator") as mock_orch_cls:

            mock_embed.from_env.return_value = MagicMock()
            mock_repo.from_env.return_value = MagicMock()
            mock_llm.from_env.return_value = MagicMock()

            mock_orch = MagicMock()
            mock_orch.process_ticket.return_value = make_agent_result()
            mock_orch_cls.return_value = mock_orch
            mock_retriever_cls.return_value = MagicMock()

            from typer.testing import CliRunner
            from support_agent.cli import app

            runner = CliRunner()
            result = runner.invoke(app, ["evaluate", "--golden", str(csv_path)])

            assert result.exit_code == 0
            mock_orch_cls.assert_called_once()
            assert mock_orch.process_ticket.call_count == 7

    def test_evaluate_missing_golden_fails(self):
        """evaluate without --golden fails with clear error."""
        from typer.testing import CliRunner
        from support_agent.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["evaluate"])

        assert result.exit_code == 1
        assert "required" in result.output.lower() or "golden" in result.output.lower()

    def test_evaluate_nonexistent_golden_fails(self):
        """evaluate with nonexistent --golden fails."""
        from typer.testing import CliRunner
        from support_agent.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["evaluate", "--golden", "nonexistent.csv"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_evaluate_reports_metrics(self, tmp_path):
        """evaluate command reports metrics in output."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed, \
             patch("support_agent.cli.DocumentChunkRepository") as mock_repo, \
             patch("support_agent.cli.OpenRouterProvider") as mock_llm, \
             patch("support_agent.cli.SemanticRetriever") as mock_retriever_cls, \
             patch("support_agent.cli.TriageOrchestrator") as mock_orch_cls:

            mock_embed.from_env.return_value = MagicMock()
            mock_repo.from_env.return_value = MagicMock()
            mock_llm.from_env.return_value = MagicMock()

            mock_orch = MagicMock()
            # Make first 3 match, last 4 mismatch
            mock_orch.process_ticket.side_effect = [
                make_agent_result(status="Replied", product_area="screen", request_type="product_issue"),
                make_agent_result(status="Escalated", product_area="", request_type="bug"),
                make_agent_result(status="Replied", product_area="screen", request_type="product_issue"),
                make_agent_result(status="Replied", product_area="community", request_type="product_issue"),
                make_agent_result(status="Replied", product_area="conversation_management", request_type="invalid"),
                make_agent_result(status="Replied", product_area="", request_type="invalid"),
                make_agent_result(status="Escalated", product_area="screen", request_type="bug"),
            ]
            mock_orch_cls.return_value = mock_orch
            mock_retriever_cls.return_value = MagicMock()

            from typer.testing import CliRunner
            from support_agent.cli import app

            runner = CliRunner()
            result = runner.invoke(app, ["evaluate", "--golden", str(csv_path)])

            assert result.exit_code == 0
            assert "EVALUATION REPORT" in result.output
            assert "Status accuracy" in result.output
            assert "Request type accuracy" in result.output
            assert "Product area accuracy" in result.output
            assert "Overall exact-match rate" in result.output

    def test_external_providers_mocked(self, tmp_path):
        """Tests ensure Supabase/Gemini/OpenRouter are never called."""
        csv_path = tmp_path / "sample.csv"
        csv_path.write_text(make_sample_csv_content())

        with patch("support_agent.cli.GeminiEmbeddingProvider") as mock_embed, \
             patch("support_agent.cli.DocumentChunkRepository") as mock_repo, \
             patch("support_agent.cli.OpenRouterProvider") as mock_llm, \
             patch("support_agent.cli.SemanticRetriever") as mock_retriever_cls, \
             patch("support_agent.cli.TriageOrchestrator") as mock_orch_cls:

            mock_embed.from_env.return_value = MagicMock()
            mock_repo.from_env.return_value = MagicMock()
            mock_llm.from_env.return_value = MagicMock()
            mock_orch = MagicMock()
            mock_orch.process_ticket.return_value = make_agent_result()
            mock_orch_cls.return_value = mock_orch
            mock_retriever_cls.return_value = MagicMock()

            from typer.testing import CliRunner
            from support_agent.cli import app

            runner = CliRunner()
            result = runner.invoke(app, ["evaluate", "--golden", str(csv_path)])

            assert result.exit_code == 0
            # Verify .from_env was called but actual APIs weren't hit
            mock_embed.from_env.assert_called_once()
            mock_repo.from_env.assert_called_once()
            mock_llm.from_env.assert_called_once()
"""Basic tests for golden-sample evaluation."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from support_agent.evaluation import (
    EvaluationReport,
    SampleTicket,
    evaluate_sample_tickets,
    format_report,
    load_sample_tickets,
    run_evaluation,
)
from support_agent.models import AgentResult


def make_result(
    status="replied",
    product_area="Screen / Managing Tests",
    request_type="product_issue",
    response="Here is the answer.",
):
    return AgentResult(
        status=status,
        product_area=product_area,
        response=response,
        justification="Based on the documentation.",
        request_type=request_type,
    )


class TestLoadSampleTickets:
    def test_load_sample_tickets(self, tmp_path):
        path = tmp_path / "golden.csv"

        path.write_text(
            "Issue,Subject,Product Area,Status,Request Type\n"
            "How do I create a test?,Create test,"
            "Screen / Managing Tests,replied,product_issue\n",
            encoding="utf-8",
        )

        samples = load_sample_tickets(path)

        assert len(samples) == 1
        assert samples[0].issue == "How do I create a test?"
        assert samples[0].subject == "Create test"
        assert samples[0].expected_status == "replied"
        assert (
            samples[0].expected_product_area
            == "screen / managing tests"
        )
        assert samples[0].expected_request_type == "product_issue"

    def test_load_sample_tickets_requires_columns(self, tmp_path):
        path = tmp_path / "invalid.csv"

        path.write_text(
            "Issue,Subject\n"
            "Test,Subject\n",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match="CSV must contain columns",
        ):
            load_sample_tickets(path)


class TestEvaluateSampleTickets:
    def test_matching_ticket(self):
        orchestrator = MagicMock()

        orchestrator.process_ticket.return_value = make_result()

        samples = [
            SampleTicket(
                issue="How do I create a test?",
                subject="Create test",
                expected_status="replied",
                expected_product_area="screen / managing tests",
                expected_request_type="product_issue",
            )
        ]

        report = evaluate_sample_tickets(
            orchestrator,
            samples,
        )

        assert report.total_tickets == 1
        assert report.status_accuracy == 1.0
        assert report.product_area_accuracy == 1.0
        assert report.request_type_accuracy == 1.0
        assert report.overall_exact_match_rate == 1.0

        assert len(report.results) == 1
        assert report.results[0].all_match is True

        orchestrator.process_ticket.assert_called_once()

    def test_mismatch_is_reported(self):
        orchestrator = MagicMock()

        orchestrator.process_ticket.return_value = make_result(
            status="escalated",
            product_area="unknown",
            request_type="bug",
            response="This request has been escalated to a human.",
        )

        samples = [
            SampleTicket(
                issue="Something is broken.",
                subject="Bug",
                expected_status="replied",
                expected_product_area="screen / managing tests",
                expected_request_type="product_issue",
            )
        ]

        report = evaluate_sample_tickets(
            orchestrator,
            samples,
        )

        assert report.total_tickets == 1
        assert report.status_accuracy == 0.0
        assert report.product_area_accuracy == 0.0
        assert report.request_type_accuracy == 0.0
        assert report.overall_exact_match_rate == 0.0

        result = report.results[0]

        assert result.status_match is False
        assert result.product_area_match is False
        assert result.request_type_match is False
        assert result.all_match is False

    def test_multiple_tickets_are_processed(self):
        orchestrator = MagicMock()

        orchestrator.process_ticket.side_effect = [
            make_result(),
            make_result(),
        ]

        samples = [
            SampleTicket(
                issue="First issue",
                subject="First",
                expected_status="replied",
                expected_product_area="screen / managing tests",
                expected_request_type="product_issue",
            ),
            SampleTicket(
                issue="Second issue",
                subject="Second",
                expected_status="replied",
                expected_product_area="screen / managing tests",
                expected_request_type="product_issue",
            ),
        ]

        report = evaluate_sample_tickets(
            orchestrator,
            samples,
        )

        assert report.total_tickets == 2
        assert len(report.results) == 2
        assert orchestrator.process_ticket.call_count == 2


class TestFormatReport:
    def test_format_report_contains_summary(self):
        report = EvaluationReport(
            total_tickets=2,
            status_accuracy=1.0,
            request_type_accuracy=0.5,
            product_area_accuracy=1.0,
            overall_exact_match_rate=0.5,
            results=[],
        )

        output = format_report(report)

        assert "EVALUATION REPORT" in output
        assert "Total tickets evaluated: 2" in output
        assert "Status accuracy:          100.0%" in output
        assert "Request type accuracy:    50.0%" in output
        assert "Product area accuracy:    100.0%" in output
        assert "Overall exact-match rate: 50.0%" in output


class TestRunEvaluation:
    def test_run_evaluation(self, tmp_path):
        path = tmp_path / "golden.csv"

        path.write_text(
            "Issue,Subject,Product Area,Status,Request Type\n"
            "How do I create a test?,Create test,"
            "Screen / Managing Tests,replied,product_issue\n",
            encoding="utf-8",
        )

        orchestrator = MagicMock()
        orchestrator.process_ticket.return_value = make_result()

        report = run_evaluation(
            path,
            orchestrator,
        )

        assert report.total_tickets == 1
        assert report.overall_exact_match_rate == 1.0
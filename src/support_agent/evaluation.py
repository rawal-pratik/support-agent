"""Sample-ticket evaluation and regression tests."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import AgentResult, Ticket
from .orchestrator import TriageOrchestrator


@dataclass
class SampleTicket:
    """A sample ticket with input fields and expected reference outputs."""
    issue: str
    subject: str
    expected_status: str
    expected_product_area: str
    expected_request_type: str


@dataclass
class EvaluationResult:
    """Result of evaluating one sample ticket."""
    ticket_index: int
    issue: str
    expected_status: str
    expected_product_area: str
    expected_request_type: str
    actual_status: str
    actual_product_area: str
    actual_request_type: str
    status_match: bool
    product_area_match: bool
    request_type_match: bool
    all_match: bool
    generated_response: str
    response_checks: dict


@dataclass
class EvaluationReport:
    """Aggregate evaluation report."""
    total_tickets: int
    status_accuracy: float
    request_type_accuracy: float
    product_area_accuracy: float
    overall_exact_match_rate: float
    results: list[EvaluationResult]


def _normalize_product_area(value: str) -> str:
    """Normalize product area for comparison (strip, lowercase)."""
    return value.strip().lower() if value else ""


def _normalize_status(value: str) -> str:
    """Normalize status for comparison (lowercase)."""
    return value.strip().lower() if value else ""


def _normalize_request_type(value: str) -> str:
    """Normalize request type for comparison."""
    return value.strip().lower() if value else ""


def load_sample_tickets(csv_path: str | Path) -> list[SampleTicket]:
    """
    Load sample tickets from CSV with reference outputs.

    Args:
        csv_path: Path to sample_support_tickets.csv

    Returns:
        List of SampleTicket objects with expected outputs
    """
    tickets: list[SampleTicket] = []

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV is missing a header row")

        # Normalize fieldnames (handle BOM)
        reader.fieldnames[0] = reader.fieldnames[0].lstrip("﻿")

        required_cols = {"Issue", "Subject", "Product Area", "Status", "Request Type"}
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            tickets.append(SampleTicket(
                issue=row.get("Issue") or "",
                subject=row.get("Subject") or "",
                expected_status=_normalize_status(row.get("Status") or ""),
                expected_product_area=_normalize_product_area(row.get("Product Area") or ""),
                expected_request_type=_normalize_request_type(row.get("Request Type") or ""),
            ))

    return tickets


def _check_response_quality(result: AgentResult) -> dict:
    """Perform lightweight deterministic checks on generated response."""
    checks = {}

    # Replied results should have non-empty response
    if result.status == "replied":
        checks["non_empty_response"] = bool(result.response.strip())
    else:
        checks["non_empty_response"] = True  # N/A for escalated

    # Escalated results should indicate escalation/human review
    if result.status == "escalated":
        checks["indicates_escalation"] = any(
            word in result.response.lower()
            for word in ["escalat", "human review", "human", "forward"]
        )
    else:
        checks["indicates_escalation"] = True  # N/A for replied

    return checks


def evaluate_sample_tickets(
    orchestrator: TriageOrchestrator,
    sample_tickets: list[SampleTicket],
) -> EvaluationReport:
    """
    Evaluate sample tickets against reference outputs.

    Args:
        orchestrator: The TriageOrchestrator to use for processing
        sample_tickets: List of SampleTicket with expected outputs

    Returns:
        EvaluationReport with metrics and per-ticket results
    """
    results: list[EvaluationResult] = []

    for idx, sample in enumerate(sample_tickets):
        ticket = Ticket(subject=sample.subject, issue=sample.issue)
        generated = orchestrator.process_ticket(ticket)

        # Compare normalized values
        actual_status = _normalize_status(generated.status)
        actual_product_area = _normalize_product_area(generated.product_area)
        actual_request_type = _normalize_request_type(generated.request_type)

        status_match = actual_status == sample.expected_status
        product_area_match = actual_product_area == sample.expected_product_area
        request_type_match = actual_request_type == sample.expected_request_type
        all_match = status_match and product_area_match and request_type_match

        response_checks = _check_response_quality(generated)

        results.append(EvaluationResult(
            ticket_index=idx,
            issue=sample.issue,
            expected_status=sample.expected_status,
            expected_product_area=sample.expected_product_area,
            expected_request_type=sample.expected_request_type,
            actual_status=actual_status,
            actual_product_area=actual_product_area,
            actual_request_type=actual_request_type,
            status_match=status_match,
            product_area_match=product_area_match,
            request_type_match=request_type_match,
            all_match=all_match,
            generated_response=generated.response,
            response_checks=response_checks,
        ))

    # Calculate aggregate metrics
    total = len(results)
    status_accuracy = sum(r.status_match for r in results) / total if total > 0 else 0.0
    request_type_accuracy = sum(r.request_type_match for r in results) / total if total > 0 else 0.0
    product_area_accuracy = sum(r.product_area_match for r in results) / total if total > 0 else 0.0
    overall_exact_match = sum(r.all_match for r in results) / total if total > 0 else 0.0

    return EvaluationReport(
        total_tickets=total,
        status_accuracy=status_accuracy,
        request_type_accuracy=request_type_accuracy,
        product_area_accuracy=product_area_accuracy,
        overall_exact_match_rate=overall_exact_match,
        results=results,
    )


def format_report(report: EvaluationReport) -> str:
    """Format evaluation report as human-readable string."""
    lines = [
        "=" * 60,
        "EVALUATION REPORT",
        "=" * 60,
        f"Total tickets evaluated: {report.total_tickets}",
        "",
        "Aggregate Metrics:",
        f"  Status accuracy:         {report.status_accuracy:.1%}",
        f"  Request type accuracy:   {report.request_type_accuracy:.1%}",
        f"  Product area accuracy:   {report.product_area_accuracy:.1%}",
        f"  Overall exact-match rate: {report.overall_exact_match_rate:.1%}",
        "",
        "Per-ticket Results:",
        "-" * 60,
    ]

    for r in report.results:
        match_symbol = "✓" if r.all_match else "✗"
        lines.append(f"  {match_symbol} Ticket {r.ticket_index + 1}")
        if not r.status_match:
            lines.append(f"    Status:       expected={r.expected_status} actual={r.actual_status}")
        if not r.product_area_match:
            lines.append(f"    Product Area: expected={r.expected_product_area or '(blank)'} actual={r.actual_product_area or '(blank)'}")
        if not r.request_type_match:
            lines.append(f"    Request Type: expected={r.expected_request_type} actual={r.actual_request_type}")
        # Response checks
        if not r.response_checks.get("non_empty_response", True):
            lines.append(f"    Response:     WARNING - Replied ticket has empty response")
        if not r.response_checks.get("indicates_escalation", True):
            lines.append(f"    Response:     WARNING - Escalated ticket doesn't indicate escalation")

    lines.append("=" * 60)
    return "\n".join(lines)


def run_evaluation(
    golden_csv_path: str | Path,
    orchestrator: TriageOrchestrator,
) -> EvaluationReport:
    """
    Main entry point for evaluation.

    Args:
        golden_csv_path: Path to sample_support_tickets.csv
        orchestrator: Configured TriageOrchestrator

    Returns:
        EvaluationReport
    """
    sample_tickets = load_sample_tickets(golden_csv_path)
    return evaluate_sample_tickets(orchestrator, sample_tickets)
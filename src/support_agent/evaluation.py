"""Golden-sample evaluation and regression reporting."""

import csv
from dataclasses import dataclass
from pathlib import Path

from .models import AgentResult, Ticket
from .orchestrator import TriageOrchestrator


@dataclass
class SampleTicket:
    issue: str
    subject: str
    expected_status: str
    expected_product_area: str
    expected_request_type: str


@dataclass
class EvaluationResult:
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
    total_tickets: int
    status_accuracy: float
    request_type_accuracy: float
    product_area_accuracy: float
    overall_exact_match_rate: float
    results: list[EvaluationResult]


def _norm(value: str) -> str:
    return (value or "").strip().lower()


def load_sample_tickets(path: str | Path) -> list[SampleTicket]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Issue", "Subject", "Product Area", "Status", "Request Type"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")

        return [
            SampleTicket(
                issue=row.get("Issue", ""),
                subject=row.get("Subject", ""),
                expected_status=_norm(row.get("Status", "")),
                expected_product_area=_norm(row.get("Product Area", "")),
                expected_request_type=_norm(row.get("Request Type", "")),
            )
            for row in reader
        ]


def _checks(result: AgentResult) -> dict:
    return {
        "non_empty_response": result.status != "replied" or bool((result.response or "").strip()),
        "indicates_escalation": (
            result.status != "escalated"
            or any(x in (result.response or "").lower() for x in ("escalat", "human", "forward"))
        ),
    }


def evaluate_sample_tickets(orchestrator: TriageOrchestrator,
                            samples: list[SampleTicket]) -> EvaluationReport:
    results = []
    for i, sample in enumerate(samples):
        actual = orchestrator.process_ticket(Ticket(subject=sample.subject, issue=sample.issue))
        status = _norm(actual.status)
        area = _norm(actual.product_area)
        kind = _norm(actual.request_type)
        results.append(EvaluationResult(
            ticket_index=i,
            issue=sample.issue,
            expected_status=sample.expected_status,
            expected_product_area=sample.expected_product_area,
            expected_request_type=sample.expected_request_type,
            actual_status=status,
            actual_product_area=area,
            actual_request_type=kind,
            status_match=status == sample.expected_status,
            product_area_match=area == sample.expected_product_area,
            request_type_match=kind == sample.expected_request_type,
            all_match=(
                status == sample.expected_status
                and area == sample.expected_product_area
                and kind == sample.expected_request_type
            ),
            generated_response=actual.response,
            response_checks=_checks(actual),
        ))

    total = len(results) or 1
    return EvaluationReport(
        total_tickets=len(results),
        status_accuracy=sum(r.status_match for r in results) / total,
        request_type_accuracy=sum(r.request_type_match for r in results) / total,
        product_area_accuracy=sum(r.product_area_match for r in results) / total,
        overall_exact_match_rate=sum(r.all_match for r in results) / total,
        results=results,
    )


def format_report(report: EvaluationReport) -> str:
    lines = [
        "=" * 60,
        "EVALUATION REPORT",
        "=" * 60,
        f"Total tickets evaluated: {report.total_tickets}",
        "",
        f"Status accuracy:          {report.status_accuracy:.1%}",
        f"Request type accuracy:    {report.request_type_accuracy:.1%}",
        f"Product area accuracy:    {report.product_area_accuracy:.1%}",
        f"Overall exact-match rate: {report.overall_exact_match_rate:.1%}",
        "",
        "Per-ticket results:",
    ]
    for r in report.results:
        mark = "✓" if r.all_match else "✗"
        lines.append(f"  {mark} Ticket {r.ticket_index + 1}")
        if not r.status_match:
            lines.append(f"    Status: expected={r.expected_status} actual={r.actual_status}")
        if not r.product_area_match:
            lines.append(f"    Product area: expected={r.expected_product_area or '(blank)'} actual={r.actual_product_area or '(blank)'}")
        if not r.request_type_match:
            lines.append(f"    Request type: expected={r.expected_request_type} actual={r.actual_request_type}")
        for name, ok in r.response_checks.items():
            if not ok:
                lines.append(f"    WARNING: {name}")
    return "\n".join(lines) + "\n" + "=" * 60


def run_evaluation(golden_csv_path: str | Path,
                   orchestrator: TriageOrchestrator) -> EvaluationReport:
    return evaluate_sample_tickets(orchestrator, load_sample_tickets(golden_csv_path))
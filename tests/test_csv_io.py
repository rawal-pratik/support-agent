import csv
import io
from pathlib import Path

import pytest

from support_agent.csv_io import OUTPUT_COLUMNS, read_tickets, write_results
from support_agent.models import AgentResult

SAMPLES_DIR = Path(__file__).parents[1] / "public" / "samples"


def test_reads_the_real_target_csv_schema_and_all_rows_in_order():
    tickets = read_tickets(SAMPLES_DIR / "support_tickets.csv")

    assert len(tickets) == 16
    assert tickets[0].issue.startswith("I completed a HackerRank test")
    assert tickets[0].subject == "Test Score Dispute"
    assert tickets[-1].issue.startswith("one of my employee")
    assert tickets[-1].subject == "Employee leaving the company"


def test_reads_the_real_sample_csv_input_columns():
    tickets = read_tickets(SAMPLES_DIR / "sample_support_tickets.csv")

    assert tickets[0].issue.startswith("I notice that people")
    assert tickets[0].subject == "Test Active in the system"
    assert tickets[1].subject == ""


def test_supports_bom_quoting_and_multiline_fields():
    source = io.StringIO(
        '\ufeffIssue,Subject\n"First line\nSecond line","Quoted, subject"\n'
    )

    tickets = read_tickets(source)

    assert [ticket.model_dump() for ticket in tickets] == [
        {
            "issue": "First line\nSecond line",
            "subject": "Quoted, subject",
        }
    ]


def test_rejects_csv_without_required_columns():
    with pytest.raises(ValueError, match=r"missing required column\(s\): Issue"):
        read_tickets(io.StringIO("Subject\nOnly subject\n"))


def test_writes_required_columns_and_preserves_result_order():
    results = [
        AgentResult(
            status="replied",
            product_area="screen",
            response="First response",
            justification="First reason",
            request_type="product_issue",
        ),
        AgentResult(
            status="escalated",
            product_area="",
            response="Second response",
            justification="Second reason",
            request_type="bug",
        ),
    ]
    destination = io.StringIO()

    write_results(destination, results)

    destination.seek(0)
    rows = list(csv.DictReader(destination))
    assert list(rows[0]) == list(OUTPUT_COLUMNS)
    assert [row["response"] for row in rows] == ["First response", "Second response"]

import csv
from pathlib import Path
from typing import Iterable, TextIO

from .models import AgentResult, Ticket

INPUT_COLUMNS = ("Issue", "Subject")
OUTPUT_COLUMNS = (
    "status",
    "product_area",
    "response",
    "justification",
    "request_type",
)


def read_tickets(source: str | Path | TextIO) -> list[Ticket]:
    """Read challenge CSV rows into tickets while preserving their order."""

    if hasattr(source, "read"):
        return _read_ticket_rows(source)

    with Path(source).open("r", encoding="utf-8-sig", newline="") as file:
        return _read_ticket_rows(file)


def _read_ticket_rows(file: TextIO) -> list[Ticket]:
    reader = csv.DictReader(file)
    if reader.fieldnames is None:
        raise ValueError("Ticket CSV is missing a header row")
    reader.fieldnames[0] = reader.fieldnames[0].lstrip("\ufeff")

    missing_columns = [column for column in INPUT_COLUMNS if column not in reader.fieldnames]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Ticket CSV is missing required column(s): {missing}")

    tickets: list[Ticket] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            tickets.append(
                Ticket(
                    issue=row.get("Issue") or "",
                    subject=row.get("Subject") or "",
                )
            )
        except ValueError as error:
            raise ValueError(f"Invalid ticket row {row_number}: {error}") from error

    return tickets


def write_results(destination: str | Path | TextIO, results: Iterable[AgentResult]) -> None:
    """Write agent results using the required five output columns."""

    if hasattr(destination, "write"):
        _write_result_rows(destination, results)
        return

    with Path(destination).open("w", encoding="utf-8", newline="") as file:
        _write_result_rows(file, results)


def _write_result_rows(file: TextIO, results: Iterable[AgentResult]) -> None:
    writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for result in results:
        writer.writerow(result.model_dump())

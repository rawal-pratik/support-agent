"""Focused tests for the terminal CLI."""

import csv
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from support_agent.cli import app
from support_agent.models import AgentResult
from support_agent.retrieval import RetrievedChunk


runner = CliRunner()


def make_chunk():
    return RetrievedChunk(
        chunk_id="doc.md#chunk-0",
        source_path="doc.md",
        title="Test Documentation",
        category="Screen / Managing Tests",
        text="Documentation content.",
        similarity=0.85,
    )


def make_result(
    status="replied",
    response="Based on the documentation, do this.",
):
    return AgentResult(
        status=status,
        product_area="Screen / Managing Tests",
        response=response,
        justification="Document 1 supports this answer.",
        request_type="product_issue",
    )


class TestHelp:
    def test_root_help(self):
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "support-agent" in result.output.lower()

    def test_commands_have_help(self):
        for command in ("ingest", "search", "triage", "evaluate"):
            result = runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0


class TestIngest:
    def test_ingest(self):
        ingest_result = MagicMock(
            document_count=10,
            chunk_count=50,
            inserted_count=45,
            skipped_count=5,
        )

        with (
            patch("support_agent.cli.DocumentChunkRepository"),
            patch("support_agent.cli.GeminiEmbeddingProvider"),
            patch(
                "support_agent.cli.ingest_corpus",
                return_value=ingest_result,
            ) as ingest,
        ):
            result = runner.invoke(app, ["ingest"])

        assert result.exit_code == 0
        assert "Documents processed: 10" in result.output
        assert "Chunks created: 50" in result.output
        assert "Chunks inserted: 45" in result.output
        assert "Chunks skipped: 5" in result.output
        ingest.assert_called_once()

    def test_ingest_options(self):
        ingest_result = MagicMock(
            document_count=1,
            chunk_count=2,
            inserted_count=2,
            skipped_count=0,
        )

        with (
            patch("support_agent.cli.DocumentChunkRepository"),
            patch("support_agent.cli.GeminiEmbeddingProvider"),
            patch(
                "support_agent.cli.ingest_corpus",
                return_value=ingest_result,
            ) as ingest,
        ):
            result = runner.invoke(
                app,
                [
                    "ingest",
                    "--chunk-size", "300",
                    "--overlap", "25",
                    "--max-chunks", "100",
                    "--no-clear",
                ],
            )

        assert result.exit_code == 0

        kwargs = ingest.call_args.kwargs
        assert kwargs["chunk_size"] == 300
        assert kwargs["overlap"] == 25
        assert kwargs["max_chunks"] == 100
        assert kwargs["clear_existing"] is False


class TestSearch:
    def test_search(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [make_chunk()]

        with patch(
            "support_agent.cli.SemanticRetriever",
            return_value=retriever,
        ):
            result = runner.invoke(
                app,
                ["search", "how do I create a test"],
            )

        assert result.exit_code == 0
        assert "Test Documentation" in result.output
        assert "Screen / Managing Tests" in result.output
        assert "0.8500" in result.output

        ticket = retriever.retrieve.call_args.args[0]
        assert ticket.issue == "how do I create a test"

    def test_search_no_results(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []

        with patch(
            "support_agent.cli.SemanticRetriever",
            return_value=retriever,
        ):
            result = runner.invoke(
                app,
                ["search", "query"],
            )

        assert result.exit_code == 0
        assert "No results found" in result.output

    def test_search_empty_query(self):
        result = runner.invoke(app, ["search", ""])

        assert result.exit_code != 0

    def test_search_rejects_invalid_candidate_k(self):
        result = runner.invoke(
            app,
            [
                "search",
                "query",
                "--top-k", "10",
                "--candidate-k", "5",
            ],
        )

        assert result.exit_code != 0


class TestTriage:
    def _write_input(self, path, rows):
        with path.open("w", encoding="utf-8", newline="") as file:
            file.write("Issue,Subject\n")
            for issue, subject in rows:
                file.write(f"{issue},{subject}\n")

    def test_triage_writes_results(self, tmp_path):
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        self._write_input(
            input_csv,
            [
                ("Issue A", "Subject A"),
                ("Issue B", "Subject B"),
            ],
        )

        orchestrator = MagicMock()
        orchestrator.process_ticket.side_effect = [
            make_result(response="Answer A"),
            make_result(response="Answer B"),
        ]

        with patch(
            "support_agent.cli.TriageOrchestrator",
            return_value=orchestrator,
        ):
            result = runner.invoke(
                app,
                ["triage", str(input_csv), str(output_csv)],
            )

        assert result.exit_code == 0
        assert "Processed 2 input tickets" in result.output
        assert "Replied: 2" in result.output
        assert "Escalated: 0" in result.output

        with output_csv.open(
            encoding="utf-8",
            newline="",
        ) as file:
            rows = list(csv.DictReader(file))

        assert len(rows) == 2
        assert rows[0]["response"] == "Answer A"
        assert rows[1]["response"] == "Answer B"

    def test_triage_counts_replies_and_escalations(self, tmp_path):
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        self._write_input(
            input_csv,
            [
                ("Issue A", "Subject A"),
                ("Issue B", "Subject B"),
                ("Issue C", "Subject C"),
            ],
        )

        orchestrator = MagicMock()
        orchestrator.process_ticket.side_effect = [
            make_result(status="replied"),
            make_result(status="replied"),
            make_result(status="escalated"),
        ]

        with patch(
            "support_agent.cli.TriageOrchestrator",
            return_value=orchestrator,
        ):
            result = runner.invoke(
                app,
                ["triage", str(input_csv), str(output_csv)],
            )

        assert result.exit_code == 0
        assert "Processed 3 input tickets" in result.output
        assert "Replied: 2" in result.output
        assert "Escalated: 1" in result.output
        assert orchestrator.process_ticket.call_count == 3

    def test_triage_preserves_order(self, tmp_path):
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        self._write_input(
            input_csv,
            [
                ("Issue A", "Subject A"),
                ("Issue B", "Subject B"),
            ],
        )

        orchestrator = MagicMock()
        orchestrator.process_ticket.side_effect = [
            make_result(response="Answer A"),
            make_result(response="Answer B"),
        ]

        with patch(
            "support_agent.cli.TriageOrchestrator",
            return_value=orchestrator,
        ):
            result = runner.invoke(
                app,
                ["triage", str(input_csv), str(output_csv)],
            )

        assert result.exit_code == 0

        calls = orchestrator.process_ticket.call_args_list

        assert calls[0].args[0].subject == "Subject A"
        assert calls[0].args[0].issue == "Issue A"
        assert calls[1].args[0].subject == "Subject B"
        assert calls[1].args[0].issue == "Issue B"

    def test_triage_rejects_invalid_candidate_k(self, tmp_path):
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        self._write_input(
            input_csv,
            [("Issue", "Subject")],
        )

        result = runner.invoke(
            app,
            [
                "triage",
                str(input_csv),
                str(output_csv),
                "--top-k", "10",
                "--candidate-k", "5",
            ],
        )

        assert result.exit_code != 0


class TestEvaluate:
    def test_evaluate_requires_golden(self):
        result = runner.invoke(app, ["evaluate"])

        assert result.exit_code != 0

    def test_evaluate(self, tmp_path):
        golden_csv = tmp_path / "golden.csv"

        golden_csv.write_text(
            "status,product_area,response,justification,request_type\n"
            "replied,Screen,Answer,Reason,product_issue\n",
            encoding="utf-8",
        )

        with (
            patch("support_agent.cli._services") as services,
            patch("support_agent.cli.TriageOrchestrator"),
            patch(
                "support_agent.cli.run_evaluation",
                return_value=MagicMock(),
            ) as evaluate,
            patch(
                "support_agent.cli.format_report",
                return_value="Evaluation report",
            ),
        ):
            services.return_value = (
                MagicMock(),
                MagicMock(),
            )

            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--golden",
                    str(golden_csv),
                ],
            )

        assert result.exit_code == 0
        assert "Evaluation report" in result.output
        evaluate.assert_called_once()
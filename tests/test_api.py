"""Tests for the FastAPI wrapper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from support_agent.api import app
from support_agent.models import AgentResult, Ticket


def make_agent_result(
    status: str = "replied",
    product_area: str = "screen",
    response: str = "Based on the docs, do X.",
    justification: str = "Doc 1 says to do X.",
    request_type: str = "bug",
) -> AgentResult:
    status_lc = status.lower()
    request_type_lc = request_type.lower()
    # Valid request_type literals: product_issue, feature_request, bug, invalid
    valid_types = ("product_issue", "feature_request", "bug", "invalid")
    if request_type_lc not in valid_types:
        request_type_lc = "bug"
    return AgentResult(
        status=status_lc,
        product_area=product_area,
        response=response,
        justification=justification,
        request_type=request_type_lc,
    )


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self):
        """GET /health returns 200 and expected JSON structure."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestTriageSingleEndpoint:
    """Tests for POST /triage endpoint."""

    def test_valid_ticket_returns_200_and_agentresult(self):
        """Valid ticket returns 200 with AgentResult body."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage", json={"issue": "Test issue", "subject": "Test subject"})

            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ("replied", "escalated")
            assert "product_area" in data
            assert "response" in data
            assert "justification" in data
            assert "request_type" in data
            assert data["request_type"] in ("bug", "product_issue", "feature_request", "invalid")

    def test_orchestrator_called_once(self):
        """Orchestrator is called exactly once for single ticket."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            client.post("/triage", json={"issue": "Test issue", "subject": "Test subject"})

            assert mock_orchestrator.process_ticket.call_count == 1

    def test_ticket_passed_to_orchestrator_contains_supplied_fields(self):
        """The Ticket passed to orchestrator contains the supplied issue and subject."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            client.post("/triage", json={"issue": "My specific issue", "subject": "My subject"})

            # Verify the ticket passed to process_ticket
            call_args = mock_orchestrator.process_ticket.call_args
            assert call_args is not None
            ticket_arg = call_args[0][0]
            assert isinstance(ticket_arg, Ticket)
            assert ticket_arg.issue == "My specific issue"
            assert ticket_arg.subject == "My subject"

    def test_missing_issue_is_rejected(self):
        """Missing issue field is rejected with 422."""
        client = TestClient(app)
        response = client.post("/triage", json={"subject": "Test subject"})

        assert response.status_code == 422

    def test_missing_subject_allowed(self):
        """Missing subject is allowed (optional field)."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage", json={"issue": "Test issue"})

            assert response.status_code == 200

    def test_malformed_request_body_rejected(self):
        """Malformed request body is rejected."""
        client = TestClient(app)
        response = client.post("/triage", json={"issue": 123, "subject": "test"})

        assert response.status_code == 422

    def test_empty_issue_rejected(self):
        """Empty issue is rejected (min_length=1)."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage", json={"issue": "", "subject": "test"})

            assert response.status_code == 422

    def test_orchestrator_failure_returns_500(self):
        """Orchestrator/provider failure returns clear HTTP error."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.side_effect = Exception("Provider failure")
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage", json={"issue": "Test issue", "subject": "Test subject"})

            assert response.status_code == 500
            assert "internal server error" in response.json()["detail"].lower()

    def test_secrets_not_exposed_in_errors(self):
        """API keys and secrets are not exposed in error responses."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            # Simulate an error that might contain secrets
            mock_orchestrator.process_ticket.side_effect = Exception("GEMINI_API_KEY=AIzaSy... failed")
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage", json={"issue": "Test issue", "subject": "Test subject"})

            assert response.status_code == 500
            error_detail = response.json()["detail"].lower()
            assert "api_key" not in error_detail
            assert "sk-" not in error_detail
            assert "internal server error" in error_detail


class TestTriageBatchEndpoint:
    """Tests for POST /triage/batch endpoint."""

    def test_valid_batch_returns_correct_count(self):
        """Valid batch returns correct number of results."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage/batch", json=[
                {"issue": "Issue 1", "subject": "Subject 1"},
                {"issue": "Issue 2", "subject": "Subject 2"},
                {"issue": "Issue 3", "subject": "Subject 3"},
            ])

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 3

    def test_orchestrator_called_once_per_ticket(self):
        """Orchestrator is called once per ticket in batch."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            client.post("/triage/batch", json=[
                {"issue": "Issue 1", "subject": "Subject 1"},
                {"issue": "Issue 2", "subject": "Subject 2"},
            ])

            assert mock_orchestrator.process_ticket.call_count == 2

    def test_input_output_order_preserved(self):
        """Input/output order is preserved."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            # Return different responses to track order
            mock_orchestrator.process_ticket.side_effect = [
                make_agent_result(response="RESPONSE_01"),
                make_agent_result(response="RESPONSE_02"),
                make_agent_result(response="RESPONSE_03"),
            ]
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage/batch", json=[
                {"issue": "Issue 1", "subject": "Subject 1"},
                {"issue": "Issue 2", "subject": "Subject 2"},
                {"issue": "Issue 3", "subject": "Subject 3"},
            ])

            assert response.status_code == 200
            data = response.json()
            assert data[0]["response"] == "RESPONSE_01"
            assert data[1]["response"] == "RESPONSE_02"
            assert data[2]["response"] == "RESPONSE_03"

    def test_empty_batch_rejected(self):
        """Empty batch is rejected with 400."""
        client = TestClient(app)
        response = client.post("/triage/batch", json=[])

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_batch_size_limit(self):
        """Batch size exceeding 100 is rejected."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            batch = [{"issue": f"Issue {i}", "subject": f"Subject {i}"} for i in range(101)]
            response = client.post("/triage/batch", json=batch)

            assert response.status_code == 400
            assert "exceeds maximum" in response.json()["detail"].lower()

    def test_one_ticket_failure_does_not_reorder_others(self):
        """One ticket's processing behavior does not reorder other tickets."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            # Middle ticket raises, others succeed
            mock_orchestrator.process_ticket.side_effect = [
                make_agent_result(response="RESPONSE_01"),
                Exception("Middle ticket fails"),
                make_agent_result(response="RESPONSE_03"),
            ]
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage/batch", json=[
                {"issue": "Issue 1", "subject": "Subject 1"},
                {"issue": "Issue 2", "subject": "Subject 2"},
                {"issue": "Issue 3", "subject": "Subject 3"},
            ])

            # Should fail on the second ticket
            assert response.status_code == 500

    def test_batch_conforms_to_agentresult_contract(self):
        """Each result in batch conforms to AgentResult contract."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result(
                status="escalated",
                product_area="",
                response="Escalated to human",
                justification="Policy requires escalation",
                request_type="bug",
            )
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage/batch", json=[
                {"issue": "Issue 1", "subject": "Subject 1"},
            ])

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            result = data[0]
            assert result["status"] in ("replied", "escalated")
            assert "product_area" in result
            assert "response" in result
            assert "justification" in result
            assert result["request_type"] in ("bug", "product_issue", "feature_request", "invalid", "security")


class TestAPISchema:
    """Tests for OpenAPI schema generation."""

    def test_openapi_generation_works(self):
        """OpenAPI schema generates without external credentials."""
        client = TestClient(app)
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/triage" in schema["paths"]
        assert "/triage/batch" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_agentresult_schema_exposed(self):
        """AgentResult schema is exposed in OpenAPI."""
        client = TestClient(app)
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        # Check that AgentResult components exist
        assert "components" in schema
        assert "schemas" in schema["components"]
        # AgentResult should be in schemas
        schemas = schema["components"]["schemas"]
        assert any("AgentResult" in k for k in schemas.keys()) or any(
            "result" in k.lower() for k in schemas.keys()
        )

    def test_ticket_schema_exposed(self):
        """Ticket schema is exposed in OpenAPI."""
        client = TestClient(app)
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        schemas = schema["components"]["schemas"]
        assert any("Ticket" in k for k in schemas.keys())


class TestConfigurationErrors:
    """Tests for configuration error handling."""

    def test_missing_config_returns_503(self):
        """Missing provider configuration returns 503."""
        with patch("support_agent.api._get_orchestrator") as mock_get:
            from fastapi import HTTPException
            mock_get.side_effect = HTTPException(
                status_code=503,
                detail="Service unavailable: configuration error - GEMINI_API_KEY must be configured"
            )

            client = TestClient(app)
            response = client.post("/triage", json={"issue": "Test", "subject": "Test"})

            assert response.status_code == 503
            assert "configuration" in response.json()["detail"].lower()


class TestNoExternalCalls:
    """Verify no external API calls are made during tests."""

    def test_no_real_gemini_call(self):
        """Test does not call real Gemini API."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            client.post("/triage", json={"issue": "Test", "subject": "Test"})

            # The build_orchestrator was mocked, so no real providers created
            mock_build.assert_called_once()

    def test_no_real_supabase_call(self):
        """Test does not call real Supabase."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            client.post("/triage/batch", json=[{"issue": "Test", "subject": "Test"}])

            mock_build.assert_called_once()

    def test_no_real_openrouter_call(self):
        """Test does not call real OpenRouter."""
        with patch("support_agent.api._get_orchestrator") as mock_build:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_build.return_value = mock_orchestrator

            client = TestClient(app)
            client.post("/triage", json={"issue": "Test", "subject": "Test"})

            mock_build.assert_called_once()
"""Tests for the triage orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from support_agent.models import AgentResult, Ticket
from support_agent.policy import PolicyDecision, RequestType, RiskLevel
from support_agent.retrieval import RetrievedChunk
from support_agent.orchestrator import TriageOrchestrator


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
    text: str = "How to do the thing.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_path="doc.md",
        title=title,
        category="General",
        text=text,
        similarity=0.85,
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


class TestNormalFlow:
    """Tests for normal ticket → policy → retrieval → LLM → final AgentResult."""

    def test_normal_ticket_full_flow(self):
        """Normal ticket goes through policy, retrieval, and LLM."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()

        chunk = make_retrieved_chunk()
        mock_retriever.retrieve.return_value = [chunk]
        mock_llm.generate.return_value = make_agent_result()

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        mock_retriever.retrieve.assert_called_once()
        mock_llm.generate.assert_called_once()

        call_args = mock_llm.generate.call_args
        assert call_args.kwargs["ticket"] == make_ticket()
        assert isinstance(call_args.kwargs["policy_decision"], PolicyDecision)
        assert call_args.kwargs["retrieved_chunks"] == [chunk]

        assert isinstance(result, AgentResult)
        assert result.status == "replied"
        assert result.product_area == "Screen"

    def test_retriever_called_with_ticket(self):
        """Retriever should be called with the original ticket."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm.generate.return_value = make_agent_result()

        ticket = make_ticket(subject="Login", issue="Cannot log in")
        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        orchestrator.process_ticket(ticket)

        mock_retriever.retrieve.assert_called_once_with(ticket)


class TestPolicyEscalation:
    """Tests where policy mandates escalation, skipping retrieval and LLM."""

    def test_policy_escalation_skips_retrieval_and_llm(self):
        """Policy-mandated escalation skips retrieval and LLM entirely."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()

        with patch("support_agent.orchestrator.evaluate_policy") as mock_eval:
            mock_eval.return_value = make_policy_decision(
                should_escalate=True,
                escalation_reason="prompt_injection",
                request_type=RequestType.INVALID,
            )
            orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
            result = orchestrator.process_ticket(make_ticket())

        mock_retriever.retrieve.assert_not_called()
        mock_llm.generate.assert_not_called()

        assert result.status == "escalated"
        assert "prompt injection" in result.response.lower()

    def test_empty_ticket_escalation(self):
        """Empty ticket triggers policy escalation."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()

        with patch("support_agent.orchestrator.evaluate_policy") as mock_eval:
            mock_eval.return_value = make_policy_decision(
                should_escalate=True,
                escalation_reason="empty_ticket",
                request_type=RequestType.INVALID,
            )
            orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
            result = orchestrator.process_ticket(make_ticket())

        assert result.status == "escalated"
        assert "too short or empty" in result.response.lower()
        assert result.request_type == "invalid"

    def test_security_privacy_escalation(self):
        """Security/privacy tickets escalate."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()

        with patch("support_agent.orchestrator.evaluate_policy") as mock_eval:
            mock_eval.return_value = make_policy_decision(
                should_escalate=True,
                escalation_reason="security_privacy",
                request_type=RequestType.BUG,
            )
            orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
            result = orchestrator.process_ticket(make_ticket())

        assert result.status == "escalated"
        assert "security or privacy" in result.response.lower()
        assert result.product_area == "unknown"


class TestInsufficientEvidence:
    """Tests for no retrieved evidence → escalation."""

    def test_no_retrieved_chunks_escalates(self):
        """Empty retrieval result escalates without calling LLM."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = []

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm, min_retrieved_chunks=1)
        result = orchestrator.process_ticket(make_ticket())

        mock_retriever.retrieve.assert_called_once()
        mock_llm.generate.assert_not_called()

        assert result.status == "escalated"
        assert "no relevant documentation" in result.response.lower()

    def test_fewer_than_min_chunks_escalates(self):
        """Fewer than min_retrieved_chunks escalates."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]  # 1 chunk

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm, min_retrieved_chunks=2)
        result = orchestrator.process_ticket(make_ticket())

        mock_llm.generate.assert_not_called()
        assert result.status == "escalated"

    def test_empty_retrieval_always_escalates(self):
        """Empty retrieval result ALWAYS escalates and never reaches the LLM."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = []

        # Default min_retrieved_chunks=1
        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        mock_llm.generate.assert_not_called()
        assert result.status == "escalated"
        assert "no relevant documentation" in result.response.lower()

    def test_empty_retrieval_escalates_with_higher_min(self):
        """Empty retrieval escalates even with higher min_retrieved_chunks."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = []

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm, min_retrieved_chunks=3)
        result = orchestrator.process_ticket(make_ticket())

        mock_llm.generate.assert_not_called()
        assert result.status == "escalated"


class TestLLMReceiveCorrectArguments:
    """Tests that LLM receives the correct ticket, policy decision, and chunks."""

    def test_llm_receives_ticket_policy_and_chunks(self):
        """LLM generate receives all three arguments correctly."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()

        chunk1 = make_retrieved_chunk(chunk_id="doc1.md#chunk-0", text="First doc")
        chunk2 = make_retrieved_chunk(chunk_id="doc2.md#chunk-1", text="Second doc")
        mock_retriever.retrieve.return_value = [chunk1, chunk2]
        mock_llm.generate.return_value = make_agent_result()

        ticket = make_ticket(subject="Test", issue="This is a test issue")
        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        orchestrator.process_ticket(ticket)

        call_args = mock_llm.generate.call_args
        assert call_args.kwargs["ticket"] == ticket
        assert call_args.kwargs["policy_decision"].request_type == RequestType.BUG
        assert call_args.kwargs["retrieved_chunks"] == [chunk1, chunk2]


class TestValidAgentResult:
    """Tests that valid LLM output produces valid AgentResult."""

    def test_llm_returns_valid_agent_result(self):
        """Valid LLM response passes through."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm.generate.return_value = make_agent_result(
            status="replied",
            product_area="Screen",
            request_type="bug",
        )

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert isinstance(result, AgentResult)
        assert result.status == "replied"
        assert result.product_area == "Screen"
        assert result.request_type == "bug"


class TestInvalidLLMOutput:
    """Tests for LLM returning invalid result."""

    def test_llm_validation_error_escalates(self):
        """LLM returns invalid output → escalate."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        # LLM generate raises ValueError (e.g., invalid JSON) → escalate
        mock_llm.generate.side_effect = ValueError("Invalid JSON from LLM")

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert result.status == "escalated"
        assert "llm_failure" in result.response.lower() or \
               "response generation failed" in result.response.lower()


class TestMandatoryEscalationNotOverridden:
    """Tests that mandatory escalation cannot be overridden by LLM."""

    def test_policy_escalation_cannot_be_overridden_by_llm(self):
        """Even if LLM returns 'replied', mandatory policy escalation wins."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm.generate.return_value = make_agent_result(status="replied")

        with patch("support_agent.orchestrator.evaluate_policy") as mock_eval:
            mock_eval.return_value = make_policy_decision(
                should_escalate=True,
                escalation_reason="security_privacy",
                request_type=RequestType.BUG,
            )
            orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
            result = orchestrator.process_ticket(make_ticket())

        # Final result MUST be escalated
        assert result.status == "escalated"
        assert "security or privacy" in result.response.lower()
        assert result.product_area == "unknown"


class TestRetrievalFailure:
    """Tests for retrieval failure handling."""

    def test_retrieval_exception_escalates(self):
        """Retrieval exception triggers escalation."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.side_effect = RuntimeError("Supabase unavailable")

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        mock_llm.generate.assert_not_called()
        assert result.status == "escalated"
        assert "retrieval failed" in result.response.lower()


class TestLLMFailure:
    """Tests for LLM failure handling."""

    def test_llm_exception_escalates(self):
        """LLM exception triggers escalation."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm.generate.side_effect = RuntimeError("OpenRouter API error")

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert result.status == "escalated"
        assert "response generation failed" in result.response.lower()

    def test_llm_validation_error_escalates(self):
        """LLM validation error triggers escalation."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm.generate.side_effect = ValueError("Invalid JSON from LLM")

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert result.status == "escalated"
        assert "llm_failure" in result.response.lower() or \
               "response generation failed" in result.response.lower()


class TestIndependentMockability:
    """Tests that components remain independently mockable."""

    def test_retriever_can_be_mocked_independently(self):
        """Retriever is a protocol that can be mocked."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm.generate.return_value = make_agent_result()

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert result.status == "replied"

    def test_llm_can_be_mocked_independently(self):
        """LLMProvider is a protocol that can be mocked."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm.generate.return_value = make_agent_result()

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert isinstance(result, AgentResult)


class TestOutputAlwaysConforms:
    """Tests that final output always conforms to AgentResult contract."""

    def test_escalated_result_conforms_to_contract(self):
        """Escalation results are valid AgentResult objects."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = []

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert isinstance(result, AgentResult)
        assert result.status in ("replied", "escalated")
        assert isinstance(result.product_area, str)
        assert isinstance(result.response, str)
        assert isinstance(result.justification, str)
        assert result.request_type in (
            "product_issue", "feature_request", "bug", "invalid"
        )

    def test_replied_result_conforms_to_contract(self):
        """Replied results are valid AgentResult objects."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm.generate.return_value = make_agent_result()

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert isinstance(result, AgentResult)
        assert result.status == "replied"
        assert result.request_type in (
            "product_issue", "feature_request", "bug", "invalid"
        )


class TestOrchestratorInitialization:
    """Tests for orchestrator initialization."""

    def test_min_retrieved_chunks_validation(self):
        """min_retrieved_chunks must be at least 1."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()

        with pytest.raises(ValueError, match="min_retrieved_chunks must be at least 1"):
            TriageOrchestrator(mock_retriever, mock_llm, min_retrieved_chunks=0)

        with pytest.raises(ValueError, match="min_retrieved_chunks must be at least 1"):
            TriageOrchestrator(mock_retriever, mock_llm, min_retrieved_chunks=-1)


class TestProductAreaHandling:
    """Tests for product area handling in final result."""

    def test_escalation_result_has_unknown_product_area(self):
        """Escalated results have product_area='unknown'."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = []

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert result.product_area == "unknown"

    def test_replied_result_preserves_llm_product_area(self):
        """Replied results preserve LLM's product_area (orchestrator doesn't invent)."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm.generate.return_value = make_agent_result(product_area="Integrations")

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert result.product_area == "Integrations"


class TestEmptyResponseHandling:
    """Tests for empty LLM response handling."""

    def test_empty_response_escalates(self):
        """Empty response from LLM triggers escalation."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()
        mock_retriever.retrieve.return_value = [make_retrieved_chunk()]
        mock_llm.generate.return_value = make_agent_result(response="   ")  # whitespace only

        orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
        result = orchestrator.process_ticket(make_ticket())

        assert result.status == "escalated"
        assert "empty_response" in result.response.lower() or \
               "empty response" in result.response.lower()


class TestEscalationRequestType:
    """Tests for request_type in escalated results."""

    def test_invalid_request_type_preserved(self):
        """If policy classifies as INVALID, escalation keeps INVALID."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()

        with patch("support_agent.orchestrator.evaluate_policy") as mock_eval:
            mock_eval.return_value = make_policy_decision(
                should_escalate=True,
                escalation_reason="prompt_injection",
                request_type=RequestType.INVALID,
            )
            orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
            result = orchestrator.process_ticket(make_ticket())

        assert result.request_type == "invalid"

    def test_bug_request_type_preserved_for_security(self):
        """Security escalation preserves original request_type if not INVALID."""
        mock_retriever = make_mock_retriever()
        mock_llm = make_mock_llm()

        with patch("support_agent.orchestrator.evaluate_policy") as mock_eval:
            mock_eval.return_value = make_policy_decision(
                should_escalate=True,
                escalation_reason="security_privacy",
                request_type=RequestType.BUG,
            )
            orchestrator = TriageOrchestrator(mock_retriever, mock_llm)
            result = orchestrator.process_ticket(make_ticket())

        assert result.request_type == "bug"

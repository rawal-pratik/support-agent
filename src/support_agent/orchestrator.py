"""Triage orchestration: connect policy, retrieval, and LLM into a single-ticket workflow."""

from typing import Protocol

from .models import AgentResult, Ticket
from .policy import PolicyDecision, evaluate_policy
from .retrieval import RetrievedChunk, Retriever
from .llm import LLMProvider


class TriageOrchestrator:
    """Orchestrates the end-to-end support triage workflow for a single ticket."""

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        min_retrieved_chunks: int = 1,
    ):
        """
        Args:
            retriever: Retrieves relevant documentation chunks for a ticket
            llm_provider: Generates structured responses from the LLM
            min_retrieved_chunks: Minimum number of chunks required to proceed to LLM (must be >= 1)
        """
        if min_retrieved_chunks < 1:
            raise ValueError("min_retrieved_chunks must be at least 1")

        self._retriever = retriever
        self._llm_provider = llm_provider
        self._min_retrieved_chunks = min_retrieved_chunks

    def process_ticket(self, ticket: Ticket) -> AgentResult:
        """
        Process a single support ticket through the triage workflow.

        Flow:
        1. Policy evaluation — if mandatory escalation, return escalated result immediately
        2. Retrieval — if insufficient evidence, return escalated result
        3. LLM generation — produce structured response
        """

        # Step 1: Policy evaluation
        policy_decision = evaluate_policy(ticket)

        # Step 1a: If policy mandates escalation, return escalated result (no LLM)
        if policy_decision.should_escalate:
            return self._make_escalated_result(
                ticket=ticket,
                policy_decision=policy_decision,
                reason=policy_decision.escalation_reason,
            )

        # Step 2: Retrieval
        try:
            retrieved_chunks = self._retriever.retrieve(ticket)
        except Exception as exc:
            # Print the underlying retrieval error while preserving escalation behavior.
            print(
                f"RETRIEVAL ERROR: {type(exc).__name__}: {exc}"
            )

            return self._make_escalated_result(
                ticket=ticket,
                policy_decision=policy_decision,
                reason="retrieval_failure",
            )

        # Step 2a: Insufficient evidence → escalate
        if len(retrieved_chunks) < self._min_retrieved_chunks:
            return self._make_escalated_result(
                ticket=ticket,
                policy_decision=policy_decision,
                reason="insufficient_evidence",
            )

        # Step 3: LLM generation
        try:
            agent_result = self._llm_provider.generate(
                ticket=ticket,
                policy_decision=policy_decision,
                retrieved_chunks=retrieved_chunks,
            )
        except Exception as exc:
            # Print the underlying LLM error while preserving escalation behavior.
            print(
                f"LLM ERROR: {type(exc).__name__}: {exc}"
            )

            return self._make_escalated_result(
                ticket=ticket,
                policy_decision=policy_decision,
                reason="llm_failure",
            )

        # Ensure response is not empty for replied tickets
        if agent_result.status == "replied" and not agent_result.response.strip():
            return self._make_escalated_result(
                ticket=ticket,
                policy_decision=policy_decision,
                reason="empty_response",
            )

        return agent_result

    def _make_escalated_result(
        self,
        ticket: Ticket,
        policy_decision: PolicyDecision,
        reason: str,
    ) -> AgentResult:
        """Create a deterministic escalation AgentResult without LLM call."""

        # Map escalation reason to appropriate request_type
        # If policy already classified as INVALID, keep it; otherwise use policy decision
        request_type = policy_decision.request_type

        if request_type != "invalid" and reason in (
            "empty_ticket",
            "prompt_injection",
            "invalid_request",
        ):
            request_type = "invalid"

        # Build a concise, deterministic response
        reason_text = self._escalation_reason_text(reason)

        response = (
            f"Your request has been escalated for human review. "
            f"Reason: {reason_text}."
        )

        justification = (
            f"Ticket automatically escalated per policy: {reason}. "
            f"Policy decision: {policy_decision.request_type.value} "
            f"(risk: {policy_decision.risk_level.value})."
        )

        # Product area remains "unknown" — the human reviewer will determine it
        product_area = "unknown"

        return AgentResult(
            status="escalated",
            product_area=product_area,
            response=response,
            justification=justification,
            request_type=request_type,
        )

    def _escalation_reason_text(self, reason: str) -> str:
        """Convert internal escalation reason to human-readable text."""

        mapping = {
            "empty_ticket": "ticket content is too short or empty",
            "prompt_injection": "potential prompt injection detected",
            "security_privacy": "security or privacy concern",
            "account_issue": "account or billing issue",
            "multiple_requests": "multiple distinct requests in one ticket",
            "insufficient_evidence": "no relevant documentation found",
            "retrieval_failure": "documentation retrieval failed",
            "llm_failure": "response generation failed",
            "empty_response": "LLM returned empty response",
            "high_risk": "high-risk content detected",
            "unsupported_request": "unsupported request type",
            "unknown_product_area": "product area could not be determined",
        }

        return mapping.get(reason, reason.replace("_", " "))


class TriageOrchestratorProtocol(Protocol):
    """Protocol for the triage orchestrator (for testing/mocking)."""

    def process_ticket(self, ticket: Ticket) -> AgentResult:
        """Process a ticket and return a structured result."""
        ...
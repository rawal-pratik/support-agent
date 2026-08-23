"""Triage orchestration: connect policy, retrieval, and LLM into a single-ticket workflow."""

from typing import Protocol

from .llm import LLMProvider
from .models import AgentResult, Ticket
from .policy import PolicyDecision, RequestType, evaluate_policy
from .retrieval import RetrievedChunk, Retriever


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
            retriever: Retrieves relevant documentation chunks for a ticket.
            llm_provider: Generates structured responses from the LLM.
            min_retrieved_chunks: Minimum number of chunks required to proceed
                to the LLM (must be >= 1).
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
        1. Evaluate deterministic policy.
        2. If policy mandates escalation, escalate immediately.
        3. If policy classifies the ticket as harmless invalid content,
           return a deterministic invalid response without retrieval.
        4. Retrieve relevant documentation.
        5. If insufficient evidence, escalate.
        6. Generate the final response with the LLM.
        """

        # --------------------------------------------------------------
        # Step 1: Policy evaluation
        # --------------------------------------------------------------
        policy_decision = evaluate_policy(ticket)

        # --------------------------------------------------------------
        # Step 1a: Mandatory policy escalation
        #
        # These decisions are authoritative. Do not send them to the LLM.
        # --------------------------------------------------------------
        if policy_decision.should_escalate:
            return self._make_escalated_result(
                ticket=ticket,
                policy_decision=policy_decision,
                reason=policy_decision.escalation_reason,
            )

        # --------------------------------------------------------------
        # Step 1b: Harmless invalid / conversational requests
        #
        # The policy layer has already determined that the ticket is
        # invalid but harmless. Do not retrieve unrelated documentation
        # and do not let the LLM reinterpret the policy classification.
        #
        # Example:
        #   "Thank you for helping me"
        #   "What is the actor in Iron Man?"
        #
        # Both should remain:
        #   status = replied
        #   request_type = invalid
        # --------------------------------------------------------------
        if policy_decision.request_type == RequestType.INVALID:
            return self._make_invalid_result(
                ticket=ticket,
                policy_decision=policy_decision,
            )

        # --------------------------------------------------------------
        # Step 2: Retrieval
        # --------------------------------------------------------------
        try:
            retrieved_chunks = self._retriever.retrieve(ticket)

        except Exception as exc:
            print(
                f"RETRIEVAL ERROR: {type(exc).__name__}: {exc}"
            )

            return self._make_escalated_result(
                ticket=ticket,
                policy_decision=policy_decision,
                reason="retrieval_failure",
            )

        # --------------------------------------------------------------
        # Step 2a: Insufficient evidence
        # --------------------------------------------------------------
        if len(retrieved_chunks) < self._min_retrieved_chunks:
            return self._make_escalated_result(
                ticket=ticket,
                policy_decision=policy_decision,
                reason="insufficient_evidence",
            )

        # --------------------------------------------------------------
        # Step 3: LLM generation
        # --------------------------------------------------------------
        try:
            agent_result = self._llm_provider.generate(
                ticket=ticket,
                policy_decision=policy_decision,
                retrieved_chunks=retrieved_chunks,
            )

        except Exception as exc:
            print(
                f"LLM ERROR: {type(exc).__name__}: {exc}"
            )

            return self._make_escalated_result(
                ticket=ticket,
                policy_decision=policy_decision,
                reason="llm_failure",
            )

        # --------------------------------------------------------------
        # Step 3a: Empty LLM response
        # --------------------------------------------------------------
        if (
            agent_result.status == "replied"
            and not agent_result.response.strip()
        ):
            return self._make_escalated_result(
                ticket=ticket,
                policy_decision=policy_decision,
                reason="empty_response",
            )

        return agent_result

    def _make_invalid_result(
        self,
        ticket: Ticket,
        policy_decision: PolicyDecision,
    ) -> AgentResult:
        """
        Create a deterministic response for harmless invalid requests.

        These tickets should not consume retrieval or LLM resources because
        the policy layer has already established that they are invalid and
        harmless.
        """

        issue = (ticket.issue or "").strip().lower()

        # Acknowledge simple conversational messages.
        conversational_markers = (
            "thank you",
            "thanks",
            "thx",
            "thank you so much",
            "okay",
            "ok",
            "got it",
            "gotcha",
            "understood",
            "great",
            "perfect",
        )

        is_conversational = any(
            marker in issue
            for marker in conversational_markers
        )

        if is_conversational:
            response = "You're welcome! Happy to help."

            justification = (
                "The ticket contains only a conversational acknowledgement "
                "and does not contain an actionable support request."
            )

        else:
            response = (
                "I'm sorry, but this request is outside the scope of "
                "HackerRank support."
            )

            justification = (
                "The request was classified by the deterministic policy "
                "layer as invalid or outside the supported HackerRank "
                "support scope."
            )

        return AgentResult(
            status="replied",
            product_area="conversation_management",
            response=response,
            justification=justification,
            request_type=RequestType.INVALID.value,
        )

    def _make_escalated_result(
        self,
        ticket: Ticket,
        policy_decision: PolicyDecision,
        reason: str,
    ) -> AgentResult:
        """Create a deterministic escalation AgentResult without an LLM call."""

        request_type = policy_decision.request_type

        if (
            request_type != RequestType.INVALID
            and reason in (
                "empty_ticket",
                "prompt_injection",
                "invalid_request",
            )
        ):
            request_type = RequestType.INVALID

        reason_text = self._escalation_reason_text(reason)

        response = (
            "Your request has been escalated for human review. "
            f"Reason: {reason_text}."
        )

        justification = (
            f"Ticket automatically escalated per policy: {reason}. "
            f"Policy decision: {policy_decision.request_type.value} "
            f"(risk: {policy_decision.risk_level.value})."
        )

        return AgentResult(
            status="escalated",
            product_area="unknown",
            response=response,
            justification=justification,
            request_type=request_type.value,
        )

    def _escalation_reason_text(self, reason: str) -> str:
        """Convert internal escalation reason to human-readable text."""

        reason_value = (
            reason.value
            if hasattr(reason, "value")
            else str(reason)
        )

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

        return mapping.get(
            reason_value,
            reason_value.replace("_", " "),
        )


class TriageOrchestratorProtocol(Protocol):
    """Protocol for the triage orchestrator (for testing/mocking)."""

    def process_ticket(self, ticket: Ticket) -> AgentResult:
        """Process a single ticket and return a structured result."""
        ...
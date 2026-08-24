"""End-to-end ticket triage orchestration."""

from typing import Protocol

from .llm import LLMProvider
from .models import AgentResult, Ticket
from .policy import PolicyDecision, RequestType, evaluate_policy
from .retrieval import Retriever


class TriageOrchestrator:
    def __init__(self, retriever: Retriever, llm_provider: LLMProvider, min_retrieved_chunks: int = 1):
        if min_retrieved_chunks < 1:
            raise ValueError("min_retrieved_chunks must be at least 1")
        self.retriever = retriever
        self.llm = llm_provider
        self.min_chunks = min_retrieved_chunks

    def process_ticket(self, ticket: Ticket) -> AgentResult:
        policy = evaluate_policy(ticket)

        if policy.should_escalate:
            return self._escalated(policy, policy.escalation_reason)

        if policy.request_type == RequestType.INVALID:
            return self._invalid(ticket, policy)

        try:
            chunks = self.retriever.retrieve(ticket)
        except Exception as exc:
            print(f"RETRIEVAL ERROR: {type(exc).__name__}: {exc}")
            return self._escalated(policy, "retrieval_failure", detail=f"{type(exc).__name__}: {exc}")

        if len(chunks) < self.min_chunks:
            return self._escalated(policy, "insufficient_evidence")

        try:
            result = self.llm.generate(ticket, policy, chunks)
        except Exception as exc:
            print(f"LLM ERROR: {type(exc).__name__}: {exc}")
            return self._escalated(policy, "llm_failure", detail=f"{type(exc).__name__}: {exc}")

        if result.status == "replied" and not (result.response or "").strip():
            return self._escalated(policy, "empty_response")

        return result

    @staticmethod
    def _invalid(ticket: Ticket, policy: PolicyDecision) -> AgentResult:
        issue = (ticket.issue or "").strip().lower()
        conversational = any(x in issue for x in ("thank you", "thanks", "thx", "got it", "okay", "ok", "great"))
        if conversational:
            response = "You're welcome! Happy to help."
            justification = "The ticket contains only a conversational acknowledgement and does not contain an actionable support request."
        else:
            response = "I'm sorry, but this request is outside the scope of HackerRank support."
            justification = "The request was classified by the deterministic policy layer as invalid or outside the supported HackerRank support scope."
        return AgentResult(
            status="replied",
            product_area="conversation_management",
            response=response,
            justification=justification,
            request_type=RequestType.INVALID.value,
        )

    @staticmethod
    def _escalated(policy: PolicyDecision, reason, detail: str | None = None) -> AgentResult:
        reason_value = reason.value if hasattr(reason, "value") else str(reason)
        descriptions = {
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
        }
        request_type = policy.request_type
        if reason_value in {"empty_ticket", "prompt_injection", "invalid_request"}:
            request_type = RequestType.INVALID

        justification = (
            f"Ticket automatically escalated per policy: {reason_value}. "
            f"Policy decision: {policy.request_type.value} "
            f"(risk: {policy.risk_level.value})."
        )
        # Internal-only diagnostic detail (never surfaced in "response"),
        # so a failure like a truncated/malformed LLM call is diagnosable
        # from the output CSV itself instead of only from stdout logs.
        if detail:
            justification += f" Internal detail: {detail[:300]}"

        return AgentResult(
            status="escalated",
            product_area="unknown",
            response=f"Your request has been escalated for human review. Reason: {descriptions.get(reason_value, reason_value.replace('_', ' '))}.",
            justification=justification,
            request_type=request_type.value,
        )


class TriageOrchestratorProtocol(Protocol):
    def process_ticket(self, ticket: Ticket) -> AgentResult:
        ...
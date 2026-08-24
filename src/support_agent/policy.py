"""Deterministic policy rules for support triage."""

import re
from enum import Enum

from pydantic import BaseModel

from .models import Ticket


class RequestType(str, Enum):
    INVALID = "invalid"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    PRODUCT_ISSUE = "product_issue"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EscalationReason(str, Enum):
    EMPTY_TICKET = "empty_ticket"
    INVALID_REQUEST = "invalid_request"
    MULTIPLE_REQUESTS = "multiple_requests"
    PROMPT_INJECTION = "prompt_injection"
    SECURITY_PRIVACY = "security_privacy"
    ACCOUNT_ISSUE = "account_issue"
    HIGH_RISK = "high_risk"
    UNSUPPORTED_REQUEST = "unsupported_request"
    UNKNOWN_PRODUCT_AREA = "unknown_product_area"


class PolicyDecision(BaseModel):
    should_escalate: bool
    escalation_reason: EscalationReason | None = None
    request_type: RequestType
    risk_level: RiskLevel
    product_area: str = "unknown"
    is_prompt_injection: bool = False
    has_multiple_requests: bool = False
    is_empty: bool = False
    is_unsupported: bool = False


def _text(ticket: Ticket) -> str:
    return f"{ticket.subject or ''} {ticket.issue or ''}".strip()


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


CONVERSATIONAL = (
    r"^\s*(?:thanks?|thank you|thank you so much|thx)(?:[.!]|\s.*)?$",
    r"^\s*(?:okay|ok|got it|gotcha|understood|great|perfect)[.!]?$",
    r"^\s*(?:hello|hi|hey|good morning|good afternoon|good evening)[!.]?$",
)

OUT_OF_SCOPE = (
    r"\b(?:actor|actress|movie|film|song|singer|celebrity)\b.*"
    r"\b(?:iron\s+man|avengers|batman|superman|hollywood|bollywood)\b",
)

INJECTION = (
    r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|directions?)",
    r"disregard\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|directions?)",
    r"forget\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?|directions?)",
    r"\byou\s+are\s+now\b",
    r"\b(?:act\s+as|pretend\s+(?:to\s+be|you\s+are))\b",
    r"(?:system\s*:|<\s*system\s*>|\[system\])",
    r"override\s+(?:all\s+)?(?:rules?|instructions?|settings?)",
    r"\b(?:jailbreak|dan\s+mode|developer\s+mode)\b",
)

SECURITY = (
    r"\bpassword\s+(?:reset|change|recovery|forgot)\b",
    r"\bsecurity\s+(?:vulnerability|issue|breach|concern)\b",
    r"\bdata\s+breach\b",
    r"\bprivacy\s+(?:concern|issue|violation)\b",
    r"\bgdpr\s+(?:request|compliance|deletion)\b",
    r"\bpersonal\s+data\s+(?:deletion|access|removal)\b",
    r"\bcredential\s+(?:leak|compromise|exposure)\b",
    r"\bauthentication\s+(?:bypass|vulnerability|issue)\b",
    r"\bunauthorized\s+(?:access|use|disclosure)\b",
    r"\bhacked\b",
)

ACCOUNT = (
    r"\baccount\s+(?:locked|suspended|deleted|compromised|hacked)\b",
    r"\bbilling\s+(?:dispute|error|issue|problem)\b",
    r"\bsubscription\s+(?:cancel|refund|change|issue)\b",
    r"\bpayment\s+(?:failed|dispute|refund|issue)\b",
    r"\binvoice\s+(?:dispute|error|missing)\b",
    r"\brefund\s+(?:request|needed|required)\b",
    r"\bcharge\s+(?:dispute|unauthorized|error)\b",
    r"\bcancel\s+(?:my\s+)?subscription\b",
)

OUTAGE = (
    r"\b(?:site|website|platform)\s+is\s+down\b",
    r"\b(?:site|website|platform)\s+(?:is\s+)?(?:completely\s+)?(?:unavailable|inaccessible)\b",
    r"\b(?:all|none\s+of\s+the)\s+(?:pages?|services?|features?)\s+(?:are\s+)?(?:down|unavailable|inaccessible|not\s+working)\b",
    r"\b(?:nothing|everything)\s+(?:is\s+)?(?:working|accessible)\b",
    r"\b(?:entire|whole)\s+(?:platform|site|website|service)\s+(?:is\s+)?(?:down|unavailable|inaccessible)\b",
    r"\b(?:platform|service)\s+(?:wide\s+)?outage\b",
)

BUG = (
    r"\bbug\b", r"\berror\b", r"\bcrash(?:es|ed|ing)?\b",
    r"\bnot\s+working\b", r"\bbroken\b", r"\bdoesn'?t\s+work\b",
    r"\bdoes\s+not\s+work\b", r"\bissue\b", r"\bproblem\b",
    r"\bfail(?:ed|s|ing|ure)?\b", r"\bincorrect(?:ly)?\b", r"\bunexpected\b",
)

FEATURE = (
    r"\bfeature\s+request\b", r"\bnew\s+feature\b",
    r"\badd\s+(?:a\s+)?feature\b",
    r"\bwould\s+(?:like|love)\s+to\s+see\b",
    r"\bwish\s+(?:you\s+)?(?:could|had)\b",
    r"\bit\s+would\s+be\s+(?:great|nice|helpful)\s+if\b",
    r"\bidea\s+for\b",
)


def _empty(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return not compact or len(compact) < 10 or bool(
        re.fullmatch(r"[.?!]+|[A-Za-z]{1,2}|(?:test|testing|asdf|qwerty|abc)", text, re.I)
    )


def _conversational(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    return _matches(normalized, CONVERSATIONAL)


def _multiple(text: str) -> bool:
    # Deliberately conservative: related multi-part questions are one request.
    return bool(re.search(
        r"(?:^|[.!?]\s+)(?:also|additionally|separately|in addition|"
        r"furthermore|moreover)\s+(?:please\s+)?"
        r"(?:can|could|would|please|help|reset|change|cancel|refund|"
        r"delete|remove|update|configure|create|add|enable|disable)\b",
        text, re.I,
    ))


def _request_type(text: str) -> RequestType:
    if _conversational(text) or _matches(text, OUT_OF_SCOPE):
        return RequestType.INVALID
    if _matches(text, BUG):
        return RequestType.BUG
    if _matches(text, FEATURE):
        return RequestType.FEATURE_REQUEST
    return RequestType.PRODUCT_ISSUE


def evaluate_policy(ticket: Ticket) -> PolicyDecision:
    text = _text(ticket)

    if _empty(text):
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.EMPTY_TICKET,
            request_type=RequestType.INVALID,
            risk_level=RiskLevel.LOW,
            is_empty=True,
        )

    if _matches(text, INJECTION):
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.PROMPT_INJECTION,
            request_type=RequestType.INVALID,
            risk_level=RiskLevel.HIGH,
            is_prompt_injection=True,
        )

    if _matches(text, SECURITY):
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.SECURITY_PRIVACY,
            request_type=_request_type(text),
            risk_level=RiskLevel.HIGH,
        )

    if _matches(text, ACCOUNT):
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.ACCOUNT_ISSUE,
            request_type=_request_type(text),
            risk_level=RiskLevel.HIGH,
        )

    if _matches(text, OUTAGE):
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.HIGH_RISK,
            request_type=RequestType.BUG,
            risk_level=RiskLevel.HIGH,
        )

    if _conversational(text):
        return PolicyDecision(
            should_escalate=False,
            request_type=RequestType.INVALID,
            risk_level=RiskLevel.LOW,
        )

    if _matches(text, OUT_OF_SCOPE):
        return PolicyDecision(
            should_escalate=False,
            request_type=RequestType.INVALID,
            risk_level=RiskLevel.LOW,
            is_unsupported=True,
        )

    if _multiple(text):
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.MULTIPLE_REQUESTS,
            request_type=_request_type(text),
            risk_level=RiskLevel.MEDIUM,
            has_multiple_requests=True,
        )

    return PolicyDecision(
        should_escalate=False,
        request_type=_request_type(text),
        risk_level=RiskLevel.LOW,
    )
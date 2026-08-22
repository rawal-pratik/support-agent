"""Deterministic policy layer for ticket classification, risk assessment, and escalation."""

import re
from enum import Enum

from pydantic import BaseModel

from .models import Ticket


class RequestType(str, Enum):
    """Classification of ticket request type with precedence ordering."""

    INVALID = "invalid"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    PRODUCT_ISSUE = "product_issue"


class RiskLevel(str, Enum):
    """Internal risk assessment levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EscalationReason(str, Enum):
    """Reasons for escalating a ticket."""

    EMPTY_TICKET = "empty_ticket"
    INVALID_REQUEST = "invalid_request"
    MULTIPLE_REQUESTS = "multiple_requests"
    PROMPT_INJECTION = "prompt_injection"
    SECURITY_PRIVACY = "security_privacy"
    ACCOUNT_ISSUE = "account_issue"
    HIGH_RISK = "high_risk"
    UNSUPPORTED_REQUEST = "unsupported_request"
    UNKNOWN_PRODUCT_AREA = "unknown_product_area"


# Patterns for detecting sensitive/escalation-worthy content
# Conservative: focus on clear security/privacy/account issues, not casual mentions
SECURITY_PRIVACY_PATTERNS = [
    re.compile(r"\bpassword\s*(reset|change|recovery|forgot)\b", re.IGNORECASE),
    re.compile(r"\bsecurity\s*(vulnerability|issue|breach|concern)\b", re.IGNORECASE),
    re.compile(r"\bdata\s*breach\b", re.IGNORECASE),
    re.compile(r"\bprivacy\s*(concern|issue|violation)\b", re.IGNORECASE),
    re.compile(r"\bgdpr\s*(request|compliance|deletion)\b", re.IGNORECASE),
    re.compile(r"\bpersonal\s*data\s*(deletion|access|removal)\b", re.IGNORECASE),
    re.compile(r"\bcredential\s*(leak|compromise|exposure)\b", re.IGNORECASE),
    re.compile(r"\bauthentication\s*(bypass|vulnerability|issue)\b", re.IGNORECASE),
    re.compile(r"\bunauthorized\s*(access|use|disclosure)\b", re.IGNORECASE),
    re.compile(r"\bhacked\b", re.IGNORECASE),
]

ACCOUNT_BILLING_PATTERNS = [
    re.compile(r"\baccount\s*(locked|suspended|deleted|compromised|hacked)\b", re.IGNORECASE),
    re.compile(r"\bbilling\s*(dispute|error|issue|problem)\b", re.IGNORECASE),
    re.compile(r"\bsubscription\s*(cancel|refund|change|issue)\b", re.IGNORECASE),
    re.compile(r"\bpayment\s*(failed|dispute|refund|issue)\b", re.IGNORECASE),
    re.compile(r"\binvoice\s*(dispute|error|missing)\b", re.IGNORECASE),
    re.compile(r"\brefund\s*(request|needed|required)\b", re.IGNORECASE),
    re.compile(r"\bcharge\s*(dispute|unauthorized|error)\b", re.IGNORECASE),
    re.compile(r"\bcancel\s*(my\s+)?subscription\b", re.IGNORECASE),
]

# Patterns for detecting prompt injection attempts
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|directions?)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|directions?)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|directions?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(to\s+be|you\s+are)", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[system\]", re.IGNORECASE),
    re.compile(r"override\s+(all\s+)?(rules?|instructions?|settings?)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"dan\s+mode", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
]

# Patterns for detecting bug reports
BUG_PATTERNS = [
    re.compile(r"\bbug\b", re.IGNORECASE),
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"\bcrash(es|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bnot\s+working\b", re.IGNORECASE),
    re.compile(r"\bbroken\b", re.IGNORECASE),
    re.compile(r"\bdoesn'?t\s+work\b", re.IGNORECASE),
    re.compile(r"\bdoes\s+not\s+work\b", re.IGNORECASE),
    re.compile(r"\bissue\b", re.IGNORECASE),
    re.compile(r"\bproblem\b", re.IGNORECASE),
    re.compile(r"\bfail(ed|s|ing|ure)?\b", re.IGNORECASE),
    re.compile(r"\bincorrect(ly)?\b", re.IGNORECASE),
    re.compile(r"\bunexpected\b", re.IGNORECASE),
]

# Patterns for detecting feature requests
FEATURE_REQUEST_PATTERNS = [
    re.compile(r"\bfeature\s+request\b", re.IGNORECASE),
    re.compile(r"\bnew\s+feature\b", re.IGNORECASE),
    re.compile(r"\badd\s+(a\s+)?feature\b", re.IGNORECASE),
    re.compile(r"\bwould\s+(like|love)\s+to\s+see\b", re.IGNORECASE),
    re.compile(r"\bwish\s+(you\s+)?(could|had)\b", re.IGNORECASE),
    re.compile(r"\bit\s+would\s+be\s+(great|nice|helpful)\s+if\b", re.IGNORECASE),
    re.compile(r"\bidea\s+for\b", re.IGNORECASE),
]

# Patterns for detecting invalid/empty tickets
INVALID_PATTERNS = [
    re.compile(r"^\s*$"),  # Empty or whitespace only
    re.compile(r"^[.]+$"),  # Only dots
    re.compile(r"^[?]+$"),  # Only question marks
    re.compile(r"^[!]+$"),  # Only exclamation marks
    re.compile(r"^[a-zA-Z]{1,2}$"),  # Single or double letters only
    re.compile(r"^(test|testing|asdf|qwerty|abc)$", re.IGNORECASE),  # Test inputs
]


class PolicyDecision(BaseModel):
    """Structured decision from the policy layer."""

    should_escalate: bool
    escalation_reason: EscalationReason | None = None
    request_type: RequestType
    risk_level: RiskLevel
    product_area: str = "unknown"
    is_prompt_injection: bool = False
    has_multiple_requests: bool = False
    is_empty: bool = False
    is_unsupported: bool = False


def _detect_empty_ticket(ticket: Ticket) -> bool:
    """Check if ticket is empty or has minimal content."""
    combined = f"{ticket.subject} {ticket.issue}".strip()
    if not combined:
        return True
    for pattern in INVALID_PATTERNS:
        if pattern.match(combined):
            return True
    # Check for very short content (< 10 characters after normalization)
    normalized = re.sub(r"\s+", "", combined)
    if len(normalized) < 10:
        return True
    return False


def _detect_prompt_injection(ticket: Ticket) -> bool:
    """Detect prompt injection attempts in ticket text."""
    combined = f"{ticket.subject} {ticket.issue}"
    for pattern in INJECTION_PATTERNS:
        if pattern.search(combined):
            return True
    return False


def _detect_security_privacy(ticket: Ticket) -> bool:
    """Detect security/privacy-related tickets that should be escalated."""
    combined = f"{ticket.subject} {ticket.issue}"
    for pattern in SECURITY_PRIVACY_PATTERNS:
        if pattern.search(combined):
            return True
    return False


def _detect_account_issues(ticket: Ticket) -> bool:
    """Detect account/billing-related tickets that should be escalated."""
    combined = f"{ticket.subject} {ticket.issue}"
    for pattern in ACCOUNT_BILLING_PATTERNS:
        if pattern.search(combined):
            return True
    return False


def _detect_multiple_requests(ticket: Ticket) -> bool:
    """Detect if ticket contains multiple distinct requests."""
    combined = f"{ticket.subject} {ticket.issue}"
    # Look for multiple question marks or numbered lists
    question_count = combined.count("?")
    # Look for numbered items like "1.", "2.", etc.
    numbered_items = re.findall(r"\b\d+\.\s+", combined)
    # Look for transition words that might indicate multiple requests
    transition_words = re.findall(
        r"\b(also|additionally|furthermore|moreover|secondly|another thing)\b",
        combined,
        re.IGNORECASE,
    )
    return question_count > 2 or len(numbered_items) >= 2 or len(transition_words) >= 2


def _classify_request_type(ticket: Ticket) -> RequestType:
    """Classify request type with precedence: invalid → bug → feature request → product issue.

    Note: This function is only called for non-escalated tickets, so invalid is already
    handled by the empty/injection checks. We check bug before feature_request to ensure
    correct precedence.
    """
    combined = f"{ticket.subject} {ticket.issue}"

    # Check for bug patterns first (bug beats feature_request)
    for pattern in BUG_PATTERNS:
        if pattern.search(combined):
            return RequestType.BUG

    # Check for feature request patterns
    for pattern in FEATURE_REQUEST_PATTERNS:
        if pattern.search(combined):
            return RequestType.FEATURE_REQUEST

    # Default to product issue
    return RequestType.PRODUCT_ISSUE


def _assess_risk_level(
    is_security: bool, is_account: bool, is_injection: bool
) -> RiskLevel:
    """Assess internal risk level."""
    if is_injection:
        return RiskLevel.HIGH
    if is_security or is_account:
        return RiskLevel.HIGH
    return RiskLevel.LOW


def evaluate_policy(ticket: Ticket) -> PolicyDecision:
    """Evaluate a ticket against policy rules and return a decision.

    This is the main entry point for the policy layer. It applies deterministic
    rules to classify the ticket, assess risk, and determine if escalation is needed.

    The policy layer is completely independent of retrieval, embeddings, database,
    and LLM services. It operates solely on the ticket text.

    Args:
        ticket: The support ticket to evaluate

    Returns:
        PolicyDecision with classification, risk assessment, and escalation status
    """
    # Check for empty/invalid ticket first
    if _detect_empty_ticket(ticket):
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.EMPTY_TICKET,
            request_type=RequestType.INVALID,
            risk_level=RiskLevel.LOW,
            is_empty=True,
        )

    # Check for prompt injection (always escalate)
    is_injection = _detect_prompt_injection(ticket)
    if is_injection:
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.PROMPT_INJECTION,
            request_type=RequestType.INVALID,
            risk_level=RiskLevel.HIGH,
            is_prompt_injection=True,
        )

    # Check for security/privacy issues (always escalate)
    is_security = _detect_security_privacy(ticket)
    if is_security:
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.SECURITY_PRIVACY,
            request_type=_classify_request_type(ticket),
            risk_level=RiskLevel.HIGH,
            # Product area is always "unknown" - orchestration will determine it
        )

    # Check for account/billing issues (always escalate)
    is_account = _detect_account_issues(ticket)
    if is_account:
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.ACCOUNT_ISSUE,
            request_type=_classify_request_type(ticket),
            risk_level=RiskLevel.HIGH,
        )

    # Check for multiple requests
    has_multiple = _detect_multiple_requests(ticket)
    if has_multiple:
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.MULTIPLE_REQUESTS,
            request_type=_classify_request_type(ticket),
            risk_level=RiskLevel.MEDIUM,
            has_multiple_requests=True,
        )

    # Classify request type
    request_type = _classify_request_type(ticket)

    # Assess risk level
    risk_level = _assess_risk_level(is_security, is_account, is_injection)

    # For normal tickets, product area remains "unknown"
    # The orchestration layer will use retrieval evidence to determine it
    return PolicyDecision(
        should_escalate=False,
        request_type=request_type,
        risk_level=risk_level,
    )

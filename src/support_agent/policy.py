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


SECURITY_PRIVACY_PATTERNS = [
    re.compile(r"\bpassword\s*(reset|change|recovery|forgot)\b", re.IGNORECASE),
    re.compile(r"\bsecurity\s*(vulnerability|issue|breach|concern)\b", re.IGNORECASE),
    re.compile(r"\bdata\s*breach\b", re.IGNORECASE),
    re.compile(r"\bprivacy\s*(concern|issue|violation)\b", re.IGNORECASE),
    re.compile(r"\bgdpr\s*(request|compliance|deletion)\b", re.IGNORECASE),
    re.compile(
        r"\bpersonal\s*data\s*(deletion|access|removal)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcredential\s*(leak|compromise|exposure)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bauthentication\s*(bypass|vulnerability|issue)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bunauthorized\s*(access|use|disclosure)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bhacked\b", re.IGNORECASE),
]


ACCOUNT_BILLING_PATTERNS = [
    re.compile(
        r"\baccount\s*(locked|suspended|deleted|compromised|hacked)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbilling\s*(dispute|error|issue|problem)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsubscription\s*(cancel|refund|change|issue)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpayment\s*(failed|dispute|refund|issue)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\binvoice\s*(dispute|error|missing)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\brefund\s*(request|needed|required)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcharge\s*(dispute|unauthorized|error)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcancel\s*(my\s+)?subscription\b",
        re.IGNORECASE,
    ),
]


INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|above|prior)\s+"
        r"(instructions?|prompts?|directions?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard\s+(all\s+)?(previous|above|prior)\s+"
        r"(instructions?|prompts?|directions?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"forget\s+(all\s+)?(previous|above|prior)\s+"
        r"(instructions?|prompts?|directions?)",
        re.IGNORECASE,
    ),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+", re.IGNORECASE),
    re.compile(
        r"pretend\s+(to\s+be|you\s+are)",
        re.IGNORECASE,
    ),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[system\]", re.IGNORECASE),
    re.compile(
        r"override\s+(all\s+)?(rules?|instructions?|settings?)",
        re.IGNORECASE,
    ),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"dan\s+mode", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
]


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


CRITICAL_OUTAGE_PATTERNS = [
    re.compile(
        r"\b(?:the\s+)?(?:site|website|platform)\s+is\s+down\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:site|website|platform)\s+(?:is\s+)?"
        r"(?:completely\s+)?"
        r"(?:unavailable|inaccessible)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:all|none\s+of\s+the)\s+"
        r"(?:pages?|services?|features?)\s+"
        r"(?:are\s+)?"
        r"(?:down|unavailable|inaccessible|not\s+working)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:nothing|everything)\s+"
        r"(?:is\s+)?(?:working|accessible)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:entire|whole)\s+"
        r"(?:platform|site|website|service)\s+"
        r"(?:is\s+)?"
        r"(?:down|unavailable|inaccessible)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:platform|service)\s+(?:wide\s+)?outage\b",
        re.IGNORECASE,
    ),
]


FEATURE_REQUEST_PATTERNS = [
    re.compile(r"\bfeature\s+request\b", re.IGNORECASE),
    re.compile(r"\bnew\s+feature\b", re.IGNORECASE),
    re.compile(r"\badd\s+(a\s+)?feature\b", re.IGNORECASE),
    re.compile(
        r"\bwould\s+(like|love)\s+to\s+see\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwish\s+(you\s+)?(could|had)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bit\s+would\s+be\s+(great|nice|helpful)\s+if\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bidea\s+for\b", re.IGNORECASE),
]


INVALID_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^[.]+$"),
    re.compile(r"^[?]+$"),
    re.compile(r"^[!]+$"),
    re.compile(r"^[a-zA-Z]{1,2}$"),
    re.compile(
        r"^(test|testing|asdf|qwerty|abc)$",
        re.IGNORECASE,
    ),
]


CONVERSATIONAL_PATTERNS = [
    re.compile(
        r"^(?:thanks|thank\s+you|thank\s+you\s+so\s+much|thx)"
        r"[.!]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:okay|ok|got\s+it|gotcha|understood|great|perfect)"
        r"[.!]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))"
        r"[!.]?$",
        re.IGNORECASE,
    ),
]


OUT_OF_SCOPE_PATTERNS = [
    re.compile(
        r"\b(?:actor|actress|movie|film|song|singer|celebrity)\b"
        r".*\b(?:iron\s+man|avengers|batman|superman|hollywood|"
        r"bollywood)\b",
        re.IGNORECASE,
    ),
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


def _combined_text(ticket: Ticket) -> str:
    """Return normalized ticket text used by policy detectors."""
    subject = ticket.subject or ""
    issue = ticket.issue or ""
    return f"{subject} {issue}".strip()


def _detect_empty_ticket(ticket: Ticket) -> bool:
    """Check if ticket is empty or has minimal content."""
    combined = _combined_text(ticket)

    if not combined:
        return True

    for pattern in INVALID_PATTERNS:
        if pattern.match(combined):
            return True

    normalized = re.sub(r"\s+", "", combined)

    return len(normalized) < 10


def _detect_conversational_message(ticket: Ticket) -> bool:
    """Detect harmless conversational messages such as thank-you messages."""

    subject = (ticket.subject or "").strip()
    issue = (ticket.issue or "").strip()

    if not subject and not issue:
        return False

    conversational_texts = (subject, issue)

    for text in conversational_texts:
        if any(
            pattern.fullmatch(text)
            for pattern in CONVERSATIONAL_PATTERNS
        ):
            return True

        normalized = re.sub(r"\s+", " ", text).strip().lower()

        if re.fullmatch(
            r"(?:thanks?|thank you)(?:\s+.*)?[.!]?",
            normalized,
        ):
            return True

    return False

def _detect_out_of_scope(ticket: Ticket) -> bool:
    """Detect clearly non-support, out-of-domain questions."""
    combined = _combined_text(ticket)

    return any(
        pattern.search(combined)
        for pattern in OUT_OF_SCOPE_PATTERNS
    )


def _detect_prompt_injection(ticket: Ticket) -> bool:
    """Detect prompt injection attempts."""
    combined = _combined_text(ticket)

    return any(
        pattern.search(combined)
        for pattern in INJECTION_PATTERNS
    )


def _detect_security_privacy(ticket: Ticket) -> bool:
    """Detect security/privacy-related tickets."""
    combined = _combined_text(ticket)

    return any(
        pattern.search(combined)
        for pattern in SECURITY_PRIVACY_PATTERNS
    )


def _detect_account_issues(ticket: Ticket) -> bool:
    """Detect account/billing-related tickets."""
    combined = _combined_text(ticket)

    return any(
        pattern.search(combined)
        for pattern in ACCOUNT_BILLING_PATTERNS
    )


def _detect_critical_outage(ticket: Ticket) -> bool:
    """Detect platform-wide outages."""
    combined = _combined_text(ticket)

    return any(
        pattern.search(combined)
        for pattern in CRITICAL_OUTAGE_PATTERNS
    )


def _detect_multiple_requests(ticket: Ticket) -> bool:
    """Detect clearly independent support requests.

    This detector is intentionally conservative.

    Multiple questions do not necessarily mean multiple requests.
    For example:

        "Should I create a variant or a different test?
         What are the advantages and disadvantages?"

    is one coherent product question.

    We therefore only escalate when the ticket contains an explicit
    transition to another independent request.
    """

    combined = _combined_text(ticket)

    # Explicit transitions are strong evidence of separate requests.
    explicit_transition_patterns = [
        re.compile(
            r"\b(?:also|additionally|separately|"
            r"in\s+addition|furthermore|moreover)\b"
            r".*\b(?:please|can\s+you|could\s+you|"
            r"help\s+me|would\s+you)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:another|one\s+more)\s+"
            r"(?:thing|question|request)\b"
            r".*\b(?:please|can\s+you|could\s+you|"
            r"help\s+me|would\s+you)\b",
            re.IGNORECASE,
        ),
    ]

    if any(
        pattern.search(combined)
        for pattern in explicit_transition_patterns
    ):
        return True

    # Explicitly separated imperative requests.
    #
    # Examples:
    #   "Reset my password. Also cancel my subscription."
    #   "Change my email and separately refund my payment."
    #
    # We require a strong second-request marker rather than simply
    # counting sentences or question marks.
    explicit_second_request = re.compile(
        r"(?:^|[.!?]\s+)"
        r"(?:also|additionally|separately|"
        r"in\s+addition|furthermore|moreover)\s+"
        r"(?:please\s+)?"
        r"(?:can|could|would|please|help|"
        r"reset|change|cancel|refund|delete|remove|"
        r"update|configure|create|add|enable|disable)\b",
        re.IGNORECASE,
    )

    if explicit_second_request.search(combined):
        return True

    return False


def _classify_request_type(ticket: Ticket) -> RequestType:
    """Classify request type.

    Precedence:
        invalid → bug → feature_request → product_issue
    """

    combined = _combined_text(ticket)

    if _detect_conversational_message(ticket):
        return RequestType.INVALID

    if _detect_out_of_scope(ticket):
        return RequestType.INVALID

    for pattern in BUG_PATTERNS:
        if pattern.search(combined):
            return RequestType.BUG

    for pattern in FEATURE_REQUEST_PATTERNS:
        if pattern.search(combined):
            return RequestType.FEATURE_REQUEST

    return RequestType.PRODUCT_ISSUE


def _assess_risk_level(
    is_security: bool,
    is_account: bool,
    is_injection: bool,
    is_outage: bool = False,
) -> RiskLevel:
    """Assess internal risk level."""

    if is_injection:
        return RiskLevel.HIGH

    if is_security or is_account:
        return RiskLevel.HIGH

    if is_outage:
        return RiskLevel.HIGH

    return RiskLevel.LOW


def evaluate_policy(ticket: Ticket) -> PolicyDecision:
    """Evaluate a ticket against deterministic policy rules."""

    # --------------------------------------------------------------
    # 1. Empty / obviously invalid input
    # --------------------------------------------------------------
    if _detect_empty_ticket(ticket):
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.EMPTY_TICKET,
            request_type=RequestType.INVALID,
            risk_level=RiskLevel.LOW,
            is_empty=True,
        )

    # --------------------------------------------------------------
    # 2. Prompt injection
    # --------------------------------------------------------------
    is_injection = _detect_prompt_injection(ticket)

    if is_injection:
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.PROMPT_INJECTION,
            request_type=RequestType.INVALID,
            risk_level=RiskLevel.HIGH,
            is_prompt_injection=True,
        )

    # --------------------------------------------------------------
    # 3. Security / privacy
    # --------------------------------------------------------------
    is_security = _detect_security_privacy(ticket)

    if is_security:
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.SECURITY_PRIVACY,
            request_type=_classify_request_type(ticket),
            risk_level=RiskLevel.HIGH,
        )

    # --------------------------------------------------------------
    # 4. Account / billing
    # --------------------------------------------------------------
    is_account = _detect_account_issues(ticket)

    if is_account:
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.ACCOUNT_ISSUE,
            request_type=_classify_request_type(ticket),
            risk_level=RiskLevel.HIGH,
        )

    # --------------------------------------------------------------
    # 5. Critical platform outage
    # --------------------------------------------------------------
    is_outage = _detect_critical_outage(ticket)

    if is_outage:
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.HIGH_RISK,
            request_type=RequestType.BUG,
            risk_level=RiskLevel.HIGH,
        )

    # --------------------------------------------------------------
    # 6. Harmless conversational input
    #
    # These do not escalate and do not need retrieval/LLM.
    # --------------------------------------------------------------
    if _detect_conversational_message(ticket):
        return PolicyDecision(
            should_escalate=False,
            request_type=RequestType.INVALID,
            risk_level=RiskLevel.LOW,
        )

    # --------------------------------------------------------------
    # 7. Clearly out-of-scope input
    #
    # Harmless invalid requests should not automatically escalate.
    # --------------------------------------------------------------
    if _detect_out_of_scope(ticket):
        return PolicyDecision(
            should_escalate=False,
            request_type=RequestType.INVALID,
            risk_level=RiskLevel.LOW,
            is_unsupported=True,
        )

    # --------------------------------------------------------------
    # 8. Multiple genuinely independent requests
    # --------------------------------------------------------------
    has_multiple = _detect_multiple_requests(ticket)

    if has_multiple:
        return PolicyDecision(
            should_escalate=True,
            escalation_reason=EscalationReason.MULTIPLE_REQUESTS,
            request_type=_classify_request_type(ticket),
            risk_level=RiskLevel.MEDIUM,
            has_multiple_requests=True,
        )

    # --------------------------------------------------------------
    # 9. Normal request classification
    # --------------------------------------------------------------
    request_type = _classify_request_type(ticket)

    risk_level = _assess_risk_level(
        is_security=is_security,
        is_account=is_account,
        is_injection=is_injection,
        is_outage=is_outage,
    )

    return PolicyDecision(
        should_escalate=False,
        request_type=request_type,
        risk_level=risk_level,
    )
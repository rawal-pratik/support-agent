import pytest
from pydantic import ValidationError

from support_agent.models import AgentResult, Ticket


def test_ticket_uses_the_observed_input_fields():
    ticket = Ticket(issue="Cannot submit", subject="Submission problem")

    assert ticket.issue == "Cannot submit"
    assert ticket.subject == "Submission problem"
    assert set(Ticket.model_fields) == {"issue", "subject"}


def test_ticket_allows_the_blank_subject_seen_in_the_sample_data():
    ticket = Ticket(issue="Account problem", subject="")

    assert ticket.subject == ""


def test_ticket_requires_an_issue():
    with pytest.raises(ValidationError):
        Ticket(issue="", subject="Missing issue")


def test_agent_result_contains_exactly_the_required_fields():
    result = AgentResult(
        status="replied",
        product_area="screen",
        response="Please check the test settings.",
        justification="The documentation describes this setting.",
        request_type="product_issue",
    )

    assert set(AgentResult.model_fields) == {
        "status",
        "product_area",
        "response",
        "justification",
        "request_type",
    }
    assert result.model_dump() == {
        "status": "replied",
        "product_area": "screen",
        "response": "Please check the test settings.",
        "justification": "The documentation describes this setting.",
        "request_type": "product_issue",
    }


@pytest.mark.parametrize("status", ["reply", "pending"])
def test_agent_result_rejects_invalid_status(status: str):
    with pytest.raises(ValidationError):
        AgentResult(
            status=status,
            product_area="screen",
            response="Response",
            justification="Reason",
            request_type="bug",
        )


@pytest.mark.parametrize("request_type", ["question", "feature", ""])
def test_agent_result_rejects_invalid_request_type(request_type: str):
    with pytest.raises(ValidationError):
        AgentResult(
            status="escalated",
            product_area="screen",
            response="Response",
            justification="Reason",
            request_type=request_type,
        )

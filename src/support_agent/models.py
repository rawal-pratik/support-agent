from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Ticket(BaseModel):
    """Internal representation of a challenge support ticket."""

    model_config = ConfigDict(str_strip_whitespace=True)

    issue: str = Field(min_length=1)
    subject: str = ""


class AgentResult(BaseModel):
    """Required customer-facing result for one support ticket."""

    status: Literal["replied", "escalated"]
    product_area: str
    response: str
    justification: str
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]

"""Basic functional tests for the OpenRouter LLM provider."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from support_agent.llm import OpenRouterProvider
from support_agent.models import AgentResult, Ticket
from support_agent.policy import (
    PolicyDecision,
    RequestType,
    RiskLevel,
)
from support_agent.retrieval import RetrievedChunk


def make_ticket():
    return Ticket(
        subject="Create a test",
        issue="How do I create a test?",
    )


def make_policy():
    return PolicyDecision(
        should_escalate=False,
        request_type=RequestType.PRODUCT_ISSUE,
        risk_level=RiskLevel.LOW,
    )


def make_chunk():
    return RetrievedChunk(
        chunk_id="chunk-1",
        source_path="screen/managing-tests/create-test.md",
        title="Create a Test",
        category="Screen / Managing Tests",
        text="Open the Tests tab and click Create Test.",
        similarity=0.9,
    )


def make_llm_json(
    status="replied",
    product_area="Screen / Managing Tests",
    response="Open the Tests tab and click Create Test.",
    justification="Document 1 explains how to create a test.",
    request_type="product_issue",
):
    return json.dumps(
        {
            "status": status,
            "product_area": product_area,
            "response": response,
            "justification": justification,
            "request_type": request_type,
        }
    )


def make_response(content):
    return httpx.Response(
        200,
        request=httpx.Request(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
        json={
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        },
    )


class TestProviderConfiguration:
    def test_requires_api_key(self):
        with pytest.raises(
            ValueError,
            match="api_key must be provided",
        ):
            OpenRouterProvider("")

    def test_from_env_requires_api_key(self, monkeypatch):
        monkeypatch.delenv(
            "OPENROUTER_API_KEY",
            raising=False,
        )

        with pytest.raises(
            ValueError,
            match="OPENROUTER_API_KEY must be configured",
        ):
            OpenRouterProvider.from_env()

    def test_from_env_uses_environment_variables(self, monkeypatch):
        monkeypatch.setenv(
            "OPENROUTER_API_KEY",
            "test-key",
        )
        monkeypatch.setenv(
            "OPENROUTER_MODEL",
            "test-model",
        )

        with patch("support_agent.llm.httpx.Client"):
            provider = OpenRouterProvider.from_env()

        assert provider.api_key == "test-key"
        assert provider.model == "test-model"

        provider.close()


class TestPrompt:
    def test_prompt_contains_ticket_policy_and_documentation(self):
        prompt = OpenRouterProvider._prompt(
            make_ticket(),
            make_policy(),
            [make_chunk()],
        )

        assert "TICKET" in prompt
        assert "Create a test" in prompt
        assert "How do I create a test?" in prompt

        assert "POLICY" in prompt
        assert "product_issue" in prompt
        assert "Should Escalate: False" in prompt

        assert "RETRIEVED DOCUMENTATION" in prompt
        assert "Create a Test" in prompt
        assert "Screen / Managing Tests" in prompt

    def test_prompt_handles_no_documents(self):
        prompt = OpenRouterProvider._prompt(
            make_ticket(),
            make_policy(),
            [],
        )

        assert "No documentation retrieved." in prompt


class TestParsing:
    def test_parse_valid_json(self):
        result = OpenRouterProvider._parse(
            make_llm_json()
        )

        assert isinstance(result, AgentResult)
        assert result.status == "replied"
        assert result.product_area == "Screen / Managing Tests"
        assert result.request_type == "product_issue"

    def test_parse_json_code_fence(self):
        text = f"```json\n{make_llm_json()}\n```"

        result = OpenRouterProvider._parse(text)

        assert isinstance(result, AgentResult)
        assert result.status == "replied"

    def test_parse_json_embedded_in_text(self):
        text = (
            "Here is the result:\n"
            + make_llm_json()
            + "\nDone."
        )

        result = OpenRouterProvider._parse(text)

        assert result.status == "replied"

    def test_parse_invalid_json(self):
        with pytest.raises(
            ValueError,
            match="invalid JSON",
        ):
            OpenRouterProvider._parse(
                "not valid json"
            )


class TestCall:
    def test_call_sends_openrouter_request(self):
        provider = OpenRouterProvider(
            "test-key",
            model="test-model",
        )

        provider.client = MagicMock()

        response = make_response(
            make_llm_json()
        )
        provider.client.post.return_value = response

        result = provider._call("test prompt")

        assert result == make_llm_json()

        provider.client.post.assert_called_once()

        call = provider.client.post.call_args

        assert (
            call.args[0]
            == "https://openrouter.ai/api/v1/chat/completions"
        )

        assert (
            call.kwargs["headers"]["Authorization"]
            == "Bearer test-key"
        )

        payload = call.kwargs["json"]

        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0
        assert payload["response_format"] == {
            "type": "json_object"
        }

        provider.close()


class TestGenerate:
    def test_generate_returns_agent_result(self):
        provider = OpenRouterProvider(
            "test-key",
            max_retries=0,
        )

        provider._call = MagicMock(
            return_value=make_llm_json()
        )

        result = provider.generate(
            make_ticket(),
            make_policy(),
            [make_chunk()],
        )

        assert isinstance(result, AgentResult)
        assert result.status == "replied"
        assert result.request_type == "product_issue"

        provider._call.assert_called_once()

        provider.close()

    def test_generate_retries_failed_response(self):
        provider = OpenRouterProvider(
            "test-key",
            max_retries=1,
        )

        provider._call = MagicMock(
            side_effect=[
                ValueError("invalid response"),
                make_llm_json(),
            ]
        )

        result = provider.generate(
            make_ticket(),
            make_policy(),
            [make_chunk()],
        )

        assert result.status == "replied"
        assert provider._call.call_count == 2

        provider.close()

    def test_generate_raises_after_retries(self):
        provider = OpenRouterProvider(
            "test-key",
            max_retries=1,
        )

        provider._call = MagicMock(
            side_effect=ValueError("invalid response")
        )

        with pytest.raises(
            ValueError,
            match="invalid response",
        ):
            provider.generate(
                make_ticket(),
                make_policy(),
                [make_chunk()],
            )

        assert provider._call.call_count == 2

        provider.close()


class TestContextManager:
    def test_provider_can_be_used_as_context_manager(self):
        provider = OpenRouterProvider("test-key")
        provider.close = MagicMock()

        with provider as active:
            assert active is provider

        provider.close.assert_called_once()
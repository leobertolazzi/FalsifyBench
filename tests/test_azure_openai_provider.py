import pytest

from src.benchmark.engine import (
    DEFAULT_AZURE_OPENAI_API_VERSION,
    OracleAgent,
    PlayerAgent,
    _detect_provider_for_model,
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _StrictAzureCreate:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if "reasoning" in kwargs:
            raise TypeError("reasoning is not supported")
        if "response_format" in kwargs:
            raise TypeError("response_format is not supported")
        if "max_tokens" in kwargs:
            raise TypeError("max_tokens is not supported with this model use max_completion_tokens")
        return _FakeResponse('{"conforms": true, "explanation": "ok"}')


class _FakeChatCompletions:
    def __init__(self, create_fn):
        self.create = create_fn


class _FakeChat:
    def __init__(self, create_fn):
        self.completions = _FakeChatCompletions(create_fn)


class _FakeAzureClient:
    def __init__(self, create_fn):
        self.chat = _FakeChat(create_fn)


def test_detect_provider_routes_gpt_5_2_chat_to_azure():
    assert _detect_provider_for_model("gpt-5.2-chat") == "azure"
    assert _detect_provider_for_model("gpt-5-nano-2025-08-07") == "openai"


def test_player_agent_uses_azure_defaults(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    player = PlayerAgent(model="gpt-5.2-chat")

    assert player.provider == "azure"
    assert player.api_key == "test-azure-key"
    assert player.azure_endpoint == "https://example.openai.azure.com/"
    assert player.api_version == DEFAULT_AZURE_OPENAI_API_VERSION


def test_oracle_agent_uses_azure_provider(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    oracle = OracleAgent(target_property="animal", model="gpt-5.2-chat")

    assert oracle.provider == "azure"
    assert oracle.api_version == "2025-01-01-preview"


def test_azure_model_requires_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT"):
        PlayerAgent(model="gpt-5.2-chat")


def test_player_agent_fallback_keeps_max_completion_tokens_for_azure(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")

    player = PlayerAgent(model="gpt-5.2-chat")
    create_fn = _StrictAzureCreate()
    player.client = _FakeAzureClient(create_fn)
    player.conversation_history = [{"role": "user", "content": "hello"}]

    result = player.take_turn("system", low_reasoning_mode=True)

    assert result["conforms"] is True
    assert len(create_fn.calls) == 2
    assert "reasoning" in create_fn.calls[0]
    assert "max_completion_tokens" in create_fn.calls[1]
    assert "max_tokens" not in create_fn.calls[1]


def test_oracle_chat_create_fallback_keeps_max_completion_tokens_for_azure(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")

    oracle = OracleAgent(target_property="animal", model="gpt-5.2-chat")
    create_fn = _StrictAzureCreate()
    oracle.client = _FakeAzureClient(create_fn)

    response = oracle._chat_create(
        prompt="hello",
        max_completion_tokens=123,
        force_json=True,
        reasoning_effort="low",
    )

    assert response.choices[0].message.content == '{"conforms": true, "explanation": "ok"}'
    assert len(create_fn.calls) == 2
    assert "response_format" in create_fn.calls[0]
    assert "max_completion_tokens" in create_fn.calls[1]
    assert "max_tokens" not in create_fn.calls[1]
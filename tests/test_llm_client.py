from mcp_chatbot.core.llm_client import LLMClient


def _client() -> LLMClient:
    return LLMClient(api_key="local", model="test-model", temp=0.0, frontend="test")


# ── Phase 11: tool_choice ──────────────────────────────────────────────────────

def test_tool_choice_defaults_to_auto():
    c = _client()
    tools = [{"type": "function", "function": {"name": "fn", "parameters": {}}}]
    _, _, payload = c._build_payload([], tools, stream=False)
    assert payload["tool_choice"] == "auto"


def test_tool_choice_forwarded_in_payload():
    c = _client()
    tools = [{"type": "function", "function": {"name": "plan_tasks", "parameters": {}}}]
    forced = {"type": "function", "function": {"name": "plan_tasks"}}
    _, _, payload = c._build_payload([], tools, stream=False, tool_choice=forced)
    assert payload["tool_choice"] == forced


def test_tool_choice_none_without_tools():
    """tool_choice is not injected when no tools are provided."""
    c = _client()
    _, _, payload = c._build_payload([], None, stream=False)
    assert "tool_choice" not in payload


def test_tool_choice_forwarded_via_stream_response_signature():
    """stream_response accepts tool_choice kwarg (compile-time check)."""
    import inspect
    sig = inspect.signature(LLMClient.stream_response)
    assert "tool_choice" in sig.parameters


def test_tool_choice_forwarded_via_get_response_signature():
    """get_response accepts tool_choice kwarg (compile-time check)."""
    import inspect
    sig = inspect.signature(LLMClient.get_response)
    assert "tool_choice" in sig.parameters


# ── Phase 11.3: response_format ───────────────────────────────────────────────

def test_response_format_forwarded_on_local_path():
    c = _client()  # api_key="local"
    fmt = {"type": "json_object"}
    _, _, payload = c._build_payload([], None, stream=False, response_format=fmt)
    assert payload.get("response_format") == fmt


def test_response_format_not_forwarded_on_remote_path():
    c = LLMClient(api_key="gsk_xxxx", model="m", temp=0.0, frontend="test")
    fmt = {"type": "json_object"}
    _, _, payload = c._build_payload([], None, stream=False, response_format=fmt)
    assert "response_format" not in payload


def test_response_format_not_injected_when_none():
    c = _client()
    _, _, payload = c._build_payload([], None, stream=False)
    assert "response_format" not in payload


def test_response_format_forwarded_via_stream_signature():
    import inspect
    sig = inspect.signature(LLMClient.stream_response)
    assert "response_format" in sig.parameters


def test_response_format_forwarded_via_get_response_signature():
    import inspect
    sig = inspect.signature(LLMClient.get_response)
    assert "response_format" in sig.parameters

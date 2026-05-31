import asyncio
from unittest.mock import MagicMock, patch
from mcp_chatbot.core.session import ChatSession, COMPRESSION_KEEP_TURNS
from mcp_chatbot.core.llm_client import LLMClient


def make_session(summary_text="This is a summary."):
    """ChatSession with mocked LLM that returns summary_text from get_response."""
    with patch("mcp_chatbot.core.session.load_settings", return_value={}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"), \
         patch("mcp_chatbot.core.session.EpisodicStore"):
        llm = MagicMock(spec=LLMClient)
        llm.max_tokens = 4096
        llm.get_response.return_value = {"role": "assistant", "content": summary_text}
        s = ChatSession([], llm)
        s._is_initialized = True
        s._skills_manager.get_index.return_value = []
        s.task_manager.is_empty.return_value = True
        s.task_manager.render_block.return_value = ""
        s.task_manager.tool_schemas = []
        s.task_manager.tool_names = set()
        return s


def _msgs(n_turns: int) -> list[dict]:
    """System prompt + n_turns of user/assistant pairs."""
    msgs = [{"role": "user", "content": "You are a helpful assistant."}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"User {i + 1}"})
        msgs.append({"role": "assistant", "content": f"Reply {i + 1}"})
    return msgs


# ── _summarize ────────────────────────────────────────────────────────────────

def test_summarize_returns_llm_content():
    session = make_session("The user asked about Python.")
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    result = session._summarize(messages)
    assert result == "The user asked about Python."


def test_summarize_calls_get_response_with_conversation_text():
    session = make_session("ok")
    messages = [
        {"role": "user", "content": "what is 2+2"},
        {"role": "assistant", "content": "4"},
    ]
    session._summarize(messages)
    call_args = session.llm_client.get_response.call_args[0][0]
    prompt_text = call_args[0]["content"]
    assert "USER: what is 2+2" in prompt_text
    assert "ASSISTANT: 4" in prompt_text


def test_summarize_excludes_none_content():
    session = make_session("ok")
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "content": None},
        {"role": "assistant", "content": "world"},
    ]
    session._summarize(messages)
    call_args = session.llm_client.get_response.call_args[0][0]
    prompt_text = call_args[0]["content"]
    assert "None" not in prompt_text


def test_summarize_returns_none_on_exception():
    session = make_session()
    session.llm_client.get_response.side_effect = Exception("API error")
    result = session._summarize([{"role": "user", "content": "hello"}])
    assert result is None


def test_summarize_returns_none_on_empty_content():
    session = make_session("")
    result = session._summarize([{"role": "user", "content": "hello"}])
    assert result is None


def test_summarize_returns_none_on_whitespace_content():
    session = make_session("   ")
    result = session._summarize([{"role": "user", "content": "hello"}])
    assert result is None


def test_summarize_returns_none_on_empty_messages():
    session = make_session("summary")
    result = session._summarize([])
    assert result is None
    session.llm_client.get_response.assert_not_called()


# ── _compress_history ─────────────────────────────────────────────────────────

def test_compress_history_noop_when_exactly_n_turns():
    session = make_session("summary")
    session.messages = _msgs(COMPRESSION_KEEP_TURNS)
    original = list(session.messages)
    asyncio.run(session._compress_history())
    assert session.messages == original


def test_compress_history_noop_when_fewer_than_n_turns():
    session = make_session("summary")
    session.messages = _msgs(COMPRESSION_KEEP_TURNS - 1)
    original = list(session.messages)
    asyncio.run(session._compress_history())
    assert session.messages == original


def test_compress_history_noop_on_empty_history():
    session = make_session("summary")
    session.messages = [{"role": "user", "content": "system only"}]
    asyncio.run(session._compress_history())
    assert len(session.messages) == 1


def test_compress_history_fires_on_n_plus_one_turns():
    session = make_session("Summary here.")
    session.messages = _msgs(COMPRESSION_KEEP_TURNS + 1)
    asyncio.run(session._compress_history())
    assert session.messages[1]["role"] == "user"
    assert session.messages[1]["content"] == "[Summary]: Summary here."


def test_compress_history_system_message_preserved():
    session = make_session("summary")
    original_system = _msgs(COMPRESSION_KEEP_TURNS + 1)[0]
    session.messages = _msgs(COMPRESSION_KEEP_TURNS + 1)
    asyncio.run(session._compress_history())
    assert session.messages[0]["content"] == original_system["content"]


def test_compress_history_keeps_last_n_user_turns():
    session = make_session("summary")
    session.messages = _msgs(COMPRESSION_KEEP_TURNS + 2)
    asyncio.run(session._compress_history())
    # messages[0]=system, messages[1]=summary, messages[2:]=kept turns
    kept = session.messages[2:]
    user_msgs = [m for m in kept if m.get("role") == "user"]
    assert len(user_msgs) == COMPRESSION_KEEP_TURNS


def test_compress_history_kept_turns_are_last_n():
    session = make_session("summary")
    n = COMPRESSION_KEEP_TURNS
    session.messages = _msgs(n + 2)
    asyncio.run(session._compress_history())
    kept = session.messages[2:]
    user_contents = [m["content"] for m in kept if m.get("role") == "user"]
    total_turns = n + 2
    expected = [f"User {i}" for i in range(total_turns - n + 1, total_turns + 1)]
    assert user_contents == expected


def test_compress_history_noop_on_summarize_failure():
    session = make_session()
    session.llm_client.get_response.side_effect = Exception("API down")
    session.messages = _msgs(COMPRESSION_KEEP_TURNS + 1)
    original = list(session.messages)
    asyncio.run(session._compress_history())
    assert session.messages == original


def test_compress_history_noop_on_empty_summary():
    session = make_session("")  # get_response returns empty content
    session.messages = _msgs(COMPRESSION_KEEP_TURNS + 1)
    original = list(session.messages)
    asyncio.run(session._compress_history())
    assert session.messages == original

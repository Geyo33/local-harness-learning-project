from unittest.mock import AsyncMock, MagicMock, patch
from mcp_chatbot.core.session import ChatSession, COMPRESSION_KEEP_TURNS
from mcp_chatbot.core.llm_client import LLMClient
import asyncio


def _msgs(n_turns: int) -> list[dict]:
    msgs = [{"role": "user", "content": "System prompt."}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"User {i + 1}"})
        msgs.append({"role": "assistant", "content": f"Reply {i + 1}"})
    return msgs


def make_session(summary_text="Summary.", mock_store=None):
    store = mock_store if mock_store is not None else MagicMock()
    store.get_recent.return_value = []
    store.get_key_facts.return_value = []    # ← stub so legacy tests that check get_key_facts are clean
    store.get_pinned_facts.return_value = [] # ← stub so "no pinned facts → block omitted" tests work
    with patch("mcp_chatbot.core.session.load_settings", return_value={}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"), \
         patch("mcp_chatbot.core.session.EpisodicStore", return_value=store):
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
    return s, store


# ── init ──────────────────────────────────────────────────────────────────────

def test_session_has_session_id():
    s, _ = make_session()
    assert isinstance(s._session_id, str)
    assert len(s._session_id) == 36  # UUID4 "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"


def test_session_ids_are_unique():
    s1, _ = make_session()
    s2, _ = make_session()
    assert s1._session_id != s2._session_id


def test_episodic_store_initialized():
    s, store = make_session()
    assert s.episodic_store is store


def test_episodic_store_none_when_init_raises(caplog):
    import logging
    with patch("mcp_chatbot.core.session.load_settings", return_value={}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"), \
         patch("mcp_chatbot.core.session.EpisodicStore", side_effect=Exception("DB error")):
        llm = MagicMock(spec=LLMClient)
        llm.max_tokens = 4096
        with caplog.at_level(logging.WARNING):
            s = ChatSession([], llm)
    assert s.episodic_store is None
    assert any("DB error" in r.message for r in caplog.records)


# ── build_system_message memory injection ─────────────────────────────────────

def test_build_system_message_injects_memory_block():
    s, store = make_session()
    store.get_recent.return_value = [
        {
            "session_id": "sess-1",
            "created_at": 1747612800.0,
            "summary": "User worked on MCP client with Gradio UI.",
            "source": "agent",
        }
    ]
    with patch("mcp_chatbot.core.session.load_settings", return_value={"memory_episodes": 3}):
        asyncio.run(s.build_system_message())
    system_content = s.messages[0]["content"]
    assert "[Memory]:" in system_content
    assert "User worked on MCP client with Gradio UI." in system_content


def test_build_system_message_resets_history_by_default():
    s, _ = make_session()
    s.messages = [{"role": "user", "content": "old system"},
                  {"role": "user", "content": "U1"},
                  {"role": "assistant", "content": "A1"}]
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    # Default contract: history is discarded, only the fresh system prompt remains.
    assert len(s.messages) == 1
    assert s.messages[0]["role"] == "user"


def test_build_system_message_preserve_history_keeps_tail():
    s, _ = make_session()
    s.messages = [{"role": "user", "content": "old system"},
                  {"role": "user", "content": "U1"},
                  {"role": "assistant", "content": "A1"}]
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message(preserve_history=True))
    # System prompt rebuilt in place; conversation tail retained.
    assert [m["content"] for m in s.messages[1:]] == ["U1", "A1"]
    assert s.messages[0]["content"] != "old system"  # system prompt was rebuilt


def test_build_system_message_no_memory_when_store_none():
    s, _ = make_session()
    s.episodic_store = None
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    assert "[Memory]:" not in s.messages[0]["content"]


def test_build_system_message_no_memory_when_empty():
    s, store = make_session()
    store.get_recent.return_value = []
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    assert "[Memory]:" not in s.messages[0]["content"]


def test_build_system_message_uses_memory_episodes_setting():
    s, store = make_session()
    store.get_recent.return_value = []
    with patch("mcp_chatbot.core.session.load_settings", return_value={"memory_episodes": 7}):
        asyncio.run(s.build_system_message())
    store.get_recent.assert_called_with(7)


def test_build_system_message_defaults_to_3_episodes():
    s, store = make_session()
    store.get_recent.return_value = []
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    store.get_recent.assert_called_with(3)


# ── build_system_message key facts injection ──────────────────────────────────

def test_build_system_message_injects_key_facts_block():
    s, store = make_session()
    store.get_pinned_facts.return_value = [
        {"id": 3, "fact": "Project uses Python 3.11", "created_at": 1747612800.0, "source": "agent"},
        {"id": 7, "fact": "User prefers no docstrings", "created_at": 1747612801.0, "source": "agent"},
    ]
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    content = s.messages[0]["content"]
    assert "[Pinned Facts]:" in content
    assert "#3 Project uses Python 3.11" in content
    assert "#7 User prefers no docstrings" in content


def test_build_system_message_no_key_facts_block_when_empty():
    s, store = make_session()
    store.get_key_facts.return_value = []
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    assert "[Key Facts]:" not in s.messages[0]["content"]


def test_build_system_message_no_key_facts_block_when_store_none():
    s, _ = make_session()
    s.episodic_store = None
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    assert "[Key Facts]:" not in s.messages[0]["content"]


def test_build_system_message_key_facts_show_ids():
    s, store = make_session()
    store.get_pinned_facts.return_value = [
        {"id": 12, "fact": "LM Studio runs on port 1234", "created_at": 1747612800.0, "source": "agent"},
    ]
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    assert "#12 LM Studio runs on port 1234" in s.messages[0]["content"]


def test_build_system_message_uses_memory_key_facts_setting():
    s, store = make_session()
    with patch("mcp_chatbot.core.session.load_settings", return_value={"memory_key_facts": 5}):
        asyncio.run(s.build_system_message())
    store.get_pinned_facts.assert_called_with(5)


def test_build_system_message_defaults_to_20_key_facts():
    s, store = make_session()
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    store.get_pinned_facts.assert_called_with(20)


# ── _compress_history episode write ───────────────────────────────────────────

def test_compress_history_writes_episode_to_store():
    store = MagicMock()
    store.get_recent.return_value = []
    s, _ = make_session("Compression summary.", mock_store=store)
    s.messages = _msgs(COMPRESSION_KEEP_TURNS + 1)

    asyncio.run(s._compress_history())

    store.add_episode.assert_called_once_with(
        s._session_id, "Compression summary.", "agent"
    )


def test_compress_history_no_episode_when_store_none():
    s, _ = make_session("Summary.")
    s.episodic_store = None
    s.messages = _msgs(COMPRESSION_KEEP_TURNS + 1)
    asyncio.run(s._compress_history())  # must not raise


def test_compress_history_no_episode_when_summary_none():
    store = MagicMock()
    store.get_recent.return_value = []
    s, _ = make_session(mock_store=store)
    s.llm_client.get_response.side_effect = Exception("API down")
    s.messages = _msgs(COMPRESSION_KEEP_TURNS + 1)

    asyncio.run(s._compress_history())

    store.add_episode.assert_not_called()


def test_compress_history_no_episode_when_too_few_turns():
    store = MagicMock()
    store.get_recent.return_value = []
    s, _ = make_session("Summary.", mock_store=store)
    s.messages = _msgs(COMPRESSION_KEEP_TURNS)  # exactly at limit — no compression fires

    asyncio.run(s._compress_history())

    store.add_episode.assert_not_called()


# ── save_episode ──────────────────────────────────────────────────────────────

def test_save_episode_writes_to_store():
    store = MagicMock()
    store.get_recent.return_value = []
    s, _ = make_session("End-of-session summary.", mock_store=store)
    s.messages = _msgs(5)  # 10 messages in history — meets ≥10 threshold

    asyncio.run(s.save_episode())

    store.add_episode.assert_called_once()
    session_id, summary, source = store.add_episode.call_args[0]
    assert session_id == s._session_id
    assert summary == "End-of-session summary."
    assert source == "agent"


def test_save_episode_skips_when_store_none():
    s, _ = make_session()
    s.episodic_store = None
    s.messages = _msgs(5)
    asyncio.run(s.save_episode())  # must not raise


def test_save_episode_skips_when_too_few_user_turns():
    store = MagicMock()
    store.get_recent.return_value = []
    s, _ = make_session(mock_store=store)
    s.messages = _msgs(2)  # 4 messages in history, below threshold of 10

    asyncio.run(s.save_episode())

    store.add_episode.assert_not_called()


def test_save_episode_skips_when_summarize_fails():
    store = MagicMock()
    store.get_recent.return_value = []
    s, _ = make_session(mock_store=store)
    s.llm_client.get_response.side_effect = Exception("API down")
    s.messages = _msgs(5)

    asyncio.run(s.save_episode())

    store.add_episode.assert_not_called()


def test_save_episode_skips_when_summarize_returns_none():
    store = MagicMock()
    store.get_recent.return_value = []
    s, _ = make_session("", mock_store=store)  # empty string → _summarize returns None
    s.messages = _msgs(5)

    asyncio.run(s.save_episode())

    store.add_episode.assert_not_called()


# ── MemoryManager wiring ──────────────────────────────────────────────────────

def test_memory_manager_initialized_with_store():
    s, store = make_session()
    assert s.memory_manager is not None
    assert s.memory_manager._store is store


def test_memory_manager_none_when_store_fails():
    with patch("mcp_chatbot.core.session.load_settings", return_value={}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"), \
         patch("mcp_chatbot.core.session.EpisodicStore", side_effect=Exception("DB error")):
        llm = MagicMock(spec=LLMClient)
        llm.max_tokens = 4096
        s = ChatSession([], llm)
    assert s.memory_manager is None


def test_build_system_message_suppresses_search_when_mcp_active():
    """MemoryManager's search_memory schema excluded when MCP server provides it."""
    s, _ = make_session()

    mock_tool = MagicMock()
    mock_tool.name = "search_memory"
    mock_tool.to_openai_schema.return_value = {
        "type": "function",
        "function": {"name": "search_memory", "description": "mcp version", "parameters": {}},
    }
    mock_server = MagicMock()
    mock_server.name = "memory-retrieval"
    mock_server.timeout = 30.0
    mock_server.list_tools = AsyncMock(return_value=[mock_tool])

    s.servers = [mock_server]
    s.active_servers = {"memory-retrieval": True}

    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())

    schema_names = [t["function"]["name"] for t in s._tool_schemas]
    # Exactly one search_memory (from MCP server), not duplicated from MemoryManager
    assert schema_names.count("search_memory") == 1
    # Write tools still present
    assert "remember_fact" in schema_names
    assert "update_fact" in schema_names
    assert "forget_fact" in schema_names


def test_build_system_message_includes_search_when_mcp_inactive():
    """MemoryManager's search_memory schema present when no MCP server provides it."""
    s, _ = make_session()
    s.servers = []
    s.active_servers = {}

    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())

    schema_names = [t["function"]["name"] for t in s._tool_schemas]
    assert "search_memory" in schema_names


def test_execute_tool_call_routes_search_memory_to_mcp_when_active():
    """search_memory call falls through to MCP server dispatch, not MemoryManager."""
    s, _ = make_session()

    mock_server = MagicMock()
    mock_server.name = "memory-retrieval"

    # Wire search_memory to mock_server in the tool map
    s._tool_server_map = {"search_memory": mock_server}
    s._tool_timeout_map = {"search_memory": 30.0}

    # MemoryManager.execute should NOT be called
    s.memory_manager.execute = MagicMock()

    # The MCP server path: server.execute_tool is an async method
    mock_result = MagicMock()
    mock_result.content = "semantic results here"
    mock_server.execute_tool = AsyncMock(return_value=mock_result)

    tool_call = {
        "id": "tc1",
        "function": {"name": "search_memory", "arguments": '{"query": "python"}'},
    }
    asyncio.run(s._execute_tool_call(tool_call))

    s.memory_manager.execute.assert_not_called()
    mock_server.execute_tool.assert_called_once()


def test_build_system_message_includes_enhanced_note_when_mcp_active():
    """System message contains enhanced mode note when MCP server provides search_memory."""
    s, _ = make_session()

    mock_tool = MagicMock()
    mock_tool.name = "search_memory"
    mock_tool.to_openai_schema.return_value = {
        "type": "function",
        "function": {"name": "search_memory", "description": "mcp version", "parameters": {}},
    }
    mock_server = MagicMock()
    mock_server.name = "memory-retrieval"
    mock_server.timeout = 30.0
    mock_server.list_tools = AsyncMock(return_value=[mock_tool])

    s.servers = [mock_server]
    s.active_servers = {"memory-retrieval": True}

    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())

    assert "connected MCP server's retrieval" in s.messages[0]["content"]


# ── load_episodes flag ────────────────────────────────────────────────────────

def test_build_system_message_skips_episodes_when_flag_false():
    """Episodes must not appear when load_episodes=False, even if store has data."""
    s, store = make_session()
    store.get_recent.return_value = [
        {
            "session_id": "sess-1",
            "created_at": 1747612800.0,
            "summary": "Previous session work.",
            "source": "agent",
        }
    ]
    s.load_episodes = False
    with patch("mcp_chatbot.core.session.load_settings", return_value={"memory_episodes": 3}):
        asyncio.run(s.build_system_message())
    assert "[Memory]:" not in s.messages[0]["content"]
    store.get_recent.assert_not_called()


def test_build_system_message_always_injects_key_facts_when_flag_false():
    """Key facts must still appear when load_episodes=False."""
    s, store = make_session()
    store.get_pinned_facts.return_value = [
        {"id": 1, "fact": "User prefers Python", "created_at": 1747612800.0, "source": "user"},
    ]
    s.load_episodes = False
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    assert "[Pinned Facts]:" in s.messages[0]["content"]
    assert "#1 User prefers Python" in s.messages[0]["content"]


def test_system_prompt_uses_pinned_facts():
    """build_system_message injects pinned facts (get_pinned_facts) not unpinned ones (get_key_facts)."""
    s, store = make_session()
    store.get_pinned_facts.return_value = [
        {"id": 1, "fact": "pinned alpha", "created_at": 0, "source": "user"}
    ]
    store.get_key_facts.return_value = [
        {"id": 2, "fact": "unpinned beta", "created_at": 0, "source": "agent"}
    ]
    with patch("mcp_chatbot.core.session.load_settings", return_value={}):
        asyncio.run(s.build_system_message())
    assert "pinned alpha" in s.messages[0]["content"]
    assert "unpinned beta" not in s.messages[0]["content"]

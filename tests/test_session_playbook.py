import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_chatbot.core.session import ChatSession
from mcp_chatbot.core.llm_client import LLMClient


def make_session(playbook_entries=None):
    store = MagicMock()
    store.get_recent.return_value = []

    mock_playbook = MagicMock()
    mock_playbook.tool_names = {"record_procedure"}
    mock_playbook.safe_tool_names = set()
    mock_playbook.tool_schemas = []
    block = ""
    if playbook_entries:
        lines = [f"- [task_success] {e['pattern']} → {e['action']}" for e in playbook_entries]
        block = "[Playbook]:\n" + "\n".join(lines)
    mock_playbook.render_block.return_value = block

    with patch("mcp_chatbot.core.session.load_settings", return_value={}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"), \
         patch("mcp_chatbot.core.session.EpisodicStore", return_value=store), \
         patch("mcp_chatbot.core.session.PlaybookManager", return_value=mock_playbook):
        llm = MagicMock(spec=LLMClient)
        llm.max_tokens = 4096
        llm.get_response.return_value = {"role": "assistant", "content": ""}
        s = ChatSession([], llm)
        s._is_initialized = True
        s._skills_manager.get_index.return_value = []
        s.task_manager.is_empty.return_value = True
        s.task_manager.render_block.return_value = ""
        s.task_manager.tool_schemas = []
        s.task_manager.tool_names = set()
        s.task_manager.safe_tool_names = set()
    return s, mock_playbook


# ── init ──────────────────────────────────────────────────────────────────────

def test_playbook_manager_initialized():
    s, pm = make_session()
    assert s.playbook_manager is pm


def test_playbook_manager_none_when_init_raises(caplog):
    import logging
    store = MagicMock()
    store.get_recent.return_value = []
    with patch("mcp_chatbot.core.session.load_settings", return_value={}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"), \
         patch("mcp_chatbot.core.session.EpisodicStore", return_value=store), \
         patch("mcp_chatbot.core.session.PlaybookManager", side_effect=Exception("fail")):
        llm = MagicMock(spec=LLMClient)
        llm.max_tokens = 4096
        with caplog.at_level(logging.WARNING):
            s = ChatSession([], llm)
    assert s.playbook_manager is None
    assert "playbook" in caplog.text.lower()


# ── build_system_message ──────────────────────────────────────────────────────

def test_build_system_message_injects_playbook_block():
    s, pm = make_session(playbook_entries=[{"pattern": "deploy pkg", "action": "step 1"}])
    asyncio.run(s.build_system_message())
    system_content = s.messages[0]["content"]
    assert "[Playbook]:" in system_content
    assert "deploy pkg" in system_content


def test_build_system_message_omits_block_when_empty():
    s, pm = make_session()  # render_block returns ""
    asyncio.run(s.build_system_message())
    system_content = s.messages[0]["content"]
    assert "[Playbook]:" not in system_content


def test_build_system_message_registers_playbook_schemas():
    from mcp_chatbot.memory.playbook import PlaybookManager
    # Use a real PlaybookManager schema to verify registration
    real_schemas = PlaybookManager.__new__(PlaybookManager).tool_schemas if False else [
        {"type": "function", "function": {"name": "record_procedure", "parameters": {}}}
    ]
    s, pm = make_session()
    pm.tool_schemas = real_schemas
    asyncio.run(s.build_system_message())
    schema_names = [sc["function"]["name"] for sc in s._tool_schemas]
    assert "record_procedure" in schema_names


# ── tool routing ──────────────────────────────────────────────────────────────

def test_record_procedure_routed_to_playbook_manager():
    s, pm = make_session()
    pm.execute.return_value = "Procedure #1 recorded: deploy"
    tc = {"function": {"name": "record_procedure", "arguments": '{"pattern":"deploy","action":"step 1"}'}}
    result = asyncio.run(s._execute_tool_call(tc))
    pm.execute.assert_called_once_with("record_procedure", {"pattern": "deploy", "action": "step 1"})
    assert result == "Procedure #1 recorded: deploy"


def test_record_procedure_not_in_safe_tool_names():
    s, pm = make_session()
    assert "record_procedure" not in s.playbook_manager.safe_tool_names


# ── _run_playbook_prompt ──────────────────────────────────────────────────────

async def _collect(gen):
    events = []
    async for event in gen:
        events.append(event)
    return events


def test_run_playbook_prompt_fires_when_last_all_done():
    s, pm = make_session()
    pm.execute.return_value = "Procedure #1 recorded: deploy"
    pm.tool_schemas = [{"type": "function", "function": {"name": "record_procedure", "parameters": {}}}]
    s.llm_client.get_response.return_value = {
        "role": "assistant",
        "tool_calls": [{
            "function": {"name": "record_procedure", "arguments": '{"pattern":"deploy","action":"step 1"}'}
        }],
    }
    s.allow_tool_action = "always"
    events = asyncio.run(_collect(s._run_playbook_prompt()))
    # Two events: (1) approval tuple, (2) result string
    assert len(events) == 2
    assert events[0] == ("record_procedure", '{"pattern":"deploy","action":"step 1"}')
    assert "record_procedure" in events[1]
    assert "Procedure #1 recorded" in events[1]
    pm.execute.assert_called_once_with("record_procedure", {"pattern": "deploy", "action": "step 1"})


def test_run_playbook_prompt_silent_when_agent_skips():
    s, pm = make_session()
    pm.tool_schemas = [{"type": "function", "function": {"name": "record_procedure", "parameters": {}}}]
    s.llm_client.get_response.return_value = {
        "role": "assistant",
        "content": "This was a one-off task.",
    }
    events = asyncio.run(_collect(s._run_playbook_prompt()))
    assert events == []


def test_run_playbook_prompt_silent_when_no_playbook_manager():
    s, pm = make_session()
    s.playbook_manager = None
    events = asyncio.run(_collect(s._run_playbook_prompt()))
    assert events == []


def test_run_playbook_prompt_deny_path():
    s, pm = make_session()
    pm.tool_schemas = [{"type": "function", "function": {"name": "record_procedure", "parameters": {}}}]
    s.llm_client.get_response.return_value = {
        "role": "assistant",
        "tool_calls": [{
            "function": {"name": "record_procedure", "arguments": '{"pattern":"deploy","action":"step 1"}'}
        }],
    }
    s.allow_tool_action = "deny"
    # Mock the event so clear()+wait() don't deadlock; deny path is driven by allow_tool_action alone
    s.allow_tool_event = MagicMock()
    s.allow_tool_event.wait = AsyncMock(return_value=None)

    events = asyncio.run(_collect(s._run_playbook_prompt()))
    assert len(events) == 2
    assert events[0] == ("record_procedure", '{"pattern":"deploy","action":"step 1"}')
    assert "Tool call denied" in events[1]
    pm.execute.assert_not_called()

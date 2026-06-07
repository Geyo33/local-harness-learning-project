import asyncio
import json
import json as _json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


PLAN_TASKS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "plan_tasks",
        "description": "Create initial task plan.",
        "parameters": {
            "type": "object",
            "properties": {"titles": {"type": "array", "items": {"type": "string"}}},
            "required": ["titles"],
        },
    },
}


def _plan_tasks_response(titles=None):
    titles = titles or ["Task A", "Task B"]
    yield {
        "type": "tool_calls_final",
        "data": [{
            "id": "tc_plan",
            "type": "function",
            "function": {
                "name": "plan_tasks",
                "arguments": _json.dumps({"titles": titles}),
            },
        }],
    }


# ── Server.timeout ────────────────────────────────────────────────────────────

def test_server_reads_timeout_from_config():
    from mcp_chatbot.core.server import Server
    server = Server("s", {"command": "uvx", "args": [], "timeout": 60.0})
    assert server.timeout == 60.0


def test_server_timeout_defaults_to_none():
    from mcp_chatbot.core.server import Server
    server = Server("s", {"command": "uvx", "args": []})
    assert server.timeout is None


# ── helpers ───────────────────────────────────────────────────────────────────

def make_session(servers=None):
    """Return a ChatSession with mocked dependencies, ready for testing."""
    from mcp_chatbot.core.session import ChatSession
    from mcp_chatbot.core.llm_client import LLMClient

    with patch("mcp_chatbot.core.session.load_settings", return_value={}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"):
        llm = MagicMock(spec=LLMClient)
        llm.max_tokens = 4096
        s = ChatSession(servers or [], llm)
        s._is_initialized = True
        s._skills_manager.get_index.return_value = []
        s.task_manager.is_empty.return_value = True
        s.task_manager.render_block.return_value = ""
        s.task_manager.tool_schemas = []
        s.task_manager.tool_names = set()
        return s


def make_mock_server(name="srv", timeout=None):
    """Return a mock Server with one tool named 'test_tool'."""
    from mcp_chatbot.core.server import Tool
    server = MagicMock()
    server.name = name
    server.timeout = timeout
    tool = Tool("test_tool", "A tool.", {"type": "object", "properties": {}})
    server.list_tools = AsyncMock(return_value=[tool])
    return server


# ── _tool_timeout_map ─────────────────────────────────────────────────────────

def test_timeout_map_uses_server_timeout():
    server = make_mock_server(timeout=45.0)
    session = make_session(servers=[server])
    session.active_servers = {"srv": True}
    asyncio.run(session.build_system_message())
    assert session._tool_timeout_map["test_tool"] == 45.0


def test_timeout_map_uses_default_when_server_timeout_is_none():
    from mcp_chatbot.core.session import DEFAULT_TOOL_TIMEOUT
    server = make_mock_server(timeout=None)
    session = make_session(servers=[server])
    session.active_servers = {"srv": True}
    asyncio.run(session.build_system_message())
    assert session._tool_timeout_map["test_tool"] == DEFAULT_TOOL_TIMEOUT


# ── tool execution timeout ────────────────────────────────────────────────────

def test_execute_tool_call_returns_message_on_timeout():
    """_execute_tool_call returns a timeout message when execute_tool raises TimeoutError."""
    server = MagicMock()
    server.execute_tool = AsyncMock(side_effect=asyncio.TimeoutError())

    session = make_session()
    session._tool_server_map = {"slow_tool": server}
    session._tool_timeout_map = {"slow_tool": 0.05}

    tool_call = {
        "id": "tc1",
        "function": {"name": "slow_tool", "arguments": "{}"},
    }
    result = asyncio.run(session._execute_tool_call(tool_call))
    assert "timed out" in result.lower()
    assert "0.05" in result


def test_execute_tool_timeout_does_not_retry():
    """execute_tool raises TimeoutError immediately without retrying."""
    from mcp_chatbot.core.server import Server

    server = Server("s", {"command": "uvx", "args": []})
    server.session = MagicMock()

    async def slow(*a, **kw):
        await asyncio.sleep(100)

    server.session.call_tool = AsyncMock(side_effect=slow)

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await server.execute_tool("tool", {}, timeout=0.05)
        assert server.session.call_tool.call_count == 1  # no retry on timeout

    asyncio.run(run())


# ── parse retry loop ──────────────────────────────────────────────────────────

def _bad_tool_response(arguments="NOT_VALID_JSON{{{"):
    """Sync generator yielding a single tool_calls_final with malformed args."""
    yield {
        "type": "tool_calls_final",
        "data": [{
            "id": "tc1",
            "type": "function",
            "function": {"name": "some_tool", "arguments": arguments},
        }]
    }


def _good_text_response(text="Done."):
    """Sync generator yielding plain content (no tool calls)."""
    yield {"type": "content", "data": text}


async def _collect(async_gen):
    results = []
    async for item in async_gen:
        results.append(item)
    return results


def test_retries_on_malformed_json_then_succeeds():
    """LLM returns bad JSON twice, then a plain text response — succeeds."""
    from mcp_chatbot.core.session import MAX_PARSE_RETRIES

    session = make_session()

    call_count = 0

    def side_effect(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return _bad_tool_response()
        return _good_text_response("All good.")

    session.llm_client.stream_response.side_effect = side_effect

    results = asyncio.run(_collect(session.handle_user_input("hello")))

    assert call_count == 3
    assert any("All good." in str(r) for r in results)


def test_errors_after_max_retries_on_malformed_json():
    """LLM always returns bad JSON — error emitted after MAX_PARSE_RETRIES."""
    from mcp_chatbot.core.session import MAX_PARSE_RETRIES

    session = make_session()
    session.llm_client.stream_response.side_effect = lambda *a, **kw: _bad_tool_response()

    results = asyncio.run(_collect(session.handle_user_input("hello")))

    error_results = [r for r in results if isinstance(r, str) and "Error" in r]
    assert error_results, f"Expected an error string, got: {results}"
    assert "malformed" in error_results[0].lower() or "invalid" in error_results[0].lower()
    assert session.llm_client.stream_response.call_count == MAX_PARSE_RETRIES + 1


def test_message_history_clean_after_retry_exhaustion():
    """After max retries, the retry messages are not left in conversation history."""
    session = make_session()
    session.llm_client.stream_response.side_effect = lambda *a, **kw: _bad_tool_response()
    initial_msg_count = len(session.messages)

    asyncio.run(_collect(session.handle_user_input("hello")))

    # "hello" user message stays; retry error messages should be trimmed
    # so we expect exactly initial + 1 (the user "hello" message)
    assert len(session.messages) == initial_msg_count + 1


# ── TaskManager routing ───────────────────────────────────────────────────────

def test_execute_tool_call_routes_plan_tasks_to_task_manager():
    session = make_session()
    session.task_manager.execute.return_value = "Plan created with 2 task(s): 1. A, 2. B"
    session.task_manager.tool_names = {"plan_tasks", "update_task", "add_task"}

    tool_call = {
        "id": "tc1",
        "function": {"name": "plan_tasks", "arguments": '{"titles": ["A", "B"]}'},
    }
    result = asyncio.run(session._execute_tool_call(tool_call))
    session.task_manager.execute.assert_called_once_with("plan_tasks", {"titles": ["A", "B"]})
    assert "Plan created" in result


def test_execute_tool_call_routes_update_task_to_task_manager():
    session = make_session()
    session.task_manager.execute.return_value = "Task 1 updated to 'done'."
    session.task_manager.tool_names = {"plan_tasks", "update_task", "add_task"}

    tool_call = {
        "id": "tc2",
        "function": {"name": "update_task", "arguments": '{"id": "1", "status": "done"}'},
    }
    result = asyncio.run(session._execute_tool_call(tool_call))
    session.task_manager.execute.assert_called_once_with("update_task", {"id": "1", "status": "done"})
    assert "done" in result


# ── _inject_task_block ────────────────────────────────────────────────────────

def test_inject_task_block_noop_when_no_task_manager():
    session = make_session()
    session.task_manager = None
    messages = [{"role": "user", "content": "hello"}]
    result = session._inject_task_block(messages)
    assert result == messages


def test_inject_task_block_notice_when_block_empty():
    session = make_session()
    session.task_manager.render_block.return_value = ""
    messages = [{"role": "user", "content": "hello"}]
    result = session._inject_task_block(messages)
    # Empty plan -> LLM is told no task list exists yet (does not mutate input)
    assert "No task list created" in result[0]["content"]
    assert result[0]["content"].endswith("hello")
    assert messages == [{"role": "user", "content": "hello"}]


def test_inject_task_block_prepends_to_string_content():
    session = make_session()
    session.task_manager.render_block.return_value = "<tasks>\n[ ] 1. A\n</tasks>"
    messages = [{"role": "user", "content": "hello"}]
    result = session._inject_task_block(messages)
    assert result[-1]["content"].startswith("(System info - Here's the list of tasks you're working on")
    assert "hello" in result[-1]["content"]


def test_inject_task_block_does_not_mutate_original_messages():
    session = make_session()
    session.task_manager.render_block.return_value = "<tasks>\n[ ] 1. A\n</tasks>"
    messages = [{"role": "user", "content": "hello"}]
    session._inject_task_block(messages)
    assert messages[-1]["content"] == "hello"  # original unchanged


def test_inject_task_block_prepends_to_multimodal_content():
    session = make_session()
    session.task_manager.render_block.return_value = "<tasks>\n[ ] 1. A\n</tasks>"
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    result = session._inject_task_block(messages)
    content = result[-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "(System info - Here's the list of tasks you're working on, never talk about it to the user, use tools to keep it up-to-date([ ]: pending, [~]: in_progress, [x]: done):\n <tasks>\n[ ] 1. A\n</tasks>)"}
    assert content[1] == {"type": "text", "text": "hello"}


def test_inject_task_block_only_modifies_last_message():
    session = make_session()
    session.task_manager.render_block.return_value = "<tasks>\n[ ] 1. A\n</tasks>"
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    result = session._inject_task_block(messages)
    assert result[0]["content"] == "first"
    assert result[1]["content"] == "second"
    assert result[2]["content"].startswith("(System info - Here's the list of tasks you're working on")


def test_llm_called_with_injected_messages_not_originals():
    """stream_response receives the ephemeral copy with the task block."""
    session = make_session()
    session.task_manager.render_block.return_value = "<tasks>\n[ ] 1. Do work\n</tasks>"
    session.task_manager.tool_names = set()

    captured_messages = []

    def side_effect(messages, tools=None):
        captured_messages.extend(messages)
        return _good_text_response("Done.")

    session.llm_client.stream_response.side_effect = side_effect

    asyncio.run(_collect(session.handle_user_input("hello")))

    last_msg_content = captured_messages[-1]["content"]
    assert "<tasks>" in last_msg_content
    # Original user message in session.messages must not be mutated
    # After the turn: [system_msg, user_msg, assistant_msg] — user is at index -2
    assert session.messages[-2]["content"] == "hello"


# ── _should_suggest_plan ──────────────────────────────────────────────────────

def test_should_suggest_plan_false_for_short_simple_text():
    session = make_session()
    assert session._should_suggest_plan("hello") is False


def test_should_suggest_plan_true_for_long_text():
    session = make_session()
    long_text = "a" * 201
    assert session._should_suggest_plan(long_text) is True


def test_should_suggest_plan_true_for_and_then():
    session = make_session()
    assert session._should_suggest_plan("Do this and then do that") is True


def test_should_suggest_plan_true_for_after_that():
    session = make_session()
    assert session._should_suggest_plan("Do this. After that, do the other thing.") is True


def test_should_suggest_plan_true_for_first_then():
    session = make_session()
    assert session._should_suggest_plan("First run the tests, then deploy.") is True


def test_should_suggest_plan_true_for_two_or_more_paths():
    session = make_session()
    # 4 slashes = 2+ paths heuristic
    assert session._should_suggest_plan("Read /foo/bar.txt and write to /baz/out.txt") is True


def test_should_suggest_plan_handles_multimodal_input():
    session = make_session()
    content = [
        {"type": "text", "text": "Do this and then do that"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]
    assert session._should_suggest_plan(content) is True


def test_should_suggest_plan_false_for_short_multimodal():
    session = make_session()
    content = [{"type": "text", "text": "hello"}]
    assert session._should_suggest_plan(content) is False


# ── nudge injection ───────────────────────────────────────────────────────────

def test_nudge_prepended_when_empty_plan_and_complex_query():
    session = make_session()
    session.task_manager.is_empty.return_value = True
    session.task_manager.tool_names = set()

    captured_messages = []

    def side_effect(messages, tools=None):
        captured_messages.extend(messages)
        return _good_text_response("Done.")

    session.llm_client.stream_response.side_effect = side_effect

    asyncio.run(_collect(session.handle_user_input("Do this and then do that.")))

    user_msg = next(m for m in captured_messages if m["role"] == "user" and "Do this" in str(m["content"]))
    assert "Hint" in str(user_msg["content"])
    assert "plan_tasks" in str(user_msg["content"])


def test_nudge_not_added_when_plan_exists():
    session = make_session()
    session.task_manager.is_empty.return_value = False
    session.task_manager.tool_names = set()

    captured_messages = []

    def side_effect(messages, tools=None):
        captured_messages.extend(messages)
        return _good_text_response("Done.")

    session.llm_client.stream_response.side_effect = side_effect

    asyncio.run(_collect(session.handle_user_input("Do this and then do that.")))

    user_msg = next(m for m in captured_messages if m["role"] == "user" and "Do this" in str(m["content"]))
    assert "Hint" not in str(user_msg["content"])


def test_nudge_not_added_for_simple_query():
    session = make_session()
    session.task_manager.is_empty.return_value = True
    session.task_manager.tool_names = set()

    captured_messages = []

    def side_effect(messages, tools=None):
        captured_messages.extend(messages)
        return _good_text_response("Done.")

    session.llm_client.stream_response.side_effect = side_effect

    asyncio.run(_collect(session.handle_user_input("hello")))

    user_msg = next(m for m in captured_messages if m["role"] == "user" and "hello" in str(m["content"]))
    assert "Hint" not in str(user_msg["content"])


def test_nudge_not_persisted_in_session_messages():
    """Hint must not appear in self.messages after the turn completes."""
    session = make_session()
    session.task_manager.is_empty.return_value = True
    session.task_manager.tool_names = set()
    session.llm_client.stream_response.side_effect = lambda *a, **kw: _good_text_response("Done.")

    asyncio.run(_collect(session.handle_user_input("Do this and then do that.")))

    for msg in session.messages:
        assert "Hint" not in str(msg["content"])


# ── _run_planner_pass ─────────────────────────────────────────────────────────

def test_planner_pass_calls_execute_when_llm_returns_plan_tasks():
    session = make_session()
    session.task_manager.tool_schemas = [PLAN_TASKS_SCHEMA]
    session.llm_client.stream_response.side_effect = lambda *a, **kw: _plan_tasks_response()

    result = asyncio.run(session._run_planner_pass("Do A then B."))

    assert result is True
    session.task_manager.execute.assert_called_once_with("plan_tasks", {"titles": ["Task A", "Task B"]})


def test_planner_pass_returns_false_when_no_tool_call():
    session = make_session()
    session.task_manager.tool_schemas = [PLAN_TASKS_SCHEMA]
    session.llm_client.stream_response.side_effect = lambda *a, **kw: _good_text_response("I'll plan later.")

    result = asyncio.run(session._run_planner_pass("Do A then B."))

    assert result is False
    session.task_manager.execute.assert_not_called()


def test_planner_pass_returns_false_when_no_plan_schema():
    session = make_session()
    session.task_manager.tool_schemas = []  # no plan_tasks schema
    result = asyncio.run(session._run_planner_pass("Do A then B."))
    assert result is False


def test_planner_pass_fires_on_auto_mode_complex_query():
    """handle_user_input triggers _run_planner_pass when mode=auto and query is complex."""
    session = make_session()
    session.planner_mode = "auto"
    session.task_manager.is_empty.return_value = True
    session.task_manager.tool_schemas = [PLAN_TASKS_SCHEMA]
    session.task_manager.tool_names = set()

    call_count = 0

    def side_effect(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1 and tools is not None and len(tools) == 1 and tools[0]["function"]["name"] == "plan_tasks":
            return _plan_tasks_response()
        return _good_text_response("Done.")

    session.llm_client.stream_response.side_effect = side_effect

    asyncio.run(_collect(session.handle_user_input("Do this and then do that.")))

    session.task_manager.execute.assert_called_once_with("plan_tasks", {"titles": ["Task A", "Task B"]})


def test_planner_pass_skipped_when_mode_off():
    session = make_session()
    session.planner_mode = "off"
    session.task_manager.is_empty.return_value = True
    session.task_manager.tool_schemas = [PLAN_TASKS_SCHEMA]
    session.task_manager.tool_names = set()
    session.llm_client.stream_response.side_effect = lambda *a, **kw: _good_text_response("Done.")

    asyncio.run(_collect(session.handle_user_input("Do this and then do that.")))

    session.task_manager.execute.assert_not_called()


def test_planner_pass_always_fires_for_simple_query():
    """mode=always triggers planner even for short simple queries."""
    session = make_session()
    session.planner_mode = "always"
    session.task_manager.is_empty.return_value = True
    session.task_manager.tool_schemas = [PLAN_TASKS_SCHEMA]
    session.task_manager.tool_names = set()

    call_count = 0

    def side_effect(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1 and tools is not None and len(tools) == 1 and tools[0]["function"]["name"] == "plan_tasks":
            return _plan_tasks_response()
        return _good_text_response("Done.")

    session.llm_client.stream_response.side_effect = side_effect

    asyncio.run(_collect(session.handle_user_input("hello")))

    session.task_manager.execute.assert_called_once()


def test_planner_pass_suppresses_hint():
    """When planner runs, the nudge hint is not injected."""
    session = make_session()
    session.planner_mode = "auto"
    session.task_manager.is_empty.return_value = True
    session.task_manager.tool_schemas = [PLAN_TASKS_SCHEMA]
    session.task_manager.tool_names = set()
    session.task_manager.render_block.return_value = ""

    captured_messages = []
    call_count = 0

    def side_effect(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_messages.extend(messages)
        if call_count == 1 and tools is not None and len(tools) == 1 and tools[0]["function"]["name"] == "plan_tasks":
            return _plan_tasks_response()
        return _good_text_response("Done.")

    session.llm_client.stream_response.side_effect = side_effect

    asyncio.run(_collect(session.handle_user_input("Do this and then do that.")))

    user_msg = next(m for m in captured_messages if m["role"] == "user" and "Do this" in str(m["content"]))
    assert "Hint" not in str(user_msg["content"])


def test_planner_pass_skipped_when_plan_exists():
    """Planner pass only fires when task list is empty."""
    session = make_session()
    session.planner_mode = "auto"
    session.task_manager.is_empty.return_value = False  # plan already exists
    session.task_manager.tool_schemas = [PLAN_TASKS_SCHEMA]
    session.task_manager.tool_names = set()
    session.llm_client.stream_response.side_effect = lambda *a, **kw: _good_text_response("Done.")

    asyncio.run(_collect(session.handle_user_input("Do this and then do that.")))

    session.task_manager.execute.assert_not_called()


# ── Workspace Guide (workspace.md) ────────────────────────────────────────────

def test_mission_brief_injected_when_plan_md_exists(tmp_path):
    (tmp_path / ".agent").mkdir()
    plan_md = tmp_path / ".agent" / "workspace.md"
    plan_md.write_text("We are building X. Constraints: Y.", encoding="utf-8")

    from mcp_chatbot.core.session import ChatSession
    from mcp_chatbot.core.llm_client import LLMClient

    with patch("mcp_chatbot.core.session.load_settings", return_value={"file_root": str(tmp_path)}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"):
        llm = MagicMock(spec=LLMClient)
        llm.max_tokens = 4096
        s = ChatSession([], llm)
        s._is_initialized = True
        s._skills_manager.get_index.return_value = []
        s.task_manager.is_empty.return_value = True
        s.task_manager.render_block.return_value = ""
        s.task_manager.tool_schemas = []
        s.task_manager.tool_names = set()
        s.active_servers = {}

        asyncio.run(s.build_system_message())

        system_msg = s.messages[0]["content"]
        assert "Workspace Guide" in system_msg
        assert "We are building X." in system_msg


def test_mission_brief_absent_when_no_plan_md(tmp_path):
    # No workspace.md in tmp_path

    from mcp_chatbot.core.session import ChatSession
    from mcp_chatbot.core.llm_client import LLMClient

    with patch("mcp_chatbot.core.session.load_settings", return_value={"file_root": str(tmp_path)}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"):
        llm = MagicMock(spec=LLMClient)
        llm.max_tokens = 4096
        s = ChatSession([], llm)
        s._is_initialized = True
        s._skills_manager.get_index.return_value = []
        s.task_manager.is_empty.return_value = True
        s.task_manager.render_block.return_value = ""
        s.task_manager.tool_schemas = []
        s.task_manager.tool_names = set()
        s.active_servers = {}

        asyncio.run(s.build_system_message())

        system_msg = s.messages[0]["content"]
        assert "Workspace Guide" not in system_msg


def test_mission_brief_reloads_on_build_system_message(tmp_path):
    """Second call to build_system_message picks up an updated workspace.md."""
    from mcp_chatbot.core.session import ChatSession
    from mcp_chatbot.core.llm_client import LLMClient

    with patch("mcp_chatbot.core.session.load_settings", return_value={"file_root": str(tmp_path)}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"):
        llm = MagicMock(spec=LLMClient)
        llm.max_tokens = 4096
        s = ChatSession([], llm)
        s._is_initialized = True
        s._skills_manager.get_index.return_value = []
        s.task_manager.is_empty.return_value = True
        s.task_manager.render_block.return_value = ""
        s.task_manager.tool_schemas = []
        s.task_manager.tool_names = set()
        s.active_servers = {}

        # First call — no workspace.md
        asyncio.run(s.build_system_message())
        assert "Workspace Guide" not in s.messages[0]["content"]

        # Create .agent/workspace.md and rebuild
        (tmp_path / ".agent").mkdir(exist_ok=True)
        (tmp_path / ".agent" / "workspace.md").write_text("New guide.", encoding="utf-8")
        asyncio.run(s.build_system_message())
        assert "Workspace Guide" in s.messages[0]["content"]
        assert "New guide." in s.messages[0]["content"]


# ── reinitialize_root ─────────────────────────────────────────────────────────

def test_reinitialize_root_creates_agent_dir(tmp_path):
    """Fresh dir gets .agent/ created and all managers initialized."""
    session = make_session()
    new_root = tmp_path / "workspace"
    new_root.mkdir()

    session.reinitialize_root(new_root)

    assert (new_root / ".agent").is_dir()
    assert session.file_manager is not None
    assert session.task_manager is not None
    assert session.episodic_store is not None
    assert session.memory_manager is not None
    assert session.playbook_manager is not None


def test_reinitialize_root_switches_to_new_root(tmp_path):
    """Second reinitialize_root replaces managers; new root's .agent/ is created."""
    session = make_session()

    root1 = tmp_path / "root1"
    root1.mkdir()
    root2 = tmp_path / "root2"
    root2.mkdir()

    session.reinitialize_root(root1)
    assert (root1 / ".agent").is_dir()
    old_task_manager = session.task_manager

    session.reinitialize_root(root2)

    assert (root2 / ".agent").is_dir()
    assert session.task_manager is not old_task_manager
    assert session.file_manager is not None
    assert session.episodic_store is not None

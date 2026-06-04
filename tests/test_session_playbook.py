import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_chatbot.core.session import ChatSession
from mcp_chatbot.core.llm_client import LLMClient


def make_session(playbook_entries=None):
    store = MagicMock()
    store.get_recent.return_value = []

    mock_playbook = MagicMock()
    mock_playbook.tool_names = {"record_procedure", "search_playbook"}
    mock_playbook.safe_tool_names = {"search_playbook"}
    mock_playbook.tool_schemas = []
    mock_playbook.search.return_value = []
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

def test_playbook_block_not_in_system_prompt():
    s, pm = make_session(playbook_entries=[{"pattern": "deploy pkg", "action": "step 1"}])
    asyncio.run(s.build_system_message())
    system_content = s.messages[0]["content"]
    assert "[Playbook]:" not in system_content


def test_search_playbook_hint_in_task_context():
    s, pm = make_session()
    asyncio.run(s.build_system_message())
    system_content = s.messages[0]["content"]
    assert "Call search_playbook(query)" in system_content


def test_search_playbook_hint_present_without_task_manager():
    s, pm = make_session()
    s.task_manager = None
    asyncio.run(s.build_system_message())
    system_content = s.messages[0]["content"]
    assert "Call search_playbook(query)" in system_content


def test_search_playbook_registered_in_tool_schemas():
    real_schemas = [
        {"type": "function", "function": {"name": "record_procedure", "parameters": {}}},
        {"type": "function", "function": {"name": "search_playbook", "parameters": {}}},
    ]
    s, pm = make_session()
    pm.tool_schemas = real_schemas
    asyncio.run(s.build_system_message())
    schema_names = [sc["function"]["name"] for sc in s._tool_schemas]
    assert "search_playbook" in schema_names
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


# ── planner procedure injection (Phase 6) ─────────────────────────────────────

def test_planner_injects_matching_procedures():
    s, pm = make_session()
    pm.search.return_value = [{"id": 1, "pattern": "deploy app", "action": "step 1; step 2"}]

    plan_schema = {"type": "function", "function": {"name": "plan_tasks", "parameters": {}}}
    s.task_manager.tool_schemas = [plan_schema]
    s.task_manager.is_empty.return_value = True

    captured_messages = []

    def mock_stream(messages, tools=None, **kwargs):
        captured_messages.extend(messages)
        return iter([{"type": "tool_calls_final", "data": [
            {"function": {"name": "plan_tasks", "arguments": '{"tasks":[{"title":"Deploy"}]}'}}
        ]}])

    s.llm_client.stream_response.side_effect = mock_stream

    result = asyncio.run(s._run_planner_pass("deploy application to production"))
    assert result is True
    assert any("[Relevant procedures" in m["content"] for m in captured_messages)
    assert any("deploy app" in m["content"] for m in captured_messages)


def test_planner_skips_injection_when_no_match():
    s, pm = make_session()
    pm.search.return_value = []

    plan_schema = {"type": "function", "function": {"name": "plan_tasks", "parameters": {}}}
    s.task_manager.tool_schemas = [plan_schema]

    captured_messages = []

    def mock_stream(messages, tools=None, **kwargs):
        captured_messages.extend(messages)
        return iter([{"type": "tool_calls_final", "data": [
            {"function": {"name": "plan_tasks", "arguments": '{"tasks":[{"title":"Deploy"}]}'}}
        ]}])

    s.llm_client.stream_response.side_effect = mock_stream

    asyncio.run(s._run_planner_pass("deploy application"))
    assert not any("[Relevant procedures" in m["content"] for m in captured_messages)


def test_planner_skips_injection_when_playbook_manager_none():
    s, pm = make_session()
    s.playbook_manager = None

    plan_schema = {"type": "function", "function": {"name": "plan_tasks", "parameters": {}}}
    s.task_manager.tool_schemas = [plan_schema]

    captured_messages = []

    def mock_stream(messages, tools=None, **kwargs):
        captured_messages.extend(messages)
        return iter([{"type": "tool_calls_final", "data": [
            {"function": {"name": "plan_tasks", "arguments": '{"tasks":[{"title":"Deploy"}]}'}}
        ]}])

    s.llm_client.stream_response.side_effect = mock_stream

    result = asyncio.run(s._run_planner_pass("deploy application"))
    assert not any("[Relevant procedures" in m["content"] for m in captured_messages)


# ── Phase 11: tool_choice forced in planner pass ──────────────────────────────

def test_planner_forces_tool_choice():
    """_run_planner_pass passes forced tool_choice for plan_tasks."""
    s, pm = make_session()
    pm.search.return_value = []

    plan_schema = {"type": "function", "function": {"name": "plan_tasks", "parameters": {}}}
    s.task_manager.tool_schemas = [plan_schema]

    captured_kwargs = {}

    def mock_stream(messages, tools=None, **kwargs):
        captured_kwargs.update(kwargs)
        return iter([{"type": "tool_calls_final", "data": [
            {"function": {"name": "plan_tasks", "arguments": '{"tasks":[{"title":"T"}]}'}}
        ]}])

    s.llm_client.stream_response.side_effect = mock_stream

    asyncio.run(s._run_planner_pass("do something"))
    assert "tool_choice" in captured_kwargs
    assert captured_kwargs["tool_choice"] == {"type": "function", "function": {"name": "plan_tasks"}}


# ── Phase 10: injected_procedure_ids tracking ─────────────────────────────────

def test_injected_ids_captured_in_planner_pass():
    """IDs of matched procedures are stored on the session after planner pass."""
    s, pm = make_session()
    pm.search.return_value = [
        {"id": 7, "pattern": "deploy app", "action": "step 1"},
        {"id": 9, "pattern": "deploy service", "action": "step 2"},
    ]

    plan_schema = {"type": "function", "function": {"name": "plan_tasks", "parameters": {}}}
    s.task_manager.tool_schemas = [plan_schema]
    s.task_manager.is_empty.return_value = True

    s.llm_client.stream_response.side_effect = lambda *a, **kw: iter([
        {"type": "tool_calls_final", "data": [
            {"function": {"name": "plan_tasks", "arguments": '{"tasks":[{"title":"Deploy"}]}'}}
        ]}
    ])

    asyncio.run(s._run_planner_pass("deploy application"))
    assert s._injected_procedures == [
        {"id": 7, "action": "step 1"},
        {"id": 9, "action": "step 2"},
    ]


def test_injected_ids_persist_across_non_planning_turn():
    """Plan-scoped, not turn-scoped: ids set by a planner pass survive a later
    turn that does not re-plan, so reinforce/penalize still fires when a
    multi-turn plan finally completes or aborts."""
    s, pm = make_session()
    injected = [{"id": 1, "action": "a"}, {"id": 2, "action": "b"}, {"id": 3, "action": "c"}]
    s._injected_procedures = list(injected)  # set by an earlier planner pass
    s.context_mgr.is_near_limit = lambda: False
    s.planner_mode = "off"  # no planner this turn
    s.llm_client.stream_response.side_effect = lambda *a, **kw: iter([
        {"type": "content", "data": "Done."}
    ])

    asyncio.run(_collect(s.handle_user_input("hello")))
    assert s._injected_procedures == injected


def test_injected_ids_reset_by_new_planner_pass():
    """A fresh planner pass replaces stale ids (here: no match → cleared)."""
    s, pm = make_session()
    s._injected_procedures = [{"id": 1, "action": "a"}]  # stale from a previous plan
    pm.search.return_value = []

    plan_schema = {"type": "function", "function": {"name": "plan_tasks", "parameters": {}}}
    s.task_manager.tool_schemas = [plan_schema]

    s.llm_client.stream_response.side_effect = lambda *a, **kw: iter([
        {"type": "tool_calls_final", "data": [
            {"function": {"name": "plan_tasks", "arguments": '{"tasks":[{"title":"T"}]}'}}
        ]}
    ])

    asyncio.run(s._run_planner_pass("do something unrelated"))
    assert s._injected_procedures == []


# ── Phase 10: penalize on abort ───────────────────────────────────────────────

def test_penalize_called_on_fingerprint_abort():
    """playbook_manager.penalize is called when the fingerprint loop guard fires."""
    s, pm = make_session()
    s.context_mgr.is_near_limit = lambda: False
    s._tool_schemas = [{"type": "function", "function": {"name": "update_task"}}]
    s.planner_mode = "always"
    s.task_manager.is_empty.return_value = True

    async def mock_planner(user_input):
        s._injected_procedures = [{"id": 5, "action": ""}, {"id": 6, "action": ""}]
        return True

    s._run_planner_pass = mock_planner

    s.task_manager.tool_names = {"update_task"}
    s.task_manager.safe_tool_names = {"update_task"}
    s.task_manager._last_all_done = False  # looping, not complete
    s.task_manager.execute.return_value = "same result"  # identical → fingerprint fires

    s.llm_client.stream_response.side_effect = lambda *a, **kw: iter([
        {"type": "tool_calls_final", "data": [{
            "id": "tc1", "type": "function",
            "function": {"name": "update_task", "arguments": '{"id":"1","status":"in_progress"}'},
        }]}
    ])

    asyncio.run(_collect(s.handle_user_input("go")))
    pm.penalize.assert_called_once_with({5: 1.0, 6: 0.5})  # rank prior (no plan text)
    assert s._injected_procedures == []  # cleared after penalize


def test_penalize_called_on_iteration_cap():
    """playbook_manager.penalize is called when the iteration cap fires."""
    import itertools
    from mcp_chatbot.core.session import MAX_AGENT_ITERATIONS

    s, pm = make_session()
    s.context_mgr.is_near_limit = lambda: False
    s._tool_schemas = [{"type": "function", "function": {"name": "update_task"}}]
    s.planner_mode = "always"
    s.task_manager.is_empty.return_value = True

    async def mock_planner(user_input):
        s._injected_procedures = [{"id": 3, "action": ""}]
        return True

    s._run_planner_pass = mock_planner

    s.task_manager.tool_names = {"update_task"}
    s.task_manager.safe_tool_names = {"update_task"}
    s.task_manager._last_all_done = False  # looping, not complete
    counter = itertools.count()
    s.task_manager.execute.side_effect = lambda name, args: f"r-{next(counter)}"

    def side_effect(messages, tools=None, **kw):
        if tools is None:
            return iter([{"type": "content", "data": "summary"}])
        return iter([{"type": "tool_calls_final", "data": [{
            "id": "tc1", "type": "function",
            "function": {"name": "update_task", "arguments": '{"id":"1","status":"in_progress"}'},
        }]}])

    s.llm_client.stream_response.side_effect = side_effect

    asyncio.run(_collect(s.handle_user_input("go")))
    pm.penalize.assert_called_once_with({3: 1.0})
    assert s._injected_procedures == []  # cleared after penalize


def test_penalize_called_on_wall_clock_timeout():
    """playbook_manager.penalize is called when the wall-clock guard fires."""
    from unittest.mock import patch

    s, pm = make_session()
    s.context_mgr.is_near_limit = lambda: False
    s._tool_schemas = [{"type": "function", "function": {"name": "update_task"}}]
    s.planner_mode = "always"
    s.task_manager.is_empty.return_value = True

    async def mock_planner(user_input):
        s._injected_procedures = [{"id": 8, "action": ""}]
        return True

    s._run_planner_pass = mock_planner
    s.agent_timeout = 300  # override the 900s default so +400s trips the cap
    s.llm_client.stream_response.side_effect = lambda *a, **kw: iter([
        {"type": "content", "data": "x"}
    ])
    # Synthesis returns empty → falls back to the plain "timed out" message.
    s.llm_client.get_response.return_value = {"role": "assistant", "content": ""}

    # First time.time() seeds _wall_start; second (loop check) is +400s → abort.
    with patch("mcp_chatbot.core.session.time.time", side_effect=[100.0, 500.0, 500.0, 500.0]):
        out = asyncio.run(_collect(s.handle_user_input("go")))

    pm.penalize.assert_called_once_with({8: 1.0})
    assert s._injected_procedures == []
    assert any("timed out" in str(x) for x in out)


def test_penalize_called_on_parse_retry_abort():
    """playbook_manager.penalize is called when the malformed-JSON parse-retry
    guard aborts; the injected-id set is cleared so it can't leak to the next turn."""
    from mcp_chatbot.core.session import MAX_PARSE_RETRIES

    s, pm = make_session()
    s.context_mgr.is_near_limit = lambda: False
    s._tool_schemas = [{"type": "function", "function": {"name": "update_task"}}]
    s.planner_mode = "always"
    s.task_manager.is_empty.return_value = True

    async def mock_planner(user_input):
        s._injected_procedures = [{"id": 11, "action": ""}, {"id": 12, "action": ""}]
        return True

    s._run_planner_pass = mock_planner
    s.task_manager.tool_names = {"update_task"}
    s.task_manager.safe_tool_names = {"update_task"}

    # Always return a tool call with invalid JSON arguments → parse error every round.
    s.llm_client.stream_response.side_effect = lambda *a, **kw: iter([
        {"type": "tool_calls_final", "data": [{
            "id": "tc1", "type": "function",
            "function": {"name": "update_task", "arguments": "{not valid json"},
        }]}
    ])

    out = asyncio.run(_collect(s.handle_user_input("go")))
    assert any("invalid tool call JSON" in str(x) for x in out)
    pm.penalize.assert_called_once_with({11: 1.0, 12: 0.5})
    assert s._injected_procedures == []
    assert s.agent_loop["active"] is False
    # the parse-retry counter only aborts after exceeding the cap
    assert s.llm_client.stream_response.call_count == MAX_PARSE_RETRIES + 1


# ── Phase 10: reinforce on success ────────────────────────────────────────────

def test_reinforce_called_on_all_done():
    """playbook_manager.reinforce is called when all tasks complete."""
    s, pm = make_session()
    pm.tool_schemas = [{"type": "function", "function": {"name": "record_procedure", "parameters": {}}}]
    s.context_mgr.is_near_limit = lambda: False
    s._tool_schemas = [{"type": "function", "function": {"name": "update_task"}}]
    s.planner_mode = "always"
    s.task_manager.is_empty.return_value = True

    async def mock_planner(user_input):
        s._injected_procedures = [{"id": 2, "action": ""}, {"id": 4, "action": ""}]
        return True

    s._run_planner_pass = mock_planner

    s.task_manager.tool_names = {"update_task"}
    s.task_manager.safe_tool_names = {"update_task"}
    s.task_manager.execute.return_value = "Task 1 updated."

    call_count = [0]

    def side_effect(messages, tools=None, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            return iter([{"type": "tool_calls_final", "data": [{
                "id": "tc1", "type": "function",
                "function": {"name": "update_task", "arguments": '{"id":"1","status":"done"}'},
            }]}])
        return iter([{"type": "content", "data": "All done."}])

    s.llm_client.stream_response.side_effect = side_effect
    s.llm_client.get_response.return_value = {"role": "assistant", "content": "ok"}

    s.task_manager._last_all_done = True
    s.allow_tool_action = "always"

    asyncio.run(_collect(s.handle_user_input("finish")))
    pm.reinforce.assert_called_once_with({2: 1.0, 4: 0.5})  # rank prior (no plan text)
    assert s._injected_procedures == []  # cleared after reinforce


def test_reinforce_skipped_when_gate_disabled():
    """playbook_reinforce=False suppresses the +0.05 bump (Phase 10.4 gate)."""
    s, pm = make_session()
    s.playbook_reinforce = False
    pm.tool_schemas = [{"type": "function", "function": {"name": "record_procedure", "parameters": {}}}]
    s.context_mgr.is_near_limit = lambda: False
    s._tool_schemas = [{"type": "function", "function": {"name": "update_task"}}]
    s.planner_mode = "always"
    s.task_manager.is_empty.return_value = True

    async def mock_planner(user_input):
        s._injected_procedures = [{"id": 2, "action": ""}, {"id": 4, "action": ""}]
        return True

    s._run_planner_pass = mock_planner
    s.task_manager.tool_names = {"update_task"}
    s.task_manager.safe_tool_names = {"update_task"}
    s.task_manager.execute.return_value = "Task 1 updated."

    call_count = [0]

    def side_effect(messages, tools=None, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            return iter([{"type": "tool_calls_final", "data": [{
                "id": "tc1", "type": "function",
                "function": {"name": "update_task", "arguments": '{"id":"1","status":"done"}'},
            }]}])
        return iter([{"type": "content", "data": "All done."}])

    s.llm_client.stream_response.side_effect = side_effect
    s.llm_client.get_response.return_value = {"role": "assistant", "content": "ok"}
    s.task_manager._last_all_done = True
    s.allow_tool_action = "always"

    asyncio.run(_collect(s.handle_user_input("finish")))
    pm.reinforce.assert_not_called()
    assert s._injected_procedures == []  # still cleared


# ── attribution weighting ─────────────────────────────────────────────────────

def test_attribution_concentrates_on_plan_matching_procedure():
    """When one procedure's action resembles the plan that ran, it takes the full
    delta and the other candidates get only the residual share."""
    s, _ = make_session()
    s._injected_procedures = [
        {"id": 1, "action": "read csv file and plot a chart with matplotlib"},
        {"id": 2, "action": "build docker image push to registry restart the service"},
        {"id": 3, "action": "write unit tests with pytest and run them"},
    ]
    s.task_manager.plan_text.return_value = (
        "build docker image and push it then restart service on production"
    )
    w = s._attribution_weights()
    assert w == {1: 0.3, 2: 1.0, 3: 0.3}


def test_attribution_falls_back_to_rank_prior_without_plan():
    """No plan text (or no match above threshold) → rank prior 1/(rank+1)."""
    s, _ = make_session()
    s._injected_procedures = [
        {"id": 5, "action": "alpha"}, {"id": 6, "action": "beta"}, {"id": 7, "action": "gamma"},
    ]
    s.task_manager.plan_text.return_value = ""
    w = s._attribution_weights()
    assert w == {5: 1.0, 6: 0.5, 7: 1 / 3}


def test_attribution_empty_when_nothing_injected():
    s, _ = make_session()
    s._injected_procedures = []
    assert s._attribution_weights() == {}

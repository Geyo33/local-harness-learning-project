import asyncio
import itertools
import json
from unittest.mock import MagicMock, patch


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
        s.task_manager.safe_tool_names = set()
        s.task_manager._last_all_done = False
        # non-empty so the main loop passes real tools (tools != None);
        # the synthesis pass is the only call made with tools=None
        s._tool_schemas = [{"type": "function", "function": {"name": "update_task"}}]
        # never compress mid-test — keeps the loop deterministic
        s.context_mgr.is_near_limit = lambda: False
        # disable planner pass: these tests drive the main loop directly
        s.planner_mode = "off"
        return s


def _tool_call_response(name, args, tc_id="tc1"):
    """Sync generator yielding a single tool_calls_final."""
    yield {
        "type": "tool_calls_final",
        "data": [{
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def _good_text_response(text="Done."):
    """Sync generator yielding plain content (no tool calls)."""
    yield {"type": "content", "data": text}


async def _collect(async_gen):
    results = []
    async for item in async_gen:
        results.append(item)
    return results


# ── 4.3 iteration cap ─────────────────────────────────────────────────────────

def test_max_iterations_triggers_synthesis():
    """LLM always returns a tool call; loop aborts at the cap and streams a
    synthesis answer (a final stream_response call made with tools=None)."""
    from mcp_chatbot.core.session import MAX_AGENT_ITERATIONS

    session = make_session()
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}
    # vary the result each call so the fingerprint guard never fires first
    counter = itertools.count()
    session.task_manager.execute.side_effect = lambda name, args: f"result-{next(counter)}"

    def side_effect(messages, tools=None):
        if tools is None:  # the synthesis pass offers no tools
            return _good_text_response("Synthesized summary.")
        return _tool_call_response("update_task", {"id": "1", "status": "in_progress"})

    session.llm_client.stream_response.side_effect = side_effect

    results = asyncio.run(_collect(session.handle_user_input("go")))

    assert any("Synthesized summary." in str(r) for r in results)
    # MAX_AGENT_ITERATIONS tool rounds + 1 synthesis round
    assert session.llm_client.stream_response.call_count == MAX_AGENT_ITERATIONS + 1
    # synthesis call was made with tools=None
    assert session.llm_client.stream_response.call_args.kwargs["tools"] is None
    assert session.agent_loop["active"] is False
    # synthesized answer is persisted so the turn closes cleanly in history
    assert session.messages[-1] == {"role": "assistant", "content": "Synthesized summary."}


# ── 4.5 loop fingerprinting ───────────────────────────────────────────────────

def test_fingerprint_aborts_on_three_identical():
    """Identical tool + result 3 times in a row aborts with the stuck-loop error."""
    session = make_session()
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}
    session.task_manager.execute.return_value = "same result"

    session.llm_client.stream_response.side_effect = (
        lambda *a, **kw: _tool_call_response("update_task", {"id": "1", "status": "in_progress"})
    )

    results = asyncio.run(_collect(session.handle_user_input("go")))

    assert any("stuck in loop" in str(r) for r in results)
    # abort fires on the 3rd identical result — exactly 3 loop iterations
    assert session.llm_client.stream_response.call_count == 3
    assert session.agent_loop["active"] is False


def test_fingerprint_does_not_abort_when_results_differ():
    """Same tool but differing results must not trip the fingerprint guard."""
    session = make_session()
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}
    counter = itertools.count()

    call_count = 0

    def side_effect(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            return _tool_call_response("update_task", {"id": "1", "status": "in_progress"})
        return _good_text_response("All done.")

    session.task_manager.execute.side_effect = lambda name, args: f"result-{next(counter)}"
    session.llm_client.stream_response.side_effect = side_effect

    results = asyncio.run(_collect(session.handle_user_input("go")))

    assert not any("stuck in loop" in str(r) for r in results)
    assert any("All done." in str(r) for r in results)


def test_fingerprint_does_not_abort_when_args_differ():
    """Same tool + same result but differing args must not trip the guard —
    e.g. search_playbook('x'), search_playbook('y'), search_playbook('z') all
    returning 'No matching procedures found.' is not a stuck loop."""
    session = make_session()
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}
    # constant result for every call — only the args vary
    session.task_manager.execute.return_value = "No matching procedures found."

    call_count = 0

    def side_effect(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            return _tool_call_response(
                "update_task", {"id": str(call_count), "status": "in_progress"}
            )
        return _good_text_response("All done.")

    session.llm_client.stream_response.side_effect = side_effect

    results = asyncio.run(_collect(session.handle_user_input("go")))

    assert not any("stuck in loop" in str(r) for r in results)
    assert any("All done." in str(r) for r in results)


# ── 4.4 wall-clock timeout ────────────────────────────────────────────────────

def test_wall_clock_timeout_aborts():
    """Clock advances past the timeout on the second iteration → abort."""
    from mcp_chatbot.core.session import AGENT_WALL_CLOCK_TIMEOUT

    session = make_session()
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}
    session.task_manager.execute.return_value = "ok"

    session.llm_client.stream_response.side_effect = (
        lambda *a, **kw: _tool_call_response("update_task", {"id": "1", "status": "in_progress"})
    )
    # On timeout the loop attempts a bounded synthesis; empty content → it falls
    # back to the plain timeout error string we assert on below.
    session.llm_client.get_response.return_value = {"role": "assistant", "content": ""}

    state = {"n": 0}

    def fake_time():
        state["n"] += 1
        if state["n"] == 1:
            return 1000.0  # _wall_start (before the loop)
        if state["n"] == 2:
            return 1000.0  # iteration 1 check — within budget
        return 1000.0 + AGENT_WALL_CLOCK_TIMEOUT + 1  # iteration 2 — over budget

    with patch("mcp_chatbot.core.session.time") as mock_time:
        mock_time.time.side_effect = fake_time
        results = asyncio.run(_collect(session.handle_user_input("go")))

    assert any(f"timed out after {AGENT_WALL_CLOCK_TIMEOUT}" in str(r) for r in results)
    # iteration 1 ran one tool round; iteration 2 aborted before calling the LLM
    assert session.llm_client.stream_response.call_count == 1
    assert session.agent_loop["active"] is False


def test_wall_clock_timeout_yields_graceful_synthesis():
    """On timeout a bounded synthesis runs; non-empty content is streamed as the
    final answer (and appended to history) instead of a bare error string."""
    from mcp_chatbot.core.session import AGENT_WALL_CLOCK_TIMEOUT

    session = make_session()
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}
    session.task_manager.execute.return_value = "ok"
    session.llm_client.stream_response.side_effect = (
        lambda *a, **kw: _tool_call_response("update_task", {"id": "1", "status": "in_progress"})
    )
    session.llm_client.get_response.return_value = {
        "role": "assistant", "content": "Here is what I got done so far.",
    }

    state = {"n": 0}

    def fake_time():
        state["n"] += 1
        if state["n"] <= 2:
            return 1000.0
        return 1000.0 + AGENT_WALL_CLOCK_TIMEOUT + 1

    with patch("mcp_chatbot.core.session.time") as mock_time:
        mock_time.time.side_effect = fake_time
        results = asyncio.run(_collect(session.handle_user_input("go")))

    assert any("Here is what I got done so far." in str(r) for r in results)
    assert not any("timed out" in str(r) for r in results)
    assert session.messages[-1] == {
        "role": "assistant", "content": "Here is what I got done so far.",
    }
    assert session.agent_loop["active"] is False


# ── 12.1 stale-task nudge ─────────────────────────────────────────────────────

def test_stale_task_nudge_injected_after_k_iterations():
    """Same in_progress task for STALE_TASK_ITERATIONS loops → nudge injected."""
    from mcp_chatbot.core.session import STALE_TASK_ITERATIONS

    session = make_session()
    session.task_manager.get_in_progress_ids.return_value = ["1"]
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}
    # Vary results per call so the fingerprint guard never fires first.
    execute_count = itertools.count()
    session.task_manager.execute.side_effect = lambda name, args: f"result-{next(execute_count)}"

    call_count = 0
    captured_msgs_list = []

    def side_effect(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_msgs_list.append(list(messages))
        if call_count <= STALE_TASK_ITERATIONS:
            return _tool_call_response("update_task", {"id": "1", "status": "in_progress"})
        return _good_text_response("Done.")

    session.llm_client.stream_response.side_effect = side_effect

    asyncio.run(_collect(session.handle_user_input("go")))

    nudge_present = any(
        any("in_progress for several steps" in str(m.get("content", "")) for m in msgs)
        for msgs in captured_msgs_list
    )
    assert nudge_present, "Stale-task nudge not injected after K identical in_progress iterations"


def test_stale_count_resets_when_in_progress_changes():
    """Stale count resets when in_progress task set changes — nudge never fires."""
    from mcp_chatbot.core.session import STALE_TASK_ITERATIONS

    session = make_session()
    ip_call_count = [0]

    def ip_side_effect():
        ip_call_count[0] += 1
        return ["1"] if ip_call_count[0] <= 3 else ["2"]

    session.task_manager.get_in_progress_ids.side_effect = ip_side_effect
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}
    execute_count = itertools.count()
    session.task_manager.execute.side_effect = lambda name, args: f"result-{next(execute_count)}"

    call_count = 0

    def side_effect(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < STALE_TASK_ITERATIONS + 2:
            return _tool_call_response("update_task", {"id": "1", "status": "in_progress"})
        return _good_text_response("Done.")

    captured_msgs_list = []
    original_side_effect = side_effect

    def capturing_side_effect(messages, tools=None, **kwargs):
        captured_msgs_list.append(list(messages))
        return original_side_effect(messages, tools=tools, **kwargs)

    session.llm_client.stream_response.side_effect = capturing_side_effect

    asyncio.run(_collect(session.handle_user_input("go")))

    nudge_present = any(
        any("in_progress for several steps" in str(m.get("content", "")) for m in msgs)
        for msgs in captured_msgs_list
    )
    assert not nudge_present, "Nudge should not fire when in_progress set changes before K iterations"


def test_stale_nudge_not_injected_when_no_in_progress():
    """No in_progress tasks → stale count never increments, no nudge."""
    from mcp_chatbot.core.session import STALE_TASK_ITERATIONS

    session = make_session()
    session.task_manager.get_in_progress_ids.return_value = []
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}
    session.task_manager.execute.return_value = "ok"

    call_count = 0
    captured_msgs_list = []

    def side_effect(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_msgs_list.append(list(messages))
        if call_count <= STALE_TASK_ITERATIONS + 1:
            return _tool_call_response("update_task", {"id": "1", "status": "done"})
        return _good_text_response("Done.")

    session.llm_client.stream_response.side_effect = side_effect

    asyncio.run(_collect(session.handle_user_input("go")))

    nudge_present = any(
        any("in_progress for several steps" in str(m.get("content", "")) for m in msgs)
        for msgs in captured_msgs_list
    )
    assert not nudge_present

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


# ── interrupt: primitives ─────────────────────────────────────────────────────

def test_request_interrupt_sets_flag_and_unblocks_gate():
    """request_interrupt sets the flag, a sentinel action, and the gate event."""
    session = make_session()
    session.agent_loop = {"active": True, "state": ""}
    assert session.interrupt_requested is False

    asyncio.run(session.request_interrupt())

    assert session.interrupt_requested is True
    assert session.allow_tool_action == "interrupt"
    assert session.allow_tool_event.is_set()


def test_request_interrupt_noop_when_loop_inactive():
    """No active turn → request_interrupt does nothing (avoids leaking into the
    next turn's approval gate)."""
    session = make_session()
    session.agent_loop = {"active": False, "state": ""}

    asyncio.run(session.request_interrupt())

    assert session.interrupt_requested is False
    assert session.allow_tool_action is None
    assert not session.allow_tool_event.is_set()


def test_abort_interrupt_keeps_partial_and_marks():
    """_abort_interrupt persists partial text + marker, resets loop, clears flag."""
    session = make_session()
    session.interrupt_requested = True
    session.agent_loop = {"active": True, "state": "streaming"}

    session._abort_interrupt("partial answer")

    assert session.messages[-1]["role"] == "assistant"
    assert "partial answer" in session.messages[-1]["content"]
    assert "_[Stopped by user]_" in session.messages[-1]["content"]
    assert session.agent_loop == {"active": False, "state": ""}
    assert session.interrupt_requested is False
    assert session.tool_call_detected is False


def test_abort_interrupt_empty_partial_appends_standalone_marker():
    """With no partial text, a standalone marker message still closes the turn."""
    session = make_session()
    session.interrupt_requested = True

    session._abort_interrupt("")

    assert session.messages[-1] == {
        "role": "assistant", "content": "_[Stopped by user]_",
    }


# ── interrupt: loop checkpoints ───────────────────────────────────────────────

def test_interrupt_between_tool_calls_aborts_and_keeps_history():
    """Flag set while a tool executes → loop aborts at the next checkpoint;
    history keeps the tool result and ends with a stopped marker."""
    session = make_session()
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}

    def execute(name, args):
        session.interrupt_requested = True  # simulate stop pressed during the tool
        return "tool ran"
    session.task_manager.execute.side_effect = execute

    session.llm_client.stream_response.side_effect = (
        lambda *a, **kw: _tool_call_response("update_task", {"id": "1", "status": "in_progress"})
    )

    results = asyncio.run(_collect(session.handle_user_input("go")))

    assert any("_[Stopped by user]_" in str(r) for r in results)
    assert session.agent_loop["active"] is False
    # exactly one LLM round happened before the abort
    assert session.llm_client.stream_response.call_count == 1
    # history is valid: the safe tool produced a paired role:tool message,
    # and the turn closes with the stopped marker
    assert session.messages[-1]["content"] == "_[Stopped by user]_"
    assert any(m["role"] == "tool" for m in session.messages)


def test_entry_resets_stale_interrupt_flag():
    """A flag left set before a turn starts must not abort the new turn."""
    session = make_session()
    session.interrupt_requested = True  # stale
    session.llm_client.stream_response.side_effect = (
        lambda *a, **kw: _good_text_response("Hello.")
    )

    results = asyncio.run(_collect(session.handle_user_input("go")))

    assert any("Hello." in str(r) for r in results)
    assert not any("_[Stopped by user]_" in str(r) for r in results)


def test_interrupt_multi_tool_round_pairs_all_calls():
    """If a round returns multiple tool calls and interrupt fires before they all
    run, every tool_call id must still get a paired role:tool result (valid
    history for strict templates)."""
    import json as _json
    session = make_session()
    session.task_manager.tool_names = {"update_task"}
    session.task_manager.safe_tool_names = {"update_task"}

    def execute(name, args):
        # stop pressed during the FIRST tool of the round
        session.interrupt_requested = True
        return "ran"
    session.task_manager.execute.side_effect = execute

    def two_calls(*a, **kw):
        yield {
            "type": "tool_calls_final",
            "data": [
                {"id": "tcA", "type": "function",
                 "function": {"name": "update_task", "arguments": _json.dumps({"id": "1", "status": "in_progress"})}},
                {"id": "tcB", "type": "function",
                 "function": {"name": "update_task", "arguments": _json.dumps({"id": "2", "status": "in_progress"})}},
            ],
        }
    session.llm_client.stream_response.side_effect = two_calls

    asyncio.run(_collect(session.handle_user_input("go")))

    # collect tool_call ids from assistant messages and tool_call_ids from tool messages
    called_ids = set()
    for m in session.messages:
        for tc in (m.get("tool_calls") or []):
            called_ids.add(tc["id"])
    paired_ids = {m["tool_call_id"] for m in session.messages if m["role"] == "tool"}

    assert called_ids, "no tool_calls recorded"
    # every tool_call id must have a paired role:tool result
    assert called_ids <= paired_ids, f"orphaned tool_calls: {called_ids - paired_ids}"
    assert session.messages[-1]["content"] == "_[Stopped by user]_"


def test_interrupt_mid_stream_keeps_partial_text():
    """Flag set after the first token → streaming stops, partial text is kept and
    marked, no further tokens are emitted."""
    session = make_session()

    def streaming_gen(*a, **kw):
        yield {"type": "content", "data": "first part "}
        session.interrupt_requested = True  # stop pressed mid-stream
        yield {"type": "content", "data": "SHOULD_NOT_APPEAR"}
    session.llm_client.stream_response.side_effect = streaming_gen

    results = asyncio.run(_collect(session.handle_user_input("go")))

    joined = "".join(str(r) for r in results)
    assert "first part " in joined
    assert "SHOULD_NOT_APPEAR" not in joined
    assert "_[Stopped by user]_" in joined
    assert session.agent_loop["active"] is False
    assert "first part " in session.messages[-1]["content"]
    assert "_[Stopped by user]_" in session.messages[-1]["content"]


def test_interrupt_at_gate_denies_then_aborts():
    """Stop pressed while an unsafe tool waits at the approval gate → reuse the
    deny path (paired role:tool result) then abort the whole loop."""
    session = make_session()
    # an unsafe tool (not in any safe set) → triggers the approval gate
    session._tool_schemas = [{"type": "function", "function": {"name": "danger_tool"}}]
    session._tool_server_map = {"danger_tool": MagicMock()}

    session.llm_client.stream_response.side_effect = (
        lambda *a, **kw: _tool_call_response("danger_tool", {"x": 1})
    )

    async def driver():
        gen = session.handle_user_input("go")
        out = []
        async for item in gen:
            out.append(item)
            # the tuple yield is the pending tool-approval request
            if isinstance(item, tuple):
                await session.request_interrupt()
        return out

    results = asyncio.run(driver())

    assert any("Tool call denied" in str(r) for r in results)
    assert any("_[Stopped by user]_" in str(r) for r in results)
    assert session.agent_loop["active"] is False
    # history validity: the assistant tool_calls entry has a paired role:tool msg
    assert any(m.get("tool_calls") for m in session.messages)
    assert any(m["role"] == "tool" for m in session.messages)
    # loop did not continue after the gate (only the one LLM round)
    assert session.llm_client.stream_response.call_count == 1


def test_request_interrupt_does_not_clobber_always():
    """request_interrupt must not overwrite auto-tool mode: in 'always' mode there
    is no pending gate wait, and overwriting would silently disable auto-tool."""
    session = make_session()
    session.agent_loop = {"active": True, "state": ""}
    session.allow_tool_action = "always"

    asyncio.run(session.request_interrupt())

    assert session.interrupt_requested is True
    assert session.allow_tool_action == "always"  # not clobbered to "interrupt"
    assert session.allow_tool_event.is_set()


def test_interrupt_preserves_auto_tool_mode():
    """Interrupting while auto-tool ('always') is on must not silently disable it:
    allow_tool_action stays 'always' through the abort so the next turn still
    bypasses the approval gate."""
    session = make_session()
    session.allow_tool_action = "always"

    def streaming_gen(*a, **kw):
        yield {"type": "content", "data": "partial "}
        session.interrupt_requested = True
        yield {"type": "content", "data": "more"}
    session.llm_client.stream_response.side_effect = streaming_gen

    asyncio.run(_collect(session.handle_user_input("go")))

    assert session.allow_tool_action == "always"
    assert session.interrupt_requested is False


def test_interrupt_at_gate_clears_tool_call_detected_before_oio_yield():
    """Regression: the gate-interrupt path must clear tool_call_detected before
    yielding its o|o string. handle_chat checks the tool_call_detected branch
    first and would otherwise unpack the string as a (name, args) tuple and raise
    ValueError. The generator pauses right after each yield, so reading the flag
    in the driver reflects its value at yield time."""
    session = make_session()
    session._tool_schemas = [{"type": "function", "function": {"name": "danger_tool"}}]
    session._tool_server_map = {"danger_tool": MagicMock()}
    session.llm_client.stream_response.side_effect = (
        lambda *a, **kw: _tool_call_response("danger_tool", {"x": 1})
    )

    async def driver():
        gen = session.handle_user_input("go")
        async for item in gen:
            if isinstance(item, tuple):
                await session.request_interrupt()
            elif isinstance(item, str) and "o|o" in item:
                # mirror handle_chat: a structured o|o event must NOT be flagged
                # as a pending tool call
                assert session.tool_call_detected is False, (
                    "tool_call_detected still True when o|o event was yielded"
                )

    asyncio.run(driver())

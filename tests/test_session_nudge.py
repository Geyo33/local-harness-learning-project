import asyncio
from unittest.mock import MagicMock, patch


def _nudge_session():
    from mcp_chatbot.core.session import ChatSession
    s = ChatSession.__new__(ChatSession)
    s._facts_nudge_counter = 0
    s._facts_nudge_threshold = 3
    s._pending_facts_nudge = False
    s._playbook_fired_this_turn = False
    return s


def test_nudge_sets_flag_at_threshold():
    s = _nudge_session()
    s._facts_nudge_counter = 3
    s._maybe_arm_facts_nudge()
    assert s._pending_facts_nudge is True


def test_nudge_suppressed_when_playbook_fired():
    s = _nudge_session()
    s._facts_nudge_counter = 3
    s._playbook_fired_this_turn = True
    s._maybe_arm_facts_nudge()
    assert s._pending_facts_nudge is False


def test_nudge_below_threshold_no_flag():
    s = _nudge_session()
    s._facts_nudge_counter = 1
    s._maybe_arm_facts_nudge()
    assert s._pending_facts_nudge is False


def test_consume_nudge_returns_text_and_resets():
    s = _nudge_session()
    s._pending_facts_nudge = True
    s._facts_nudge_counter = 5
    text = s._consume_facts_nudge()
    assert "remember_fact" in text
    assert s._pending_facts_nudge is False
    assert s._facts_nudge_counter == 0


def test_consume_nudge_empty_when_not_armed():
    s = _nudge_session()
    assert s._consume_facts_nudge() == ""


# ── integration: counter wiring + _hint merge through handle_user_input ────────

def _good_text_response(text="Done."):
    """Sync generator mimicking a plain final answer (no tool calls)."""
    yield {"type": "content", "data": text}


async def _collect(async_gen):
    return [item async for item in async_gen]


def _loop_session():
    """A ChatSession wired to drive the real handle_user_input loop, with the
    store/skills/tasks mocked so each turn resolves to one streamed final answer."""
    from mcp_chatbot.core.session import ChatSession
    from mcp_chatbot.core.llm_client import LLMClient

    with patch("mcp_chatbot.core.session.load_settings", return_value={}), \
         patch("mcp_chatbot.core.session.SkillsManager"), \
         patch("mcp_chatbot.core.session.TaskManager"), \
         patch("mcp_chatbot.core.session.EpisodicStore"):
        llm = MagicMock(spec=LLMClient)
        llm.max_tokens = 4096
        s = ChatSession([], llm)
        s._is_initialized = True
        s._skills_manager.get_index.return_value = []
        s.task_manager.is_empty.return_value = True
        s.task_manager.render_block.return_value = ""
        s.task_manager.tool_schemas = []
        s.task_manager.tool_names = set()
        s.task_manager.safe_tool_names = set()
        s.task_manager._last_all_done = False
        s._tool_schemas = [{"type": "function", "function": {"name": "update_task"}}]
        s.context_mgr.is_near_limit = lambda: False
        s.planner_mode = "off"
        # No-op retrieval so the [Relevant Facts] block stays empty and no real DB
        # file is touched; the nudge text is the only thing we assert on.
        s.episodic_store.search_memory.return_value = []
        return s


def test_nudge_injected_into_next_turn_prompt():
    """End-to-end: with threshold 1, turn 1 arms the nudge (not yet injected),
    turn 2 consumes it — the nudge text must appear in turn 2's LLM prompt and
    the counter must reset. Guards the counter-increment / _consume / _hint-merge
    wiring inside handle_user_input that the unit tests exercise only in isolation."""
    s = _loop_session()
    s._facts_nudge_threshold = 1
    s._facts_nudge_counter = 0
    s.llm_client.stream_response.side_effect = (
        lambda messages, tools=None, **kw: _good_text_response("Done.")
    )

    asyncio.run(_collect(s.handle_user_input("first turn")))
    turn1_msgs = s.llm_client.stream_response.call_args_list[0].args[0]
    assert not any("remember_fact" in str(m.get("content", "")) for m in turn1_msgs)
    assert s._pending_facts_nudge is True  # armed at end of turn 1

    asyncio.run(_collect(s.handle_user_input("second turn")))
    turn2_msgs = s.llm_client.stream_response.call_args_list[1].args[0]
    assert any("remember_fact" in str(m.get("content", "")) for m in turn2_msgs)
    assert s._facts_nudge_counter == 0   # reset on consume
    assert s._pending_facts_nudge is False


def test_nudge_not_injected_before_threshold():
    """Below threshold the nudge never arms, so its text never reaches the prompt."""
    s = _loop_session()
    s._facts_nudge_threshold = 5
    s._facts_nudge_counter = 0
    s.llm_client.stream_response.side_effect = (
        lambda messages, tools=None, **kw: _good_text_response("Done.")
    )

    asyncio.run(_collect(s.handle_user_input("one")))
    asyncio.run(_collect(s.handle_user_input("two")))
    all_msgs = [
        m for call in s.llm_client.stream_response.call_args_list
        for m in call.args[0]
    ]
    assert not any("remember_fact for each now" in str(m.get("content", "")) for m in all_msgs)
    assert s._pending_facts_nudge is False
    assert s._facts_nudge_counter == 2  # incremented per turn, never consumed

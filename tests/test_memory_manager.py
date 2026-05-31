from unittest.mock import MagicMock
from mcp_chatbot.memory.memory_manager import MemoryManager


def _manager():
    store = MagicMock()
    return MemoryManager(store), store


# ── metadata ──────────────────────────────────────────────────────────────────

def test_safe_tool_names_contains_only_search_memory():
    mgr, _ = _manager()
    assert mgr.safe_tool_names == {"search_memory"}


def test_tool_names_contains_all_four():
    mgr, _ = _manager()
    assert mgr.tool_names == {"remember_fact", "update_fact", "forget_fact", "search_memory"}


def test_tool_schemas_count():
    mgr, _ = _manager()
    assert len(mgr.tool_schemas) == 4


def test_tool_schemas_names():
    mgr, _ = _manager()
    names = {s["function"]["name"] for s in mgr.tool_schemas}
    assert names == {"remember_fact", "update_fact", "forget_fact", "search_memory"}


# ── remember_fact ─────────────────────────────────────────────────────────────

def test_remember_fact_calls_store_with_agent_source():
    mgr, store = _manager()
    store.find_similar_fact.return_value = None
    store.remember_fact.return_value = 7
    result = mgr.execute("remember_fact", {"fact": "User prefers dark mode."})
    store.remember_fact.assert_called_once_with("User prefers dark mode.", source="agent")
    assert "Remembered fact #7" in result
    assert "User prefers dark mode." in result


def test_remember_fact_empty_returns_error():
    mgr, store = _manager()
    result = mgr.execute("remember_fact", {"fact": ""})
    store.find_similar_fact.assert_not_called()
    store.remember_fact.assert_not_called()
    assert "Error" in result


def test_remember_fact_missing_key_returns_error():
    mgr, store = _manager()
    result = mgr.execute("remember_fact", {})
    store.find_similar_fact.assert_not_called()
    store.remember_fact.assert_not_called()
    assert "Error" in result


# ── update_fact ───────────────────────────────────────────────────────────────

def test_update_fact_calls_store_on_success():
    mgr, store = _manager()
    store.update_fact.return_value = True
    result = mgr.execute("update_fact", {"fact_id": 3, "new_fact": "New text."})
    store.update_fact.assert_called_once_with(3, "New text.")
    assert "Updated fact #3" in result


def test_update_fact_error_for_missing_id():
    mgr, store = _manager()
    store.update_fact.return_value = False
    result = mgr.execute("update_fact", {"fact_id": 99, "new_fact": "Text."})
    assert "Error" in result and "99" in result


def test_update_fact_missing_new_fact_returns_error():
    mgr, store = _manager()
    result = mgr.execute("update_fact", {"fact_id": 1})
    store.update_fact.assert_not_called()
    assert "Error" in result


# ── forget_fact ───────────────────────────────────────────────────────────────

def test_forget_fact_calls_store_on_success():
    mgr, store = _manager()
    store.forget_fact.return_value = True
    result = mgr.execute("forget_fact", {"fact_id": 5})
    store.forget_fact.assert_called_once_with(5)
    assert "Deleted fact #5" in result


def test_forget_fact_error_for_missing_id():
    mgr, store = _manager()
    store.forget_fact.return_value = False
    result = mgr.execute("forget_fact", {"fact_id": 99})
    assert "Error" in result and "99" in result


def test_forget_fact_missing_args_returns_error():
    mgr, store = _manager()
    result = mgr.execute("forget_fact", {})
    store.forget_fact.assert_not_called()
    assert "Error" in result


# ── search_memory ─────────────────────────────────────────────────────────────

def test_search_memory_formats_results():
    mgr, store = _manager()
    store.search_memory.return_value = [
        {"type": "episode", "id": 1, "text": "MCP work.", "created_at": 1747612800.0, "source": "agent", "score": -0.5},
    ]
    result = mgr.execute("search_memory", {"query": "MCP"})
    store.search_memory.assert_called_once_with("MCP")
    assert "[episode#1" in result
    assert "MCP work." in result


def test_search_memory_no_results_message():
    mgr, store = _manager()
    store.search_memory.return_value = []
    result = mgr.execute("search_memory", {"query": "xyz"})
    assert result == "No memory matches found."


def test_search_memory_empty_query_returns_error():
    mgr, store = _manager()
    result = mgr.execute("search_memory", {"query": ""})
    store.search_memory.assert_not_called()
    assert "Error" in result


# ── remember_fact dedup ───────────────────────────────────────────────────────

def test_remember_fact_blocks_on_duplicate():
    mgr, store = _manager()
    store.find_similar_fact.return_value = {"id": 3, "fact": "User prefers dark mode."}
    result = mgr.execute("remember_fact", {"fact": "User prefers dark mode setting."})
    store.remember_fact.assert_not_called()
    assert "Similar fact already exists" in result
    assert "#3" in result
    assert "User prefers dark mode." in result
    assert "update_fact" in result


def test_remember_fact_inserts_when_no_duplicate():
    mgr, store = _manager()
    store.find_similar_fact.return_value = None
    store.remember_fact.return_value = 5
    result = mgr.execute("remember_fact", {"fact": "Project uses uv for packaging."})
    store.remember_fact.assert_called_once_with("Project uses uv for packaging.", source="agent")
    assert "Remembered fact #5" in result


def test_remember_fact_duplicate_check_uses_stripped_fact():
    mgr, store = _manager()
    store.find_similar_fact.return_value = None
    store.remember_fact.return_value = 1
    mgr.execute("remember_fact", {"fact": "  User prefers dark mode.  "})
    store.find_similar_fact.assert_called_once_with("User prefers dark mode.")


def test_remember_fact_user_source_still_checked_for_duplicates():
    mgr, store = _manager()
    store.find_similar_fact.return_value = {"id": 2, "fact": "User likes Python."}
    result = mgr.execute("remember_fact", {"fact": "User likes Python.", "source": "user"})
    store.remember_fact.assert_not_called()
    assert "Similar fact already exists" in result


# ── unknown tool ──────────────────────────────────────────────────────────────

def test_unknown_tool_returns_error():
    mgr, _ = _manager()
    result = mgr.execute("does_not_exist", {})
    assert "Error" in result

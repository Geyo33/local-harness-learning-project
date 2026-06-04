import pytest
from mcp_chatbot.memory.store import EpisodicStore
from mcp_chatbot.memory.playbook import PlaybookManager


def _pm(tmp_path) -> PlaybookManager:
    store = EpisodicStore(tmp_path / ".agent" / "memory.db")
    return PlaybookManager(store)


# ── tool interface ────────────────────────────────────────────────────────────

def test_tool_names(tmp_path):
    assert _pm(tmp_path).tool_names == {"record_procedure", "search_playbook"}


def test_safe_tool_names_contains_search_playbook(tmp_path):
    assert _pm(tmp_path).safe_tool_names == {"search_playbook"}


def test_search_playbook_tool_is_safe(tmp_path):
    assert "search_playbook" in _pm(tmp_path).safe_tool_names


# ── execute: record_procedure ─────────────────────────────────────────────────

def test_execute_record_procedure_valid(tmp_path):
    pm = _pm(tmp_path)
    result = pm.execute("record_procedure", {"pattern": "deploy", "action": "step 1"})
    assert "Procedure #1 recorded" in result
    assert "deploy" in result


def test_execute_record_procedure_blank_pattern(tmp_path):
    pm = _pm(tmp_path)
    result = pm.execute("record_procedure", {"pattern": "", "action": "step 1"})
    assert result.startswith("Error:")


def test_execute_record_procedure_blank_action(tmp_path):
    pm = _pm(tmp_path)
    result = pm.execute("record_procedure", {"pattern": "deploy", "action": ""})
    assert result.startswith("Error:")


def test_execute_unknown_tool(tmp_path):
    pm = _pm(tmp_path)
    result = pm.execute("unknown_tool", {})
    assert result.startswith("Error:")


# ── execute: search_playbook ──────────────────────────────────────────────────

def test_search_playbook_execute_formats_output(tmp_path):
    pm = _pm(tmp_path)
    pm.execute("record_procedure", {"pattern": "deploy package", "action": "bump version; run tests; upload"})
    result = pm.execute("search_playbook", {"query": "deploy package"})
    assert "#1" in result
    assert "[task_success]" in result
    assert "deploy package" in result
    assert "→" in result


def test_search_playbook_execute_empty_query(tmp_path):
    pm = _pm(tmp_path)
    result = pm.execute("search_playbook", {"query": ""})
    assert result.startswith("Error:")


def test_search_playbook_execute_no_match(tmp_path):
    pm = _pm(tmp_path)
    result = pm.execute("search_playbook", {"query": "zzzzz qqqqq mmmmm"})
    assert result == "No matching procedures found."


# ── search method ─────────────────────────────────────────────────────────────

def test_search_returns_matching_procedures(tmp_path):
    pm = _pm(tmp_path)
    pm.execute("record_procedure", {"pattern": "deploy docker container to production", "action": "build; push; restart"})
    pm.execute("record_procedure", {"pattern": "write unit tests for feature", "action": "create test; implement; run"})
    results = pm.search("deploy container")
    assert len(results) >= 1
    assert results[0]["pattern"] == "deploy docker container to production"


def test_search_returns_empty_on_no_match(tmp_path):
    pm = _pm(tmp_path)
    pm.execute("record_procedure", {"pattern": "deploy docker container", "action": "build; push; restart"})
    results = pm.search("zzzzz qqqqq mmmmm")
    assert results == []


def test_search_respects_top_n(tmp_path):
    pm = _pm(tmp_path)
    for i in range(10):
        pm.execute("record_procedure", {"pattern": f"deploy service {i} to production environment", "action": f"step {i}"})
    results = pm.search("deploy service to production", top_n=3)
    assert len(results) <= 3


# ── frontend-facing shims (app.py Playbook tab) ───────────────────────────────

def test_get_top_n_returns_entries_with_render_keys(tmp_path):
    pm = _pm(tmp_path)
    pm.execute("record_procedure", {"pattern": "deploy package", "action": "bump; test; upload"})
    entries = pm.get_top_n(50)
    assert len(entries) == 1
    # keys consumed by app.render_playbook
    for key in ("id", "source", "pattern", "action", "created_at", "confidence"):
        assert key in entries[0]


def test_clear_all_removes_all_procedures(tmp_path):
    pm = _pm(tmp_path)
    pm.execute("record_procedure", {"pattern": "A", "action": "a"})
    pm.execute("record_procedure", {"pattern": "B different topic entirely", "action": "b"})
    pm.clear_all()
    assert pm.get_top_n(50) == []

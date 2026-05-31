import json
import os
import time
from pathlib import Path
import pytest


def _pm(tmp_path):
    from mcp_chatbot.memory.playbook import PlaybookManager
    return PlaybookManager(tmp_path)


# ── init ──────────────────────────────────────────────────────────────────────

def test_init_creates_agent_dir(tmp_path):
    _pm(tmp_path)
    assert (tmp_path / ".agent").is_dir()


def test_init_creates_empty_playbook_file(tmp_path):
    _pm(tmp_path)
    data = json.loads((tmp_path / ".agent" / "playbook.json").read_text())
    assert data == {"procedures": []}


def test_init_preserves_existing_data(tmp_path):
    pm = _pm(tmp_path)
    pm.record("pattern A", "action A", "task_success")
    from mcp_chatbot.memory.playbook import PlaybookManager
    pm2 = PlaybookManager(tmp_path)
    assert len(pm2.get_top_n(10)) == 1


def test_init_resets_corrupt_file(tmp_path, caplog):
    import logging
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    (agent_dir / "playbook.json").write_text("not json")
    with caplog.at_level(logging.WARNING):
        pm = _pm(tmp_path)
    assert pm.get_top_n(10) == []
    assert "corrupt" in caplog.text.lower()


# ── record ────────────────────────────────────────────────────────────────────

def test_record_returns_id(tmp_path):
    pm = _pm(tmp_path)
    proc_id = pm.record("deploy", "step 1", "task_success")
    assert proc_id == 1


def test_record_increments_ids(tmp_path):
    pm = _pm(tmp_path)
    id1 = pm.record("A", "a", "task_success")
    id2 = pm.record("B", "b", "task_success")
    assert id1 == 1
    assert id2 == 2


def test_record_stores_all_fields(tmp_path):
    pm = _pm(tmp_path)
    before = time.time()
    pm.record("deploy package", "bump version; run tests; upload", "task_success")
    after = time.time()
    entries = pm.get_top_n(10)
    assert len(entries) == 1
    e = entries[0]
    assert e["id"] == 1
    assert e["pattern"] == "deploy package"
    assert e["action"] == "bump version; run tests; upload"
    assert e["confidence"] == 0.8
    assert e["source"] == "task_success"
    assert before <= e["created_at"] <= after


# ── get_top_n ─────────────────────────────────────────────────────────────────

def test_get_top_n_returns_sorted_by_confidence_desc(tmp_path):
    pm = _pm(tmp_path)
    pm.record("A", "a", "agent")        # 0.6
    pm.record("B", "b", "user")         # 1.0
    pm.record("C", "c", "task_success") # 0.8
    entries = pm.get_top_n(3)
    assert [e["pattern"] for e in entries] == ["B", "C", "A"]


def test_get_top_n_respects_limit(tmp_path):
    pm = _pm(tmp_path)
    for i in range(5):
        pm.record(f"P{i}", f"A{i}", "task_success")
    assert len(pm.get_top_n(3)) == 3


def test_get_top_n_returns_all_when_n_exceeds_count(tmp_path):
    pm = _pm(tmp_path)
    pm.record("A", "a", "task_success")
    assert len(pm.get_top_n(100)) == 1


# ── render_block ──────────────────────────────────────────────────────────────

def test_render_block_empty_string_when_no_entries(tmp_path):
    assert _pm(tmp_path).render_block(10) == ""


def test_render_block_formats_entries_correctly(tmp_path):
    pm = _pm(tmp_path)
    pm.record("deploy package", "bump version; run tests; upload", "task_success")
    block = pm.render_block(10)
    assert block.startswith("[Playbook]:")
    assert "[task_success]" in block
    assert "deploy package" in block
    assert "bump version" in block
    assert "→" in block


# ── clear_all ─────────────────────────────────────────────────────────────────

def test_clear_all_removes_all_entries(tmp_path):
    pm = _pm(tmp_path)
    pm.record("A", "a", "task_success")
    pm.clear_all()
    assert pm.get_top_n(10) == []


# ── tool interface ────────────────────────────────────────────────────────────

def test_tool_names(tmp_path):
    assert _pm(tmp_path).tool_names == {"record_procedure"}


def test_safe_tool_names_is_empty_set(tmp_path):
    assert _pm(tmp_path).safe_tool_names == set()


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


# ── fuzzy dedup ───────────────────────────────────────────────────────────────

def test_record_exact_duplicate_bumps_confidence(tmp_path):
    pm = _pm(tmp_path)
    id1 = pm.record("deploy docker container", "build image; push; restart", "task_success")
    id2 = pm.record("deploy docker container", "build image; push; restart", "task_success")
    assert id1 == id2
    entries = pm.get_top_n(10)
    assert len(entries) == 1
    assert entries[0]["confidence"] == pytest.approx(0.85)


def test_record_similar_paraphrase_consolidates(tmp_path):
    pm = _pm(tmp_path)
    id1 = pm.record("deploy a docker container to production", "build; push; restart", "task_success")
    id2 = pm.record("deploy a docker container to production env", "build; push; reload", "task_success")
    assert id1 == id2
    entries = pm.get_top_n(10)
    assert len(entries) == 1
    assert entries[0]["action"] == "build; push; reload"
    assert entries[0]["confidence"] == pytest.approx(0.85)


def test_record_distinct_patterns_create_separate_entries(tmp_path):
    pm = _pm(tmp_path)
    pm.record("deploy docker container", "step A", "task_success")
    pm.record("write unit tests for new feature", "step B", "task_success")
    assert len(pm.get_top_n(10)) == 2


def test_record_confidence_capped_at_1(tmp_path):
    pm = _pm(tmp_path)
    pm.record("deploy docker container", "step", "task_success")
    for _ in range(5):
        pm.record("deploy docker container", "step", "task_success")
    entries = pm.get_top_n(10)
    assert entries[0]["confidence"] <= 1.0


# ── atomic write ──────────────────────────────────────────────────────────────

def test_atomic_write_uses_tmp_file(tmp_path, monkeypatch):
    written_paths = []
    original_replace = os.replace

    def tracking_replace(src, dst):
        written_paths.append(Path(src).name)
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", tracking_replace)
    pm = _pm(tmp_path)
    pm.record("A", "a", "task_success")
    assert any("tmp" in p for p in written_paths)

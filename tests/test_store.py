import json
import time
import pytest
from pathlib import Path
from mcp_chatbot.memory.store import EpisodicStore


def _store(tmp_path) -> EpisodicStore:
    return EpisodicStore(tmp_path / ".agent" / "memory.db")


# ── record_procedure ──────────────────────────────────────────────────────────

def test_record_procedure_inserts_and_returns_id(tmp_path):
    s = _store(tmp_path)
    pid = s.record_procedure("deploy docker container", "build; push; restart")
    assert pid == 1


def test_record_procedure_increments_ids(tmp_path):
    s = _store(tmp_path)
    id1 = s.record_procedure("deploy docker container", "step A")
    id2 = s.record_procedure("write unit tests for new feature", "step B")
    assert id1 == 1
    assert id2 == 2


def test_record_procedure_stores_all_fields(tmp_path):
    s = _store(tmp_path)
    before = time.time()
    s.record_procedure("deploy package", "bump version; run tests; upload", source="task_success")
    after = time.time()
    results = s.get_top_procedures(10)
    assert len(results) == 1
    p = results[0]
    assert p["pattern"] == "deploy package"
    assert p["action"] == "bump version; run tests; upload"
    assert p["confidence"] == pytest.approx(0.8)
    assert p["source"] == "task_success"
    assert before <= p["created_at"] <= after


def test_record_procedure_exact_dedup_bumps_confidence(tmp_path):
    s = _store(tmp_path)
    id1 = s.record_procedure("deploy docker container", "build; push; restart")
    id2 = s.record_procedure("deploy docker container", "build; push; restart")
    assert id1 == id2
    results = s.get_top_procedures(10)
    assert len(results) == 1
    assert results[0]["confidence"] == pytest.approx(0.85)


def test_record_procedure_similar_paraphrase_dedup(tmp_path):
    s = _store(tmp_path)
    id1 = s.record_procedure("deploy a docker container to production", "build; push; restart")
    id2 = s.record_procedure("deploy a docker container to production env", "build; push; reload")
    assert id1 == id2
    results = s.get_top_procedures(10)
    assert len(results) == 1
    assert results[0]["action"] == "build; push; reload"


def test_record_procedure_distinct_patterns_no_dedup(tmp_path):
    s = _store(tmp_path)
    s.record_procedure("deploy docker container", "step A")
    s.record_procedure("write unit tests for new feature", "step B")
    assert len(s.get_top_procedures(10)) == 2


def test_record_procedure_confidence_capped_at_1(tmp_path):
    s = _store(tmp_path)
    s.record_procedure("deploy docker container", "step")
    for _ in range(10):
        s.record_procedure("deploy docker container", "step")
    results = s.get_top_procedures(10)
    assert results[0]["confidence"] <= 1.0


def test_record_procedure_source_confidence(tmp_path):
    s = _store(tmp_path)
    s.record_procedure("A", "a", source="user")
    s.record_procedure("B", "b", source="agent")
    results = s.get_top_procedures(10)
    confs = {p["pattern"]: p["confidence"] for p in results}
    assert confs["A"] == pytest.approx(1.0)
    assert confs["B"] == pytest.approx(0.6)


# ── get_top_procedures ────────────────────────────────────────────────────────

def test_get_top_procedures_respects_limit(tmp_path):
    s = _store(tmp_path)
    for i in range(5):
        s.record_procedure(f"task {i}", f"action {i}")
    assert len(s.get_top_procedures(3)) == 3


def test_get_top_procedures_returns_all_when_n_exceeds(tmp_path):
    s = _store(tmp_path)
    s.record_procedure("A", "a")
    assert len(s.get_top_procedures(100)) == 1


def test_get_top_procedures_decay_ordering(tmp_path):
    s = _store(tmp_path)
    # Insert two procedures with the same confidence but different last_accessed
    now = time.time()
    s.record_procedure("recent task", "recent action")
    # Manually backdating not possible via API, so we use adjust_confidence
    # to give the second entry a lower effective score: insert via store directly.
    # Use SQLite directly via a second store instance for backdating.
    import sqlite3
    from contextlib import closing
    db = tmp_path / ".agent" / "memory.db"
    with closing(sqlite3.connect(db)) as conn:
        with conn:
            conn.execute(
                "UPDATE procedures SET last_accessed = ? WHERE id = 1",
                (now - 86400 * 30,),  # 30 days ago
            )
    s.record_procedure("old task", "old action")
    results = s.get_top_procedures(10)
    # "old task" was just inserted (recent last_accessed) → higher decay score
    assert results[0]["pattern"] == "old task"


# ── search_procedures ─────────────────────────────────────────────────────────

def test_search_procedures_finds_match(tmp_path):
    s = _store(tmp_path)
    s.record_procedure("deploy docker container to production", "build; push; restart")
    s.record_procedure("write unit tests for feature", "create test; implement; run")
    results = s.search_procedures("deploy container")
    assert len(results) >= 1
    assert results[0]["pattern"] == "deploy docker container to production"


def test_search_procedures_empty_on_no_match(tmp_path):
    s = _store(tmp_path)
    s.record_procedure("deploy docker container", "build; push; restart")
    results = s.search_procedures("zzzzz qqqqq mmmmm")
    assert results == []


def test_search_procedures_recalls_on_long_natural_language_query(tmp_path):
    # The planner injection passes the full user request as the query; FTS5's
    # default implicit-AND would match nothing, so tokens are OR-joined.
    s = _store(tmp_path)
    s.record_procedure("deploy docker container to production", "build; push; restart")
    results = s.search_procedures(
        "I need to deploy my docker container to the production environment right now please"
    )
    assert len(results) >= 1
    assert results[0]["pattern"] == "deploy docker container to production"


def test_search_procedures_ranks_more_matching_tokens_higher(tmp_path):
    s = _store(tmp_path)
    s.record_procedure("deploy docker container to production", "build; push; restart")
    s.record_procedure("write unit tests for new feature", "create; implement; run")
    results = s.search_procedures("deploy a docker container then write unit tests")
    assert results[0]["pattern"] == "deploy docker container to production"


def test_search_procedures_weights_pattern_over_action(tmp_path):
    # A query term matching a procedure's trigger (pattern) should outrank one
    # where the term only appears buried in another procedure's action steps.
    s = _store(tmp_path)
    s.record_procedure("restart the web service", "stop; deploy; start; deploy; verify deploy logs")
    s.record_procedure("deploy application", "build; ship")
    results = s.search_procedures("deploy my app")
    assert results[0]["pattern"] == "deploy application"


def test_search_procedures_demotes_low_confidence_match(tmp_path):
    # Two equally-relevant matches (same keywords, ~equal bm25). The learning
    # loop's penalty must sink the low-confidence one below the healthy one.
    s = _store(tmp_path)
    pid_lo = s.record_procedure("alpha deploy docker container", "step A")
    pid_hi = s.record_procedure("deploy docker container omega", "step B")
    s.adjust_confidence([pid_lo], -0.6)  # 0.8 -> 0.2, still above prune floor

    results = s.search_procedures("deploy docker container")
    ids = [r["id"] for r in results]
    assert pid_lo in ids and pid_hi in ids  # both still recalled
    assert results[0]["id"] == pid_hi  # healthy procedure ranks first

    # Flip the penalty to prove ordering tracks confidence, not insertion order.
    s.adjust_confidence([pid_hi], -0.6)  # both now 0.2
    s.adjust_confidence([pid_lo], +0.6)  # lo back to 0.8
    results = s.search_procedures("deploy docker container")
    assert results[0]["id"] == pid_lo


def test_search_procedures_handles_fts_special_chars(tmp_path):
    # Quoting tokens neutralizes FTS5 operator chars (quotes, parens, colons).
    s = _store(tmp_path)
    s.record_procedure("deploy docker container to production", "build; push; restart")
    results = s.search_procedures('deploy: "container" (prod)!')
    assert len(results) >= 1
    assert results[0]["pattern"] == "deploy docker container to production"


def test_search_procedures_respects_limit(tmp_path):
    s = _store(tmp_path)
    for i in range(10):
        s.record_procedure(f"deploy service {i} to production environment", f"step {i}")
    results = s.search_procedures("deploy service to production", limit=3)
    assert len(results) <= 3


def test_search_procedures_refreshes_last_accessed(tmp_path):
    s = _store(tmp_path)
    s.record_procedure("deploy docker container", "build; push")
    before = time.time()
    s.search_procedures("deploy docker")
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(tmp_path / ".agent" / "memory.db")) as conn:
        row = conn.execute("SELECT last_accessed FROM procedures WHERE id = 1").fetchone()
    assert row[0] >= before


# ── adjust_confidence ─────────────────────────────────────────────────────────

def test_adjust_confidence_increment(tmp_path):
    s = _store(tmp_path)
    pid = s.record_procedure("deploy docker container", "build; push")
    s.adjust_confidence([pid], +0.1)
    results = s.get_top_procedures(10)
    assert results[0]["confidence"] == pytest.approx(0.9)


def test_adjust_confidence_decrement(tmp_path):
    s = _store(tmp_path)
    pid = s.record_procedure("deploy docker container", "build; push")
    s.adjust_confidence([pid], -0.1)
    results = s.get_top_procedures(10)
    assert results[0]["confidence"] == pytest.approx(0.7)


def test_adjust_confidence_caps_at_1(tmp_path):
    s = _store(tmp_path)
    pid = s.record_procedure("deploy docker container", "build; push", source="user")  # 1.0
    s.adjust_confidence([pid], +0.5)
    results = s.get_top_procedures(10)
    assert results[0]["confidence"] == pytest.approx(1.0)


def test_adjust_confidence_prunes_at_floor(tmp_path):
    s = _store(tmp_path)
    pid = s.record_procedure("weak procedure", "old action", source="agent")  # 0.6
    s.adjust_confidence([pid], -0.5)  # → 0.1 → pruned (≤ floor)
    assert s.get_top_procedures(10) == []


def test_adjust_confidence_noop_on_empty_ids(tmp_path):
    s = _store(tmp_path)
    s.record_procedure("A", "a")
    s.adjust_confidence([], -0.5)
    assert len(s.get_top_procedures(10)) == 1


# ── migration ─────────────────────────────────────────────────────────────────

def test_migration_imports_playbook_json(tmp_path):
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    legacy = agent_dir / "playbook.json"
    legacy.write_text(json.dumps({"procedures": [
        {"pattern": "deploy docker", "action": "build; push", "confidence": 0.8,
         "source": "task_success", "created_at": time.time()},
    ]}), encoding="utf-8")
    s = EpisodicStore(agent_dir / "memory.db")
    results = s.get_top_procedures(10)
    assert len(results) == 1
    assert results[0]["pattern"] == "deploy docker"
    assert (agent_dir / "playbook.json.migrated").exists()
    assert not legacy.exists()


def test_migration_idempotent_skips_if_procedures_exist(tmp_path):
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    legacy = agent_dir / "playbook.json"
    legacy.write_text(json.dumps({"procedures": [
        {"pattern": "deploy docker", "action": "build; push", "confidence": 0.8,
         "source": "task_success", "created_at": time.time()},
    ]}), encoding="utf-8")
    # Pre-populate the procedures table by creating the store first
    s = EpisodicStore(agent_dir / "memory.db")
    # Migration ran, now re-create the legacy file to test idempotence
    (agent_dir / "playbook.json.migrated").rename(legacy)
    assert not (agent_dir / "playbook.json.migrated").exists()
    # Second store init: procedures table is not empty → migration skipped
    s2 = EpisodicStore(agent_dir / "memory.db")
    results = s2.get_top_procedures(10)
    assert len(results) == 1
    assert (agent_dir / "playbook.json.migrated").exists()


def test_migration_skips_when_no_legacy_file(tmp_path):
    s = _store(tmp_path)
    assert s.get_top_procedures(10) == []

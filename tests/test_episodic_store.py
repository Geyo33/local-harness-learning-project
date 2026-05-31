from pathlib import Path


def _store(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    return EpisodicStore(tmp_path / ".agent" / "memory.db")


# ── init ──────────────────────────────────────────────────────────────────────

def test_init_creates_db_file(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    EpisodicStore(tmp_path / ".agent" / "memory.db")
    assert (tmp_path / ".agent" / "memory.db").exists()


def test_init_creates_parent_dirs(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    EpisodicStore(tmp_path / "deep" / "nested" / "memory.db")
    assert (tmp_path / "deep" / "nested" / "memory.db").exists()


def test_init_is_idempotent(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    db = tmp_path / "memory.db"
    EpisodicStore(db)
    EpisodicStore(db)  # second init should not raise or corrupt


# ── get_recent ────────────────────────────────────────────────────────────────

def test_get_recent_empty_when_no_episodes(tmp_path):
    assert _store(tmp_path).get_recent(5) == []


def test_get_recent_returns_added_episode(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess-1", "User worked on MCP client.", "agent")
    results = store.get_recent(5)
    assert len(results) == 1
    assert results[0]["session_id"] == "sess-1"
    assert results[0]["summary"] == "User worked on MCP client."
    assert results[0]["source"] == "agent"
    assert isinstance(results[0]["created_at"], float)


def test_get_recent_returns_last_n_by_created_at(tmp_path):
    store = _store(tmp_path)
    # Use explicit created_at values to avoid sub-millisecond timestamp ties
    for i in range(5):
        store.add_episode("sess", f"Summary {i}", "agent", created_at=float(i))
    results = store.get_recent(3)
    assert len(results) == 3
    assert results[0]["summary"] == "Summary 4"
    assert results[1]["summary"] == "Summary 3"
    assert results[2]["summary"] == "Summary 2"


def test_get_recent_returns_all_when_fewer_than_n(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "Only one", "agent")
    assert len(store.get_recent(10)) == 1


# ── clear_all ─────────────────────────────────────────────────────────────────

def test_clear_all_removes_episodes(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "A summary", "agent")
    store.clear_all()
    assert store.get_recent(5) == []


def test_clear_all_is_idempotent(tmp_path):
    store = _store(tmp_path)
    store.clear_all()
    store.clear_all()
    assert store.get_recent(5) == []


# ── persistence ───────────────────────────────────────────────────────────────

def test_data_survives_reconnect(tmp_path):
    from mcp_chatbot.memory.store import EpisodicStore
    db_path = tmp_path / "memory.db"
    EpisodicStore(db_path).add_episode("sess", "Persistent", "agent")
    results = EpisodicStore(db_path).get_recent(5)
    assert len(results) == 1
    assert results[0]["summary"] == "Persistent"


# ── WAL + FTS5 ────────────────────────────────────────────────────────────────

def test_wal_mode_enabled(tmp_path):
    import sqlite3
    from contextlib import closing
    from mcp_chatbot.memory.store import EpisodicStore
    EpisodicStore(tmp_path / "memory.db")
    with closing(sqlite3.connect(str(tmp_path / "memory.db"))) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_fts5_table_created(tmp_path):
    import sqlite3
    from contextlib import closing
    from mcp_chatbot.memory.store import EpisodicStore
    EpisodicStore(tmp_path / "memory.db")
    with closing(sqlite3.connect(str(tmp_path / "memory.db"))) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='episodes_fts'"
        ).fetchone()
    assert row is not None


# ── remember_fact ─────────────────────────────────────────────────────────────

def test_remember_fact_returns_int_id(tmp_path):
    store = _store(tmp_path)
    fact_id = store.remember_fact("User prefers dark mode.")
    assert isinstance(fact_id, int) and fact_id > 0


def test_remember_fact_source_defaults_to_agent(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("Test fact.")
    facts = store.get_key_facts()
    assert facts[0]["source"] == "agent"


def test_remember_fact_custom_source(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("User stated this.", source="user")
    facts = store.get_key_facts()
    assert facts[0]["source"] == "user"


# ── update_fact ───────────────────────────────────────────────────────────────

def test_update_fact_returns_true_on_success(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("Old fact.")
    assert store.update_fact(fid, "New fact.") is True


def test_update_fact_updates_text(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("Old fact.")
    store.update_fact(fid, "New fact.")
    facts = store.get_key_facts()
    assert facts[0]["fact"] == "New fact."


def test_update_fact_returns_false_for_missing_id(tmp_path):
    store = _store(tmp_path)
    assert store.update_fact(999, "Anything.") is False


# ── forget_fact ───────────────────────────────────────────────────────────────

def test_forget_fact_returns_true_on_success(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("To be deleted.")
    assert store.forget_fact(fid) is True


def test_forget_fact_removes_row(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("To be deleted.")
    store.forget_fact(fid)
    assert store.get_key_facts() == []


def test_forget_fact_returns_false_for_missing_id(tmp_path):
    store = _store(tmp_path)
    assert store.forget_fact(999) is False


# ── get_key_facts ─────────────────────────────────────────────────────────────

def test_get_key_facts_empty_when_none(tmp_path):
    assert _store(tmp_path).get_key_facts() == []


def test_get_key_facts_returns_inserted(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("Fact A.")
    facts = store.get_key_facts()
    assert len(facts) == 1
    assert facts[0]["fact"] == "Fact A."


def test_get_key_facts_ordered_by_last_accessed(tmp_path):
    # still valid: update_fact bumps last_accessed, which drives the decay score in get_key_facts
    store = _store(tmp_path)
    id1 = store.remember_fact("Fact 1.")
    store.remember_fact("Fact 2.")
    store.update_fact(id1, "Fact 1 updated.")  # bumps last_accessed for id1
    facts = store.get_key_facts()
    assert facts[0]["id"] == id1


# ── clear_all includes key_facts ──────────────────────────────────────────────

def test_clear_all_removes_key_facts(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("Some fact.")
    store.clear_all()
    assert store.get_key_facts() == []


# ── search_memory ─────────────────────────────────────────────────────────────

def test_search_memory_finds_episode_via_fts5(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "MCP client Gradio development session", "agent")
    results = store.search_memory("Gradio")
    assert any(r["type"] == "episode" for r in results)
    assert any("Gradio" in r["text"] for r in results)


def test_search_memory_finds_fact_via_like(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("User prefers dark mode.")
    results = store.search_memory("dark mode")
    assert any(r["type"] == "fact" for r in results)
    assert any("dark mode" in r["text"] for r in results)


def test_search_memory_no_match_returns_empty(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "Something unrelated.", "agent")
    results = store.search_memory("xyzzy_no_match_abc")
    assert results == []


def test_search_memory_result_has_required_keys(tmp_path):
    store = _store(tmp_path)
    store.add_episode("sess", "Test episode content here.", "agent")
    results = store.search_memory("episode")
    assert len(results) > 0
    assert {"type", "id", "text", "created_at", "source", "score"} <= results[0].keys()


def test_search_memory_results_sorted_by_score(tmp_path):
    store = _store(tmp_path)
    # Episodes and facts with varying keyword density create FTS5 BM25 score spreads.
    # When episodes are appended first then facts, the concatenated list is unsorted
    # unless search_memory globally sorts by score.
    store.add_episode("s1", "Python", "agent")
    store.add_episode("s2", "Python Python Python Python Python", "agent")
    store.remember_fact("Python Python Python Python")
    store.remember_fact("Python")
    results = store.search_memory("Python", limit=10)
    assert len(results) >= 4, "need multiple results to test ordering"
    scores = [r["score"] for r in results]
    assert scores == sorted(scores), "results must be sorted by score ascending"


def test_search_memory_respects_limit(tmp_path):
    store = _store(tmp_path)
    for i in range(5):
        store.add_episode("s", f"Python tutorial episode number {i}", "agent")
    for i in range(5):
        store.remember_fact(f"Python fact number {i}")
    results = store.search_memory("Python", limit=4)
    assert len(results) <= 4, "results must not exceed limit"


# ── _fact_similar ─────────────────────────────────────────────────────────────

def test_fact_similar_identical_strings():
    from mcp_chatbot.memory.store import _fact_similar
    assert _fact_similar("User prefers dark mode.", "User prefers dark mode.") is True


def test_fact_similar_near_duplicate_phrasing():
    from mcp_chatbot.memory.store import _fact_similar
    # same words, minor rephrasing — should match
    assert _fact_similar(
        "User prefers dark mode interface.",
        "User prefers dark mode interface setting.",
    ) is True


def test_fact_similar_distinct_facts():
    from mcp_chatbot.memory.store import _fact_similar
    # completely different content
    assert _fact_similar("User prefers dark mode.", "Project uses Python 3.12.") is False


def test_fact_similar_case_insensitive():
    from mcp_chatbot.memory.store import _fact_similar
    assert _fact_similar("User Prefers Dark Mode.", "user prefers dark mode.") is True


# ── find_similar_fact ─────────────────────────────────────────────────────────

def test_find_similar_fact_empty_store_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.find_similar_fact("User prefers dark mode.") is None


def test_find_similar_fact_returns_none_for_distinct_fact(tmp_path):
    store = _store(tmp_path)
    store.remember_fact("User prefers dark mode.")
    assert store.find_similar_fact("Project uses Python 3.12.") is None


def test_find_similar_fact_returns_match_for_near_duplicate(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("User prefers dark mode interface.")
    match = store.find_similar_fact("User prefers dark mode interface setting.")
    assert match is not None
    assert match["id"] == fid
    assert "dark mode" in match["fact"]


def test_find_similar_fact_returns_dict_with_id_and_fact(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("User prefers dark mode.")
    match = store.find_similar_fact("User prefers dark mode.")
    assert isinstance(match, dict)
    assert "id" in match and "fact" in match
    assert match["id"] == fid


def test_fact_similar_numeric_difference_blocked():
    # Facts differing only by a number are treated as updates, not new facts.
    # "API timeout is 60s" is a correction of "API timeout is 30s" — use update_fact.
    from mcp_chatbot.memory.store import _fact_similar
    assert _fact_similar("API timeout is 30 seconds.", "API timeout is 60 seconds.") is True

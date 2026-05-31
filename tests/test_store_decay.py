import time
from contextlib import closing
from pathlib import Path
from mcp_chatbot.memory.store import EpisodicStore


def _store(tmp_path: Path) -> EpisodicStore:
    return EpisodicStore(tmp_path / "memory.db")


def _insert_fact(store: EpisodicStore, fact: str, source: str, days_ago: float) -> int:
    ts = time.time() - days_ago * 86400
    with closing(store._connect()) as conn:
        with conn:
            cursor = conn.execute(
                "INSERT INTO key_facts (fact, created_at, last_accessed, source) "
                "VALUES (?, ?, ?, ?)",
                (fact, ts, ts, source),
            )
            return cursor.lastrowid


def test_get_key_facts_user_ranks_above_agent_same_age(tmp_path):
    store = _store(tmp_path)
    _insert_fact(store, "agent fact", "agent", days_ago=30)
    _insert_fact(store, "user fact", "user", days_ago=30)
    facts = store.get_key_facts(10)
    assert facts[0]["fact"] == "user fact"
    assert facts[1]["fact"] == "agent fact"


def test_get_key_facts_recently_accessed_ranks_above_old(tmp_path):
    store = _store(tmp_path)
    _insert_fact(store, "old fact", "agent", days_ago=90)
    _insert_fact(store, "recent fact", "agent", days_ago=1)
    facts = store.get_key_facts(10)
    assert facts[0]["fact"] == "recent fact"
    assert facts[1]["fact"] == "old fact"


def test_get_key_facts_respects_limit_after_sort(tmp_path):
    store = _store(tmp_path)
    for i, days in enumerate([90, 60, 45, 30, 1]):
        _insert_fact(store, f"fact_{i}", "agent", days_ago=days)
    facts = store.get_key_facts(3)
    assert len(facts) == 3
    assert facts[0]["fact"] == "fact_4"  # 1 day ago — highest score


def test_search_memory_refreshes_last_accessed(tmp_path):
    store = _store(tmp_path)
    fact_id = _insert_fact(store, "python project", "agent", days_ago=60)

    before = time.time()
    store.search_memory("python")
    after = time.time()

    with closing(store._connect()) as conn:
        cursor = conn.execute(
            "SELECT last_accessed FROM key_facts WHERE id = ?", (fact_id,)
        )
        row = cursor.fetchone()

    assert before <= row[0] <= after

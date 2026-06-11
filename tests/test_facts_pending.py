from mcp_chatbot.memory.store import EpisodicStore


def _store(tmp_path):
    return EpisodicStore(tmp_path / "memory.db")


def test_remember_fact_defaults_to_approved(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("default fact")
    assert store.get_pending_facts() == []
    assert any(f["id"] == fid for f in store.get_key_facts())


def test_remember_fact_pending_goes_to_queue(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("queued fact", status="pending")
    pending = store.get_pending_facts()
    assert [p["id"] for p in pending] == [fid]
    assert store.count_pending() == 1


def test_approve_fact_promotes_and_pins(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("promote me", status="pending")
    assert store.approve_fact(fid, pinned=True) is True
    assert store.count_pending() == 0
    pinned = store.get_pinned_facts(20)
    assert [p["id"] for p in pinned] == [fid]


def test_approve_fact_with_edited_text(tmp_path):
    store = _store(tmp_path)
    fid = store.remember_fact("typo fakt", status="pending")
    store.approve_fact(fid, new_fact="corrected fact", pinned=False)
    facts = {f["id"]: f["fact"] for f in store.get_key_facts()}
    assert facts[fid] == "corrected fact"


def test_approve_unknown_id_returns_false(tmp_path):
    store = _store(tmp_path)
    assert store.approve_fact(99999, pinned=False) is False

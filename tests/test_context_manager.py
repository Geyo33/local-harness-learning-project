import json
import pytest
from mcp_chatbot.core.context_manager import ContextManager


# ── _count ────────────────────────────────────────────────────────────────────

def test_count_short_text():
    assert ContextManager._count("abcd") == 1

def test_count_longer_text():
    assert ContextManager._count("a" * 400) == 100

def test_count_empty_string_returns_one():
    assert ContextManager._count("") == 1


# ── _extract_text ─────────────────────────────────────────────────────────────

def test_extract_text_string():
    assert ContextManager._extract_text("hello") == "hello"

def test_extract_text_list():
    content = [{"type": "text", "text": "hello"}, {"type": "image_url", "url": "..."}]
    assert ContextManager._extract_text(content) == "hello"

def test_extract_text_empty_list():
    assert ContextManager._extract_text([]) == ""

def test_extract_text_none():
    assert ContextManager._extract_text(None) == ""


# ── snapshot ──────────────────────────────────────────────────────────────────

def test_snapshot_returns_expected_keys():
    cm = ContextManager(max_tokens=1000)
    result = cm.snapshot([{"role": "user", "content": "system"}], [])
    assert set(result.keys()) == {"system", "tools", "history", "memory", "reserve", "total", "max", "pct"}

def test_snapshot_memory_always_zero():
    cm = ContextManager(max_tokens=1000)
    result = cm.snapshot([{"role": "user", "content": "system"}], [])
    assert result["memory"] == 0

def test_snapshot_max_reflects_constructor():
    cm = ContextManager(max_tokens=1000)
    result = cm.snapshot([{"role": "user", "content": "system"}], [])
    assert result["max"] == 1000

def test_snapshot_system_counts_first_message():
    cm = ContextManager(max_tokens=1000)
    result = cm.snapshot([{"role": "user", "content": "a" * 40}], [])
    assert result["system"] == 10

def test_snapshot_history_counts_remaining_messages():
    cm = ContextManager(max_tokens=1000)
    messages = [
        {"role": "user", "content": "system"},
        {"role": "user", "content": "a" * 40},
        {"role": "assistant", "content": "b" * 40},
    ]
    result = cm.snapshot(messages, [])
    assert result["history"] == 20

def test_snapshot_tools_counts_serialized_schemas():
    cm = ContextManager(max_tokens=1000)
    schemas = [{"function": {"name": "t"}}]
    result = cm.snapshot([{"role": "user", "content": "s"}], schemas)
    assert result["tools"] == ContextManager._count(json.dumps(schemas))

def test_snapshot_reserve_is_max_minus_total():
    cm = ContextManager(max_tokens=1000)
    result = cm.snapshot([{"role": "user", "content": "a" * 400}], [])
    assert result["reserve"] == 1000 - result["total"]

def test_snapshot_pct_is_total_over_max():
    cm = ContextManager(max_tokens=1000)
    result = cm.snapshot([{"role": "user", "content": "a" * 400}], [])
    assert result["pct"] == pytest.approx(result["total"] / 1000)

def test_snapshot_empty_messages():
    cm = ContextManager(max_tokens=1000)
    result = cm.snapshot([], [])
    assert result["system"] == 0
    assert result["history"] == 0
    assert result["total"] == 0

def test_snapshot_skips_none_content():
    cm = ContextManager(max_tokens=1000)
    messages = [
        {"role": "user", "content": "system"},
        {"role": "tool", "content": None},
    ]
    result = cm.snapshot(messages, [])
    assert result["history"] == 0

def test_snapshot_system_handles_missing_content_key():
    cm = ContextManager(max_tokens=1000)
    messages = [{"role": "tool", "tool_call_id": "abc"}]
    result = cm.snapshot(messages, [])
    assert result["system"] == 1  # _count("") = max(1, 0) = 1

def test_breakdown_property_matches_last_snapshot():
    cm = ContextManager(max_tokens=1000)
    result = cm.snapshot([{"role": "user", "content": "system"}], [])
    assert cm.breakdown == result


# ── is_near_limit ─────────────────────────────────────────────────────────────

def test_is_near_limit_at_threshold_returns_true():
    cm = ContextManager(max_tokens=100, threshold=0.70)
    # "a" * 280 → _count = 70 → pct = 0.70
    cm.snapshot([{"role": "user", "content": "a" * 280}], [])
    assert cm.is_near_limit() is True

def test_is_near_limit_below_threshold_returns_false():
    cm = ContextManager(max_tokens=1000, threshold=0.70)
    cm.snapshot([{"role": "user", "content": "a" * 4}], [])
    assert cm.is_near_limit() is False

def test_is_near_limit_default_threshold():
    cm = ContextManager(max_tokens=100)
    assert cm.threshold == 0.70

def test_is_near_limit_custom_threshold():
    cm = ContextManager(max_tokens=100, threshold=0.50)
    # "a" * 200 → _count = 50 → pct = 0.50
    cm.snapshot([{"role": "user", "content": "a" * 200}], [])
    assert cm.is_near_limit() is True

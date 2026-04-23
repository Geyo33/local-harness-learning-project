from pathlib import Path
import pytest


def _tm(tmp_path):
    from mcp_chatbot.core.task_manager import TaskManager
    return TaskManager(tmp_path)


# ── init ──────────────────────────────────────────────────────────────────────

def test_init_creates_agent_dir(tmp_path):
    _tm(tmp_path)
    assert (tmp_path / ".agent").is_dir()


def test_init_creates_empty_tasks_file(tmp_path):
    import json
    _tm(tmp_path)
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    assert data == {"tasks": []}


def test_init_resets_existing_state(tmp_path):
    import json
    tm = _tm(tmp_path)
    tm.plan_tasks(["Task A"])
    # Re-initialise — should reset
    from mcp_chatbot.core.task_manager import TaskManager
    TaskManager(tmp_path)
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    assert data == {"tasks": []}


# ── is_empty ──────────────────────────────────────────────────────────────────

def test_is_empty_true_when_no_tasks(tmp_path):
    assert _tm(tmp_path).is_empty() is True


def test_is_empty_false_after_plan(tmp_path):
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])
    assert tm.is_empty() is False


# ── plan_tasks ────────────────────────────────────────────────────────────────

def test_plan_tasks_creates_tasks_with_sequential_ids(tmp_path):
    import json
    tm = _tm(tmp_path)
    tm.plan_tasks(["Parse CSV", "Compute totals"])
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    assert data["tasks"][0] == {"id": "1", "title": "Parse CSV", "status": "pending"}
    assert data["tasks"][1] == {"id": "2", "title": "Compute totals", "status": "pending"}


def test_plan_tasks_replaces_existing_plan(tmp_path):
    tm = _tm(tmp_path)
    tm.plan_tasks(["Old task"])
    tm.plan_tasks(["New task"])
    assert tm.is_empty() is False
    import json
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    assert len(data["tasks"]) == 1
    assert data["tasks"][0]["title"] == "New task"
    assert data["tasks"][0]["id"] == "1"


def test_plan_tasks_returns_confirmation_string(tmp_path):
    tm = _tm(tmp_path)
    result = tm.plan_tasks(["A", "B"])
    assert "2" in result
    assert "A" in result


# ── render_block ──────────────────────────────────────────────────────────────

def test_render_block_empty_when_no_tasks(tmp_path):
    assert _tm(tmp_path).render_block() == ""


def test_render_block_format(tmp_path):
    tm = _tm(tmp_path)
    tm.plan_tasks(["Parse CSV", "Write report"])
    block = tm.render_block()
    assert block.startswith("<tasks>")
    assert block.endswith("</tasks>")
    assert "[ ] 1. Parse CSV" in block
    assert "[ ] 2. Write report" in block


def test_render_block_status_glyphs(tmp_path):
    from mcp_chatbot.core.task_manager import TaskManager
    tm = TaskManager(tmp_path)
    tm.plan_tasks(["A", "B", "C", "D"])
    tm.update_task("1", "in_progress")
    tm.update_task("2", "done")
    tm.update_task("3", "cancelled")
    block = tm.render_block()
    assert "[~] 1. A" in block
    assert "[x] 2. B" in block
    assert "[-] 3. C" in block
    assert "[ ] 4. D" in block


# ── update_task ───────────────────────────────────────────────────────────────

def test_update_task_changes_status(tmp_path):
    import json
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])
    tm.update_task("1", "in_progress")
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    assert data["tasks"][0]["status"] == "in_progress"


def test_update_task_unknown_id_returns_error(tmp_path):
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])
    result = tm.update_task("99", "done")
    assert "No task with id=99" in result
    assert "1" in result  # lists current ids


def test_update_task_invalid_status_returns_error(tmp_path):
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])
    result = tm.update_task("1", "flying")
    assert "Error" in result or "invalid" in result.lower()


def test_update_task_returns_confirmation(tmp_path):
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])
    result = tm.update_task("1", "done")
    assert "1" in result
    assert "done" in result


# ── add_task ──────────────────────────────────────────────────────────────────

def test_add_task_appends_when_no_after(tmp_path):
    import json
    tm = _tm(tmp_path)
    tm.plan_tasks(["A", "B"])
    tm.add_task("C")
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    assert data["tasks"][-1]["title"] == "C"
    assert data["tasks"][-1]["id"] == "3"


def test_add_task_inserts_after_given_id(tmp_path):
    import json
    tm = _tm(tmp_path)
    tm.plan_tasks(["A", "C"])
    tm.add_task("B", after="1")
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    titles = [t["title"] for t in data["tasks"]]
    assert titles == ["A", "B", "C"]


def test_add_task_auto_increments_id(tmp_path):
    tm = _tm(tmp_path)
    tm.plan_tasks(["A", "B"])
    tm.add_task("C")
    import json
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    ids = [t["id"] for t in data["tasks"]]
    assert ids == ["1", "2", "3"]


def test_add_task_unknown_after_returns_error(tmp_path):
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])
    result = tm.add_task("B", after="99")
    assert "No task with id=99" in result


def test_add_task_returns_confirmation(tmp_path):
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])
    result = tm.add_task("B")
    assert "B" in result
    assert "2" in result


# ── tool_names ────────────────────────────────────────────────────────────────

def test_tool_names(tmp_path):
    tm = _tm(tmp_path)
    assert tm.tool_names == {"plan_tasks", "update_task", "add_task"}


# ── execute dispatcher ────────────────────────────────────────────────────────

def test_execute_plan_tasks(tmp_path):
    tm = _tm(tmp_path)
    result = tm.execute("plan_tasks", {"titles": ["X", "Y"]})
    assert "X" in result
    assert not tm.is_empty()


def test_execute_update_task(tmp_path):
    import json
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])
    tm.execute("update_task", {"id": "1", "status": "done"})
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    assert data["tasks"][0]["status"] == "done"


def test_execute_add_task(tmp_path):
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])
    tm.execute("add_task", {"title": "B"})
    import json
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    assert len(data["tasks"]) == 2


def test_execute_unknown_tool_returns_error(tmp_path):
    tm = _tm(tmp_path)
    result = tm.execute("unknown_tool", {})
    assert "Error" in result or "unknown" in result.lower()


# ── tool_schemas ──────────────────────────────────────────────────────────────

def test_tool_schemas_returns_three_schemas(tmp_path):
    tm = _tm(tmp_path)
    assert len(tm.tool_schemas) == 3


def test_tool_schemas_names(tmp_path):
    tm = _tm(tmp_path)
    names = {s["function"]["name"] for s in tm.tool_schemas}
    assert names == {"plan_tasks", "update_task", "add_task"}


def test_tool_schemas_plan_tasks_has_titles_array(tmp_path):
    tm = _tm(tmp_path)
    schema = next(s for s in tm.tool_schemas if s["function"]["name"] == "plan_tasks")
    props = schema["function"]["parameters"]["properties"]
    assert "titles" in props
    assert props["titles"]["type"] == "array"
    assert props["titles"]["items"]["type"] == "string"
    assert "titles" in schema["function"]["parameters"]["required"]


def test_tool_schemas_update_task_has_enum_status(tmp_path):
    tm = _tm(tmp_path)
    schema = next(s for s in tm.tool_schemas if s["function"]["name"] == "update_task")
    props = schema["function"]["parameters"]["properties"]
    assert set(props["status"]["enum"]) == {"pending", "in_progress", "done", "cancelled"}
    assert "id" in schema["function"]["parameters"]["required"]
    assert "status" in schema["function"]["parameters"]["required"]


def test_tool_schemas_add_task_after_is_optional(tmp_path):
    tm = _tm(tmp_path)
    schema = next(s for s in tm.tool_schemas if s["function"]["name"] == "add_task")
    required = schema["function"]["parameters"].get("required", [])
    assert "title" in required
    assert "after" not in required


def test_tool_schemas_type_is_function(tmp_path):
    tm = _tm(tmp_path)
    for schema in tm.tool_schemas:
        assert schema["type"] == "function"


# ── edge cases (2.7) ──────────────────────────────────────────────────────────

def test_init_warns_on_corrupt_tasks_file(tmp_path, caplog):
    import logging
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    (agent_dir / "tasks.json").write_text("{corrupt!", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        from mcp_chatbot.core.task_manager import TaskManager
        TaskManager(tmp_path)

    assert any("corrupt" in r.message.lower() for r in caplog.records)


def test_init_resets_corrupt_file_to_empty(tmp_path):
    import json
    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    (agent_dir / "tasks.json").write_text("{corrupt!", encoding="utf-8")

    from mcp_chatbot.core.task_manager import TaskManager
    tm = TaskManager(tmp_path)
    assert tm.is_empty()
    data = json.loads((tmp_path / ".agent" / "tasks.json").read_text())
    assert data == {"tasks": []}


def test_init_no_corrupt_warning_when_file_missing(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        from mcp_chatbot.core.task_manager import TaskManager
        TaskManager(tmp_path)

    assert not any("corrupt" in r.message.lower() for r in caplog.records)


def test_read_warns_on_mid_session_corruption(tmp_path, caplog):
    import logging
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])
    (tmp_path / ".agent" / "tasks.json").write_text("{corrupt!", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        block = tm.render_block()

    assert block == ""
    assert any("corrupt" in r.message.lower() for r in caplog.records)


def test_plan_tasks_warns_when_overwriting(tmp_path, caplog):
    import logging
    tm = _tm(tmp_path)
    tm.plan_tasks(["A"])

    with caplog.at_level(logging.WARNING):
        tm.plan_tasks(["B"])

    assert any("replacing" in r.message.lower() for r in caplog.records)

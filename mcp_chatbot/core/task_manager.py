from __future__ import annotations

import json
import logging
import os
from pathlib import Path

_VALID_STATUSES = {"pending", "in_progress", "done"}
_STATUS_GLYPHS = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "done": "[x]",
}


class TaskManager:
    def __init__(self, root: Path) -> None:
        self._agent_dir = root / ".agent"
        self._state_file = self._agent_dir / "tasks.json"
        self._agent_dir.mkdir(parents=True, exist_ok=True)
        if self._state_file.exists():
            try:
                json.loads(self._state_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logging.warning(
                    "tasks.json at %s is corrupt — resetting to empty plan.",
                    self._state_file,
                )
        self._write({"tasks": []})
        self._last_all_done: bool = False
        logging.info("TaskManager initialized at %s", self._state_file)

    def _write(self, data: dict) -> None:
        tmp = self._state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self._state_file)

    def clear_plan(self) -> None:
        """Wipe the plan. Called once after all tasks complete + playbook prompt."""
        self._write({"tasks": []})

    def _read(self) -> dict:
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"tasks": []}
        except json.JSONDecodeError:
            logging.warning("tasks.json is corrupt — using empty plan.")
            return {"tasks": []}

    def is_completed(self, data) -> bool:
        tasks = data.get("tasks", [])
        if not tasks:
            return False
        for task in tasks:
            if task.get("steps"):
                if not all(s["status"] == "done" for s in task["steps"]):
                    return False
            else:
                if task["status"] != "done":
                    return False
        return True

    def is_empty(self) -> bool:
        return not self._read()["tasks"]

    def plan_text(self) -> str:
        """Concatenated task + step titles of the current plan.

        Used for procedure attribution: comparing what actually got planned/run
        against each injected procedure's action text to estimate which one the
        model followed."""
        parts: list[str] = []
        for t in self._read().get("tasks", []):
            if t.get("title"):
                parts.append(t["title"])
            for s in t.get("steps") or []:
                if s.get("title"):
                    parts.append(s["title"])
        return " ".join(parts)

    def get_in_progress_ids(self) -> list[str]:
        """Return sorted list of task/step ids currently in_progress."""
        data = self._read()
        ids: list[str] = []
        for t in data.get("tasks", []):
            if t["status"] == "in_progress":
                ids.append(t["id"])
            for s in t.get("steps", []):
                if s["status"] == "in_progress":
                    ids.append(s["id"])
        return sorted(ids)

    def plan_tasks(self, tasks: list[dict | str]) -> str:
        existing = self._read()["tasks"]
        if existing:
            logging.warning("plan_tasks called while plan exists — replacing.")
        new_tasks = []
        for i, t in enumerate(tasks):
            task_id = str(i + 1)
            title = t if isinstance(t, str) else t["title"]
            step_titles = [] if isinstance(t, str) else t.get("steps", [])
            steps = [
                {
                    "id": f"{task_id}.{j + 1}",
                    "title": st if isinstance(st, str) else (st.get("title") or ""),
                    "status": "pending",
                }
                for j, st in enumerate(step_titles)
            ]
            entry = {"id": task_id, "title": title, "status": "pending"}
            if steps:
                entry["steps"] = steps
            new_tasks.append(entry)
        self._write({"tasks": new_tasks})
        summary = ", ".join(f"{t['id']}. {t['title']}" for t in new_tasks)
        return f"Plan created with {len(new_tasks)} task(s): {summary}"

    def render_block(self) -> str:
        tasks = self._read()["tasks"]
        if not tasks:
            return ""
        lines = []
        for t in tasks:
            lines.append(f"{_STATUS_GLYPHS.get(t['status'], '[ ]')} {t['id']}. {t['title']}")
            for s in t.get("steps", []):
                lines.append(f"  {_STATUS_GLYPHS.get(s['status'], '[ ]')} {s['id']}. {s['title']}")
        return "<tasks>\n" + "\n".join(lines) + "\n</tasks>"

    def _ordering_block(self, siblings: list, idx: int, status: str) -> str | None:
        """Sequential-ordering failsafe. Returns an error string if setting
        siblings[idx] to `status` would skip ahead, else None.

        `done` and `in_progress` are both gated: every earlier sibling must be
        `done` first (no marking out of order, no second item started while a
        prior one is unfinished). `pending` (a reset) is never blocked. Applied
        independently at each level — task ids gate against sibling tasks, step
        ids against sibling steps under the same parent."""
        if status not in ("done", "in_progress"):
            return None
        unfinished = [s["id"] for s in siblings[:idx] if s["status"] != "done"]
        if unfinished:
            target = siblings[idx]["id"]
            return (
                f"Blocked: cannot set {target} to '{status}' — earlier "
                f"item(s) not done yet: {', '.join(unfinished)}. "
                f"Complete them in order first."
            )
        return None

    def update_task(self, id: str, status: str) -> str:
        if status not in _VALID_STATUSES:
            return f"Error: invalid status '{status}'. Valid: {sorted(_VALID_STATUSES)}"
        data = self._read()

        if "." in id:
            parent_id, _ = id.split(".", 1)
            for task in data["tasks"]:
                if task["id"] == parent_id:
                    steps = task.get("steps", [])
                    for i, step in enumerate(steps):
                        if step["id"] == id:
                            block = self._ordering_block(steps, i, status)
                            if block:
                                return block
                            step["status"] = status
                            auto_msg = ""
                            if status == "done" and all(s["status"] == "done" for s in steps):
                                task["status"] = "done"
                                auto_msg = f" Task {parent_id} auto-completed (all steps done)."
                            self._last_all_done = self.is_completed(data)
                            self._write(data)
                            return f"Step {id} updated to '{status}'.{auto_msg}"
                    return f"No step with id={id} in task {parent_id}."
            return f"No task with id={parent_id}."

        for i, task in enumerate(data["tasks"]):
            if task["id"] == id:
                block = self._ordering_block(data["tasks"], i, status)
                if block:
                    return block
                task["status"] = status
                cascade_msg = ""
                if status == "done" and task.get("steps"):
                    for s in task["steps"]:
                        s["status"] = "done"
                    cascade_msg = f" ({len(task['steps'])} step(s) also marked done.)"
                self._last_all_done = self.is_completed(data)
                self._write(data)
                return f"Task {id} updated to '{status}'.{cascade_msg}"
        current_ids = [t["id"] for t in data["tasks"]]
        return f"No task with id={id}. Current ids: {current_ids}"

    def add_task(self, title: str, after: str | None = None) -> str:
        data = self._read()
        tasks = data["tasks"]
        new_id = str(int(after) + 1) if after else str(max((int(t["id"]) for t in tasks), default=0) + 1)
        new_task = {"id": new_id, "title": title, "status": "pending"}
        if after is not None:
            idx = next((i for i, t in enumerate(tasks) if t["id"] == after), None)
            if idx is None:
                current_ids = [t["id"] for t in tasks]
                return f"No task with id={after}. Current ids: {current_ids}"
            tasks.insert(idx + 1, new_task)
            for i in range(idx + 2, len(tasks)):
                new_tid = str(int(tasks[i]["id"]) + 1)
                tasks[i]["id"] = new_tid
                for s in tasks[i].get("steps", []):
                    s["id"] = f"{new_tid}.{s['id'].split('.', 1)[1]}"
        else:
            tasks.append(new_task)
        self._write(data)
        return f"Added task {new_id}: '{title}'."

    def add_step(self, parent_id: str, title: str, after: str | None = None) -> str:
        data = self._read()
        for task in data["tasks"]:
            if task["id"] == parent_id:
                steps = task.setdefault("steps", [])
                existing_nums = [int(s["id"].split(".")[1]) for s in steps] if steps else [0]
                new_num = max(existing_nums) + 1
                new_step = {"id": f"{parent_id}.{new_num}", "title": title, "status": "pending"}
                if after is not None:
                    idx = next((i for i, s in enumerate(steps) if s["id"] == after), None)
                    if idx is None:
                        return f"No step with id={after}."
                    steps.insert(idx + 1, new_step)
                    for j, s in enumerate(steps):
                        s["id"] = f"{parent_id}.{j + 1}"
                else:
                    steps.append(new_step)
                self._write(data)
                return f"Added step {new_step['id']}: '{title}'."
        return f"No task with id={parent_id}. Current ids: {[t['id'] for t in data['tasks']]}"

    @property
    def tool_names(self) -> set[str]:
        return {"plan_tasks", "update_task", "add_task", "add_step"}

    @property
    def safe_tool_names(self) -> set[str]:
        return {"update_task"}

    def execute(self, tool_name: str, arguments: dict) -> str:
        try:
            if tool_name == "plan_tasks":
                tasks = arguments.get("tasks") or arguments.get("titles")
                if not tasks:
                    return "Error: 'tasks' must be a non-empty list."
                return self.plan_tasks(tasks)
            if tool_name == "update_task":
                return self.update_task(arguments.get("id", ""), arguments.get("status", ""))
            if tool_name == "add_task":
                title = arguments.get("title")
                if not title:
                    return "Error: 'title' is required for add_task."
                return self.add_task(title, arguments.get("after"))
            if tool_name == "add_step":
                parent_id = arguments.get("parent_id")
                title = arguments.get("title")
                if not parent_id or not title:
                    return "Error: 'parent_id' and 'title' are required."
                return self.add_step(parent_id, title, arguments.get("after"))
            return f"Error: unknown task tool '{tool_name}'."
        except Exception as e:
            return f"Error: {e}"

    @property
    def tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "plan_tasks",
                    "description": (
                        "If no task list yet, ALWAYS call this first for any multi-step request (3+ steps). "
                        "Pass an ordered list of tasks. Each task can optionally include "
                        "a 'steps' list for finer-grained sub-steps."
                        "\nAlways gather enough context before planning(ask user if unsure) to be able to organize the list into clear actionnable steps."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {
                                            "type": "string",
                                            "description": "Task title (≤80 chars).",
                                        },
                                        "steps": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Optional ordered sub-steps. Each item is a PLAIN STRING (≤80 chars), NOT an object.",
                                        },
                                    },
                                    "required": ["title"],
                                },
                                "description": "Ordered list of tasks.",
                            }
                        },
                        "required": ["tasks"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_task",
                    "description": "Update the status of a task or step by its id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Task id (e.g. '1') or step id (e.g. '1.2').",
                            },
                            "status": {
                                "type": "string",
                                "enum": sorted(_VALID_STATUSES),
                                "description": "New status.",
                            },
                        },
                        "required": ["id", "status"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "add_task",
                    "description": "Add a new task to the plan. Optionally insert it after a specific task id. Use only when you need an additional task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Title of the new task.",
                            },
                            "after": {
                                "type": "string",
                                "description": "Insert after this task id. Omit to append at the end.",
                            },
                        },
                        "required": ["title"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "add_step",
                    "description": "Add a new step to an existing task. Use only when you need an additional step for a task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "parent_id": {
                                "type": "string",
                                "description": "Task id to add the step to (e.g. '1').",
                            },
                            "title": {
                                "type": "string",
                                "description": "Step title (≤80 chars).",
                            },
                            "after": {
                                "type": "string",
                                "description": "Insert after this step id (e.g. '1.2'). Omit to append.",
                            },
                        },
                        "required": ["parent_id", "title"],
                    },
                },
            },
        ]

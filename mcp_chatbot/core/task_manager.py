from __future__ import annotations

import json
import logging
import os
from pathlib import Path

_VALID_STATUSES = {"pending", "in_progress", "done", "cancelled"}
_STATUS_GLYPHS = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "done": "[x]",
    "cancelled": "[-]",
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
        logging.info("TaskManager initialized at %s", self._state_file)

    def _write(self, data: dict) -> None:
        tmp = self._state_file.with_suffix(".json.tmp")
        if self.is_completed(data):
            data = {"tasks": []}
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self._state_file)

    def _read(self) -> dict:
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"tasks": []}
        except json.JSONDecodeError:
            logging.warning("tasks.json is corrupt — using empty plan.")
            return {"tasks": []}

    def is_completed(self, data) -> bool:
        completed_check = [True if task["status"] == "done" else False for task in data["tasks"]]
        return all(completed_check)

    def is_empty(self) -> bool:
        return not self._read()["tasks"]

    def plan_tasks(self, titles: list[str]) -> str:
        existing = self._read()["tasks"]
        if existing:
            logging.warning("plan_tasks called while plan exists — replacing.")
        tasks = [
            {"id": str(i + 1), "title": t, "status": "pending"}
            for i, t in enumerate(titles)
        ]
        self._write({"tasks": tasks})
        summary = ", ".join(f"{t['id']}. {t['title']}" for t in tasks)
        return f"Plan created with {len(tasks)} task(s): {summary}"

    def render_block(self) -> str:
        tasks = self._read()["tasks"]
        if not tasks:
            return ""
        lines = [
            f"{_STATUS_GLYPHS.get(t['status'], '[ ]')} {t['id']}. {t['title']}"
            for t in tasks
        ]
        return "<tasks>\n" + "\n".join(lines) + "\n</tasks>"

    def update_task(self, id: str, status: str) -> str:
        if status not in _VALID_STATUSES:
            return f"Error: invalid status '{status}'. Valid: {sorted(_VALID_STATUSES)}"
        data = self._read()
        for task in data["tasks"]:
            if task["id"] == id:
                task["status"] = status
                self._write(data)
                return f"Success: Task {id} updated to '{status}'."
        current_ids = [t["id"] for t in data["tasks"]]
        return f"No task with id={id}. Current ids: {current_ids}"

    def add_task(self, title: str, after: str | None = None) -> str:
        data = self._read()
        tasks = data["tasks"]
        new_id = str(int(after)+1) if after else str(max((int(t["id"]) for t in tasks), default=0) + 1)
        new_task = {"id": new_id, "title": title, "status": "pending"}
        if after is not None:
            idx = next((i for i, t in enumerate(tasks) if t["id"] == after), None)
            if idx is None:
                current_ids = [t["id"] for t in tasks]
                return f"No task with id={after}. Current ids: {current_ids}"
            tasks.insert(idx + 1, new_task)
            for i in range(idx + 2, len(tasks)):
                # Increment the ID of the task at after insert index 'i'
                current_id = int(tasks[i]["id"])
                tasks[i]["id"] = str(current_id + 1)
        else:
            tasks.append(new_task)
        self._write(data)
        return f"Added task {new_id}: '{title}'."

    @property
    def tool_names(self) -> set[str]:
        return {"plan_tasks", "update_task", "add_task"}

    @property
    def safe_tool_names(self) -> set[str]:
        return {"update_task"}

    def execute(self, tool_name: str, arguments: dict) -> str:
        try:
            if tool_name == "plan_tasks":
                titles = arguments.get("titles")
                if not titles:
                    return "Error: 'titles' must be a non-empty list for plan_tasks."
                return self.plan_tasks(titles)
            if tool_name == "update_task":
                return self.update_task(arguments.get("id", ""), arguments.get("status", ""))
            if tool_name == "add_task":
                title = arguments.get("title")
                if not title:
                    return "Error: 'title' is required for add_task."
                return self.add_task(title, arguments.get("after"))
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
                    "description": "Create or replace the current task plan. Use at the start of a multi-step task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "titles": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Ordered list of task titles.",
                            }
                        },
                        "required": ["titles"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_task",
                    "description": "Update the status of a task by its id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Task id (e.g. '1', '2').",
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
                    "description": "Add a new task to the plan. Optionally insert it after a specific task id.",
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
        ]

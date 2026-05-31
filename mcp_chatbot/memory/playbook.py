from __future__ import annotations

import json
import logging
import os
import time
from difflib import SequenceMatcher
from pathlib import Path

_PATTERN_SIMILARITY_THRESHOLD = 0.85


def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= _PATTERN_SIMILARITY_THRESHOLD


_CONFIDENCE_BY_SOURCE = {
    "task_success": 0.8,
    "user": 1.0,
    "agent": 0.6,
}


class PlaybookManager:
    def __init__(self, root: Path) -> None:
        self._agent_dir = root / ".agent"
        self._state_file = self._agent_dir / "playbook.json"
        self._agent_dir.mkdir(parents=True, exist_ok=True)
        if not self._state_file.exists():
            self._write({"procedures": []})
        else:
            try:
                data = json.loads(self._state_file.read_text(encoding="utf-8"))
                if "procedures" not in data:
                    raise ValueError("Missing 'procedures' key")
            except (json.JSONDecodeError, ValueError):
                logging.warning(
                    "playbook.json at %s is corrupt — resetting.", self._state_file
                )
                self._write({"procedures": []})
        logging.info("PlaybookManager initialized at %s", self._state_file)

    def _write(self, data: dict) -> None:
        tmp = self._state_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self._state_file)

    def _read(self) -> dict:
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"procedures": []}
        except json.JSONDecodeError:
            logging.warning("playbook.json is corrupt — using empty playbook.")
            return {"procedures": []}

    def record(self, pattern: str, action: str, source: str) -> int:
        data = self._read()
        procedures = data["procedures"]
        existing = next((p for p in procedures if _similar(p["pattern"], pattern)), None)
        if existing:
            existing["confidence"] = min(1.0, existing["confidence"] + 0.05)
            existing["action"] = action
            self._write(data)
            return existing["id"]
        new_id = max((p["id"] for p in procedures), default=0) + 1
        procedures.append({
            "id": new_id,
            "pattern": pattern,
            "action": action,
            "confidence": _CONFIDENCE_BY_SOURCE.get(source, 0.7),
            "source": source,
            "created_at": time.time(),
        })
        self._write(data)
        return new_id

    def get_top_n(self, n: int) -> list[dict]:
        data = self._read()
        return sorted(data["procedures"], key=lambda p: p["confidence"], reverse=True)[:n]

    def render_block(self, n: int) -> str:
        entries = self.get_top_n(n)
        if not entries:
            return ""
        lines = [f"- [{e['source']}] {e['pattern']} → {e['action']}" for e in entries]
        return "[Playbook]:\n" + "(Collection of successful procedures to apply in similar situations)\n\n" + "\n\n".join(lines)

    def clear_all(self) -> None:
        self._write({"procedures": []})

    @property
    def tool_names(self) -> set[str]:
        return {"record_procedure"}

    @property
    def safe_tool_names(self) -> set[str]:
        return set()

    def execute(self, tool_name: str, arguments: dict) -> str:
        try:
            if tool_name == "record_procedure":
                pattern = arguments.get("pattern", "").strip()
                action = arguments.get("action", "").strip()
                if not pattern:
                    return "Error: 'pattern' is required."
                if not action:
                    return "Error: 'action' is required."
                proc_id = self.record(pattern, action, source="task_success")
                return f"Procedure #{proc_id} recorded: {pattern}"
            return f"Error: unknown playbook tool '{tool_name}'."
        except Exception as e:
            logging.error("PlaybookManager.%s error: %s", tool_name, e)
            return f"Error: {e}"

    @property
    def tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "record_procedure",
                    "description": (
                        "Record a reusable procedure in the playbook. "
                        "Call after completing a multi-step task that is likely to recur. "
                        "pattern = the general situation; action = the step-by-step approach."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The general situation or trigger for this procedure, has to be short and concise.",
                            },
                            "action": {
                                "type": "string",
                                "description": "The step-by-step approach to take.",
                            },
                        },
                        "required": ["pattern", "action"],
                    },
                },
            }
        ]

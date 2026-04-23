from __future__ import annotations

import json
import os
from pathlib import Path

_SETTINGS_PATH = Path("mcp_chatbot") / "settings.json"


def load_settings() -> dict:
    """Load settings from mcp_chatbot/settings.json. Returns {} if missing or malformed."""
    try:
        return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(data: dict) -> None:
    """Atomically write data to mcp_chatbot/settings.json."""
    tmp = _SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, _SETTINGS_PATH)

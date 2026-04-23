import json
import os
import tempfile
from pathlib import Path
import pytest


def test_load_settings_returns_empty_dict_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from mcp_chatbot.settings import load_settings
    assert load_settings() == {}


def test_load_settings_returns_empty_dict_when_malformed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mcp_chatbot").mkdir()
    (tmp_path / "mcp_chatbot" / "settings.json").write_text("not json")
    from mcp_chatbot.settings import load_settings
    assert load_settings() == {}


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mcp_chatbot").mkdir()
    from mcp_chatbot.settings import load_settings, save_settings
    save_settings({"file_root": "/some/path"})
    assert load_settings() == {"file_root": "/some/path"}


def test_save_is_atomic(tmp_path, monkeypatch):
    """Temp file must not exist after save."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mcp_chatbot").mkdir()
    from mcp_chatbot.settings import save_settings
    save_settings({"file_root": "/x"})
    tmp_file = tmp_path / "mcp_chatbot" / "settings.json.tmp"
    assert not tmp_file.exists()

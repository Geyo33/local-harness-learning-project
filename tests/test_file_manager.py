from pathlib import Path
import pytest
import subprocess
from unittest.mock import patch, MagicMock


@pytest.fixture()
def base(tmp_path):
    """A temp dir with some files and a .gitignore."""
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "secret.env").write_text("SECRET=abc", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.env\n# comment\n\nbuild/\n", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested", encoding="utf-8")
    return tmp_path


def _fm(base):
    from mcp_chatbot.tools.file_manager import FileManager
    return FileManager(base)


# ── init ──────────────────────────────────────────────────────────────────────

def test_raises_if_base_dir_missing():
    from mcp_chatbot.tools.file_manager import FileManager
    with pytest.raises(ValueError):
        FileManager(Path("/nonexistent/path/xyz"))


# ── list_files ────────────────────────────────────────────────────────────────

def test_list_files_root_returns_files(base):
    fm = _fm(base)
    result = fm.list_files("")
    assert "hello.txt" in result
    assert "subdir" in result


def test_list_files_filters_gitignore(base):
    fm = _fm(base)
    result = fm.list_files("")
    assert "secret.env" not in result


def test_list_files_subdir(base):
    fm = _fm(base)
    result = fm.list_files("subdir")
    assert "nested.txt" in result


def test_list_files_nonexistent_path_returns_error(base):
    fm = _fm(base)
    result = fm.list_files("no_such_dir")
    assert "Error" in result or "error" in result


# ── read_file ─────────────────────────────────────────────────────────────────

def test_read_file_returns_content(base):
    fm = _fm(base)
    result = fm.read_file("hello.txt")
    assert result.startswith("Successfully read from 'hello.txt'")
    assert "hello world" in result
    assert "1 |" in result


def test_read_file_no_wrapping_quotes(base):
    (base / "q.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.read_file("q.txt")
    # gutter kept, but no single quotes wrapping the line content
    assert "1 | alpha" in result
    assert "2 | beta" in result
    assert "'alpha'" not in result


def test_read_file_blocked_by_gitignore(base):
    fm = _fm(base)
    result = fm.read_file("secret.env")
    assert "Error" in result or "blocked" in result.lower()


def test_read_file_missing_returns_error(base):
    fm = _fm(base)
    result = fm.read_file("nope.txt")
    assert "Error" in result or "error" in result


def test_read_file_traversal_blocked(base):
    fm = _fm(base)
    result = fm.read_file("../../etc/passwd")
    assert "Error" in result or "blocked" in result.lower()


def test_sibling_dir_with_shared_prefix_is_blocked(tmp_path):
    """A sibling directory whose name starts with the base dir name must be blocked."""
    from mcp_chatbot.tools.file_manager import FileManager
    # Create base dir and a sibling that shares its name as a prefix
    base = tmp_path / "sandbox"
    sibling = tmp_path / "sandboxevil"
    base.mkdir()
    sibling.mkdir()
    (sibling / "secret.txt").write_text("evil", encoding="utf-8")
    fm = FileManager(base)
    # Attempting to escape via a relative traversal into the sibling must be blocked
    result = fm.read_file("../sandboxevil/secret.txt")
    assert "Error" in result or "blocked" in result.lower()


# ── build/ directory pattern ──────────────────────────────────────────────────

def test_list_files_filters_build_dir(base):
    """build/ pattern in .gitignore must hide the build directory from listing."""
    (base / "build").mkdir()
    (base / "build" / "output.js").write_text("compiled", encoding="utf-8")
    fm = _fm(base)
    result = fm.list_files("")
    assert "build" not in result


def test_read_file_inside_build_dir_blocked(base):
    """Files inside a build/ ignored directory must be blocked."""
    (base / "build").mkdir()
    (base / "build" / "output.js").write_text("compiled", encoding="utf-8")
    fm = _fm(base)
    result = fm.read_file("build/output.js")
    assert "Error" in result or "blocked" in result.lower()


def test_list_files_build_dir_itself_blocked(base):
    """list_files('build') on an ignored directory must return an error."""
    (base / "build").mkdir()
    (base / "build" / "output.js").write_text("compiled", encoding="utf-8")
    fm = _fm(base)
    result = fm.list_files("build")
    assert "Error" in result or "blocked" in result.lower()


# ── tool_entries & is_safe ────────────────────────────────────────────────────

def test_tool_entries_has_four_items(base):
    fm = _fm(base)
    assert len(fm.tool_entries) == 4


def test_list_files_is_safe(base):
    fm = _fm(base)
    by_name = {e["schema"]["function"]["name"]: e["safe"] for e in fm.tool_entries}
    assert by_name["list_files"] is True
    assert by_name["read_file"] is False
    assert by_name["bash"] is False
    assert by_name["edit_file"] is False


def test_is_safe_returns_correct_values(base):
    fm = _fm(base)
    assert fm.is_safe("list_files") is True
    assert fm.is_safe("read_file") is False
    assert fm.is_safe("bash") is False
    assert fm.is_safe("edit_file") is False


# ── execute dispatcher ────────────────────────────────────────────────────────

def test_execute_list_files(base):
    fm = _fm(base)
    result = fm.execute("list_files", {"path": ""})
    assert "hello.txt" in result


def test_execute_read_file(base):
    fm = _fm(base)
    result = fm.execute("read_file", {"path": "hello.txt"})
    assert "hello world" in result
    assert "1 |" in result


def test_execute_unknown_tool(base):
    fm = _fm(base)
    result = fm.execute("unknown_tool", {})
    assert "Error" in result or "unknown" in result.lower()


# ── edit_file — happy path ────────────────────────────────────────────────────

def test_edit_file_single_match(base):
    (base / "m.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "bbb", "BBB")
    assert "Error" not in result
    assert (base / "m.txt").read_text(encoding="utf-8") == "aaa\nBBB\nccc\n"


def test_edit_file_multiline_find(base):
    (base / "m.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "two\nthree", "X\nY\nZ")
    assert "Error" not in result
    assert (base / "m.txt").read_text(encoding="utf-8") == "one\nX\nY\nZ\nfour\n"


def test_edit_file_delete_via_empty_replace(base):
    (base / "m.txt").write_text("keep\ndrop_me\nkeep2\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "drop_me\n", "")
    assert "Error" not in result
    assert (base / "m.txt").read_text(encoding="utf-8") == "keep\nkeep2\n"


def test_edit_file_preserves_indentation(base):
    original = "def f():\n    return 1\n"
    (base / "m.py").write_text(original, encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.py", "    return 1", "    return 2")
    assert "Error" not in result
    assert (base / "m.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_edit_file_success_message_no_file_content(base):
    (base / "m.txt").write_text("unique_sentinel\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "bbb", "NEW")
    assert "Applied" in result
    assert "unique_sentinel" not in result


# ── edit_file — match guards ──────────────────────────────────────────────────

def test_edit_file_not_found_no_write(base):
    original = "aaa\nbbb\n"
    (base / "m.txt").write_text(original, encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "zzz", "QQQ")
    assert "Error" in result and "not found" in result.lower()
    assert (base / "m.txt").read_text(encoding="utf-8") == original


def test_edit_file_multi_match_no_write(base):
    original = "x = 1\nx = 1\nx = 1\n"
    (base / "m.txt").write_text(original, encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "x = 1", "x = 2")
    assert "Error" in result and "3 times" in result
    assert (base / "m.txt").read_text(encoding="utf-8") == original


def test_edit_file_safe_failure_on_absent_text(base):
    """Genuinely-absent text -> not found -> file untouched (safe failure)."""
    original = "def f():\n    return 1\n"
    (base / "m.py").write_text(original, encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.py", "    return 99", "    return 2")
    assert "Error" in result and "not found" in result.lower()
    assert (base / "m.py").read_text(encoding="utf-8") == original


def test_edit_file_empty_find_returns_error(base):
    (base / "m.txt").write_text("aaa\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "", "X")
    assert "Error" in result and "find" in result.lower()


def test_edit_file_whitespace_only_find_returns_error(base):
    (base / "m.txt").write_text("a   b\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "   ", "X")
    assert "Error" in result and "find" in result.lower()
    assert (base / "m.txt").read_text(encoding="utf-8") == "a   b\n"


def test_edit_file_replace_identical_to_find_returns_error(base):
    (base / "m.txt").write_text("aaa\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "bbb", "bbb")
    assert "Error" in result


# ── edit_file — gutter strip fallback ─────────────────────────────────────────

def test_edit_file_strips_line_number_gutter(base):
    (base / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.py", "2 |     return 1", "    return 2")
    assert "Error" not in result
    assert (base / "m.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_edit_file_strips_gutter_with_old_quotes(base):
    (base / "m.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "1 | 'alpha'", "ALPHA")
    assert "Error" not in result
    assert (base / "m.txt").read_text(encoding="utf-8") == "ALPHA\nbeta\n"


# ── edit_file — EOL handling ──────────────────────────────────────────────────

def test_edit_file_preserves_crlf(base):
    p = base / "m.txt"
    p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "two", "TWO")
    assert "Error" not in result
    assert p.read_bytes() == b"one\r\nTWO\r\nthree\r\n"


def test_edit_file_crlf_multiline_find(base):
    """Multi-line find joined with \\n still matches a CRLF file."""
    p = base / "m.txt"
    p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    fm = _fm(base)
    result = fm.edit_file("m.txt", "two\nthree", "X")
    assert "Error" not in result
    assert p.read_bytes() == b"one\r\nX\r\n"


def test_edit_file_preserves_no_trailing_newline(base):
    p = base / "m.txt"
    p.write_bytes(b"one\ntwo\nthree")  # no trailing newline
    fm = _fm(base)
    fm.edit_file("m.txt", "three", "THREE")
    assert p.read_bytes() == b"one\ntwo\nTHREE"


# ── edit_file — path & access guards ──────────────────────────────────────────

def test_edit_file_missing_path_returns_error(base):
    fm = _fm(base)
    result = fm.edit_file("", "a", "b")
    assert "Error" in result and "path" in result.lower()


def test_edit_file_missing_file_returns_error(base):
    fm = _fm(base)
    result = fm.edit_file("no_such_file.txt", "a", "b")
    assert "Error" in result


def test_edit_file_blocked_by_gitignore(base):
    fm = _fm(base)
    result = fm.edit_file("secret.env", "SECRET=abc", "SECRET=xyz")
    assert "Error" in result or "blocked" in result.lower()


def test_edit_file_traversal_blocked(base):
    fm = _fm(base)
    result = fm.edit_file("../../evil.txt", "a", "b")
    assert "Error" in result or "blocked" in result.lower()


# ── edit_file — execute dispatcher ────────────────────────────────────────────

def test_execute_edit_file(base):
    (base / "m.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.execute("edit_file", {"path": "m.txt", "find": "bbb", "replace": "REPLACED"})
    assert "Error" not in result
    assert "Applied" in result
    assert (base / "m.txt").read_text(encoding="utf-8") == "aaa\nREPLACED\nccc\n"


# ── bash — unit tests (no WSL required) ──────────────────────────────────────

def _completed(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


def test_bash_empty_commands_no_subprocess(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run") as mock_run:
        result = fm.bash([])
        mock_run.assert_not_called()
    assert "Error" in result


def test_bash_uses_stdin_flag(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run", return_value=_completed()) as mock_run:
        fm.bash(["echo hello"])
    assert mock_run.call_args.args[0] == ["wsl.exe", "bash", "-s"]


def test_bash_input_is_bytes(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run", return_value=_completed()) as mock_run:
        fm.bash(["echo hello"])
    assert isinstance(mock_run.call_args.kwargs["input"], bytes)


def test_bash_crlf_normalized_to_lf(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run", return_value=_completed()) as mock_run:
        fm.bash(["line1\r\nline2\r\n"])
    assert mock_run.call_args.kwargs["input"] == b"line1\nline2\n"


def test_bash_lf_only_unchanged(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run", return_value=_completed()) as mock_run:
        fm.bash(["line1\nline2\n"])
    assert mock_run.call_args.kwargs["input"] == b"line1\nline2\n"


def test_bash_lone_cr_normalized_to_lf(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run", return_value=_completed()) as mock_run:
        fm.bash(["line1\rline2\r"])
    assert mock_run.call_args.kwargs["input"] == b"line1\nline2\n"


def test_bash_output_decoded_from_bytes(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run", return_value=_completed(stdout=b"hello world")):
        result = fm.bash(["echo hello"])
    assert "hello world" in result


def test_bash_stdout_and_stderr_combined(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run",
               return_value=_completed(stdout=b"out", stderr=b"err")):
        result = fm.bash(["cmd"])
    assert "out" in result and "err" in result


def test_bash_multiple_commands_run_separately(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run", return_value=_completed()) as mock_run:
        fm.bash(["cmd1", "cmd2"])
    assert mock_run.call_count == 2


def test_bash_nonzero_exit_no_output_reports_exit_code(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run", return_value=_completed(returncode=1)):
        result = fm.bash(["exit 1"])
    assert "Success" not in result
    assert "Exit code 1" in result


def test_bash_zero_exit_no_output_still_reports_success(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run", return_value=_completed(returncode=0)):
        result = fm.bash(["true"])
    assert "Success" in result


def test_bash_timeout_returns_error_string(base):
    fm = _fm(base)
    with patch("mcp_chatbot.tools.file_manager.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd=["wsl.exe", "bash", "-s"], timeout=30)):
        result = fm.bash(["sleep 999"])
    assert "Error" in result
    assert "timed out" in result.lower()


# ── bash — integration tests (require WSL) ────────────────────────────────────

import shutil as _shutil

_wsl_available = pytest.mark.skipif(
    _shutil.which("wsl.exe") is None,
    reason="WSL not available"
)


@_wsl_available
def test_bash_heredoc_backtick_literal(base):
    """Backticks inside quoted heredoc must appear literally in file."""
    fm = _fm(base)
    cmd = "cat > out.md << 'EOF'\nUse `code` here\nEOF"
    result = fm.bash([cmd])
    assert "Error" not in result
    content = (base / "out.md").read_text(encoding="utf-8")
    assert "`code`" in content


@_wsl_available
def test_bash_heredoc_no_var_expansion(base):
    """$VAR inside single-quoted heredoc must not expand."""
    fm = _fm(base)
    cmd = "cat > out.txt << 'EOF'\n$HOME should not expand\nEOF"
    result = fm.bash([cmd])
    assert "Error" not in result
    content = (base / "out.txt").read_text(encoding="utf-8")
    assert "$HOME" in content


@_wsl_available
def test_bash_multiline_output(base):
    fm = _fm(base)
    result = fm.bash(["printf 'line1\\nline2\\nline3'"])
    assert "line1" in result
    assert "line2" in result
    assert "line3" in result


@_wsl_available
def test_bash_stderr_captured(base):
    fm = _fm(base)
    result = fm.bash(["ls /nonexistent_xyz_path_abc 2>&1"])
    assert "No such file" in result or "cannot access" in result or "ls:" in result

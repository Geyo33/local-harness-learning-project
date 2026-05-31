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
    assert by_name["replace_lines"] is False


def test_is_safe_returns_correct_values(base):
    fm = _fm(base)
    assert fm.is_safe("list_files") is True
    assert fm.is_safe("read_file") is False
    assert fm.is_safe("bash") is False
    assert fm.is_safe("replace_lines") is False


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


# ── replace_lines — happy path ────────────────────────────────────────────────

def test_replace_lines_middle_range(base):
    (base / "multi.txt").write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 2, "end_line": 3, "expected": ["line2", "line3"], "new_lines": ["NEW2", "NEW3"]}])
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "line1\nNEW2\nNEW3\nline4\n"


def test_replace_lines_single_line(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 2, "end_line": 2, "expected": ["bbb"], "new_lines": ["XXX"]}])
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "aaa\nXXX\nccc\n"


def test_replace_lines_first_line(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 1, "end_line": 1, "expected": ["aaa"], "new_lines": ["NEW_FIRST"]}])
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "NEW_FIRST\nbbb\nccc\n"


def test_replace_lines_last_line(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 3, "end_line": 3, "expected": ["ccc"], "new_lines": ["NEW_LAST"]}])
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "aaa\nbbb\nNEW_LAST\n"


def test_replace_lines_entire_file(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 1, "end_line": 3, "expected": ["aaa", "bbb", "ccc"], "new_lines": ["ONLY"]}])
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "ONLY\n"


def test_replace_lines_delete_lines(base):
    (base / "multi.txt").write_text("keep1\ndelete_me\nkeep2\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 2, "end_line": 2, "expected": ["delete_me"], "new_lines": []}])
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "keep1\nkeep2\n"


def test_replace_lines_end_clamped_silently(base):
    (base / "multi.txt").write_text("aaa\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 1, "end_line": 9999, "expected": ["aaa", "bbb"], "new_lines": ["ONLY"]}])
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "ONLY\n"


def test_replace_lines_success_message(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 2, "end_line": 2, "expected": ["bbb"], "new_lines": ["NEW"]}])
    assert "Applied" in result and "1 edit" in result


# ── replace_lines — error cases ───────────────────────────────────────────────

def test_replace_lines_start_zero_returns_error(base):
    (base / "multi.txt").write_text("aaa\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    assert "Error" in fm.replace_lines("multi.txt", [{"start_line": 0, "end_line": 1, "new_lines": ["x"]}])


def test_replace_lines_start_negative_returns_error(base):
    (base / "multi.txt").write_text("aaa\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    assert "Error" in fm.replace_lines("multi.txt", [{"start_line": -1, "end_line": 1, "new_lines": ["x"]}])


def test_replace_lines_start_greater_than_end_returns_error(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 3, "end_line": 1, "new_lines": ["x"]}])
    assert "Error" in result


def test_replace_lines_start_beyond_file_returns_error(base):
    (base / "multi.txt").write_text("aaa\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 10, "end_line": 10, "new_lines": ["x"]}])
    assert "Error" in result and "beyond" in result.lower()


def test_replace_lines_missing_file_returns_error(base):
    fm = _fm(base)
    assert "Error" in fm.replace_lines("no_such_file.txt", [{"start_line": 1, "end_line": 1, "new_lines": ["x"]}])


def test_replace_lines_blocked_by_gitignore(base):
    fm = _fm(base)
    result = fm.replace_lines("secret.env", [{"start_line": 1, "end_line": 1, "new_lines": ["x"]}])
    assert "Error" in result or "blocked" in result.lower()


def test_replace_lines_traversal_blocked(base):
    fm = _fm(base)
    result = fm.replace_lines("../../evil.txt", [{"start_line": 1, "end_line": 1, "new_lines": ["x"]}])
    assert "Error" in result or "blocked" in result.lower()


# ── replace_lines — execute dispatcher ───────────────────────────────────────

def test_execute_replace_lines(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.execute("replace_lines", {
        "path": "multi.txt",
        "edits": [{"start_line": 2, "end_line": 2, "expected": ["bbb"], "new_lines": ["REPLACED"]}],
    })
    assert "Error" not in result
    assert "Applied" in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "aaa\nREPLACED\nccc\n"


# ── replace_lines — batch ─────────────────────────────────────────────────────

def test_replace_lines_batch_two_edits_non_overlapping(base):
    """Two edits on separate regions both applied correctly."""
    (base / "multi.txt").write_text("aaa\nbbb\nccc\nddd\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [
        {"start_line": 1, "end_line": 1, "expected": ["aaa"], "new_lines": ["AAA"]},
        {"start_line": 3, "end_line": 3, "expected": ["ccc"], "new_lines": ["CCC"]},
    ])
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "AAA\nbbb\nCCC\nddd\n"


def test_replace_lines_batch_order_independent(base):
    """Passing edits top-to-bottom produces the same result as bottom-to-top."""
    (base / "a.txt").write_text("1\n2\n3\n4\n5\n", encoding="utf-8")
    (base / "b.txt").write_text("1\n2\n3\n4\n5\n", encoding="utf-8")
    fm = _fm(base)
    fm.replace_lines("a.txt", [
        {"start_line": 2, "end_line": 2, "expected": ["2"], "new_lines": ["X"]},
        {"start_line": 4, "end_line": 4, "expected": ["4"], "new_lines": ["Y"]},
    ])
    fm.replace_lines("b.txt", [
        {"start_line": 4, "end_line": 4, "expected": ["4"], "new_lines": ["Y"]},
        {"start_line": 2, "end_line": 2, "expected": ["2"], "new_lines": ["X"]},
    ])
    expected = "1\nX\n3\nY\n5\n"
    assert (base / "a.txt").read_text(encoding="utf-8") == expected
    assert (base / "b.txt").read_text(encoding="utf-8") == expected


def test_replace_lines_batch_overlap_returns_error_and_no_write(base):
    """Overlapping edits return an error and leave the file unchanged."""
    original = "aaa\nbbb\nccc\nddd\n"
    (base / "multi.txt").write_text(original, encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [
        {"start_line": 2, "end_line": 4, "expected": ["bbb", "ccc", "ddd"], "new_lines": ["X"]},
        {"start_line": 3, "end_line": 5, "expected": ["ccc", "ddd"], "new_lines": ["Y"]},
    ])
    assert "Error" in result and "overlap" in result.lower()
    assert (base / "multi.txt").read_text(encoding="utf-8") == original


def test_replace_lines_batch_adjacent_not_overlapping(base):
    """Adjacent ranges [1,2] and [3,4] don't trigger overlap error."""
    (base / "multi.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [
        {"start_line": 1, "end_line": 2, "expected": ["a", "b"], "new_lines": ["A"]},
        {"start_line": 3, "end_line": 4, "expected": ["c", "d"], "new_lines": ["B"]},
    ])
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "A\nB\n"


def test_replace_lines_batch_empty_list_returns_error(base):
    (base / "multi.txt").write_text("aaa\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [])
    assert "Error" in result


def test_replace_lines_batch_invalid_one_edit_no_write(base):
    """If any edit is invalid, no write occurs (all-or-nothing)."""
    original = "aaa\nbbb\nccc\n"
    (base / "multi.txt").write_text(original, encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [
        {"start_line": 1, "end_line": 1, "expected": ["aaa"], "new_lines": ["OK"]},
        {"start_line": 0, "end_line": 1, "new_lines": ["BAD"]},
    ])
    assert "Error" in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == original


def test_replace_lines_batch_success_message_n_edits(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [
        {"start_line": 1, "end_line": 1, "expected": ["aaa"], "new_lines": ["AAA"]},
        {"start_line": 3, "end_line": 3, "expected": ["ccc"], "new_lines": ["CCC"]},
    ])
    assert "Applied 2 edit(s)" in result
    assert "New line count:" in result


def test_replace_lines_success_no_file_content_in_result(base):
    """Success response must not contain file content — no reinject."""
    (base / "multi.txt").write_text("unique_sentinel_value\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", [{"start_line": 2, "end_line": 2, "expected": ["bbb"], "new_lines": ["NEW"]}])
    assert "unique_sentinel_value" not in result


# ── replace_lines — new_lines interface ──────────────────────────────────────

def test_replace_lines_multiline_replacement(base):
    """Replace one line with three lines using new_lines array."""
    (base / "f.txt").write_text("header\nold\nfooter\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("f.txt", [{"start_line": 2, "end_line": 2, "expected": ["old"], "new_lines": ["new_a", "new_b", "new_c"]}])
    assert "Error" not in result
    assert (base / "f.txt").read_text(encoding="utf-8") == "header\nnew_a\nnew_b\nnew_c\nfooter\n"


def test_replace_lines_blank_line_via_empty_element(base):
    """An empty string element in new_lines produces a blank line."""
    (base / "f.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("f.txt", [{"start_line": 2, "end_line": 2, "expected": ["bbb"], "new_lines": ["content", ""]}])
    assert "Error" not in result
    assert (base / "f.txt").read_text(encoding="utf-8") == "aaa\ncontent\n\nccc\n"


def test_replace_lines_line_count_accurate_for_multiline(base):
    """Reported line count must reflect actual lines, not list-element count."""
    (base / "f.txt").write_text("x\ny\nz\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("f.txt", [{"start_line": 2, "end_line": 2, "expected": ["y"], "new_lines": ["a", "b", "c"]}])
    # Replaced 1 line with 3 → 5 lines total (x, a, b, c, z)
    assert "New line count: 5" in result


# ── replace_lines — path & expected guards (post-hardening) ───────────────────

def test_replace_lines_missing_path_returns_actionable_error(base):
    fm = _fm(base)
    result = fm.replace_lines("", [{"start_line": 1, "end_line": 1, "expected": ["x"], "new_lines": ["y"]}])
    assert "Error" in result and "path" in result.lower()


def test_replace_lines_missing_expected_returns_error(base):
    (base / "m.txt").write_text("aaa\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("m.txt", [{"start_line": 1, "end_line": 1, "new_lines": ["X"]}])
    assert "Error" in result and "expected" in result.lower()


def test_replace_lines_expected_mismatch_no_write(base):
    """Wrong indentation in expected -> rejected, file untouched."""
    original = "def f():\n    return 1\n"
    (base / "m.py").write_text(original, encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("m.py", [{"start_line": 2, "end_line": 2, "expected": ["return 1"], "new_lines": ["    return 2"]}])
    assert "Error" in result and "does not match" in result
    assert (base / "m.py").read_text(encoding="utf-8") == original


def test_replace_lines_expected_match_applies(base):
    (base / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("m.py", [{"start_line": 2, "end_line": 2, "expected": ["    return 1"], "new_lines": ["    return 2"]}])
    assert "Error" not in result
    assert (base / "m.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"


def test_replace_lines_expected_not_list_returns_error(base):
    (base / "m.txt").write_text("x\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("m.txt", [{"start_line": 1, "end_line": 1, "expected": "x", "new_lines": ["Y"]}])
    assert "Error" in result and "array" in result.lower()


def test_replace_lines_preserves_no_trailing_newline(base):
    p = base / "m.txt"
    p.write_bytes(b"one\ntwo\nthree")  # no trailing newline
    fm = _fm(base)
    fm.replace_lines("m.txt", [{"start_line": 3, "end_line": 3, "expected": ["three"], "new_lines": ["THREE"]}])
    assert p.read_bytes() == b"one\ntwo\nTHREE"


def test_replace_lines_preserves_trailing_newline(base):
    p = base / "m.txt"
    p.write_bytes(b"one\ntwo\nthree\n")
    fm = _fm(base)
    fm.replace_lines("m.txt", [{"start_line": 3, "end_line": 3, "expected": ["three"], "new_lines": ["THREE"]}])
    assert p.read_bytes() == b"one\ntwo\nTHREE\n"


def test_replace_lines_preserves_crlf_eol(base):
    """New lines adopt the file's CRLF convention; untouched lines unchanged."""
    p = base / "m.txt"
    p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    fm = _fm(base)
    fm.replace_lines("m.txt", [{"start_line": 1, "end_line": 1, "expected": ["one"], "new_lines": ["ONE", "EXTRA"]}])
    assert p.read_bytes() == b"ONE\r\nEXTRA\r\ntwo\r\nthree\r\n"


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

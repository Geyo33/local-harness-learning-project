from pathlib import Path
import pytest


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


# ── edit_file — create ────────────────────────────────────────────────────────

def test_edit_file_creates_new_file(base):
    fm = _fm(base)
    result = fm.edit_file("new.txt", "", "content here")
    assert "Error" not in result
    assert (base / "new.txt").read_text() == "content here"


def test_edit_file_create_errors_if_exists(base):
    fm = _fm(base)
    result = fm.edit_file("hello.txt", "", "overwrite")
    assert "Error" in result or "exists" in result.lower()


def test_edit_file_create_makes_parent_dirs(base):
    fm = _fm(base)
    result = fm.edit_file("deep/dir/file.txt", "", "hi")
    assert "Error" not in result
    assert (base / "deep" / "dir" / "file.txt").read_text() == "hi"


# ── edit_file — replace ───────────────────────────────────────────────────────

def test_edit_file_replaces_first_occurrence(base):
    fm = _fm(base)
    result = fm.edit_file("hello.txt", "hello", "goodbye")
    assert "Error" not in result
    assert (base / "hello.txt").read_text() == "goodbye world"


def test_edit_file_old_str_not_found_returns_error(base):
    fm = _fm(base)
    result = fm.edit_file("hello.txt", "NOTHERE", "x")
    assert "Error" in result or "not found" in result.lower()


def test_edit_file_blocked_by_gitignore(base):
    fm = _fm(base)
    result = fm.edit_file("secret.env", "", "bad")
    assert "Error" in result or "blocked" in result.lower()


def test_edit_file_traversal_blocked(base):
    fm = _fm(base)
    result = fm.edit_file("../../evil.txt", "", "evil")
    assert "Error" in result or "blocked" in result.lower()


# ── tool_entries & is_safe ────────────────────────────────────────────────────

def test_tool_entries_has_five_items(base):
    fm = _fm(base)
    assert len(fm.tool_entries) == 5


def test_list_files_is_safe(base):
    fm = _fm(base)
    by_name = {e["schema"]["function"]["name"]: e["safe"] for e in fm.tool_entries}
    assert by_name["list_files"] is True
    assert by_name["read_file"] is False
    assert by_name["edit_file"] is False
    assert by_name["bash"] is False
    assert by_name["replace_lines"] is False


def test_is_safe_returns_correct_values(base):
    fm = _fm(base)
    assert fm.is_safe("list_files") is True
    assert fm.is_safe("read_file") is False
    assert fm.is_safe("edit_file") is False
    assert fm.is_safe("bash") is False
    assert fm.is_safe("replace_lines") is False


def test_tool_names_set(base):
    fm = _fm(base)
    assert fm.tool_names == {"list_files", "read_file", "edit_file", "bash", "replace_lines"}


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


def test_execute_edit_file(base):
    fm = _fm(base)
    result = fm.execute("edit_file", {"path": "new.txt", "old_str": "", "new_str": "hi"})
    assert "Error" not in result
    assert (base / "new.txt").exists()


def test_execute_unknown_tool(base):
    fm = _fm(base)
    result = fm.execute("unknown_tool", {})
    assert "Error" in result or "unknown" in result.lower()


# ── replace_lines — happy path ────────────────────────────────────────────────

def test_replace_lines_middle_range(base):
    (base / "multi.txt").write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", 2, 3, "NEW2\nNEW3")
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "line1\nNEW2\nNEW3\nline4\n"


def test_replace_lines_single_line(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", 2, 2, "XXX")
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "aaa\nXXX\nccc\n"


def test_replace_lines_first_line(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", 1, 1, "NEW_FIRST")
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "NEW_FIRST\nbbb\nccc\n"


def test_replace_lines_last_line(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", 3, 3, "NEW_LAST")
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "aaa\nbbb\nNEW_LAST\n"


def test_replace_lines_entire_file(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", 1, 3, "ONLY")
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "ONLY"


def test_replace_lines_delete_lines(base):
    (base / "multi.txt").write_text("keep1\ndelete_me\nkeep2\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", 2, 2, "")
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "keep1\nkeep2\n"


def test_replace_lines_end_clamped_silently(base):
    (base / "multi.txt").write_text("aaa\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", 1, 9999, "ONLY")
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "ONLY"


def test_replace_lines_success_message(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", 2, 2, "NEW")
    assert "2" in result and ("Replaced" in result or "replaced" in result)


# ── replace_lines — error cases ───────────────────────────────────────────────

def test_replace_lines_start_zero_returns_error(base):
    (base / "multi.txt").write_text("aaa\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    assert "Error" in fm.replace_lines("multi.txt", 0, 1, "x")


def test_replace_lines_start_negative_returns_error(base):
    (base / "multi.txt").write_text("aaa\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    assert "Error" in fm.replace_lines("multi.txt", -1, 1, "x")


def test_replace_lines_start_greater_than_end_returns_error(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", 3, 1, "x")
    assert "Error" in result


def test_replace_lines_start_beyond_file_returns_error(base):
    (base / "multi.txt").write_text("aaa\nbbb\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.replace_lines("multi.txt", 10, 10, "x")
    assert "Error" in result and "beyond" in result.lower()


def test_replace_lines_missing_file_returns_error(base):
    fm = _fm(base)
    assert "Error" in fm.replace_lines("no_such_file.txt", 1, 1, "x")


def test_replace_lines_blocked_by_gitignore(base):
    fm = _fm(base)
    result = fm.replace_lines("secret.env", 1, 1, "x")
    assert "Error" in result or "blocked" in result.lower()


def test_replace_lines_traversal_blocked(base):
    fm = _fm(base)
    result = fm.replace_lines("../../evil.txt", 1, 1, "x")
    assert "Error" in result or "blocked" in result.lower()


# ── replace_lines — execute dispatcher ───────────────────────────────────────

def test_execute_replace_lines(base):
    (base / "multi.txt").write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    fm = _fm(base)
    result = fm.execute("replace_lines", {
        "path": "multi.txt",
        "start_line": 2,
        "end_line": 2,
        "new_str": "REPLACED"
    })
    assert "Error" not in result
    assert (base / "multi.txt").read_text(encoding="utf-8") == "aaa\nREPLACED\nccc\n"

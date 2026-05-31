from __future__ import annotations

import fnmatch
from pathlib import Path
import subprocess


class FileManager:
    """
    Scoped filesystem access for LLM tool calls.

    Enforces:
    - All paths resolved under base_dir (blocks path traversal).
    - .gitignore patterns deny read and write access.
    """

    _TOOL_NAMES = {"list_files", "read_file", "bash", "replace_lines"}

    def __init__(self, base_dir: Path) -> None:
        if not base_dir.exists():
            raise ValueError(f"Base directory does not exist: {base_dir}")
        self._base = base_dir.resolve()
        self._ignored: list[str] = self._load_gitignore()

    # ── .gitignore ─────────────────────────────────────────────────────────

    def _load_gitignore(self) -> list[str]:
        gitignore = self._base / ".gitignore"
        if not gitignore.exists():
            return []
        patterns = []
        for line in gitignore.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
        patterns.append(".git")
        return patterns

    def _is_ignored(self, rel: str) -> bool:
        name = Path(rel).name
        # Normalize to forward slashes for cross-platform fnmatch
        rel_fwd = rel.replace("\\", "/")
        for pattern in self._ignored:
            if fnmatch.fnmatch(name, pattern):
                return True
            if pattern.endswith("/"):
                # Directory-style pattern: match the dir name or any path inside it
                dir_name = pattern.rstrip("/")
                if fnmatch.fnmatch(name, dir_name) or dir_name in Path(rel_fwd).parts:
                    return True
            elif fnmatch.fnmatch(rel_fwd, pattern):
                return True
        return False

    # ── path safety ────────────────────────────────────────────────────────

    def _safe_resolve(self, path: str) -> Path | None:
        """Return resolved absolute path if under base_dir, else None."""
        try:
            resolved = (self._base / path).resolve()
        except Exception:
            return None
        if not resolved.is_relative_to(self._base):
            return None
        return resolved

    def _check(self, path: str) -> tuple[Path | None, str | None]:
        """Return (resolved_path, error_string). One of the two is always None."""
        resolved = self._safe_resolve(path)
        if resolved is None:
            return None, f"Error: path '{path}' is outside the allowed directory."
        rel = str(resolved.relative_to(self._base))
        if self._is_ignored(rel):
            return None, f"Error: path '{path}' is blocked by .gitignore."
        return resolved, None

    def _walk(self, directory: Path) -> list[Path]:
        entries = []
        for item in sorted(directory.iterdir()):
            rel = str(item.relative_to(self._base)).replace("\\", "/")
            if self._is_ignored(rel):
                continue
            entries.append(item)
            if item.is_dir():
                entries.extend(self._walk(item))
        return entries

    # ── tool operations ────────────────────────────────────────────────────

    def list_files(self, path: str = "") -> str:
        target = self._safe_resolve(path) if path else self._base
        if target is None:
            return f"Error: path '{path}' is outside the allowed directory."
        if not target.exists():
            return f"Error: path '{path}' does not exist."
        # Guard: don't list a directory that is itself .gitignore-blocked
        if target != self._base:
            rel = str(target.relative_to(self._base)).replace("\\", "/")
            if self._is_ignored(rel):
                return f"Error: path '{path}' is blocked by .gitignore."
        entries = []
        for item in self._walk(target):
            rel = str(item.relative_to(self._base)).replace("\\", "/")
            entries.append(rel + ("/" if item.is_dir() else ""))
        return "\n\nFiles found:\n\n"+"\n".join(entries) if entries else "(empty)"

    def read_file(self, path: str) -> str:
        resolved, err = self._check(path)
        if err:
            return err
        if not resolved.exists():
            return f"Error: '{path}' does not exist."
        if not resolved.is_file():
            return f"Error: '{path}' is not a file."
        try:
            content = resolved.read_text(encoding="utf-8")
            lines = content.splitlines(keepends=False)
            width = len(str(len(lines)))
            numbered = "".join(f"{i + 1:>{width}} | '{line}'\n" for i, line in enumerate(lines))
            plural = "" if len(lines) == 1 else "s"
            return f"Successfully read from '{path}' ({len(lines)} line{plural})\nContent:\n{numbered}"
        except Exception as e:
            return f"Error reading '{path}': {e}"

    def replace_lines(self, path: str, edits: list[dict]) -> str:
        if not path or not path.strip():
            return (
                "Error: no 'path' provided. replace_lines requires the relative file "
                "path as the first argument. Retry with path set."
            )
        resolved, err = self._check(path)
        if err:
            return err
        if not resolved.exists():
            return f"Error: '{path}' does not exist."
        if not resolved.is_file():
            return f"Error: '{path}' is not a file."
        if not edits:
            return "Error: edits list is empty."
        try:
            # newline="" disables newline translation so EOLs round-trip unchanged
            with open(resolved, "r", encoding="utf-8", newline="") as f:
                content = f.read()
        except Exception as e:
            return f"Error reading '{path}': {e}"
        lines = content.splitlines(keepends=True)
        total = len(lines)

        # Validate all edits before touching the file (all-or-nothing)
        for i, edit in enumerate(edits):
            start = edit.get("start_line", 0)
            end = edit.get("end_line", 0)
            if start < 1:
                return f"Error: edit {i} start_line must be >= 1."
            if end < start:
                return f"Error: edit {i} end_line ({end}) must be >= start_line ({start})."
            if start > total:
                return (
                    f"Error: edit {i} start_line ({start}) is beyond the file's "
                    f"line count ({total})."
                )
            expected = edit.get("expected")
            if expected is None:
                return (
                    f"Error: edit {i} missing required 'expected' "
                    "(the exact current lines being replaced)."
                )
            if not isinstance(expected, list):
                return (
                    f"Error: edit {i} 'expected' must be an array of strings "
                    "(one element per line), not a single string."
                )
            end_clamped = min(end, total)
            actual = [lines[j].rstrip("\r\n") for j in range(start - 1, end_clamped)]
            if actual != expected:
                return (
                    f"Error: edit {i} 'expected' does not match file lines "
                    f"{start}-{end_clamped}.\n"
                    "File has:\n"
                    + "\n".join(f"{start + k} | {a!r}" for k, a in enumerate(actual))
                    + "\nYou sent:\n"
                    + "\n".join(f"{e!r}" for e in expected)
                    + "\nRe-read the file and retry with the exact lines "
                    "(including indentation)."
                )

        # Sort descending by start_line; clamp end_line to file length
        sorted_edits = sorted(
            enumerate(edits),
            key=lambda x: x[1]["start_line"],
            reverse=True,
        )
        clamped = [
            (orig_i, e["start_line"], min(e["end_line"], total), e.get("new_lines", []))
            for orig_i, e in sorted_edits
        ]

        # Overlap check: sorted descending, so clamped[i+1] sits above clamped[i]
        for i in range(len(clamped) - 1):
            curr_orig_i, curr_start, curr_end, _ = clamped[i]
            next_orig_i, next_start, next_end, _ = clamped[i + 1]
            if next_end >= curr_start:
                return (
                    f"Error: edits {next_orig_i} and {curr_orig_i} overlap "
                    f"(lines {next_start}-{next_end} and {curr_start}-{curr_end})."
                )

        # Detect the file's EOL so new lines match its convention (avoid mixed EOLs)
        eol = "\r\n" if "\r\n" in content else "\r" if "\r" in content else "\n"
        had_trailing_nl = content.endswith(("\n", "\r"))

        # Apply bottom-up (sorted descending = bottom of file first)
        for _, start, end, new_lines in clamped:
            before = lines[: start - 1]
            after = lines[end:]
            if not new_lines:
                new_segment: list[str] = []
            else:
                joined = eol.join(new_lines)
                if after:
                    new_segment = [joined + eol]
                else:
                    # segment now ends the file: match the file's original EOF state
                    new_segment = [joined + (eol if had_trailing_nl else "")]
            lines = before + new_segment + after

        try:
            with open(resolved, "w", encoding="utf-8", newline="") as f:
                f.write("".join(lines))
        except Exception as e:
            return f"Error writing '{path}': {e}"

        n = len(edits)
        final_content = "".join(lines)
        return f"Applied {n} edit(s) to '{path}'. New line count: {len(final_content.splitlines())}."

    def bash(self, commands: list[str]) -> str:
        if not commands:
            return "Error: no commands provided."
        results = []
        for cmd in commands:
            try:
                r = subprocess.run(
                    ["wsl.exe", "bash", "-s"],
                    input=cmd.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"),
                    capture_output=True,
                    timeout=30,
                    cwd=self._base,
                )
                output = (r.stdout + r.stderr).decode("utf-8", errors="replace").strip()
                if not output:
                    output = "Success" if r.returncode == 0 else f"Exit code {r.returncode}"
            except subprocess.TimeoutExpired:
                output = "Error: timed out after 30 seconds."
            results.append(f"$ {cmd}\n{output}")
        return "\n\n".join(results)

    def execute(self, tool_name: str, arguments: dict) -> str:
        try:
            if tool_name == "list_files":
                return self.list_files(arguments.get("path", ""))
            if tool_name == "read_file":
                return self.read_file(arguments.get("path", ""))
            if tool_name == "replace_lines":
                return self.replace_lines(
                    arguments.get("path", ""),
                    arguments.get("edits", []),
                )
            if tool_name == "bash":
                return self.bash(arguments.get("commands", []))
            return f"Error: unknown file tool '{tool_name}'."
        except Exception as e:
            return f"Error: {e}"


    # ── schema & metadata ──────────────────────────────────────────────────

    @property
    def tool_names(self) -> set[str]:
        return self._TOOL_NAMES

    def is_safe(self, tool_name: str) -> bool:
        return tool_name in {"list_files"}

    @property
    def tool_entries(self) -> list[dict]:
        """List of {schema, safe} dicts. schema is OpenAI-compatible; safe is not sent to LLM."""
        return [
            {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "description": (
                            f"List files, directories, subdirectories and their files under the configured base directory ({self._base}). "
                            "Pass an empty string or a relative subdirectory path."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Relative subdirectory to list. Empty string lists the root.",
                                }
                            },
                            "required": [],
                        },
                    },
                },
                "safe": True,
            },
            {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read the UTF-8 contents of a file at the given relative path.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Relative path to the file to read.",
                                }
                            },
                            "required": ["path"],
                        },
                    },
                },
                "safe": False,
            },
            {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": (
                            f"""Run one or more bash commands sequentially via WSL, starting in the base directory ({self._base}). 
Use standard bash/Linux syntax (e.g. `ls`, `mkdir`, `git`, `python3`, `cat`). 
Results are returned labeled. 

⚠️ State Warning: Each command in the list is executed in an independent shell session starting from the base directory. 
To perform sequential operations like changing directories (cd) followed by another action, 
chain them into a single string using '&&' or ';'. 
Variables and state do not persist across separate commands—use command substitution $() or pipe (|) within a single command to share data.

⚠️ Quoting Warning: Always wrap file and directory paths in double quotes when they contain spaces
(e.g. `cat "./a2a agent/src/main.py"` not `cat ./a2a agent/src/main.py`).
Failure to quote will cause the shell to split the path into separate arguments.

⚠️ Heredoc Warning: When writing file content that contains backticks or $ (e.g. Markdown code blocks, Python/shell source), always use a single-quoted heredoc delimiter to prevent bash from expanding command substitutions and variables:
  CORRECT:   cat > file.md <<'EOF'
  INCORRECT: cat > file.md <<EOF
With an unquoted EOF, backtick expressions inside the body are executed as commands, corrupting the output.

✓ Best Practices:
- Chain operations with && to ensure earlier commands succeed before running later ones.
- Use cat to verify file contents immediately after creating them.
- Use python3 explicitly (not just python).
- Check directory structure with pwd and ls before file operations if uncertain.
- Use test -f "file.txt" to verify file existence before operating on it.
- Redirect stderr with 2>&1 to capture error messages: command 2>&1"""
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "commands": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of bash commands to run sequentially.",
                                }
                            },
                            "required": ["commands"],
                        },
                    },
                },
                "safe": False,
            },
            {
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "replace_lines",
                        "description": (
                            "Replace one or more line ranges in an existing file. "
                            "ALWAYS call read_file first to get current line numbers and content. "
                            "Specify start_line through end_line (1-indexed, inclusive). "
                            "For each edit you MUST echo the exact current lines in 'expected' — "
                            "copy them verbatim from read_file (including all leading indentation); "
                            "the tool rejects the edit if 'expected' does not match the file. "
                            "Preserve indentation in new_lines too. "
                            "Pass all targeted edits for a file in a single call. "
                            "Set new_lines to an empty array to delete lines. "
                            "Each element is one line — no newline characters needed. "
                            "Use \"\" as an element for a blank line. "
                            "end_line beyond the last line is clamped silently."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Relative path to the file to edit.",
                                },
                                "edits": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "start_line": {
                                                "type": "integer",
                                                "description": "1-indexed first line of the range to replace (inclusive).",
                                            },
                                            "end_line": {
                                                "type": "integer",
                                                "description": "1-indexed last line of the range to replace (inclusive).",
                                            },
                                            "expected": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "The EXACT current lines being replaced (1 element per line, no newline chars, include all leading whitespace). Tool verifies this against the file and rejects the edit on mismatch. Copy verbatim from read_file output.",
                                            },
                                            "new_lines": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "Replacement lines. Each element is one line — no newline characters needed. Preserve exact indentation (⚠️ include all leading spaces or tabs ⚠️). Use \"\" for a blank line. Empty array deletes the range.",
                                            },
                                        },
                                        "required": ["start_line", "end_line", "expected", "new_lines"],
                                    },
                                    "description": "List of line-range edits to apply. All edits applied atomically.",
                                },
                            },
                            "required": ["path", "edits"],
                        },
                    },
                },
                "safe": False,
            },
        ]

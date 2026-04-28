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

    _TOOL_NAMES = {"list_files", "read_file", "edit_file", "bash", "replace_lines"}

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
        for item in sorted(target.iterdir()):
            rel = str(item.relative_to(self._base)).replace("\\", "/")
            if not self._is_ignored(rel):
                entries.append(rel + ("/" if item.is_dir() else ""))
        return "\n".join(entries) if entries else "(empty)"

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
            lines = content.splitlines(keepends=True)
            width = len(str(len(lines)))
            numbered = "".join(f"{i + 1:>{width}} | {line}" for i, line in enumerate(lines))
            plural = "" if len(lines) == 1 else "s"
            return f"Successfully read from '{path}' ({len(lines)} line{plural})\nContent:\n{numbered}"
        except Exception as e:
            return f"Error reading '{path}': {e}"

    def edit_file(self, path: str, old_str: str, new_str: str) -> str:
        resolved, err = self._check(path)
        if err:
            return err
        if old_str == "":
            if resolved.exists():
                return f"Error: '{path}' already exists. Use old_str/new_str to edit it."
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(new_str, encoding="utf-8")
            return f"Created '{path}'."
        if not resolved.exists():
            return f"Error: '{path}' does not exist."
        content = resolved.read_text(encoding="utf-8")
        if old_str not in content:
            return f"Error: text not found in '{path}'."
        resolved.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
        return f"Edited '{path}'."

    def replace_lines(self, path: str, start_line: int, end_line: int, new_str: str) -> str:
        resolved, err = self._check(path)
        if err:
            return err
        if not resolved.exists():
            return f"Error: '{path}' does not exist."
        if not resolved.is_file():
            return f"Error: '{path}' is not a file."
        if start_line < 1:
            return "Error: start_line must be >= 1."
        try:
            content = resolved.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading '{path}': {e}"
        lines = content.splitlines(keepends=True)
        total = len(lines)
        if start_line > total:
            return (
                f"Error: start_line ({start_line}) is beyond the file's "
                f"line count ({total})."
            )
        if end_line < start_line:
            return f"Error: start_line ({start_line}) must be <= end_line ({end_line})."
        end_line = min(end_line, total)
        before = lines[:start_line - 1]
        after = lines[end_line:]
        if new_str == "":
            new_segment = []
        else:
            # Add trailing newline when there is surrounding context (before or after),
            # to avoid fusing with adjacent lines or stripping the file's final newline.
            if (after or before) and not new_str.endswith("\n"):
                insertion = new_str + "\n"
            else:
                insertion = new_str
            new_segment = [insertion]
        try:
            resolved.write_text("".join(before + new_segment + after), encoding="utf-8")
        except Exception as e:
            return f"Error writing '{path}': {e}"
        return f"Replaced lines {start_line}-{end_line} in '{path}'."

    def bash(self, commands: list[str]) -> str:
        if not commands:
            return "Error: no commands provided."
        results = []
        for cmd in commands:
            r = subprocess.run(
                ["wsl.exe", "bash", "-ls"],
                input=cmd,
                capture_output=True, text=True, timeout=30, cwd=self._base,
                encoding="utf-8",
            )
            output = (r.stdout + r.stderr).strip()
            results.append(f"$ {cmd}\n{output if output else '(no output) - Success'}")
        return "\n\n".join(results)

    def execute(self, tool_name: str, arguments: dict) -> str:
        try:
            if tool_name == "list_files":
                return self.list_files(arguments.get("path", ""))
            if tool_name == "read_file":
                return self.read_file(arguments.get("path", ""))
            if tool_name == "edit_file":
                return self.edit_file(
                    arguments.get("path", ""),
                    arguments.get("old_str", ""),
                    arguments.get("new_str", ""),
                )
            if tool_name == "replace_lines":
                return self.replace_lines(
                    arguments.get("path", ""),
                    arguments.get("start_line", 0),
                    arguments.get("end_line", 0),
                    arguments.get("new_str", ""),
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
                            f"List files and directories under the configured base directory ({self._base}). "
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
                        "name": "edit_file",
                        "description": (
                            "Create or edit a file. "
                            "If old_str is empty, creates a new file with new_str as content (fails if file exists). "
                            "If file exists and you don't know its content use the read_file tool to help you populate old_str."
                            "If old_str is provided, replaces the first occurrence of old_str with new_str."
                            "ALWAYS prioritize replace_lines tool for editing a file !!!"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Relative path to the file to create or edit.",
                                },
                                "old_str": {
                                    "type": "string",
                                    "description": "Text to find and replace. Empty string to create a new file.",
                                },
                                "new_str": {
                                    "type": "string",
                                    "description": "Replacement text, or full content when creating.",
                                },
                            },
                            "required": ["path", "old_str", "new_str"],
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

⚠️ File Creation Warning: For multi-line code or files with special characters, use heredoc syntax with cat:
  cat > "file.py" << 'EOF'
  def hello():
      print("world")
  EOF
Do NOT use echo or echo -e to create code files—they produce literal \n characters instead of actual newlines.

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
                            "Replace a range of lines in an existing file by line number. "
                            "Use read_file first to see line numbers, then specify start_line through "
                            "end_line (1-indexed, inclusive). "
                            "Set new_str to empty string to delete those lines. "
                            "new_str is used verbatim — include a trailing newline if you want one. "
                            "end_line beyond the last line is clamped silently. "
                            "Always replace lines from bottom to top when using replace_lines mutliple time on the same file as lines count will change with each edit. "
                            "Prefer this over edit_file for targeted changes to large files."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Relative path to the file to edit.",
                                },
                                "start_line": {
                                    "type": "integer",
                                    "description": "1-indexed first line of the range to replace (inclusive).",
                                },
                                "end_line": {
                                    "type": "integer",
                                    "description": "1-indexed last line of the range to replace (inclusive).",
                                },
                                "new_str": {
                                    "type": "string",
                                    "description": "Replacement content. Empty string deletes the lines.",
                                },
                            },
                            "required": ["path", "start_line", "end_line", "new_str"],
                        },
                    },
                },
                "safe": False,
            },
        ]

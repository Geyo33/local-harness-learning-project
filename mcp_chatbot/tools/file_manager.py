from __future__ import annotations

import fnmatch
import re
from pathlib import Path
import subprocess


class FileManager:
    """
    Scoped filesystem access for LLM tool calls.

    Enforces:
    - All paths resolved under base_dir (blocks path traversal).
    - .gitignore patterns deny read and write access.
    """

    _TOOL_NAMES = {"list_files", "read_file", "bash", "edit_file"}

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
            numbered = "".join(f"{i + 1:>{width}} | {line}\n" for i, line in enumerate(lines))
            plural = "" if len(lines) == 1 else "s"
            return f"Successfully read from '{path}' ({len(lines)} line{plural})\nContent:\n{numbered}"
        except Exception as e:
            return f"Error reading '{path}': {e}"

    @staticmethod
    def _strip_gutter(text: str) -> str:
        """Remove a read_file-style 'N | ' line-number gutter (and any wrapping
        quotes) the model may have copied into find. Returns the cleaned text."""
        out = []
        for ln in text.split("\n"):
            s = re.sub(r"^\s*\d+\s*\|\s?", "", ln)
            if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
                s = s[1:-1]
            out.append(s)
        return "\n".join(out)

    def edit_file(self, path: str, find: str, replace: str) -> str:
        if not path or not path.strip():
            return (
                "Error: no 'path' provided. edit_file requires the relative file "
                "path. Retry with path set."
            )
        resolved, err = self._check(path)
        if err:
            return err
        if not resolved.exists():
            return f"Error: '{path}' does not exist."
        if not resolved.is_file():
            return f"Error: '{path}' is not a file."
        if not find.strip():
            return "Error: 'find' is empty. Provide the exact text to locate."
        try:
            # newline="" disables newline translation so EOLs round-trip unchanged
            with open(resolved, "r", encoding="utf-8", newline="") as f:
                content = f.read()
        except Exception as e:
            return f"Error reading '{path}': {e}"

        # Detect EOL, match on \n-normalized copies (so multi-line find works on CRLF)
        eol = "\r\n" if "\r\n" in content else "\r" if "\r" in content else "\n"

        def _norm(s: str) -> str:
            return s.replace("\r\n", "\n").replace("\r", "\n")

        content_n = _norm(content)
        find_n = _norm(find)
        replace_n = _norm(replace)

        if replace_n == find_n:
            return "Error: 'replace' is identical to 'find' — no change to make."

        matched = find_n
        count = content_n.count(find_n)
        if count == 0:
            # Defensive: model may have copied a 'N | ' gutter from read_file.
            stripped = self._strip_gutter(find_n)
            if stripped and stripped != find_n and content_n.count(stripped) == 1:
                matched = stripped
                count = 1
        if count == 0:
            # Defensive: model often pads blank/continuation lines with trailing
            # whitespace it invented. Retry on a per-line rstrip'd comparison and,
            # if a single line-aligned window matches, splice the original span.
            c_lines = content_n.split("\n")
            f_lines = [ln.rstrip() for ln in find_n.split("\n")]
            n = len(f_lines)
            if n:
                hits = [
                    i
                    for i in range(len(c_lines) - n + 1)
                    if [ln.rstrip() for ln in c_lines[i : i + n]] == f_lines
                ]
                if len(hits) == 1:
                    matched = "\n".join(c_lines[hits[0] : hits[0] + n])
                    count = 1
        if count == 0:
            return (
                f"Error: find text not found in '{path}'. "
                "Re-read the file and copy the exact text (same blank spaces, no line numbers)."
            )
        if count > 1:
            return (
                f"Error: find text found {count} times in '{path}'. "
                "Add surrounding lines to make it unique."
            )

        new_content_n = content_n.replace(matched, replace_n, 1)
        out = new_content_n.replace("\n", eol)
        try:
            with open(resolved, "w", encoding="utf-8", newline="") as f:
                f.write(out)
        except Exception as e:
            return f"Error writing '{path}': {e}"
        return f"Applied edit to '{path}'."

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
            if tool_name == "edit_file":
                return self.edit_file(
                    arguments.get("path", ""),
                    arguments.get("find", ""),
                    arguments.get("replace", ""),
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
                        "name": "edit_file",
                        "description": (
                            "Edit a file by replacing an exact piece of text. "
                            "ALWAYS call read_file first. Copy the 'find' text VERBATIM "
                            "from the file — do NOT include the 'N | ' line numbers. "
                            "Include enough surrounding lines so 'find' appears EXACTLY "
                            "ONCE in the file (the tool errors if it matches zero or "
                            "multiple places). Preserve all indentation in both 'find' "
                            "and 'replace'. Set 'replace' to an empty string to delete "
                            "the found text. For a large change that rewrites most of a "
                            "file, prefer rewriting the whole file with the bash tool "
                            "(cat > file <<'EOF' … EOF) instead of many edit_file calls."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Relative path to the file to edit.",
                                },
                                "find": {
                                    "type": "string",
                                    "description": "Exact text to locate, copied verbatim from the file (no line numbers, keep indentation). Must be unique in the file.",
                                },
                                "replace": {
                                    "type": "string",
                                    "description": "Replacement text (keep indentation). Empty string deletes the found text.",
                                },
                            },
                            "required": ["path", "find", "replace"],
                        },
                    },
                },
                "safe": False,
            },
        ]

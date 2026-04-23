from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path


_SKILLS_DIR = Path(__file__).parent.parent / "skills"


def _load_dispatch(path: Path) -> dict:
    """Dynamically import a scripts.py and return its DISPATCH dict."""
    module_name = f"skill_script_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, "DISPATCH", {})


def _python_type_to_json(annotation) -> str:
    """Map a Python type annotation to a JSON Schema type string."""
    if annotation in (int, float):
        return "number"
    if annotation is str:
        return "string"
    if annotation is bool:
        return "boolean"
    return "string"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """
    Extract YAML-style frontmatter from a markdown string.

    Returns (metadata_dict, body) where metadata_dict contains the
    key/value pairs from the --- block and body is the remaining markdown.
    Only handles simple scalar string values (no lists, nested maps, etc.).
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    meta: dict[str, str] = {}
    end_index = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = i
            break
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    if end_index is None:
        return {}, text

    body = "\n".join(lines[end_index + 1:]).strip()
    return meta, body


class SkillsManager:
    """
    Scans mcp_chatbot/skills/ for SKILL.md files and exposes:
      - get_index()       → list of {name, description} for system prompt injection
      - read_skill(name)  → full SKILL.md content for LLM activation
    """

    def __init__(self, skills_dir: Path = _SKILLS_DIR) -> None:
        self._skills_dir = skills_dir
        # name -> {"description": str, "path": Path}
        self._index: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self._skills_dir.exists():
            logging.info("Skills directory not found at %s — no skills loaded.", self._skills_dir)
            return

        for skill_file in sorted(self._skills_dir.glob("*/SKILL.md")):
            try:
                text = skill_file.read_text(encoding="utf-8")
                meta, _ = _parse_frontmatter(text)
                name = meta.get("name")
                description = meta.get("description")
                if not name or not description:
                    logging.warning("Skipping %s — missing 'name' or 'description' in frontmatter.", skill_file)
                    continue
                script_path = skill_file.parent / "scripts.py"
                dispatch = {}
                if script_path.exists():
                    try:
                        dispatch = _load_dispatch(script_path)
                        logging.info("Loaded %d function(s) from %s", len(dispatch), script_path)
                    except Exception as e:
                        logging.warning("Failed to load scripts from %s: %s", script_path, e)
                self._index[name] = {"description": description, "path": skill_file, "dispatch": dispatch}
                logging.info("Loaded skill: %s", name)
            except Exception as e:
                logging.warning("Failed to load skill from %s: %s", skill_file, e)

    def get_index(self) -> list[dict[str, str]]:
        """Return a list of {name, description} for all discovered skills."""
        return [
            {"name": name, "description": info["description"]}
            for name, info in self._index.items()
        ]

    def execute_function(self, skill_name: str, function_name: str, kwargs: dict) -> str:
        """Call a function from the given skill's DISPATCH dict and return the result as a string."""
        entry = self._index.get(skill_name)
        if entry is None:
            available = ", ".join(self._index.keys()) or "none"
            return f"Error: skill '{skill_name}' not found. Available skills: {available}."
        fn = entry.get("dispatch", {}).get(function_name)
        if fn is None:
            available_fns = ", ".join(entry.get("dispatch", {}).keys()) or "none"
            return f"Error: function '{function_name}' not found in skill '{skill_name}'. Available functions: {available_fns}."
        try:
            return str(fn(**kwargs))
        except Exception as e:
            return f"Error executing '{function_name}': {e}"

    def read_skill(self, skill_name: str) -> str:
        """Return the full SKILL.md content for the given skill name."""
        entry = self._index.get(skill_name)
        if entry is None:
            available = ", ".join(self._index.keys()) or "none"
            return f"Skill '{skill_name}' not found. Available skills: {available}."
        try:
            return entry["path"].read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading skill '{skill_name}': {e}"

    def build_function_schemas(self, skill_name: str) -> list[dict]:
        """
        Return OpenAI tool schemas for all functions in the given skill's DISPATCH.
        Tool names are namespaced as skill__<skill_name>__<function_name>.
        Returns [] if skill not found or has no DISPATCH.
        """
        entry = self._index.get(skill_name)
        if not entry:
            return []
        schemas = []
        for fn_name, fn in entry.get("dispatch", {}).items():
            sig = inspect.signature(fn)
            properties: dict[str, dict] = {}
            required: list[str] = []
            for param_name, param in sig.parameters.items():
                if param.annotation is inspect.Parameter.empty:
                    logging.warning(
                        "Parameter '%s' in skill '%s' function '%s' has no type annotation; defaulting to 'string'.",
                        param_name, skill_name, fn_name,
                    )
                json_type = _python_type_to_json(param.annotation)
                properties[param_name] = {"type": json_type}
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
            description = (fn.__doc__ or f"Call {fn_name} from the {skill_name} skill.").strip()
            schemas.append({
                "type": "function",
                "function": {
                    "name": f"skill__{skill_name}__{fn_name}",
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return schemas

    @property
    def tool_schema(self) -> dict:
        """OpenAI-compatible function schema for the built-in read_skill tool."""
        return {
            "type": "function",
            "function": {
                "name": "read_skill",
                "description": (
                    "Load the full instructions for a skill by name. "
                    "Call this when a task matches one of the available skills "
                    "listed in the system prompt."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "The exact skill name as listed in the system prompt.",
                        }
                    },
                    "required": ["skill_name"],
                },
            },
        }

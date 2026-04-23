from pathlib import Path
import pytest
import tempfile
import textwrap


def _make_skill(tmp_path: Path, name: str, scripts_src: str) -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill.\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    (skill_dir / "scripts.py").write_text(textwrap.dedent(scripts_src), encoding="utf-8")
    return skill_dir


def test_build_function_schemas_returns_openai_schemas(tmp_path):
    from mcp_chatbot.tools.skills_manager import SkillsManager

    _make_skill(tmp_path, "calc", """
        def add(a: float, b: float) -> str:
            \"\"\"Add two numbers.\"\"\"
            return str(a + b)

        DISPATCH = {"add": add}
    """)

    sm = SkillsManager(skills_dir=tmp_path)
    schemas = sm.build_function_schemas("calc")

    assert len(schemas) == 1
    s = schemas[0]
    assert s["type"] == "function"
    fn = s["function"]
    assert fn["name"] == "skill__calc__add"
    assert fn["description"] == "Add two numbers."
    props = fn["parameters"]["properties"]
    assert props["a"]["type"] == "number"
    assert props["b"]["type"] == "number"
    assert fn["parameters"]["required"] == ["a", "b"]


def test_build_function_schemas_str_type(tmp_path):
    from mcp_chatbot.tools.skills_manager import SkillsManager

    _make_skill(tmp_path, "text", """
        def reverse(text: str) -> str:
            \"\"\"Reverse a string.\"\"\"
            return text[::-1]

        DISPATCH = {"reverse": reverse}
    """)

    sm = SkillsManager(skills_dir=tmp_path)
    schemas = sm.build_function_schemas("text")
    props = schemas[0]["function"]["parameters"]["properties"]
    assert props["text"]["type"] == "string"


def test_build_function_schemas_unknown_skill_returns_empty(tmp_path):
    from mcp_chatbot.tools.skills_manager import SkillsManager

    sm = SkillsManager(skills_dir=tmp_path)
    assert sm.build_function_schemas("nonexistent") == []


def test_build_function_schemas_no_dispatch_returns_empty(tmp_path):
    from mcp_chatbot.tools.skills_manager import SkillsManager

    skill_dir = tmp_path / "empty"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: empty\ndescription: No scripts.\n---\n",
        encoding="utf-8",
    )
    # No scripts.py

    sm = SkillsManager(skills_dir=tmp_path)
    assert sm.build_function_schemas("empty") == []


def test_build_function_schemas_fallback_description(tmp_path):
    from mcp_chatbot.tools.skills_manager import SkillsManager

    _make_skill(tmp_path, "nodoc", """
        def fn(x: int) -> str:
            return str(x)

        DISPATCH = {"fn": fn}
    """)

    sm = SkillsManager(skills_dir=tmp_path)
    schemas = sm.build_function_schemas("nodoc")
    assert schemas[0]["function"]["description"] != ""


def test_build_function_schemas_bool_type(tmp_path):
    from mcp_chatbot.tools.skills_manager import SkillsManager

    _make_skill(tmp_path, "flags", """
        def toggle(enabled: bool) -> str:
            \"\"\"Toggle a flag.\"\"\"
            return str(not enabled)

        DISPATCH = {"toggle": toggle}
    """)

    sm = SkillsManager(skills_dir=tmp_path)
    schemas = sm.build_function_schemas("flags")
    props = schemas[0]["function"]["parameters"]["properties"]
    assert props["enabled"]["type"] == "boolean"

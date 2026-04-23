from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mcp_chatbot.core.server import Server
from mcp_chatbot.core.llm_client import LLMClient
from mcp_chatbot.tools.skills_manager import SkillsManager
from mcp_chatbot.tools.file_manager import FileManager
from mcp_chatbot.settings import load_settings
from mcp_chatbot.core.task_manager import TaskManager


MAX_PARSE_RETRIES = 3
DEFAULT_TOOL_TIMEOUT = 30.0


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self._is_initialized: bool = False
        self.messages: list[dict[str, Any]] = []
        self.token_usage: dict = {"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0, "completion_tokens_details": {"reasoning_tokens": 0}}, "context_size": self.llm_client.max_tokens}
        self._tool_schemas: list[dict[str, Any]] = []
        self._tool_server_map: dict[str, Server] = {}
        self._tool_timeout_map: dict[str, float] = {}
        self._skills_manager: SkillsManager = SkillsManager()
        self.active_servers: dict[str, bool] = {}
        self.tool_call_detected: bool = False
        self._active_skill: str | None = None
        self._injected_skill_schemas: list[dict] = []
        self.allow_tool_event = asyncio.Event()
        self.allow_tool_action: str | None = None
        self.deny_reason: str = ""
        self.file_manager: FileManager | None = None
        settings = load_settings()
        file_root = settings.get("file_root")
        if file_root:
            try:
                self.file_manager = FileManager(Path(file_root))
                logging.info("FileManager initialized with root: %s", file_root)
            except Exception as e:
                logging.warning("Could not initialize FileManager: %s", e)

        task_root = Path(file_root) if file_root else Path(".")
        try:
            self.task_manager: TaskManager = TaskManager(task_root)
            logging.info("TaskManager initialized at root: %s", task_root)
        except Exception as e:
            logging.warning("Could not initialize TaskManager: %s", e)
            self.task_manager = None

        self.planner_mode: str = settings.get("planner_mode", "auto")

    async def list_servers(self) -> None:
        try:
            for server in self.servers:
                self.active_servers[server.name] = False
            logging.info("All servers listed as inactive")
        except Exception as e:
            logging.error("Listing servers failed for one or more servers: %s", e)
            raise

    async def initialize_servers(self) -> bool:
        try:
            for server in self.servers:
                if self.active_servers[server.name] == True:
                    await server.initialize()
            self._is_initialized = True
            logging.info("All servers initialized successfully.")
            return True
        except Exception as e:
            logging.error("Initialization failed for one or more servers: %s", e)
            await self.cleanup_servers()
            self._is_initialized = False
            return False

    async def build_system_message(self) -> None:
        """
        Collect all tool schemas from active servers and build the initial
        messages list.  Tool descriptions are no longer injected as plain
        text — they are registered as proper OpenAI-style function schemas
        and passed to the API on every request.
        """
        try:
            self._tool_schemas = []
            self._tool_server_map = {}
            self._tool_timeout_map = {}
            self._active_skill = None
            self._injected_skill_schemas = []

            for server in self.servers:
                if self.active_servers[server.name]:
                    tools = await server.list_tools()
                    for tool in tools:
                        self._tool_schemas.append(tool.to_openai_schema())
                        self._tool_server_map[tool.name] = server
                        self._tool_timeout_map[tool.name] = (
                            server.timeout if server.timeout is not None else DEFAULT_TOOL_TIMEOUT
                        )

            skill_index = self._skills_manager.get_index()
            if skill_index:
                skills_lines = "\n".join(
                    f"- {s['name']}: {s['description']}" for s in skill_index
                )
                skills_prompt = (
                    f"\n\n### Skills\n\nYou have access to skills. When a task matches a skill:"
                    f"\n1. Call read_skill(skill_name) to load the skill instructions."
                    f"\n2. Follow the instructions. If the skill has callable functions, they will be available as tools — use them via normal tool calls."
                    f"\n\nAvailable skills:\n---\n"
                    f"{skills_lines}\n---\n"
                )
            else:
                skills_prompt = ""

            # Load optional mission brief
            mission_brief = ""
            try:
                plan_path = Path(load_settings().get("file_root") or ".") / "plan.md"
                if plan_path.exists():
                    mission_brief = "\n\n### Mission Brief\n\n" + plan_path.read_text(encoding="utf-8").strip()
                    logging.info("Loaded mission brief from %s", plan_path)
                else:
                    logging.info("Optional mission brief not loaded, %s doesn't exist", plan_path)
            except Exception as e:
                logging.warning("Could not load plan.md: %s", e)

            task_context = ""
            if self.task_manager:
                task_context = (
                    "\n\n## Tasks\n"
                    "Tasks appear in <tasks> before each message. "
                    "For any multi-step request: call plan_tasks(titles) first if no plan exists. "
                    "Call update_task(id, in_progress) before starting each task; "
                    "update_task(id, done) when finished. "
                    "If task block get messed up you can rebuild it using plan_tasks(titles)"
                    "Task tools run silently — no approval needed."
                )

            file_context = ""
            if self.file_manager:
                file_context = (
                    "\n\n## Files\n"
                    "list_files runs silently, needs to be re-used to check sub-directories. "
                    "read_file, edit_file, and bash require approval. "
                    "bash: each command is a separate, stateless WSL shell starting at base_dir — "
                    "chain dependent steps with && in a single command."
                )

            system_content = (
                "You are a helpful assistant. "
                "Use tools when they help; reply directly when they don't."
                + task_context
                + file_context
                + skills_prompt
                + mission_brief
            )

            self.messages = [{"role": "user", "content": system_content}]

            if skill_index:
                self._tool_schemas.append(self._skills_manager.tool_schema)

            if self.file_manager:
                for entry in self.file_manager.tool_entries:
                    self._tool_schemas.append(entry["schema"])

            if self.task_manager:
                for schema in self.task_manager.tool_schemas:
                    self._tool_schemas.append(schema)

            logging.info("Registered %d tool(s) and %d skill(s).", len(self._tool_schemas), len(skill_index))
        except Exception as e:
            error_msg = f"Error building system messages: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)

    def _inject_task_block(self, messages: list[dict], hint: str = "") -> list[dict]:
        if self.task_manager is None and not hint:
            return messages
        block = self.task_manager.render_block() if self.task_manager else ""
        prefix_parts = []
        if hint:
            prefix_parts.append(hint)
        if block:
            prefix_parts.append(block)
        if not prefix_parts:
            return messages
        prefix = "\n\n".join(prefix_parts)
        last = messages[-1]
        content = last["content"]
        if isinstance(content, str):
            new_content = prefix + "\n\n" + content
        else:
            new_content = [{"type": "text", "text": prefix}] + list(content)
        return messages[:-1] + [{**last, "content": new_content}]

    def _should_suggest_plan(self, user_input: str | list) -> bool:
        if isinstance(user_input, str):
            text = user_input
        else:
            text = " ".join(
                p["text"] for p in user_input
                if isinstance(p, dict) and p.get("type") == "text"
            )
        text_lower = text.lower()
        triggers = ["and then", "after that", "first ", "then "]
        file_path_count = text.count("/") + text.count("\\")
        return (
            len(text) > 200
            or any(t in text_lower for t in triggers)
            or file_path_count >= 4
        )

    async def _run_planner_pass(self, user_input: str | list) -> bool:
        """One LLM call restricted to plan_tasks only. Returns True if plan was seeded."""
        if not self.task_manager:
            return False
        plan_schema = next(
            (s for s in self.task_manager.tool_schemas if s["function"]["name"] == "plan_tasks"),
            None,
        )
        if plan_schema is None:
            return False

        if isinstance(user_input, str):
            text = user_input
        else:
            text = " ".join(
                p.get("text", "") for p in user_input
                if isinstance(p, dict) and p.get("type") == "text"
            )

        planner_messages = [
            {
                "role": "user",
                "content": (
                    "You are a task planner. Your only job is to call plan_tasks with a list of "
                    "short, actionable task titles for the following request. Do nothing else.\n\n"
                    f"Request: {text}"
                ),
            }
        ]

        tool_calls = None
        for event in self.llm_client.stream_response(planner_messages, tools=[plan_schema]):
            if event["type"] == "tool_calls_final":
                tool_calls = event["data"]

        if not tool_calls:
            return False

        for tc in tool_calls:
            if tc["function"]["name"] == "plan_tasks":
                try:
                    args = json.loads(tc["function"].get("arguments", "{}"))
                except json.JSONDecodeError:
                    return False
                self.task_manager.execute("plan_tasks", args)
                logging.info("Planner pass seeded plan.")
                return True

        return False

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        logging.info("Starting server cleanup...")
        for server in reversed(self.servers):
            self.active_servers[server.name] = False
            await server.cleanup()
        self._is_initialized = False

    async def _execute_tool_call(self, tool_call: dict[str, Any]) -> str:
        """
        Execute a single tool_call dict (OpenAI format) and return the
        result as a string suitable for a ``tool`` role message.
        """
        name = tool_call["function"]["name"]
        try:
            arguments = json.loads(tool_call["function"].get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}

        logging.info("Executing tool: %s", name)
        logging.info("With arguments: %s", arguments)

        if name == "read_skill":
            skill_name = arguments.get("skill_name", "")
            return self._skills_manager.read_skill(skill_name)

        if name.startswith("skill__"):
            parts = name.split("__", 2)
            if len(parts) == 3:
                _, skill_name, fn_name = parts
                return self._skills_manager.execute_function(skill_name, fn_name, arguments)
            return f"Error: malformed skill tool name '{name}'"

        if self.task_manager and name in self.task_manager.tool_names:
            return self.task_manager.execute(name, arguments)

        if self.file_manager and name in self.file_manager.tool_names:
            return self.file_manager.execute(name, arguments)

        server = self._tool_server_map.get(name)
        if server is None:
            return f"Error: no server found for tool '{name}'"

        timeout = self._tool_timeout_map.get(name, DEFAULT_TOOL_TIMEOUT)
        try:
            result = await server.execute_tool(name, arguments, timeout=timeout)
            if isinstance(result, dict) and "progress" in result:
                progress = result["progress"]
                total = result["total"]
                logging.info("Progress: %d/%d (%.1f%%)", progress, total, (progress / total) * 100)
            return str(result.content)
        except asyncio.TimeoutError:
            error_msg = f"Tool timed out after {timeout}s"
            logging.warning("Tool '%s' timed out after %.1fs", name, timeout)
            return error_msg
        except Exception as e:
            error_msg = f"Error executing tool '{name}': {str(e)}"
            logging.error(error_msg)
            return error_msg

    async def handle_user_input(self, user_input: str | list[dict[str, Any]]):
        """
        Handles one turn of conversation from an external source (e.g. Gradio).

        Agentic loop:
          1. Send messages + tool schemas to the LLM (non-streaming).
          2. If the model returns tool_calls, optionally ask the user for
             permission, execute each call, append tool-role results, and
             repeat from step 1.
          3. Once the model returns a plain text response (no tool_calls),
             stream that final answer to the caller for display.

        Yields protocol strings compatible with the existing Gradio frontend:
          - Plain text chunks for the final streamed answer.
          - ``"<tool_name>o|o<assistant_msg>o|o<tool_result>"`` for tool events.
          - ``"Tool call deniedo|o<assistant_msg>o|o---"`` when the user denies.
        """
        if not self._is_initialized:
            yield "Error: Client is not ready. Please wait for initialization."
            return

        _planner_ran = False
        if self.planner_mode != "off" and self.task_manager and self.task_manager.is_empty():
            should_plan = (self.planner_mode == "always") or self._should_suggest_plan(user_input)
            if should_plan:
                _planner_ran = await self._run_planner_pass(user_input)

        _hint = ""
        if not _planner_ran and self.task_manager and self.task_manager.is_empty() and self._should_suggest_plan(user_input):
            _hint = "[Hint: You have no tasks yet. Consider calling plan_tasks first.]"

        self.messages.append({"role": "user", "content": user_input})

        if self._active_skill:
            for s in self._injected_skill_schemas:
                try:
                    self._tool_schemas.remove(s)
                except ValueError:
                    pass
            self._injected_skill_schemas = []
            self._active_skill = None

        parse_retries = 0  # tracks consecutive malformed-JSON tool call responses
        messages_checkpoint = len(self.messages)

        while True:
            tools = self._tool_schemas or None
            with open("mcp_chatbot/log.json", "w") as f:
                json.dump(self.messages, f, indent=2)
            assistant_msg = ""
            tool_calls = None
            tool_names = {}
            tool_args = {}
            call_messages = self._inject_task_block(self.messages, hint=_hint)
            _hint = ""  # consume — subsequent iterations don't repeat hint
            with open("mcp_chatbot/logs_with_tasks.json", "w") as f:
                json.dump(call_messages, f, indent=2)
            for event in self.llm_client.stream_response(call_messages, tools=tools):
                if event["type"] == "content":
                    assistant_msg += event["data"]
                    yield event["data"]
                elif event["type"] == "tool_name":
                    tool_names[str(event["index"])] = {"name": event["data"]}
                elif event["type"] == "tool_arguments":
                    tool_args[str(event["index"])] = {"args": event["data"]}
                elif event["type"] == "tool_calls_final":
                    tool_calls = event["data"]
                elif event["type"] == "usage":
                    self.token_usage["usage"] = event["data"]
                elif event["type"] == "error":
                    assistant_msg += event["data"]
            logging.debug("tool_names: %s", tool_names)
            logging.debug("tool_args: %s", tool_args)

            # pop the "Use the above tool call result..." nudge from the previous iteration
            if (
                self.messages
                and self.messages[-1].get("content") == (
                    "If more tool calls are needed, call them now with no text. If task is complete, give your final answer. Take the above tool call result into account. Always use update_task when any task status change."
                )
            ):
                self.messages.pop(-1)

            # validate tool call argument JSON before executing
            if tool_calls:
                parse_error: Exception | None = None
                for tc in tool_calls:
                    try:
                        json.loads(tc["function"].get("arguments", "{}") or "{}")
                    except json.JSONDecodeError as e:
                        parse_error = e
                        break

                if parse_error is not None:
                    if parse_retries >= MAX_PARSE_RETRIES:
                        del self.messages[messages_checkpoint:]
                        yield (
                            f"Error: LLM produced invalid tool call JSON after "
                            f"{MAX_PARSE_RETRIES} retries. Aborting."
                        )
                        return
                    parse_retries += 1
                    logging.warning(
                        "Malformed tool call JSON (attempt %d/%d): %s",
                        parse_retries, MAX_PARSE_RETRIES, parse_error,
                    )
                    self.messages.append({
                        "role": "user",
                        "content": (
                            f"Your last response contained invalid JSON in tool arguments. "
                            f"Error: {parse_error}. "
                            f"Please retry your tool call with valid JSON arguments."
                        ),
                    })
                    continue  # re-enter while loop; do NOT execute the bad call
                else:
                    parse_retries = 0  # clean response; reset counter

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": assistant_msg})
                logging.info("Final response: %s", assistant_msg)
                if self._active_skill:
                    for s in self._injected_skill_schemas:
                        try:
                            self._tool_schemas.remove(s)
                        except ValueError:
                            pass
                    self._injected_skill_schemas = []
                    self._active_skill = None
                with open("mcp_chatbot/log.json", "w") as f:
                    json.dump(self.messages, f, indent=2)
                break

            if assistant_msg:
                self.messages.append({"role": "assistant", "content": assistant_msg})

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_arg = tool_call["function"]["arguments"]
                logging.info("Attempting Tool Call: %s", tool_name)

                is_safe_file_tool = (
                    self.file_manager is not None
                    and self.file_manager.is_safe(tool_name)
                )
                is_safe_task_tool = (
                    self.task_manager is not None
                    and tool_name in self.task_manager.tool_names
                )

                if not (is_safe_file_tool or is_safe_task_tool):
                    self.tool_call_detected = True
                    yield tool_name, tool_arg
                    if self.allow_tool_action != "always":
                        self.allow_tool_event.clear()
                        await self.allow_tool_event.wait()

                    if self.allow_tool_action == "deny":
                        self.tool_call_detected = False
                        self.allow_tool_action = None
                        with_reason = f" Reason for denying tool call: {self.deny_reason}"
                        denial_text = f"**Result:**\nTool call was denied by user.\n\n**Arguments:**\n```json\n{tool_arg}\n```"
                        yield f"Tool call deniedo|o{tool_arg}o|o{denial_text}o|oFalse"
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", tool_name),
                            "content": denial_text,
                        })
                        self.messages.append({"role": "user", "content": with_reason if self.deny_reason else ''})
                        break

                    if self.allow_tool_action == "allow":
                        self.allow_tool_action = None

                    self.tool_call_detected = False

                result_text = await self._execute_tool_call(tool_call)

                yield f"{tool_name}o|o{json.dumps(tool_call)}o|o{result_text}o|o{str(is_safe_file_tool or is_safe_task_tool)}"

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", tool_name),
                    "content": f"**Result:**\n\n{result_text}",
                })
                if tool_name == "read_skill":
                    skill_name = json.loads(
                        tool_call["function"].get("arguments", "{}")
                    ).get("skill_name", "")
                    fn_schemas = self._skills_manager.build_function_schemas(skill_name)
                    if fn_schemas:
                        for s in self._injected_skill_schemas:
                            try:
                                self._tool_schemas.remove(s)
                            except ValueError:
                                pass
                        self._injected_skill_schemas = fn_schemas
                        self._tool_schemas.extend(fn_schemas)
                        self._active_skill = skill_name
                        logging.info(
                            "Injected %d skill function schema(s) for '%s'.",
                            len(fn_schemas), skill_name,
                        )
                else:
                    nudge = (
                        "If more tool calls are needed, call them now with no text. "
                        "If task is complete, give your final answer. "
                        "Take the above tool call result into account."
                    )
                    # if self.task_manager:
                    #     nudge += " Always use update_task when any task status change."
                    self.messages.append({"role": "user", "content": nudge})
            self.tool_call_detected = False

    async def set_allow_tool_action(self, action: str, reason: str = "") -> None:
        """Called by the UI (e.g. Gradio) to approve or deny a pending tool call."""
        self.deny_reason = reason
        self.allow_tool_action = action
        self.allow_tool_event.set()
        logging.debug("set_allow_tool_action called with '%s'", action)

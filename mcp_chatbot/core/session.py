from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any
import time
import datetime

from mcp_chatbot.core.server import Server
from mcp_chatbot.core.llm_client import LLMClient
from mcp_chatbot.core.schemas import EpisodeResult
from mcp_chatbot.tools.skills_manager import SkillsManager
from mcp_chatbot.tools.file_manager import FileManager
from mcp_chatbot.settings import load_settings
from mcp_chatbot.core.task_manager import TaskManager
from mcp_chatbot.core.context_manager import ContextManager
from mcp_chatbot.memory.store import EpisodicStore
from mcp_chatbot.memory.memory_manager import MemoryManager
from mcp_chatbot.memory.playbook import PlaybookManager


MAX_PARSE_RETRIES = 3
MAX_AGENT_ITERATIONS = 20
AGENT_WALL_CLOCK_TIMEOUT = 900  # seconds; fallback default, override via settings.json["agent_timeout"]
TIMEOUT_SYNTHESIS_BUDGET = 60  # seconds; bounded sub-call to synthesize a final answer on wall-clock abort
# Procedure attribution: the planner injects the top-N candidate procedures but
# the model typically follows only one. On reinforce/penalize we estimate which
# by similarity of each procedure's action to the plan that actually ran; the
# best match gets the full delta, the rest a reduced residual share.
_PROC_ATTRIB_THRESHOLD = 0.12  # min token-Jaccard(plan, action) to trust attribution; else rank prior
_PROC_RESIDUAL_FRAC = 0.3      # delta fraction applied to non-attributed injected procedures
STALE_TASK_ITERATIONS = 5
DEFAULT_TOOL_TIMEOUT = 30.0
COMPRESSION_KEEP_TURNS = 4
# Appended to the assistant bubble (and persisted to history) when the user
# interrupts a turn via the stop button. Kept as one constant so the yielded
# display string and the persisted history marker can never drift apart.
STOPPED_MARKER = "_[Stopped by user]_"
# Appended to the assistant bubble when the backend stops generation because the
# completion hit max_output_tokens (finish_reason == "length"). The answer is
# truncated mid-output; this tells the user (and persists to history) rather than
# leaving a silently cut-off message.
TRUNCATED_MARKER = "_[Output truncated — hit max_output_tokens cap]_"
# Yielded after a tool call is approved, before the (possibly slow) execution,
# so the frontend can retract the approval gate immediately instead of leaving
# it spinning until the result arrives. Consumed in handle_chat — never shown.
GATE_CLOSE_MARKER = "\x00gate_close\x00"


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self._is_initialized: bool = False
        self.messages: list[dict[str, Any]] = []
        self.token_usage: dict = {
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "context_size": self.llm_client.max_tokens,
            "breakdown": {},
        }
        self.context_mgr = ContextManager(self.llm_client.max_tokens)
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
        file_root = settings.get("file_root") or "."
        if file_root:
            try:
                self.file_manager = FileManager(Path(file_root))
                logging.info("FileManager initialized with root: %s", file_root)
            except Exception as e:
                logging.warning("Could not initialize FileManager: %s", e)

        task_root = Path(file_root)
        try:
            self.task_manager: TaskManager = TaskManager(task_root)
            logging.info("TaskManager initialized at root: %s", task_root)
        except Exception as e:
            logging.warning("Could not initialize TaskManager: %s", e)
            self.task_manager = None

        self._session_id: str = str(uuid.uuid4())
        self.episodic_store: EpisodicStore | None = None
        try:
            db_path = Path(file_root) / ".agent" / "memory.db"
            self.episodic_store = EpisodicStore(db_path)
            logging.info("EpisodicStore initialized at: %s", db_path)
        except Exception as e:
            logging.warning("Could not initialize EpisodicStore: %s", e)

        self.memory_manager: MemoryManager | None = None
        if self.episodic_store:
            self.memory_manager = MemoryManager(self.episodic_store)
            logging.info("MemoryManager initialized.")

        self.playbook_manager: PlaybookManager | None = None
        if self.episodic_store:
            try:
                self.playbook_manager = PlaybookManager(self.episodic_store)
                logging.info("PlaybookManager initialized (store-backed).")
            except Exception as e:
                logging.warning("Could not initialize PlaybookManager: %s", e)

        self.planner_mode: str = settings.get("planner_mode", "auto")
        # Wall-clock cap for the agentic loop (seconds). Local LLMs run far slower
        # than cloud endpoints, so a single multi-step plan can legitimately exceed
        # the 300s default; raise via settings.json["agent_timeout"] for local use.
        self.agent_timeout: float = float(
            settings.get("agent_timeout", AGENT_WALL_CLOCK_TIMEOUT)
        )
        # +0.05 reinforce on plan completion. Default on (current behavior);
        # disable to avoid double-counting with the record_procedure dedup bump.
        self.playbook_reinforce: bool = settings.get("playbook_reinforce", True)
        self.agent_loop = {"active": False, "state": ""}
        self.interrupt_requested: bool = False
        self.load_episodes: bool = True
        # Plan-scoped: each entry {"id": int, "action": str}, in retrieval rank
        # order. Drives weighted reinforce/penalize via _attribution_weights.
        # Populated two ways: the planner pass sets it directly; a model-issued
        # search_playbook + plan_tasks commits the turn buffer below into it.
        self._injected_procedures: list[dict] = []
        # Turn-scoped buffer of procedures the model looked up via search_playbook
        # this turn (same {id, action} shape as above). Committed into
        # _injected_procedures when the model calls plan_tasks, so self-retrieved
        # procedures feed the same reinforce/penalize loop the planner-injected
        # ones do. Reset each turn.
        self._retrieved_procedures: list[dict] = []
        # On-demand fact retrieval: sliding window of recently-retrieved facts.
        # Each entry {"id": int, "fact": str, "ttl": int}; ttl decremented each
        # user turn, dropped at 0. Ephemeral — never written to history.
        self._fact_window: list[dict] = []
        self._retrieval_k: int = settings.get("memory_retrieval_k", 5)
        self._retrieval_ttl: int = settings.get("memory_retrieval_ttl", 3)
        self._memory_block: str = ""  # rendered [Relevant Facts] block for current turn
        # Loop-end nudge to capture durable facts. Counts user turns; persisted
        # to settings.json so short sessions still eventually trigger it.
        self._facts_nudge_counter: int = settings.get("facts_nudge_progress", 0)
        self._facts_nudge_threshold: int = settings.get("facts_nudge_turns", 8)
        self._pending_facts_nudge: bool = False
        self._playbook_fired_this_turn: bool = False
        self._consumed_nudge: str = ""

    def reinitialize_root(self, root: Path) -> None:
        """Reinitialize all file-root-dependent components when the root changes."""
        try:
            self.file_manager = FileManager(root)
        except Exception as e:
            logging.warning("Could not reinitialize FileManager: %s", e)
            self.file_manager = None

        try:
            self.task_manager = TaskManager(root)
        except Exception as e:
            logging.warning("Could not reinitialize TaskManager: %s", e)
            self.task_manager = None

        try:
            db_path = root / ".agent" / "memory.db"
            self.episodic_store = EpisodicStore(db_path)
        except Exception as e:
            logging.warning("Could not reinitialize EpisodicStore: %s", e)
            self.episodic_store = None

        if self.episodic_store:
            self.memory_manager = MemoryManager(self.episodic_store)
            try:
                self.playbook_manager = PlaybookManager(self.episodic_store)
            except Exception as e:
                logging.warning("Could not reinitialize PlaybookManager: %s", e)
                self.playbook_manager = None
        else:
            self.memory_manager = None
            self.playbook_manager = None

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

    async def build_system_message(self, preserve_history: bool = False) -> None:
        """
        Collect all tool schemas from active servers and build the initial
        messages list.  Tool descriptions are no longer injected as plain
        text — they are registered as proper OpenAI-style function schemas
        and passed to the API on every request.

        By default this resets `self.messages` to just the system prompt
        (startup / settings-change / memory-reset contract). Pass
        `preserve_history=True` to rebuild the system prompt in place while
        keeping the existing conversation tail — used when a mid-session action
        (e.g. approving/pinning a fact) must refresh the prompt without
        wiping the live chat.
        """
        try:
            self._tool_schemas = []
            self._tool_server_map = {}
            self._tool_timeout_map = {}
            self._active_skill = None
            self._injected_skill_schemas = []
            settings = load_settings()

            blacklist = settings.get("tool_blacklist") or {}
            blacklisted_tools = set(blacklist.get("tools") or [])
            blacklisted_skills = set(blacklist.get("skills") or [])

            for server in self.servers:
                if self.active_servers[server.name]:
                    tools = await server.list_tools()
                    for tool in tools:
                        self._tool_schemas.append(tool.to_openai_schema())
                        self._tool_server_map[tool.name] = server
                        self._tool_timeout_map[tool.name] = (
                            server.timeout if server.timeout is not None else DEFAULT_TOOL_TIMEOUT
                        )

            def timestamp_to_datetime(unix_time):
                return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %H:%M:%S")
            timestamp = time.time()
            timestring = timestamp_to_datetime(timestamp)

            identity = "You are a helpful assistant. "

            skill_index = self._skills_manager.get_index()
            if blacklisted_skills:
                skill_index = [s for s in skill_index if s["name"] not in blacklisted_skills]
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

            # Load optional workspace guide
            mission_brief = ""
            try:
                plan_path = Path(settings.get("file_root") or ".") / ".agent" / "workspace.md"
                if plan_path.exists():
                    mission_brief = "\n\n### Workspace Guide\n\n" + plan_path.read_text(encoding="utf-8").strip()
                    logging.info("Loaded workspace guide from %s", plan_path)
                else:
                    logging.info("Optional workspace guide not loaded, %s doesn't exist", plan_path)
            except Exception as e:
                logging.warning("Could not load workspace.md: %s", e)

            task_context = ""
            if self.task_manager:
                task_context = (
                    "\n\n## Tasks\n"
                    "Tasks (and optional steps) appear in <tasks> before each message.\n"
                    "Rules you MUST follow:\n"
                    "- For any multi-step request (3+ actions): ALWAYS call plan_tasks first.\n"
                    "- ALWAYS call update_task(id, 'in_progress') before starting each task or step.\n"
                    "- ALWAYS call update_task(id, 'done') when each task or step finishes.\n"
                    "- Keep only ONE task or step in_progress at a time.\n"
                    "- Completed tasks stay visible while the plan is in progress; do not remove "
                    "them yourself. The whole plan clears automatically once every task/step is done.\n"
                    # "- Use steps (via plan_tasks 'steps' field or add_step) for tasks with 3+ sub-actions.\n"
                    # "- plan_tasks, add_task, and add_step require user approval. update_task runs silently."
                )

            file_context = ""
            if self.file_manager:
                file_context = (
                    "\n\n## Files\n"
                    "list_files runs silently, needs to be re-used to check sub-directories. "
                    "read_file shows numbered lines (N | line) — for targeted edits use edit_file, copying the exact text to change into 'find' (no line numbers) and the new text into 'replace'; 'find' must match exactly once. "
                    "For large changes that rewrite most of a file, don't chain many edit_file calls — rewrite the whole file in one bash command (cat > file <<'EOF' … EOF). "
                    "read_file, edit_file, and bash require approval. "
                    "bash: each command is a separate, stateless WSL shell starting at base_dir — "
                    "chain dependent steps with && in a single command."
                )

            mcp_has_search = "search_memory" in self._tool_server_map

            memory_tool_context = ""
            if self.memory_manager:
                search_note = (
                    "\nsearch_memory uses the connected MCP server's retrieval."
                    if mcp_has_search
                    else "\nsearch_memory uses BM25 keyword retrieval locally."
                )
                memory_tool_context = (
                    "\n\n## Memory Tools\n"
                    "Pinned facts injected above show IDs — use those IDs directly with update_fact or forget_fact. "
                    "search_memory runs silently — use it to search episodes and for facts beyond the pinned set. "
                    "remember_fact queues a fact for the user to review later (no inline prompt) — call it freely for durable facts. "
                    "update_fact and forget_fact persist across sessions and require user approval."
                    + search_note
                )

            memory_context = ""
            if self.load_episodes and self.episodic_store:
                n = settings.get("memory_episodes", 3)
                episodes = await asyncio.to_thread(self.episodic_store.get_recent, n)
                if episodes:
                    lines = []
                    for ep in episodes:
                        date_str = datetime.datetime.fromtimestamp(ep["created_at"]).strftime("%Y-%m-%d")
                        lines.append(f"\n<{date_str}> {ep['source'].title()}: {ep['summary']}\n")
                    memory_context = "\n\n[Memory]:\n" + "\n".join(lines)

            key_facts_context = ""
            if self.episodic_store:
                n_facts = settings.get("memory_key_facts", 20)
                facts = await asyncio.to_thread(self.episodic_store.get_pinned_facts, n_facts)
                if facts:
                    fact_lines = [f"#{f['id']} {f['fact']}" for f in facts]
                    key_facts_context = "\n\n[Pinned Facts]:\n" + "\n".join(fact_lines)

            playbook_hint = ""
            if self.playbook_manager:
                playbook_hint = (
                    "\n\n## Playbook\n"
                    "Call search_playbook(query) to retrieve relevant past procedures before planning."
                )

            system_content = (
                "--- SYSTEM INSTRUCTIONS ---\n\n"
                "# INSTRUCTIONS\n\n"
                f"{identity}"
                f"\nCurrent date and time : {timestring}\n\n"
                "\nUse tools when they help; reply directly when they don't."
                "\nWrite a short and concise one sentence description of what you're trying to accomplish before calling tools."
                + task_context
                + playbook_hint
                + file_context
                + memory_tool_context
                + skills_prompt
                + mission_brief
                + "\n\n# CONTEXT AND MEMORY\n\n"
                + memory_context
                + key_facts_context
                + "\n\n--- SYSTEM INSTRUCTION END ---"
            )

            with open("system_prompt_log.md", "w", encoding="utf-8") as f:
                f.write(system_content)

            if preserve_history and len(self.messages) > 1:
                # Refresh the system prompt in place (e.g. after approving/pinning
                # a fact mid-session) without discarding the live conversation.
                self.messages = [{"role": "user", "content": system_content}] + self.messages[1:]
            else:
                self.messages = [{"role": "user", "content": system_content}]

            if skill_index:
                self._tool_schemas.append(self._skills_manager.tool_schema)

            if self.file_manager:
                for entry in self.file_manager.tool_entries:
                    self._tool_schemas.append(entry["schema"])

            if self.task_manager:
                for schema in self.task_manager.tool_schemas:
                    self._tool_schemas.append(schema)

            if self.memory_manager:
                memory_schemas = self.memory_manager.tool_schemas
                if mcp_has_search:
                    memory_schemas = [
                        s for s in memory_schemas
                        if s["function"]["name"] != "search_memory"
                    ]
                for schema in memory_schemas:
                    self._tool_schemas.append(schema)

            if self.playbook_manager:
                for schema in self.playbook_manager.tool_schemas:
                    self._tool_schemas.append(schema)

            if blacklisted_tools:
                self._tool_schemas = [
                    s for s in self._tool_schemas
                    if s.get("function", {}).get("name") not in blacklisted_tools
                ]

            logging.info("Registered %d tool(s) and %d skill(s).", len(self._tool_schemas), len(skill_index))
        except Exception as e:
            error_msg = f"Error building system messages: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)

    def _inject_task_block(self, messages: list[dict], hint: str = "", extra: str = "") -> list[dict]:
        if self.task_manager is None and not hint and not extra:
            return messages
        block = self.task_manager.render_block() if self.task_manager else ""
        prefix_parts = []
        if extra:
            prefix_parts.append(extra)
        if hint:
            prefix_parts.append(hint)
        if block:
            prefix_parts.append("(System info - Here's the list of tasks you're working on, never talk about it to the user, use tools to keep it up-to-date([ ]: pending, [~]: in_progress, [x]: done):\n "+block+")")
        elif self.task_manager is not None and not hint:
            prefix_parts.append("(System info - No task list created. Call plan_tasks if the request needs a multi-step plan.)")
        if not prefix_parts:
            return messages
        prefix = "\n\n".join(prefix_parts)
        with open("prefix_log.md", "w", encoding="utf-8") as f:
            f.write(prefix)
        last = messages[-1]
        content = last["content"]
        if isinstance(content, str):
            new_content = prefix + "\n\n" + content
        else:
            new_content = [{"type": "text", "text": prefix}] + list(content)
        return messages[:-1] + [{**last, "content": new_content}]

    def _update_fact_window(self, user_text: str) -> None:
        """Decay the existing window one step, then merge in fresh approved-fact
        hits for this turn's user input (TTL reset on hit). Episodes excluded."""
        for entry in self._fact_window:
            entry["ttl"] -= 1
        self._fact_window = [e for e in self._fact_window if e["ttl"] > 0]
        if not (self.episodic_store and user_text.strip()):
            return
        try:
            results = self.episodic_store.search_facts(user_text, limit=self._retrieval_k)
        except Exception as e:
            logging.warning("Fact retrieval failed: %s", e)
            return
        by_id = {e["id"]: e for e in self._fact_window}
        for r in results:
            if r.get("type") != "fact":
                continue
            if r["id"] in by_id:
                by_id[r["id"]]["ttl"] = self._retrieval_ttl
            else:
                entry = {"id": r["id"], "fact": r["text"], "ttl": self._retrieval_ttl}
                self._fact_window.append(entry)
                by_id[r["id"]] = entry

    def _render_fact_window(self) -> str:
        if not self._fact_window:
            return ""
        lines = [f"#{e['id']} {e['fact']}" for e in self._fact_window]
        return "(System info - [Relevant Facts] retrieved from memory:\n" + "\n".join(lines) + ")"

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

        procedure_block = ""
        # A planner pass produces a fresh plan, so the injected-procedure set is
        # reset here (plan-scoped, not turn-scoped): the ids persist across the
        # later turns that execute the same plan so reinforce/penalize can still
        # fire when a multi-turn plan completes or aborts.
        self._injected_procedures = []
        if self.playbook_manager:
            matches = self.playbook_manager.search(text, top_n=3)
            if matches:
                self._injected_procedures = [
                    {"id": p["id"], "action": p.get("action", "")} for p in matches
                ]
                lines = [f"- {p['pattern']} → {p['action']}" for p in matches]
                procedure_block = (
                    "[Relevant procedures from memory — consider these when building the task plan]:\n"
                    + "\n".join(lines)
                    + "\n\n"
                )

        planner_messages = [
            {
                "role": "user",
                "content": (
                    f"{procedure_block}"
                    "You are a task planner. Your only job is to call the plan_tasks tool "
                    "with an ordered list of tasks for the following request. "
                    "Each task may have an optional 'steps' list for sub-actions. "
                    "Do nothing else.\n\n"
                    f"Request: {text}"
                ),
            }
        ]

        tool_calls = None
        for event in self.llm_client.stream_response(
            planner_messages,
            tools=[plan_schema],
            tool_choice={"type": "function", "function": {"name": "plan_tasks"}},
        ):
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

    async def _run_playbook_prompt(self):
        """Fire a non-streaming LLM call offering record_procedure after all tasks complete."""
        if not self.playbook_manager:
            return

        if not self.playbook_manager.tool_schemas:
            return
        record_schema = self.playbook_manager.tool_schemas[0]
        prompt_messages = self.messages + [{
            "role": "user",
            "content": (
                "[auto] All tasks completed. If this workflow is likely to recur and is "
                "worth reusing, call record_procedure(pattern, action) now.\n"
                "pattern = a keyword-rich trigger for this kind of request, phrased the way a "
                "user would ask it — include the key nouns, tools, file types, and verbs (and "
                "synonyms) so future searches can find it. Keep the domain keywords; drop only "
                "one-off specifics (exact filenames, values).\n"
                "action = the generalized step-by-step recipe.\n"
                "Skip and call nothing if this was a one-off task unlikely to repeat."
            ),
        }]

        try:
            response = await asyncio.to_thread(
                self.llm_client.get_response, prompt_messages, [record_schema]
            )
        except Exception as e:
            logging.warning("Playbook auto-prompt failed: %s", e)
            return

        tool_calls = response.get("tool_calls")
        if not tool_calls:
            return

        for tc in tool_calls:
            if tc.get("function", {}).get("name") != "record_procedure":
                continue
            tool_arg = tc["function"].get("arguments", "{}")

            self.tool_call_detected = True
            yield "record_procedure", tool_arg
            if self.allow_tool_action != "always":
                self.allow_tool_event.clear()
                await self.allow_tool_event.wait()

            if self.allow_tool_action == "deny":
                self.tool_call_detected = False
                self.allow_tool_action = None
                denial_text = "Playbook recording denied by user."
                yield f"Tool call deniedo|o{tool_arg}o|o{denial_text}o|oFalse"
                return

            if self.allow_tool_action == "allow":
                self.allow_tool_action = None

            self.tool_call_detected = False
            try:
                args = json.loads(tool_arg or "{}")
            except json.JSONDecodeError:
                args = {}
            result_text = self.playbook_manager.execute("record_procedure", args)
            yield f"record_procedureo|o{tool_arg}o|o{result_text}o|oFalse"

    def _maybe_arm_facts_nudge(self) -> None:
        """At a completed turn: arm the nudge if enough user turns have passed and
        the playbook prompt did not already fire this turn. Counter is NOT reset
        here — it resets only when the nudge is consumed, so a session that ends
        right after arming re-arms next session instead of losing the count."""
        if (
            self._facts_nudge_counter >= self._facts_nudge_threshold
            and not self._playbook_fired_this_turn
        ):
            self._pending_facts_nudge = True

    def _consume_facts_nudge(self) -> str:
        """Return the nudge text if armed (and reset), else empty string."""
        if not self._pending_facts_nudge:
            return ""
        self._pending_facts_nudge = False
        self._facts_nudge_counter = 0
        return (
            "[Memory] Several turns have passed. If this conversation surfaced durable "
            "facts worth keeping (user preferences, project constants, decisions), call "
            "remember_fact for each now. Skip if nothing is worth persisting."
        )

    def persist_nudge_state(self) -> None:
        """Persist the user-turn counter across sessions (short-session fix)."""
        try:
            from mcp_chatbot.settings import load_settings, save_settings
            s = load_settings()
            s["facts_nudge_progress"] = self._facts_nudge_counter
            save_settings(s)
        except Exception as e:
            logging.warning("Could not persist nudge state: %s", e)

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

        if self.memory_manager:
            _memory_names = (
                self.memory_manager.tool_names - {"search_memory"}
                if "search_memory" in self._tool_server_map
                else self.memory_manager.tool_names
            )
            if name in _memory_names:
                return self.memory_manager.execute(name, arguments)

        if self.playbook_manager and name in self.playbook_manager.tool_names:
            if name == "search_playbook":
                query = (arguments.get("query") or "").strip()
                if not query:
                    return "Error: 'query' is required."
                # Single query: full dicts render the result string; the buffer
                # keeps only {id, action} to match the _injected_procedures
                # contract (line 128) the planner path also follows.
                results = self.playbook_manager.search(query)
                seen = {p["id"] for p in self._retrieved_procedures}
                self._retrieved_procedures.extend(
                    {"id": p["id"], "action": p.get("action", "")}
                    for p in results if p["id"] not in seen
                )
                return self.playbook_manager.render_results(results)
            return self.playbook_manager.execute(name, arguments)

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

    def _attribution_weights(self) -> dict[int, float]:
        """Estimate which injected procedure the model actually followed and
        return per-id weights in (0, 1].

        The planner shows the top-N candidates but the model usually leans on one.
        We compare each procedure's action text against the plan that actually ran
        (token-Jaccard): the best match above ``_PROC_ATTRIB_THRESHOLD`` is treated
        as the one used (full weight 1.0) and the rest get ``_PROC_RESIDUAL_FRAC``
        — non-zero so a procedure that is repeatedly injected-but-never-matched
        still slowly decays. With no plan text or no clear match we fall back to a
        rank prior (``1/(rank+1)``): better-ranked retrievals get more weight."""
        procs = self._injected_procedures
        if not procs:
            return {}
        plan_text = self.task_manager.plan_text() if self.task_manager else ""
        if not isinstance(plan_text, str):
            plan_text = ""
        plan_tokens = set(re.findall(r"\w+", plan_text.lower()))

        best_idx, best_score = -1, 0.0
        if plan_tokens:
            for i, p in enumerate(procs):
                a_tokens = set(re.findall(r"\w+", (p.get("action") or "").lower()))
                if not a_tokens:
                    continue
                union = plan_tokens | a_tokens
                score = len(plan_tokens & a_tokens) / len(union) if union else 0.0
                if score > best_score:
                    best_idx, best_score = i, score

        if best_idx >= 0 and best_score >= _PROC_ATTRIB_THRESHOLD:
            return {
                p["id"]: (1.0 if i == best_idx else _PROC_RESIDUAL_FRAC)
                for i, p in enumerate(procs)
            }
        return {p["id"]: 1.0 / (i + 1) for i, p in enumerate(procs)}

    def _penalize_injected_procedures(self) -> None:
        """Penalize the procedures injected into the current plan after a
        loop-safety abort, then clear the set so a later unrelated abort cannot
        re-penalize the same ids. Weighted by attribution so the procedure the
        model actually followed takes the brunt, not every shown candidate."""
        if self._injected_procedures and self.playbook_manager:
            self.playbook_manager.penalize(self._attribution_weights())
        self._injected_procedures = []

    def _reinforce_injected_procedures(self) -> None:
        """Reinforce the procedures injected into the current plan when it
        completes, then clear the set. Gated by ``playbook_reinforce`` so the
        +0.05 bump can be disabled if it double-counts with the record dedup
        bump. Attribution-weighted like the penalty path."""
        if self.playbook_reinforce and self._injected_procedures and self.playbook_manager:
            self.playbook_manager.reinforce(self._attribution_weights())
        self._injected_procedures = []

    async def _synthesize_on_timeout(self) -> str:
        """Best-effort final answer when the wall-clock cap fires. Runs a
        non-streaming synthesis call in a thread bounded by
        ``TIMEOUT_SYNTHESIS_BUDGET`` so a slow/hung backend (the likely cause of
        the timeout) cannot hang the abort path too. Returns the synthesized text,
        or "" on timeout/error so the caller can fall back to a plain message."""
        synthesis_msgs = self.messages + [{
            "role": "user",
            "content": (
                "(System: time budget exceeded. Stop calling tools. "
                "Summarize what was accomplished so far and give a best-effort "
                "final answer based on the work done.)"
            ),
        }]
        try:
            msg = await asyncio.wait_for(
                asyncio.to_thread(self.llm_client.get_response, synthesis_msgs, None),
                timeout=TIMEOUT_SYNTHESIS_BUDGET,
            )
            return (msg.get("content") or "").strip()
        except Exception as e:
            logging.warning("Timeout synthesis failed: %s", e)
            return ""

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

        _user_text = user_input if isinstance(user_input, str) else " ".join(
            p.get("text", "") for p in user_input
            if isinstance(p, dict) and p.get("type") == "text"
        )
        self._update_fact_window(_user_text)
        self._memory_block = self._render_fact_window()

        self._facts_nudge_counter += 1
        self._playbook_fired_this_turn = False
        self._consumed_nudge = self._consume_facts_nudge()

        self.interrupt_requested = False
        if self.allow_tool_action == "interrupt":
            self.allow_tool_action = None
            self.allow_tool_event.clear()

        if self._active_skill:
            for s in self._injected_skill_schemas:
                try:
                    self._tool_schemas.remove(s)
                except ValueError:
                    pass
            self._injected_skill_schemas = []
            self._active_skill = None

        parse_retries = 0  # tracks consecutive malformed-JSON tool call responses
        # Reset only the search buffer, not _injected_procedures: a plan in
        # progress from a prior turn (model paused to ask the user, then resumes)
        # must keep its tracked procedures so they are still scored on completion.
        self._retrieved_procedures = []
        messages_checkpoint = len(self.messages)
        _compressed_this_turn = False
        _turn_count = 0
        _fingerprints: list[str] = []
        _wall_start = time.time()
        _stale_count = 0
        _last_in_progress_hash: str | None = None

        while True:
            self.agent_loop = {"active": True, "state": ""}
            _turn_count += 1
            if self.interrupt_requested:
                self._abort_interrupt("")
                yield f"\n\n{STOPPED_MARKER}"
                return
            if _turn_count > MAX_AGENT_ITERATIONS:
                self._penalize_injected_procedures()
                synthesis_msgs = self.messages + [{
                    "role": "user",
                    "content": (
                        "(System: maximum iterations reached. "
                        "Summarize what was accomplished and give a best-effort final answer. "
                        "Do not call any more tools.)"
                    ),
                }]
                synthesis_text = ""
                for event in self.llm_client.stream_response(synthesis_msgs, tools=None):
                    if event["type"] == "content":
                        synthesis_text += event["data"]
                        yield event["data"]
                if synthesis_text:
                    self.messages.append({"role": "assistant", "content": synthesis_text})
                self.agent_loop = {"active": False, "state": ""}
                return
            if time.time() - _wall_start > self.agent_timeout:
                self._penalize_injected_procedures()
                self.agent_loop["state"] = "synthesizing"
                synthesis_text = await self._synthesize_on_timeout()
                if synthesis_text:
                    yield synthesis_text
                    self.messages.append({"role": "assistant", "content": synthesis_text})
                else:
                    yield (
                        f"Error: agent timed out after {self.agent_timeout:.0f} seconds "
                        "and could not synthesize a summary."
                    )
                self.agent_loop = {"active": False, "state": ""}
                return
            tools = self._tool_schemas or None
            with open("mcp_chatbot/log.json", "w") as f:
                json.dump(self.messages, f, indent=2)
            assistant_msg = ""
            tool_calls = None
            tool_names = {}
            tool_args = {}
            if self._consumed_nudge:
                _hint = (_hint + "\n\n" + self._consumed_nudge).strip() if _hint else self._consumed_nudge
                self._consumed_nudge = ""  # merge once; don't re-inject on loop iterations
            call_messages = self._inject_task_block(self.messages, hint=_hint, extra=self._memory_block)
            _hint = ""  # consume — subsequent iterations don't repeat hint

            if self.task_manager:
                ip_ids = self.task_manager.get_in_progress_ids()
                current_ip_hash = str(ip_ids)
                if ip_ids:
                    if current_ip_hash == _last_in_progress_hash:
                        _stale_count += 1
                        if _stale_count >= STALE_TASK_ITERATIONS:
                            ids_str = ", ".join(ip_ids)
                            stale_nudge = (
                                f"[System: task {ids_str} has been in_progress for several steps "
                                "without an update — update it or re-plan.]"
                            )
                            call_messages = call_messages + [{"role": "user", "content": stale_nudge}]
                            _stale_count = 0
                    else:
                        _stale_count = 0
                    _last_in_progress_hash = current_ip_hash
                else:
                    _stale_count = 0
                    _last_in_progress_hash = None

            with open("mcp_chatbot/logs_with_tasks.json", "w") as f:
                json.dump(call_messages, f, indent=2)
            breakdown = self.context_mgr.snapshot(call_messages, tools or [])
            self.token_usage["breakdown"] = breakdown
            if self.context_mgr.is_near_limit() and not _compressed_this_turn:
                logging.warning(
                    "Context at %.0f%% — compressing history", breakdown["pct"] * 100
                )
                self.agent_loop["state"] = "compressing"
                await self._compress_history()
                self.agent_loop["state"] = ""
                call_messages = self._inject_task_block(self.messages, hint="", extra=self._memory_block)
                _compressed_this_turn = True
            self.agent_loop["state"] = "streaming"
            truncated = False
            for event in self.llm_client.stream_response(call_messages, tools=tools):
                if event["type"] == "content":
                    if self.interrupt_requested:
                        break
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
                elif event["type"] == "truncated":
                    truncated = True
                elif event["type"] == "error":
                    assistant_msg += event["data"]
            if truncated:
                logging.warning(
                    "LLM completion hit max_output_tokens (finish_reason=length); "
                    "output truncated%s.", " mid tool-call" if tool_calls else ""
                )
                # Only surface to the user on a text answer; a truncated tool-call
                # falls through to the parse-retry guard, which handles the bad JSON.
                if not tool_calls:
                    assistant_msg += f"\n\n{TRUNCATED_MARKER}"
                    yield f"\n\n{TRUNCATED_MARKER}"
            logging.debug("tool_names: %s", tool_names)
            logging.debug("tool_args: %s", tool_args)
            self.agent_loop["state"] = ""

            if self.interrupt_requested:
                self._abort_interrupt(assistant_msg)
                yield f"\n\n{STOPPED_MARKER}"
                return

            # pop the "Use the above tool call result..." nudge from the previous iteration
            if (
                self.messages
                and self.messages[-1].get("content") == (
                    "(System instruction - If more tool calls are needed, call them now with no text. If all tasks are complete, give your final answer. Take the above tool call result into account.)"
                )
            ):
                self.messages.pop(-1)

            # validate tool call argument JSON before executing
            if tool_calls:
                parse_error: Exception | None = None
                for tc in tool_calls:
                    # Normalize empty/whitespace argument strings to "{}". An empty
                    # "arguments": "" is invalid in the OpenAI tool-call format;
                    # local backends (LM Studio / llama.cpp) reject the whole
                    # history with HTTP 500 when re-serializing such a turn, which
                    # poisons every subsequent call since the bad assistant message
                    # stays in self.messages. Rewrite it before it is persisted.
                    if not (tc["function"].get("arguments") or "").strip():
                        tc["function"]["arguments"] = "{}"
                    try:
                        json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError as e:
                        parse_error = e
                        break

                if parse_error is not None:
                    if parse_retries >= MAX_PARSE_RETRIES:
                        self._penalize_injected_procedures()
                        del self.messages[messages_checkpoint:]
                        yield (
                            f"Error: LLM produced invalid tool call JSON after "
                            f"{MAX_PARSE_RETRIES} retries. Aborting."
                        )
                        self.agent_loop = {"active": False, "state": ""}
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
                self.agent_loop = {"active": False, "state": ""}
                self._maybe_arm_facts_nudge()
                with open("mcp_chatbot/log.json", "w") as f:
                    json.dump(self.messages, f, indent=2)
                break

            # Persist the assistant turn WITH its tool_calls. Required so the
            # following role:tool messages can be paired back to their call —
            # strict templates (e.g. Gemma) only render tool results by
            # forward-scanning from an assistant message that carries
            # tool_calls; without it the results are invisible to the model.
            # Also the OpenAI spec mandates this ordering.
            self.messages.append({
                "role": "assistant",
                "content": assistant_msg or "",
                "tool_calls": tool_calls,
            })

            for _idx, tool_call in enumerate(tool_calls):
                if self.interrupt_requested:
                    # Pair every not-yet-executed call in this round so the
                    # appended assistant{tool_calls} entry has no orphaned ids
                    # (strict templates / OpenAI spec require a result per call).
                    for _pending in tool_calls[_idx:]:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": _pending.get("id", _pending["function"]["name"]),
                            "content": "Tool call skipped — stopped by user.",
                        })
                    self._abort_interrupt("")
                    yield f"\n\n{STOPPED_MARKER}"
                    return
                tool_name = tool_call["function"]["name"]
                tool_arg = tool_call["function"]["arguments"]
                logging.info("Attempting Tool Call: %s", tool_name)

                is_safe_file_tool = (
                    self.file_manager is not None
                    and self.file_manager.is_safe(tool_name)
                )
                is_safe_task_tool = (
                    self.task_manager is not None
                    and tool_name in self.task_manager.safe_tool_names
                )
                is_safe_memory_tool = (
                    self.memory_manager is not None
                    and tool_name in self.memory_manager.safe_tool_names
                )
                is_safe_playbook_tool = (
                    self.playbook_manager is not None
                    and tool_name in self.playbook_manager.safe_tool_names
                )

                if not (is_safe_file_tool or is_safe_task_tool or is_safe_memory_tool or is_safe_playbook_tool):
                    self.tool_call_detected = True
                    yield tool_name, tool_arg
                    if self.allow_tool_action not in ("always", "interrupt") and not self.interrupt_requested:
                        self.allow_tool_event.clear()
                        await self.allow_tool_event.wait()

                    if self.interrupt_requested or self.allow_tool_action == "interrupt":
                        # Reuse the deny path so the already-appended
                        # assistant{tool_calls} entry gets a paired tool result,
                        # then abort the whole loop. Clear tool_call_detected
                        # BEFORE yielding the o|o string — handle_chat checks the
                        # tool_call_detected branch first and would otherwise try
                        # to unpack this string as a (name, args) tuple.
                        self.tool_call_detected = False
                        denial_text = (
                            f"\nTool call '{tool_name}' was stopped by the user.\n\n"
                            f"**Arguments:**\n```json\n{tool_arg}\n```"
                        )
                        yield f"Tool call deniedo|o{tool_arg}o|o{denial_text}o|oFalse"
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", tool_name),
                            "content": denial_text,
                        })
                        self.messages.append({
                            "role": "user",
                            "content": "Reason for stopping: Stopped by user.",
                        })
                        self._abort_interrupt("")
                        yield f"\n\n{STOPPED_MARKER}"
                        return

                    if self.allow_tool_action == "deny":
                        self.tool_call_detected = False
                        self.allow_tool_action = None
                        with_reason = f" Reason for denying tool call: {self.deny_reason}"
                        denial_text = f"\nTool call '{tool_name}' was denied by user.\n\n**Arguments:**\n```json\n{tool_arg}\n```"
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

                    # Gate was shown for this non-safe call; retract it now,
                    # before the slow execution, so it doesn't spin/linger.
                    yield GATE_CLOSE_MARKER

                result_text = await self._execute_tool_call(tool_call)

                # A fresh plan supersedes any prior attribution tracking: the
                # procedures the model looked up this turn (buffer) become the
                # plan-scoped set scored on completion/abort. Empty buffer (planned
                # without searching) correctly clears stale ids. Note: a plan_tasks
                # issued after the planner pass already seeded _injected_procedures
                # will overwrite it — acceptable, the model replaced that plan.
                if tool_name == "plan_tasks":
                    self._injected_procedures = self._retrieved_procedures
                    self._retrieved_procedures = []

                # Include tool_arg so distinct calls (e.g. search_playbook with
                # different queries that all return "no match") don't collide as
                # a false loop. A genuine stuck loop repeats name+args+result.
                _fp = hashlib.md5(f"{tool_name}:{tool_arg}:{result_text[:120]}".encode()).hexdigest()
                _fingerprints.append(_fp)
                if len(_fingerprints) >= 3 and len(set(_fingerprints[-3:])) == 1:
                    self._penalize_injected_procedures()
                    yield "Error: agent stuck in loop (same tool + result 3 consecutive times). Aborting."
                    self.agent_loop = {"active": False, "state": ""}
                    return

                yield f"{tool_name}o|o{json.dumps(tool_call)}o|o{result_text}o|o{str(is_safe_file_tool or is_safe_task_tool or is_safe_memory_tool or is_safe_playbook_tool)}"

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", tool_name),
                    "content": f"**Result of your '{tool_name}' tool call:**\n\n{result_text}",
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
                        nudge_func = "\nDo not make any comment and call the appropriate tool immediately !"
                        nudge = f"(System instruction - Use the tool call result above to complete the user's request.{nudge_func if fn_schemas else ''})"
                        self.messages.append({"role": "user", "content": nudge})
                else:
                    nudge = (
                        # "(System instruction - If more tool calls are needed, call them now with no text. "
                        # "If all tasks are complete, give your final answer. "
                        # "(System instruction - If more tool calls are needed, call them now with no text. "
                        "(System instruction - Take the above tool call result into account.)"
                    )
                    self.messages.append({"role": "user", "content": nudge})
                if (
                    tool_name == "update_task"
                    and self.task_manager is not None
                    and self.task_manager._last_all_done
                ):
                    self._playbook_fired_this_turn = True
                    self._reinforce_injected_procedures()
                    self.task_manager._last_all_done = False
                    async for event in self._run_playbook_prompt():
                        yield event
                    self.task_manager.clear_plan()
            self.tool_call_detected = False

    async def set_allow_tool_action(self, action: str, reason: str = "") -> None:
        """Called by the UI (e.g. Gradio) to approve or deny a pending tool call."""
        self.deny_reason = reason
        self.allow_tool_action = action
        self.allow_tool_event.set()
        logging.debug("set_allow_tool_action called with '%s'", action)

    async def request_interrupt(self) -> None:
        """Request a cooperative abort of the current agent turn. No-op when no
        turn is active (so a stray click cannot leak a sentinel into the next
        turn's approval gate). Sets the gate event so a pending approval wait
        unblocks and is handled as an interrupt rather than allow/deny."""
        if not self.agent_loop.get("active"):
            return
        self.interrupt_requested = True
        # Don't clobber auto-tool mode: in "always" mode there is no pending gate
        # wait to unblock, and overwriting it would silently disable auto-tool
        # after the turn aborts (teardown only resets the "interrupt" sentinel).
        # The interrupt checkpoints all fire on interrupt_requested regardless.
        if self.allow_tool_action != "always":
            self.allow_tool_action = "interrupt"
        self.allow_tool_event.set()
        logging.info("Interrupt requested by user.")

    def _abort_interrupt(self, assistant_msg: str) -> None:
        """Clean teardown shared by every interrupt checkpoint: penalize any
        injected procedures, persist the partial answer (or a standalone marker)
        so the turn closes with valid history, then reset loop + flags."""
        self._penalize_injected_procedures()
        marker = STOPPED_MARKER
        if assistant_msg.strip():
            self.messages.append({
                "role": "assistant",
                "content": assistant_msg + "\n\n" + marker,
            })
        else:
            self.messages.append({"role": "assistant", "content": marker})
        self.agent_loop = {"active": False, "state": ""}
        self.interrupt_requested = False
        self.tool_call_detected = False
        if self.allow_tool_action == "interrupt":
            self.allow_tool_action = None
            self.allow_tool_event.clear()

    async def save_episode(self) -> None:
        """Summarize the current session and persist it as an episode."""
        if self.episodic_store is None:
            return
        history = self.messages[1:]
        convo_length = [m for m in history]
        if len(convo_length) < 10:
            logging.info("Short session - Session episode not saved.")
            return
        def _summarize_and_save() -> bool:
            summary, facts = self._summarize_with_facts(history)
            if not summary:
                return False
            self.episodic_store.add_episode(self._session_id, summary, "agent")
            for fact in facts:
                try:
                    if self.episodic_store.find_similar_fact(fact) is None:
                        self.episodic_store.remember_fact(fact, source="agent", status="pending")
                except Exception as e:
                    logging.warning("Co-extraction fact write failed: %s", e)
            return True

        logging.info("Saving session episode — summarizing history...")
        try:
            saved = await asyncio.to_thread(_summarize_and_save)
            if saved:
                logging.info("Session episode saved.")
        except Exception as e:
            logging.warning("Could not save session episode: %s", e)

    def _summarize(self, messages: list[dict]) -> str | None:
        """Summarize conversation history using the LLM."""
        def _format_message(message):
            content = message.get('content', '')
            if isinstance(content, str):
                return f"{message['role'].upper()}: {content}"
            elif isinstance(content, list) and content:
                text = content[0].get('text', '')
                return f"{message['role'].upper()}: {text} (user submitted an image)"
            return None
        text = "\n".join(
            _format_message(m)
            for m in messages
            if m.get("content")
        )
        if not text:
            return None
        prompt = [{"role": "user", "content": (
            "Summarize the following conversation concisely. "
            "Preserve key decisions, facts, tool results, and context "
            "needed to continue the conversation.\n\n"
            f"<conversation>\n{text}\n</conversation>"
        )}]
        try:
            response = self.llm_client.get_response(prompt)
            return response.get("content", "").strip() or None
        except Exception as e:
            logging.warning("Summarization failed: %s", e)
            return None

    def _summarize_with_facts(self, messages: list[dict]) -> tuple[str | None, list[str]]:
        """Structured shutdown summary: one call yields both the episode summary
        and candidate durable facts. Uses response_format on the local path; on
        any parse failure (e.g. a remote backend that ignores response_format)
        falls back to treating the whole response as the summary with no facts.

        Separate from _summarize because _summarize is shared with
        _compress_history and must keep returning a plain string."""
        def _format_message(message):
            content = message.get("content", "")
            if isinstance(content, str):
                return f"{message['role'].upper()}: {content}"
            elif isinstance(content, list) and content:
                text = content[0].get("text", "")
                return f"{message['role'].upper()}: {text} (user submitted an image)"
            return None

        text = "\n".join(m for m in (_format_message(x) for x in messages if x.get("content")) if m)
        if not text:
            return None, []
        prompt = [{"role": "user", "content": (
            "Summarize the following conversation concisely, preserving key decisions, "
            "facts, tool results, and context needed to continue. Then extract any "
            "durable facts worth remembering long-term (user preferences, project "
            "preferences, info about user) as short self-contained statements; use an empty "
            "list if there are none.\n\n"
            f"<conversation>\n{text}\n</conversation>"
        )}]
        fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": "episode_result",
                "strict": True,
                "schema": EpisodeResult.model_json_schema(),
            },
        }
        try:
            response = self.llm_client.get_response(prompt, response_format=fmt)
            content = (response.get("content") or "").strip()
        except Exception as e:
            logging.warning("Co-extraction call failed: %s", e)
            return None, []
        if not content:
            return None, []
        try:
            data = EpisodeResult.model_validate_json(content)
            return (data.summary.strip() or None), [f.strip() for f in data.facts if f.strip()]
        except Exception:
            # Backend ignored response_format or returned prose — keep the summary.
            return content, []

    async def _compress_history(self) -> None:
        """Compress old conversation by summarizing and keeping only recent turns."""
        history = self.messages[1:]
        user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
        if len(user_indices) <= COMPRESSION_KEEP_TURNS:
            return
        cut = user_indices[-COMPRESSION_KEEP_TURNS]
        to_compress = history[:cut]
        to_keep = history[cut:]
        try:
            summary = await asyncio.to_thread(self._summarize, to_compress)
        except Exception as e:
            logging.warning("Could not compress messages into summary.: %s", e)
        if summary is None:
            return
        self.messages = (
            [self.messages[0]]
            + [{"role": "user", "content": f"[Summary]: {summary}"}]
            + to_keep
        )
        logging.info("Compressed %d messages into summary.", len(to_compress))
        if self.episodic_store:
            await asyncio.to_thread(
                self.episodic_store.add_episode, self._session_id, summary, "agent"
            )

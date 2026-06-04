from __future__ import annotations

import logging

from mcp_chatbot.memory.store import EpisodicStore


class PlaybookManager:
    def __init__(self, store: EpisodicStore) -> None:
        self._store = store
        logging.info("PlaybookManager initialized (store-backed).")

    def search(self, query: str, top_n: int = 5) -> list[dict]:
        return self._store.search_procedures(query, limit=top_n)

    def get_top_n(self, n: int) -> list[dict]:
        return self._store.get_top_procedures(n)

    # Base deltas; scaled per-procedure by the caller's attribution weights so the
    # procedure the model actually followed moves more than the other candidates.
    _PENALTY = -0.1
    _REWARD = 0.05

    def penalize(self, weights: dict[int, float]) -> None:
        """Reduce confidence for procedures that preceded a failed/looping run.

        ``weights`` maps procedure id -> attribution weight in (0, 1]; each id is
        adjusted by ``_PENALTY * weight``."""
        self._apply_weighted(weights, self._PENALTY)

    def reinforce(self, weights: dict[int, float]) -> None:
        """Boost confidence for procedures that preceded a successful run.

        ``weights`` maps procedure id -> attribution weight in (0, 1]; each id is
        adjusted by ``_REWARD * weight``."""
        self._apply_weighted(weights, self._REWARD)

    def _apply_weighted(self, weights: dict[int, float], base: float) -> None:
        for pid, w in weights.items():
            if w:
                self._store.adjust_confidence([pid], base * w)

    def clear_all(self) -> None:
        self._store.clear_procedures()

    @property
    def tool_names(self) -> set[str]:
        return {"record_procedure", "search_playbook"}

    @property
    def safe_tool_names(self) -> set[str]:
        return {"search_playbook"}

    def execute(self, tool_name: str, arguments: dict) -> str:
        try:
            if tool_name == "record_procedure":
                pattern = arguments.get("pattern", "").strip()
                action = arguments.get("action", "").strip()
                if not pattern:
                    return "Error: 'pattern' is required."
                if not action:
                    return "Error: 'action' is required."
                proc_id = self._store.record_procedure(pattern, action, source="task_success")
                return f"Procedure #{proc_id} recorded: {pattern}"
            if tool_name == "search_playbook":
                query = arguments.get("query", "").strip()
                if not query:
                    return "Error: 'query' is required."
                results = self._store.search_procedures(query, limit=5)
                if not results:
                    return "No matching procedures found."
                lines = [f"#{p['id']} [{p['source']}] {p['pattern']} → {p['action']}" for p in results]
                return "\n".join(lines)
            return f"Error: unknown playbook tool '{tool_name}'."
        except Exception as e:
            logging.error("PlaybookManager.%s error: %s", tool_name, e)
            return f"Error: {e}"

    @property
    def tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "record_procedure",
                    "description": (
                        "Record a reusable procedure in the playbook. "
                        "Call after completing a multi-step task that is likely to recur. "
                        "The pattern is the search key used to retrieve this procedure later, "
                        "so make it keyword-rich; the action is the reusable recipe."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": (
                                    "A keyword-rich trigger describing the kind of request this "
                                    "applies to, phrased the way a user would ask it. Include the "
                                    "concrete nouns, tools, file types, and action verbs involved "
                                    "(and common synonyms). This is the search key, so prefer "
                                    "specific, varied keywords over short or abstract phrasing. "
                                    "Generalize away one-off specifics (exact filenames, values) "
                                    "but keep the domain vocabulary. "
                                    "E.g. 'deploy / ship a docker container image to production, "
                                    "push registry, restart service'."
                                ),
                            },
                            "action": {
                                "type": "string",
                                "description": (
                                    "The reusable step-by-step recipe. Generalize over-specific "
                                    "values so it applies next time; concise but complete."
                                ),
                            },
                        },
                        "required": ["pattern", "action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_playbook",
                    "description": (
                        "Search for relevant past procedures by description or keyword. "
                        "Call before planning a complex or recurring task."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Description of the task or situation to find procedures for.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

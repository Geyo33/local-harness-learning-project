from __future__ import annotations

import datetime
import logging

from mcp_chatbot.memory.store import EpisodicStore


class MemoryManager:
    def __init__(self, store: EpisodicStore) -> None:
        self._store = store

    @property
    def tool_names(self) -> set[str]:
        return {"remember_fact", "update_fact", "forget_fact", "search_memory"}

    @property
    def safe_tool_names(self) -> set[str]:
        return {"search_memory"}

    def execute(self, tool_name: str, arguments: dict) -> str:
        try:
            if tool_name == "remember_fact":
                fact = arguments.get("fact", "").strip()
                if not fact:
                    return "Error: 'fact' is required."
                source = arguments.get("source", "agent")
                match = self._store.find_similar_fact(fact)
                if match:
                    return (
                        f"Similar fact already exists: #{match['id']} '{match['fact']}'. "
                        f"Call update_fact with fact_id={match['id']} and new_fact='<replacement text>, "
                        f"or forget_fact({match['id']}) if it is no longer relevant."
                    )
                fact_id = self._store.remember_fact(fact, source=source)
                return f"Remembered fact #{fact_id}: {fact}"

            if tool_name == "update_fact":
                fact_id = arguments.get("fact_id")
                new_fact = arguments.get("new_fact", "").strip()
                if fact_id is None or not new_fact:
                    return "Error: 'fact_id' and 'new_fact' are required."
                updated = self._store.update_fact(int(fact_id), new_fact)
                if updated:
                    return f"Updated fact #{fact_id}: {new_fact}"
                return f"Error: no fact with id={fact_id}."

            if tool_name == "forget_fact":
                fact_id = arguments.get("fact_id")
                if fact_id is None:
                    return "Error: 'fact_id' is required."
                deleted = self._store.forget_fact(int(fact_id))
                if deleted:
                    return f"Deleted fact #{fact_id}."
                return f"Error: no fact with id={fact_id}."

            if tool_name == "search_memory":
                query = arguments.get("query", "").strip()
                if not query:
                    return "Error: 'query' is required."
                results = self._store.search_memory(query)
                if not results:
                    return "No memory matches found."
                lines = []
                for r in results:
                    date_str = datetime.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d")
                    tag = f"[{r['type']}#{r['id']} {date_str} {r['source']}]"
                    lines.append(f"{tag} {r['text']}")
                return "\n".join(lines)

            return f"Error: unknown memory tool '{tool_name}'."
        except Exception as e:
            logging.error("MemoryManager.%s error: %s", tool_name, e)
            return f"Error: {e}"

    @property
    def tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "remember_fact",
                    "description": (
                        "Store a persistent fact in long-term memory. "
                        "Use for user preferences, project constants, or important knowledge "
                        "that should survive across sessions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact": {
                                "type": "string",
                                "description": "The fact to remember, as a self-contained statement.",
                            },
                            "source": {
                                "type": "string",
                                "enum": ["user", "agent"],
                                "description": "Who originated this fact. Use 'user' when the user explicitly stated it; use 'agent' for inferences or derived knowledge.",
                            },
                        },
                        "required": ["fact"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_fact",
                    "description": "Correct or update an existing stored fact by its id. Use when a fact is stale or wrong.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact_id": {
                                "type": "integer",
                                "description": "The id of the fact to update (from search_memory results).",
                            },
                            "new_fact": {
                                "type": "string",
                                "description": "Replacement text for the fact.",
                            },
                        },
                        "required": ["fact_id", "new_fact"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "forget_fact",
                    "description": "Permanently delete a stored fact by its id. Use when a fact is known to be wrong or no longer relevant.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fact_id": {
                                "type": "integer",
                                "description": "The id of the fact to delete (from search_memory results).",
                            }
                        },
                        "required": ["fact_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memory",
                    "description": (
                        "Search long-term memory for episodes and facts matching a query. "
                        "Returns ranked results. Use before answering questions about past sessions or stored knowledge."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Keywords or phrase to search for.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

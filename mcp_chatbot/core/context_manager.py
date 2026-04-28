from __future__ import annotations

import json


class ContextManager:
    def __init__(self, max_tokens: int, threshold: float = 0.70) -> None:
        self.max_tokens = max_tokens
        self.threshold = threshold
        self._breakdown: dict = {
            "system": 0, "tools": 0, "history": 0, "memory": 0,
            "reserve": max_tokens, "total": 0, "max": max_tokens, "pct": 0.0,
        }

    @staticmethod
    def _count(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _extract_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""

    def snapshot(self, messages: list, tool_schemas: list) -> dict:
        system = self._count(self._extract_text(messages[0].get("content"))) if messages else 0
        history = sum(
            self._count(self._extract_text(m["content"]))
            for m in messages[1:]
            if "content" in m and m["content"] is not None
        )
        tools = self._count(json.dumps(tool_schemas)) if tool_schemas else 0
        total = system + history + tools
        reserve = self.max_tokens - total
        pct = total / self.max_tokens if self.max_tokens else 0.0
        self._breakdown = {
            "system": system,
            "tools": tools,
            "history": history,
            "memory": 0,
            "reserve": reserve,
            "total": total,
            "max": self.max_tokens,
            "pct": pct,
        }
        return self._breakdown

    def is_near_limit(self) -> bool:
        return self._breakdown["pct"] >= self.threshold

    @property
    def breakdown(self) -> dict:
        return self._breakdown

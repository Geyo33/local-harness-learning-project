from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self, frontend: str, temp: float) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.groq_api_key = os.getenv("LLM_API_KEY")
        self.local_model = True
        self.api_key = "local" if self.local_model == True else self.groq_api_key
        self.model = "model-identifier" if self.local_model == True else "meta-llama/llama-4-scout-17b-16e-instruct"
        self.temp = temp
        self.frontend = frontend

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file."""
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key

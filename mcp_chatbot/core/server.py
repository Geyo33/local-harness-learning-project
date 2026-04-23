from __future__ import annotations

import asyncio
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self,
        name: str,
        description: str,
        inputSchema: dict[str, Any],
        title: str | None = None,
    ) -> None:
        self.name: str = name
        self.title: str | None = title
        self.description: str = description
        self.inputSchema: dict[str, Any] = inputSchema

    def to_openai_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function tool object."""
        description = self.description or ""
        if self.title:
            description = f"{self.title}: {description}"
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": self.inputSchema,
            },
        }


class Server:
    """
    Manages MCP server connections and tool execution.

    The key design constraint: anyio cancel scopes (used internally by
    stdio_client / ClientSession) MUST be entered and exited within the
    same OS task.  To satisfy this we run the entire connection lifetime
    inside a single, long-lived background asyncio.Task (_lifecycle_task).

    External callers use:
      - await server.initialize()  -> starts the background task and waits
                                      until the session is ready
      - await server.cleanup()     -> signals shutdown and waits for the
                                      background task to finish cleanly
    """

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.timeout: float | None = config.get("timeout", None)  # seconds; None = use session default
        self.session: ClientSession | None = None

        self._ready_event: asyncio.Event = asyncio.Event()
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._init_error: Exception | None = None
        self._lifecycle_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """
        Spawn the background lifecycle task and block until the server
        session is fully ready (or raises if initialisation failed).
        """
        self._lifecycle_task = asyncio.create_task(
            self._run_lifecycle(), name=f"mcp-server-{self.name}"
        )
        await self._ready_event.wait()
        if self._init_error is not None:
            raise self._init_error

    async def cleanup(self) -> None:
        """
        Signal the background task to shut down and wait for it to finish.
        The exit_stack.aclose() call happens inside the background task,
        in the same task where the contexts were entered — no cancel-scope
        cross-task violation.
        """
        if self._lifecycle_task is None or self._lifecycle_task.done():
            return
        self._shutdown_event.set()
        try:
            await self._lifecycle_task
        except Exception as e:
            logging.error("Error waiting for lifecycle task of %s to finish: %s", self.name, e)

    async def _run_lifecycle(self) -> None:
        """
        Open the MCP connection and keep it alive until shutdown is
        requested.  Enter AND exit all async contexts in this single task
        so that anyio cancel scopes are always closed by the same task
        that created them.
        """
        exit_stack = AsyncExitStack()
        try:
            async with exit_stack:
                command = (
                    shutil.which("npx")
                    if self.config["command"] == "npx"
                    else self.config["command"]
                )
                if command is None:
                    raise ValueError("The command must be a valid string and cannot be None.")

                server_params = StdioServerParameters(
                    command=command,
                    args=self.config["args"],
                    env={**os.environ, **self.config["env"]} if self.config.get("env") else None,
                )

                stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
                read, write = stdio_transport
                session = await exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                self.session = session
                logging.info("Server %s initialised successfully.", self.name)
                self._ready_event.set()
                await self._shutdown_event.wait()
                logging.info("Shutdown signal received for server %s; closing.", self.name)

        except Exception as e:
            self._init_error = e
            logging.error("Error in lifecycle of server %s: %s", self.name, e)
        finally:
            self.session = None
            self._ready_event.set()
            logging.info("Cleanup complete for server %s.", self.name)

    async def list_tools(self) -> list[Tool]:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools: list[Tool] = []
        for item in tools_response:
            if item[0] == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema, tool.title)
                    for tool in item[1]
                )
        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
        timeout: float | None = None,
    ) -> Any:
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info("Executing %s...", tool_name)
                if timeout is not None:
                    result = await asyncio.wait_for(
                        self.session.call_tool(tool_name, arguments),
                        timeout=timeout,
                    )
                else:
                    result = await self.session.call_tool(tool_name, arguments)
                return result
            except asyncio.TimeoutError:
                raise  # don't retry on timeout — propagate immediately
            except Exception as e:
                attempt += 1
                logging.warning("Error executing tool: %s. Attempt %d of %d.", e, attempt, retries)
                if attempt < retries:
                    logging.info("Retrying in %s seconds...", delay)
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

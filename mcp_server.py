"""
TariffIQ MCP server.

Exposes the full LangGraph pipeline as a single MCP tool for Claude Desktop.
"""

from __future__ import annotations

import json
import logging
import os
import io
import sys
import contextlib
from pathlib import Path
from typing import Any

import anyio
from dotenv import load_dotenv
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


# Load .env once at startup (project root).
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _sanitize_proxy_env() -> None:
    """
    Prevent litellm/httpx startup crashes from malformed proxy env vars.
    Removes proxy vars that are empty strings or missing URL scheme.
    """
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    for key in proxy_keys:
        val = os.environ.get(key)
        if val is None:
            continue
        cleaned = val.strip()
        if not cleaned:
            os.environ.pop(key, None)
            continue
        if "://" not in cleaned:
            os.environ.pop(key, None)


_sanitize_proxy_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

server = Server("tariffiq")
_RUN_PIPELINE = None


def _get_run_pipeline():
    global _RUN_PIPELINE
    if _RUN_PIPELINE is None:
        # Lazy import so env sanitization above applies before litellm/httpx init.
        from agents.graph import run_pipeline
        _RUN_PIPELINE = run_pipeline
    return _RUN_PIPELINE


def _run_tariffiq_query_impl(user_query: str) -> str:
    """
    Run the full TariffIQ pipeline and return final synthesis output.

    Pipeline: query -> classification -> rate -> policy -> trade -> synthesis
    """
    query = (user_query or "").strip()
    if not query:
        return "Error: user_query is required."

    try:
        run_pipeline = _get_run_pipeline()
        # MCP stdio uses stdout for protocol frames; any stray stdout from
        # downstream libs would corrupt JSON-RPC and disconnect the server.
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_pipeline(query)
    except Exception as exc:
        logger.exception("run_tariffiq_query failed")
        return f"Error running TariffIQ pipeline: {exc}"

    final_response = result.get("final_response")
    if final_response:
        return str(final_response)

    # Fallback: provide a compact structured summary when synthesis is unavailable.
    fallback = {
        "message": "Pipeline completed but no final synthesis was produced.",
        "hitl_required": result.get("hitl_required"),
        "hitl_reason": result.get("hitl_reason"),
        "error": result.get("error"),
        "hts_code": result.get("hts_code"),
        "pipeline_confidence": result.get("pipeline_confidence"),
    }
    return json.dumps(fallback, ensure_ascii=True)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="run_tariffiq_query",
            description=(
                "Run the full TariffIQ agent pipeline "
                "(query -> classification -> rate -> policy -> trade -> synthesis)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "The user's natural-language tariff question.",
                    }
                },
                "required": ["user_query"],
                "additionalProperties": False,
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name != "run_tariffiq_query":
        raise ValueError(f"Unknown tool: {name}")

    user_query = str((arguments or {}).get("user_query", "")).strip()
    result_text = _run_tariffiq_query_impl(user_query)
    return [types.TextContent(type="text", text=result_text)]


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    # Required for Claude Desktop MCP integration.
    anyio.run(_main)


# Claude Desktop config snippet:
# {
#   "mcpServers": {
#     "tariffiq": {
#       "command": "python",
#       "args": ["/absolute/path/to/your/project/mcp_server.py"],
#       "env": {
#         "PYTHONPATH": "/absolute/path/to/your/project"
#       }
#     }
#   }
# }

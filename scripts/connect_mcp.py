"""Minimal MCP client for local testing against this project's MCP server.

Usage examples:
  python scripts/connect_mcp.py --list
  python scripts/connect_mcp.py --call query_knowledge_hub --arg query "What is RAG?" --arg top_k 3

This script launches the server as a subprocess using the stdio transport
and speaks JSON-RPC lines over stdin/stdout (same approach as tests/e2e/test_mcp_client.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def start_server() -> subprocess.Popen:
    env = dict(**__import__("os").environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def send_jsonrpc(
    proc: subprocess.Popen,
    messages: List[Dict[str, Any]],
    expected_responses: int,
    timeout: float = 15.0,
) -> List[Dict[str, Any]]:
    assert proc.stdin is not None
    assert proc.stdout is not None

    for msg in messages:
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    responses: List[Dict[str, Any]] = []
    stop_event = threading.Event()

    def _reader() -> None:
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if "id" in data and ("result" in data or "error" in data):
                responses.append(data)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    deadline = time.time() + timeout
    while len(responses) < expected_responses and time.time() < deadline:
        time.sleep(0.05)

    stop_event.set()
    return responses


def build_init_messages() -> List[Dict[str, Any]]:
    return [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "clientInfo": {"name": "scripts/connect_mcp.py", "version": "0.1"},
                "capabilities": {},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="Call tools/list and print tools")
    parser.add_argument("--call", type=str, help="Tool name to call")
    parser.add_argument("--arg", action="append", nargs=2, metavar=("KEY", "VALUE"), help="Tool argument as key value (can repeat)")
    args = parser.parse_args()

    proc = start_server()

    try:
        # Prepare initial init messages and optionally an action
        messages = build_init_messages()
        next_id = 2

        if args.list:
            messages.append({"jsonrpc": "2.0", "id": next_id, "method": "tools/list", "params": {}})
            expected = 2
        elif args.call:
            # Build arguments dict
            arguments: Dict[str, Any] = {}
            if args.arg:
                for k, v in args.arg:
                    # try to parse numbers, else keep string
                    try:
                        parsed = int(v)
                    except ValueError:
                        try:
                            parsed = float(v)
                        except ValueError:
                            parsed = v
                    arguments[k] = parsed

            messages.append(
                {
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "tools/call",
                    "params": {"name": args.call, "arguments": arguments},
                }
            )
            expected = 2
        else:
            print("No action specified. Use --list or --call.")
            proc.terminate()
            return 2

        responses = send_jsonrpc(proc, messages, expected_responses=expected, timeout=30.0)

        # Print all received responses with nice formatting
        for r in responses:
            print(json.dumps(r, indent=2, ensure_ascii=False))

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""MCP 接入的两种姿势，外加起一个本地 http server 的小工具。

姿势一：client.get_tools()  —— 一行拿到全部工具，每次工具调用新建会话
姿势二：client.session(name) + load_mcp_tools(session) —— 会话复用，自己管生命周期
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

ROOT = Path(__file__).resolve().parent.parent
SERVERS = ROOT / "servers"
PYTHON = sys.executable


def _py_script(name: str) -> str:
    return str(SERVERS / name)


def stdio_conn(script_name: str = "order_server.py") -> dict:
    """stdio 连接：客户端把 server 当子进程拉起来。"""
    return {
        "command": PYTHON,
        "args": [_py_script(script_name)],
        "transport": "stdio",
        "env": {**os.environ, "PYTHONIOENCODING": "utf-8"},
    }


def http_conn(port: int, transport: str = "streamable_http") -> dict:
    """http 连接：server 是常驻进程，客户端只管连。transport 也可填 sse。"""
    path = "/mcp" if transport == "streamable_http" else "/sse"
    return {"url": f"http://127.0.0.1:{port}{path}", "transport": transport}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_http_server(script_name: str = "kb_server.py", port: int | None = None):
    """把 http 型 server 拉起来，返回 (进程, 端口)。"""
    port = port or free_port()
    proc = subprocess.Popen(
        [PYTHON, _py_script(script_name), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    wait_port(port, proc=proc, timeout=30.0)
    return proc, port


def wait_port(port: int, proc: subprocess.Popen | None = None, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            out = b""
            if proc.stdout is not None:
                out = proc.stdout.read() or b""
            raise RuntimeError(
                f"http server 提前退出（returncode={proc.returncode}）：\n"
                + out.decode("utf-8", "replace")
            )
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"等待端口 {port} 就绪超时")


async def pose_get_tools(connections: dict) -> list[BaseTool]:
    """姿势一：一行拿到工具列表。"""
    client = MultiServerMCPClient(connections)
    return await client.get_tools()


def pose_session(client: MultiServerMCPClient, server_name: str):
    """姿势二：进会话内部，在同一个会话里加载工具。

    用法：
        async with pose_session(client, "kb") as session:
            tools = await load_mcp_tools(session)

    有状态 server（浏览器、数据库连接、登录态）必须走这条路径。
    """
    return client.session(server_name)


async def to_text(result) -> str:
    """MCP 工具的返回值是 content blocks 列表，取文本要自己拼。"""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(result)

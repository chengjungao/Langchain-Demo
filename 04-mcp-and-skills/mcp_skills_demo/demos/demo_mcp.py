# -*- coding: utf-8 -*-
"""姿势一与姿势二对照：不需要模型，直连 MCP 工具，最快验证链路。

运行：
    python demos/demo_mcp.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: E402
from langchain_mcp_adapters.tools import load_mcp_tools  # noqa: E402

from src.mcp_client import (  # noqa: E402
    http_conn,
    pose_session,
    spawn_http_server,
    stdio_conn,
    to_text,
)


def title(t: str) -> None:
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


async def main() -> int:
    title("1. 姿势一：get_tools() 一行拿到工具")

    connections = {
        "order": stdio_conn("order_server.py"),
    }
    client = MultiServerMCPClient(connections)
    tools = await client.get_tools()

    print(f"拿到 {len(tools)} 个工具")
    for t in tools:
        print(f"  - {t.name:<16} schema={type(t.args_schema).__name__:<10} "
              f"description={ (t.description or '').splitlines()[0][:34] }")
    print("\n注意两件事：")
    print("  args_schema 是 dict（JSON Schema），不是 Pydantic 模型")
    print("  description 就是 server 端函数的 docstring 原文")

    title("2. 不带模型直接调用工具")
    target = next(t for t in tools if t.name == "query_order")
    raw = await target.ainvoke({"order_no": "SO20260912001", "tenant_id": "t-acme"})
    print(f"raw type : {type(raw).__name__}")
    print(f"raw value: {str(raw)[:120]}")
    print(f"取文本后 : {await to_text(raw)}"[:160])

    refund = next(t for t in tools if t.name == "calc_refund")
    raw2 = await refund.ainvoke(
        {"order_no": "SO20260912002", "reason_code": "quality", "has_invoice": True}
    )
    print(f"\n试算结果 : {await to_text(raw2)}")

    title("3. 网上教程里的 with 写法，现在会怎样")
    try:
        with MultiServerMCPClient(connections):
            print("with 仍然可用")
    except TypeError as e:
        print(f"失败：TypeError: {e}")
        print("教程里普遍还在写 with，标准版本已经换了。")

    title("4. 姿势二：会话复用")

    holder = {}

    async with pose_session(client, "order") as session:
        session_tools = await load_mcp_tools(session)
        holder["session"] = session
        print(f"会话类型: {type(session).__name__}")
        print(f"会话内工具: {[t.name for t in session_tools]}")
        raw3 = await session_tools[0].ainvoke(
            {"order_no": "SO20260912001", "tenant_id": "t-acme"}
        )
        print(f"会话内调用一次: {await to_text(raw3)}"[:150])

    title("5. 换个传输：streamable-http")

    proc, port = spawn_http_server("kb_server.py")
    try:
        http_client = MultiServerMCPClient({"kb": http_conn(port)})
        kb_tools = await http_client.get_tools()
        print(f"http server 端口 {port}，拿到 {len(kb_tools)} 个工具: "
              f"{[t.name for t in kb_tools]}")

        search = kb_tools[0]
        hits = await search.ainvoke({"query": "审批", "top_k": 2})
        print(f"检索结果: {await to_text(hits)}"[:300])

        async with pose_session(http_client, "kb") as s2:
            res = await s2.read_resource("policy://refund/rules")
            text = res.contents[0].text if hasattr(res, "contents") else str(res)
            print(f"\nresource 读取（前 120 字）: {text[:120]}")
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

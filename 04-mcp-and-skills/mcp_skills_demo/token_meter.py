# -*- coding: utf-8 -*-
"""工具定义的 token 账单实测。

用法：
    python token_meter.py                 # 用合成工具算账
    python token_meter.py --count 7000
    python token_meter.py --params        # 按参数个数分档
    python token_meter.py --mcp           # 换成真实 MCP 工具

一个坑写在前面：count_tokens_approximately 必须用 tools= 传工具。
把工具定义塞进 messages，会走 convert_to_messages 解析，算的是另一套口径，
数字从一开始就不可比。算账算错，比不算更糟。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.messages.utils import count_tokens_approximately  # noqa: E402

from src.tool_gateway import synthetic_tools  # noqa: E402

WINDOW = 200_000
_PROBE = [HumanMessage(content="x")]


def tokens_of(tools: list) -> int:
    """只算工具定义那部分，把消息本身的基准量减掉。"""
    base = count_tokens_approximately(_PROBE)
    with_tools = count_tokens_approximately(_PROBE, tools=list(tools))
    return with_tools - base


def report(tools: list, label: str = "工具") -> dict:
    total = tokens_of(tools)
    n = len(tools) or 1
    avg = total / n
    rows = []
    for scale in (10, 50, 200, 1000, 7000):
        t = int(avg * scale)
        rows.append((scale, t, t / WINDOW))

    print(f"\n{label}：{len(tools)} 个，合计 {total:,} token，单工具均值 {avg:.1f}")
    print("\n按均值外推：")
    print(f"  {'工具数':>8} {'token':>12} {'占 200K 窗口':>14}")
    for scale, t, ratio in rows:
        flag = "  ← 建议切渐进式发现" if 0.01 <= ratio <= 0.05 else ""
        flag = flag or ("  ← 装不下" if ratio > 1 else "")
        print(f"  {scale:>8} {t:>12,} {ratio * 100:>13.1f}%{flag}")
    return {"count": len(tools), "total": total, "avg": avg, "scales": rows}


async def real_mcp_tools() -> list:
    from src.mcp_client import pose_get_tools, stdio_conn

    return await pose_get_tools({"order": stdio_conn("order_server.py")})


def by_param_count() -> None:
    """参数越多越贵。同一套描述模板，只改参数个数。"""
    from src.tool_gateway import make_tool

    print("\n按参数个数看（描述模板一致）：")
    print(f"  {'参数个数':>8} {'token':>8}")
    rows = []
    for n in (3, 4, 5, 6, 7, 8):
        params = [f"p{i}_id" for i in range(n - 1)] + ["tenant_id"]
        name = f"dom00_order_query_{n:04d}"
        desc = ("订单查询。用于 dom00 域的业务操作，"
                f"需要提供 {n} 个参数，调用前请确认 tenant_id 与调用方一致。")
        t = make_tool(name, desc, params)
        one = tokens_of([t])
        rows.append((n, one))
        print(f"  {n:>8} {one:>8}")
    if rows:
        avg = sum(v for _, v in rows) / len(rows)
        print(f"  均值 {avg:.1f} token/工具")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=7000)
    parser.add_argument("--mcp", action="store_true", help="用真实 MCP 工具")
    parser.add_argument("--params", action="store_true", help="按参数个数分档实测")
    parser.add_argument("--sample", type=int, default=6)
    args = parser.parse_args()

    if args.params:
        by_param_count()
        return 0

    if args.mcp:
        tools = asyncio.run(real_mcp_tools())
        print("真实 MCP 工具（来自 order_server），逐个看：")
        for t in tools:
            one = tokens_of([t])
            print(f"  {t.name:<16} {one:>5} token")
        report(tools, "真实 MCP 工具")
        return 0

    print(f"造 {args.count} 个业务风格工具（4~7 个参数、中文描述）")
    sample = synthetic_tools(args.sample, domains=args.sample)
    for t in sample:
        print(f"  {t.name:<28} {tokens_of([t]):>5} token")

    tools = synthetic_tools(args.count)
    report(tools, "合成工具")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

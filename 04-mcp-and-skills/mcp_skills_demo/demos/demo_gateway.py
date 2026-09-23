# -*- coding: utf-8 -*-
"""货架模式：全量注册，模型每轮只看得到检索命中的几个。

用测试替身模型跑，不需要 API key。断言模型侧实际拿到的工具数量。

运行：
    python demos/demo_gateway.py
    设 GATEWAY_TOOLS=800 可以缩小货架，跑得更快
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import build_agent, mcp_tools  # noqa: E402
from src.fake_model import RecordingModel  # noqa: E402
from src.tool_gateway import ToolGateway, synthetic_tools  # noqa: E402
from token_meter import tokens_of  # noqa: E402

QUERY = "帮我处理一笔退款，订单号 SO20260912001，客户说键盘有质量问题"
TOP_K = 8
WINDOW = 200_000


def title(t: str) -> None:
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


async def main() -> int:
    shelf_size = int(os.environ.get("GATEWAY_TOOLS", "7000"))

    title("1. 上货架")
    real = await mcp_tools()
    gateway = ToolGateway(synthetic_tools(shelf_size) + real)
    print(gateway.describe())
    print(f"其中真实 MCP 工具：{[t.name for t in real]}")

    title("2. 货架全量进上下文的账单")
    full_tokens = tokens_of(gateway.all_tools())
    print(f"全量 {gateway.size} 个工具：{full_tokens:,} token"
          f" ＝ 200K 窗口的 {full_tokens / WINDOW * 100:.1f}%")

    title("3. 检索：一个退款问题能捞出什么")
    hits = gateway.search(QUERY, top_k=TOP_K)
    for t in hits:
        print(f"  - {t.name}")
    print(f"\n命中 {len(hits)} 个（上限 {TOP_K}），"
          f"模型侧账单从 {full_tokens:,} 降到 {tokens_of(hits):,} token")

    title("4. 跑一遍 Agent，看模型侧实际拿到几个工具")
    model = RecordingModel()
    built = build_agent(model, gateway=gateway, top_k=TOP_K)
    print(f"中间件自己声明的工具：{[t.name for t in built.skills.tools]}")

    await built.agent.ainvoke({"messages": [{"role": "user", "content": QUERY}]})

    seen = model.seen_tools[0] if model.seen_tools else []
    print(f"\nAgent 注册的工具总数 : {built.gateway.shelf.size}")
    print(f"模型这一轮实际看到   : {len(seen)} 个")
    for name in seen:
        print(f"  - {name}")

    call = built.gateway.calls[-1]
    print(f"\n网关记录：{call['before']} → {call['after']}"
          f"｜原样保留 {call['kept']}｜检索意图「{call['query'][:16]}…」")

    assert len(seen) <= TOP_K + 1, f"模型侧工具数超出预期：{len(seen)}"
    assert len(seen) < built.gateway.shelf.size / 2, "网关没有起到收窄作用"
    print("\n断言通过：模型只看到检索命中的那几个。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

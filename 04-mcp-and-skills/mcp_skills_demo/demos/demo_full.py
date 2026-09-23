# -*- coding: utf-8 -*-
"""合体全流程：Skills 给章法，MCP 给工具。

走一遍真实工序：读技能正文 → 查政策 → 查订单 → 试算 → 提交 → 出回执。

模型用剧本替身，不需要 API key；MCP 侧是真连（stdio + streamable-http）。

运行：
    python demos/demo_full.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402

from src.agent import build_agent  # noqa: E402
from src.fake_model import ScriptedModel, call  # noqa: E402
from src.mcp_client import http_conn, pose_get_tools, spawn_http_server, stdio_conn  # noqa: E402
from src.tool_gateway import ToolGateway  # noqa: E402

QUESTION = "客户说键盘有质量问题，要退款，订单号 SO20260912002，已经开票了。帮我处理。"

RECEIPT = (
    "已受理。工单号 RF912002，退款金额 458.00 元，走质量问题通道，运费与税点全免。\n"
    "到账时间 1 到 3 个工作日。您需要把发票原件寄回，地址我发到您手机上。\n"
    "依据：refund.window 质量问题 30 天内可退；refund.invoice 已开票需财务先冲红。"
)


def title(t: str) -> None:
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


async def main() -> int:
    proc, port = spawn_http_server("kb_server.py")
    try:
        title("1. 两个 server 各走一种传输")
        connections = {
            "order": stdio_conn("order_server.py"),
            "kb": http_conn(port),
        }
        tools = await pose_get_tools(connections)
        print(f"stdio : order_server -> "
              f"{[t.name for t in tools if t.name != 'search_policy']}")
        print(f"http  : kb_server    -> "
              f"{[t.name for t in tools if t.name == 'search_policy']}")

        title("2. 装进同一个 Agent（Skills + MCP）")
        script = [
            call("read_skill", {"skill_name": "refund-flow"}, "c1"),
            call("search_policy", {"query": "质量问题 审批 税点", "top_k": 2}, "c2"),
            call("query_order", {"order_no": "SO20260912002", "tenant_id": "t-acme"}, "c3"),
            call("calc_refund",
                 {"order_no": "SO20260912002", "reason_code": "quality", "has_invoice": True},
                 "c4"),
            call("submit_refund",
                 {"order_no": "SO20260912002", "amount": 458.0,
                  "reason_code": "quality", "operator": "agent"}, "c5"),
            AIMessage(content=RECEIPT),
        ]
        model = ScriptedModel(script=script)
        built = build_agent(model, tools=tools)
        print(f"注册工具 : {[t.name for t in built.tools]}")
        print(f"中间件   : {[type(m).__name__ for m in built.middleware]}")

        title("3. 技能清单进上下文了吗")
        result = await built.agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]})
        first_system = model.seen_system[0]
        has_index = "refund-flow" in first_system and "技能库" in first_system
        print(f"system message 里出现技能清单 : {has_index}")
        print(f"system message 里出现技能正文 : {'第 1 步：查订单' in first_system}"
              "（正文不预加载，这一步应当是 False）")
        print(f"system message 长度 : {len(first_system)} 字符")

        title("4. 工序逐步执行")
        for i, (tools_i, msg) in enumerate(zip(model.seen_tools, script), 1):
            label = "最终回答" if not getattr(msg, "tool_calls", None) else msg.tool_calls[0]["name"]
            print(f"  第 {i} 轮｜模型看到 {len(tools_i):>2} 个工具｜动作：{label}")

        title("5. 技能正文是按需读进来的")
        contents = [getattr(m, "content", "") for m in result["messages"]
                    if getattr(m, "type", "") == "tool"]
        body = next((c for c in contents if "第 1 步" in str(c)), "")
        print(f"read_skill 返回的正文长度 : {len(str(body))} 字符")
        print(f"正文首行 : {str(body).splitlines()[0][:40]}")

        title("6. 工具真的被调用了")
        print("工具返回（截断）：")
        for c in contents:
            s = str(c).replace("\n", " ")
            print(f"  - {s[:96]}")

        title("7. 最终回执")
        print(result["messages"][-1].content)

        title("8. 换成货架模式（同样的 Agent，多一个中间件）")
        from src.tool_gateway import synthetic_tools

        gateway = ToolGateway(synthetic_tools(400) + tools)
        model2 = ScriptedModel(script=[AIMessage(content="收到。")])
        built2 = build_agent(model2, gateway=gateway, top_k=8)
        await built2.agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]})
        c = built2.gateway.calls[-1]
        print(f"货架 {gateway.size} 个工具，模型看到 {c['after']} 个：{c['names']}")

        assert has_index, "技能清单没有进 system message"
        assert "第 1 步：查订单" in str(body), "技能正文没有被按需读取"
        assert any("RF912002" in str(c) for c in contents), "提交退款没有被真实执行"
        print("\n断言通过：技能清单常驻、正文按需读、MCP 工具真调用。")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

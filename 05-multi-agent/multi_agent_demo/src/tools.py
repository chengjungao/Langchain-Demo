# -*- coding: utf-8 -*-
"""合成工具：与《MCP 与 Skills 接入》那篇同一套口径。

为什么用合成工具而不是真工具：
    这里要量的是「工具定义占多少 token」，而 token 量由工具名、描述、参数个数决定，
    跟工具背后干了什么无关。合成工具能把这三个量固定住，让不同形态之间可比。

口径对齐：
    单工具定义里，描述用中文业务风格写满一段，参数名带 _id 后缀。
    与上一篇的实测均值 219.6 token/工具 同源，所以两篇的数字可以直接比。
"""
from __future__ import annotations

from typing import Any, Sequence

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field, create_model


def make_tool(name: str, description: str, params: Sequence[str]) -> BaseTool:
    """按名称与参数清单造一个结构化工具。"""
    fields = {p: (str, Field(description=f"{p} 参数")) for p in params}
    schema = create_model(f"{name}_args", **fields)

    def _fn(**kwargs: Any) -> str:
        return f"{name} -> " + ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

    return StructuredTool(name=name, description=description, args_schema=schema, func=_fn)


# ---------------- 单 Agent 手里的全套工具（6 个） ----------------
TOOLS_SINGLE: list[BaseTool] = [
    make_tool("order_query", "按订单号或用户标识查询订单详情，用于售后与物流核对。", ["order_no", "tenant_id"]),
    make_tool("order_refund_calc", "试算一笔订单的可退金额，含优惠分摊与运费处理。", ["order_no", "reason_code", "tenant_id"]),
    make_tool("order_refund_submit", "提交退款申请，涉及资金变动，需先试算并确认。", ["order_no", "amount", "operator_id", "tenant_id"]),
    make_tool("kb_search", "检索内部知识库，返回与问题最相关的若干段落。", ["query", "top_k", "tenant_id"]),
    make_tool("report_build", "生成运营报表，支持按日/周/月聚合。", ["metric", "start_date", "end_date", "tenant_id"]),
    make_tool("ticket_create", "创建工单并指派给对应处理人。", ["title", "detail", "assignee", "tenant_id"]),
]

# ---------------- 拆开之后，每个专家只拿自己那一份 ----------------
TOOLS_ORDER = TOOLS_SINGLE[0:3]
TOOLS_KB = TOOLS_SINGLE[3:4]
TOOLS_REPORT = TOOLS_SINGLE[4:6]

# ---------------- 主管手里的转交工具 ----------------
TRANSFER_TOOLS: list[BaseTool] = [
    make_tool(
        f"transfer_to_{n}",
        f"把当前任务转交给{n}处理，task 参数写清它要做什么。",
        ["task"],
    )
    for n in ("order_agent", "kb_agent", "report_agent")
]


# ---------------- Agent 的提示词（长度模拟真实产线） ----------------
SINGLE_SYSTEM = (
    "你是某电商平台的运营助理，负责帮运营同学处理日常事务。"
    "你可以查询订单、试算与提交退款、检索内部知识库、生成运营报表。"
    "遇到需要用户身份的操作，先确认调用方身份再执行。"
    "查询类操作可以直接做，涉及资金变动的操作必须先把试算结果说清楚，等确认后再提交。"
    "回答用中文，先给结论再给依据，涉及数字要标注来源工具与查询条件。"
    "如果一次请求里包含多件事，按顺序逐件处理，每件处理完给一行小结。"
    "不要臆造工具没有返回的数据，拿不到就直说拿不到。"
) * 3  # 叠到约 900 字，模拟写了几百行的那种系统提示词

SUPERVISOR_SYSTEM = (
    "你是运营助理团队的主管。下面有三位专家：订单专家、知识专家、报表专家。"
    "你的职责是判断用户请求属于哪一类，然后把任务转交给对应专家。"
    "转交时要把用户的原始诉求写清楚，不要自己加戏。"
    "专家返回结果后，你负责汇总成一段给用户的最终回答。"
) * 2

SUB_SYSTEM = (
    "你是订单专家，只处理订单查询与退款相关的请求。"
    "只做被交代的那件事，做完给一段简短结论，不要展开无关内容。"
)

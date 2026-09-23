# -*- coding: utf-8 -*-
"""两笔账：窄任务（只动一个专家）与宽任务（要动三个专家）。

计数口径与《MCP 与 Skills 接入》那篇完全一致：count_tokens_approximately，
工具定义通过 tools= 传进去。不要用别的路径计工具定义，口径不同数字不可比。

这里的账只算模型输入侧，不含模型输出。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from src.tools import (
    SINGLE_SYSTEM,
    SUB_SYSTEM,
    SUPERVISOR_SYSTEM,
    TOOLS_KB,
    TOOLS_ORDER,
    TOOLS_REPORT,
    TOOLS_SINGLE,
    TRANSFER_TOOLS,
)

WINDOW = 200_000


def tok(messages: list, tools: list | None = None) -> int:
    return count_tokens_approximately(messages, tools=tools or [])


def split(messages: list, tools: list | None = None) -> tuple[int, int]:
    """返回 (消息部分, 工具定义部分)。"""
    base = tok(messages)
    return base, tok(messages, tools) - base


# ---------------- 一、固定开销 ----------------
def fixed_overhead() -> dict:
    s, s_tools = split([SystemMessage(content=SINGLE_SYSTEM)], TOOLS_SINGLE)
    sup_s, sup_tools = split([SystemMessage(content=SUPERVISOR_SYSTEM)], TRANSFER_TOOLS)
    sub_s, sub_tools = split([SystemMessage(content=SUB_SYSTEM)], TOOLS_ORDER)
    _, kb_tools = split([SystemMessage(content=SUB_SYSTEM)], TOOLS_KB)
    _, rep_tools = split([SystemMessage(content=SUB_SYSTEM)], TOOLS_REPORT)

    sub_total = sub_s + sub_tools + kb_tools + rep_tools
    return {
        "single": {"system": s, "tools": s_tools, "total": s + s_tools},
        "multi": {
            "supervisor": sup_s + sup_tools,
            "supervisor_system": sup_s,
            "transfer_tools": sup_tools,
            "order": sub_s + sub_tools,
            "kb": sub_s + kb_tools,
            "report": sub_s + rep_tools,
            "total": sup_s + sup_tools + sub_total,
        },
    }


# ---------------- 二、窄任务：只动一个专家 ----------------
NARROW_Q = "帮我查一下订单 SO20260915001 现在什么状态，顺便看看能不能退，能退多少。"


def narrow_task() -> dict:
    """同一件只动一个专家的事，两种形态各走一遍。"""
    user_q = HumanMessage(content=NARROW_Q)

    # 单 Agent：问 → 查订单 → 试算 → 作答，3 次调用
    total_single = 0
    history = [user_q]
    m = [SystemMessage(content=SINGLE_SYSTEM)] + history
    b, t = split(m, TOOLS_SINGLE)
    total_single += b + t
    history += [
        AIMessage(content="", tool_calls=[{"name": "order_query", "args": {"order_no": "SO...", "tenant_id": "t1"}, "id": "c1"}]),
        ToolMessage(content="订单状态：已签收；金额 328.00；签收日期 2026-09-10", tool_call_id="c1"),
    ]
    b, t = split([SystemMessage(content=SINGLE_SYSTEM)] + history, TOOLS_SINGLE)
    total_single += b + t
    history += [
        AIMessage(content="", tool_calls=[{"name": "order_refund_calc", "args": {"order_no": "SO...", "reason_code": "7天无理由", "tenant_id": "t1"}, "id": "c2"}]),
        ToolMessage(content="可退金额 296.00（优惠分摊 24.00、运费 8.00 不退）", tool_call_id="c2"),
    ]
    b, t = split([SystemMessage(content=SINGLE_SYSTEM)] + history, TOOLS_SINGLE)
    total_single += b + t

    # 多 Agent：主管判断 → 订单专家查 → 试算 → 回主管收口，4 次调用
    total_multi = 0
    sup_hist = [user_q]
    b, t = split([SystemMessage(content=SUPERVISOR_SYSTEM)] + sup_hist, TRANSFER_TOOLS)
    total_multi += b + t
    sup_hist += [AIMessage(content="", tool_calls=[{"name": "transfer_to_order_agent", "args": {"task": "查订单状态并试算退款"}, "id": "t1"}])]

    # 隔离姿势：专家只拿到任务描述，不拿父图全历史
    sub_hist = [HumanMessage(content="查订单状态并试算退款")]
    b, t = split([SystemMessage(content=SUB_SYSTEM)] + sub_hist, TOOLS_ORDER)
    total_multi += b + t
    sub_hist += [
        AIMessage(content="", tool_calls=[{"name": "order_query", "args": {"order_no": "SO...", "tenant_id": "t1"}, "id": "c1"}]),
        ToolMessage(content="订单状态：已签收；金额 328.00", tool_call_id="c1"),
    ]
    b, t = split([SystemMessage(content=SUB_SYSTEM)] + sub_hist, TOOLS_ORDER)
    total_multi += b + t

    sup_hist += [ToolMessage(content="订单已签收，可退 296.00", tool_call_id="t1")]
    b, t = split([SystemMessage(content=SUPERVISOR_SYSTEM)] + sup_hist, TRANSFER_TOOLS)
    total_multi += b + t

    return {
        "single": {"calls": 3, "tokens": total_single},
        "multi": {"calls": 4, "tokens": total_multi},
        "ratio": total_multi / total_single,
    }


# ---------------- 三、宽任务：要动三个专家 ----------------
WIDE_Q = "三件事：1）订单 SO20260915001 什么状态、能不能退；2）查一下七天无理由的规则原文；3）出一下上周的退款率报表。"


def wide_task() -> dict:
    q = HumanMessage(content=WIDE_Q)

    # 单 Agent：一轮并发发 3 个 tool_calls，再一轮作答，共 2 次调用
    h = [q]
    b, t = split([SystemMessage(content=SINGLE_SYSTEM)] + h, TOOLS_SINGLE)
    total_single = b + t
    h += [
        AIMessage(content="", tool_calls=[
            {"name": "order_query", "args": {"order_no": "SO...", "tenant_id": "t1"}, "id": "a"},
            {"name": "kb_search", "args": {"query": "七天无理由", "top_k": 3, "tenant_id": "t1"}, "id": "b"},
            {"name": "report_build", "args": {"metric": "refund_rate", "start_date": "09-07", "end_date": "09-13", "tenant_id": "t1"}, "id": "c"},
        ]),
        ToolMessage(content="订单：已签收 328.00", tool_call_id="a"),
        ToolMessage(content="规则：签收后 7 天内可申请……", tool_call_id="b"),
        ToolMessage(content="退款率 3.2%", tool_call_id="c"),
    ]
    b, t = split([SystemMessage(content=SINGLE_SYSTEM)] + h, TOOLS_SINGLE)
    total_single += b + t

    # 多 Agent：主管来回 4 次 + 三个专家各 2 次，共 10 次调用
    total_multi = 0
    n_call = 0
    sh = [q]
    tool_map = {"order": TOOLS_ORDER, "kb": TOOLS_KB, "report": TOOLS_REPORT}
    for name in ("order", "kb", "report"):
        b, t = split([SystemMessage(content=SUPERVISOR_SYSTEM)] + sh, TRANSFER_TOOLS)
        total_multi += b + t
        n_call += 1
        sh += [
            AIMessage(content="", tool_calls=[{"name": f"transfer_to_{name}_agent", "args": {"task": f"{name} 相关任务"}, "id": f"t{n_call}"}]),
            ToolMessage(content=f"已转交 {name}_agent，它已完成：结论……", tool_call_id=f"t{n_call}"),
        ]
        sub_tools = tool_map[name]
        subh = [HumanMessage(content=f"{name} 相关任务")]
        b, t = split([SystemMessage(content=SUB_SYSTEM)] + subh, sub_tools)
        total_multi += b + t
        n_call += 1
        subh += [
            AIMessage(content="", tool_calls=[{"name": sub_tools[0].name, "args": {}, "id": "s1"}]),
            ToolMessage(content="工具返回结果……", tool_call_id="s1"),
        ]
        b, t = split([SystemMessage(content=SUB_SYSTEM)] + subh, sub_tools)
        total_multi += b + t
        n_call += 1

    b, t = split([SystemMessage(content=SUPERVISOR_SYSTEM)] + sh, TRANSFER_TOOLS)
    total_multi += b + t
    n_call += 1

    return {
        "single": {"calls": 2, "tokens": total_single},
        "multi": {"calls": n_call, "tokens": total_multi},
        "ratio": total_multi / total_single,
    }

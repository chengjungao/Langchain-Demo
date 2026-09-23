# -*- coding: utf-8 -*-
"""专家与主管：整套 demo 的 Agent 装配都在这里。

三个专家各带一块工具：
    订单专家  order_query / order_refund_calc / order_refund_submit
    知识专家  kb_search
    报表专家  report_build / ticket_create

主管是一个普通节点，它做的判断在本 demo 里由脚本写死（script 驱动），
换成真模型后把 script 去掉即可。
"""
from __future__ import annotations

from langchain.agents import create_agent

from src.spy_model import SpyChatModel
from src.tools import (
    SUB_SYSTEM,
    TOOLS_KB,
    TOOLS_ORDER,
    TOOLS_REPORT,
)

ORDER_SYSTEM = "你是订单专家，只处理订单查询与退款相关的请求。" + SUB_SYSTEM
KB_SYSTEM = "你是知识专家，只回答内部政策与规则类问题。" + SUB_SYSTEM
REPORT_SYSTEM = "你是报表专家，只负责出报表与建工单。" + SUB_SYSTEM


def _check_script(script) -> list:
    """挡住「把 model 当成 script 位置传参」这类手误。

    pydantic 模型是可迭代的，迭代出来是一串 (键, 值) 元组。
    一旦误传，脚本字段会变成这种元组列表，报错却发生在几百行之外。
    """
    if script is None:
        return []
    if not isinstance(script, list):
        raise TypeError(
            "script 必须是列表，形如 [{'text': '...'}]；"
            "如果要传模型，请用关键字 model=xxx"
        )
    return list(script)


def make_order_expert(script: list | None = None, model=None):
    """订单专家。"""
    m = model or SpyChatModel(tag="order", script=_check_script(script))
    return create_agent(model=m, tools=TOOLS_ORDER, system_prompt=ORDER_SYSTEM, name="order_expert")


def make_kb_expert(script: list | None = None, model=None):
    """知识专家。"""
    m = model or SpyChatModel(tag="kb", script=_check_script(script))
    return create_agent(model=m, tools=TOOLS_KB, system_prompt=KB_SYSTEM, name="kb_expert")


def make_report_expert(script: list | None = None, model=None):
    """报表专家。"""
    m = model or SpyChatModel(tag="report", script=_check_script(script))
    return create_agent(model=m, tools=TOOLS_REPORT, system_prompt=REPORT_SYSTEM, name="report_expert")


def make_all_experts(scripts: dict | None = None) -> dict:
    """一次建好三个专家，scripts 形如 {"order": [...], "kb": [...], "report": [...]}。"""
    scripts = scripts or {}
    return {
        "order_expert": make_order_expert(scripts.get("order")),
        "kb_expert": make_kb_expert(scripts.get("kb")),
        "report_expert": make_report_expert(scripts.get("report")),
    }

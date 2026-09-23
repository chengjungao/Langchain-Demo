# -*- coding: utf-8 -*-
"""三种隔离手段：默认姿势、Send 只传任务、独立 state schema。

这一章要回答的问题是：把 Agent 拆开，到底隔离了什么？
三个函数各自搭一张图，配合 demos/demo_isolation.py 跑出对照数字。
"""
from __future__ import annotations

from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from typing_extensions import TypedDict

from src.spy_model import SpyChatModel

# 一段业务味的父图历史，用来放大差异
FILLER = (
    "客服系统的订单查询链路今天又出了一次超时，排查下来是下游接口在高峰期响应变慢，"
    "我们临时加了缓存并且把超时时间从三秒调到五秒，观察了半小时没有复现。"
)
TASK = "请查一下订单 20260915001 的物流状态。"


def build_history(rounds: int = 12) -> list:
    """造一段父图历史，默认 12 轮共 24 条消息。"""
    history: list = []
    for i in range(rounds):
        history.append(HumanMessage(content=f"第 {i} 轮：{FILLER}"))
        history.append(AIMessage(content=f"第 {i} 轮回复：我记下了。{FILLER}"))
    return history


# ---------------- 手段零：默认姿势（并不隔离） ----------------
class PlainState(TypedDict):
    messages: Annotated[list, add_messages]


def build_naive(sub_agent):
    """把子 Agent 直接当节点嵌进父图。父图的全部历史都会传给它。"""
    g = StateGraph(PlainState)
    g.add_node("sub", sub_agent)
    g.add_edge(START, "sub")
    g.add_edge("sub", END)
    return g.compile()


# ---------------- 手段一：Send 只传挑出来的那几样 ----------------
class ParentWithFanState(TypedDict):
    messages: Annotated[list, add_messages]


class WorkerState(TypedDict):
    task: str
    result: Annotated[list, add_messages]


def build_send_payload(worker_model: SpyChatModel, sub_agent_factory):
    """用 Send 把任务单独送进 worker，父图历史一概进不去。

    worker 用的是一套只认 task 的 schema，父图里那些 messages 没有落脚的地方。
    """

    def dispatch(state):
        return [Send("worker", {"task": TASK})]

    def worker(state: WorkerState):
        # 注意用关键字传参：位置传参会把 model 塞进 script 形参
        sub = sub_agent_factory(model=worker_model)
        out = sub.invoke({"messages": [HumanMessage(content=state["task"])]})
        return {"result": out["messages"]}

    g = StateGraph(ParentWithFanState)
    g.add_node("dispatch", lambda s: {})
    g.add_node("worker", worker)
    g.add_edge(START, "dispatch")
    g.add_conditional_edges("dispatch", dispatch, ["worker"])
    g.add_edge("worker", END)
    return g.compile()


# ---------------- 手段二：独立 state schema ----------------
class ChildState(TypedDict):
    task: str  # 父图里叫 query，名字不同
    messages: Annotated[list, add_messages]  # 名字撞上了
    note: str


class SchemaParentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    note: str


SEEN_KEYS: list = []


def build_schema_isolation():
    """子图声明自己的 schema：同名键照样穿进去，异名键被挡住。"""

    def child(state: ChildState):
        SEEN_KEYS.append({
            "keys": sorted(state.keys()),
            "messages_n": len(state.get("messages") or []),
        })
        return {"note": "子图跑完"}

    g = StateGraph(SchemaParentState)
    g.add_node("child", child)
    g.add_edge(START, "child")
    g.add_edge("child", END)
    return g.compile()


# ---------------- 汇总用的计数口径 ----------------
def count_tokens(messages: list, tools: list | None = None) -> int:
    """与《MCP 与 Skills 接入》那篇同一套计数口径。"""
    from langchain_core.messages.utils import count_tokens_approximately

    return count_tokens_approximately(messages, tools=tools or [])


def first_call_input(model: SpyChatModel) -> tuple[int, int, int]:
    """返回 (消息条数, 字符数, token 数)，取该模型第一次调用的输入。"""
    calls = SpyChatModel.calls_of(model.tag)
    if not calls:
        return 0, 0, 0
    msgs = calls[0]["messages"]
    chars = sum(len(m.content or "") for m in msgs)
    return len(msgs), chars, count_tokens(msgs)


__all__ = [
    "TASK",
    "build_history",
    "build_naive",
    "build_send_payload",
    "build_schema_isolation",
    "create_agent",
    "SystemMessage",
    "SEEN_KEYS",
    "count_tokens",
    "first_call_input",
    "Any",
]

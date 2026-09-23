# -*- coding: utf-8 -*-
"""并行分发：Send 的正确姿势与错误姿势。

正确：从条件边返回 Send 列表，几个 worker 同时跑，各拿自己的 payload。
错误：从普通节点返回 Send 列表，1.x 直接抛 InvalidUpdateError。
"""
from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from typing_extensions import TypedDict


class FanState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    results: Annotated[list[AnyMessage], add_messages]


class WorkerState(TypedDict):
    seed: str
    results: Annotated[list[AnyMessage], add_messages]


LANDED: list = []


def worker(state: WorkerState):
    """每个 worker 只该看到自己那份 payload。"""
    LANDED.append(state.get("seed"))
    return {"results": [AIMessage(content=f"worker 收到 {state.get('seed')}")]}


def build_parallel_graph(seeds: list[str]):
    """正确的并行分发：Send 从条件边返回。"""
    LANDED.clear()

    def dispatch(state):
        return [Send("worker", {"seed": s}) for s in seeds]

    g = StateGraph(FanState)
    g.add_node("dispatch", lambda s: {})
    g.add_node("worker", worker)
    g.add_edge(START, "dispatch")
    g.add_conditional_edges("dispatch", dispatch, ["worker"])
    g.add_edge("worker", END)
    return g.compile()


def build_wrong_way(seeds: list[str]):
    """错误写法：Send 从普通节点返回。留着是为了复现那句报错原文。"""

    def fan_out(state):
        return [Send("worker", {"seed": s}) for s in seeds]

    g = StateGraph(FanState)
    g.add_node("fan_out", fan_out)
    g.add_node("worker", worker)
    g.add_edge(START, "fan_out")
    g.add_edge("fan_out", "worker")
    g.add_edge("worker", END)
    return g.compile()


def try_wrong_way(seeds: list[str] | None = None) -> str:
    """跑一次错误写法，把报错类型与原文交出来。"""
    app = build_wrong_way(seeds or ["A", "B", "C"])
    try:
        app.invoke({"messages": [HumanMessage(content="go")], "results": []})
        return "居然没报错（版本行为可能变了，请核对）"
    except InvalidUpdateError as e:
        return f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"

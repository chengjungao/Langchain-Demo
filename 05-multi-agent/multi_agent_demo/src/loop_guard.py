# -*- coding: utf-8 -*-
"""转交环的检测与兜底。

主管式编排最典型的翻车方式：主管每次都把活推出去，专家每次都推回来。
两个函数：
    build_pingpong_graph()  故意造的环，用来复现 GraphRecursionError 原文
    build_guarded_graph()   加了计数守卫的版本，环会被拦在预算内
"""
from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

DEFAULT_LIMIT = 8


class LoopState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    hops: int


def _bump(state: LoopState, who: str) -> dict:
    return {
        "messages": [AIMessage(content=f"[{who}] 第 {state.get('hops', 0) + 1} 次来回", name=who)],
        "hops": state.get("hops", 0) + 1,
    }


def build_pingpong_graph():
    """没有守卫的版本：主管与专家互相推，直到撞上 recursion_limit。"""

    def supervisor(state: LoopState):
        return _bump(state, "supervisor")

    def expert(state: LoopState):
        return _bump(state, "expert")

    def route(state: LoopState):
        return "expert" if state.get("hops", 0) % 2 else "supervisor"

    g = StateGraph(LoopState)
    g.add_node("supervisor", supervisor)
    g.add_node("expert", expert)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: "expert", ["expert"])
    g.add_edge("expert", "supervisor")
    return g.compile()


def build_guarded_graph(limit: int = DEFAULT_LIMIT):
    """加了计数守卫的版本：跳够预算就收口，不再往下推。

    守卫逻辑只有两行：进节点先看 hops，超预算就直接回 END。
    真实产线里建议把这个判断做成装饰器套在主管与专家节点上。
    """

    def supervisor(state: LoopState) -> Any:
        if state.get("hops", 0) >= limit:
            return {
                "messages": [AIMessage(content="转交预算用尽，我直接收口。", name="supervisor")],
            }
        return _bump(state, "supervisor")

    def expert(state: LoopState) -> Any:
        if state.get("hops", 0) >= limit:
            return {
                "messages": [AIMessage(content="转交预算用尽，我把手上的结果交回。", name="expert")],
            }
        return _bump(state, "expert")

    def after_supervisor(state: LoopState):
        return END if state.get("hops", 0) >= limit else "expert"

    def after_expert(state: LoopState):
        return END if state.get("hops", 0) >= limit else "supervisor"

    g = StateGraph(LoopState)
    g.add_node("supervisor", supervisor)
    g.add_node("expert", expert)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", after_supervisor, ["expert", END])
    g.add_conditional_edges("expert", after_expert, ["supervisor", END])
    return g.compile()


def run_unbounded(recursion_limit: int = DEFAULT_LIMIT) -> dict:
    """跑没守卫的那张图，返回报错信息与已产生的轨迹长度。"""
    app = build_pingpong_graph()
    try:
        out = app.invoke(
            {"messages": [HumanMessage(content="一桩没人认领的活")], "hops": 0},
            {"recursion_limit": recursion_limit},
        )
        return {"ok": True, "msgs": len(out["messages"]), "hops": out["hops"], "err": None}
    except GraphRecursionError as e:
        return {"ok": False, "msgs": None, "hops": None, "err": f"{type(e).__name__}: {e}"}


def run_guarded(limit: int = DEFAULT_LIMIT) -> dict:
    """跑加了守卫的版本。"""
    app = build_guarded_graph(limit)
    out = app.invoke(
        {"messages": [HumanMessage(content="一桩没人认领的活")], "hops": 0},
        {"recursion_limit": max(limit * 3, 25)},
    )
    return {"ok": True, "msgs": len(out["messages"]), "hops": out["hops"], "tail": out["messages"][-1].content}

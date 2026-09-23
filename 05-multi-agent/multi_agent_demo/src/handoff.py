# -*- coding: utf-8 -*-
"""手写交接：用 Command(goto=...) 把控制权交给下一个 Agent。

预制件装不下的场景得自己画。这里给一张最小可用的主管图：
    __start__ → supervisor ⇄ 各专家 → __end__
主管用 Command 决定去哪个专家，专家跑完回主管收口。
"""
from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import TypedDict

from src.spy_model import SpyChatModel


class TeamState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turn: int


def make_supervisor_node(model: SpyChatModel, expert_names: list[str], max_turns: int = 3):
    """主管节点：每轮挑一个专家交出去，挑不动了就自己收口。

    这里用脚本驱动模型，换成真模型时把 script 换成真调用即可。
    注意 return 的是 Command，不是普通 dict。这是交接的标准写法。
    """

    def supervisor(state: TeamState) -> Command:
        turn = state.get("turn", 0)
        if turn >= max_turns:
            model.script = [{"text": "三件事都办完了，我汇总一下。"}]
        out = model.invoke([*state["messages"]])
        tool_calls = getattr(out, "tool_calls", None) or []

        if not tool_calls:
            return Command(
                goto=END,
                update={"messages": [AIMessage(content=out.content, name="supervisor")]},
            )

        target = tool_calls[0]["name"]
        args = tool_calls[0].get("args") or {}
        handoff_note = AIMessage(
            content=f"转交给 {target}：{args.get('task', '')}",
            name="supervisor",
        )
        # graph=Command.PARENT 是给「在子图里往父图跳」用的。
        # 主管本身就在父图上，所以这里不需要它。
        return Command(
            goto=target,
            update={"messages": [handoff_note], "turn": turn + 1},
        )

    return supervisor


def build_handoff_graph(supervisor_model: SpyChatModel, experts: dict):
    """把主管与专家拼成一张图。"""
    g = StateGraph(TeamState)
    names = list(experts.keys())
    g.add_node("supervisor", make_supervisor_node(supervisor_model, names))
    for name, agent in experts.items():
        g.add_node(name, agent)
        g.add_edge(name, "supervisor")  # 专家跑完一律回主管
    g.add_edge(START, "supervisor")
    return g.compile()


def describe_trajectory(messages: list) -> list[str]:
    """把消息列表压成一眼能看的轨迹。"""
    traj = []
    for m in messages:
        label = getattr(m, "name", None) or m.type
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            label += "(" + ",".join(tc["name"] for tc in tool_calls) + ")"
        traj.append(label)
    return traj

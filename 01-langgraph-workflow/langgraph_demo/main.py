# -*- coding: utf-8 -*-
"""高危操作审核 Agent：三关闭环 demo（零 API key，内置假模型，直接可跑）。

流程：Agent 生成高危指令 -> interrupt 挂起等人工放行 -> 放行后执行出错
     -> 时间旅行回到出错前 -> 换安全指令重跑成功。

运行：python main.py
依赖：pip install "langgraph>=1.0" "langgraph-checkpoint-sqlite" langchain-core
"""
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


# ---------- 1. 危险工具：interrupt 人工闸口 ----------
@tool
def exec_change(cmd: str) -> str:
    """执行一条运维变更指令"""
    ok = interrupt({"cmd": cmd, "question": "放行这条指令？"})
    if ok != "yes":
        return "已驳回"
    if "rm " in cmd:  # 模拟执行出事
        raise RuntimeError("误伤生产目录（模拟）")
    return f"已执行：{cmd}"


TOOLS = [exec_change]

# ---------- 2. 假模型：无需 API key，按对话内容出牌 ----------
def call_model(state: MessagesState):
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content="变更流程已结束。")]}
    user_text = next((m.content for m in reversed(state["messages"])
                      if getattr(m, "type", "") == "human"), "")
    if "归档" in user_text:
        cmd = "archive /data/tmp/old"
    else:
        cmd = "rm -rf /data/tmp/old"
    return {"messages": [AIMessage(content="", tool_calls=[
        {"name": "exec_change", "args": {"cmd": cmd}, "id": "call_1"},
    ])]}


# ---------- 3. 组图：model <-> tools 循环 ----------
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile(checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "ops-demo"}}

print("== 第一次 invoke：撞上 interrupt，图挂起 ==")
graph.invoke({"messages": [("user", "清理 /data/tmp 下的旧日志")]}, cfg)
snap = graph.get_state(cfg)
task = snap.tasks[0] if snap.tasks else None
if task is not None and getattr(task, "interrupts", None):
    print("审批单:", task.interrupts[0].value)
print("图已挂起，next =", snap.next)

print("\n== 放行后执行出错，但每步状态已存档 ==")
try:
    graph.invoke(Command(resume="yes"), cfg)
except Exception as e:
    print("执行异常:", e)

print("\n== 时间旅行：列出全部 checkpoint ==")
history = list(graph.get_state_history(cfg))
for i, s in enumerate(history):
    print(f"[{i}] next={s.next}")

print("\n== 回到最初 model 之前，换安全指令重跑 ==")
model_before = [s for s in history if s.next == ("model",)][-1]  # 最早的一档
fixed = graph.update_state(
    model_before.config,
    {"messages": [("user", "改用归档脚本清理")]},
)
final = graph.invoke(None, fixed)
# 重跑会再次经过 interrupt 闸口（恢复即重跑，天然再审一次），补一次放行
final = graph.invoke(Command(resume="yes"), cfg)
print("最终结果:", final["messages"][-1].content)

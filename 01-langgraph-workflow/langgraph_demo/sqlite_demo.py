# -*- coding: utf-8 -*-
"""SqliteSaver 持久化版：验证"进程死了也能续"。

两步跑法（模拟进程重启）：
  第 1 步：python sqlite_demo.py run    # 跑到 interrupt 挂起，状态写入 checkpoints.db 后退出
  第 2 步：python sqlite_demo.py resume # 重新启动，按同一 thread_id 从断点恢复放行
"""
import sys
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

DB = "checkpoints.db"
THREAD = "ops-sqlite"


@tool
def exec_change(cmd: str) -> str:
    """执行一条运维变更指令"""
    ok = interrupt({"cmd": cmd, "question": "放行这条指令？"})
    if ok != "yes":
        return "已驳回"
    return f"已执行：{cmd}"


def call_model(state: MessagesState):
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content="变更流程已结束。")]}
    return {"messages": [AIMessage(content="", tool_calls=[
        {"name": "exec_change", "args": {"cmd": "archive /data/tmp/old"}, "id": "call_1"},
    ])]}


def build():
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode([exec_change]))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    return builder


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"

    with SqliteSaver.from_conn_string(DB) as checkpointer:
        graph = build().compile(checkpointer=checkpointer)
        cfg = {"configurable": {"thread_id": THREAD}}

        if mode == "run":
            print("== 跑到 interrupt 挂起，状态已写入", DB, "==")
            graph.invoke({"messages": [("user", "清理 /data/tmp 下的旧日志")]}, cfg)
            print("图已挂起。现在可以关掉进程（Ctrl+C），再执行: python sqlite_demo.py resume")
            return

        if mode == "resume":
            snap = graph.get_state(cfg)
            print("重启后读到的断点 next =", snap.next)
            print("== 从数据库断点恢复，放行 ==")
            final = graph.invoke(Command(resume="yes"), cfg)
            print("结果:", final["messages"][-1].content)
            return

        print("用法: python sqlite_demo.py run | resume")


if __name__ == "__main__":
    main()

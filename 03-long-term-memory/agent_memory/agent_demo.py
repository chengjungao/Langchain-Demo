"""LangGraph Agent 接入演示：两种记忆接入姿势。

    python agent_demo.py

用脚本化模型驱动，整条链路不需要 API key。
看点是第 2 幕和第 4 幕：全新的 thread、空的历史消息，
Agent 依然知道自己面对的是谁、该用什么语言答话。
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from agent_memory import InMemoryStore, LocalHashEmbedder, MemoryManager
from agent_memory.agent import build_memory_agent

USER_A = "alice"
USER_B = "bob"


def title(text: str) -> None:
    print(f"\n{'=' * 66}\n{text}\n{'=' * 66}")


def new_thread(memory: MemoryManager, user_id: str, label: str, text: str) -> None:
    """开一条全新的 thread 跑一轮对话。thread 之间不共享任何消息历史。"""
    from agent_memory.agent import AgentContext

    print(f"\n--- {label}")
    print(f"    用户：{text}")
    result = graph.invoke(
        {"messages": [("user", text)]},
        config={"configurable": {"thread_id": f"{user_id}-{label}"}},
        context=AgentContext(user_id=user_id),
    )
    for message in result["messages"]:
        name = type(message).__name__
        if name == "AIMessage" and message.tool_calls:
            for call in message.tool_calls:
                print(f"    Agent 调用工具：{call['name']}({call['args']})")
        elif name == "AIMessage" and message.content:
            print(f"    Agent：{message.content}")
        elif name == "ToolMessage":
            print(f"    工具返回：{str(message.content)[:100]}")


def dump(memory: MemoryManager, user_id: str) -> None:
    print(f"    [{user_id} 当前有效记忆]")
    for mem in memory.store.list(user_id):
        print(f"      {mem.content}")


memory = MemoryManager(
    store=InMemoryStore(embedder=LocalHashEmbedder(256)),
    namespace=("users", USER_A, "memories"),
)
graph = build_memory_agent(memory, checkpointer=InMemorySaver())

print("=" * 66)
print("记忆系统已就绪：零依赖存储 + 脚本化模型 + LangGraph 图")
print("=" * 66)

title("第 1 幕｜首次会话（thread-a）：自我介绍被记住")
new_thread(memory, USER_A, "thread-a",
           "我叫程工，我不用 Windows，只认 Mac。回答尽量精简，结论先行。")
dump(memory, USER_A)

title("第 2 幕｜全新会话（thread-b）：空历史，但 Agent 认识她")
new_thread(memory, USER_A, "thread-b", "我用什么操作系统？回答该详细还是简洁？")

title("第 3 幕｜全新会话（thread-c）：模型主动写入一条新记忆")
new_thread(memory, USER_A, "thread-c", "我以后都用 Rust 写后端，记住这一点。")
dump(memory, USER_A)

title("第 4 幕｜全新会话（thread-d）：跨会话召回刚才写的")
new_thread(memory, USER_A, "thread-d", "我用什么语言写后端？")

title("第 5 幕｜另一位用户（bob）：同一张图，不同记忆")
new_thread(memory, USER_B, "thread-a", "我叫 Bob，我用 Windows。")
new_thread(memory, USER_B, "thread-b", "我用什么操作系统？")

title("第 6 幕｜隔离核验")
print(f"    bob 的记忆条数：{len(memory.store.list(USER_B))}")
print(f"    bob 的记忆里含 alice 信息的条数："
      f"{len([m for m in memory.store.list(USER_B) if '程工' in m.content])}")
print(f"    alice 的记忆里含 bob 信息的条数："
      f"{len([m for m in memory.store.list(USER_A) if 'Bob' in m.content])}")

print("\n完成。同一张编译好的图服务了两个用户，记忆互不干扰。")

"""完整剧本演示：一个技术助手 Agent 的长期记忆。

跑这个文件就能看到记忆系统的全部关键行为：
跨会话召回、偏好变更触发旧值失效、经历整合成经验、低价值记忆被遗忘、
以及多租户之间的硬隔离。

    python demo.py

不需要任何 API key，也不依赖第三方向量库。
"""

from __future__ import annotations

import time

from agent_memory import (
    InMemoryStore,
    LocalHashEmbedder,
    Memory,
    MemoryManager,
    MemoryType,
    utc_now,
)

USER_A = "alice"
USER_B = "bob"


def title(text: str) -> None:
    print(f"\n{'=' * 66}\n{text}\n{'=' * 66}")


def step(text: str) -> None:
    print(f"\n--- {text}")


def show_memories(memories, limit: int | None = None) -> None:
    if not memories:
        print("    （空）")
        return
    for mem in (memories[:limit] if limit else memories):
        score = f"  分数 {mem.score}" if mem.score is not None else ""
        print(f"    [{mem.type.value:<10}] {mem.content}{score}")


def build() -> MemoryManager:
    """组装记忆系统。换后端、换嵌入、换抽取器都只改这一处。"""
    store = InMemoryStore(embedder=LocalHashEmbedder(dim=256))
    return MemoryManager(store=store)


def act1_first_meeting(memory: MemoryManager) -> None:
    title("第 1 幕｜首次会话：Agent 从对话里挑出值得记的东西")

    step("alice 说话")
    text_a = "我叫程工，我不用 Windows，只认 Mac。回答尽量精简，结论先行。"
    print(f"    用户：{text_a}")
    report = memory.remember(USER_A, text_a)
    print(f"    抽取结果：{report.summary()}")
    show_memories(report.inserted)
    if report.merged:
        print(f"    合并说明：命中既有记忆，只刷新时间，不新增")

    step("alice 补充自己在做什么")
    text_a2 = "我在做一个搜索 Agent 的项目，后端用 Python 写。"
    print(f"    用户：{text_a2}")
    report = memory.remember(USER_A, text_a2)
    print(f"    抽取结果：{report.summary()}")
    show_memories(report.inserted)

    step("bob 说话（另一位用户，记忆互不相干）")
    text_b = "我叫 Bob，我用 Windows，编辑器是 VS Code。"
    print(f"    用户：{text_b}")
    report = memory.remember(USER_B, text_b)
    print(f"    抽取结果：{report.summary()}")
    show_memories(report.inserted)


def act2_cross_session(memory: MemoryManager) -> None:
    title("第 2 幕｜换个会话：上下文清零，但人还在")

    step("新会话开始，历史消息为空。alice 直接提问")
    query = "帮我看看这个检索延迟的问题"
    print(f"    用户：{query}")
    print("    （注意：这一轮没有任何历史消息，全部上下文由记忆系统提供）")

    step("检索到的长期记忆（低于阈值的噪声直接砍掉）")
    recalled = memory.recall(USER_A, query, k=5, min_score=0.02)
    show_memories(recalled)

    step("注入提示词的样子")
    context = memory.build_context(USER_A, query, k=4, min_score=0.02)
    print("    " + context.replace("\n", "\n    "))


def act3_preference_change(memory: MemoryManager) -> None:
    title("第 3 幕｜用户换系统了：旧值失效，但不删除")

    step("alice 说换了系统")
    text = "我换到 Ubuntu 了。"
    print(f"    用户：{text}")
    report = memory.remember(USER_A, text)
    print(f"    抽取结果：{report.summary()}")
    if report.inserted:
        print(f"    新增：{report.inserted[0].content}")
    for old in report.invalidated:
        print(f"    失效：{old.content}  （原因：同槽位出现新值）")

    step("这个槽位的完整历史（旧记录还在，只是不再参与检索）")
    for mem in memory.history_of_slot(USER_A, "preference.os"):
        state = "有效" if mem.is_active else f"已失效 -> 被 {mem.superseded_by} 取代"
        print(f"    [{state:<28}] {mem.content}")

    step("再问一次系统偏好，召回的是新值")
    recalled = memory.recall(USER_A, "我用什么操作系统", k=3)
    show_memories(recalled)


def act4_style_change(memory: MemoryManager) -> None:
    title("第 4 幕｜用户改主意了：回答风格被覆盖")

    step("alice 改了表达偏好")
    text = "其实以后回答详细一点，多举例。"
    print(f"    用户：{text}")
    report = memory.remember(USER_A, text)
    print(f"    抽取结果：{report.summary()}")
    for new in report.inserted:
        print(f"    新增：{new.content}")
    for old in report.invalidated:
        print(f"    失效：{old.content}")

    step("现在的风格记忆")
    recalled = memory.recall(USER_A, "你该怎么回答我", memory_types=[MemoryType.SEMANTIC], k=3)
    for mem in recalled:
        if mem.slot == "preference.style":
            print(f"    {mem.content}  （重要性 {mem.importance}）")


def act5_consolidate(memory: MemoryManager) -> None:
    title("第 5 幕｜经历攒够了：多条经验整合成一条结论")

    step("alice 陆续聊了三次检索延迟的排查（情景记忆）")
    episodes = [
        "上次排查检索延迟用了火焰图，定位到是慢查询",
        "上次排查检索延迟最后发现是缓存穿透，加了布隆过滤器",
        "上次排查检索延迟调了批量大小就好了",
    ]
    for text in episodes:
        print(f"    用户：{text}")
        memory.remember(USER_A, text, memory_type=MemoryType.EPISODIC)

    step("整合前的经历条目")
    show_memories(memory.recall(USER_A, "检索延迟怎么排查", memory_types=[MemoryType.EPISODIC], k=5))

    step("执行整合")
    created = memory.consolidate(USER_A, threshold=0.45, min_size=3)
    for mem in created:
        print(f"    生成结论：{mem.content}")
        print(f"    来源条数：{mem.metadata.get('count')} 条原始经历已标记失效")

    step("整合后再查，拿到的是结论而不是三条碎片")
    show_memories(memory.recall(USER_A, "检索延迟怎么排查", k=3, min_score=0.02))


def act6_forget(memory: MemoryManager) -> None:
    title("第 6 幕｜清理：低价值记忆该真删")

    step("写入一条很久以前、重要性很低的记忆（模拟历史脏数据）")
    stale = Memory(
        content="用户随口问过今天天气怎么样",
        user_id=USER_A,
        type=MemoryType.EPISODIC,
        importance=0.1,
        created_at=utc_now() - 200 * 86400,
        updated_at=utc_now() - 200 * 86400,
        last_access_at=utc_now() - 200 * 86400,
        source="simulated_history",
    )
    memory.store.add(stale)
    print(f"    {stale.content}  （200 天前，重要性 {stale.importance}）")

    step("先出计划，不动数据")
    plan = memory.forget(USER_A)
    for memory_id in plan.delete_ids:
        print(f"    待删除 {memory_id}：{plan.reasons[memory_id]}")

    step("确认执行")
    plan = memory.forget(USER_A, apply=True)
    print(f"    已删除 {len(plan.delete_ids)} 条低价值记忆")


def act7_isolation(memory: MemoryManager) -> None:
    title("第 7 幕｜租户隔离：同一个问题，两个人两个答案")

    query = "我用什么操作系统"
    step(f"bob 问「{query}」")
    show_memories(memory.recall(USER_B, query, k=3), limit=2)

    step(f"alice 问「{query}」")
    show_memories(memory.recall(USER_A, query, k=3), limit=2)

    step("交叉验证：alice 的记忆里搜不到 bob 的任何信息")
    cross = [m for m in memory.store.list(USER_A) if "Bob" in m.content or "VS Code" in m.content]
    print(f"    alice 名下含 bob 信息的记忆条数：{len(cross)}")

    step("越权读取测试：用 bob 的身份读 alice 的记忆 id")
    alice_memory = memory.store.list(USER_A)[0]
    stolen = memory.store.get(USER_B, alice_memory.id)
    print(f"    bob 拿着 alice 的记忆 id 去取：{stolen}")


def act8_report(memory: MemoryManager) -> None:
    title("第 8 幕｜体检报告")

    for user in (USER_A, USER_B):
        stats = memory.report(user)
        print(f"\n    用户 {user}")
        print(f"      记忆总数 {stats['total']}｜有效 {stats['active']}｜已失效 {stats['invalidated']}")
        print(f"      按类型 {stats['by_type']}")
        print(f"      平均重要性 {stats['avg_importance']}")
        print(f"      命名空间 {stats['namespaces']}")


def main() -> None:
    started = time.time()
    memory = build()

    act1_first_meeting(memory)
    act2_cross_session(memory)
    act3_preference_change(memory)
    act4_style_change(memory)
    act5_consolidate(memory)
    act6_forget(memory)
    act7_isolation(memory)
    act8_report(memory)

    memory.close()
    print(f"\n全部演示耗时 {time.time() - started:.2f} 秒。跑完它，记忆系统的主要行为就都见过了。")


if __name__ == "__main__":
    main()

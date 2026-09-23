"""交互式命令行：手动玩这套记忆系统。

    python cli.py

命令：
    /user <名字>     切换当前用户（隔离边界就跟着切）
    /say <文本>      当作这个用户说了一句话，交给抽取器
    /recall <问题>   检索长期记忆
    /context <问题>  看注入提示词的样子
    /list            列出当前用户的全部有效记忆
    /history <槽位>  看某个槽位被改写过几次（含已失效记录）
    /consolidate     把零散经历整合成结论
    /forget          按策略清理低价值记忆（会先给计划）
    /forget --apply  真的删
    /report          体检报告
    /help  /quit
"""

from __future__ import annotations

import sys

from agent_memory import InMemoryStore, LocalHashEmbedder, MemoryManager

BANNER = """记忆系统命令行。当前用户 alice。
先试试：/say 我叫程工，我不用 Windows，只认 Mac，回答尽量精简
然后：  /recall 我用什么操作系统
         /list
输入 /help 看全部命令，/quit 退出。"""


def build() -> MemoryManager:
    return MemoryManager(store=InMemoryStore(embedder=LocalHashEmbedder(256)))


def show(memories) -> None:
    if not memories:
        print("  （空）")
        return
    for mem in memories:
        score = f"  分数 {mem.score}" if mem.score is not None else ""
        state = "" if mem.is_active else "  [已失效]"
        print(f"  [{mem.type.value:<10}]{state} {mem.content}{score}")


def main() -> None:
    memory = build()
    user = "alice"
    print(BANNER)

    while True:
        try:
            raw = input(f"\n{user}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        if not raw.startswith("/"):
            report = memory.remember(user, raw)
            print(f"  抽取结果：{report.summary()}")
            show(report.inserted + report.merged)
            continue

        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command in ("/quit", "/exit", "/q"):
            break
        elif command == "/help":
            print(__doc__)
        elif command == "/user":
            if arg:
                user = arg
                print(f"  已切换到 {user}")
            else:
                print(f"  当前用户 {user}")
        elif command == "/say":
            if not arg:
                print("  用法：/say <文本>")
                continue
            report = memory.remember(user, arg)
            print(f"  抽取结果：{report.summary()}")
            show(report.inserted + report.merged)
            for old in report.invalidated:
                print(f"  旧值失效：{old.content}")
        elif command == "/recall":
            show(memory.recall(user, arg or "用户偏好", k=6, min_score=0.0))
        elif command == "/context":
            context = memory.build_context(user, arg or "用户偏好", k=5, min_score=0.01)
            print("  " + (context.replace("\n", "\n  ") if context else "（无可用记忆）"))
        elif command == "/list":
            show(memory.store.list(user))
        elif command == "/history":
            if not arg:
                print("  用法：/history <槽位>，例如 /history preference.os")
                continue
            for mem in memory.history_of_slot(user, arg):
                state = "有效" if mem.is_active else f"已失效 -> {mem.superseded_by}"
                print(f"  [{state:<26}] {mem.content}")
        elif command == "/consolidate":
            created = memory.consolidate(user)
            if not created:
                print("  没有达到整合阈值的经历（默认同主题至少 3 条）")
            for mem in created:
                print(f"  已生成结论：{mem.content}")
        elif command == "/forget":
            plan = memory.forget(user, apply=arg == "--apply")
            if not plan.delete_ids:
                print("  没有该忘的记忆")
            for memory_id in plan.delete_ids:
                print(f"  {memory_id}：{plan.reasons[memory_id]}")
            if plan.delete_ids and arg != "--apply":
                print("  以上只是计划，确认后加 --apply 真正删除")
        elif command == "/report":
            stats = memory.report(user)
            print(f"  记忆总数 {stats['total']}｜有效 {stats['active']}｜"
                  f"已失效 {stats['invalidated']}")
            print(f"  按类型 {stats['by_type']}｜平均重要性 {stats['avg_importance']}")
            print(f"  命名空间 {stats['namespaces']}")
        else:
            print(f"  未知命令 {command}，输入 /help 查看")

    memory.close()
    print("再见。")


if __name__ == "__main__":
    sys.exit(main())

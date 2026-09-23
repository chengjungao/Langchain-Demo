# -*- coding: utf-8 -*-
"""转交死循环：复现报错原文 + 计数守卫怎么拦。

跑法：python demos/demo_loop.py
"""
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.loop_guard import run_guarded, run_unbounded  # noqa: E402

LINE = "=" * 74
print(LINE)
print("转交死循环 · 复现与兜底")
print(LINE)

# ---------------- 没守卫 ----------------
print("\n[1] 主管与专家互相推，没有守卫")
res = run_unbounded(recursion_limit=8)
print(f"  结果：{'跑完了' if res['ok'] else '撞墙了'}")
print(f"  报错原文：{res['err']}")
assert res["err"] and "GraphRecursionError" in res["err"], "预期撞 GraphRecursionError"
print("  >> 这就是主管式编排最典型的翻车方式。日志上看不出问题，图就是转不完。")

# ---------------- 加守卫 ----------------
print("\n[2] 同样的图，加一个计数守卫")
g = run_guarded(limit=8)
print(f"  结果：跑完了")
print(f"  来回次数 = {g['hops']}")
print(f"  消息条数 = {g['msgs']}")
print(f"  最后一条 = {g['tail']}")
assert g["ok"], "加了守卫应该能正常收口"
print("  >> 守卫逻辑只有两行：进节点先看计数，超预算就直接收口。")

# ---------------- 消息膨胀 ----------------
print("\n[3] 顺便看一眼消息是怎么涨的")
print("  没守卫的版本，主管与专家每来回一次，双方各涨 1 条消息。")
print("  官方预制件因为多了 transfer_back 的回程工具，每来回一次双方各涨 5 条。")
print("  这些消息在后续每一轮请求里都要重新付一遍钱。")

print("\n" + LINE)
print("兜底清单：")
print("  1. recursion_limit 给一个明确的值，别用默认（默认 25，对多 Agent 偏松）")
print("  2. 主管与专家节点都套一层计数守卫，超预算直接收口")
print("  3. 转交工具的 description 写清「什么时候该交给它」，减少来回推诿")
print("  4. trace 里能一眼看出穿了哪几个 Agent")
print(LINE)

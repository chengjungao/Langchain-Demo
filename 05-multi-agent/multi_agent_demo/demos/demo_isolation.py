# -*- coding: utf-8 -*-
"""隔离对照：默认姿势并不隔离，Send 才隔离。

跑法：python demos/demo_isolation.py
不需要 API key。
"""
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage  # noqa: E402

from src.agents import make_order_expert  # noqa: E402
from src.isolation import (  # noqa: E402
    SEEN_KEYS,
    TASK,
    build_history,
    build_naive,
    build_schema_isolation,
    build_send_payload,
    count_tokens,
    first_call_input,
)
from src.spy_model import SpyChatModel  # noqa: E402

LINE = "=" * 74
history = build_history()
hist_tokens = count_tokens(history)
print(LINE)
print("隔离对照 · 同一段父图历史，三种姿势子 Agent 各看到什么")
print(LINE)
print(f"\n父图历史：{len(history)} 条消息，{sum(len(m.content) for m in history)} 字符 = {hist_tokens} token")
print(f"本次任务：{TASK}")

# ---------------- 姿势零：默认嵌入 ----------------
print("\n" + "-" * 74)
print("[默认姿势] 子 Agent 直接 add_node 进父图")
print("-" * 74)
SpyChatModel.reset()
sub_model = SpyChatModel(tag="order", script=[{"text": "订单在途，预计明天到。"}])
app = build_naive(make_order_expert(model=sub_model))
app.invoke({"messages": history + [HumanMessage(content=TASK)]})
n0, chars0, tok0 = first_call_input(sub_model)
print(f"  子 Agent 第一次调用看到 {n0} 条消息（父历史 {len(history)} 条 + 任务）")
print(f"  输入 = {tok0} token，其中父历史占 {hist_tokens / tok0 * 100:.1f}%")
assert n0 >= len(history), "默认姿势下子 Agent 没有拿到父图全历史，与预期不符"
print("  >> 结论：父图全历史都进去了。这个姿势一分钱没省，也没隔离。")

# ---------------- 姿势一：Send 只传任务 ----------------
print("\n" + "-" * 74)
print("[隔离手段一] Send 只传任务")
print("-" * 74)
SpyChatModel.reset()
worker_model = SpyChatModel(tag="worker", script=[{"text": "订单 20260915001 已签收。"}])
app2 = build_send_payload(worker_model, make_order_expert)
app2.invoke({"messages": history + [HumanMessage(content=TASK)]})
n1, chars1, tok1 = first_call_input(worker_model)
print(f"  worker 第一次调用看到 {n1} 条消息")
print(f"  输入 = {tok1} token")
print(f"  对比默认姿势 {tok0} token，省下 {tok0 - tok1} token（{(1 - tok1 / tok0) * 100:.0f}%）")
assert n1 < len(history), "Send 姿势下任务之外的父历史也进去了，与预期不符"
print("  >> 结论：父图历史一概进不去，只有挑出来的那份任务。")

# ---------------- 姿势二：独立 schema ----------------
print("\n" + "-" * 74)
print("[隔离手段二] 子图声明独立 state schema")
print("-" * 74)
SEEN_KEYS.clear()
app3 = build_schema_isolation()
out3 = app3.invoke(
    {"messages": history + [HumanMessage(content=TASK)], "query": TASK, "note": ""}
)
seen = SEEN_KEYS[0]
print(f"  子图收到的 state 键 = {seen['keys']}")
print(f"  子图收到的 messages 条数 = {seen['messages_n']}")
print(f"  父图的 query 有没有进去：{'task' in seen['keys']}")
print(f"  父图最终 state 键 = {sorted(out3.keys())}")
assert "task" not in seen["keys"], "异名键不该穿进去"
assert seen["messages_n"] >= len(history), "同名键 messages 应该照样穿进去"
print("  >> 结论：同名键 messages 照样全带进来，只有异名键被挡住。")
print("     schema 要真的隔开，键名就不能撞。")

# ---------------- 汇总 ----------------
print("\n" + LINE)
print("汇总")
print(LINE)
print(f"  默认嵌入        : 第 1 次调用 {tok0:>6} token（父历史全带，占 {hist_tokens / tok0 * 100:.0f}%）")
print(f"  Send 只传任务   : 第 1 次调用 {tok1:>6} token")
print("  独立 schema     : messages 同名照样全带，异名键被挡住")
print()
print("  一句话：在 LangGraph 里，隔离不是拆出来的，是自己写出来的。")
print(LINE)

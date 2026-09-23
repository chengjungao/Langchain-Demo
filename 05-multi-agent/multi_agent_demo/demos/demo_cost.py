# -*- coding: utf-8 -*-
"""三笔账：固定开销、窄任务、宽任务。

跑法：python demos/demo_cost.py
计数口径与上一篇《MCP 与 Skills 接入》完全一致，两篇的数字可以直接比。
"""
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.cost import WINDOW, fixed_overhead, narrow_task, wide_task  # noqa: E402

LINE = "=" * 74

# ---------------- 一、固定开销 ----------------
print(LINE)
print("一、固定开销（跟对话内容无关、每次请求都要重付的部分）")
print(LINE)
fx = fixed_overhead()
s, m = fx["single"], fx["multi"]
print("\n单 Agent（6 个工具）：")
print(f"   系统提示词        {s['system']:>6,} token")
print(f"   工具定义          {s['tools']:>6,} token")
print(f"   合计              {s['total']:>6,} token")
print("\n多 Agent（1 主管 + 3 专家）：")
print(f"   主管：提示词 {m['supervisor_system']:>5,} + 转交工具 {m['transfer_tools']:>5,} = {m['supervisor']:>6,}")
print(f"   订单专家          {m['order']:>6,} token")
print(f"   知识专家          {m['kb']:>6,} token")
print(f"   报表专家          {m['report']:>6,} token")
print(f"   合计              {m['total']:>6,} token")
print(f"\n   倍数：{m['total'] / s['total']:.2f}x")
print(f"   占 200K 窗口：单 Agent {s['total'] / WINDOW * 100:.2f}%   多 Agent {m['total'] / WINDOW * 100:.2f}%")

# ---------------- 二、窄任务 ----------------
print("\n" + LINE)
print("二、窄任务：一次请求只动 1 个专家")
print(LINE)
nr = narrow_task()
print(f"\n   单 Agent   {nr['single']['calls']} 次调用 / {nr['single']['tokens']:>6,} token")
print(f"   多 Agent   {nr['multi']['calls']} 次调用 / {nr['multi']['tokens']:>6,} token")
print(f"   倍数       {nr['ratio']:.2f}x   {'多 Agent 更省' if nr['ratio'] < 1 else '多 Agent 更贵'}")
print("\n   为什么省：每个专家手上只有自己那两三个工具，每次请求都更轻。")

# ---------------- 三、宽任务 ----------------
print("\n" + LINE)
print("三、宽任务：一次请求要动 3 个专家")
print(LINE)
wd = wide_task()
print(f"\n   单 Agent   {wd['single']['calls']} 次调用 / {wd['single']['tokens']:>6,} token（一轮并发发 3 个 tool_calls）")
print(f"   多 Agent   {wd['multi']['calls']} 次调用 / {wd['multi']['tokens']:>6,} token（主管串行转交 3 次）")
print(f"   倍数       {wd['ratio']:.2f}x   {'多 Agent 更省' if wd['ratio'] < 1 else '多 Agent 更贵'}")

# ---------------- 四、对照 ----------------
print("\n" + LINE)
print("对照")
print(LINE)
print(f"   {'场景':<26}{'单 Agent':>12}{'多 Agent':>12}{'倍数':>10}")
print(f"   {'固定开销':<26}{s['total']:>12,}{m['total']:>12,}{m['total'] / s['total']:>9.2f}x")
print(f"   {'窄任务（动 1 个专家）':<22}{nr['single']['tokens']:>12,}{nr['multi']['tokens']:>12,}{nr['ratio']:>9.2f}x")
print(f"   {'宽任务（动 3 个专家）':<22}{wd['single']['tokens']:>12,}{wd['multi']['tokens']:>12,}{wd['ratio']:>9.2f}x")
print()
print("  分水岭：一次请求要动几个专家。")
print("  动一个，多 Agent 省一半；动三个，多 Agent 贵一半，模型调用次数从 2 次涨到 10 次。")
print(LINE)

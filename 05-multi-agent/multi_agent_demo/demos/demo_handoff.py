# -*- coding: utf-8 -*-
"""手写交接：Command 的完整轨迹。

跑法：python demos/demo_handoff.py
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

from src.agents import make_all_experts  # noqa: E402
from src.handoff import build_handoff_graph, describe_trajectory  # noqa: E402
from src.spy_model import SpyChatModel  # noqa: E402

LINE = "=" * 74
print(LINE)
print("手写交接 · Command(goto=...) 的完整轨迹")
print(LINE)

SpyChatModel.reset()

# 主管脚本：依次把三件事交出去，最后自己收口
supervisor_model = SpyChatModel(tag="supervisor", script=[
    {"calls": [("order_expert", {"task": "查订单 SO20260915001 的状态"})]},
    {"calls": [("kb_expert", {"task": "查七天无理由的规则原文"})]},
    {"calls": [("report_expert", {"task": "出上周的退款率报表"})]},
])

# 三个专家：各跑一轮工具 + 一轮作答
experts = make_all_experts({
    "order": [
        {"calls": [("order_query", {"order_no": "SO20260915001", "tenant_id": "t1"})]},
        {"text": "订单已签收，可退 296.00。"},
    ],
    "kb": [
        {"calls": [("kb_search", {"query": "七天无理由", "top_k": 3, "tenant_id": "t1"})]},
        {"text": "签收后 7 天内可申请，运费不退。"},
    ],
    "report": [
        {"calls": [("report_build", {"metric": "refund_rate", "start_date": "09-07", "end_date": "09-13", "tenant_id": "t1"})]},
        {"text": "上周退款率 3.2%。"},
    ],
})

app = build_handoff_graph(supervisor_model, experts)
out = app.invoke(
    {"messages": [HumanMessage(content="三件事：查订单、查规则、出报表")], "turn": 0},
    {"recursion_limit": 40},
)

print("\n轨迹（按消息顺序）：")
for i, label in enumerate(describe_trajectory(out["messages"]), 1):
    print(f"  {i:>2}. {label}")

print(f"\n消息总数 = {len(out['messages'])}")
print(f"模型调用总数 = {len(SpyChatModel.calls)}")
for tag in ("supervisor", "order", "kb", "report"):
    n = len(SpyChatModel.calls_of(tag))
    if n:
        print(f"   {tag:<12} 被调用 {n} 次")

print("\n注意几件事：")
print("  1. 主管每轮都要重新读一遍会话历史，转交次数越多，它读的越多")
print("  2. 专家的回复会回到主管手里，主管负责收口")
print("  3. 这一版没有回程工具，专家跑完由图的边直接回主管")
print("     官方预制件会额外生成 transfer_back_to_supervisor，一次来回多 2 条消息")
print(LINE)

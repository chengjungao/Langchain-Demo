"""演示 5：长会话的指标采集 —— 治理到底省了多少。

跑一个 12 轮的会话，两种策略各一遍：不治理（全部历史都带上）、治理后（只保留
最近几条）。看单轮输入 token 怎么走。

这个演示要说明的是一件容易忽略的事：**监控不能只看「本轮平均 token」**。
单轮看每一轮都没超预算，累计账单却在按会话长度悄悄滑走。

运行：python demos/demo_monitor.py
"""
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src import paths  # noqa: E402
from src.monitor import aggregate, alert_rules, compare_sessions, print_sessions  # noqa: E402
from src.runtime import banner, force_offline, outbound_guard, section  # noqa: E402

force_offline()
banner("演示 5 · 长会话指标采集：单轮没超，账单在滑")

section("12 轮会话，两种策略")
with outbound_guard() as guard:
    cmp = compare_sessions(window=4)
print_sessions(cmp)
print(f"\n  对外连接被守卫拦下 {guard.blocked} 次（0 表示连试都没试过）")

section("这段曲线上该挂什么告警")
rows = cmp["ungoverned"]
alerts = alert_rules(rows)
for a in alerts:
    print(f"  ⚠ {a}")
print("\n  三条可告警的指标：")
print("    ① 会话内单轮输入 token 的斜率：连续 3 轮持续上升且没有回落，就该查上下文治理")
print("    ② 单会话累计 token 的 P95：比平均值更容易发现那批「聊得特别长」的会话")
print("    ③ 消息条数：它比 token 更早暴露问题，因为条数是线性涨、token 是加速涨")

section("治理后的曲线为什么走平")
print(f"  治理后单轮输入在 {cmp['governed_input_range'][0]} ~ {cmp['governed_input_range'][1]} 之间，")
print("  因为它的输入只跟「最近几条」有关，跟会话总长度无关。")
print(f"  不治理那条从 {cmp['first_turn_input']} 一路涨到 {cmp['last_turn_input']}，"
      f"第 {cmp['turns']} 轮是第 1 轮的 {cmp['growth_x']} 倍。")

# ---------------------------------------------------------------- 接上真实 trace
section("接上真实采集到的 trace（如果跑过演示 2）")
trace = paths.TRACES
if trace.exists():
    recs = [json.loads(ln) for ln in trace.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if recs:
        m = aggregate(recs)
        print(f"  {trace.name}：{m['runs']} 条 run，{m['sessions']} 个会话")
        print(f"  单次 llm 平均 {m['avg_tokens_per_llm_call']} token ｜ "
              f"耗时 p95 {m['latency_ms_p95']} ms ｜ 错误率 {m['error_rate'] * 100:.2f}%")
        print(f"  单会话平均 {m['tokens_per_session']} token")
    else:
        print(f"  {trace.name} 是空的")
else:
    print(f"  还没跑过演示 2（找不到 {trace.name}），跳过。")
    print("  先跑 python demos/demo_eval.py 就能在这里看到真实数据的聚合结果。")

section("滚动基线告警怎么用")
print("  阈值不拍绝对值，拿最近 N 次的中位数当参照，本次超过它的若干倍才报。")
print("  这样业务量涨了、模型换了，阈值会跟着自己走。")
print("  样本不足 N 次时不报（宁可不报，也不要一开始天天误报）。")
print("\n  命令行：python -m src.monitor --trace reports/traces.jsonl --factor 1.5")

print("\n" + "=" * 84)
print("演示 5 结束")
print("=" * 84)

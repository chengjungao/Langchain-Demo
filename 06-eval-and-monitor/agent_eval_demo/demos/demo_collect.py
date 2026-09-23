"""演示 1：本地 trace 采集 —— 不上平台，能不能把一次请求的账算清楚。

跑一个带工具调用的 Agent，用 collect_runs 抓 trace，看四件事：

  1. 一次请求抓到几条 run
  2. 调用链是不是拍平的（对照：普通链式 Runnable 有完整父子关系）
  3. token 到底藏在哪一层
  4. 不带凭据时，本地采集能不能独立完成

运行：python demos/demo_collect.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from src import paths  # noqa: E402
from src.agent import build_tool_calling_agent, invoke_quiet  # noqa: E402
from src.runtime import banner, force_offline, outbound_guard, section  # noqa: E402
from src.trace_store import TraceStore, is_flat, print_summary, run_to_record  # noqa: E402

force_offline()
banner("演示 1 · 本地 trace 采集：不上云也能把账算清楚")

# ---------------------------------------------------------------- 对照组
section("对照组：普通链式 Runnable 的父子关系")
chain = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
with outbound_guard() as g:
    _ = chain.invoke(1)
print(f"  对外连接被守卫拦下 {g.blocked} 次（本地回环放行）")
print("  链式 Runnable 不产生 run 记录，除非显式 @traceable。")
print("  下面换成真正带 run 的两条路对比。")

# ---------------------------------------------------------------- 主体
section("主体：一次带工具调用的 Agent 请求")
agent = build_tool_calling_agent()
out, runs = invoke_quiet(agent, [HumanMessage("查订单 A1001")])

print(f"  Agent 最终回复：{out['messages'][-1].content}")
print(f"  抓到的 run：{len(runs)} 条")

print(f"\n  {'name':<18}{'type':<8}{'parent_run_id':<16}{'耗时ms':<10}{'入tok':<8}{'出tok':<8}{'工具调用数'}")
print("  " + "-" * 76)
for r in runs:
    rec = run_to_record(r)
    print(f"  {rec['name'][:17]:<18}{rec['run_type']:<8}{(rec['parent_run_id'] or 'None')[:15]:<16}"
          f"{str(rec['ms']):<10}{str(rec.get('input_tokens') or '-'):<8}"
          f"{str(rec.get('output_tokens') or '-'):<8}{str(rec.get('tool_call_count') or '-')}")

types: dict[str, int] = {}
for r in runs:
    types[r.run_type] = types.get(r.run_type, 0) + 1
print(f"\n  类型分布：" + "、".join(f"{v} 条 {k}" for k, v in sorted(types.items())))

flat = is_flat([run_to_record(r) for r in runs])
print(f"\n  调用链是否拍平：{'是' if flat else '否'}")
if flat:
    print("    每条 run 的 parent_run_id 都是 None，dotted_order 全是单段，")
    print("    根 run 的 child_runs 是空的。想重建时序只能按 start_time 排序。")

section("重建时序（按 start_time 排序）")
for i, r in enumerate(sorted(runs, key=lambda r: r.start_time), 1):
    print(f"    {i}. {r.start_time.strftime('%H:%M:%S.%f')[:-3]}  {r.run_type:<6}{r.name}")
print(f"\n  dotted_order 示例：{runs[0].dotted_order}")
print("    单段 = 平级。如果有层级，这里会是 20260917T....xxxx.yyyy 这种多点结构。")

section("token 藏在哪一层")
llm_runs = [r for r in runs if r.run_type == "llm"]
if llm_runs:
    r = llm_runs[0]
    print(f"  想直接取？{hasattr(r, 'total_tokens')}")
    print("    run 上没有 total_tokens / prompt_tokens 这类现成属性，getattr 取不到。")
    cell = r.outputs["generations"][0][0]
    msg = cell["message"]
    kw = msg.get("kwargs", {})
    print("\n  真值路径：run.outputs['generations'][0][0]['message']['kwargs']['usage_metadata']")
    print(f"    ⚠ 注意这里 message 是序列化后的 dict，不是对象，属性访问会报 AttributeError")
    print(f"    usage_metadata = {kw.get('usage_metadata')}")
    print(f"    tool_call_count 在 run.extra 里：{(r.extra or {}).get('tool_call_count')}")

section("顺带记够监控要用的字段")
store = TraceStore(paths.REPORTS / "traces_collect.jsonl")
store.clear()
n = store.append(runs, session="collect-demo", sample_id="C01")
print(f"  已落盘 {n} 条 run → {store.path.name}")
print_summary(store.summary(), "汇总（这些字段就是监控看板的数据源）")

section("结论")
print("  不上平台，本地也能拿到：条数、类型、耗时、输入输出 token、工具调用数、错误。")
print("  两个坑记住：run 上没有现成的 token 属性；LangGraph 会把调用链拍平。")

print("\n" + "=" * 84)
print("演示 1 结束")
print("=" * 84)

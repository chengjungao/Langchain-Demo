"""演示 2：离线评测 —— 两版 Agent 跑同一套评测集，看分档分数。

同一套 6 条样本，两个版本：
  good  全部答对
  bad   常见类目照旧答对，长尾与多意图全部塞进常见类目，并编造一个订单号

重点不在「谁分高」，而在**分档看**：只看总分你看到的是掉了三成，看分档才看到
简单样本一条没坏、困难样本崩了三分之二。

运行：python demos/demo_eval.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src import dataset as ds  # noqa: E402
from src import paths  # noqa: E402
from src.runner import build_target, run_offline  # noqa: E402
from src.runtime import banner, force_offline, outbound_guard, section  # noqa: E402
from src.trace_store import TraceStore, print_summary  # noqa: E402

force_offline()
banner("演示 2 · 离线评测：同一套尺子量两个版本")

# ---------------------------------------------------------------- 评测集
section("评测集")
items = ds.load(paths.EVAL_SET)
problems = ds.validate(items)
print(f"  读到 {len(items)} 条：{items[0]['id']} ~ {items[-1]['id']}")
for it in items:
    print(f"    {it['id']}  {it['tier']:<6}{it['msg'][:22]:<24}→ {it['intent']}/{it['priority']}")
if problems:
    print(f"\n  ✘ 评测集有问题：{problems}")
    sys.exit(1)
print("\n  ✔ 校验通过（每行有 tier，easy 与 hard 两档都在）")

fp = ds.freeze(paths.EVAL_SET, paths.EVAL_SET_FROZEN)
ok, msg = ds.check_frozen(paths.EVAL_SET, paths.EVAL_SET_FROZEN)
print(f"\n  版本冻结：{fp['version']}（{fp['samples']} 条，{fp['tiers']}）")
print(f"  冻结校验：{msg}")

# ---------------------------------------------------------------- 跑两个版本
store = TraceStore(paths.TRACES)
store.clear()

reports = {}
with outbound_guard() as guard:
    for variant in ("good", "bad"):
        section(f"跑 {variant} 版")
        target = build_target(variant=variant, trace_store=store, session=variant)
        rep = run_offline(target, items, variant=variant,
                          eval_set_version=fp["version"])
        reports[variant] = rep
        rep.print_report(f"评测结果 · {variant} 版")
        rep.save(paths.REPORTS / f"current_{variant}.json")

print(f"\n  对外连接被守卫拦下 {guard.blocked} 次"
      "（0 表示整个过程连试都没试过，不是「试了但被拦住」）")

section("顺带采集到的监控数据")
print_summary(store.summary(), "两个版本的 run 合在一起（正式项目里 trace 是持续上报的）")

# ---------------------------------------------------------------- 对照
good, bad = reports["good"], reports["bad"]
section("两版对照")
print(f"  {'':<22}{'good':<10}{'bad':<10}{'变化'}")
print("  " + "-" * 52)
for k in good.per_key:
    g, b = good.per_key[k], bad.per_key.get(k, 0)
    mark = "  <== 掉了" if g - b > 0.01 else ""
    print(f"  {k:<22}{g:<10}{b:<10}{-(g - b):+.3f}{mark}")

print(f"\n  {'分档（意图准确率）':<22}{'good':<10}{'bad':<10}{'变化'}")
print("  " + "-" * 52)
for t in ("easy", "hard"):
    key = f"{good.primary}@{t}"
    g, b = good.per_key_tier.get(key, 0), bad.per_key_tier.get(key, 0)
    print(f"  {t:<22}{g:<10}{b:<10}{-(g - b):+.3f}")

print(f"\n  主指标 overall：good {good.overall} → bad {bad.overall}")
print(f"  只看总分，你看到的是掉了 {(good.overall - bad.overall) * 100:.0f}%。")
print(f"  看分档：easy 档 {good.per_key_tier.get(f'{good.primary}@easy')} 一条没坏，"
      f"hard 档掉到 {bad.per_key_tier.get(f'{good.primary}@hard')}。")

section("这一版真正的教训")
jv = bad.per_key.get("json_valid")
print(f"  json_valid 在 bad 版仍然是 {jv}，它一条问题都没报出来。")
print("  结构没坏、话术没坏、订单号也只是编了一个 —— 单看任何一个判定器都可能放行。")
print("  门禁要靠多个判定器组合着看，这就是为什么判定器不是一个而是六个。")

print("\n  trace 已写入：" + (paths.TRACES).name)
print("  接着可以跑：python -m src.monitor --trace reports/traces.jsonl")

print("\n" + "=" * 84)
print("演示 2 结束")
print("=" * 84)

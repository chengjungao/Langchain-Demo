"""演示 7：官方 evaluate 的三种 data 形态 —— 报三种不同的错。

如果你想走官方链路，这个坑必须先知道：`evaluate()` 的 `data` 参数有三种写法，
配错一个就会在几百行之外崩掉。

  A  data = list[dict] + 默认上传          → 报错
  B  data = list[dict] + upload_results=False → 表面跑通，一迭代就崩
  C  data = list[Example] + upload_results=False → 完全跑通

全程不需要任何云端凭据，也不需要联网（脚本会把对外网络断掉）。

运行：python demos/demo_official.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src import dataset as ds  # noqa: E402
from src import paths  # noqa: E402
from src.agent import build_agent, invoke_quiet, parse_output  # noqa: E402
from src.runner import run_official  # noqa: E402
from src.runtime import banner, force_offline, outbound_guard, section  # noqa: E402

from langchain_core.messages import HumanMessage  # noqa: E402

force_offline()
banner("演示 7 · 官方 evaluate：三种 data 形态，三种错法")

items = ds.load(paths.EVAL_SET)
agent = build_agent(variant="good")


def target(inputs: dict) -> dict:
    """注意 invoke_quiet：它把 LangChain 自己的 run 上报关掉了。

    不关的话，日志里会刷一片 401。evaluate(upload_results=False) 内部会把追踪
    上下文设成 local 模式，那个模式管得住 langsmith 自己的 @traceable run，
    但管不住 LangChain 的 run。
    """
    pred, _ = invoke_quiet(agent, [HumanMessage(inputs["msg"])], capture=False)
    return parse_output(pred)


def as_dicts():
    return [{"inputs": {"msg": it["msg"], "tier": it["tier"]},
             "outputs": {"intent": it["intent"], "priority": it["priority"],
                         "order_id": it["order_id"]}} for it in items]


def evaluator(run, example):
    from langsmith.evaluation import EvaluationResult

    pred = (run.outputs or {}).get("intent")
    gold = (example.outputs or {}).get("intent")
    return EvaluationResult(key="intent_exact", score=float(pred == gold))


def show_exc(tag: str, e: Exception) -> None:
    print(f"    {tag}：{type(e).__name__}: {str(e)[:110]}")


with outbound_guard() as guard:
    # ------------------------------------------------------------ A
    section("A  data = list[dict]，用默认上传")
    from langsmith import evaluate

    try:
        evaluate(target, data=as_dicts(), evaluators=[evaluator],
                 experiment_prefix="shape-a", max_concurrency=1)
        print("    ✘ 竟然没报错")
    except Exception as e:
        show_exc("报错", e)
    print("    这是网上教程里最常见的写法。报错内容与凭据无关，是 data 的形态问题。")
    print("    容易把人带到错方向：以为是 key 没配好。")

    # ------------------------------------------------------------ B
    section("B  data = list[dict]，配 upload_results=False")
    from langsmith import evaluate as ev_b

    res_b = ev_b(target, data=as_dicts(), evaluators=[evaluator],
                 experiment_prefix="shape-b", upload_results=False, max_concurrency=1)
    print("    调用本身没报错，看起来跑通了：")
    print(f"      experiment_name = {getattr(res_b, 'experiment_name', None)!r}")
    try:
        rows = list(res_b)
        print(f"      ✘ 迭代成功，拿到 {len(rows)} 条（本环境与实测时不同）")
    except Exception as e:
        show_exc("一迭代就崩", e)
    print("    表面成功，可能等你写完解析脚本才发现结果一条都读不出来。")

    # ------------------------------------------------------------ C
    section("C  data = list[Example]，配 upload_results=False")
    res_c, rep_c = run_official(target, items, experiment_prefix="shape-c")
    print(f"    ✔ 完全跑通：{rep_c.n} 条样本，主指标 {rep_c.primary} = {rep_c.overall}")
    print(f"      experiment_name = {res_c.experiment_name!r}")
    try:
        print(f"      get_dataset_id() = {res_c.get_dataset_id()!r}")
    except Exception as e:
        show_exc("get_dataset_id", e)

    print("\n    离线模式取不到的那几个属性：")
    for name in ("experiment_id", "url", "comparison_url"):
        try:
            print(f"      {name} = {getattr(res_c, name)!r}")
        except Exception as e:
            show_exc(name, e)
    print("    能跑门禁，回链不到平台页面。这是离线评测的真实边界，够用。")

    print("\n    逐条明细：")
    for r in rep_c.rows:
        print(f"      {r['id']:<7}{r['tier']:<6}{r['scores']}")

print(f"\n  对外连接被守卫拦下 {guard.blocked} 次 —— 三种形态全部在没有凭据、没有外网的环境下跑完。")
print("  0 表示整个过程连试都没试过。这也是为什么 invoke_quiet() 那一步不能省：")
print("  不关掉 LangChain 的 run 上报，后台线程会拿着空 key 去连，守卫就会开始计数。")

section("小结")
print("  A 当场报错，你会去查。")
print("  B 表面成功，这才是危险的那个。")
print("  C 完全可用，代价是 experiment_id 与 url 取不到。")
print("  本包的 runner.run_offline() 绕开了这一整套：直接用普通 dict 跑，")
print("  零依赖、零网络、结果一条不落地能读出来。要用官方链路时再切到 run_official()。")

print("\n" + "=" * 84)
print("演示 7 结束")
print("=" * 84)

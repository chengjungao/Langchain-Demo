"""演示 4：回放缓存 —— 让评测不烧钱。

把模型调用按 (模型标识 + system prompt + 用户输入) 的哈希落盘，重跑时直接读盘。
跑四轮看它的实际效果，外加一组不缓存的对照。

第 4 轮是最值得看的一轮：只在 system prompt 里加一句话，全部未命中。
缓存 key 里必须包含 prompt，少这一项你会拿到过期答案，而且是静默的。

运行：python demos/demo_replay.py
"""
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src import dataset as ds  # noqa: E402
from src import paths  # noqa: E402
from src.cassette import Cassette, CountedModel  # noqa: E402
from src.runtime import banner, force_offline, section  # noqa: E402

force_offline()
banner("演示 4 · 最小回放缓存：评测跑一百遍也只花第一遍的钱")

items = ds.load(paths.EVAL_SET)
QUESTIONS = [it["msg"] for it in items]
EXTRA = ["订单能改地址吗", "优惠券没生效"]
SYSTEM_V1 = "你是客服工单分诊助手，只输出 JSON。"
SYSTEM_V2 = "你是客服工单分诊助手，判断意图与优先级，只输出 JSON。"  # 只加了一句

cache = Cassette(paths.CASSETTES)
removed = cache.clear()
section("准备")
print(f"  评测集 {len(QUESTIONS)} 条 + 待扩的 {len(EXTRA)} 条")
print(f"  清空缓存目录 {paths.CASSETTES.name}（删掉 {removed} 个旧文件）")

rows = []


def run_round(name: str, questions: list[str], sys_prompt: str, no_cache: bool = False) -> dict:
    model = CountedModel("scripted-triage-v1", sys_prompt)
    hits = misses = 0
    t0 = time.perf_counter()
    for q in questions:
        if no_cache:
            CountedModel.no_cache(q)
            model.real_calls += 1
            status = "nocache"
        else:
            _, status = model.invoke(q, cache)
        hits += status == "hit"
        misses += status == "miss"
    wall = time.perf_counter() - t0
    row = {"name": name, "samples": len(questions), "real": model.real_calls,
           "hits": hits, "misses": misses, "wall": wall, "files": cache.count()}
    rows.append(row)
    return row


section("四轮 + 一组对照")
print(f"  {'轮次':<26}{'样本':<7}{'真实调用':<11}{'命中':<7}{'未命中':<9}{'耗时':<11}{'缓存文件'}")
print("  " + "-" * 82)
run_round("第 1 轮 冷启动", QUESTIONS, SYSTEM_V1)
run_round("第 2 轮 热重跑", QUESTIONS, SYSTEM_V1)
run_round("第 3 轮 扩了 2 条样本", QUESTIONS + EXTRA, SYSTEM_V1)
run_round("第 4 轮 改了 system prompt", QUESTIONS, SYSTEM_V2)
run_round("对照：不用缓存", QUESTIONS, SYSTEM_V1, no_cache=True)

for r in rows:
    print(f"  {r['name']:<26}{r['samples']:<7}{r['real']:<11}{r['hits']:<7}"
          f"{r['misses']:<9}{r['wall']:.3f}s{'':<6}{r['files']}")

r1, r2, r3, r4, r5 = rows
section("读法")
print(f"  第 1 轮是最贵的一轮：{r1['real']} 次真实调用。")
print(f"  第 2 轮同样的评测集：真实调用 {r2['real']} 次，耗时从 {r1['wall']:.2f}s 降到 {r2['wall']:.2f}s。")
print(f"  第 3 轮只在新增的 {len(EXTRA)} 条上付钱：真实调用 {r3['real']} 次。")
print(f"  第 4 轮 system prompt 改了一句，真实调用回到 {r4['real']} 次 —— key 里必须包含 prompt。")
print(f"  对照：完全不缓存，每一轮都是 {r5['real']} 次真实调用、{r5['wall']:.2f}s。")

section("累计省下多少")
total_with = r1["real"] + r2["real"] + r3["real"] + r4["real"]
total_without = sum(r["samples"] for r in rows[:4])
print(f"  四轮下来：带缓存 {total_with} 次真实调用，不带缓存 {total_without} 次。")
print(f"  省掉 {total_without - total_with} 次，约 "
      f"{(1 - total_with / total_without) * 100:.0f}%。")
print("  演示里每次调用只睡 0.08 秒。真实模型一次几百毫秒到几秒，省的是钱。")

section("它测不了什么")
print("  换模型那一版，缓存命中意味着你根本没调模型，新模型的表现看不出来。")
print("  任何依赖采样随机性的环节，回放也测不出来。")
print("  做法是分两条线：日常改动用缓存跑，换模型和发版前跑真实调用。")

section("三条纪律")
print("  ① key = 模型标识 + system prompt + 用户输入。少任何一项都会拿错答案。")
print("  ② 缓存目录提交进仓库，CI 里就能零调用跑回归。")
print("     官方对 LANGSMITH_TEST_CACHE 也是这个建议（注意它需要额外装 vcrpy，")
print("     环境里默认没有，不装的话缓存开关是静默不生效的）。")
print("  ③ 缓存损坏或缺失时回退到真实调用，不要抛错。缓存是加速器，不是依赖。")

# ---------------------------------------------------------------- 自检
section("自检")
expect = {"r1_real": 6, "r2_real": 0, "r3_real": 2, "r4_real": 6}
actual = {"r1_real": r1["real"], "r2_real": r2["real"],
          "r3_real": r3["real"], "r4_real": r4["real"]}
ok = actual == expect
for k, want in expect.items():
    got = actual[k]
    print(f"  {k:<10}实际 {got:<6}期望 {want:<6}{'✔' if got == want else '✘'}")
print("  ✔ 行为符合预期" if ok else "  ✘ 与预期不符（评测集条数或 system prompt 被改过？）")

print("\n" + "=" * 84)
print("演示 4 结束")
print("=" * 84)
sys.exit(0 if ok else 1)

"""演示 3：门禁 —— 跑不过就拦住发布。

这个脚本演的不是「怎么跑评测」，而是「怎么让评测真的能拦住东西」。

做法：
  1. 好的版本跑一遍，记成基线
  2. 拿基线判定好的版本     → 期望退出码 0
  3. 人为退化的版本跑一遍
  4. 拿基线判定退化的版本   → 期望退出码 1

第 4 步是真去调命令行、真的读退出码，不是打印一行红字。只打印不改退出码的
脚本，在 CI 里永远是绿的。

运行：python demos/demo_gate.py
"""
import json
import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src import dataset as ds  # noqa: E402
from src import paths  # noqa: E402
from src.gate import decide, print_decision  # noqa: E402
from src.runner import build_target, run_offline  # noqa: E402
from src.runtime import banner, force_offline, section  # noqa: E402
from src.trace_store import TraceStore  # noqa: E402

force_offline()
banner("演示 3 · 门禁：能阻塞的评测才算门禁")

items = ds.load(paths.EVAL_SET)
fp = ds.freeze(paths.EVAL_SET, paths.EVAL_SET_FROZEN)
store = TraceStore(paths.TRACES)
store.clear()


def run_variant(variant: str):
    target = build_target(variant=variant, trace_store=store, session=variant)
    rep = run_offline(target, items, variant=variant, eval_set_version=fp["version"])
    rep.save(paths.CURRENT)
    print(f"  {variant} 版：overall {rep.overall}（{rep.per_key}）")
    return rep


def call_gate(*extra: str) -> int:
    """真去跑命令行，拿真实退出码。"""
    cmd = [sys.executable, "-m", "src.gate",
           "--current", str(paths.CURRENT), "--baseline", str(paths.BASELINE), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_root))
    out = (p.stdout or "") + (p.stderr or "")
    print("\n" + "\n".join("  | " + ln for ln in out.strip().splitlines() if ln.strip()))
    return p.returncode


results: list[tuple[str, int, int]] = []  # (说明, 实际退出码, 期望退出码)

# ---------------------------------------------------------------- 1 记基线
section("第 1 步：好的版本跑一遍，记成基线")
good = run_variant("good")
code = call_gate("--write-baseline")
results.append(("记基线（--write-baseline）", code, 0))
print(f"\n  基线已写入 {paths.BASELINE.name}")

# ---------------------------------------------------------------- 2 好的版本应放行
section("第 2 步：拿基线判定好的版本，期望放行（exit 0）")
code = call_gate()
results.append(("判定 good 版（期望放行）", code, 0))

# ---------------------------------------------------------------- 3 退化版本
section("第 3 步：跑人为退化的版本")
bad = run_variant("bad")
print("\n  退化版的改动只有三处：")
print("    E04 东西坏了但我更想直接退钱   → 意图答成 logistics")
print("    E05 发票要改公司名+少一件      → 优先级该 high 答成 normal")
print("    E06 问保修到什么时候           → 意图答成 invoice，还编了个订单号 A9999")

# ---------------------------------------------------------------- 4 退化版本应拦住
section("第 4 步：拿基线判定退化版本，期望拦住（exit 1）")
code = call_gate()
results.append(("判定 bad 版（期望拦住）", code, 1))

# ---------------------------------------------------------------- 5 只改阈值的对照
section("第 5 步：对照 —— 阈值放宽到 0.6 会怎样")
code = call_gate("--threshold", "0.6")
results.append(("阈值放宽到 0.6", code, 0))
print("\n  这一版每个判定器都掉了，但阈值放到 0.6 就一条都拦不住。")
print("  阈值本身就是一次决策：太松，真退化也放行。")

# ---------------------------------------------------------------- 6 分档定位
section("第 6 步：门禁能不能定位到问题")
dec = decide(
    json.loads(paths.BASELINE.read_text(encoding="utf-8")),
    json.loads(paths.CURRENT.read_text(encoding="utf-8")),
    threshold=0.05,
)
print_decision(dec)
print(f"\n  定位结论：{dec['hint'] if dec['hint'] else '—'}")

# ---------------------------------------------------------------- 汇总
section("自检汇总")
ok = True
print(f"  {'步骤':<32}{'实际':<8}{'期望':<8}{'结果'}")
print("  " + "-" * 58)
for name, got, want in results:
    flag = "✔" if got == want else "✘"
    if got != want:
        ok = False
    print(f"  {name:<32}{got:<8}{want:<8}{flag}")

if ok:
    print("\n  ✔ 门禁行为全部符合预期：好的放行、退化拦住、阈值太松就会漏")
else:
    print("\n  ✘ 有步骤不符合预期，检查上面的输出")

print("\n" + "=" * 84)
print("演示 3 结束")
print("=" * 84)
sys.exit(0 if ok else 1)

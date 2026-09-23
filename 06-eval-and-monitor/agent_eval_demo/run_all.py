"""一键跑通全部演示与测试。

这个脚本的存在是为了让「验收」这件事可重复：解压 → 建 venv → 装依赖 →
python run_all.py，所有演示和测试都跑一遍，最后给出汇总。

判定标准：
  除 demo_gate 内部会故意触发一次非 0 退出码（它就是要证明门禁能拦住东西，
  这是脚本自己的自检，对外仍返回 0）之外，其余脚本都该返回 0。

run_all.py 自己返回 0，说明「全部脚本的行为都符合设计预期」。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEMOS = [
    ("演示 1  本地 trace 采集", "demos/demo_collect.py", []),
    ("演示 2  离线评测（good / bad 两版）", "demos/demo_eval.py", []),
    ("演示 3  门禁（含退化拦截自检）", "demos/demo_gate.py", []),
    ("演示 4  回放缓存", "demos/demo_replay.py", []),
    ("演示 5  长会话指标采集", "demos/demo_monitor.py", []),
    ("演示 6  脱敏规则对照", "demos/demo_redact.py", []),
    ("演示 7  官方 evaluate 三种 data 形态", "demos/demo_official.py", []),
    ("演示 8  LLM 裁判（本地模型，不在就跳过）", "demos/demo_judge.py", []),
]

MONITOR_CLI = ("监控命令行 src.monitor", ["-m", "src.monitor"])


def run(cmd: list[str], quiet: bool = True) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT),
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if not quiet:
        print(out)
    return p.returncode, out


def main() -> int:
    t0 = time.time()
    print("=" * 84)
    print("agent_eval_demo · 一键验收")
    print("=" * 84)
    print(f"  解释器 {sys.executable}")
    print(f"  目录   {ROOT}")

    results: list[tuple[str, int, str]] = []

    print("\n" + "-" * 84)
    print("演示脚本")
    print("-" * 84)
    for name, rel, extra in DEMOS:
        code, out = run([sys.executable, rel, *extra])
        note = ""
        if code != 0:
            tail = [ln for ln in out.strip().splitlines() if ln.strip()][-4:]
            note = " | ".join(tail)[:160]
        results.append((name, code, note))
        print(f"  {name:<44}exit {code}{'  ' + note if note else ''}")

    print("\n" + "-" * 84)
    print("命令行工具")
    print("-" * 84)
    code, out = run([sys.executable, *MONITOR_CLI[1], "--trace", "reports/traces.jsonl",
                     "--no-save"])
    note = ""
    if code != 0:
        note = " | ".join([ln for ln in out.strip().splitlines() if ln.strip()][-3:])[:160]
    results.append((MONITOR_CLI[0], code, note))
    print(f"  {MONITOR_CLI[0]:<44}exit {code}{'  ' + note if note else ''}")

    code, out = run([sys.executable, "-m", "src.gate", "--write-baseline"])
    results.append(("门禁 · 记基线", code, ""))
    print(f"  {'门禁 · 记基线':<44}exit {code}")

    code, out = run([sys.executable, "-m", "src.gate"])
    results.append(("门禁 · 判定（当前=基线，应放行）", code, ""))
    print(f"  {'门禁 · 判定（当前=基线，应放行）':<44}exit {code}")

    print("\n" + "-" * 84)
    print("冒烟测试")
    print("-" * 84)
    code, out = run([sys.executable, "-m", "pytest", "tests", "-q"])
    hits = [ln.strip(" .=+") for ln in out.splitlines()
            if "passed" in ln or "failed" in ln or "error" in ln.lower()]
    summary_line = hits[-1] if hits else ""
    results.append(("pytest tests", code, summary_line))
    print(f"  {'pytest tests':<44}exit {code}  {summary_line}")

    # ---------------------------------------------------------------- 汇总
    print("\n" + "=" * 84)
    print("汇总")
    print("=" * 84)
    print(f"  {'项目':<44}{'退出码'}")
    print("  " + "-" * 56)
    bad = []
    for name, code, _ in results:
        mark = "✔" if code == 0 else "✘"
        print(f"  {name:<44}{code}  {mark}")
        if code != 0:
            bad.append(name)

    print(f"\n  共 {len(results)} 项，{len(results) - len(bad)} 项通过，耗时 {time.time() - t0:.1f}s")
    if bad:
        print(f"  ✘ 未通过：{bad}")
        print("\n  提示：演示 8 需要本地模型，没起服务时它会跳过并返回 0。")
        print("       如果它返回了非 0，先看它的输出。")
        return 1

    print("\n  ✔ 全部通过。")
    print("    门禁的退化拦截、缓存的零调用、分档分数的复现，都在这轮里验过了。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())

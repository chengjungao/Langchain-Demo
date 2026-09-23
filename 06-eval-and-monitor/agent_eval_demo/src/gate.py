"""门禁：跑不过就拦住发布。

一个门禁要能满足三件事，缺一件在 CI 里就是废的：

  1. 能阻塞 —— 不通过时返回非 0 退出码。只打印一行红字不改退出码的脚本，
     在 CI 里永远是绿的。
  2. 阈值用滚动基线 —— 不拍绝对值。绝对阈值要么定得太松抓不到退化，要么
     定得太紧天天误报。
  3. 失败能定位 —— 只报一个总分没有用。要能说出是哪个判定器、哪一档掉的。

定位逻辑里最有价值的一条：**简单样本没坏、困难样本崩了，说明是模型退化，
不是评测数据变难了**。这就是评测集里必须留 easy 档的原因。

用法：
  python -m src.gate --write-baseline          # 把当前结果记成基线
  python -m src.gate                            # 判定，通过 exit 0，退化 exit 1
  python -m src.gate --threshold 0.03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import paths
from . import evaluators as ev


def load_json(path: Path | str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def decide(baseline: dict, current: dict, threshold: float = 0.05,
           primary: str | None = None) -> dict:
    """比对基线，判定是否放行。"""
    primary = primary or current.get("primary") or ev.PRIMARY
    base_overall = baseline.get("overall", 0.0)
    cur_overall = current.get("overall", 0.0)
    drop = round(base_overall - cur_overall, 4)

    # 逐个判定器看掉了多少
    drops: dict[str, dict] = {}
    for k, cur in (current.get("scores") or {}).items():
        base = (baseline.get("scores") or {}).get(k)
        if base is None:
            continue
        d = round(base - cur, 4)
        drops[k] = {"baseline": base, "current": cur, "drop": d, "over": d > threshold}

    # 分档看：定位是长尾退化还是数据变难
    tier_drops: dict[str, dict] = {}
    for k, cur in (current.get("tier") or {}).items():
        base = (baseline.get("tier") or {}).get(k)
        if base is None:
            continue
        tier_drops[k] = {"baseline": base, "current": cur, "drop": round(base - cur, 4)}

    failed = drop > threshold or any(v["over"] for v in drops.values())

    hint = ""
    if failed:
        for t in ("easy", "hard"):
            key = f"{primary}@{t}"
            if key in tier_drops and f"{primary}@{'hard' if t == 'easy' else 'easy'}" in tier_drops:
                other = "hard" if t == "easy" else "easy"
                de = tier_drops[f"{primary}@easy"]["drop"]
                dh = tier_drops[f"{primary}@hard"]["drop"]
                if dh > de:
                    hint = (f"简单样本掉 {de}，困难样本掉 {dh}。"
                            "典型的长尾退化，不是评测数据变难了。")
                else:
                    hint = (f"简单样本掉 {de}，困难样本掉 {dh}。"
                            "两档一起掉，先怀疑改动本身，别急着怪长尾。")
                break
        if not hint:
            worst = max(drops.items(), key=lambda kv: kv[1]["drop"], default=(None, None))
            if worst[0]:
                hint = f"掉得最多的是 {worst[0]}（-{worst[1]['drop']}）。"

    return {
        "failed": failed,
        "threshold": threshold,
        "primary": primary,
        "baseline_overall": base_overall,
        "current_overall": cur_overall,
        "drop": drop,
        "drops": drops,
        "tier": tier_drops,
        "hint": hint,
        "baseline_version": baseline.get("eval_set_version", ""),
        "current_version": current.get("eval_set_version", ""),
    }


def print_decision(dec: dict) -> None:
    print("=" * 84)
    print("门禁判定")
    print("=" * 84)
    print(f"\n  主指标 {dec['primary']}：基线 {dec['baseline_overall']} "
          f"→ 本次 {dec['current_overall']}（变化 {-dec['drop']:+.3f}）")
    print(f"  阈值 {dec['threshold']}")

    if dec["baseline_version"] and dec["baseline_version"] != dec["current_version"]:
        print(f"\n  ⚠ 评测集版本变了：基线 {dec['baseline_version']} → 本次 {dec['current_version']}")
        print("    改样本等于改尺子，这次对比不成立，需要重记基线。")

    print(f"\n  {'判定器':<22}{'基线':<10}{'本次':<10}{'变化'}")
    print("  " + "-" * 56)
    for k, v in dec["drops"].items():
        mark = "  <== 掉了" if v["over"] else ""
        print(f"  {k:<22}{v['baseline']:<10}{v['current']:<10}{-v['drop']:+.3f}{mark}")

    print(f"\n  {'分档':<22}{'基线':<10}{'本次':<10}{'变化'}")
    print("  " + "-" * 56)
    for k, v in dec["tier"].items():
        print(f"  {k:<22}{v['baseline']:<10}{v['current']:<10}{-v['drop']:+.3f}")

    if dec["failed"]:
        print(f"\n  ✘ 门禁不通过：主指标掉 {dec['drop']} > 阈值 {dec['threshold']}"
              if dec["drop"] > dec["threshold"] else "\n  ✘ 门禁不通过：有判定器掉超阈值")
        if dec["hint"]:
            print(f"    定位：{dec['hint']}")
        print("  => exit 1")
    else:
        print(f"\n  ✔ 门禁通过（变化都在阈值 {dec['threshold']} 之内）")
        print("  => exit 0")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测门禁")
    ap.add_argument("--current", default=str(paths.CURRENT), help="本次评测结果 JSON")
    ap.add_argument("--baseline", default=str(paths.BASELINE), help="基线 JSON")
    ap.add_argument("--threshold", type=float, default=0.05, help="允许掉多少（默认 0.05）")
    ap.add_argument("--primary", default=None, help=f"主指标（默认 {ev.PRIMARY}）")
    ap.add_argument("--write-baseline", action="store_true", help="把当前结果记成基线")
    ap.add_argument("--json", action="store_true", help="只输出判定 JSON")
    args = ap.parse_args(argv)

    current = load_json(args.current)
    if current is None:
        print(f"✘ 找不到本次结果：{args.current}")
        return 2

    if args.write_baseline:
        Path(args.baseline).write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✔ 已记基线 {args.baseline}：主指标 {current.get('overall')}"
              f"（评测集 {current.get('eval_set_version')}）")
        return 0

    baseline = load_json(args.baseline)
    if baseline is None:
        print(f"✘ 找不到基线 {args.baseline}，无法判定。先跑一次 --write-baseline。")
        return 2

    dec = decide(baseline, current, threshold=args.threshold, primary=args.primary)
    if args.json:
        print(json.dumps(dec, ensure_ascii=False, indent=2))
    else:
        print_decision(dec)
    return 1 if dec["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

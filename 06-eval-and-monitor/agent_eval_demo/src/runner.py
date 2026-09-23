"""执行器：离线跑一遍评测。

为什么不直接用 langsmith.evaluate()：那是个好工具，但它的 data 参数有三种形态、
报三种错，配错一个就会在几百行之外崩掉（详见 dataset.to_examples 的注释）。
本执行器直接用普通 dict 跑，零依赖、零网络、结果一条不落地能读出来。

要上官方链路时用 run_official()，它构造真正的 Example 对象，并且把 target 里
LangChain 的 run 上报关掉。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import agent as agent_mod
from . import dataset as ds
from . import evaluators as ev


@dataclass
class Report:
    """一次评测的全部产出。门禁只认它。"""

    variant: str = ""
    n: int = 0
    wall: float = 0.0
    rows: list[dict] = field(default_factory=list)
    per_key: dict[str, float] = field(default_factory=dict)
    per_key_tier: dict[str, float] = field(default_factory=dict)
    primary: str = ev.PRIMARY
    eval_set_version: str = ""
    trace: dict = field(default_factory=dict)
    experiment_name: str = ""
    mode: str = "offline"

    @property
    def overall(self) -> float:
        """门禁盯的数字：主指标的整体均值。"""
        return self.per_key.get(self.primary, 0.0)

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "mode": self.mode,
            "n": self.n,
            "wall": round(self.wall, 3),
            "primary": self.primary,
            "overall": self.overall,
            "scores": self.per_key,
            "tier": self.per_key_tier,
            "eval_set_version": self.eval_set_version,
            "trace": self.trace,
            "experiment_name": self.experiment_name,
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # ------------------------------------------------------------ 打印
    def print_report(self, title: str = "评测结果") -> None:
        print("=" * 84)
        print(f"{title}  [variant={self.variant} · {self.mode} · 评测集 {self.eval_set_version}]")
        print("=" * 84)
        print(f"\n【① 执行】{self.n} 条样本 / {len(self.per_key)} 个判定器 / 耗时 {self.wall:.2f} s")

        print("\n【② 逐条明细】少了这张表，分数掉了你找不到是哪条掉的")
        print(f"  {'id':<7}{'档':<6}{'预测(意图/优先级)':<26}{'期望(意图/优先级)':<26}{'结论'}")
        print("  " + "-" * 78)
        for r in self.rows:
            got = (r["pred"].get("intent"), r["pred"].get("priority"))
            want = (r["gold"].get("intent"), r["gold"].get("priority"))
            flag = "✔" if got == want else "✘"
            print(f"  {r['id']:<7}{r['tier']:<6}{str(got):<26}{str(want):<26}{flag}")

        print("\n【③ 分档分数】门禁能不能定位问题，全看这张表")
        tiers = sorted({r["tier"] for r in self.rows})
        keys = [k for k in self.per_key]
        print(f"  {'判定器':<22}" + "".join(f"{t:<10}" for t in tiers) + "合计")
        print("  " + "-" * (22 + 10 * len(tiers) + 8))
        for k in keys:
            cells = "".join(f"{self.per_key_tier.get(f'{k}@{t}', 0):<10}" for t in tiers)
            print(f"  {k:<22}{cells}{self.per_key[k]}")
        print(f"\n  主指标（{self.primary}）整体 = {self.overall}")

        if self.trace:
            from .trace_store import print_summary

            print_summary(self.trace, "trace 采集（本次评测顺带记下来的）")


# ---------------------------------------------------------------- 目标函数
def build_target(variant: str = "good", trace_store: Any = None,
                 session: str | None = None, use_local: bool = False) -> Callable[[dict], dict]:
    """把 Agent 包成一个 (inputs) -> outputs 的目标函数。

    这是官方 evaluate() 和本包离线执行器都认的签名。
    """
    agent = agent_mod.build_agent(variant=variant, use_local=use_local)

    def target(inputs: dict) -> dict:
        pred, runs = agent_mod.ask(agent, inputs["msg"])
        if trace_store is not None:
            trace_store.append(runs, session=session, sample_id=inputs.get("id"))
        return pred

    return target


def as_inputs(item: dict) -> dict:
    return {"id": item["id"], "msg": item["msg"], "tier": item["tier"]}


# ---------------------------------------------------------------- 离线执行器
def run_offline(
    target: Callable[[dict], dict],
    items: list[dict],
    names: list[str] | None = None,
    variant: str = "",
    eval_set_version: str = "",
) -> Report:
    """逐条跑，逐条判。串行执行，保证结果可复现。"""
    names = names or ev.DEFAULT
    rows = []
    t0 = time.perf_counter()
    for item in items:
        inputs = as_inputs(item)
        started = time.perf_counter()
        pred = target(inputs) or {}
        ms = round((time.perf_counter() - started) * 1000, 2)
        scores = ev.run_all(pred, item, names)
        rows.append({
            "id": item["id"],
            "tier": item["tier"],
            "msg": item["msg"],
            "pred": pred,
            "gold": item,
            "ms": ms,
            "scores": {s.key: s.score for s in scores},
            "comments": {s.key: s.comment for s in scores if s.comment},
        })
    wall = time.perf_counter() - t0

    per_key: dict[str, list[float]] = {}
    per_key_tier: dict[str, list[float]] = {}
    for r in rows:
        for k, v in r["scores"].items():
            if v is None:
                continue  # 跳过的不进分母
            per_key.setdefault(k, []).append(v)
            per_key_tier.setdefault(f"{k}@{r['tier']}", []).append(v)

    def avg(v: list[float]) -> float:
        return round(sum(v) / len(v), 3) if v else 0.0

    return Report(
        variant=variant,
        n=len(rows),
        wall=wall,
        rows=rows,
        per_key={k: avg(v) for k, v in per_key.items()},
        per_key_tier={k: avg(v) for k, v in per_key_tier.items()},
        eval_set_version=eval_set_version,
        mode="offline",
    )


# ---------------------------------------------------------------- 官方链路
def run_official(
    target: Callable[[dict], dict],
    items: list[dict],
    names: list[str] | None = None,
    experiment_prefix: str = "agent-eval",
    max_concurrency: int = 1,
) -> tuple[Any, Report]:
    """走官方 evaluate(upload_results=False)。

    两件事必须做对：
      1. data 传真正的 Example 对象（不然报三种不同的错）
      2. target 里关掉 LangChain 的 run 上报（不然日志刷 401）

    返回 (原始结果对象, Report)。原始对象上 experiment_id / url 在离线模式取不到，
    会抛 ValueError: Experiment not started yet。
    """
    from langsmith import evaluate

    names = names or ev.DEFAULT
    examples = ds.to_examples(items)
    ls_evaluators = [ev.to_langsmith(ev.REGISTRY[n]) for n in names]

    t0 = time.perf_counter()
    res = evaluate(
        target,
        data=examples,
        evaluators=ls_evaluators,
        experiment_prefix=experiment_prefix,
        upload_results=False,
        max_concurrency=max_concurrency,
    )
    rows_raw = list(res)
    wall = time.perf_counter() - t0

    rows = []
    for item, row in zip(items, rows_raw):
        run = row["run"]
        out = dict(run.outputs or {})
        scores = {r.key: r.score for r in row["evaluation_results"]["results"]}
        rows.append({
            "id": item["id"],
            "tier": item["tier"],
            "msg": item["msg"],
            "pred": out,
            "gold": item,
            "ms": None,
            "scores": scores,
            "comments": {},
        })

    per_key: dict[str, list[float]] = {}
    per_key_tier: dict[str, list[float]] = {}
    for r in rows:
        for k, v in r["scores"].items():
            per_key.setdefault(k, []).append(v)
            per_key_tier.setdefault(f"{k}@{r['tier']}", []).append(v)

    def avg(v: list[float]) -> float:
        return round(sum(v) / len(v), 3) if v else 0.0

    rep = Report(
        variant="official",
        n=len(rows),
        wall=wall,
        rows=rows,
        per_key={k: avg(v) for k, v in per_key.items()},
        per_key_tier={k: avg(v) for k, v in per_key_tier.items()},
        experiment_name=getattr(res, "experiment_name", "") or "",
        mode="official",
    )
    return res, rep

"""观测：把 collect_runs 抓到的 run 落成本地 JSONL。

这一层解决的问题是「不上平台，能不能把一次请求的账算清楚」。答案是能，但要
知道两个坑：

  1. run 上没有 total_tokens / prompt_tokens 这类现成属性，getattr 取不到。
     真值藏在 run.outputs["generations"][0][0]["message"]["kwargs"]["usage_metadata"]。
     message 在这里是序列化后的 dict，不是对象，用属性访问会报 AttributeError。

  2. LangGraph 会把调用链拍平。一次带工具调用的请求抓到 5 条 run，
     parent_run_id 全是 None、dotted_order 全是单段、根 run 的 child_runs 为 0。
     想重建时序只能按 start_time 排序。普通链式 Runnable 的父子关系是完整的，
     所以这是 LangGraph 执行器的行为。

落盘格式：一行一条 run，字段见 run_to_record()。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

# 平级判定用：如果所有 run 的 parent_run_id 都为空，说明这条链被拍平了。


def _usage(run: Any) -> dict:
    """从 llm run 里把 usage_metadata 挖出来。

    结构层层嵌套，任何一层缺失都返回空 dict，不抛错。观测代码最忌讳自己把
    主流程搞崩。
    """
    try:
        cell = run.outputs["generations"][0][0]
        msg = cell["message"] if isinstance(cell, dict) else cell
        kwargs = msg.get("kwargs", {}) if isinstance(msg, dict) else {}
        return kwargs.get("usage_metadata") or {}
    except Exception:
        return {}


def _ms(run: Any) -> float | None:
    if run.end_time and run.start_time:
        return round((run.end_time - run.start_time).total_seconds() * 1000, 2)
    return None


def run_to_record(run: Any, session: str | None = None, sample_id: str | None = None) -> dict:
    """把一条 run 拍成一行可落盘的记录。"""
    rec: dict[str, Any] = {
        "ts": time.time(),
        "run_id": str(run.id),
        "name": run.name,
        "run_type": run.run_type,
        "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
        "dotted_order": run.dotted_order,
        "start_time": run.start_time.isoformat() if run.start_time else None,
        "ms": _ms(run),
        "error": bool(run.error),
        "tags": list(run.tags or []),
        "sample_id": sample_id,
        "session": session,
    }
    if run.run_type == "llm":
        use = _usage(run)
        rec["input_tokens"] = use.get("input_tokens")
        rec["output_tokens"] = use.get("output_tokens")
        rec["total_tokens"] = use.get("total_tokens")
        rec["tool_call_count"] = (run.extra or {}).get("tool_call_count")
    return rec


def is_flat(records: Iterable[dict]) -> bool:
    """这批 run 是不是被拍平的（没有父子关系）。"""
    recs = list(records)
    if len(recs) <= 1:
        return False
    return all(r.get("parent_run_id") is None for r in recs)


class TraceStore:
    """一个极简的本地 trace 仓库：JSONL，只追加。"""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    # ------------------------------------------------------------ 写
    def append(self, runs: Iterable[Any], session: str | None = None,
               sample_id: str | None = None) -> int:
        recs = [run_to_record(r, session=session, sample_id=sample_id) for r in runs]
        if not recs:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(recs)

    # ------------------------------------------------------------ 读
    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # 半行写坏就跳过，别让观测数据拖垮主流程
        return out

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)

    # ------------------------------------------------------------ 汇总
    def summary(self) -> dict:
        recs = self.read_all()
        return summarize(recs)


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数。列表为空返回 0。"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def summarize(recs: list[dict]) -> dict:
    """把一批 run 记录汇总成监控指标。"""
    by_type: dict[str, int] = {}
    for r in recs:
        by_type[r["run_type"]] = by_type.get(r["run_type"], 0) + 1

    llm = [r for r in recs if r["run_type"] == "llm"]
    in_tok = sum(r.get("input_tokens") or 0 for r in llm)
    out_tok = sum(r.get("output_tokens") or 0 for r in llm)
    lat = [r["ms"] for r in recs if r.get("ms") is not None]

    # 按样本算端到端耗时：一条样本的所有 run 时间跨度
    per_sample: dict[str, float] = {}
    for r in recs:
        sid = r.get("sample_id")
        if sid and r.get("ms"):
            per_sample[sid] = per_sample.get(sid, 0.0) + r["ms"]
    sample_lat = list(per_sample.values())

    return {
        "runs": len(recs),
        "by_type": by_type,
        "llm_runs": len(llm),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "avg_tokens_per_llm_call": round((in_tok + out_tok) / len(llm), 1) if llm else 0.0,
        "latency_ms_total": round(sum(lat), 2),
        "latency_ms_p50": round(percentile(lat, 0.5), 2),
        "latency_ms_p95": round(percentile(lat, 0.95), 2),
        "sample_latency_ms_p95": round(percentile(sample_lat, 0.95), 2),
        "errors": sum(1 for r in recs if r.get("error")),
        "flat": is_flat(recs),
    }


def print_summary(s: dict, title: str = "trace 汇总") -> None:
    print(f"\n  【{title}】")
    print(f"    run 总数        {s['runs']}（{', '.join(f'{k} {v}' for k, v in sorted(s['by_type'].items()))}）")
    print(f"    调用链是否拍平   {'是（parent_run_id 全空）' if s['flat'] else '否'}")
    print(f"    模型侧 token     输入 {s['input_tokens']} / 输出 {s['output_tokens']} / 合计 {s['total_tokens']}")
    print(f"    单次 llm 平均    {s['avg_tokens_per_llm_call']} token")
    print(f"    耗时             p50 {s['latency_ms_p50']} ms ｜ p95 {s['latency_ms_p95']} ms")
    print(f"    样本端到端 p95   {s['sample_latency_ms_p95']} ms")
    print(f"    错误 run         {s['errors']}")

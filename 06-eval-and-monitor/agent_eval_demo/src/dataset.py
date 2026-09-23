"""评测集：读写、版本冻结、从 trace 回捞样本。

三条纪律写进代码里，不靠自觉：

  1. 每行必须有 tier（easy / hard）。没有简单样本，分数掉了你分不清是模型
     退化还是这批数据本身变难了。
  2. 一旦用于门禁就要冻结版本。改样本等于改尺子，历史分数全部作废。
     freeze() 存指纹，check_frozen() 在开跑前校验。
  3. 评测集不是越大越好，是越稳定越好。回捞来的候选样本要人工过一遍再入集。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

REQUIRED = ("id", "tier", "msg")
TIERS = ("easy", "hard")


# ---------------------------------------------------------------- 读写
def load(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    items = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(json.loads(line))
    return items


def save(path: Path | str, items: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    items = list(items)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(items)


def validate(items: Iterable[dict]) -> list[str]:
    """返回问题清单，空列表代表没问题。"""
    problems = []
    for i, it in enumerate(items):
        for key in REQUIRED:
            if key not in it:
                problems.append(f"第 {i + 1} 行缺字段 {key}")
        if it.get("tier") not in TIERS:
            problems.append(f"{it.get('id')}: tier 必须是 {TIERS} 之一，实际 {it.get('tier')!r}")
    tiers = {it.get("tier") for it in items}
    if "easy" not in tiers or "hard" not in tiers:
        problems.append(f"缺少分档：当前只有 {sorted(t for t in tiers if t)}，两档都要有")
    return problems


# ---------------------------------------------------------------- 冻结
def fingerprint(path: Path | str) -> dict:
    """算指纹。版本号用内容哈希，不靠人手工维护。"""
    p = Path(path)
    raw = p.read_bytes()
    items = load(p)
    tiers: dict[str, int] = {}
    for it in items:
        tiers[it.get("tier", "?")] = tiers.get(it.get("tier", "?"), 0) + 1
    return {
        "version": "v" + hashlib.sha256(raw).hexdigest()[:8],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "samples": len(items),
        "tiers": tiers,
    }


def freeze(path: Path | str, frozen_path: Path | str) -> dict:
    """冻结评测集版本，把指纹写到旁边一个文件里。"""
    fp = fingerprint(path)
    Path(frozen_path).write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp


def check_frozen(path: Path | str, frozen_path: Path | str) -> tuple[bool, str]:
    """开跑前校验：评测集有没有在冻结之后被人改过。"""
    fp = Path(frozen_path)
    if not fp.exists():
        return True, "未冻结过（首次运行）"
    frozen = json.loads(fp.read_text(encoding="utf-8"))
    now = fingerprint(path)
    if frozen["sha256"] != now["sha256"]:
        return False, (
            f"评测集已变更：冻结时 {frozen['version']}（{frozen['samples']} 条）"
            f"→ 当前 {now['version']}（{now['samples']} 条）。"
            "改样本等于改尺子，历史分数作废，需要人工评审后重新冻结。"
        )
    return True, f"与冻结版本一致（{now['version']}，{now['samples']} 条）"


# ---------------------------------------------------------------- 回捞
def from_traces(trace_path: Path | str, limit: int = 20) -> list[dict]:
    """从线上 trace 里捞候选样本。

    四类信号：出错、重试、耗时异常、用户不满（这里用 tags 模拟）。
    捞出来的是候选，不是评测集 —— 必须人工标注后才能入集。
    """
    p = Path(trace_path)
    if not p.exists():
        return []
    recs = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    latency = [r["ms"] for r in recs if r.get("ms")]
    p95 = sorted(latency)[int(len(latency) * 0.95)] if len(latency) > 1 else None

    picked: dict[str, dict] = {}
    for r in recs:
        sid = r.get("sample_id") or r["run_id"]
        reason = None
        if r.get("error"):
            reason = "error"
        elif "retry" in (r.get("tags") or []):
            reason = "retry"
        elif p95 and (r.get("ms") or 0) > p95:
            reason = "slow"
        elif "user_dissatisfied" in (r.get("tags") or []):
            reason = "user_dissatisfied"
        if reason and sid not in picked:
            picked[sid] = {
                "id": f"MINED-{sid[:8]}",
                "tier": "hard",  # 回捞来的基本都是长尾，默认进 hard 档
                "msg": "<待人工填写：这条 trace 的用户原话>",
                "reason": reason,
                "source_run": r["run_id"],
                "needs_label": True,
            }
    return list(picked.values())[:limit]


# ---------------------------------------------------------------- 给官方 evaluate 用
def to_examples(items: Iterable[dict]) -> list:
    """转成 langsmith 的 Example 对象。

    这一步是必须的。官方 evaluate() 有三种 data 形态，报三种错：
      传 list[dict] 配默认上传  → 'dict' object has no attribute 'dataset_id'
      传 list[dict] 配 upload_results=False → 表面跑通，一迭代就崩在 modified_at
      传 list[Example]         → 完全跑通
    所以要么走本包的离线执行器（runner.run_offline），要么老老实实构造 Example。
    """
    import uuid

    from langsmith.schemas import Example

    ds = uuid.uuid4()
    out = []
    for it in items:
        inputs = {"msg": it["msg"], "tier": it["tier"]}
        outputs = {k: it[k] for k in ("intent", "priority", "order_id", "must_include") if k in it}
        out.append(
            Example(
                id=uuid.uuid4(),
                dataset_id=ds,
                inputs=inputs,
                outputs=outputs,
                created_at="2026-09-17T00:00:00Z",
            )
        )
    return out

"""判定器：6 个，从纯规则到 LLM 裁判。

判定器的输入是普通 dict，不是 langsmith 的 Run 对象。这样同一套判定逻辑既能
在本包的离线执行器里跑，也能通过 to_langsmith() 挂到官方 evaluate() 上。

  1. json_valid                结构：输出能不能解析、字段齐不齐
  2. intent_exact              精确匹配：意图
  3. priority_exact            精确匹配：优先级
  4. order_id_grounded         结构化断言：输出里的订单号必须真在输入里出现过
  5. must_include              必含关键词：按样本配置必须有的话
  6. judge_rubric              LLM 裁判骨架：开放式质量，需要模型，没有就跳过

一条经验：规则能判的别找模型。前 5 个都是确定性的，成本接近 0 而且永不变卦。
把裁判留给真正没法写规则的那部分。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

CONTRACT_KEYS = ("intent", "priority", "order_id", "reply")


@dataclass
class Score:
    """score=None 表示这一条被跳过（缺模型的裁判），统计时不计入分母。"""

    key: str
    score: float | None
    comment: str = ""


Evaluator = Callable[[dict, dict], Score]


# ---------------------------------------------------------------- 1 结构
def json_valid(pred: dict, gold: dict) -> Score:
    if "_parse_error" in pred:
        return Score("json_valid", 0.0, f"输出不是合法 JSON：{pred['_parse_error'][:40]}")
    missing = [k for k in CONTRACT_KEYS if k not in pred]
    if missing:
        return Score("json_valid", 0.0, f"缺字段 {missing}")
    return Score("json_valid", 1.0, "")


# ---------------------------------------------------------------- 2 意图
def intent_exact(pred: dict, gold: dict) -> Score:
    got, want = pred.get("intent"), gold.get("intent")
    return Score("intent_exact", float(got == want),
                 "" if got == want else f"预测 {got} / 期望 {want}")


# ---------------------------------------------------------------- 3 优先级
def priority_exact(pred: dict, gold: dict) -> Score:
    got, want = pred.get("priority"), gold.get("priority")
    return Score("priority_exact", float(got == want),
                 "" if got == want else f"预测 {got} / 期望 {want}")


# ---------------------------------------------------------------- 4 结构化断言
def order_id_grounded(pred: dict, gold: dict) -> Score:
    """输出里出现的订单号必须真的在用户输入里出现过。

    这是防幻觉最便宜的一招：不判断「答得好不好」，只判断「有没有编」。
    """
    got = pred.get("order_id")
    src = str(gold.get("msg") or "")
    if got in (None, "", "null"):
        return Score("order_id_grounded", 1.0, "")
    ok = str(got) in src
    return Score("order_id_grounded", float(ok),
                 "" if ok else f"编造订单号 {got}，输入里没有")


# ---------------------------------------------------------------- 5 必含关键词
def must_include(pred: dict, gold: dict) -> Score:
    """按样本配置的必备片段，逐条查是否出现在原始输出里。"""
    need = gold.get("must_include") or []
    if not need:
        return Score("must_include", None, "该样本未配置必备片段")
    raw = str(pred.get("_raw") or "")
    miss = [w for w in need if w not in raw]
    return Score("must_include", float(not miss),
                 "" if not miss else f"缺少 {'、'.join(miss)}")


# ---------------------------------------------------------------- 6 LLM 裁判
def judge_rubric(pred: dict, gold: dict) -> Score:
    """骨架：占位实现，真实项目里换成对 judge 模型的调用。

    之所以默认返回 None 而不是 0，是因为「没跑」和「跑砸了」是两回事。
    把没跑的算成 0 分，你的平均分会被无端拉低。
    """
    return Score("judge_rubric", None, "跳过：未接入裁判模型（见 src/evaluators.py 的 judge_* 函数）")


REGISTRY: dict[str, Evaluator] = {
    "json_valid": json_valid,
    "intent_exact": intent_exact,
    "priority_exact": priority_exact,
    "order_id_grounded": order_id_grounded,
    "must_include": must_include,
    "judge_rubric": judge_rubric,
}

# 默认跑的五个确定性判定器。judge 要另外接模型，不进默认集。
DEFAULT = ["json_valid", "intent_exact", "priority_exact", "order_id_grounded", "must_include"]

# 门禁盯的主指标。它在结果里必须能被单独取出来。
PRIMARY = "intent_exact"


def run_all(pred: dict, gold: dict, names: list[str] | None = None) -> list[Score]:
    names = names or DEFAULT
    return [REGISTRY[n](pred, gold) for n in names]


# ---------------------------------------------------------------- 挂到官方 evaluate 上
def to_langsmith(fn: Evaluator):
    """把本模块的判定器适配成 langsmith 的 (run, example) 签名。"""

    def _ev(run: Any, example: Any):
        from langsmith.evaluation import EvaluationResult

        pred = dict(run.outputs or {})
        if "_raw" in (run.outputs or {}):
            pred["_raw"] = run.outputs["_raw"]
        gold = dict(example.outputs or {})
        gold["msg"] = (example.inputs or {}).get("msg", "")
        s = fn(pred, gold)
        return EvaluationResult(key=s.key, score=s.score if s.score is not None else 0.0,
                               comment=s.comment)

    _ev.__name__ = getattr(fn, "__name__", "evaluator")
    return _ev


# ================================================================ LLM 裁判
# 下面这段是「真跑裁判」的那部分，需要 OpenAI 兼容端点（默认 LM Studio）。
# 三个要点：成对比较比绝对打分更稳；位置要交换着测；裁判自己也要有回归。

PAIR_SYS = """你是客服回复质量评审。用户会给你同一个用户问题的两条候选回复。
请判断哪一条更好，判断依据：是否直接回答了问题、是否给了可执行的步骤、是否说明了后续。
只输出 JSON，不要任何解释，格式：{"better": "A" 或 "B"}"""

ABS_SYS = """你是客服回复质量评审。请给候选回复打分，1 到 5 分。
判断依据：是否直接回答了问题、是否给了可执行的步骤、是否说明了后续。
只输出 JSON，不要任何解释，格式：{"score": 1-5 的整数}"""


@dataclass
class JudgeCost:
    """裁判的账。按「可见回复的长度」估成本会低估一个数量级。"""

    calls: int = 0
    out_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0
    truncations: int = 0
    details: list[dict] = field(default_factory=list)

    def add(self, d: dict) -> None:
        self.calls += 1
        self.out_tokens += d.get("out_tok") or 0
        self.reasoning_tokens += d.get("reason_tok") or 0
        self.seconds += d.get("s") or 0.0
        if d.get("finish") == "length":
            self.truncations += 1
        self.details.append(d)

    def report(self) -> dict:
        return {
            "calls": self.calls,
            "out_tokens": self.out_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "reasoning_share": round(self.reasoning_tokens / self.out_tokens * 100, 1)
            if self.out_tokens else 0.0,
            "avg_seconds": round(self.seconds / self.calls, 2) if self.calls else 0.0,
            "truncations": self.truncations,
        }


def _call(model: Any, system: str, user: str) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    t0 = time.perf_counter()
    r = model.invoke([SystemMessage(system), HumanMessage(user)])
    dt = time.perf_counter() - t0
    u = r.usage_metadata or {}
    det = u.get("output_token_details") or {}
    return {
        "raw": r.content or "",
        "s": round(dt, 2),
        "out_tok": u.get("output_tokens"),
        "reason_tok": det.get("reasoning", 0),
        "finish": (r.response_metadata or {}).get("finish_reason"),
    }


def parse_better(text: str) -> str | None:
    m = re.search(r'"better"\s*:\s*"([AB])"', text or "")
    if m:
        return m.group(1)
    m = re.search(r"\b([AB])\b", text or "")
    return m.group(1) if m else None


def parse_score(text: str) -> int | None:
    m = re.search(r'"score"\s*:\s*(\d+)', text or "")
    return int(m.group(1)) if m else None


def judge_absolute(model: Any, question: str, reply: str, cost: JudgeCost | None = None) -> int | None:
    user = f"用户问题：{question}\n\n候选回复：\n{reply}"
    d = _call(model, ABS_SYS, user)
    d["score"] = parse_score(d["raw"])
    if cost:
        cost.add(d)
    return d["score"]


def judge_pairwise(model: Any, question: str, a: str, b: str,
                   cost: JudgeCost | None = None) -> str | None:
    user = f"用户问题：{question}\n\n候选 A：\n{a}\n\n候选 B：\n{b}"
    d = _call(model, PAIR_SYS, user)
    d["better"] = parse_better(d["raw"])
    if cost:
        cost.add(d)
    return d["better"]


def pairwise_swap(model: Any, question: str, a: str, b: str,
                  cost: JudgeCost | None = None) -> dict:
    """同一对答案跑两个顺序，看结论会不会跟着位置走。

    返回 picked（两次选中了谁）、stable（是否稳定）、flip（是否翻转）。
    """
    v1 = judge_pairwise(model, question, a, b, cost)
    v2 = judge_pairwise(model, question, b, a, cost)
    if v1 is None or v2 is None:
        return {"v1": v1, "v2": v2, "stable": None, "flip": None,
                "picked": [], "note": "解析失败，不计入对照"}
    picked = {a if v1 == "A" else b, b if v2 == "A" else a}
    stable = len(picked) == 1
    return {"v1": v1, "v2": v2, "stable": stable, "flip": not stable, "picked": sorted(picked)}

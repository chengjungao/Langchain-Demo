"""监控：指标聚合 + 滚动基线告警。

这一层跟评测的分工要说清楚：

  开发期你要的是「哪些样本变差了」，是逐条的。
  监控你要的是「整体有没有异常」，是聚合的。
  把逐条明细塞进监控看板，没人会看。把聚合数字拿去做回归定位，你找不到问题在哪。

告警阈值用滚动基线，不拍绝对值：拿最近 N 次的中位数当参照，本次超过它的
若干倍才报。这样业务量涨了、模型换了，阈值会跟着自己走。

长会话还要看一个容易被忽略的指标：**单轮输入 token 的斜率**。单轮看每一轮
都没超预算，累计账单却在按会话长度悄悄滑走。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from . import paths
from .trace_store import percentile, summarize

# 监控指标里最该有几个「比率」而不是「总量」：
# 总量会随业务量涨，比率不会。告警要盯比率。


def aggregate(recs: list[dict]) -> dict:
    """把一批 run 记录聚合成看板上的一行。"""
    s = summarize(recs)
    llm = [r for r in recs if r["run_type"] == "llm"]
    sessions = {r.get("session") for r in recs if r.get("session")}
    s["sessions"] = len(sessions)
    s["error_rate"] = round(s["errors"] / s["runs"], 4) if s["runs"] else 0.0
    s["tokens_per_session"] = round(s["total_tokens"] / len(sessions), 1) if sessions else 0.0
    s["tool_calls"] = sum(r.get("tool_call_count") or 0 for r in llm)
    return s


# ---------------------------------------------------------------- 会话曲线
def session_rows(recs: list[dict], session: str) -> list[dict]:
    """把一次会话的 trace 按时间理成逐轮曲线。"""
    rows = [r for r in recs if r.get("session") == session and r["run_type"] == "llm"]
    rows.sort(key=lambda r: r.get("start_time") or "")
    out = []
    cum_in = cum_out = 0
    for i, r in enumerate(rows, 1):
        ti = r.get("input_tokens") or 0
        to = r.get("output_tokens") or 0
        cum_in += ti
        cum_out += to
        out.append({"turn": i, "input_tokens": ti, "output_tokens": to,
                    "cum_in": cum_in, "cum_total": cum_in + cum_out})
    return out


def slope(values: list[float]) -> list[float]:
    """相邻差分，用来找「单轮输入一路往上」的会话。"""
    return [round(values[i + 1] - values[i], 2) for i in range(len(values) - 1)]


def alert_rules(rows: list[dict]) -> list[str]:
    """会话级告警：斜率、P95、消息条数。"""
    alerts = []
    if not rows:
        return alerts

    diffs = slope([r["input_tokens"] for r in rows])
    # 连续 3 轮持续上升且没有回落
    run = best = 0
    for d in diffs:
        run = run + 1 if d > 0 else 0
        best = max(best, run)
    if best >= 3:
        alerts.append(f"单轮输入 token 连续 {best} 轮上升且没有回落，去查上下文治理")

    if len(rows) >= 5:
        peak = max(r["input_tokens"] for r in rows)
        first = rows[0]["input_tokens"]
        if first and peak / first >= 5:
            alerts.append(f"末轮单轮输入是首轮的 {peak / first:.1f} 倍（{first} → {peak}），"
                          "按会话长度滑走的账单")

    return alerts


# ---------------------------------------------------------------- 滚动基线
def load_history(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def append_history(path: Path | str, metrics: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), **metrics}, ensure_ascii=False) + "\n")


WATCH = ("latency_ms_p95", "sample_latency_ms_p95", "avg_tokens_per_llm_call",
         "error_rate", "tokens_per_session")


def rolling_alerts(current: dict, history: list[dict], window: int = 7,
                   factor: float = 1.5) -> list[dict]:
    """拿最近 window 次的中位数当参照，超过 factor 倍就报。

    样本不足 window 次时不报 —— 宁可不报，也不要一开始天天误报。
    """
    alerts = []
    recent = history[-window:]
    if len(recent) < 3:
        return alerts
    for key in WATCH:
        vals = [h[key] for h in recent if h.get(key) is not None]
        if len(vals) < 3:
            continue
        base = statistics.median(vals)
        cur = current.get(key)
        if cur is None:
            continue
        if base > 0 and cur > base * factor:
            alerts.append({
                "metric": key,
                "rolling_median": round(base, 4),
                "current": cur,
                "ratio": round(cur / base, 2),
                "threshold": factor,
            })
    return alerts


# ---------------------------------------------------------------- 长会话模拟
SYSTEM = "你是订单助手，回答要简短。"

TURNS = [
    "帮我查订单 A1001",
    "它什么时候到",
    "能改成明天下午送吗",
    "如果改不了就退了吧",
    "退款多久到账",
    "退到原支付方式吗",
    "要扣手续费吗",
    "那我还是等收货吧",
    "麻烦帮我催一下快递",
    "催了之后大概多久有反馈",
    "如果明天还没动我就退",
    "好，先这样吧",
]


def _windowed_model(window: int):
    """造一个带窗口裁剪的模型：window=0 表示不治理（全部历史都带上）。"""
    from typing import Any, Sequence

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class WindowedModel(BaseChatModel):
        win: int = 0

        @property
        def _llm_type(self) -> str:
            return "windowed"

        def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            hist = [m for m in messages if not isinstance(m, SystemMessage)]
            if self.win and len(hist) > self.win:
                hist = hist[-self.win:]
            prompt_chars = len(SYSTEM) + sum(len(str(m.content)) for m in hist)
            content = f"已处理：{messages[-1].content[:12]}"
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=content,
                usage_metadata={"input_tokens": prompt_chars,
                                "output_tokens": len(content),
                                "total_tokens": prompt_chars + len(content)},
                response_metadata={"model_name": "windowed"},
            ))])

    return WindowedModel(win=window)


def simulate_session(window: int = 0, turns: list[str] | None = None, session: str = "sim") -> list[dict]:
    """跑一会长会话，逐轮记下模型侧的输入 token。

    这里的窗口裁剪模拟的是上下文治理（只保留最近几条）。真实项目里治理发生在
    编排层，这里为了可观察把它放进模型里。
    """
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage

    from .agent import invoke_quiet

    turns = turns or TURNS
    agent = create_agent(model=_windowed_model(window), tools=[], system_prompt=SYSTEM)
    msgs: list = []
    rows = []
    cum_in = cum_out = 0
    for i, q in enumerate(turns, 1):
        msgs.append(HumanMessage(q))
        out, runs = invoke_quiet(agent, msgs)
        msgs = out["messages"]
        ti = to = 0
        for r in runs:
            if r.run_type != "llm":
                continue
            kw = (r.outputs.get("generations", [[{}]])[0][0].get("message", {}) or {}).get("kwargs", {})
            u = kw.get("usage_metadata") or {}
            ti += u.get("input_tokens") or 0
            to += u.get("output_tokens") or 0
        cum_in += ti
        cum_out += to
        rows.append({"turn": i, "messages": len(msgs), "input_tokens": ti,
                     "output_tokens": to, "cum_in": cum_in, "cum_total": cum_in + cum_out})
    return rows


def compare_sessions(turns: list[str] | None = None, window: int = 4) -> dict:
    """两种策略各跑一遍，把差别算出来。"""
    turns = turns or TURNS
    a = simulate_session(window=0, turns=turns)
    b = simulate_session(window=window, turns=turns)
    total_a, total_b = a[-1]["cum_total"], b[-1]["cum_total"]
    n5 = min(5, len(turns))
    share5 = (b[n5 - 1]["cum_total"] / a[n5 - 1]["cum_total"])

    return {
        "turns": len(turns),
        "ungoverned": a,
        "governed": b,
        "total_ungoverned": total_a,
        "total_governed": total_b,
        "saving_pct": round((1 - total_b / total_a) * 100, 1) if total_a else 0.0,
        "saving_pct_first5": round((1 - share5) * 100, 1),
        "first_turn_input": a[0]["input_tokens"],
        "last_turn_input": a[-1]["input_tokens"],
        "growth_x": round(a[-1]["input_tokens"] / a[0]["input_tokens"], 1) if a[0]["input_tokens"] else 0.0,
        "governed_input_range": (min(r["input_tokens"] for r in b), max(r["input_tokens"] for r in b)),
    }


def print_sessions(cmp: dict) -> None:
    print("=" * 88)
    print(f"长会话指标采集：{cmp['turns']} 轮，两种策略各跑一遍")
    print("=" * 88)
    for label, rows in (("不治理（全部历史都带上）", cmp["ungoverned"]),
                        (f"治理后（只保留最近几条）", cmp["governed"])):
        print(f"\n  【{label}】")
        print(f"    {'轮':<4}{'消息数':<8}{'本轮输入tok':<13}{'累计输入':<11}{'累计总'}")
        print("    " + "-" * 48)
        for r in rows:
            print(f"    {r['turn']:<4}{r['messages']:<8}{r['input_tokens']:<13}"
                  f"{r['cum_in']:<11}{r['cum_total']}")

    print("\n  【对比】")
    print(f"    不治理：累计总 token {cmp['total_ungoverned']}")
    print(f"    治理后：累计总 token {cmp['total_governed']}"
          f"（降到 {cmp['total_governed'] / cmp['total_ungoverned'] * 100:.0f}%，省 {cmp['saving_pct']}%）")
    print(f"    会话越长差距越大：前 5 轮省 {cmp['saving_pct_first5']}%，"
          f"前 {cmp['turns']} 轮省 {cmp['saving_pct']}%")
    print(f"\n  【曲线的形状】")
    print(f"    不治理：单轮输入从 {cmp['first_turn_input']} 涨到 {cmp['last_turn_input']}，"
          f"第 {cmp['turns']} 轮是第 1 轮的 {cmp['growth_x']} 倍")
    print(f"    治理后：单轮输入在 {cmp['governed_input_range'][0]} ~ "
          f"{cmp['governed_input_range'][1]} 之间走平")
    print("\n    这就是为什么监控不能只看「本轮平均 token」：")
    print("    单轮看每一轮都没超预算，累计账单却在按会话长度悄悄滑走。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="监控指标聚合与滚动基线告警")
    ap.add_argument("--trace", default=str(paths.TRACES))
    ap.add_argument("--history", default=str(paths.REPORTS / "metrics_history.jsonl"))
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--no-save", action="store_true", help="不把本次指标写进历史")
    args = ap.parse_args(argv)

    p = Path(args.trace)
    if not p.exists():
        print(f"✘ 找不到 trace 文件 {p}，先跑一次采集。")
        return 2
    recs = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not recs:
        print(f"✘ {p} 是空的。")
        return 2

    metrics = aggregate(recs)
    print("=" * 84)
    print(f"监控指标聚合  [{p.name} · {metrics['runs']} 条 run]")
    print("=" * 84)
    print(f"\n  run 总数 {metrics['runs']}（{', '.join(f'{k} {v}' for k, v in sorted(metrics['by_type'].items()))}）")
    print(f"  调用链拍平      {'是' if metrics['flat'] else '否'}")
    print(f"  token           输入 {metrics['input_tokens']} / 输出 {metrics['output_tokens']} / 合计 {metrics['total_tokens']}")
    print(f"  单次 llm 平均   {metrics['avg_tokens_per_llm_call']} token")
    print(f"  耗时            p50 {metrics['latency_ms_p50']} ms / p95 {metrics['latency_ms_p95']} ms")
    print(f"  样本端到端 p95  {metrics['sample_latency_ms_p95']} ms")
    print(f"  错误率          {metrics['error_rate'] * 100:.2f}%（{metrics['errors']} 条）")
    print(f"  工具调用次数    {metrics['tool_calls']}")
    if metrics["sessions"]:
        print(f"  会话数          {metrics['sessions']}，单会话平均 {metrics['tokens_per_session']} token")

    # 会话曲线
    sessions = sorted({r.get("session") for r in recs if r.get("session")})
    all_alerts: list[str] = []
    for s in sessions[:5]:
        rows = session_rows(recs, s)
        if len(rows) < 3:
            continue
        a = alert_rules(rows)
        if a:
            print(f"\n  【会话 {s} 的告警】")
            for x in a:
                print(f"    ⚠ {x}")
            all_alerts += a

    # 滚动基线
    hist = load_history(args.history)
    alerts = rolling_alerts(metrics, hist, window=args.window, factor=args.factor)
    print(f"\n  【滚动基线告警】历史 {len(hist)} 次，参照最近 {min(len(hist), args.window)} 次的中位数")
    if not hist:
        print("    首次运行，还没有历史，本次只记不报。")
    elif not alerts:
        print("    ✔ 各项指标都在滚动基线之内。")
    else:
        for a in alerts:
            print(f"    ⚠ {a['metric']}：滚动中位数 {a['rolling_median']} → 本次 {a['current']}"
                  f"（{a['ratio']} 倍，阈值 {a['threshold']}）")

    if not args.no_save:
        append_history(args.history, {k: metrics[k] for k in WATCH})
        print(f"\n  ✔ 本次指标已写入 {args.history}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

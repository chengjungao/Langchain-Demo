"""冒烟测试：把每个模块的关键行为钉住。

这些断言不是凑数的，每一条都对应正文里的一个结论。跑通了，说明正文里引用的
那组数字在这台机器上能复现。

运行：pytest -q   （或 python -m pytest tests -q）
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from src import dataset as ds  # noqa: E402
from src import evaluators as ev  # noqa: E402
from src import monitor as mon  # noqa: E402
from src import paths  # noqa: E402
from src import redact as rd  # noqa: E402
from src.cassette import Cassette, CountedModel, cache_key  # noqa: E402
from src.gate import decide  # noqa: E402
from src.runner import build_target, run_offline  # noqa: E402
from src.trace_store import TraceStore, is_flat, summarize  # noqa: E402


# ================================================================ 评测集
def test_eval_set_valid():
    items = ds.load(paths.EVAL_SET)
    assert len(items) == 6
    assert ds.validate(items) == []
    assert {it["tier"] for it in items} == {"easy", "hard"}


def test_eval_set_needs_both_tiers(tmp_path):
    p = tmp_path / "only_easy.jsonl"
    ds.save(p, [{"id": "X1", "tier": "easy", "msg": "hi"}])
    problems = ds.validate(ds.load(p))
    assert any("缺少分档" in x for x in problems)


def test_freeze_detects_tampering(tmp_path):
    p = tmp_path / "s.jsonl"
    frozen = tmp_path / "s.frozen.json"
    ds.save(p, ds.load(paths.EVAL_SET))
    ds.freeze(p, frozen)
    ok, _ = ds.check_frozen(p, frozen)
    assert ok

    items = ds.load(p)
    items[0]["msg"] = "被改过了"
    ds.save(p, items)
    ok, msg = ds.check_frozen(p, frozen)
    assert not ok
    assert "评测集已变更" in msg


def test_to_examples_shape():
    ex = ds.to_examples(ds.load(paths.EVAL_SET))
    assert len(ex) == 6
    assert ex[0].inputs["msg"]
    assert ex[0].dataset_id is not None
    # 六条样本共用同一个 dataset_id，否则会被当成六个数据集
    assert len({str(e.dataset_id) for e in ex}) == 1


# ================================================================ 判定器
def test_json_valid():
    ok = ev.json_valid({"intent": "a", "priority": "b", "order_id": None, "reply": "c"}, {})
    assert ok.score == 1.0
    bad = ev.json_valid({"_parse_error": "boom"}, {})
    assert bad.score == 0.0
    missing = ev.json_valid({"intent": "a"}, {})
    assert missing.score == 0.0
    assert "缺字段" in missing.comment


def test_exact_match():
    assert ev.intent_exact({"intent": "refund"}, {"intent": "refund"}).score == 1.0
    assert ev.intent_exact({"intent": "logistics"}, {"intent": "refund"}).score == 0.0
    assert ev.priority_exact({"priority": "high"}, {"priority": "normal"}).score == 0.0


def test_order_id_grounded_catches_hallucination():
    gold = {"msg": "我要退货，订单 A1001 还没到"}
    assert ev.order_id_grounded({"order_id": "A1001"}, gold).score == 1.0
    assert ev.order_id_grounded({"order_id": None}, gold).score == 1.0
    s = ev.order_id_grounded({"order_id": "A9999"}, gold)
    assert s.score == 0.0
    assert "编造订单号" in s.comment


def test_must_include():
    gold = {"must_include": ["退款", "A1001"]}
    assert ev.must_include({"_raw": "已受理您的退款申请，订单号 A1001。"}, gold).score == 1.0
    s = ev.must_include({"_raw": "已为您查询物流进度。"}, gold)
    assert s.score == 0.0
    assert "退款" in s.comment
    # 没配置必备片段的样本要跳过，不能算 0 分
    assert ev.must_include({"_raw": "x"}, {"must_include": []}).score is None


def test_judge_defaults_to_skipped():
    s = ev.judge_rubric({}, {})
    assert s.score is None, "没接裁判时必须返回 None，否则平均分被无端拉低"


def test_skipped_scores_do_not_enter_average():
    items = [{"id": "T1", "tier": "easy", "msg": "x", "intent": "a",
              "priority": "b", "order_id": None, "must_include": []}]
    rep = run_offline(lambda i: {"intent": "a", "priority": "b", "order_id": None,
                                 "reply": "r", "_raw": "r"},
                      items, names=["json_valid", "must_include", "judge_rubric"])
    assert rep.per_key["json_valid"] == 1.0
    # 这两项都被跳过（一个没配必备片段，一个没接裁判），不该出现在分母里
    assert "must_include" not in rep.per_key
    assert "judge_rubric" not in rep.per_key


# ================================================================ 评测与分档
@pytest.fixture(scope="module")
def reports():
    items = ds.load(paths.EVAL_SET)
    out = {}
    for v in ("good", "bad"):
        out[v] = run_offline(build_target(variant=v), items, variant=v)
    return out


def test_good_variant_scores_perfect(reports):
    assert reports["good"].overall == 1.0
    assert reports["good"].per_key["json_valid"] == 1.0


def test_bad_variant_overall_and_tiers(reports):
    bad = reports["bad"]
    assert bad.overall == 0.667, "主指标整体应掉到 0.667"
    assert bad.per_key_tier["intent_exact@easy"] == 1.0, "简单样本一条都不该坏"
    assert bad.per_key_tier["intent_exact@hard"] == 0.333, "困难样本应崩掉三分之二"


def test_structural_check_alone_would_miss_it(reports):
    """这一条是正文的核心论点：结构没坏，单看结构判定器会放行。"""
    assert reports["bad"].per_key["json_valid"] == 1.0


def test_hallucination_is_caught(reports):
    assert reports["bad"].per_key["order_id_grounded"] < 1.0


# ================================================================ 门禁
def test_gate_passes_same_version(reports):
    good = reports["good"].to_dict()
    dec = decide(good, good, threshold=0.05)
    assert not dec["failed"]


def test_gate_blocks_regression(reports):
    dec = decide(reports["good"].to_dict(), reports["bad"].to_dict(), threshold=0.05)
    assert dec["failed"]
    assert dec["drop"] == pytest.approx(0.333, abs=0.002)
    assert "困难样本" in dec["hint"]


def test_gate_lets_it_slip_with_loose_threshold(reports):
    dec = decide(reports["good"].to_dict(), reports["bad"].to_dict(), threshold=0.6)
    assert not dec["failed"], "阈值 0.6 时这个退化会被放行 —— 阈值本身就是一次决策"


def test_gate_cli_exit_codes(tmp_path, reports):
    base = tmp_path / "b.json"
    cur = tmp_path / "c.json"
    base.write_text(json.dumps(reports["good"].to_dict(), ensure_ascii=False), encoding="utf-8")
    cur.write_text(json.dumps(reports["bad"].to_dict(), ensure_ascii=False), encoding="utf-8")
    p = subprocess.run(
        [sys.executable, "-m", "src.gate", "--current", str(cur), "--baseline", str(base)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert p.returncode == 1, "门禁必须返回非 0，否则在 CI 里永远是绿的"

    cur.write_text(json.dumps(reports["good"].to_dict(), ensure_ascii=False), encoding="utf-8")
    p = subprocess.run(
        [sys.executable, "-m", "src.gate", "--current", str(cur), "--baseline", str(base)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert p.returncode == 0


def test_gate_missing_baseline_exits_2(tmp_path, reports):
    cur = tmp_path / "c.json"
    cur.write_text(json.dumps(reports["good"].to_dict(), ensure_ascii=False), encoding="utf-8")
    p = subprocess.run(
        [sys.executable, "-m", "src.gate", "--current", str(cur),
         "--baseline", str(tmp_path / "nope.json")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert p.returncode == 2


# ================================================================ 回放缓存
def test_cache_key_covers_prompt_and_model():
    a = cache_key("m1", "sys v1", "q")
    assert a != cache_key("m1", "sys v2", "q"), "改了 system prompt 必须换 key"
    assert a != cache_key("m2", "sys v1", "q"), "换了模型必须换 key"
    assert a != cache_key("m1", "sys v1", "q2")
    assert a == cache_key("m1", "sys v1", "q")


def test_cassette_rounds(tmp_path):
    cache = Cassette(tmp_path / "c")
    qs = [f"q{i}" for i in range(6)]
    extra = ["x1", "x2"]
    sys1, sys2 = "sys v1", "sys v2"

    def run(questions, prompt):
        m = CountedModel("m", prompt, latency=0)
        for q in questions:
            m.invoke(q, cache)
        return m.real_calls

    assert run(qs, sys1) == 6, "冷启动全部未命中"
    assert run(qs, sys1) == 0, "热重跑零真实调用"
    assert run(qs + extra, sys1) == 2, "扩样本只付新增的那两条"
    assert run(qs, sys2) == 6, "改了 prompt 全部未命中"


def test_cassette_survives_corruption(tmp_path):
    cache = Cassette(tmp_path / "c")
    key = cache_key("m", "s", "q")
    (tmp_path / "c" / f"{key}.json").write_text("{坏掉的", encoding="utf-8")
    assert cache.get(key) is None, "缓存坏了要回退，不能抛"
    answer, status = cache.call("m", "s", "q", lambda: "新答案")
    assert status == "miss" and answer == "新答案"


# ================================================================ trace 采集
def test_trace_store_and_flatness(tmp_path):
    from langchain_core.messages import HumanMessage

    from src.agent import build_tool_calling_agent, invoke_quiet

    agent = build_tool_calling_agent()
    _, runs = invoke_quiet(agent, [HumanMessage("查订单 A1001")])
    store = TraceStore(tmp_path / "t.jsonl")
    assert store.append(runs, session="s1", sample_id="C1") == len(runs)

    recs = store.read_all()
    assert len(recs) == 5, "一次带工具调用的请求应抓到 5 条 run"
    types = {}
    for r in recs:
        types[r["run_type"]] = types.get(r["run_type"], 0) + 1
    assert types == {"llm": 2, "tool": 2, "chain": 1}
    assert is_flat(recs), "LangGraph 会把调用链拍平"

    s = summarize(recs)
    assert s["total_tokens"] == 450 + 552
    assert s["flat"] is True


def test_run_has_no_token_attribute():
    """正文点名的坑：run 上没有现成的 token 属性，真值在 outputs 里。"""
    from langchain_core.messages import HumanMessage
    from src.agent import build_tool_calling_agent, invoke_quiet

    _, runs = invoke_quiet(build_tool_calling_agent(), [HumanMessage("查订单")])
    llm = [r for r in runs if r.run_type == "llm"][0]
    assert not hasattr(llm, "total_tokens")
    assert llm.outputs["generations"][0][0]["message"]["kwargs"]["usage_metadata"]


# ================================================================ 监控
def test_session_curve_reproducible():
    cmp = mon.compare_sessions()
    assert cmp["turns"] == 12
    assert cmp["first_turn_input"] == 24
    assert cmp["last_turn_input"] == 249
    assert cmp["growth_x"] == 10.4
    assert cmp["total_ungoverned"] == 1756
    assert cmp["total_governed"] == 755
    assert cmp["saving_pct"] == 57.0
    assert cmp["saving_pct_first5"] == 24.4


def test_session_alerts_fire():
    rows = mon.compare_sessions()["ungoverned"]
    alerts = mon.alert_rules(rows)
    assert any("连续" in a for a in alerts)
    assert any("倍" in a for a in alerts)


def test_governed_session_stays_flat():
    cmp = mon.compare_sessions()
    lo, hi = cmp["governed_input_range"]
    assert hi - lo < 100, "治理后的单轮输入应该走平"


def test_rolling_alerts_need_history():
    cur = {"latency_ms_p95": 999.0}
    assert mon.rolling_alerts(cur, [], window=7) == [], "没有历史时不报，宁可不报也别误报"
    hist = [{"latency_ms_p95": 100.0} for _ in range(5)]
    alerts = mon.rolling_alerts(cur, hist, window=7, factor=1.5)
    assert alerts and alerts[0]["metric"] == "latency_ms_p95"


# ================================================================ 脱敏
def test_default_rules_miss_cn_cloud():
    secret = dict(rd.probe_table("secret"))
    assert secret["AWS AccessKey"]
    assert not secret["阿里云 AccessKey"], "官方默认规则不认阿里云 LTAI 前缀"
    assert not secret["腾讯云 SecretId"]
    assert not secret["手机号"]


def test_business_level_covers_cn_cloud_and_business():
    biz = dict(rd.probe_table("business"))
    for name in ("阿里云 AccessKey", "腾讯云 SecretId", "手机号", "身份证号", "邮箱"):
        assert biz[name], f"{name} 应该被 business 档盖住"


def test_redact_does_not_touch_source_object():
    src = {"q": "用户 13812345678 的订单", "meta": {"tel": "13900000000"}}
    before = copy.deepcopy(src)
    out = rd.redact(src, level="business")
    assert src == before, "默认必须深拷贝，绝不能原地改业务对象"
    assert "13812345678" not in json.dumps(out, ensure_ascii=False)


def test_redactor_leaks_detection():
    r = rd.Redactor("business")
    nested = {"deep": {"deeper": {"x": "13812345678"}}}
    assert "手机号" in rd.Redactor("secret").leaks(nested, nested)
    assert "手机号" not in r.leaks(nested, r.apply(nested))


def test_strict_level_masks_order_id():
    text = json.dumps(rd.redact({"o": "订单 A1001"}, level="strict"), ensure_ascii=False)
    assert "A1001" not in text
    text2 = json.dumps(rd.redact({"o": "订单 A1001"}, level="business"), ensure_ascii=False)
    assert "A1001" in text2, "订单号默认不脱敏，放开会误伤"

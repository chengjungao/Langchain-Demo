# -*- coding: utf-8 -*-
"""冒烟测试：把 demo 里那几条关键结论钉成断言。

跑法：python tests/test_smoke.py
不需要 API key。
"""
import io
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage  # noqa: E402

from src.agents import make_all_experts, make_order_expert  # noqa: E402
from src.cost import fixed_overhead, narrow_task, wide_task  # noqa: E402
from src.handoff import build_handoff_graph  # noqa: E402
from src.isolation import (  # noqa: E402
    SEEN_KEYS,
    TASK,
    build_history,
    build_naive,
    build_schema_isolation,
    build_send_payload,
    first_call_input,
)
from src.loop_guard import run_guarded, run_unbounded  # noqa: E402
from src.parallel import LANDED, build_parallel_graph, try_wrong_way  # noqa: E402
from src.spy_model import SpyChatModel  # noqa: E402


class TestIsolation(unittest.TestCase):
    """第三章的结论：默认不隔离，Send 才隔离，schema 只挡异名键。"""

    def setUp(self):
        SpyChatModel.reset()
        self.history = build_history()

    def test_default_embeds_full_history(self):
        """默认姿势：子 Agent 拿到父图全部历史。"""
        model = SpyChatModel(tag="order", script=[{"text": "答"}])
        app = build_naive(make_order_expert(model=model))
        app.invoke({"messages": self.history + [HumanMessage(content=TASK)]})
        n, _, _ = first_call_input(model)
        self.assertGreaterEqual(n, len(self.history))

    def test_send_payload_blocks_history(self):
        """Send 姿势：任务之外的父图历史一概进不去。"""
        model = SpyChatModel(tag="worker", script=[{"text": "答"}])
        app = build_send_payload(model, make_order_expert)
        app.invoke({"messages": self.history + [HumanMessage(content=TASK)]})
        n, _, _ = first_call_input(model)
        self.assertLess(n, len(self.history))

    def test_send_saves_tokens(self):
        """Send 姿势应该明显更省。"""
        m1 = SpyChatModel(tag="a", script=[{"text": "答"}])
        build_naive(make_order_expert(model=m1)).invoke(
            {"messages": self.history + [HumanMessage(content=TASK)]}
        )
        _, _, tok_default = first_call_input(m1)

        m2 = SpyChatModel(tag="b", script=[{"text": "答"}])
        build_send_payload(m2, make_order_expert).invoke(
            {"messages": self.history + [HumanMessage(content=TASK)]}
        )
        _, _, tok_isolated = first_call_input(m2)
        self.assertLess(tok_isolated, tok_default)

    def test_schema_blocks_only_differently_named_keys(self):
        """独立 schema：同名键穿进去，异名键被挡住。"""
        SEEN_KEYS.clear()
        app = build_schema_isolation()
        app.invoke({"messages": self.history + [HumanMessage(content=TASK)], "query": TASK, "note": ""})
        seen = SEEN_KEYS[0]
        self.assertNotIn("task", seen["keys"])
        self.assertGreaterEqual(seen["messages_n"], len(self.history))


class TestParallel(unittest.TestCase):
    """第五章：Send 的位置要求。"""

    def test_send_from_conditional_edge_works(self):
        app = build_parallel_graph(["A", "B", "C"])
        app.invoke({"messages": [HumanMessage(content="go")], "results": []})
        self.assertEqual(sorted(LANDED), ["A", "B", "C"])

    def test_send_from_plain_node_raises(self):
        err = try_wrong_way(["A", "B"])
        self.assertIn("InvalidUpdateError", err)


class TestLoopGuard(unittest.TestCase):
    """第七章：转交环要自己有兜底。"""

    def test_unbounded_hits_recursion_error(self):
        res = run_unbounded(recursion_limit=8)
        self.assertFalse(res["ok"])
        self.assertIn("GraphRecursionError", res["err"])

    def test_guarded_stops_within_budget(self):
        res = run_guarded(limit=8)
        self.assertTrue(res["ok"])
        self.assertLessEqual(res["hops"], 9)


class TestHandoff(unittest.TestCase):
    """第五章：手写 Command 交接能跑通，且主管被调用次数 = 专家数 + 1。"""

    def test_handoff_runs(self):
        SpyChatModel.reset()
        sup = SpyChatModel(tag="supervisor", script=[
            {"calls": [("order_expert", {"task": "查订单"})]},
        ])
        experts = make_all_experts({
            "order": [{"text": "订单已签收。"}],
            "kb": [{"text": "7 天。"}],
            "report": [{"text": "3.2%。"}],
        })
        app = build_handoff_graph(sup, experts)
        out = app.invoke(
            {"messages": [HumanMessage(content="查订单")], "turn": 0},
            {"recursion_limit": 30},
        )
        self.assertGreaterEqual(len(out["messages"]), 3)
        self.assertEqual(len(SpyChatModel.calls_of("supervisor")), 2)  # 转交 1 次 + 收口 1 次


class TestCost(unittest.TestCase):
    """第七章：三笔账的关系要对得上。"""

    def test_fixed_overhead_multi_costs_more(self):
        fx = fixed_overhead()
        self.assertGreater(fx["multi"]["total"], fx["single"]["total"])

    def test_narrow_task_favors_multi(self):
        nr = narrow_task()
        self.assertLess(nr["multi"]["tokens"], nr["single"]["tokens"])
        self.assertLess(nr["ratio"], 1.0)

    def test_wide_task_favors_single(self):
        wd = wide_task()
        self.assertGreater(wd["multi"]["tokens"], wd["single"]["tokens"])
        self.assertGreater(wd["ratio"], 1.0)
        self.assertEqual(wd["multi"]["calls"], 10)

    def test_wide_task_multi_calls_more(self):
        wd = wide_task()
        self.assertGreater(wd["multi"]["calls"], wd["single"]["calls"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

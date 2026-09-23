# -*- coding: utf-8 -*-
"""冒烟测试。只用标准库 unittest，不联网、不需要 API key。

运行：
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.messages.utils import count_tokens_approximately  # noqa: E402

from src.skills_middleware import (  # noqa: E402
    MAX_DESCRIPTION,
    SkillLibrary,
    parse_frontmatter,
)
from src.tool_gateway import ToolGateway, classify, make_tool, synthetic_tools  # noqa: E402
from token_meter import tokens_of  # noqa: E402


def write_skill(base: Path, dirname: str, body: str) -> Path:
    d = base / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")
    return d


class TestFrontmatter(unittest.TestCase):
    def test_parse_basic(self):
        meta, body = parse_frontmatter("---\nname: a-b\ndescription: 说 明\n---\n正文")
        self.assertEqual(meta["name"], "a-b")
        self.assertEqual(meta["description"], "说 明")
        self.assertEqual(body, "正文")

    def test_parse_list(self):
        meta, _ = parse_frontmatter("---\nname: a\nallowed-tools: [x, y]\n---\n")
        self.assertEqual(meta["allowed-tools"], ["x", "y"])

    def test_no_frontmatter(self):
        meta, body = parse_frontmatter("# 标题\n正文")
        self.assertEqual(meta, {})
        self.assertIn("正文", body)


class TestSkillLibrary(unittest.TestCase):
    def test_missing_frontmatter_is_skipped_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_skill(base, "good", "---\nname: good\ndescription: 能用\n---\nok")
            write_skill(base, "bad", "# 没有 frontmatter")
            lib = SkillLibrary([base])
            metas, warn = lib.scan()
            self.assertEqual([m.name for m in metas], ["good"])
            self.assertTrue(any("缺少" in w for w in warn))

    def test_description_is_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            long_desc = "字" * (MAX_DESCRIPTION + 300)
            write_skill(base, "long", f"---\nname: long\ndescription: {long_desc}\n---\n")
            lib = SkillLibrary([base])
            metas, warn = lib.scan()
            self.assertEqual(len(metas[0].description), MAX_DESCRIPTION)
            self.assertTrue(any("截断" in w for w in warn))

    def test_name_dir_mismatch_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_skill(base, "dir-name", "---\nname: other-name\ndescription: x\n---\n")
            lib = SkillLibrary([base])
            metas, warn = lib.scan()
            self.assertEqual(len(metas), 1)
            self.assertTrue(any("不一致" in w for w in warn))

    def test_path_traversal_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_skill(base, "demo", "---\nname: demo\ndescription: x\n---\n密文")
            (base / "secret.txt").write_text("SECRET", encoding="utf-8")
            lib = SkillLibrary([base])
            lib.scan()

            self.assertIn("密文", lib.read("demo"))
            for bad in ("../secret.txt", "../../secret.txt", "sub/../../secret.txt"):
                out = lib.read("demo", bad)
                self.assertIn("越界", out, f"没有拦住 {bad}")
                self.assertNotIn("SECRET", out)

    def test_absolute_path_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_skill(base, "demo", "---\nname: demo\ndescription: x\n---\n")
            lib = SkillLibrary([base])
            lib.scan()
            out = lib.read("demo", str(base / "secret.txt"))
            self.assertTrue("只接受" in out or "不存在" in out)

    def test_read_strips_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_skill(base, "demo", "---\nname: demo\ndescription: x\n---\n# 正文标题")
            lib = SkillLibrary([base])
            lib.scan()
            self.assertTrue(lib.read("demo").startswith("# 正文标题"))

    def test_reference_file_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d = write_skill(base, "demo", "---\nname: demo\ndescription: x\n---\n")
            (d / "references").mkdir()
            (d / "references" / "r.md").write_text("细则", encoding="utf-8")
            lib = SkillLibrary([base])
            lib.scan()
            self.assertEqual(lib.read("demo", "references/r.md"), "细则")


class TestToolGateway(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify("order_query_0001"), ("order", "read"))
        self.assertEqual(classify("order_update_0001"), ("order", "write"))
        self.assertEqual(classify("order_delete_0001"), ("order", "danger"))

    def test_danger_is_not_auto_exposed(self):
        tools = [
            make_tool("order_query_0001", "订单查询", ["a", "b", "c", "tenant_id"]),
            make_tool("order_delete_0001", "订单删除", ["a", "b", "c", "tenant_id"]),
        ]
        gw = ToolGateway(tools)
        names = [t.name for t in gw.search("订单", top_k=8)]
        self.assertIn("order_query_0001", names)
        self.assertNotIn("order_delete_0001", names)
        names2 = [t.name for t in gw.search("订单", top_k=8, allow_danger=True)]
        self.assertIn("order_delete_0001", names2)

    def test_chinese_query_hits(self):
        gw = ToolGateway(synthetic_tools(200, domains=4))
        hits = gw.search("帮我处理一笔退款", top_k=8)
        self.assertTrue(hits)
        self.assertTrue(all("refund" in t.name for t in hits))

    def test_levels_and_groups(self):
        gw = ToolGateway(synthetic_tools(80, domains=4))
        self.assertEqual(sum(gw.levels().values()), 80)
        self.assertEqual(sum(gw.groups().values()), 80)
        self.assertIn("danger", gw.levels())

    def test_middleware_keeps_foreign_tools(self):
        from langchain.agents.middleware import ModelRequest
        from src.tool_gateway import ToolGatewayMiddleware

        shelf_tool = make_tool("order_query_0001", "订单查询", ["a", "b", "c", "tenant_id"])
        foreign = make_tool("read_skill", "读技能", ["skill_name"])
        gw = ToolGateway([shelf_tool])
        mw = ToolGatewayMiddleware(gw, top_k=8)

        req = ModelRequest(
            model=None, messages=[HumanMessage(content="订单查询")],
            system_message=None, tool_choice=None,
            tools=[shelf_tool, foreign], response_format=None,
            state={}, runtime=None, model_settings=None,
        )
        captured = {}

        def handler(r):
            captured["names"] = [t.name for t in r.tools]
            return r

        mw.wrap_model_call(req, handler)
        self.assertIn("read_skill", captured["names"])


class TestTokenMeter(unittest.TestCase):
    def test_cost_grows_with_tool_count(self):
        a = tokens_of(synthetic_tools(20, domains=2))
        b = tokens_of(synthetic_tools(40, domains=2))
        self.assertGreater(b, a)

        per_tool = b / 40
        self.assertTrue(150 < per_tool < 320, f"单工具均值落在 {per_tool:.1f}，超出预期区间")

    def test_more_params_cost_more(self):
        from src.tool_gateway import make_tool

        costs = []
        for n in (3, 4, 5, 6, 7):
            params = [f"p{i}_id" for i in range(n - 1)] + ["tenant_id"]
            desc = ("订单查询。用于 dom00 域的业务操作，"
                    f"需要提供 {n} 个参数，调用前请确认 tenant_id 与调用方一致。")
            costs.append(tokens_of([make_tool(f"dom00_order_query_{n:04d}", desc, params)]))
        self.assertEqual(costs, sorted(costs), f"参数越多应当越贵，实测 {costs}")

    def test_two_paths_are_not_comparable(self):
        """把工具定义当 messages 传，走的是另一套计数口径，数字对不上。"""
        import json

        tools = synthetic_tools(20, domains=2)
        t = tools[0]
        payload = json.dumps(
            {"name": t.name, "description": t.description,
             "parameters": t.args_schema.model_json_schema()},
            ensure_ascii=False,
        )
        via_messages = count_tokens_approximately([HumanMessage(content=payload)])
        via_tools = count_tokens_approximately([HumanMessage(content="x")], tools=[t])
        self.assertNotEqual(via_messages, via_tools, "两条路径的数字不该相同")


if __name__ == "__main__":
    unittest.main(verbosity=2)

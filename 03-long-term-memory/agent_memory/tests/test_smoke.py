"""冒烟测试：把记忆系统的关键行为钉住，防止改坏。

    python -m unittest discover -s tests -v

只测行为，不测实现。每条用例对应正文里的一个结论。
"""

from __future__ import annotations

import unittest

from agent_memory import (
    InMemoryStore,
    LocalHashEmbedder,
    Memory,
    MemoryManager,
    MemoryType,
    utc_now,
)


def make_manager() -> MemoryManager:
    return MemoryManager(store=InMemoryStore(embedder=LocalHashEmbedder(128)))


class TestExtraction(unittest.TestCase):
    """抽取：什么该记、记成什么样。"""

    def setUp(self) -> None:
        self.memory = make_manager()

    def test_preference_with_alias_normalized(self):
        """别名要归一到同一个实体，否则同一件事会散成两条记忆。"""
        report = self.memory.remember("u1", "我不用 Windows，只认 Mac")
        self.assertEqual(len(report.inserted), 1)
        content = report.inserted[0].content
        self.assertIn("macOS", content)
        self.assertIn("Windows", content)
        self.assertEqual(report.inserted[0].slot, "preference.os")

    def test_multiword_alias(self):
        """带空格的别名（VS Code）也要能对上。"""
        report = self.memory.remember("u1", "我用 Windows，编辑器是 VS Code")
        slots = {m.slot for m in report.inserted}
        self.assertIn("preference.editor", slots)

    def test_smalltalk_is_ignored(self):
        report = self.memory.remember("u1", "今天天气不错")
        self.assertEqual(report.summary(), "没有值得记的内容")

    def test_style_preference_uses_fixed_slot(self):
        report = self.memory.remember("u1", "回答尽量精简，结论先行")
        style = [m for m in report.inserted if m.slot == "preference.style"]
        self.assertEqual(len(style), 1)


class TestWritePolicy(unittest.TestCase):
    """写入：合并、失效、历史留存。"""

    def setUp(self) -> None:
        self.memory = make_manager()

    def test_same_fact_merges_instead_of_duplicating(self):
        self.memory.remember("u1", "我叫程工")
        report = self.memory.remember("u1", "我叫程工")
        self.assertEqual(len(report.inserted), 0)
        self.assertEqual(len(report.merged), 1)

    def test_new_value_invalidates_old_but_keeps_it(self):
        self.memory.remember("u1", "我只认 Mac")
        report = self.memory.remember("u1", "我换到 Ubuntu 了")

        self.assertEqual(len(report.invalidated), 1)
        self.assertIsNotNone(report.invalidated[0].invalid_at)
        self.assertEqual(report.invalidated[0].superseded_by, report.inserted[0].id)

        active = self.memory.store.all_slot_memories("u1", "preference.os")
        self.assertEqual(len(active), 1)
        self.assertIn("Ubuntu", active[0].content)

        history = self.memory.history_of_slot("u1", "preference.os")
        self.assertEqual(len(history), 2)


class TestRetrieval(unittest.TestCase):
    """检索：跨会话召回与租户隔离。"""

    def setUp(self) -> None:
        self.memory = make_manager()

    def test_recall_without_any_history(self):
        """新会话没有任何消息历史，只靠记忆也能答上话。"""
        self.memory.remember("u1", "我只认 Mac，回答尽量精简")
        got = self.memory.recall("u1", "我用什么操作系统", k=3)
        self.assertTrue(any("macOS" in m.content for m in got))

    def test_tenant_isolation(self):
        self.memory.remember("alice", "我只认 Mac")
        self.memory.remember("bob", "我用 Windows")
        got = self.memory.recall("bob", "我用什么操作系统", k=5)
        self.assertFalse(any("macOS" in m.content for m in got))
        self.assertTrue(any("Windows" in m.content for m in got))

    def test_cross_tenant_read_denied(self):
        """拿着别人的记忆 id 也读不到，隔离在存储层。"""
        self.memory.remember("alice", "我只认 Mac")
        target = self.memory.store.list("alice")[0]
        self.assertIsNone(self.memory.store.get("bob", target.id))

    def test_min_score_filters_noise(self):
        self.memory.remember("alice", "我只认 Mac")
        got = self.memory.recall("alice", "完全不相干的另一件事", k=5, min_score=0.5)
        self.assertEqual(got, [])

    def test_recall_touches_access_stats(self):
        self.memory.remember("u1", "我只认 Mac")
        self.memory.recall("u1", "我用什么操作系统", k=1)
        hit = [m for m in self.memory.store.list("u1") if "macOS" in m.content][0]
        self.assertGreaterEqual(hit.access_count, 1)


class TestGovernance(unittest.TestCase):
    """治理：遗忘与整合。"""

    def test_forget_removes_stale_low_value(self):
        memory = make_manager()
        old = utc_now() - 200 * 86400
        stale = Memory(
            content="用户随口问过天气", user_id="u1", type=MemoryType.EPISODIC,
            importance=0.1, created_at=old, updated_at=old, last_access_at=old,
        )
        memory.store.add(stale)

        plan = memory.forget("u1")
        self.assertIn(stale.id, plan.delete_ids)

        memory.forget("u1", apply=True)
        self.assertIsNone(memory.store.get("u1", stale.id))

    def test_consolidate_turns_episodes_into_conclusion(self):
        memory = make_manager()
        for text in (
            "上次排查检索延迟用了火焰图",
            "上次排查检索延迟发现是缓存穿透",
            "上次排查检索延迟调了批量大小",
        ):
            memory.remember("u1", text, memory_type=MemoryType.EPISODIC)

        created = memory.consolidate("u1", threshold=0.45, min_size=3)
        self.assertEqual(len(created), 1)
        self.assertIn("检索延迟", created[0].content)
        self.assertEqual(created[0].metadata["count"], 3)

        remaining = memory.store.list("u1", memory_types=[MemoryType.EPISODIC])
        self.assertEqual(remaining, [])

    def test_consolidate_needs_enough_episodes(self):
        memory = make_manager()
        memory.remember("u1", "上次排查检索延迟用了火焰图", memory_type=MemoryType.EPISODIC)
        self.assertEqual(memory.consolidate("u1", min_size=3), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

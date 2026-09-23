"""记忆管理器：对外唯一入口。

上层（Agent、工具、中间件）只跟这个类打交道，三件事：
remember 写入、recall 召回、治理（forget / consolidate）。

所有"决策"都下沉到了 policy 层，这里只负责编排与落库，
所以替换任何一层（存储、嵌入、抽取、策略）都不用改调用方。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .embedder import Embedder
from .extractor import BaseExtractor, ExtractedMemory, RuleBasedExtractor
from .policy import (
    ForgetConfig,
    ForgetPlan,
    ScoringWeights,
    find_clusters,
    plan_forget,
    plan_write,
    rank,
    summarize,
)
from .schema import Memory, MemoryType, utc_now
from .store import MemoryStore, SQLiteMemoryStore


@dataclass
class WriteReport:
    """一次 remember 做了什么，全部可观察。"""

    inserted: list[Memory] = field(default_factory=list)
    merged: list[Memory] = field(default_factory=list)
    invalidated: list[Memory] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.inserted:
            parts.append(f"新增 {len(self.inserted)} 条")
        if self.merged:
            parts.append(f"合并 {len(self.merged)} 条")
        if self.invalidated:
            parts.append(f"旧值失效 {len(self.invalidated)} 条")
        return "，".join(parts) if parts else "没有值得记的内容"


class MemoryManager:
    """长期记忆门面。"""

    def __init__(
        self,
        store: MemoryStore | None = None,
        embedder: Embedder | None = None,
        extractor: BaseExtractor | None = None,
        weights: ScoringWeights | None = None,
        forget_config: ForgetConfig | None = None,
        namespace: tuple[str, ...] = ("memories",),
    ) -> None:
        self.store = store or SQLiteMemoryStore(":memory:")
        self.embedder = embedder or self.store.embedder
        self.extractor = extractor or RuleBasedExtractor()
        self.weights = weights or ScoringWeights()
        self.forget_config = forget_config or ForgetConfig()
        self.namespace = namespace

    # ------------------------------------------------------------ 写入

    def remember(
        self,
        user_id: str,
        text: str,
        *,
        namespace: tuple[str, ...] | None = None,
        source: str = "conversation",
        memory_type: MemoryType | None = None,
        importance: float | None = None,
    ) -> WriteReport:
        """从一段话里抽记忆并落库。

        传了 memory_type 且抽取器没抽出东西时，允许强制写入一条，
        供 Agent 的显式"记住这个"动作使用。
        """
        extracted = self.extractor.extract(text)
        if memory_type is not None and not extracted:
            extracted = [ExtractedMemory(
                content=text.strip(),
                type=memory_type,
                importance=importance if importance is not None else 0.6,
            )]

        report = WriteReport()
        for item in extracted:
            memory = Memory(
                content=item.content,
                user_id=user_id,
                type=memory_type or item.type,
                namespace=namespace or self.namespace,
                slot=item.slot,
                importance=importance if importance is not None else item.importance,
                source=source,
                metadata=dict(item.metadata),
            )
            memory.embedding = self.embedder.embed(memory.content)
            self._persist(memory, report)
        return report

    def _persist(self, memory: Memory, report: WriteReport) -> None:
        same_slot = (
            self.store.all_slot_memories(memory.user_id, memory.slot) if memory.slot else []
        )
        all_active = self.store.list(memory.user_id)
        plan = plan_write(memory, same_slot, all_active)

        if plan.action == "merge" and plan.merge_target is not None:
            target = plan.merge_target
            target.content = memory.content
            target.updated_at = utc_now()
            target.importance = max(target.importance, memory.importance)
            target.metadata.update(memory.metadata)
            target.embedding = memory.embedding
            self.store.update(target)
            report.merged.append(target)
            return

        self.store.add(memory)
        report.inserted.append(memory)
        for old in plan.supersede:
            old.invalidate(successor_id=memory.id)
            self.store.update(old)
            report.invalidated.append(old)

    # ------------------------------------------------------------ 召回

    def recall(
        self,
        user_id: str,
        query: str,
        *,
        k: int = 5,
        namespace: tuple[str, ...] | None = None,
        memory_types: Iterable[MemoryType] | None = None,
        include_invalid: bool = False,
        touch: bool = True,
        min_score: float = 0.0,
    ) -> list[Memory]:
        """召回与 query 最相关的记忆。

        touch=True 会把命中的记忆标记为"被用过"：访问次数加一、刷新最近访问时间。
        这是检索之外的副作用，但很有必要，它让衰减模型知道哪些记忆真的有用。

        min_score 用来砍掉低分噪声。带噪声的记忆比没有记忆更糟：
        模型会拿一条不相关的事实硬套当前问题。
        """
        hits = self.store.search(
            user_id,
            query_text=query,
            namespace=namespace,
            memory_types=memory_types,
            limit=max(k * 4, 20),
            include_invalid=include_invalid,
        )
        ranked = rank(hits, weights=self.weights)
        top = [
            hit for hit in ranked
            if hit.memory.score is not None and hit.memory.score >= min_score
        ][:k]

        if touch:
            for hit in top:
                if hit.memory.is_active:
                    hit.memory.touch()
                    self.store.update(hit.memory)

        return [hit.memory for hit in top]

    def build_context(
        self, user_id: str, query: str, *, k: int = 5, min_score: float = 0.0
    ) -> str:
        """把召回到的记忆拼成可直接塞进提示词的一段文本。"""
        memories = self.recall(user_id, query, k=k, min_score=min_score)
        if not memories:
            return ""
        lines = "\n".join(f"- {m.content}" for m in memories)
        return f"关于该用户已知的信息：\n{lines}"

    # ------------------------------------------------------------ 治理

    def forget(self, user_id: str, *, apply: bool = False) -> ForgetPlan:
        """算出该忘哪些。apply=False 时只出计划，不动数据。"""
        plan = plan_forget(self.store.list(user_id), self.forget_config)
        if apply and plan.delete_ids:
            self.store.delete(user_id, plan.delete_ids)
        return plan

    def consolidate(
        self,
        user_id: str,
        *,
        threshold: float = 0.45,
        min_size: int = 3,
    ) -> list[Memory]:
        """把零散的多条经历整合成一条结论。

        整合后原始经历标记失效（不再单独参与检索，但记录留着可追溯），
        新生成的结论是语义记忆，衰减更慢。
        """
        episodes = self.store.list(user_id, memory_types=[MemoryType.EPISODIC])
        created: list[Memory] = []

        for cluster in find_clusters(episodes, threshold=threshold, min_size=min_size):
            content = summarize(cluster)
            memory = Memory(
                content=content,
                user_id=user_id,
                type=MemoryType.SEMANTIC,
                namespace=self.namespace,
                slot=None,
                importance=0.8,
                source="consolidation",
                metadata={"from_ids": [m.id for m in cluster.memories], "count": cluster.size},
            )
            memory.embedding = self.embedder.embed(memory.content)
            self.store.add(memory)
            created.append(memory)

            for old in cluster.memories:
                old.invalidate(successor_id=memory.id)
                self.store.update(old)

        return created

    # ------------------------------------------------------------ 观测

    def report(self, user_id: str) -> dict:
        stats = self.store.stats(user_id)
        stats["namespaces"] = self.store.list_namespaces(user_id)
        return stats

    def history_of_slot(self, user_id: str, slot: str) -> list[Memory]:
        """看一个槽位被改写过几次（含已失效记录）。审计与排查都靠它。"""
        return self.store.all_slot_memories(user_id, slot, include_invalid=True)

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if callable(close):
            close()

"""记忆治理策略：排序、去重、失效、遗忘、整合。

这一层是记忆系统里最不容易抄的部分。检索排序公式、冲突怎么判、什么时候忘，
每个团队都会长出自己的一套。这里给的是一套能直接用的最小实现，
每一处判断都写了理由，方便按业务改。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from .embedder import cosine, tokenize
from .schema import DEFAULT_HALF_LIFE_DAYS, Memory, MemoryType, utc_now
from .store import Hit

# ---------------------------------------------------------------- 排序


@dataclass
class ScoringWeights:
    """排序权重。两路召回分相加为 1，importance 单独控制影响力度。"""

    relevance: float = 0.6
    keyword: float = 0.4
    importance_influence: float = 0.4


@dataclass
class ForgetConfig:
    """遗忘配置。三个维度可以单独开关，也可以一起用。"""

    min_importance: float = 0.25     # 低于这个重要性的低价值记忆可被淘汰
    grace_days: float = 7.0          # 新记忆的保护期，期内不因重要性被淘汰
    max_age_days: dict[MemoryType, float] = field(default_factory=lambda: {
        MemoryType.EPISODIC: 90.0,   # 经历类：三个月前的琐碎经历意义不大
        MemoryType.SEMANTIC: 720.0,
        MemoryType.PROCEDURAL: 3650.0,
    })
    capacity: int = 500              # 单用户上限，超出按综合分淘汰


def recency_factor(memory: Memory, now: float | None = None) -> float:
    """时间衰减因子。半衰期按记忆类型取，越常用越不容易掉。

    基准取"最近一次更新或访问"，所以一条被反复命中的记忆会一直保持新鲜。
    """
    now = now or utc_now()
    half_life = DEFAULT_HALF_LIFE_DAYS.get(memory.type, 180.0)
    return 0.5 ** (memory.age_days(now) / half_life)


def final_score(
    hit: Hit,
    now: float | None = None,
    weights: ScoringWeights | None = None,
) -> float:
    """一条召回结果的最终排序分。

    公式：两路相关性 × 时间衰减 × 重要性系数。
    三者的量级都是 0 到 1，乘起来天然不会有某一项独大。
    """
    weights = weights or ScoringWeights()
    base = weights.relevance * hit.relevance + weights.keyword * hit.keyword
    importance_factor = (
        1.0 - weights.importance_influence
        + weights.importance_influence * hit.memory.importance
    )
    return base * recency_factor(hit.memory, now) * importance_factor


def rank(
    hits: Sequence[Hit],
    now: float | None = None,
    weights: ScoringWeights | None = None,
) -> list[Hit]:
    """重排并把最终分回填到 memory.score，方便打印与排查。"""
    now = now or utc_now()
    weights = weights or ScoringWeights()
    scored = sorted(hits, key=lambda h: final_score(h, now, weights), reverse=True)
    for hit in scored:
        hit.memory.score = round(final_score(hit, now, weights), 4)
    return scored


# ---------------------------------------------------------------- 写入决策


@dataclass
class WritePlan:
    """一次写入的决策结果。

    action = merge 时只更新已有记忆（同一事实又说了一遍）；
    action = insert 时新增一条，并把 supersede 里的旧记忆标记失效。
    """

    action: str = "insert"
    merge_target: Memory | None = None
    supersede: list[Memory] = field(default_factory=list)


# 无槽位记忆的查重阈值：高到这个程度才算同一件事
DUPLICATE_SIMILARITY = 0.93


def _normalize(text: str) -> str:
    """去掉空白与标点，用于"说的是一模一样的话"这种判断。"""
    return "".join(ch for ch in text if ch.isalnum()).lower()


def similarity(a: Memory, b: Memory) -> float:
    if a.embedding is None or b.embedding is None:
        return 0.0
    return cosine(a.embedding, b.embedding)


def _same_fact(new: Memory, old: Memory) -> bool:
    """判断两条记忆是不是在讲同一件事。

    优先看槽位里的结构化值（最准），没有值就退化成内容比对。
    """
    new_value = new.metadata.get("value")
    old_value = old.metadata.get("value")
    if new_value and old_value:
        return new_value == old_value
    return _normalize(new.content) == _normalize(old.content)


def plan_write(
    new: Memory,
    same_slot: Sequence[Memory],
    all_active: Sequence[Memory],
) -> WritePlan:
    """决定这条新记忆是新增、合并还是替换。

    三种情况：
    1. 同槽位且值相同 -> 同一事实又说了一遍，合并（只刷新时间）
    2. 同槽位且值不同 -> 用户改主意了，旧值失效，新值入库
    3. 无槽位 -> 全库查重，只有几乎完全重复才合并
    """
    for old in same_slot:
        if _same_fact(new, old):
            return WritePlan(action="merge", merge_target=old)

    if same_slot:
        return WritePlan(action="insert", supersede=list(same_slot))

    if new.slot is None and all_active:
        best = max(all_active, key=lambda m: similarity(new, m))
        if similarity(new, best) >= DUPLICATE_SIMILARITY:
            return WritePlan(action="merge", merge_target=best)

    return WritePlan(action="insert")


# ---------------------------------------------------------------- 遗忘


@dataclass
class ForgetPlan:
    """遗忘决策：要删哪些、各自因为什么。"""

    delete_ids: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)


def plan_forget(
    memories: Sequence[Memory],
    config: ForgetConfig | None = None,
    now: float | None = None,
) -> ForgetPlan:
    """三种遗忘策略一起算，返回要物理删除的记忆。

    注意这是"删除"而不是"失效"：失效是事实被取代，记录必须留着；
    遗忘是这条记忆不再有价值，留着只是噪声，该真删。
    """
    config = config or ForgetConfig()
    now = now or utc_now()
    plan = ForgetPlan()

    survivors: list[Memory] = []
    for mem in memories:
        age = mem.age_days(now)

        if age > config.max_age_days.get(mem.type, 720.0):
            plan.delete_ids.append(mem.id)
            plan.reasons[mem.id] = f"超过时效（{age:.0f} 天）"
            continue

        if age > config.grace_days and mem.importance < config.min_importance:
            plan.delete_ids.append(mem.id)
            plan.reasons[mem.id] = f"重要性过低（{mem.importance}）"
            continue

        survivors.append(mem)

    # 容量淘汰：按"重要性 + 使用频次"排序，末尾的淘汰
    if len(survivors) > config.capacity:
        survivors.sort(
            key=lambda m: (m.importance, min(m.access_count, 10)),
            reverse=True,
        )
        for mem in survivors[config.capacity:]:
            plan.delete_ids.append(mem.id)
            plan.reasons[mem.id] = "容量上限淘汰"

    return plan


# ---------------------------------------------------------------- 整合


@dataclass
class Cluster:
    """一组讲同一件事的记忆。"""

    memories: list[Memory]

    @property
    def size(self) -> int:
        return len(self.memories)


def find_clusters(
    memories: Sequence[Memory],
    threshold: float = 0.45,
    min_size: int = 3,
) -> list[Cluster]:
    """把相似的情景记忆聚成簇，供整合使用。

    用最简单的贪心聚类：逐条与已有簇比，够像就进去，否则新开一簇。
    规模上万条时该换真聚类，但情景记忆本来就只在小窗口内整合，
    贪心在这里够用，而且结果可解释。
    """
    clusters: list[Cluster] = []
    for mem in memories:
        placed = False
        for cluster in clusters:
            if similarity(mem, cluster.memories[0]) >= threshold:
                cluster.memories.append(mem)
                placed = True
                break
        if not placed:
            clusters.append(Cluster(memories=[mem]))
    return [c for c in clusters if c.size >= min_size]


_TIME_PREFIX = re.compile(r"^(?:上次|上回|之前|前几天|昨天|那次|以前|刚刚)+")


def _common_phrase(texts: Sequence[str]) -> str:
    """逐条收敛，求出这批记录共有的那段话。"""
    if not texts:
        return ""
    best = texts[0]
    for text in texts[1:]:
        match = SequenceMatcher(None, best, text).find_longest_match(
            0, len(best), 0, len(text)
        )
        best = best[match.a: match.a + match.size]
        if len(best) < 2:
            return ""
    return best


def summarize(cluster: Cluster) -> str:
    """把一簇经历概括成一条结论。

    做法是取这批记录的最长公共片段当主题，再套一句话。
    不调模型，结果可追溯到原始记录，而公共片段天然就是它们真正在讲的那件事。
    比按词频拼凑稳得多：词频会拼出"查检检索"这种碎词。
    """
    texts = [m.content for m in cluster.memories]
    topic = _TIME_PREFIX.sub("", _common_phrase(texts)).strip("，。、：: ")
    if len(topic) < 2:
        topic = "高频问题"
    return f"用户在「{topic}」这类问题上已积累 {cluster.size} 次处理经验"

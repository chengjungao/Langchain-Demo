"""记忆的数据模型。

三个设计决定，后面所有能力都长在上面：

1. 记忆是一等公民对象，不是一段字符串。类型、重要性、来源、槽位都是字段，
   不是一个纯文本列表。
2. 双时序失效。invalid_at 记录的是"这条记忆从什么时候起不再为真"，
   而不是把它删掉。旧事实留在库里，审计和回溯都靠它。
3. superseded_by 把"被谁取代"串成链，一条偏好被改写过几次一目了然。
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """CoALA 框架对长期记忆的三分类。"""

    SEMANTIC = "semantic"      # 语义记忆：事实与偏好。用户是谁、喜欢什么
    EPISODIC = "episodic"      # 情景记忆：经历过什么。上次那类问题怎么解的
    PROCEDURAL = "procedural"  # 程序性记忆：规则与方法。该怎么答、有什么约定


# 三类记忆的默认半衰期（天）。偏好相对稳定，情景经验掉得最快。
DEFAULT_HALF_LIFE_DAYS: dict[MemoryType, float] = {
    MemoryType.SEMANTIC: 180.0,
    MemoryType.EPISODIC: 30.0,
    MemoryType.PROCEDURAL: 365.0,
}


def utc_now() -> float:
    """当前 UTC 时间戳（秒）。全项目统一用它取时间，方便测试时打桩。"""
    return datetime.now(timezone.utc).timestamp()


@dataclass
class Memory:
    """一条长期记忆。"""

    content: str
    user_id: str
    type: MemoryType = MemoryType.SEMANTIC
    namespace: tuple[str, ...] = ("memories",)

    # 槽位：同类事实的"位置"。同一个槽位出现新值时，旧记忆失效。
    # 例：preference.os 这个槽位先后装过 windows 和 macos。
    slot: str | None = None

    importance: float = 0.5
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    created_at: float = field(default_factory=utc_now)
    updated_at: float = field(default_factory=utc_now)
    last_access_at: float = field(default_factory=utc_now)
    access_count: int = 0

    # 双时序：什么时候失效、被谁取代。都为 None 表示仍然有效。
    invalid_at: float | None = None
    superseded_by: str | None = None

    source: str = "conversation"
    metadata: dict[str, Any] = field(default_factory=dict)

    embedding: list[float] | None = None
    # 检索时回填的排序分，不持久化
    score: float | None = None

    @property
    def is_active(self) -> bool:
        return self.invalid_at is None

    @property
    def ns(self) -> str:
        """namespace 元组的字符串形式，便于打印与检索。"""
        return "/".join(self.namespace)

    def touch(self, at: float | None = None) -> None:
        """记一次"被使用"：访问次数加一，刷新最近访问时间。

        最近访问时间参与时间衰减，所以常用的记忆不容易被淘汰。
        """
        self.access_count += 1
        self.last_access_at = at or utc_now()

    def invalidate(self, successor_id: str | None = None, at: float | None = None) -> None:
        """标记失效，不删除。"""
        self.invalid_at = at or utc_now()
        self.superseded_by = successor_id

    def age_days(self, now: float | None = None) -> float:
        """距最近一次更新或访问过了多少天。"""
        return ((now or utc_now()) - max(self.updated_at, self.last_access_at)) / 86400.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        data["namespace"] = list(self.namespace)
        return data

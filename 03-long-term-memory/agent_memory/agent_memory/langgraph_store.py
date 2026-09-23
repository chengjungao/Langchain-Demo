"""把记忆层接成 LangGraph 官方 BaseStore 的实现。

这一步是"接自己的向量库"的落点。LangGraph 对自定义 store 的要求很轻：
实现 batch 一个方法即可，get / put / search / list_namespaces 都有默认实现走它。
内部的存储与检索完全可以是自己的东西，这里是 SQLite 版，
换成 Milvus 只需要替换下面 _records 指向的对象。

映射关系：
    namespace -> namespace，多租户就把它设计成 ("users", user_id, ...)
    key       -> 记忆 id
    value     -> {"content": ..., "type": ..., "importance": ..., "slot": ...}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)

from .manager import MemoryManager
from .schema import Memory, MemoryType, utc_now
from .store import SQLiteMemoryStore

# value 里这几个键由记忆层自己管，不放进 metadata
_RESERVED_KEYS = frozenset({"content", "type", "importance", "slot"})


def _to_datetime(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _parse_type(raw: Any) -> MemoryType:
    try:
        return MemoryType(str(raw))
    except ValueError:
        return MemoryType.SEMANTIC


class LangGraphMemoryStore(BaseStore):
    """用 agent_memory 的存储层实现 LangGraph 的 store 接口。

    LangGraph 侧看到的是标准 KV 语义，底层仍然是我们自己的记忆模型：
    embedding 落在自己的库里，衰减与治理规则也还是自己那套。
    """

    def __init__(self, manager: MemoryManager) -> None:
        super().__init__()
        self._manager = manager
        self._records: SQLiteMemoryStore = manager.store  # type: ignore[assignment]
        self._embedder = manager.embedder

    # ------------------------------------------------------------ 租户解析

    def tenant_of(self, namespace: tuple[str, ...]) -> str:
        """从 namespace 里取出租户标识。

        默认约定 ("users", user_id, ...) 或 (user_id, ...)。
        接到自己系统时改这里，让它对上你们的账号体系，
        隔离边界就跟着账号体系走，不用迁就任何第三方库的抽象。
        """
        if not namespace:
            raise ValueError("namespace 不能为空")
        if namespace[0] == "users" and len(namespace) > 1:
            return namespace[1]
        return namespace[0]

    # ------------------------------------------------------------ BaseStore

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        return [self._run(op) for op in ops]

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        return self.batch(ops)

    def _run(self, op: Any) -> Any:
        if isinstance(op, PutOp):
            return self._put(op)
        if isinstance(op, GetOp):
            return self._get(op)
        if isinstance(op, SearchOp):
            return self._search(op)
        if isinstance(op, ListNamespacesOp):
            return self._list_namespaces(op)
        raise NotImplementedError(f"不支持的 op：{type(op).__name__}")

    # ------------------------------------------------------------ 各操作

    def _put(self, op: PutOp) -> None:
        user_id = self.tenant_of(op.namespace)

        # value 为 None 表示删除
        if op.value is None:
            self._records.delete(user_id, [op.key])
            return None

        value = dict(op.value)
        content = str(value.get("content", "")).strip()
        if not content:
            return None

        existing = self._records.get(user_id, op.key)
        if existing is not None:
            existing.content = content
            existing.type = _parse_type(value.get("type", existing.type.value))
            existing.importance = float(value.get("importance", existing.importance))
            existing.slot = value.get("slot", existing.slot)
            existing.embedding = self._embedder.embed(content)
            existing.updated_at = utc_now()
            self._records.update(existing)
            return None

        memory = Memory(
            id=op.key,
            content=content,
            user_id=user_id,
            type=_parse_type(value.get("type", "semantic")),
            namespace=tuple(op.namespace),
            slot=value.get("slot"),
            importance=float(value.get("importance", 0.5)),
            source="langgraph",
            metadata={"extra": {k: v for k, v in value.items() if k not in _RESERVED_KEYS}},
        )
        memory.embedding = self._embedder.embed(content)
        self._records.add(memory)
        return None

    def _get(self, op: GetOp) -> Item | None:
        memory = self._records.get(self.tenant_of(op.namespace), op.key)
        return self._to_item(memory) if memory else None

    def _search(self, op: SearchOp) -> list[SearchItem]:
        user_id = self.tenant_of(op.namespace_prefix)
        prefix = tuple(op.namespace_prefix) or None
        limit = op.limit or 10
        offset = op.offset or 0

        hits = self._records.search(
            user_id,
            query_text=op.query,
            namespace=prefix,
            limit=max(limit + offset, 20),
        )

        items: list[SearchItem] = []
        for hit in hits:
            if not _match_filter(hit.memory, op.filter):
                continue
            items.append(SearchItem(
                namespace=hit.memory.namespace,
                key=hit.memory.id,
                value=self._to_value(hit.memory),
                created_at=_to_datetime(hit.memory.created_at),
                updated_at=_to_datetime(hit.memory.updated_at),
                score=hit.relevance,
            ))
        return items[offset:offset + limit]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces = self._records.list_all_namespaces()
        depth = op.max_depth if op.max_depth else None
        result: list[tuple[str, ...]] = []

        for ns in namespaces:
            parts = tuple(ns.split("/"))
            if depth:
                parts = parts[:depth]
            if parts not in result:
                result.append(parts)

        if op.match_conditions:
            for condition in op.match_conditions:
                path = getattr(condition, "path", None)
                if path:
                    result = [ns for ns in result if _path_matches(ns, tuple(path))]

        offset = op.offset or 0
        limit = op.limit or len(result)
        return result[offset:offset + limit]

    # ------------------------------------------------------------ 值转换

    @staticmethod
    def _to_value(memory: Memory) -> dict[str, Any]:
        return {
            "content": memory.content,
            "type": memory.type.value,
            "importance": memory.importance,
            "slot": memory.slot,
            "active": memory.is_active,
        }

    @classmethod
    def _to_item(cls, memory: Memory) -> Item:
        return Item(
            namespace=memory.namespace,
            key=memory.id,
            value=cls._to_value(memory),
            created_at=_to_datetime(memory.created_at),
            updated_at=_to_datetime(memory.updated_at),
        )


def _match_filter(memory: Memory, flt: dict | None) -> bool:
    """支持等值与 $eq / $ne / $in 三种过滤，够日常用。"""
    if not flt:
        return True
    value = LangGraphMemoryStore._to_value(memory)
    for key, expect in flt.items():
        actual = value.get(key)
        if isinstance(expect, dict):
            for operator, operand in expect.items():
                if operator == "$eq" and actual != operand:
                    return False
                if operator == "$ne" and actual == operand:
                    return False
                if operator == "$in" and actual not in operand:
                    return False
        elif actual != expect:
            return False
    return True


def _path_matches(namespace: tuple[str, ...], path: tuple[str, ...]) -> bool:
    if len(path) > len(namespace):
        return False
    return all(p == "*" or p == n for p, n in zip(path, namespace))

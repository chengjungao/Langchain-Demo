"""存储层：记忆的落盘与召回。

MemoryStore 是契约，SQLiteMemoryStore 是默认实现：单文件、零依赖、带租户隔离。
要换成 Milvus 或 Postgres，只需再实现一个 MemoryStore，上层一行都不用改。

两种召回方式各管一段：
- 向量召回负责"意思相近但用词不同"
- 关键词召回负责"专有名词、型号、错别字"这类向量容易漏的情况
两者在 policy 层加权合成，这就是最小可用的混合检索。
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

from .embedder import Embedder, LocalHashEmbedder, cosine, tokenize
from .schema import Memory, MemoryType, utc_now

# 关键词命中统计时忽略的高频词，避免"的/了/我"把分数拉虚
_STOPWORDS = frozenset({
    "的", "了", "是", "我", "你", "他", "她", "在", "有", "和", "就", "都", "也",
    "不", "要", "被", "把", "这", "那", "个", "给", "对", "吗", "呢", "吧",
    "a", "an", "the", "is", "are", "of", "to", "and", "or", "i", "you", "it",
})


@dataclass
class Hit:
    """一条召回结果：记忆本体 + 两路原始分数。最终排序分由 policy 算。"""

    memory: Memory
    relevance: float = 0.0  # 向量余弦
    keyword: float = 0.0    # 关键词命中率


def _pack(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class MemoryStore(ABC):
    """记忆存储契约。"""

    @abstractmethod
    def add(self, memory: Memory) -> Memory: ...

    @abstractmethod
    def update(self, memory: Memory) -> Memory: ...

    @abstractmethod
    def get(self, user_id: str, memory_id: str) -> Memory | None: ...

    @abstractmethod
    def list(
        self,
        user_id: str,
        namespace: tuple[str, ...] | None = None,
        memory_types: Iterable[MemoryType] | None = None,
        include_invalid: bool = False,
    ) -> list[Memory]: ...

    @abstractmethod
    def search(
        self,
        user_id: str,
        query_text: str | None = None,
        query_vec: Sequence[float] | None = None,
        namespace: tuple[str, ...] | None = None,
        memory_types: Iterable[MemoryType] | None = None,
        limit: int = 10,
        include_invalid: bool = False,
    ) -> list[Hit]: ...

    @abstractmethod
    def delete(self, user_id: str, memory_ids: Sequence[str]) -> int: ...

    @abstractmethod
    def stats(self, user_id: str) -> dict: ...


class SQLiteMemoryStore(MemoryStore):
    """SQLite 实现。

    设计上的两个取舍：
    1. 所有查询强制带 user_id。租户隔离放在存储层，上层写错也越不了权。
    2. 召回在小数据量下走应用层算分。真实规模要把向量检索下沉到向量库，
       接口签名保持一致，所以换实现不影响调用方。
    """

    def __init__(self, path: str = ":memory:", embedder: Embedder | None = None) -> None:
        self.embedder = embedder or LocalHashEmbedder()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    # ------------------------------------------------------------ 建表

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id             TEXT PRIMARY KEY,
                    user_id        TEXT NOT NULL,
                    ns             TEXT NOT NULL,
                    type           TEXT NOT NULL,
                    slot           TEXT,
                    content        TEXT NOT NULL,
                    importance     REAL NOT NULL,
                    created_at     REAL NOT NULL,
                    updated_at     REAL NOT NULL,
                    last_access_at REAL NOT NULL,
                    access_count   INTEGER NOT NULL DEFAULT 0,
                    invalid_at     REAL,
                    superseded_by  TEXT,
                    source         TEXT,
                    meta           TEXT,
                    emb            BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_user_ns   ON memories(user_id, ns);
                CREATE INDEX IF NOT EXISTS idx_user_slot ON memories(user_id, slot);
                CREATE INDEX IF NOT EXISTS idx_state     ON memories(user_id, invalid_at);
                """
            )

    # ------------------------------------------------------------ 基础读写

    def add(self, memory: Memory) -> Memory:
        if memory.embedding is None:
            memory.embedding = self.embedder.embed(memory.content)
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO memories (id, user_id, ns, type, slot, content, importance,
                       created_at, updated_at, last_access_at, access_count,
                       invalid_at, superseded_by, source, meta, emb)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory.id, memory.user_id, memory.ns, memory.type.value, memory.slot,
                    memory.content, memory.importance,
                    memory.created_at, memory.updated_at, memory.last_access_at,
                    memory.access_count, memory.invalid_at, memory.superseded_by,
                    memory.source, json.dumps(memory.metadata, ensure_ascii=False),
                    _pack(memory.embedding),
                ),
            )
        return memory

    def update(self, memory: Memory) -> Memory:
        if memory.embedding is None:
            memory.embedding = self.embedder.embed(memory.content)
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE memories SET content=?, importance=?, updated_at=?, last_access_at=?,
                       access_count=?, invalid_at=?, superseded_by=?, slot=?, meta=?, emb=?
                   WHERE id=? AND user_id=?""",
                (
                    memory.content, memory.importance, memory.updated_at, memory.last_access_at,
                    memory.access_count, memory.invalid_at, memory.superseded_by, memory.slot,
                    json.dumps(memory.metadata, ensure_ascii=False), _pack(memory.embedding),
                    memory.id, memory.user_id,
                ),
            )
        return memory

    def get(self, user_id: str, memory_id: str) -> Memory | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id=? AND user_id=?", (memory_id, user_id)
        ).fetchone()
        return _row_to_memory(row) if row else None

    def list(
        self,
        user_id: str,
        namespace: tuple[str, ...] | None = None,
        memory_types: Iterable[MemoryType] | None = None,
        include_invalid: bool = False,
    ) -> list[Memory]:
        sql = "SELECT * FROM memories WHERE user_id=?"
        params: list = [user_id]
        sql, params = self._apply_filters(sql, params, namespace, memory_types, include_invalid)
        sql += " ORDER BY created_at ASC"
        return [_row_to_memory(r) for r in self._conn.execute(sql, params).fetchall()]

    def delete(self, user_id: str, memory_ids: Sequence[str]) -> int:
        if not memory_ids:
            return 0
        placeholders = ",".join("?" * len(memory_ids))
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"DELETE FROM memories WHERE user_id=? AND id IN ({placeholders})",
                [user_id, *memory_ids],
            )
        return cur.rowcount

    # ------------------------------------------------------------ 召回

    def search(
        self,
        user_id: str,
        query_text: str | None = None,
        query_vec: Sequence[float] | None = None,
        namespace: tuple[str, ...] | None = None,
        memory_types: Iterable[MemoryType] | None = None,
        limit: int = 10,
        include_invalid: bool = False,
    ) -> list[Hit]:
        sql = "SELECT * FROM memories WHERE user_id=?"
        params: list = [user_id]
        sql, params = self._apply_filters(sql, params, namespace, memory_types, include_invalid)
        rows = self._conn.execute(sql, params).fetchall()

        if query_vec is None and query_text is not None:
            query_vec = self.embedder.embed(query_text)

        q_tokens = {t for t in tokenize(query_text) if t not in _STOPWORDS} if query_text else set()

        hits: list[Hit] = []
        for row in rows:
            memory = _row_to_memory(row)
            relevance = 0.0
            if query_vec and memory.embedding:
                relevance = cosine(query_vec, memory.embedding)
            keyword = 0.0
            if q_tokens:
                m_tokens = set(tokenize(memory.content))
                keyword = len(q_tokens & m_tokens) / len(q_tokens)
            hits.append(Hit(memory=memory, relevance=relevance, keyword=keyword))

        # 先按两路原始分的粗排收敛候选集，精细排序交给 policy
        hits.sort(key=lambda h: (h.relevance, h.keyword), reverse=True)
        return hits[: max(limit, 1)]

    def all_slot_memories(self, user_id: str, slot: str, include_invalid: bool = False) -> list[Memory]:
        """取某个槽位下的全部记忆。冲突检测与去重都靠它。"""
        sql = "SELECT * FROM memories WHERE user_id=? AND slot=?"
        params: list = [user_id, slot]
        if not include_invalid:
            sql += " AND invalid_at IS NULL"
        sql += " ORDER BY updated_at DESC"
        return [_row_to_memory(r) for r in self._conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------ 统计

    def stats(self, user_id: str) -> dict:
        rows = self._conn.execute(
            "SELECT type, invalid_at, importance, last_access_at FROM memories WHERE user_id=?",
            (user_id,),
        ).fetchall()
        total = len(rows)
        active = [r for r in rows if r["invalid_at"] is None]
        by_type: dict[str, int] = {}
        for r in active:
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        return {
            "total": total,
            "active": len(active),
            "invalidated": total - len(active),
            "by_type": by_type,
            "avg_importance": round(
                sum(r["importance"] for r in active) / len(active), 3
            ) if active else 0.0,
        }

    def list_namespaces(self, user_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT ns FROM memories WHERE user_id=? ORDER BY ns", (user_id,)
        ).fetchall()
        return [r["ns"] for r in rows]

    def list_all_namespaces(self) -> list[str]:
        """跨租户列出全部命名空间。LangGraph 侧的 list_namespaces 需要它。"""
        rows = self._conn.execute("SELECT DISTINCT ns FROM memories ORDER BY ns").fetchall()
        return [r["ns"] for r in rows]

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------ 内部

    @staticmethod
    def _apply_filters(
        sql: str,
        params: list,
        namespace: tuple[str, ...] | None,
        memory_types: Iterable[MemoryType] | None,
        include_invalid: bool,
    ) -> tuple[str, list]:
        if namespace:
            prefix = "/".join(namespace)
            sql += " AND (ns = ? OR ns LIKE ?)"
            params += [prefix, prefix + "/%"]
        if memory_types:
            values = [t.value if isinstance(t, MemoryType) else str(t) for t in memory_types]
            sql += f" AND type IN ({','.join('?' * len(values))})"
            params += values
        if not include_invalid:
            sql += " AND invalid_at IS NULL"
        return sql, params


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        user_id=row["user_id"],
        type=MemoryType(row["type"]),
        namespace=tuple(row["ns"].split("/")) if row["ns"] else ("memories",),
        slot=row["slot"],
        importance=row["importance"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_access_at=row["last_access_at"],
        access_count=row["access_count"],
        invalid_at=row["invalid_at"],
        superseded_by=row["superseded_by"],
        source=row["source"] or "conversation",
        metadata=json.loads(row["meta"]) if row["meta"] else {},
        embedding=_unpack(row["emb"]) if row["emb"] else None,
    )


class InMemoryStore(SQLiteMemoryStore):
    """进程内存储，跑测试和 demo 用。数据随进程结束消失。"""

    def __init__(self, embedder: Embedder | None = None) -> None:
        super().__init__(path=":memory:", embedder=embedder)

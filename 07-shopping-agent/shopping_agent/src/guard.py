# -*- coding: utf-8 -*-
"""护栏：把「会出事的动作」挡在模型之外。

模型本身不保证任何事。同一句话问两次，它可能一次查订单一次去下单；
超时没设就挂住；金额没有上限就能给用户报出一个不该报的数。
所以护栏不写在提示词里 —— 提示词是建议，代码才是约束。

这里落三道：

- **幂等**：同一个动作重放，第二次直接返回第一次的结果，不重复产生副作用
- **金额**：超过阈值必须人工确认，确认走的是中断机制，不是模型自觉
- **轮数**：单个子图里的循环上限，防止模型在两个工具之间反复横跳

幂等键落在 SQLite 而不是内存里，是因为重复下单这件事**重启之后依然要拦住**。
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "runtime" / "guard.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS idempotency (
    ikey       TEXT PRIMARY KEY,
    result     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS low_speed (
    ikey       TEXT PRIMARY KEY,
    hits       INTEGER NOT NULL DEFAULT 1,
    first_at   TEXT NOT NULL,
    last_at    TEXT NOT NULL
);
"""


class GuardError(Exception):
    """护栏拒绝执行。带上给人看的理由。"""


@dataclass
class Guard:
    """一次会话的护栏上下文。阈值来自配置，所以能现场调。"""

    max_rounds: int = 6
    max_amount: float = 2000.0
    db_path: Path | str = DB_PATH
    rounds: dict = field(default_factory=dict)      # role -> 已跑轮数
    blocked: list = field(default_factory=list)     # 被拦下的动作，供界面展示

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── 轮数 ────────────────────────────────────────────
    def tick(self, role: str) -> int:
        """进一轮。超上限就抛，让图去收尾而不是继续烧。"""
        n = self.rounds.get(role, 0) + 1
        self.rounds[role] = n
        if n > self.max_rounds:
            self.blocked.append({"kind": "rounds", "role": role,
                                 "detail": f"{role} 超过 {self.max_rounds} 轮上限"})
            raise GuardError(f"轮数已达上限 {self.max_rounds}，先停下来把已知信息给你。")
        return n

    # ── 金额 ────────────────────────────────────────────
    def needs_confirm(self, amount: float) -> bool:
        return amount > self.max_amount

    # ── 幂等 ────────────────────────────────────────────
    @staticmethod
    def make_key(session_id: str, action: str, *parts) -> str:
        raw = "|".join([session_id, action] + [str(p) for p in parts])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def lookup(self, ikey: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT result FROM idempotency WHERE ikey=?", (ikey,)).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "INSERT INTO low_speed (ikey, hits, first_at, last_at) VALUES (?,1,?,?) "
                "ON CONFLICT(ikey) DO UPDATE SET hits=hits+1, last_at=excluded.last_at",
                (ikey, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            self._conn.commit()
            self._replays = getattr(self, "_replays", 0) + 1
            return row["result"]

    def remember(self, ikey: str, result: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO idempotency (ikey,result,created_at) VALUES (?,?,?)",
                (ikey, result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            self._conn.commit()

    def replay_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(SUM(hits),0) AS n FROM low_speed").fetchone()
            return int(row["n"]) if row else 0

    def recent_blocks(self) -> list[dict]:
        return self.blocked[-10:]

    def reset(self) -> None:
        self.rounds.clear()
        self.blocked.clear()

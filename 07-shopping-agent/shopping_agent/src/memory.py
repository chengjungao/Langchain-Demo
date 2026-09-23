# -*- coding: utf-8 -*-
"""记忆系统：三层存储 + 写入策略 + 召回排序 + 遗忘。

多数 demo 的记忆就是一个列表，往里塞消息，取出来全带上。这撑不过三天：
消息越滚越长，模型开始忽略中间的内容，成本一路涨，而且用户改了口径之后
系统还在用旧偏好。

所以这里拆成三层，各有各的写入时机和生命周期：

| 层 | 装什么 | 存多久 | 什么时候写 |
|---|---|---|---|
| 会话摘要 | 这个会话聊过什么 | 跟着会话 | 消息数超过阈值时压缩 |
| 用户画像 | 用户是谁、稳定偏好 | 长期 | 抽到新的偏好时按 key 覆盖 |
| 事件记忆 | 发生过哪些具体的事 | 中期，会衰减 | 每轮结尾，有信号才写 |

三条纪律贯穿这一层：

1. **不是每轮都写**。先进一个廉价的信号检测器，没信号直接跳过，
   省下的是一次模型调用。真实系统里这笔账很重要。
2. **同 key 覆盖，但要够强才翻转**。用户随口一句「这个牌子也行」不该
   推翻他之前明确说过的偏好。翻转需要置信度或证据数站得住。
3. **用过的记忆会变强，没用的会变弱**。命中一次就更新 last_hit，
   排序时会加分；长期不命中的记忆自然沉底，而不是永远占用注入预算。
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .retrieval import tokenize

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "runtime"
DB_PATH = DB_DIR / "memory.db"

# 时间衰减的半衰期。三十天之后权重减半 —— 用户上个月的口味
# 不该和上周的一样重。
HALF_LIFE_DAYS = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    user_id        TEXT NOT NULL,
    key            TEXT NOT NULL,
    value          TEXT NOT NULL,
    kind           TEXT NOT NULL DEFAULT '其他',
    polarity       TEXT NOT NULL DEFAULT 'neutral',
    quote          TEXT NOT NULL DEFAULT '',
    confidence     REAL NOT NULL DEFAULT 0.7,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    last_hit       TEXT,
    hits           INTEGER NOT NULL DEFAULT 0,
    active         INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS episode (
    ep_id      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    summary    TEXT NOT NULL,
    sku_id     TEXT NOT NULL DEFAULT '',
    order_id   TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_hit   TEXT,
    hits       INTEGER NOT NULL DEFAULT 0,
    weight     REAL NOT NULL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_ep_user ON episode(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS session_meta (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    title      TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    turns      INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now() -> datetime:
    return datetime.now()


def _iso(t: datetime | None = None) -> str:
    return (t or _now()).strftime("%Y-%m-%d %H:%M:%S")


def _age_days(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        return max(0.0, (_now() - datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")).total_seconds() / 86400)
    except ValueError:
        return 0.0


def _decay(ts: str | None) -> float:
    """按距今时间给权重打折扣。刚发生的算 1，一个半衰期前算 0.5。"""
    return 0.5 ** (_age_days(ts) / HALF_LIFE_DAYS)


@dataclass
class MemoryHit:
    """一条被召回的长期记忆，带上它是怎么被选中的。"""

    source: str          # profile / episode / session
    label: str
    text: str
    score: float
    meta: dict

    def line(self) -> str:
        return f"- [{self.label}] {self.text}"


class MemoryStore:
    """SQLite 落盘。单连接加锁，避免多线程下的写冲突。"""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── 写入：用户画像 ──────────────────────────────────
    def upsert_profile(self, user_id: str, item) -> str:
        """按「槽位」写入，返回 'new' / 'update' / 'skip'。

        这里处理的是记忆系统最容易翻车的地方：**键不稳定**。

        模型每次给同一件事起的名字都不一样 —— 这次叫「品牌黑名单」，下次叫
        `avoid_brand`，再下次叫「回避品牌」。如果不做归一化，用户的同一条偏好
        会在库里躺着五六份，召回时一起塞进提示词，既费 token 又互相矛盾。

        做法是把 kind 归到一张固定词表，再用「类别 + 方向」当槽位。
        同一个槽位里，值按重叠度合并：说得更全的就替换，说的是另一件事就追加。
        """
        value = (item.value or "").strip()
        if not value:
            return "skip"
        kind = canon_kind(item.kind, item.key, value)
        slot = f"{kind}·{_pol_label(item.polarity)}"
        now = _iso()

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM profile WHERE user_id=? AND key=?",
                (user_id, slot)).fetchone()

            if row is None:
                self._conn.execute(
                    "INSERT INTO profile (user_id,key,value,kind,polarity,quote,"
                    "confidence,evidence_count,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (user_id, slot, value, kind, item.polarity, item.quote,
                     max(0.7, item.confidence), 1, now, now))
                self._drop_conflicts(user_id, kind, item.polarity, value)
                self._conn.commit()
                return "new"

            merged, changed = _merge_values(row["value"], value)
            conf = min(0.99, row["confidence"] + (0.08 if changed else 0.04))
            self._conn.execute(
                "UPDATE profile SET value=?, confidence=?, evidence_count=evidence_count+1, "
                "quote=?, updated_at=?, active=1 WHERE user_id=? AND key=?",
                (merged, conf, item.quote or row["quote"], now, user_id, slot))
            self._conn.commit()
            return "update" if changed else "skip"

    def _drop_conflicts(self, user_id: str, kind: str, polarity: str, value: str) -> None:
        """同一个类别下，方向相反且说的还是同一件事的旧记录，停用它。

        用户从「喜欢云雀」改口成「不想要云雀」时，两条留着会互相打架。
        只处理值高度重合的那些，同类别下的其他偏好不受影响。
        """
        if polarity == "neutral":
            return
        rows = self._conn.execute(
            "SELECT key, value FROM profile WHERE user_id=? AND kind=? AND polarity!=? "
            "AND polarity!='neutral' AND active=1", (user_id, kind, polarity)).fetchall()
        for r in rows:
            for part in str(r["value"]).split("；"):
                if _overlap(part, value) >= 0.6:
                    self._conn.execute(
                        "UPDATE profile SET active=0, updated_at=? WHERE user_id=? AND key=?",
                        (_iso(), user_id, r["key"]))
                    break

    # ── 写入：事件 ──────────────────────────────────────
    def add_episode(self, user_id: str, item, session_id: str = "") -> str | None:
        """写一条事件。同一件事重复出现时只更新旧记录，不新增。"""
        summary = (item.summary or "").strip()
        if len(summary) < 6:
            return None
        with self._lock:
            dup = self._find_similar_episode(user_id, summary)
            if dup is not None:
                self._conn.execute(
                    "UPDATE episode SET hits=hits+1, weight=MIN(2.0, weight+0.2), "
                    "created_at=? WHERE ep_id=?",
                    (_iso(), dup))
                self._conn.commit()
                return dup
            ep_id = "EP" + uuid.uuid4().hex[:10]
            self._conn.execute(
                "INSERT INTO episode (ep_id,user_id,kind,summary,sku_id,order_id,"
                "session_id,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (ep_id, user_id, item.kind, summary, item.sku_id or "",
                 item.order_id or "", session_id, _iso()))
            self._conn.commit()
            return ep_id

    def _find_similar_episode(self, user_id: str, summary: str, thresh: float = 0.82) -> str | None:
        """最近三天内、内容高度重合的事件视为同一件。"""
        toks = set(tokenize(summary))
        if not toks:
            return None
        since = _iso(_now() - timedelta(days=3))
        rows = self._conn.execute(
            "SELECT ep_id, summary FROM episode WHERE user_id=? AND created_at>=? "
            "ORDER BY created_at DESC LIMIT 20", (user_id, since)).fetchall()
        for r in rows:
            other = set(tokenize(r["summary"]))
            if not other:
                continue
            jaccard = len(toks & other) / len(toks | other)
            if jaccard >= thresh:
                return r["ep_id"]
        return None

    # ── 写入：会话摘要 ──────────────────────────────────
    def upsert_session(self, session_id: str, user_id: str, *,
                       title: str | None = None, summary: str | None = None,
                       turns_delta: int = 0) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id FROM session_meta WHERE session_id=?",
                (session_id,)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO session_meta (session_id,user_id,title,summary,turns,"
                    "created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                    (session_id, user_id, title or "", summary or "", turns_delta,
                     _iso(), _iso()))
            else:
                sets, vals = ["updated_at=?", "turns=turns+?"], [_iso(), turns_delta]
                if title:
                    sets.append("title=?")
                    vals.append(title)
                if summary is not None:
                    sets.append("summary=?")
                    vals.append(summary)
                vals.append(session_id)
                self._conn.execute(
                    f"UPDATE session_meta SET {', '.join(sets)} WHERE session_id=?", vals)
            self._conn.commit()

    # ── 召回 ────────────────────────────────────────────
    def session_summary(self, session_id: str) -> str:
        """这个会话之前压出来的摘要。没有就返回空串。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT summary FROM session_meta WHERE session_id=?",
                (session_id,)).fetchone()
        return (row["summary"] if row else "") or ""

    def session_stats(self, session_id: str) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM session_meta WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else {}

    def recall(self, user_id: str, query: str, limit: int = 6,
               char_budget: int = 600) -> list[MemoryHit]:
        """召回长期记忆。

        排序不是单纯按时间，而是三件事相乘：**置信度 × 时间衰减 × 与当前话题的相关度**。
        这样既有「最近说的优先」，又有「说了很多次的很稳」，还不会把
        跟当前话题毫无关系的记忆一起塞进去。
        """
        q_toks = set(tokenize(query))
        hits: list[MemoryHit] = []

        with self._lock:
            for r in self._conn.execute(
                    "SELECT * FROM profile WHERE user_id=? AND active=1", (user_id,)):
                d = dict(r)
                rel = _relevance(q_toks, f"{d['key']} {d['value']} {d['kind']}")
                score = d["confidence"] * (0.6 + 0.4 * rel) * 1.15
                if d["hits"]:
                    score *= 1 + min(0.2, d["hits"] * 0.02)
                hits.append(MemoryHit(
                    source="profile",
                    label=_profile_label(d["kind"]),
                    text=f"{d['key']}：{d['value']}",
                    score=round(score, 3),
                    meta={"key": d["key"], "confidence": round(d["confidence"], 2),
                          "evidence": d["evidence_count"], "polarity": d["polarity"],
                          "updated_at": d["updated_at"]}))

            rows = self._conn.execute(
                "SELECT * FROM episode WHERE user_id=? ORDER BY created_at DESC LIMIT 40",
                (user_id,)).fetchall()
            for r in rows:
                d = dict(r)
                rel = _relevance(q_toks, d["summary"])
                score = 0.9 * _decay(d["created_at"]) * (0.35 + 0.65 * rel) * d["weight"]
                hits.append(MemoryHit(
                    source="episode",
                    label=_episode_label(d["kind"]),
                    text=f"{d['created_at'][:10]} {d['summary']}",
                    score=round(score, 3),
                    meta={"ep_id": d["ep_id"], "kind": d["kind"],
                          "age_days": round(_age_days(d["created_at"]), 1),
                          "hits": d["hits"]}))

            for r in self._conn.execute(
                    "SELECT * FROM session_meta WHERE user_id=? AND summary!='' "
                    "ORDER BY updated_at DESC LIMIT 5", (user_id,)):
                d = dict(r)
                rel = _relevance(q_toks, d["summary"])
                score = 0.7 * _decay(d["updated_at"]) * (0.35 + 0.65 * rel)
                hits.append(MemoryHit(
                    source="session", label="上次聊过",
                    text=d["summary"][:120], score=round(score, 3),
                    meta={"session_id": d["session_id"], "turns": d["turns"]}))

        hits.sort(key=lambda h: -h.score)
        picked, used = [], 0
        for h in hits[:limit * 2]:
            if len(picked) >= limit:
                break
            if used + len(h.text) > char_budget and picked:
                break
            picked.append(h)
            used += len(h.text)
        self._mark_hit(user_id, picked)
        return picked

    def _mark_hit(self, user_id: str, hits: list[MemoryHit]) -> None:
        """被用上的记忆记一笔。这是「越用越准」的来源。"""
        if not hits:
            return
        with self._lock:
            for h in hits:
                if h.source == "profile":
                    self._conn.execute(
                        "UPDATE profile SET hits=hits+1, last_hit=? WHERE user_id=? AND key=?",
                        (_iso(), user_id, h.meta["key"]))
                elif h.source == "episode":
                    self._conn.execute(
                        "UPDATE episode SET hits=hits+1, last_hit=? WHERE ep_id=?",
                        (_iso(), h.meta["ep_id"]))
            self._conn.commit()

    # ── 遗忘与运维 ──────────────────────────────────────
    def forget(self, user_id: str, key: str) -> bool:
        """用户说「忘掉这个」时调用。软删除，留证据。"""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE profile SET active=0, updated_at=? WHERE user_id=? AND key=?",
                (_iso(), user_id, key))
            self._conn.commit()
            return cur.rowcount > 0

    def drop_episode(self, ep_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM episode WHERE ep_id=?", (ep_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def decay_scan(self, user_id: str) -> int:
        """把长期没被用上、置信度又低的画像停用。返回停用条数。

        不删数据，只置 active=0 —— 用户哪天再提起，它会带着旧证据回来。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, confidence, last_hit, updated_at FROM profile "
                "WHERE user_id=? AND active=1", (user_id,)).fetchall()
            dead = [r["key"] for r in rows
                    if r["confidence"] < 0.45
                    and _age_days(r["last_hit"] or r["updated_at"]) > 45]
            for k in dead:
                self._conn.execute(
                    "UPDATE profile SET active=0 WHERE user_id=? AND key=?", (user_id, k))
            self._conn.commit()
            return len(dead)

    # ── 给界面看的快照 ──────────────────────────────────
    def snapshot(self, user_id: str) -> dict:
        """一次性把三层记忆摊开，供界面展示。"""
        with self._lock:
            profile = [dict(r) for r in self._conn.execute(
                "SELECT * FROM profile WHERE user_id=? ORDER BY active DESC, "
                "confidence DESC, updated_at DESC", (user_id,))]
            episodes = [dict(r) for r in self._conn.execute(
                "SELECT * FROM episode WHERE user_id=? ORDER BY created_at DESC LIMIT 50",
                (user_id,))]
            sessions = [dict(r) for r in self._conn.execute(
                "SELECT * FROM session_meta WHERE user_id=? ORDER BY updated_at DESC LIMIT 20",
                (user_id,))]
        for p in profile:
            p["age_days"] = round(_age_days(p["last_hit"] or p["updated_at"]), 1)
        for e in episodes:
            e["age_days"] = round(_age_days(e["created_at"]), 1)
            e["decay"] = round(_decay(e["created_at"]), 2)
        return {"profile": profile, "episodes": episodes, "sessions": sessions,
                "stats": {"profile_active": sum(1 for p in profile if p["active"]),
                          "profile_muted": sum(1 for p in profile if not p["active"]),
                          "episodes": len(episodes), "sessions": len(sessions)}}

    def clear(self, user_id: str, layer: str = "all") -> None:
        with self._lock:
            if layer in ("all", "profile"):
                self._conn.execute("DELETE FROM profile WHERE user_id=?", (user_id,))
            if layer in ("all", "episode"):
                self._conn.execute("DELETE FROM episode WHERE user_id=?", (user_id,))
            if layer in ("all", "session"):
                self._conn.execute("DELETE FROM session_meta WHERE user_id=?", (user_id,))
            self._conn.commit()


# ── 画像归类 ────────────────────────────────────────────────
# 模型给 kind 的写法每次都可能不一样，直接拿来当主键，同一条偏好会写进去好几遍。
# 所以写入前先归到一张固定词表上，一个槽位只留一条。

_KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("品牌",     ("brand", "品牌", "牌子", "厂商")),
    ("预算",     ("budget", "预算", "价格", "价位", "多少钱")),
    ("尺码",     ("size", "尺码", "头围", "码数", "尺寸")),
    ("身体条件", ("body", "head", "wear", "glasses", "身体", "头", "佩戴", "眼镜",
                 "敏感", "夹", "耳朵", "耳", "视力", "身高", "体重")),
    ("使用场景", ("usage", "scene", "场景", "通勤", "办公", "地铁", "出差", "在家", "出行")),
    ("禁忌",     ("avoid", "dislike", "hate", "禁忌", "回避", "不要", "别给")),
    ("习惯",     ("habit", "prefer", "习惯", "偏好", "喜欢", "一直用", "常用")),
)

_POL_LABELS = {"like": "偏好", "dislike": "回避", "neutral": "事实"}


def canon_kind(kind: str, key: str = "", value: str = "") -> str:
    """把一个自由填写的类别名归到固定词表上。"""
    blob = f"{kind} {key}".lower()
    for name, keys in _KIND_RULES:
        if any(k in blob for k in keys):
            return name
    # 类别名本身没线索时，看值里有没有特征词
    blob2 = value.lower()
    for name, keys in _KIND_RULES:
        if name in ("禁忌", "习惯"):
            continue
        if any(k in blob2 for k in keys):
            return name
    return "其他"


def _pol_label(polarity: str) -> str:
    return _POL_LABELS.get(polarity, "事实")


def _overlap(a: str, b: str) -> float:
    """两段短文本的字面重合度，用二元组 Jaccard。"""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _merge_values(old: str, new: str, max_items: int = 3) -> tuple[str, bool]:
    """把新值并进旧值。返回 (合并结果, 是否真的变了)。

    同一件事换个说法时替换掉旧的，而不是两条都留着 —— 留着的结果是
    提示词里出现两句意思相同、措辞不同的话，模型会以为这是两件事。
    """
    items = [x.strip() for x in str(old).split("；") if x.strip()]
    for i, it in enumerate(items):
        if it == new:
            return old, False
        if _overlap(it, new) >= 0.45:
            items[i] = new if len(new) >= len(it) else it
            return "；".join(items), items[i] != it
    items.append(new)
    if len(items) > max_items:
        items = items[-max_items:]
    return "；".join(items), True


# ── 召回辅助 ────────────────────────────────────────────────

def _relevance(q_toks: set[str], text: str) -> float:
    """轻量相关度：二元组重叠率。够用，且不引入第二次模型调用。"""
    if not q_toks:
        return 0.5
    t = set(tokenize(text))
    if not t:
        return 0.0
    return len(q_toks & t) / math.sqrt(len(q_toks) * len(t))


_EP_LABELS = {"咨询": "咨询过", "下单": "下过单", "退换": "退换过",
              "投诉": "投诉过", "浏览": "看过"}


def _profile_label(kind: str) -> str:
    """类别名在写入时已经归一化过，这里直接用。"""
    return kind or "其他"


def _episode_label(kind: str) -> str:
    return _EP_LABELS.get(kind, kind or "事件")


# ── 写入信号检测 ────────────────────────────────────────────

_PROFILE_SIGNALS = ("我喜欢", "我偏好", "我不要", "我不喜欢", "我讨厌", "别给我",
                    "我预算", "预算", "我住", "我在", "我对", "过敏", "敏感",
                    "习惯", "一直用", "以后都用", "下次别")
_EPISODE_SIGNALS = ("买了", "下单", "退", "换货", "投诉", "发货", "物流", "订单",
                    "发票", "保修", "到货", "签收", "不满意", "坏了")


def looks_memorable(user_text: str, assistant_text: str = "") -> dict:
    """先花零成本判断这轮值不值得抽记忆。

    每轮都抽一次记忆，等于每轮多一次模型调用 —— 在真实账单里这是笔不小的开销。
    大多数日常寒暄里没有可记的东西，用一个字符串匹配先筛掉，能省掉大部分调用。
    这是「先把便宜的检查做在前面」的一个具体例子。
    """
    text = (user_text or "")
    p = [s for s in _PROFILE_SIGNALS if s in text]
    e = [s for s in _EPISODE_SIGNALS if s in text]
    return {"worth": bool(p or e), "profile_signals": p, "episode_signals": e}


def format_memory_block(hits: list[MemoryHit]) -> str:
    """把召回结果拼成一段能放进系统提示词的文字。"""
    if not hits:
        return ""
    lines = [h.line() for h in hits]
    return ("关于这位用户，你之前记下了这些（可能过时，与当前说法冲突时以用户当下为主）：\n"
            + "\n".join(lines))


_store: MemoryStore | None = None
_store_lock = threading.Lock()


def get_memory(path: Path | str | None = None) -> MemoryStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = MemoryStore(path)
    return _store


def export_json(user_id: str) -> str:
    return json.dumps(get_memory().snapshot(user_id), ensure_ascii=False, indent=2)

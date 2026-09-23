# -*- coding: utf-8 -*-
"""运行时：把图跑起来，并把过程中发生的事推给调用方。

服务层不该知道 Agent 内部有几个节点、工具是怎么调的。它只做三件事：
拿到一个会话、把用户这句话推进去、把推出来的事件转发给浏览器。

中间的桥是一根队列。图在后台线程里跑，事件往队列里丢，HTTP 那层从队列里读。
这样做的原因是同步：模型生成要几秒到几十秒，如果 HTTP 请求一直等着，
浏览器那边就只能白屏转圈。有了队列，第一个字出来就能显示出去。
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

from langgraph.types import Command

from .brain import make_brain
from .config import get_config
from .dataset import get_catalog
from .graph import build_graph, get_checkpointer
from .guard import Guard
from .memory import get_memory
from .models import Order
from .session import Session
from .tools import Shelf
from .trace import Trace

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
MAX_SESSIONS = 200          # 内存里最多留多少个活跃会话


class Runtime:
    def __init__(self) -> None:
        self.catalog = get_catalog()
        self.memory = get_memory()
        self._merge_written_orders()
        self._sessions: dict[str, Session] = {}
        self._graphs: dict[str, Any] = {}
        self._lock = threading.RLock()

    # ── 启动时把之前下过的单读回来 ──────────────────────
    def _merge_written_orders(self) -> None:
        """这个工程产生的订单写在 runtime 目录，重启后要能查到。

        演示数据（data/）始终是只读的，运行期产生的数据另存一处。
        这样「把数据换成你自己的」这件事不会因为跑过一次就被污染。
        """
        p = RUNTIME / "placed_orders.jsonl"
        if not p.exists():
            return
        seen = {o.order_id for o in self.catalog.orders}
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("order_id") in seen:
                        continue
                    o = Order(**{k: v for k, v in row.items()
                                 if k in Order.model_fields})
                    self.catalog.orders.append(o)
                    self.catalog.by_order[o.order_id] = o
                    self.catalog.orders_of.setdefault(o.user_id, []).append(o)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    # ── 会话 ────────────────────────────────────────────
    def _key(self, session_id: str, user_id: str) -> str:
        return f"{session_id}:{user_id}"

    def _prepare(self, session_id: str, user_id: str,
                 sink) -> tuple[Session, Any, dict]:
        key = self._key(session_id, user_id)
        with self._lock:
            sess = self._sessions.get(key)
            graph = self._graphs.get(key)
            if sess is None or graph is None:
                cfg = get_config()
                sess = Session(
                    user_id=user_id, session_id=session_id, catalog=self.catalog,
                    memory=self.memory, cfg=cfg,
                    guard=Guard(max_rounds=cfg.max_rounds, max_amount=cfg.max_amount),
                    trace=Trace(session_id))
                sess.brain = make_brain(cfg)
                graph = build_graph(sess, Shelf(sess))
                self._sessions[key] = sess
                self._graphs[key] = graph
                self._trim()
        sess.emitter = sink
        return sess, graph, {"configurable": {"thread_id": key}}

    def _trim(self) -> None:
        """会话缓存满了就丢最早的那批。丢掉的会话历史还在 SQLite 里，
        下一轮会重新建对象，对话不受影响。"""
        if len(self._sessions) <= MAX_SESSIONS:
            return
        for k in list(self._sessions)[:len(self._sessions) - MAX_SESSIONS]:
            self._sessions.pop(k, None)
            self._graphs.pop(k, None)

    def invalidate(self) -> None:
        """配置变了要全部重建 —— 换了模型，旧的 brain 还指着老端点。"""
        with self._lock:
            self._sessions.clear()
            self._graphs.clear()

    # ── 一轮对话 ────────────────────────────────────────
    def start(self, session_id: str, user_id: str, text: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        sess, graph, config = self._prepare(session_id, user_id, q.put)
        threading.Thread(
            target=self._worker, args=(sess, graph, config, q, text, None),
            daemon=True).start()
        return q

    def resume(self, session_id: str, user_id: str, approved: bool) -> queue.Queue:
        """人工确认的结果送回去。图会从被打断的那一步接着跑。"""
        q: queue.Queue = queue.Queue()
        sess, graph, config = self._prepare(session_id, user_id, q.put)
        threading.Thread(
            target=self._worker, args=(sess, graph, config, q, "", approved),
            daemon=True).start()
        return q

    def _worker(self, sess: Session, graph, config: dict, q: queue.Queue,
                text: str, resume: bool | None) -> None:
        try:
            sess.guard.reset()
            sess.clear_turn()
            # 这两个是本轮的工作台，每轮开工前都得清空。
            # `last_quote` 尤其不能留 —— 写操作的接管要靠它判断
            # 「用户说的成交是针对哪一笔」，留着上一轮的报价，
            # 用户这一轮说的「确认」就会被算到旧的那笔上。
            sess.facts.pop("prepared", None)
            sess.facts.pop("last_quote", None)
            sess.facts.pop("confirmed_amount", None)
            sess.facts.pop("order_rejected", None)
            sess.brain.sink = sess.trace.count_usage

            payload = Command(resume=resume) if resume is not None else \
                {"user_text": text, "events": []}

            reply = ""
            pending: dict | None = None
            for chunk in graph.stream(payload, config=config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    pending = _interrupt_payload(chunk["__interrupt__"])
                    continue
                for _node, upd in chunk.items():
                    if not isinstance(upd, dict):
                        continue
                    if upd.get("reply"):
                        reply = upd["reply"]
                    for ev in upd.get("events") or []:
                        q.put(ev)

            # 兜底：有些路径的 reply 只在最终状态里
            if not reply:
                try:
                    snap = graph.get_state(config)
                    reply = (snap.values or {}).get("reply") or ""
                except Exception:
                    reply = ""

            if pending:
                q.put({"kind": "confirm", "payload": pending})
                q.put({"kind": "awaiting", "reply": reply})
            else:
                trace_summary = sess.trace.summary()
                sess.trace.flush({"user_id": sess.user_id})
                q.put({"kind": "final", "reply": reply,
                       "summary": trace_summary,
                       "usage": sess.trace.usage,
                       "guard": {"blocked": sess.guard.recent_blocks(),
                                 "replays": sess.guard.replay_count()},
                       "mode": sess.brain.kind})
        except Exception as e:
            q.put({"kind": "error",
                   "message": f"{type(e).__name__}: {e}"})
        finally:
            sess.emitter = None
            q.put({"kind": "__end__"})

    # ── 读 ──────────────────────────────────────────────
    def history(self, session_id: str, user_id: str) -> list[dict]:
        key = self._key(session_id, user_id)
        with self._lock:
            graph = self._graphs.get(key)
        if graph is None:
            sess, graph, config = self._prepare(session_id, user_id, None)
            config = {"configurable": {"thread_id": key}}
        else:
            config = {"configurable": {"thread_id": key}}
        try:
            snap = graph.get_state(config)
            msgs = (snap.values or {}).get("messages") or []
            return [{"role": m.get("role", "assistant"),
                     "content": m.get("content", "")} for m in msgs]
        except Exception:
            return []

    def reset(self, session_id: str, user_id: str) -> None:
        """清空这个会话的短期历史。长期记忆不受影响。

        注意这里不能只把内存里的对象丢掉。状态是落在检查点里的，
        只丢对象的话，下一轮重建图会从同一个检查点把历史读回来 ——
        用户点了「清空会话」，界面上却还看得见之前说过的话。
        """
        key = self._key(session_id, user_id)
        with self._lock:
            self._sessions.pop(key, None)
            self._graphs.pop(key, None)
        try:
            get_checkpointer().delete_thread(key)
        except Exception:
            pass

    def snapshot(self, session_id: str, user_id: str) -> dict:
        key = self._key(session_id, user_id)
        with self._lock:
            sess = self._sessions.get(key)
        if sess is None:
            return {"active": False}
        return {
            "active": True,
            "mode": sess.brain.kind if sess.brain else "unknown",
            "label": sess.brain.label if sess.brain else "",
            "rounds": dict(sess.guard.rounds),
            "blocked": sess.guard.recent_blocks(),
            "replays": sess.guard.replay_count(),
            "usage": sess.trace.usage,
            "events": sess.trace.events[-60:],
        }

    def stats(self) -> dict:
        return {
            "data": self.catalog.stats(),
            "sessions_in_memory": len(self._sessions),
            "cache_dir": str(RUNTIME),
        }


def _interrupt_payload(intr) -> dict:
    """把中断对象里的内容取出来。不同版本的包装层不同，取最里面那层。"""
    try:
        first = intr[0] if isinstance(intr, (list, tuple)) else intr
        v = getattr(first, "value", first)
        return v if isinstance(v, dict) else {"value": v}
    except Exception:
        return {}


_runtime: Runtime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> Runtime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = Runtime()
        return _runtime

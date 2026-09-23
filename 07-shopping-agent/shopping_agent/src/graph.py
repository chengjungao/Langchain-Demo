# -*- coding: utf-8 -*-
"""主图：一轮对话的完整链路。

    装载 → 路由 → 支线 → 归档

四条支线（导购 / 订单 / 客服 / 闲聊）是四张独立的子图，各自有自己的
循环、工具清单、轮数预算和提示词。主图只负责决定走哪条，以及负责
所有支线都要做的那两件事：**进门前把上下文治理好，出门后把该记的记下来**。

为什么治理和归档要放在主图上，而不是每条支线各写一遍：

- 治理放在进门前，意味着三条支线拿到的历史是**同一份**。各写一遍的话，
  改一处裁剪策略要改三处，而且很容易改漏一处，于是同一句话在不同支线里
  行为不一致。
- 归档放在出门后，意味着不管用户走的是哪条线，记忆都会被记下来。
  如果只在导购线归档，用户在客服线说过的偏好就丢了 —— 这是很常见的漏。

状态里的 `messages` 保留全量，送给模型的却是裁剪过的。这两件事要分开，
因为**保留全量是为了可追溯，裁剪是为了省钱和让模型看得清**。
"""
from __future__ import annotations

import operator
import sqlite3
import threading
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents.loop import build_all_loops
from .agents.prompts import build_system
from .brain import BrainCtx
from .context import compose_memory_note, compose_request_messages, need_summary
from .memory import looks_memorable
from .session import Session
from .tools import Shelf

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"

REPLACE = "__REPLACE__"          # state 里的重置标记
MAX_KEEP_IN_STATE = 80           # state 里最多留多少条消息


def merge_messages(old: list | None, new: list | None) -> list:
    """消息列表的合并规则。

    默认是追加，遇到 `__REPLACE__` 开头的批次则整体替换。
    替换是必须的 —— 状态只增不减的话，聊上半天内存里就堆着一份完整记录，
    而且每一轮都要把它序列化一次。
    """
    if not new:
        return list(old or [])
    if isinstance(new, list) and new and new[0] == REPLACE:
        return list(new[1:])
    return list(old or []) + list(new)


class TurnState(TypedDict, total=False):
    user_text: str
    messages: Annotated[list, merge_messages]
    route: dict
    context_report: dict
    memory_note: str
    memory_hits: list
    reply: str
    events: Annotated[list, operator.add]
    archived: dict


# ── 检查点存储 ──────────────────────────────────────────────
# 会话历史落在 SQLite 里，进程重启后同一个 session_id 还能接着聊。
# 这是「会话可续」的落点：状态不放在进程内存里，就不会因为部署而丢。

_cp_lock = threading.Lock()
_checkpointer = None


def get_checkpointer():
    global _checkpointer
    with _cp_lock:
        if _checkpointer is None:
            RUNTIME.mkdir(parents=True, exist_ok=True)
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn = sqlite3.connect(str(RUNTIME / "sessions.db"),
                                   check_same_thread=False)
            _checkpointer = SqliteSaver(conn)
            _checkpointer.setup()
        return _checkpointer


# ── 图装配 ──────────────────────────────────────────────────

def build_graph(sess: Session, shelf: Shelf | None = None,
                with_checkpointer: bool = True):
    shelf = shelf or Shelf(sess)
    loops = build_all_loops(sess, shelf)
    cfg = sess.cfg

    # ── 节点一：装载 ────────────────────────────────────
    def load(state: TurnState) -> dict:
        """进门前做两件事：把长期记忆捞出来，把上下文裁到位。"""
        user_text = state.get("user_text", "")
        history = state.get("messages") or []

        # 1) 长期记忆召回
        sess.trace.start("recall")
        hits = sess.memory.recall(sess.user_id, user_text,
                                  limit=cfg.memory_limit if cfg else 6)
        note = compose_memory_note(hits)
        sess.facts["memory_note"] = note
        sess.facts["last_user_text"] = user_text
        recall_ms = sess.trace.stop("recall")

        events = []
        if hits:
            sess.trace.add("memory", "召回长期记忆",
                           "；".join(f"{h.label} {h.text[:24]}" for h in hits), recall_ms)
            events.append({"kind": "memory", "title": f"召回 {len(hits)} 条长期记忆",
                           "detail": "\n".join(f"· {h.label}：{h.text}" for h in hits),
                           "ms": recall_ms,
                           "items": [{"source": h.source, "label": h.label,
                                      "text": h.text, "score": h.score} for h in hits]})
        else:
            events.append({"kind": "memory", "title": "没有可用的长期记忆",
                           "detail": "这是第一次对话，或者记忆里没有跟当前话题相关的条目"})

        # 2) 上下文治理
        summary = sess.memory.session_summary(sess.session_id)
        prepared, report = compose_request_messages(
            history, user_text, summary=summary,
            memory_hits=hits, keep=cfg.history_limit if cfg else 12)

        # 被裁掉的内容够多时，压成一段摘要存起来。这一步要花一次模型调用，
        # 所以设了门槛（见 context.need_summary），不是每次都做。
        dropped_n = report["dropped"]
        if dropped_n >= 6 and need_summary(history[:dropped_n]):
            sess.trace.start("summary")
            new_summary = sess.brain.summarize(history[:dropped_n])
            ms = sess.trace.stop("summary")
            if new_summary:
                sess.memory.upsert_session(sess.session_id, sess.user_id,
                                           summary=new_summary)
                events.append({"kind": "govern", "title": "早期对话已压缩成摘要",
                               "detail": new_summary[:200], "ms": ms})
                prepared, report = compose_request_messages(
                    history, user_text,
                    summary=(summary + "；" + new_summary) if summary else new_summary,
                    memory_hits=hits, keep=cfg.history_limit if cfg else 12)
                report["summary_used"] = True

        sess.facts["prepared"] = prepared
        sess.trace.add("govern", "上下文治理",
                       f"历史 {report['history_total']} 条 → 保留 {report['kept']} 条，"
                       f"裁掉 {report['dropped']} 条；注入记忆 {report['memory_injected']} 条；"
                       f"估算 {report['est_tokens']} token")
        events.append({"kind": "govern", "title": "上下文治理",
                       "detail": f"历史 {report['history_total']} 条 → 保留 {report['kept']}，"
                                 f"裁掉 {report['dropped']}；"
                                 f"注入记忆 {report['memory_injected']} 条；"
                                 f"本次请求约 {report['est_tokens']} token",
                       "report": report})
        return {"memory_note": note, "context_report": report,
                "memory_hits": [{"label": h.label, "text": h.text,
                                 "score": h.score, "source": h.source} for h in hits],
                "events": events}

    # ── 节点二：路由 ────────────────────────────────────
    def route_node(state: TurnState) -> dict:
        sess.trace.start("route")
        r = sess.brain.route(state.get("user_text", ""), state.get("messages") or [])
        ms = sess.trace.stop("route")
        sess.trace.add("route", f"路由到 {_branch_label(r.route)}", r.reason, ms)

        # 路由这一步顺手抽出来的槽位（类目、预算、订单号、商品编码）要留下来。
        # 支线如果不知道这些，就会把用户刚说过的话再问一遍 ——
        # 而这次抽取的钱已经花过了，信息丢掉等于白花。
        slots = {k: v for k, v in r.model_dump().items()
                 if k not in ("route", "reason") and v}
        sess.facts["route"] = r.route
        sess.facts["route_slots"] = slots

        return {"route": r.model_dump(),
                "events": [{"kind": "route", "title": f"路由 → {_branch_label(r.route)}",
                            "detail": r.reason, "ms": ms,
                            "branch": r.route, "slots": slots}]}

    # ── 节点三：支线 ────────────────────────────────────
    def make_branch(role: str):
        def node(state: TurnState) -> dict:
            out = loops[role].invoke({
                "user_text": state.get("user_text", ""),
                "history": sess.facts.get("prepared") or [],
                "observations": [],
                "rounds": 0,
            })
            return {"reply": out.get("reply") or "", "events": out.get("events") or []}
        return node

    def chat_node(state: TurnState) -> dict:
        """闲聊兜底：不调工具，只回一句然后把话题引回能做事的范围。

        这条线同样要带上长期记忆。用户问「上次我说过什么」时，路由很可能
        判成闲聊，如果这里不带记忆，他会得到一句「没查到」—— 而记忆其实
        就在手边。这类「兜底支线忘了带上下文」的漏，在实际项目里很常见。
        """
        sess.trace.start("chat")
        ctx = BrainCtx(role="chat", user_text=state.get("user_text", ""),
                       system=build_system("chat", sess,
                                           memory_note=sess.facts.get("memory_note", ""),
                                           slots=sess.facts.get("route_slots")),
                       history=sess.facts.get("prepared") or [], tool_names=[])
        d = sess.brain.decide(ctx, [])
        ms = sess.trace.stop("chat")
        reply = d.reply or "我在。想找点什么，或者要查订单物流，都可以直接说。"
        sess.trace.add("chat", "闲聊兜底", reply[:120], ms)
        return {"reply": reply,
                "events": [{"kind": "chat", "title": "闲聊兜底（这条线不调工具）",
                            "detail": reply[:200], "ms": ms}]}

    # ── 节点四：归档 ────────────────────────────────────
    def archive(state: TurnState) -> dict:
        """出门后做三件事：写记忆、更新会话、把状态瘦下来。"""
        user_text = state.get("user_text", "")
        reply = state.get("reply") or ""
        events: list[dict] = []
        archived: dict = {}

        # 1) 记忆抽取。先过一个零成本的信号检测 —— 大部分日常寒暄里
        #    没有可记的东西，为它们各花一次模型调用不划算。
        sig = looks_memorable(user_text, reply)
        archived["signal"] = sig
        if sig["worth"]:
            sess.trace.start("extract")
            draft = sess.brain.extract(
                (sess.facts.get("prepared") or [])[-6:] +
                [{"role": "assistant", "content": reply}])
            ms = sess.trace.stop("extract")
            written = {"profile": [], "episode": []}
            for item in draft.profile:
                action = sess.memory.upsert_profile(sess.user_id, item)
                written["profile"].append({"key": item.key, "value": item.value,
                                           "action": action})
            for item in draft.episodes:
                ep = sess.memory.add_episode(sess.user_id, item, sess.session_id)
                if ep:
                    written["episode"].append({"summary": item.summary, "id": ep})
            archived["written"] = written
            if written["profile"] or written["episode"]:
                sess.trace.add("archive", "写入长期记忆",
                               f"画像 {len(written['profile'])} 条、事件 {len(written['episode'])} 条",
                               ms)
                events.append({"kind": "archive", "title": "写入长期记忆",
                               "detail": _describe_written(written), "ms": ms,
                               "written": written})
        else:
            sess.trace.add("archive", "跳过记忆抽取", "这轮没有可记的信号")
            events.append({"kind": "archive", "title": "跳过记忆抽取",
                           "detail": "这轮没检测到偏好或事件信号，省掉一次模型调用"})

        # 2) 会话元信息
        title = (user_text[:18] + "…") if len(user_text) > 18 else user_text
        sess.memory.upsert_session(sess.session_id, sess.user_id,
                                   title=title, turns_delta=1)

        # 3) 状态瘦身
        new_msgs: list = [{"role": "user", "content": user_text},
                          {"role": "assistant", "content": reply}]
        old = state.get("messages") or []
        if len(old) + 2 > MAX_KEEP_IN_STATE:
            new_msgs = [REPLACE] + old[-(MAX_KEEP_IN_STATE - 2):] + new_msgs
        return {"messages": new_msgs, "archived": archived, "events": events}

    # ── 装配 ────────────────────────────────────────────
    g = StateGraph(TurnState)
    g.add_node("load", load)
    g.add_node("route", route_node)
    g.add_node("advisor", make_branch("advisor"))
    g.add_node("order", make_branch("order"))
    g.add_node("service", make_branch("service"))
    g.add_node("chat", chat_node)
    g.add_node("archive", archive)

    g.add_edge(START, "load")
    g.add_edge("load", "route")
    g.add_conditional_edges("route", lambda s: (s.get("route") or {}).get("route", "chat"), {
        "advisor": "advisor", "order": "order",
        "service": "service", "chat": "chat"})
    for role in ("advisor", "order", "service", "chat"):
        g.add_edge(role, "archive")
    g.add_edge("archive", END)

    return g.compile(checkpointer=get_checkpointer() if with_checkpointer else None)


def _branch_label(route: str) -> str:
    return {"advisor": "导购线", "order": "订单线",
            "service": "客服线", "chat": "闲聊"}.get(route, route)


def _describe_written(w: dict) -> str:
    bits = []
    for p in w.get("profile", []):
        bits.append(f"画像·{p['key']}={p['value']}（{_action_label(p['action'])}）")
    for e in w.get("episode", []):
        bits.append(f"事件·{e['summary'][:30]}")
    return "\n".join(bits) or "没有新增"


def _action_label(a: str) -> str:
    return {"new": "新增", "update": "更新", "skip": "证据不足未改判"}.get(a, a)

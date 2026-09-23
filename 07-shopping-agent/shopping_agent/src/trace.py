# -*- coding: utf-8 -*-
"""轨迹：一次会话里发生了什么。

Agent 出问题时，「它答错了」这句话没有信息量。需要知道的是：路由判到了哪条线、
调了哪个工具、传了什么参数、工具真的返回了什么、卡在第几轮。

这份轨迹一边推给浏览器实时显示，一边落盘。落盘的理由很实际 ——
用户报问题的时候，你手上得有那次会话的完整记录，而不是让他复述。
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = ROOT / "runtime" / "traces"


class Trace:
    """收集一次会话的事件。线程安全，因为 SSE 推送和主流程不在同一个线程。"""

    def __init__(self, session_id: str, persist: bool = True) -> None:
        self.session_id = session_id
        self.events: list[dict] = []
        self.persist = persist
        self._lock = threading.RLock()
        self._t0 = time.time()
        self._marks: dict[str, float] = {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def add(self, kind: str, title: str, detail: str = "",
            ms: int | None = None, **extra) -> dict:
        """记一件事。kind 决定界面上用什么颜色和图标渲染。"""
        ev = {
            "seq": len(self.events) + 1,
            "at": datetime.now().strftime("%H:%M:%S"),
            "elapsed_ms": int((time.time() - self._t0) * 1000),
            "kind": kind,
            "title": title,
            "detail": detail,
            "ms": ms,
        }
        ev.update(extra)
        with self._lock:
            self.events.append(ev)
        return ev

    # ── 计时辅助 ────────────────────────────────────────
    def start(self, key: str) -> None:
        self._marks[key] = time.time()

    def stop(self, key: str) -> int:
        t = self._marks.pop(key, None)
        return int((time.time() - t) * 1000) if t else 0

    def count_usage(self, prompt: int, completion: int, reasoning: int = 0) -> None:
        """记录一次模型调用的用量。推理 token 单独记，因为它不产生回答，
        但照样计费 —— 本地跑是费电，云端跑是费钱。"""
        with self._lock:
            self.usage["prompt_tokens"] += prompt
            self.usage["completion_tokens"] += completion
            self.usage["reasoning_tokens"] = self.usage.get("reasoning_tokens", 0) + reasoning
            self.usage["calls"] += 1

    # ── 落盘 ────────────────────────────────────────────
    def flush(self, meta: dict | None = None) -> Path | None:
        if not self.persist:
            return None
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        p = TRACE_DIR / f"{self.session_id}.jsonl"
        with self._lock:
            events = list(self.events)
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"type": "meta", **(meta or {}),
                                    "session_id": self.session_id,
                                    "total_ms": int((time.time() - self._t0) * 1000),
                                    "usage": self.usage}, ensure_ascii=False) + "\n")
                for ev in events:
                    f.write(json.dumps({"type": "event", **ev}, ensure_ascii=False) + "\n")
        except OSError:
            return None
        return p

    def summary(self) -> dict:
        tools = [e for e in self.events if e["kind"] == "tool"]
        return {
            "events": len(self.events),
            "tool_calls": len(tools),
            "tools": [e["title"] for e in tools],
            "total_ms": int((time.time() - self._t0) * 1000),
            "usage": self.usage,
        }

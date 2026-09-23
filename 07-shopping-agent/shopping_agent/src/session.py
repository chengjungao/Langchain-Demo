# -*- coding: utf-8 -*-
"""会话上下文：一次对话里，各个部件共用的东西。

工具需要知道「当前是谁」，护栏需要知道「这次会话的阈值」，轨迹需要有人往里记。
这些都不该做成全局变量 —— 两个用户同时对话就会串。

所以每次对话创建一个 Session，把目录数据、记忆、护栏、轨迹、模型判断层
都挂在上面，工具通过闭包拿到属于自己这次会话的那一份。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .brain import Brain
from .config import ModelConfig
from .dataset import Catalog
from .guard import Guard
from .memory import MemoryStore
from .trace import Trace


@dataclass
class Session:
    user_id: str
    session_id: str = field(default_factory=lambda: "S" + uuid.uuid4().hex[:10])
    catalog: Catalog | None = None
    memory: MemoryStore | None = None
    guard: Guard | None = None
    trace: Trace | None = None
    brain: Brain | None = None
    cfg: ModelConfig | None = None

    # 会话内的短期状态。注意它不落盘 —— 落盘的是消息历史（checkpointer）
    # 和三层长期记忆，这里只放「这一轮正在处理什么」。
    facts: dict = field(default_factory=dict)
    observations: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # 往外推事件用的回调。由服务层接上，Agent 层只管往里丢 ——
    # 这样 Agent 不需要知道自己在被谁观察，也不需要知道推给了谁。
    emitter: object = None

    def emit(self, ev: dict) -> None:
        if callable(self.emitter):
            try:
                self.emitter(ev)
            except Exception:
                pass

    def emit_token(self, text: str) -> None:
        if text:
            self.emit({"kind": "token", "text": text})

    def remember_obs(self, tool: str, args: dict, result: str,
                     call_id: str = "") -> None:
        """记下一次工具调用，供下一轮拼消息时用。

        call_id 是模型给出这次调用时的编号。带着它，下一轮才能把结果
        当作**协议里的工具返回**回灌，而不是伪装成用户说的一句话。
        这两者对模型的影响完全不同 —— 见 brain.decide 里的说明。
        """
        self.observations.append({"tool": tool, "args": args,
                                  "result": result, "call_id": call_id})

    def clear_turn(self) -> None:
        self.observations.clear()

    @property
    def user(self):
        return self.catalog.by_user.get(self.user_id)

    def profile_note(self) -> str:
        """用户档案里那些不随对话变化的事实。"""
        u = self.user
        if u is None:
            return ""
        bits = [f"昵称 {u.nickname}", f"常驻 {u.city}", f"会员等级 {u.member_level}"]
        if u.budget_range:
            bits.append(f"历史预算区间 {u.budget_range[0]}~{u.budget_range[1]} 元")
        if u.preferred_brands:
            bits.append(f"偏好品牌 {'、'.join(u.preferred_brands)}")
        if u.disliked_brands:
            bits.append(f"回避品牌 {'、'.join(u.disliked_brands)}")
        if u.notes:
            bits.append("备注：" + "；".join(u.notes))
        return "；".join(bits)

    def describe(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "nickname": self.user.nickname if self.user else self.user_id,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.brain.kind if self.brain else "unknown",
        }

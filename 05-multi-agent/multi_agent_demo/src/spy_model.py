# -*- coding: utf-8 -*-
"""记录型假模型：不需要 API key，就能把每个 Agent 到底收到了什么记下来。

用法：
    from src.spy_model import SpyChatModel

    model = SpyChatModel(tag="order", script=[{"text": "答好了"}])
    # script 里每一项二选一：
    #   {"text": "..."}                      直接作答
    #   {"calls": [("tool_name", {...}), ...]}  发起工具调用（可一次发多个）
    # script 用完后，默认回复 "[tag] done"

记录：
    SpyChatModel.reset()         清空记录
    SpyChatModel.calls           本次运行里所有模型调用，每项含 who / n / messages
    SpyChatModel.calls_of("order")   只看某个 Agent 的调用

换真模型：把 SpyChatModel(...) 换成 ChatOpenAI(...) 之类的真实模型即可，
调用方代码不用动。
"""
from __future__ import annotations

import io
import sys
from typing import Any, ClassVar, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Windows 控制台按 UTF-8 输出，避免中文乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class SpyChatModel(BaseChatModel):
    """按脚本作答、并把每次调用看到的消息全记下来的假模型。"""

    tag: str = "spy"
    script: list = []

    # 类级共享记录，方便跨 Agent 对照
    calls: ClassVar[list] = []
    tools_offered: ClassVar[dict] = {}

    @property
    def _llm_type(self) -> str:
        return "spy"

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.tools_offered = {}

    @classmethod
    def calls_of(cls, tag: str) -> list:
        return [c for c in cls.calls if c["who"] == tag]

    # 1.x 里 BaseChatModel.bind_tools 默认抛 NotImplementedError，假模型必须自己实现
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        SpyChatModel.tools_offered[self.tag] = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        SpyChatModel.calls.append({"who": self.tag, "n": len(messages), "messages": list(messages)})

        step = self.script.pop(0) if self.script else {"text": f"[{self.tag}] done"}
        calls = step.get("calls")
        if calls:
            tool_calls = [
                {
                    "name": name if isinstance(name, str) else name[0],
                    "args": (name[1] if isinstance(name, tuple) and len(name) > 1 else {}),
                    "id": f"{self.tag}-c{i}",
                }
                for i, name in enumerate(calls)
            ]
            message = AIMessage(content="", tool_calls=tool_calls)
        else:
            message = AIMessage(content=step.get("text", ""))

        return ChatResult(generations=[ChatGeneration(message=message)], llm_output={})


def make_model(tag: str, script: list) -> SpyChatModel:
    """建一个假模型，脚本可以后续替换。"""
    return SpyChatModel(tag=tag, script=list(script))

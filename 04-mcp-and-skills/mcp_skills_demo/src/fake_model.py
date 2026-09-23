# -*- coding: utf-8 -*-
"""测试替身模型：不联网、不需要 API key，用来验证中间件与链路行为。

两个：
- RecordingModel   记录每一轮模型实际看到的工具与 system message
- ScriptedModel    按剧本吐 tool_call，用来跑通完整工序
"""
from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class RecordingModel(BaseChatModel):
    """把每一轮绑上来的工具与 system message 记下来，然后回一句固定文本。"""

    reply: str = "（测试替身）收到。"
    seen_tools: list[list[str]] = Field(default_factory=list)
    seen_system: list[str] = Field(default_factory=list)
    seen_messages: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        # 这里拿到的就是模型真正会被绑定的工具清单
        self.seen_tools.append([getattr(t, "name", str(t)) for t in tools])
        return self.bind(tools=list(tools), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = ""
        for m in messages:
            if getattr(m, "type", "") in ("system", "developer") and isinstance(m.content, str):
                system = m.content
                break
        self.seen_system.append(system)
        self.seen_messages.append(len(messages))
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.reply))],
            llm_output={"model_name": self._llm_type},
        )


class ScriptedModel(BaseChatModel):
    """按剧本依次返回消息，用来把一条完整工序跑到底。"""

    script: list[AIMessage] = Field(default_factory=list)
    cursor: int = 0
    seen_tools: list[list[str]] = Field(default_factory=list)
    seen_system: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        self.seen_tools.append([getattr(t, "name", str(t)) for t in tools])
        return self.bind(tools=list(tools), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = ""
        for m in messages:
            if getattr(m, "type", "") in ("system", "developer") and isinstance(m.content, str):
                system = m.content
                break
        self.seen_system.append(system)

        if self.cursor >= len(self.script):
            msg = AIMessage(content="（剧本已走完）")
        else:
            msg = self.script[self.cursor]
            self.cursor += 1
        return ChatResult(
            generations=[ChatGeneration(message=msg)],
            llm_output={"model_name": self._llm_type},
        )


def call(name: str, args: dict, call_id: str) -> AIMessage:
    """构造一条带单个 tool_call 的 AI 消息。"""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def last_human_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "human" and isinstance(m.content, str):
            return m.content
    return ""

"""把长期记忆接进 LangGraph Agent：两种接入姿势。

姿势一 · 工具内访问
    把"查记忆"和"记事情"做成工具，交给模型决定什么时候用。
    适合用户偏好这种需要模型判断时机的内容。

姿势二 · 流程内固定写入
    在回来的路上挂一个节点，把这一轮用户说的话固定交给记忆系统。
    不依赖模型的判断，该落的一定会落。适合审计、合规这类硬要求。

生产上两者混用：硬规则走姿势二兜底，需要判断的走姿势一补充。

一个容易踩的坑：节点函数里注入的是 Runtime，工具函数里注入的是 ToolRuntime。
两者字段不同（ToolRuntime 多带 state 与 tool_call_id），注解写错会在调用时报
"Error invoking tool"，而且错误信息不告诉你注解错了。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime, tools_condition
from langgraph.runtime import Runtime

from .manager import MemoryManager
from .schema import MemoryType


@dataclass
class AgentContext:
    """随每次调用传入的上下文。user_id 是记忆隔离的依据。

    用 context 传身份，而不是用闭包把 user_id 绑进工具：
    这样一张编译好的图能服务所有用户，否则一个用户就得编译一张图。
    """

    user_id: str


# ---------------------------------------------------------------- 姿势一：工具


def make_memory_tools(memory: MemoryManager) -> list:
    """把记忆能力做成工具。"""

    @tool
    def recall_memory(query: str, runtime: ToolRuntime[AgentContext]) -> str:
        """检索当前用户的长期记忆。

        当用户提到"我之前说过""我的偏好""你记得吗"这类内容时调用。
        query 用自然语言描述你要找什么。
        """
        user_id = runtime.context.user_id
        memories = memory.recall(user_id, query, k=5, min_score=0.02)
        if not memories:
            return "没有检索到相关记忆。"
        return "；".join(m.content for m in memories)

    @tool
    def save_memory(
        content: str,
        memory_type: str,
        runtime: ToolRuntime[AgentContext],
    ) -> str:
        """把一条关于用户的稳定信息写入长期记忆。

        content 要写成自洽的第三人称陈述，例如"用户偏好用表格对比方案"。
        memory_type 取 semantic（事实与偏好）、episodic（经历）、procedural（规则）。
        """
        user_id = runtime.context.user_id
        try:
            parsed = MemoryType(memory_type)
        except ValueError:
            parsed = MemoryType.SEMANTIC
        report = memory.remember(
            user_id, content, memory_type=parsed, source="agent_tool"
        )
        return f"已写入长期记忆：{report.summary()}"

    return [recall_memory, save_memory]


# ---------------------------------------------------------------- 姿势二：固定写入


def make_writeback_node(memory: MemoryManager):
    """回来的路上固定写一次记忆，不看模型脸色。"""

    def writeback(state: MessagesState, runtime: Runtime[AgentContext]) -> dict:
        user_id = runtime.context.user_id
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is not None:
            memory.remember(user_id, str(last_human.content), source="writeback")
        return {}

    return writeback


# ---------------------------------------------------------------- 本地演示用模型


# 脚本化模型的判定词。放在模块级是因为 BaseChatModel 继承自 pydantic 模型，
# 类属性不加注解会被当成字段定义，直接抛 PydanticUserError。
QUERY_HINTS = ("什么", "怎么", "该", "吗", "？", "?", "帮我", "推荐", "记得")
SAVE_HINTS = ("以后", "记住", "我喜欢", "我偏好", "我不用", "我只认", "我叫", "换成")


class ScriptedMemoryModel(BaseChatModel):
    """脚本化模型：按固定规则决定调哪个工具。

    它的唯一用途是让整条链路在没有 API key 的情况下也能跑完。
    生产上换成任意 LangChain 1.x 的 chat model 即可，图本身不用改。
    """

    @property
    def _llm_type(self) -> str:
        return "scripted-memory-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedMemoryModel":
        """脚本化模型不需要真的绑定工具，它按规则决定调哪个。

        但 LangChain 1.x 里 BaseChatModel.bind_tools 的默认实现直接抛
        NotImplementedError，所以必须显式返回自己，否则编译图这一步就挂。
        """
        return self

    def _generate(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]

        # 工具刚返回结果，收尾作答
        if isinstance(last, ToolMessage):
            message = AIMessage(content=f"（据我记住的）{last.content}")
            return ChatResult(generations=[ChatGeneration(message=message)])

        text = str(getattr(last, "content", ""))

        if any(hint in text for hint in SAVE_HINTS):
            message = AIMessage(
                content="",
                tool_calls=[{
                    "name": "save_memory",
                    "args": {"content": text, "memory_type": "semantic"},
                    "id": f"call_{uuid4().hex[:8]}",
                }],
            )
        elif any(hint in text for hint in QUERY_HINTS):
            message = AIMessage(
                content="",
                tool_calls=[{
                    "name": "recall_memory",
                    "args": {"query": text},
                    "id": f"call_{uuid4().hex[:8]}",
                }],
            )
        else:
            message = AIMessage(content="好的。")

        return ChatResult(generations=[ChatGeneration(message=message)])


# ---------------------------------------------------------------- 组装


def build_memory_agent(
    memory: MemoryManager,
    model: BaseChatModel | None = None,
    checkpointer: Any = None,
    writeback: bool = True,
):
    """编译一张带长期记忆的 Agent 图。

    图结构：
        START -> model -> (有工具调用？) -> tools -> model
                       \\-> writeback -> END
    """
    tools = make_memory_tools(memory)
    bound_model = (model or ScriptedMemoryModel()).bind_tools(tools)

    def call_model(state: MessagesState, runtime: Runtime[AgentContext]) -> dict:
        return {"messages": [bound_model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState, context_schema=AgentContext)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "model")
    if writeback:
        builder.add_node("writeback", make_writeback_node(memory))
        builder.add_conditional_edges(
            "model", tools_condition, {"tools": "tools", "__end__": "writeback"}
        )
        builder.add_edge("writeback", "__end__")
    else:
        builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    return builder.compile(checkpointer=checkpointer)

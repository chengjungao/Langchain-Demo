# -*- coding: utf-8 -*-
"""把 MCP 与 Skills 装进同一个 Agent。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

from src.mcp_client import pose_get_tools
from src.mcp_client import stdio_conn
from src.skills_middleware import SkillLibraryMiddleware
from src.tool_gateway import ToolGateway, ToolGatewayMiddleware

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"

DEFAULT_SYSTEM_PROMPT = (
    "你是一名售后处理助手。回答时先给结论，再给依据。"
    "涉及金额、时效、政策条款时，只用工具返回值里的数字。"
)


async def mcp_tools(connections: dict | None = None) -> list[BaseTool]:
    """按需从 MCP server 取工具。"""
    connections = connections or {"order": stdio_conn("order_server.py")}
    return await pose_get_tools(connections)


@dataclass
class Assembled:
    """组装结果。中间件单独留出来，方便 demo 断言。"""

    agent: Any
    tools: list[BaseTool]
    middleware: list = field(default_factory=list)

    @property
    def skills(self) -> SkillLibraryMiddleware | None:
        for m in self.middleware:
            if isinstance(m, SkillLibraryMiddleware):
                return m
        return None

    @property
    def gateway(self) -> ToolGatewayMiddleware | None:
        for m in self.middleware:
            if isinstance(m, ToolGatewayMiddleware):
                return m
        return None


def build_agent(
    model,
    *,
    tools: list[BaseTool] | None = None,
    gateway: ToolGateway | None = None,
    use_skills: bool = True,
    skill_dirs: list[str | Path] | None = None,
    top_k: int = 8,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Assembled:
    """组装 Agent。

    gateway 给了就走货架模式：全量工具注册，模型每轮只看检索命中的几个。
    """
    middleware = []
    if gateway is not None:
        middleware.append(ToolGatewayMiddleware(gateway, top_k=top_k))
        agent_tools = gateway.all_tools()
    else:
        agent_tools = list(tools or [])

    if use_skills:
        middleware.append(SkillLibraryMiddleware(skill_dirs or [SKILLS_DIR]))

    agent = create_agent(
        model=model,
        tools=agent_tools,
        middleware=middleware,
        system_prompt=system_prompt,
    )
    return Assembled(agent=agent, tools=agent_tools, middleware=middleware)

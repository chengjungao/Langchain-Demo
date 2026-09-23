# -*- coding: utf-8 -*-
"""知识库服务 MCP Server（streamable-http 传输）。

除了一个检索工具，还暴露一个 resource，用来对比两类原语的区别：
resource 是「可读的数据」，由客户端按需拉取；
tool 是「可调用的动作」，由模型决定何时调用。

启动：
    python servers/kb_server.py --port 8931
"""
from __future__ import annotations

import argparse
import logging

from mcp.server.fastmcp import FastMCP

logging.getLogger("mcp").setLevel(logging.WARNING)

mcp = FastMCP("kb-server")

_POLICY = {
    "refund.window": "签收后 7 天内可申请无理由退款，质量问题 30 天内。",
    "refund.shipping": "已发货订单退款扣除运费 12 元，质量问题除外。",
    "refund.invoice": "已开票订单退款需扣除 6% 税点，或由财务先冲红再全额退。",
    "refund.approval": "单笔退款金额超过 500 元须走人工审批，审批时效 1 个工作日。",
    "refund.priority": "客户等级为 VIP 的退款申请优先处理，须在 4 小时内给出结论。",
}


@mcp.tool()
def search_policy(query: str, top_k: int = 3) -> list[dict]:
    """检索退款与售后政策知识库。

    按关键词匹配政策条目，返回命中条目的键、正文与相关度。
    在判断退款规则、审批门槛或时效时调用本工具，不要凭记忆回答政策问题。

    Args:
        query: 检索关键词，例如「审批」「运费」「VIP」
        top_k: 返回条数上限
    """
    hits = []
    for key, text in _POLICY.items():
        score = 0
        for token in [t for t in query.replace("，", " ").split() if t]:
            if token in text or token in key:
                score += 1
        if query and query in text:
            score += 2
        if score:
            hits.append({"key": key, "text": text, "score": score})
    hits.sort(key=lambda x: (-x["score"], x["key"]))
    return hits[:top_k]


@mcp.resource("policy://refund/rules")
def refund_rules() -> str:
    """退款政策全文（resource 原语）。"""
    return "\n".join(f"[{k}] {v}" for k, v in _POLICY.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8931)
    args = parser.parse_args()
    mcp.settings.port = args.port
    mcp.settings.host = "127.0.0.1"
    mcp.run(transport="streamable-http")

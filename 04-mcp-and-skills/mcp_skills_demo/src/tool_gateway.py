# -*- coding: utf-8 -*-
"""货架式工具网关。

思路：全量工具注册在货架上，模型每轮只看得到检索命中的几个。

- 命名规范：{域}_{对象}_{动作}，前缀就是分组
- 分级：read / write / danger 三档，危险操作不进自动检索结果
- 清单与货架分离：注册是注册，可见是可见

同时提供合成工具工厂，用来验证「工具成千上万」时的行为。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.tools import BaseTool, StructuredTool

# 危险动作关键词：即使检索命中也不自动放进上下文
DANGER_WORDS = ("delete", "drop", "purge", "refund_submit", "reset", "revoke")
WRITE_WORDS = ("create", "update", "submit", "write", "import", "publish")


def classify(name: str) -> tuple[str, str]:
    """按命名推断 (分组, 危险等级)。"""
    group = name.split("_", 1)[0] if "_" in name else "misc"
    low = name.lower()
    if any(w in low for w in DANGER_WORDS):
        return group, "danger"
    if any(w in low for w in WRITE_WORDS):
        return group, "write"
    return group, "read"


@dataclass
class ToolCard:
    """货架上的工具卡片。"""

    tool: BaseTool
    group: str
    level: str
    keywords: set[str]

    @property
    def name(self) -> str:
        return self.tool.name


def _tokenize(text: str) -> set[str]:
    """切词。英文按分隔符切，中文按连续片段切二元组，这样中文查询才检得动。"""
    tokens: set[str] = set()
    buf = ""
    run: list[str] = []

    def flush_word() -> None:
        nonlocal buf
        if len(buf) > 1:
            tokens.add(buf)
        buf = ""

    def flush_run() -> None:
        nonlocal run
        for i in range(len(run) - 1):
            tokens.add(run[i] + run[i + 1])
        run = []

    for ch in text.lower():
        if "\u4e00" <= ch <= "\u9fff":
            flush_word()
            run.append(ch)
            continue
        flush_run()
        if ch.isalnum() and ord(ch) < 128:
            buf += ch
        else:
            flush_word()
    flush_run()
    flush_word()
    return tokens


class ToolGateway:
    """工具货架。"""

    def __init__(self, tools: Iterable[BaseTool], always_visible: Sequence[str] = ()):
        self._cards: dict[str, ToolCard] = {}
        self.always_visible = list(always_visible)
        self.register(tools)

    def register(self, tools: Iterable[BaseTool]) -> None:
        for t in tools:
            group, level = classify(t.name)
            self._cards[t.name] = ToolCard(
                tool=t,
                group=group,
                level=level,
                keywords=_tokenize(t.name) | _tokenize(t.description or ""),
            )

    # ---------- 货架视图 ----------

    @property
    def size(self) -> int:
        return len(self._cards)

    def names(self) -> set[str]:
        return set(self._cards)

    def all_tools(self) -> list[BaseTool]:
        return [c.tool for c in self._cards.values()]

    def groups(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self._cards.values():
            out[c.group] = out.get(c.group, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def levels(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self._cards.values():
            out[c.level] = out.get(c.level, 0) + 1
        return out

    # ---------- 检索 ----------

    def search(self, query: str, top_k: int = 8,
               allow_danger: bool = False) -> list[BaseTool]:
        """按意图检索出可见工具。先过滤，再打分。"""
        picked: list[BaseTool] = []
        for name in self.always_visible:
            card = self._cards.get(name)
            if card is not None:
                picked.append(card.tool)

        q = _tokenize(query)
        scored: list[tuple[float, str]] = []
        for name, card in self._cards.items():
            if name in self.always_visible:
                continue
            if card.level == "danger" and not allow_danger:
                continue
            hit = len(q & card.keywords)
            if hit == 0:
                continue
            score = hit + (0.5 if card.level == "read" else 0.0)
            scored.append((score, name))

        scored.sort(key=lambda x: (-x[0], x[1]))
        for _, name in scored[:max(0, top_k - len(picked))]:
            picked.append(self._cards[name].tool)
        return picked

    def describe(self) -> str:
        g = "，".join(f"{k} {v}" for k, v in list(self.groups().items())[:6])
        lv = "，".join(f"{k} {v}" for k, v in self.levels().items())
        return f"货架共 {self.size} 个工具｜分组：{g}｜分级：{lv}"


class ToolGatewayMiddleware(AgentMiddleware):
    """用 wrap_model_call 把货架上的工具按需注入。

    request.tools 是模型这一轮真正拿到的工具清单，
    override 之后写回，模型侧看到的就只有检索命中的那几个。
    """

    def __init__(self, gateway: ToolGateway, top_k: int = 8,
                 query_of: Callable[[Any], str] | None = None):
        super().__init__()
        self.shelf = gateway
        self.top_k = top_k
        self.query_of = query_of or _default_query
        self.calls: list[dict] = []

    def _narrow(self, request: ModelRequest) -> ModelRequest:
        query = self.query_of(request)
        on_shelf = self.shelf.names()

        # 只接管货架上的工具。中间件自己注入的工具（例如 read_skill）原样保留，
        # 否则一个 override 就把别的中间件加的工具一起抹掉了。
        keep = [t for t in (request.tools or []) if t.name not in on_shelf]
        visible = keep + self.shelf.search(query, top_k=self.top_k)

        self.calls.append({
            "query": query,
            "before": len(request.tools or []),
            "after": len(visible),
            "kept": [t.name for t in keep],
            "names": [t.name for t in visible],
        })
        return request.override(tools=visible)

    def wrap_model_call(self, request: ModelRequest, handler):
        return handler(self._narrow(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        return await handler(self._narrow(request))


def _default_query(request: ModelRequest) -> str:
    """默认取最后一条人类消息作为检索意图。"""
    for msg in reversed(request.messages or []):
        if getattr(msg, "type", "") == "human":
            content = msg.content
            if isinstance(content, str):
                return content
            return " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
    return ""


# ---------- 合成工具工厂（用于验证规模化行为） ----------

_VERBS = ("query", "list", "get", "create", "update", "submit", "delete", "export")
_OBJECTS = ("order", "refund", "product", "customer", "ticket", "invoice",
            "warehouse", "coupon", "review", "settlement")
_VERB_CN = {"query": "查询", "list": "列表", "get": "获取", "create": "创建",
            "update": "更新", "submit": "提交", "delete": "删除", "export": "导出"}
_OBJ_CN = {"order": "订单", "refund": "退款", "product": "商品", "customer": "客户",
           "ticket": "工单", "invoice": "发票", "warehouse": "仓储", "coupon": "优惠券",
           "review": "评价", "settlement": "结算"}


def make_tool(name: str, description: str, params: Sequence[str]) -> BaseTool:
    """按名称与参数清单造一个结构化工具。"""
    from pydantic import Field, create_model

    fields = {p: (str, Field(description=f"{p} 参数")) for p in params}
    schema = create_model(f"{name}_args", **fields)

    def _fn(**kwargs: Any) -> str:
        return f"{name} -> " + ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

    return StructuredTool(name=name, description=description,
                          args_schema=schema, func=_fn)


def synthetic_tools(count: int, domains: int = 20) -> list[BaseTool]:
    """造 count 个业务风格工具，4~7 个参数，中文描述。"""
    tools: list[BaseTool] = []
    domains = max(1, domains)
    for i in range(count):
        dom = f"dom{i % domains:02d}"
        verb = _VERBS[i % len(_VERBS)]
        obj = _OBJECTS[(i // len(_VERBS)) % len(_OBJECTS)]
        name = f"{dom}_{obj}_{verb}_{i:04d}"
        n_params = 4 + (i % 4)
        params = [f"p{j}_id" for j in range(n_params - 1)] + ["tenant_id"]
        desc = (f"{_OBJ_CN[obj]}{_VERB_CN[verb]}。用于 {dom} 域的业务操作，"
                f"需要提供 {n_params} 个参数，调用前请确认 tenant_id 与调用方一致。")
        tools.append(make_tool(name, desc, params))
    return tools

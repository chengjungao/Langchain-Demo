# -*- coding: utf-8 -*-
"""工具货架：把工具收进来，按支线分发。

这里的做法对应前文讲过的**工具分层**：

- **货架**（shelf）是全部工具，一个进程里只定义一次
- **清单**（for_role）按支线发放子集

为什么要分：如果三条支线都拿到全部工具，导购线就会看到 `place_order`，
然后在用户只是问「这款怎么样」的时候顺手把单下了。工具越多，误选的概率越高，
而且每一次工具定义都要占提示词的 token。发它用得上的那几个，是同时省成本和降风险。
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from ..session import Session
from .catalog import build_catalog_tools
from .order import build_order_tools
from .service import build_service_tools

# 每条支线能拿到的工具。改这里就是改权限，比改提示词可靠。
TOOL_GROUPS: dict[str, tuple[str, ...]] = {
    "advisor": ("search_products", "get_product", "compare_products",
                "get_reviews", "recall_preferences"),
    # 订单线额外带上 search_products。原因是下单要先「认货」：
    # 用户说的是商品名，而 place_order 要的是商品编码，中间这一步
    # 如果没工具可调，模型只能回头反问用户「SKU 是多少」—— 这在真实
    # 对话里是很扫兴的一轮。它是只读工具，放进这条线不增加写风险。
    "order": ("search_products", "list_my_orders", "get_order",
              "get_logistics", "quote_price", "place_order"),
    "service": ("search_policies", "check_return", "create_ticket"),
}

# 会改变系统状态的工具。它们需要额外对待：显示上要突出，执行前要过确认。
WRITE_TOOLS = {"place_order", "create_ticket"}


class Shelf:
    """一次会话的工具集合。"""

    def __init__(self, sess: Session) -> None:
        self.sess = sess
        all_tools: list[BaseTool] = []
        all_tools += build_catalog_tools(sess)
        all_tools += build_order_tools(sess)
        all_tools += build_service_tools(sess)
        self.by_name: dict[str, BaseTool] = {t.name: t for t in all_tools}

    def for_role(self, role: str) -> list[BaseTool]:
        names = TOOL_GROUPS.get(role, ())
        return [self.by_name[n] for n in names if n in self.by_name]

    def invoke(self, name: str, args: dict) -> str:
        """统一执行入口。

        所有工具调用都从这里过，好处是三个横切关注点只需要写一遍：
        参数兜底、轨迹记录、异常处理。工具本身因此可以写得很干净。
        """
        tool = self.by_name.get(name)
        if tool is None:
            return f"没有这个工具：{name}。可用的有：{'、'.join(self.by_name)}"
        clean = {k: v for k, v in (args or {}).items()
                 if k in getattr(tool, "args_schema", type("x", (), {"model_fields": {}})).model_fields}
        try:
            out = tool.invoke(clean)
            return out if isinstance(out, str) else str(out)
        except Exception as e:                      # 工具不该把异常抛给模型
            return f"工具 {name} 执行出错：{type(e).__name__}: {e}"

    def schema_lines(self, role: str) -> str:
        """给提示词用的工具清单。只列名字和一句话说明。"""
        out = []
        for t in self.for_role(role):
            first = (t.description or "").strip().splitlines()[0]
            out.append(f"- {t.name}：{first}")
        return "\n".join(out)

    def names(self, role: str) -> list[str]:
        return [t.name for t in self.for_role(role)]

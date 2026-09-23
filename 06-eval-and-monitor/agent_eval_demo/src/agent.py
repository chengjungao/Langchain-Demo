"""一个可替换的 Agent。

设计意图：整篇《评测与监控》要讲的是「怎么知道它变好了还是变坏了」，
所以被测对象必须能被人为改坏，否则没法演示门禁抓退化。

本模块提供两条路：

  1. 脚本模型（默认）——按输入查表作答，完全确定、零成本、可复现。
     两个版本：
       good  全部答对（当基线）
       bad   常见类目照旧答对，长尾与多意图全部塞进常见类目，并编造一个订单号
     这就是演示里的「人为退化」。

  2. 本地真模型——指向 LM Studio（OpenAI 兼容端点），模型名走环境变量。
     需要真模型时用 build_agent(use_local=True)，没起服务会抛错。

另外提供一个「安静的调用」包装 invoke_quiet()：把 LangChain 自己的 run
上报关掉。不关的话，离线评测时后台线程会刷一批 401 到日志里。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

# ---------------------------------------------------------------- 评测集配套答案
# 键是用户原话，值是 (intent, priority, order_id)。
# 与 data/eval_set.jsonl 里的 gold 标签一一对应。

GOOD: dict[str, tuple[str, str, str | None]] = {
    "我要退货，订单 A1001 还没到": ("refund", "normal", "A1001"),
    "发票开错了，抬头写错了": ("invoice", "normal", None),
    "快递三天没动了": ("logistics", "normal", None),
    "东西坏了，但我更想直接退钱": ("refund", "high", None),
    "发票要改成公司名，另外包裹少了一件": ("invoice", "high", None),
    "我不急，就是想问问保修到什么时候": ("warranty", "low", None),
}

# 退化版：前三条（easy 档）一条没坏，后三条（hard 档）错两条、并编造订单号。
BAD: dict[str, tuple[str, str, str | None]] = {
    "我要退货，订单 A1001 还没到": ("refund", "normal", "A1001"),
    "发票开错了，抬头写错了": ("invoice", "normal", None),
    "快递三天没动了": ("logistics", "normal", None),
    "东西坏了，但我更想直接退钱": ("logistics", "normal", None),  # 意图 + 优先级都错
    "发票要改成公司名，另外包裹少了一件": ("invoice", "normal", None),  # 优先级错
    "我不急，就是想问问保修到什么时候": ("invoice", "normal", "A9999"),  # 意图错 + 编造订单号
}

VARIANTS = {"good": GOOD, "bad": BAD}

# 话术模板。回复里出现的类目词，正是 must_include 判定器要查的东西。
REPLY = {
    "refund": "已受理您的退款申请，预计 48 小时内处理。",
    "invoice": "已受理您的发票申请，预计 24 小时内开具。",
    "logistics": "已为您查询物流进度，请留意后续更新。",
    "warranty": "已为您查询保修信息，具体以商品页为准。",
    "unknown": "抱歉，暂时无法识别您的问题。",
}

SYSTEM_PROMPT = "你是客服工单分诊助手，只输出 JSON。"


@tool
def lookup_policy(intent: str) -> str:
    """查询某类意图的处理时限。"""
    return {"refund": "48 小时", "invoice": "24 小时"}.get(intent, "3 个工作日")


@tool
def get_order(order_id: str) -> str:
    """查询订单状态。"""
    return f"订单 {order_id}：已发货"


@tool
def calc_refund(amount: float) -> str:
    """试算退款金额。"""
    return f"可退 {amount * 0.9:.2f} 元"


TOOLS = [lookup_policy]
TOOL_CALLING_TOOLS = [get_order, calc_refund]


class TriageModel(BaseChatModel):
    """按输入查表的脚本模型。

    它同时造出可复现的 token 数，让监控指标有东西可算：
    输入 token 折算成「用户原话长度 + 系统提示开销 180」。
    """

    variant: str = "good"
    latency: float = 0.01  # 每次调用固定磨一小会儿，留出可测量的耗时

    @property
    def _llm_type(self) -> str:
        return "scripted-triage"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self

    def _answer(self, text: str) -> str:
        intent, priority, order_id = VARIANTS[self.variant].get(text, ("unknown", "low", None))
        reply = REPLY.get(intent, REPLY["unknown"])
        if order_id:
            reply = f"{reply}订单号 {order_id}。"
        return json.dumps(
            {"intent": intent, "priority": priority, "order_id": order_id, "reply": reply},
            ensure_ascii=False,
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        last = messages[-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        payload = self._answer(text)
        time.sleep(self.latency)
        msg = AIMessage(
            content=payload,
            usage_metadata={
                "input_tokens": len(text) + 180,  # 180 折算系统提示与工具定义的开销
                "output_tokens": len(payload),
                "total_tokens": len(text) + 180 + len(payload),
            },
            response_metadata={"model_name": f"scripted-triage-{self.variant}", "finish_reason": "stop"},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class ToolCallingModel(BaseChatModel):
    """先并发调两个工具、再收口的脚本模型。

    用来演示「一次带工具调用的请求，本地能抓到几条 run」。
    在 langchain 1.x + langgraph 下，一次这样的请求会拍到
    2 条 llm + 2 条 tool + 1 条 chain，共 5 条，且全部平级。
    """

    idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.idx == 0:
            self.idx = 1
            msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_order", "args": {"order_id": "A1001"}, "id": "call_1"},
                    {"name": "calc_refund", "args": {"amount": 200.0}, "id": "call_2"},
                ],
                usage_metadata={"input_tokens": 412, "output_tokens": 38, "total_tokens": 450},
                response_metadata={"model_name": "scripted-tool-calling", "finish_reason": "tool_calls"},
            )
        else:
            self.idx = 0
            msg = AIMessage(
                content="订单 A1001 已发货，可退 180.00 元。",
                usage_metadata={"input_tokens": 530, "output_tokens": 22, "total_tokens": 552},
                response_metadata={"model_name": "scripted-tool-calling", "finish_reason": "stop"},
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# ---------------------------------------------------------------- Agent 装配
def build_agent(variant: str = "good", use_local: bool = False, model: Any = None) -> Any:
    """装一个 Agent。

    variant   good / bad —— 只对脚本模型有效
    use_local True 时接本地 LM Studio（需要服务在跑）
    model     直接塞一个模型实例，优先级最高
    """
    if model is None:
        model = local_model() if use_local else TriageModel(variant=variant)
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def build_tool_calling_agent() -> Any:
    """装一个会并发调用两个工具的 Agent，用于 trace 结构演示。"""
    return create_agent(
        model=ToolCallingModel(), tools=TOOL_CALLING_TOOLS, system_prompt="你是订单助手。"
    )


# ---------------------------------------------------------------- 本地真模型
def local_model(
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    timeout: int = 180,
) -> Any:
    """接一个 OpenAI 兼容端点。默认指向 LM Studio。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=base_url or os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:1234/v1"),
        api_key=os.environ.get("LOCAL_API_KEY", "lm-studio"),
        model=model or os.environ.get("LOCAL_MODEL", "qwen/qwen3.5-9b"),
        temperature=temperature,
        timeout=timeout,
    )


def local_llm_available(base_url: str | None = None, timeout: float = 2.0) -> bool:
    """本地端点通不通。不通就让相关脚本体面地跳过，而不是崩掉。"""
    import urllib.error
    import urllib.request

    url = (base_url or os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:1234/v1")).rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


# ---------------------------------------------------------------- 安静的调用
def invoke_quiet(agent: Any, messages: list, capture: bool = True):
    """跑一次 Agent，可选把 trace 抓下来，同时关掉对外的 run 上报。

    返回 (输出, runs)。

    为什么必须关上报：evaluate(upload_results=False) 内部会把追踪上下文设成
    local 模式，它管得住 langsmith 自己的 @traceable run，但管不住 LangChain
    的 run。target 里只要跑了 LangChain Agent，后台线程就会拿着空 key 去上报，
    日志里刷一片 401。这里显式关掉。
    """
    from langchain_core.tracers.context import collect_runs
    from langsmith import run_helpers as rh

    with rh.tracing_context(enabled=False):
        if capture:
            with collect_runs() as cb:
                out = agent.invoke({"messages": messages})
            return out, list(cb.traced_runs)
        out = agent.invoke({"messages": messages})
        return out, []


def parse_output(out: Any) -> dict:
    """把 Agent 的原始返回解析成评测用的 dict。

    解析失败时给一个 _parse_error 标记，判定器里的 json_valid 会认出来。
    宁可让判定器判 0 分，也不要在这里抛错把整轮评测打断。
    """
    text = out["messages"][-1].content
    if not isinstance(text, str):
        text = str(text)
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"_parse_error": text[:200]}
    parsed["_raw"] = text
    return parsed


def ask(agent: Any, question: str):
    """按评测集的输入格式跑一次，返回 (解析后的 dict, runs)。"""
    out, runs = invoke_quiet(agent, [HumanMessage(question)])
    return parse_output(out), runs

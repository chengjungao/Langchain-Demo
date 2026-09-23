#!/usr/bin/env python
"""上下文治理三招实测：裁剪、删除、摘要。

这份脚本回答一个具体问题：会话越聊越长，token 账单怎么按住。

它不需要任何 API key，因为它把所有模型调用换成了一个只记账的假模型。
切换成真实模型只需要改 main() 里 build_model() 的返回值。

四个实验：
  幕 1  不做治理，看账单怎么涨
  幕 2  第一招：裁剪（before_model + RemoveMessage）
  幕 3  第二招：删除（after_model + RemoveMessage(id=...)）
  幕 4  第三招：摘要（SummarizationMiddleware）

口径说明（重要）：
  脚本统计的是**字符数**，不是真实 token 数，为的是不依赖分词器、谁跑都一样。
  所以摘要那幕里 trigger=600 指的是 600 个字符。换成官方默认的 token 计数，
  同一个 600 会小到一次都触发不了，脚本里专门跑了一遍对照给你看。

运行：
    python context_governance_demo.py
"""
from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    after_model,
    before_model,
    wrap_model_call,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, RemoveMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import Field

SUMMARY_MARK = "[[SUMMARIZE]]"
SECRET = "sk-live-9f3a2b7c"
ROUNDS = 20
LINE = "=" * 66


def char_counter(msgs) -> int:
    """字符计数，本脚本里的 token 替身。

    用它图的是可复现：不依赖任何模型的分词器，谁跑都是同一个数。
    代价是 trigger 的数字含义跟着变了，见 show_trigger_unit()。
    """
    return sum(len(str(getattr(m, "content", ""))) for m in msgs)


class LedgerModel(BaseChatModel):
    """只记账的假模型：记录每次调用看到的上下文规模。

    换成真实模型：把 build_model() 里的返回改成
        init_chat_model("deepseek:deepseek-chat")
    其余代码一行都不用动。
    """

    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "ledger-model"

    def bind_tools(self, tools, **kwargs):
        """BaseChatModel 在 1.x 里默认抛 NotImplementedError，必须自己实现。"""
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = "\n".join(str(getattr(m, "content", "")) for m in messages)
        is_summary = SUMMARY_MARK in text
        self.calls.append({
            "messages": len(messages),
            "chars": len(text),
            "kind": "summary" if is_summary else "chat",
        })
        if is_summary:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="用户在做客服 Agent 的上下文治理，已聊过退款流程、物流延迟与日志排查。"
            ))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="收到，我查一下。"))])


def build_model() -> LedgerModel:
    return LedgerModel()


def bill(model: LedgerModel, kind: str = "chat") -> int:
    """本轮实验里，模型为『正常对话』吃掉的输入字符总量。"""
    return sum(c["chars"] for c in model.calls if c["kind"] == kind)


def peak(model: LedgerModel, kind: str = "chat") -> int:
    hits = [c["messages"] for c in model.calls if c["kind"] == kind]
    return max(hits) if hits else 0


def user_text(round_no: int) -> str:
    return (
        f"第 {round_no} 轮：用户 U-{round_no:03d} 问订单 A{1000 + round_no} 的退款进度，"
        "顺便抱怨物流太慢，需要查订单、查退款流水、再解释一遍时效规则。"
    )


def preface(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def show_bill(model: LedgerModel, label: str) -> None:
    chats = [c for c in model.calls if c["kind"] == "chat"]
    print(f"  对话轮数：{len(chats)}")
    print(f"  模型输入累计：{bill(model)} 字符")
    print(f"  单轮峰值：{peak(model)} 条消息")
    summaries = [c for c in model.calls if c["kind"] == "summary"]
    if summaries:
        extra = sum(c["chars"] for c in summaries)
        print(f"  摘要调用：{len(summaries)} 次，额外成本 {extra} 字符")
    print(f"  [{label}]")


def run_plain() -> LedgerModel:
    model = build_model()
    graph = create_agent(model, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "plain"}}
    for i in range(1, ROUNDS + 1):
        graph.invoke({"messages": [("user", user_text(i))]}, cfg)
    return model


# ---------------------------------------------------------------- 第一招 裁剪

@before_model
def trim_history(state, runtime):
    """发给模型前，把消息列表重写成「首条 + 最近若干条」。

    返回 REMOVE_ALL_MESSAGES 加新列表，等于整体替换。
    只重写列表不叫治理：这里没有按 token 算，真实项目建议用
    trim_messages(..., token_counter=模型, max_tokens=窗口的六成) 来定预算。
    """
    messages = state["messages"]
    if len(messages) <= 6:
        return None
    kept = [messages[0], *messages[-4:]]
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}


def run_trim() -> LedgerModel:
    model = build_model()
    graph = create_agent(model, middleware=[trim_history], checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "trim"}}
    for i in range(1, ROUNDS + 1):
        graph.invoke({"messages": [("user", user_text(i))]}, cfg)
    return model


# ---------------------------------------------------------------- 第二招 删除

@after_model
def forget_on_request(state, runtime):
    """用户明确要求忘掉某段内容时，把那条消息标记为删除。

    注意两点：
    1. after_model 里 state 已经包含模型刚产出的回复，所以不能拿 messages[-1]
       当用户请求，要按类型和内容去找。
    2. 加存在性判断，否则每轮都会重复触发删除。
    """
    messages = state["messages"]
    target = next((m for m in messages
                   if getattr(m, "type", "") == "human" and SECRET in str(m.content)), None)
    if target is None:
        return None
    asked = any(getattr(m, "type", "") == "human" and "忘掉" in str(m.content) for m in messages)
    if not asked:
        return None
    return {"messages": [RemoveMessage(id=target.id)]}


def run_forget() -> tuple[LedgerModel, Any]:
    model = build_model()
    graph = create_agent(model, middleware=[forget_on_request], checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "forget"}}
    graph.invoke({"messages": [("user", f"这段是生产密钥 {SECRET}，帮我看看有没有泄漏风险")]}, cfg)
    graph.invoke({"messages": [("user", "刚才那段忘掉")]}, cfg)
    graph.invoke({"messages": [("user", "我们接着聊退款流程")]}, cfg)
    return model, graph


# ---------------------------------------------------------------- 第三招 摘要

def run_summarize(trigger: int = 600, real_tokens: bool = False) -> LedgerModel:
    """跑一遍摘要治理。

    trigger 的单位取决于 token_counter：
      默认（real_tokens=False）用 char_counter，trigger=600 指 600 个字符；
      real_tokens=True 用官方近似计数，trigger=600 才是 600 个 token。
    这两种口径下同一个 600 的效果差得很远，别混着看。
    """
    model = build_model()
    middleware = SummarizationMiddleware(
        model=model,
        trigger=("tokens", trigger),
        keep=("messages", 4),
        token_counter=(count_tokens_approximately if real_tokens else char_counter),
        summary_prompt=SUMMARY_MARK + "\n下面的对话请压缩成要点：\n{messages}",
    )
    graph = create_agent(model, middleware=[middleware], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"summary-{trigger}-{real_tokens}"}}
    for i in range(1, ROUNDS + 1):
        graph.invoke({"messages": [("user", user_text(i))]}, config)
    return model


def show_trigger_unit() -> None:
    """trigger 的单位陷阱：换计数口径，同一个 600 完全是两回事。

    上面那几行数字是在「字符口径」下跑出来的。换成官方默认的 token 计数，
    trigger=600 太小，一次都不会触发，账单跟不做治理一模一样。
    """
    print()
    print("  同一个 600，换成官方默认的 token 计数再跑一遍：")
    print(f"  {'trigger':<12}{'累计输入字符':>14}{'单轮峰值':>10}{'摘要调用':>10}")
    for trig in (600, 400, 300, 200):
        model = run_summarize(trigger=trig, real_tokens=True)
        n_sum = sum(1 for c in model.calls if c["kind"] == "summary")
        print(f"  {trig:<12}{bill(model):>14}{peak(model):>10}{n_sum:>10}")
    print("  600 token 时摘要一次都没触发，账单和无治理完全相同。")
    print("  结论：trigger 的数字要和 token_counter 的口径一起看，单独抄一个数字没意义。")


def show_trim_boundary() -> None:
    """官方 trim_messages 的边界行为，含一个必须知道的陷阱。"""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    messages = [
        SystemMessage("你是客服 Agent"),
        HumanMessage("订单 A100 退款到哪一步了"),
        AIMessage("", tool_calls=[{"name": "query", "args": {"no": "A100"}, "id": "c1"}]),
        ToolMessage("A100 已退款", tool_call_id="c1"),
        AIMessage("A100 昨天已完成退款"),
        HumanMessage("那 B200 呢"),
        AIMessage("", tool_calls=[{"name": "query", "args": {"no": "B200"}, "id": "c2"}]),
        ToolMessage("B200 待审核", tool_call_id="c2"),
        AIMessage("B200 还在审核中"),
    ]

    def brief(msgs) -> str:
        names = {"SystemMessage": "系统", "HumanMessage": "用户", "AIMessage": "AI", "ToolMessage": "工具"}
        return " -> ".join(names.get(type(m).__name__, "?") for m in msgs)

    tight = trim_messages(messages, max_tokens=4, token_counter=len, strategy="last",
                          include_system=True, start_on="human")
    ok = trim_messages(messages, max_tokens=8, token_counter=len, strategy="last",
                       include_system=True, start_on="human")
    print(f"  预算给到刚好 4 条：{brief(tight)}   <- 只剩系统提示，用户问题整段丢了")
    print(f"  预算给到 8 条：  {brief(ok)}")
    print("  结论：预算别贴着下限设，要给系统提示和最近一轮留出余量。")


@wrap_model_call
def trim_transient(request, handler):
    """瞬态裁剪：改写本次请求，状态一个字节都不动。

    模型看到的内容和 before_model 版一样少，但完整历史留在状态里，
    后续要做审计、回溯、长期记忆抽取都还有原料。
    """
    messages = request.messages
    if len(messages) > 6:
        request = request.override(messages=[messages[0], *messages[-4:]])
    return handler(request)


def compare_trim_modes(rounds: int = 6) -> None:
    """两种裁剪的差别：都省钱，区别在状态动不动。"""
    def run(middleware) -> tuple[list[int], int]:
        model = build_model()
        graph = create_agent(model, middleware=middleware, checkpointer=InMemorySaver())
        cfg = {"configurable": {"thread_id": f"cmp-{id(middleware)}"}}
        for i in range(1, rounds + 1):
            graph.invoke({"messages": [("user", f"第 {i} 轮：查一下订单 A{1000 + i} 的退款进度")]}, cfg)
        state = graph.get_state(cfg).values["messages"]
        return [c["messages"] for c in model.calls], len(state)

    seen_p, size_p = run([trim_history])
    seen_t, size_t = run([trim_transient])
    print(f"  持久裁剪 before_model   逐轮模型看到 {seen_p}   最终状态: {size_p} 条")
    print(f"  瞬态裁剪 wrap_model_call 逐轮模型看到 {seen_t}   最终状态: {size_t} 条")
    print("  注意前几轮还没触发裁剪，条数是慢慢涨到 5 条的。")
    print("  结论：省钱效果一样，区别在状态留不留。要审计和回溯就用瞬态那一版。")


def main() -> None:
    preface("幕 1：不做任何治理，20 轮对话的账单")
    plain = run_plain()
    show_bill(plain, "无治理")

    preface("幕 2：第一招 裁剪（before_model 里重写消息列表）")
    trimmed = run_trim()
    show_bill(trimmed, "裁剪后")
    print(f"  账单变化：{bill(plain)} -> {bill(trimmed)} 字符"
          f"，降到 {bill(trimmed) / bill(plain):.0%}")
    print(f"  单轮峰值：{peak(plain)} -> {peak(trimmed)} 条消息")

    preface("幕 2 附：官方 trim_messages 的边界行为")
    show_trim_boundary()

    print()
    print("  两种裁剪的差别（before_model 与 wrap_model_call）：")
    compare_trim_modes()

    preface("幕 3：第二招 删除（标记删除，不是从列表里抠掉）")
    forget_model, forget_graph = run_forget()
    current = forget_graph.get_state({"configurable": {"thread_id": "forget"}}).values["messages"]
    print("  当前状态里的消息：")
    for m in current:
        print(f"    - {type(m).__name__:<14} {str(m.content)[:34]}")
    alive = any(SECRET in str(m.content) for m in current)
    print(f"  当前状态里还能找到密钥：{alive}")
    history = list(forget_graph.get_state_history({"configurable": {"thread_id": "forget"}}))
    hidden = sum(1 for s in history
                 if any(SECRET in str(getattr(m, 'content', '')) for m in s.values.get('messages', [])))
    print(f"  存档 checkpoint 共 {len(history)} 个，其中 {hidden} 个仍含密钥")
    print("  结论：删除让消息从『当前状态』消失，历史存档里它还在。")
    print("       用户要求删除敏感信息时，存档清理要单独考虑。")

    preface("幕 4：第三招 摘要（SummarizationMiddleware）")
    summarized = run_summarize(trigger=600)
    show_bill(summarized, "摘要后")
    print("  （本次口径：token_counter 按字符计数，所以 600 指 600 个字符）")
    print(f"  账单变化：{bill(plain)} -> {bill(summarized)} 字符"
          f"，降到 {bill(summarized) / bill(plain):.0%}")

    show_trigger_unit()

    print()
    print("  触发阈值是笔取舍账（同一段对话，只改 trigger，均为字符口径）：")
    print(f"  {'trigger':<10}{'累计输入字符':>14}{'单轮峰值':>10}{'摘要调用':>10}")
    for trig in (600, 400):
        model = run_summarize(trigger=trig)
        n_sum = sum(1 for c in model.calls if c["kind"] == "summary")
        print(f"  {trig:<10}{bill(model):>14}{peak(model):>10}{n_sum:>10}")
    print("  阈值调紧省得更多，但摘要本身也要调模型，次数一多成本会还回来。")

    preface("四组对比")
    print(f"  {'方案':<8}{'累计输入字符':>14}{'单轮峰值条数':>14}")
    for name, model in (("无治理", plain), ("裁剪", trimmed), ("摘要", summarized)):
        print(f"  {name:<8}{bill(model):>14}{peak(model):>14}")


if __name__ == "__main__":
    main()

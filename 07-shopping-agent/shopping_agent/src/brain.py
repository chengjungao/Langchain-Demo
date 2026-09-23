# -*- coding: utf-8 -*-
"""模型抽象层：把「谁来做判断」这件事收成一个接口。

整条 Agent 链路里有四类判断要做：路由、选工具、抽记忆、写摘要。
它们都依赖模型，但依赖的方式不一样 —— 路由要结构化输出，选工具要 function
calling，抽记忆要能处理长文本，写摘要只要能写。

如果每个节点各自去 `ChatOpenAI(...)`，会有两个后果：一是模型换不了，
二是没有模型时整条链路直接断掉。所以这里把四种能力收成一个 `Brain` 接口，
再给两个实现：

- `LLMBrain`  —— 接任意 OpenAI 兼容端点，本地服务或云端都行
- `RuleBrain` —— 关键词与启发式规则，不需要任何模型

第二个不是占位符。它让这个工程在**没有配置模型时也能完整跑通**，
也提供了一个对照：同一份数据、同一张图，换成规则判断会退化在哪一步。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, ToolMessage

from .config import ModelConfig
from .models import Route, looks_like_purchase


# ── 统一的数据结构 ──────────────────────────────────────────

@dataclass
class Decision:
    """Agent 循环里的一步。要么调工具，要么给答复，二选一。"""

    tool: str | None = None
    args: dict = field(default_factory=dict)
    reply: str = ""
    thought: str = ""
    # 模型为这次调用给出的编号。下一轮要把结果当成协议里的「工具返回」
    # 回灌，就得带上它，否则模型收到的是一条来历不明的用户消息。
    call_id: str = ""


@dataclass
class BrainCtx:
    """一次判断需要的全部输入。

    长成这个样子是因为两个实现要的东西不一样：模型需要完整消息，
    规则只需要用户这句话和已经拿到的工具结果。
    """

    role: str                          # advisor / order / service
    user_text: str
    system: str
    history: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)   # 已执行的工具及结果
    tool_names: list[str] = field(default_factory=list)
    hints: dict = field(default_factory=dict)                # 路由阶段抽到的槽位


# ── 记忆抽取的输出模型 ──────────────────────────────────────

class ProfileItem(BaseModel):
    """一条用户画像。"""

    kind: str = Field(description="偏好类型：品牌/预算/禁忌/尺码/场景/其他")
    key: str = Field(description="稳定的键名，如「偏好品牌」「预算上限」，同键会覆盖")
    value: str = Field(description="值，写成一句短语，如「漫步者」")
    polarity: Literal["like", "dislike", "neutral"] = "neutral"
    quote: str = Field(default="", description="用户原话片段，用于回溯")
    confidence: float = 0.7


class EpisodeItem(BaseModel):
    """一条事件记忆。"""

    kind: str = Field(description="事件类型：咨询/下单/退换/投诉/浏览")
    summary: str = Field(description="一句话记下发生了什么")
    sku_id: str = ""
    order_id: str = ""


class MemoryDraft(BaseModel):
    profile: list[ProfileItem] = Field(default_factory=list)
    episodes: list[EpisodeItem] = Field(default_factory=list)


# ── 接口 ────────────────────────────────────────────────────

class Brain:
    kind = "base"
    label = "未配置"

    def route(self, text: str, history: list[dict]) -> Route:
        raise NotImplementedError

    def decide(self, ctx: BrainCtx, tools: list[Any]) -> Decision:
        raise NotImplementedError

    def extract(self, messages: list[dict]) -> MemoryDraft:
        raise NotImplementedError

    def summarize(self, messages: list[dict], limit: int = 200) -> str:
        raise NotImplementedError


# ── 实现一：真实模型 ────────────────────────────────────────

_ROUTE_SYSTEM = """你在一个电商购物助手的意图识别环节工作。读用户这句话，判断它属于哪条支线。

支线定义：
- advisor：找商品、要推荐、比价、问参数。用户想买点什么或者想了解商品。
- order：查订单、查物流、问发货、**要下单**。用户已经在关心某笔具体交易。
- service：问平台规则、退换货政策、发票、保修、运费责任。用户关心的是规则本身。
- chat：寒暄、与购物无关、说不清楚。

分界只看一件事：**这句话要的是「了解」还是「办掉」。**
「这款耳机怎么样」「有没有一千以内的」是要了解，走 advisor；
「帮我下单」「就这个，买了」「订单到哪了」是要办掉，走 order ——
哪怕句子里带了商品名和规格，也不要因为「提到了商品」就判成 advisor。

同时把能从这句话里直接读到的槽位填进 category / budget_max / sku_id / order_id。
读不到就留空，不要猜。reason 用一句话说明判断依据。"""

_MEMORY_SYSTEM = """你在一个电商购物助手的记忆归档环节工作。

从这段对话里抽出两类东西。

**一、profile：用户表现出的稳定偏好。**

只记用户自己说的、或者他明确认同的。**不要从助手的话里推断** ——
助手说「您对夹头敏感」，如果用户没说过，那就不是事实。

kind 必须从这几个里选一个，不要自己造：
  品牌 / 预算 / 尺码 / 身体条件 / 使用场景 / 禁忌 / 习惯 / 其他

value 写成一句短语，尽量带上具体信息。比如：
  kind=身体条件  value=对夹头敏感，戴眼镜
  kind=品牌      value=云雀（不要）
  kind=预算      value=耳机 1000 元以内

同一个类别下的同类信息合并成一条，不要拆成多条。
比如「对夹头敏感」和「戴眼镜」都是身体条件，写一条就够了。

**二、episodes：这次对话里发生过的具体事情。**

只记**用户这一侧**的动作和诉求：问了什么、买了什么、要退什么、投诉什么。
不要记助手做了什么 —— 「助手推荐了某款」不是用户的事件。

写成一句话，带上是哪个商品或哪笔订单。没有就返回空列表。

三类都不要为了凑数编内容。信息不足时宁可少记，也不要记错的。"""


class LLMBrain(Brain):
    kind = "live"

    def __init__(self, cfg: ModelConfig) -> None:
        from langchain_openai import ChatOpenAI
        self.cfg = cfg
        self.label = f"{cfg.model} @ {cfg.base_url}"
        # 用量回收口。由服务层接上，用来算这次对话真的花了多少。
        self.sink = None
        # 这个端点的流式到底能不能用。第一次发现拿不到内容就记下来，
        # 后面不再重复试 —— 每次重试都是一次完整的模型调用。
        self._stream_ok: bool | None = None

        kwargs: dict = {}
        if cfg.disable_thinking and _is_local(cfg.base_url):
            # 本地推理服务上的推理型模型（Qwen3 一类）默认会先输出一大段思考。
            # 思考也占生成预算，Agent 里那些「分个类、挑个工具」的判断用不上它，
            # 开着反而会出现「跑完了但一个字都没有」。
            kwargs["extra_body"] = {"reasoning_effort": "none"}

        self._llm = ChatOpenAI(
            model=cfg.model,
            base_url=cfg.base_url or None,
            api_key=cfg.api_key or "not-needed",
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            max_retries=1,
            **kwargs,
        )

    def _report(self, resp) -> None:
        """把服务端返回的用量记一笔。

        自己数 token 只能估，服务端返回的才是准的。OpenAI 兼容端点的
        usage 字段规范一致，本地服务也遵守，所以这里统一读它。
        """
        u = getattr(resp, "usage_metadata", None)
        if not u or not callable(self.sink):
            return
        try:
            detail = u.get("output_token_details") or {}
            self.sink(int(u.get("input_tokens") or 0),
                      int(u.get("output_tokens") or 0),
                      int(detail.get("reasoning") or 0))
        except Exception:
            pass

    # ── 路由 ────────────────────────────────────────────
    def route(self, text: str, history: list[dict]) -> Route:
        msgs = [("system", _ROUTE_SYSTEM)]
        for h in history[-4:]:
            msgs.append((h["role"], h["content"]))
        msgs.append(("user", text))
        try:
            r = self._structured(Route, msgs, _ROUTE_FALLBACK_KEYS)
        except Exception:
            return _route_by_rule(text)

        # 涉及钱的话不让模型自由判。实测里它会把「帮我下单某款显示器」
        # 判成「了解这类商品」，于是走进导购线 —— 那条线没有下单工具，
        # 用户的要求就落空了。这里做一次确定性改判。
        if looks_like_purchase(text) and r.route == "advisor":
            r.route = "order"
            r.reason = (r.reason + "　（用户说的是要办掉，不是要了解，"
                        "已按成交意图改判到订单线）").strip()
        return r

    # ── 选工具 ──────────────────────────────────────────
    def _build_messages(self, ctx: BrainCtx) -> list:
        """把「这一轮发生了什么」拼成一份完整的消息序列。

        工具结果按协议回灌：先是助手那条 tool_calls，紧接着 role=tool 的
        结果消息。这不是格式洁癖 —— 换个写法（把结果写成一句用户的
        「工具返回了这些」），模型会当成用户又说了句话，于是判断对话
        该收尾了，直接给答复。明明还有一步没做，它却宣布已经做完。

        序列末尾再补一条显式提醒，把「用户原始诉求」和「已经办到哪一步」
        摆在一起，让模型自己对齐还缺什么这一步。
        """
        msgs: list = [("system", ctx.system)]
        for h in ctx.history[-6:]:
            msgs.append((h["role"], h["content"]))
        msgs.append(("user", ctx.user_text))

        for i, o in enumerate(ctx.observations):
            cid = o.get("call_id") or f"call_{i}"
            msgs.append(AIMessage(
                content="",
                tool_calls=[{"name": o.get("tool") or "",
                             "args": o.get("args") or {},
                             "id": cid, "type": "tool_call"}]))
            msgs.append(ToolMessage(content=str(o.get("result") or ""),
                                    tool_call_id=cid))

        if ctx.observations:
            used = "；".join(f"{o.get('tool')}（{_brief(o.get('args'))}）"
                             for o in ctx.observations)
            msgs.append(("user",
                         "这一轮你已经调用过：" + used +
                         "\n用户原始诉求：" + ctx.user_text +
                         "\n照着原始诉求对着看：里面每个动作都有工具成功返回过了吗？"
                         "还缺的，现在就继续调用工具把它办掉。"
                         "对齐了就直接给出最终答复，同一组参数不要重复调用。"))
        return msgs

    def decide(self, ctx: BrainCtx, tools: list[Any], on_token=None) -> Decision:
        """Agent 循环里的一步：接着动手，还是收尾回答。

        这里有一条容易踩的坑：**不能因为「已经拿到过工具结果」就把工具收走**。
        真实任务常常是多步的 —— 先认出是哪个商品、再算价、最后下单，
        三步要连着走完。如果第一轮之后就不给工具，链路只能走一步，
        剩下的步骤模型没有工具可用，它会**编**一个结果继续往下说。
        「订单号 ORD20260917001，已下单成功」这种话就是这么来的。

        还有一条同样重要：**工具结果要按协议回灌**。用 assistant 的
        tool_calls + role=tool 的消息对，而不是把它写成一句「用户说：工具
        返回了这些」。后者看着差不多，但模型会把它当成用户又说了句话，
        于是认为「对话可以结束了」，然后直接收尾 —— 明明还有一步没做。
        """
        msgs = self._build_messages(ctx)
        bound = self._llm.bind_tools(tools) if tools else self._llm

        # 已经有工具结果时，这一步多半是在组织答复，走流式让用户边看边等。
        # 但它也可能还要再调一次工具，所以工具照样绑着 —— 两种结果都要接得住。
        if ctx.observations and on_token:
            try:
                d = self._stream_decide(bound, msgs, on_token)
                if d is not None:
                    return d
            except Exception:
                pass

        return self._invoke_decide(bound, msgs, on_token)

    def _invoke_decide(self, bound, msgs: list, on_token) -> Decision:
        """一次非流式的判断。流式不可用时也落到这里。

        拿到工具调用就返回工具，否则把答复交出去。分段吐字保留着，
        是因为有些本地推理端点流式确实不可用，但用户还是希望字一个个
        出现，而不是等十几秒被一整段砸中。
        """
        try:
            resp = bound.invoke(msgs)
            self._report(resp)
        except Exception as e:
            return Decision(reply=f"（模型调用失败：{e}）")

        calls = getattr(resp, "tool_calls", None) or []
        if calls:
            c = calls[0]
            return Decision(tool=c.get("name"), args=c.get("args") or {},
                            thought=_text_of(resp), call_id=c.get("id") or "")
        text = _text_of(resp) or _empty_reply_hint(resp, self.cfg)
        if on_token and text:
            _emit_paced(text, on_token)
        return Decision(reply=text)

    def _stream_decide(self, bound, msgs: list, on_token) -> Decision | None:
        """边流边判断这一步是「动手」还是「收尾」。

        流式返回里两者的形态不同：工具调用走 tool_call_chunks，
        答复走 content。所以可以在同一条流上把两件事分开接住，
        不必事先猜是哪一种。

        返回 None 表示这条流拿不到可用信息（端点不支持流式、或者内容
        全进了推理字段），交给调用方走非流式路径。这个判断只做一次，
        之后记住该端点不适合流式，免得每轮白跑一次失败请求。
        """
        if self._stream_ok is False:
            return None

        buf: list[str] = []
        chunks: list[dict] = []
        last = None
        try:
            for chunk in bound.stream(msgs, stream_usage=True):
                last = chunk
                t = _text_of(chunk)
                if t:
                    buf.append(t)
                    on_token(t)
                chunks.extend(getattr(chunk, "tool_call_chunks", None) or [])
        except Exception:
            self._stream_ok = False
            if buf:                     # 已经吐出去一半了，就不要再调一次
                return Decision(reply="".join(buf))
            return None

        if last is not None:
            self._report(last)

        if chunks:
            self._stream_ok = True
            merged = _merge_tool_chunks(chunks)
            if merged:
                return Decision(tool=merged.get("name"),
                                args=merged.get("args") or {},
                                call_id=merged.get("id") or "")
            return None

        text = "".join(buf)
        if text:
            self._stream_ok = True
            return Decision(reply=text)

        # 流式走完了但一个字都没有：多半是内容全进了推理字段。
        self._stream_ok = False
        return None

    # ── 记忆抽取 ────────────────────────────────────────
    def extract(self, messages: list[dict]) -> MemoryDraft:
        convo = "\n".join(f"{m['role']}：{m['content']}" for m in messages[-8:])
        msgs = [("system", _MEMORY_SYSTEM), ("user", convo or "（无内容）")]
        try:
            return self._structured(MemoryDraft, msgs, _MEMORY_FALLBACK_KEYS)
        except Exception:
            return MemoryDraft()

    def summarize(self, messages: list[dict], limit: int = 200) -> str:
        convo = "\n".join(f"{m['role']}：{m['content']}" for m in messages)
        try:
            r = self._llm.invoke([
                ("system", "把下面的对话压缩成一段不超过 80 字的中文摘要，只保留"
                           "用户的目标、已确认的结论、明确的偏好。不要客套话。"),
                ("user", convo[:4000]),
            ])
            return _text_of(r)[:limit]
        except Exception:
            return ""

    # ── 通用：结构化输出 + 文本兜底 ──────────────────────
    def _structured(self, model_cls: type[BaseModel], msgs: list, keys: list[str]):
        """先试原生结构化输出，不行再退到「要一段 JSON 然后自己解析」。

        小模型和部分推理框架对 function calling 支持不完整，这一步兜底
        是真实项目里常见的做法，不是多此一举。
        """
        try:
            return self._llm.with_structured_output(model_cls).invoke(msgs)
        except Exception:
            pass
        try:
            fields = ", ".join(f'"{k}"' for k in keys)
            probe = list(msgs) + [
                ("user", f"只输出一个 JSON 对象，字段包含 {fields}，不要任何解释文字。")]
            raw = _text_of(self._llm.invoke(probe))
            data = _pluck_json(raw)
            if data:
                return model_cls(**{k: v for k, v in data.items()
                                    if k in model_cls.model_fields})
        except Exception:
            pass
        raise RuntimeError("结构化输出失败")


_ROUTE_FALLBACK_KEYS = ["route", "reason", "category", "budget_max", "sku_id", "order_id"]
_MEMORY_FALLBACK_KEYS = ["profile", "episodes"]


def _text_of(resp) -> str:
    c = getattr(resp, "content", resp)
    if isinstance(c, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return str(c or "")


def _merge_tool_chunks(chunks: list[dict]) -> dict | None:
    """把流式返回里碎片化的工具调用拼回完整的一次调用。

    流式下工具调用不是一个完整对象过来的：名字先到，参数是一段段
    JSON 字符串随后陆续到达。所以先按 index 归拢，再把参数拼起来解析。
    一次只取第一个调用 —— 这个循环每步只做一件事，多给的忽略掉。
    """
    slots: dict[int, dict] = {}
    for c in chunks:
        if not isinstance(c, dict):
            continue
        i = int(c.get("index") or 0)
        s = slots.setdefault(i, {"name": "", "args": "", "id": ""})
        if c.get("name"):
            s["name"] = c["name"]
        if c.get("args"):
            s["args"] += c["args"]
        if c.get("id"):
            s["id"] = c["id"]
    if not slots:
        return None
    s = slots[min(slots)]
    if not s["name"]:
        return None
    try:
        args = json.loads(s["args"]) if s["args"].strip() else {}
    except json.JSONDecodeError:
        args = {}
    return {"name": s["name"], "args": args if isinstance(args, dict) else {},
            "id": s["id"]}


def _brief(args) -> str:
    """把一次工具调用的参数压成一行，用来提醒模型「这个已经调过了」。"""
    if not args:
        return "无参数"
    try:
        return json.dumps(args, ensure_ascii=False)[:80]
    except (TypeError, ValueError):
        return str(args)[:80]


def _emit_paced(text: str, on_token) -> None:
    """整段文本切块吐出去，让它有点节奏，别一瞬间全出来。"""
    import time as _time
    step = max(6, len(text) // 36)
    for i in range(0, len(text), step):
        on_token(text[i:i + step])
        _time.sleep(0.018)


def _pluck_json(raw: str) -> dict | None:
    """从一段可能裹着散文的回复里挖出第一个 JSON 对象。"""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def _is_local(base_url: str) -> bool:
    """是不是本机或内网的服务。

    只对这个范围发关思考的请求参数。云端端点大多不认识这个字段，
    发过去轻则被忽略重则直接报错，没必要冒这个险。
    """
    try:
        host = (urlparse(base_url or "").hostname or "").lower()
    except ValueError:
        return False
    if host in ("127.0.0.1", "localhost", "0.0.0.0", "::1", "host.docker.internal"):
        return True
    return host.startswith(("192.168.", "10.", "172.16.", "172.17.", "172.18.",
                            "172.19.", "172.2", "172.30.", "172.31."))


def _empty_reply_hint(resp, cfg: ModelConfig) -> str:
    """模型一个字没吐时，告诉用户真实原因，而不是留一片空白。

    这种情况在本地跑推理型模型时很常见：生成预算被思考链吃光了。
    与其显示空白让人以为程序坏了，不如把观察到的数字摆出来。
    """
    u = getattr(resp, "usage_metadata", None) or {}
    detail = u.get("output_token_details") or {}
    rt = int(detail.get("reasoning") or 0)
    out = int(u.get("output_tokens") or 0)
    if rt and rt >= out * 0.9:
        return (f"（模型这次没有产出回答：{out} 个输出 token 里 {rt} 个是它的推理内容。\n"
                f"去设置里把「单次最大生成」调大一些，或者确认「关闭思考模式」是开着的。）")
    return "（模型返回了空内容。可能是这个端点不兼容当前参数，或者回复被内容策略拦下了。）"


# ── 实现二：规则 ────────────────────────────────────────────

_ADVISOR_WORDS = ("推荐", "买", "想要", "想入", "选", "哪个", "哪个好", "对比", "比价",
                  "预算", "便宜", "贵", "性价比", "有没有", "看看", "找", "求")
_ORDER_WORDS = ("订单", "物流", "快递", "到哪", "什么时候到", "发货", "下单", "买了",
                "签收", "运单", "单号", "催")
_SERVICE_WORDS = ("退货", "换货", "退款", "保修", "发票", "运费", "政策", "规则",
                  "能不能退", "七天", "无理由", "售后", "价保", "保价", "满减", "优惠券",
                  # 口语说法。用户很少完整说「退货」，多的是「拆封了还能退吗」
                  # 「想退了」这类，不补进来就会掉进闲聊兜底。
                  "能退", "想退", "退了", "拆封", "换个", "维修")
_CATEGORIES = ("耳机", "键盘", "显示器", "屏幕", "机械键盘", "降噪")


def _route_by_rule(text: str) -> Route:
    """规则路由。分数制而不是「先匹配先返回」，因为一句话里常混着多类词。"""
    score = {"advisor": 0, "order": 0, "service": 0}
    for w in _ADVISOR_WORDS:
        score["advisor"] += text.count(w)
    for w in _ORDER_WORDS:
        score["order"] += text.count(w)
    for w in _SERVICE_WORDS:
        score["service"] += text.count(w)
    if re.search(r"[A-Z]{2}\d{8,}", text):          # 像订单号的东西
        score["order"] += 3
    best = max(score, key=lambda k: score[k])
    route = best if score[best] > 0 else "chat"

    cat = next((c for c in _CATEGORIES if c in text), None)
    if cat in ("屏幕",):
        cat = "显示器"
    if cat in ("机械键盘",):
        cat = "键盘"

    budget = None
    m = re.search(r"(\d{3,5})\s*(?:元|块)?\s*(?:以内|以下|左右|上下)", text)
    if m:
        budget = float(m.group(1))
    else:
        m = re.search(r"预算\s*(\d{3,5})", text)
        if m:
            budget = float(m.group(1))

    sku = None
    m = re.search(r"SKU-[A-Z]\d{4}", text, re.I)
    if m:
        sku = m.group(0).upper()
    order = None
    m = re.search(r"[A-Z]{2}\d{8,}", text)
    if m:
        order = m.group(0)

    return Route(route=route, reason=f"规则命中：{score}", category=cat,
                 budget_max=budget, sku_id=sku, order_id=order)


class RuleBrain(Brain):
    """不需要模型的判断层。

    它的价值有两个：让工程在零配置下完整可跑；以及在正文里做一个诚实的对照 ——
    把模型换成规则，链路能走完，但槽位抽取和自然语言组织会明显变差。
    """

    kind = "demo"

    def __init__(self) -> None:
        self.label = "内置规则（未配置模型）"

    def route(self, text: str, history: list[dict]) -> Route:
        return _route_by_rule(text)

    def decide(self, ctx: BrainCtx, tools: list[Any], on_token=None) -> Decision:
        if ctx.observations:
            text = _compose_reply(ctx)
            if on_token:
                on_token(text)
            return Decision(reply=text)
        tool = _pick_tool(ctx)
        if tool is None:
            text = _compose_reply(ctx)
            if on_token:
                on_token(text)
            return Decision(reply=text)
        return Decision(tool=tool[0], args=tool[1], thought="规则选择")

    def extract(self, messages: list[dict]) -> MemoryDraft:
        return _extract_by_rule(messages)

    def summarize(self, messages: list[dict], limit: int = 200) -> str:
        parts = [m["content"] for m in messages if m["role"] == "user"]
        joined = "；".join(p.strip() for p in parts if p.strip())
        return (joined[:limit] + "…") if len(joined) > limit else joined


# ── 规则实现的内脏 ──────────────────────────────────────────

def _pick_tool(ctx: BrainCtx) -> tuple[str, dict] | None:
    """按支线决定第一个该调的工具。

    真实项目里这里是模型的事，规则实现只覆盖最常见的那条路径 ——
    这本身就是「规则能走多远」的证据。
    """
    t = ctx.user_text
    names = set(ctx.tool_names)

    if ctx.role == "advisor":
        if "search_products" in names:
            return "search_products", {"query": t}

    if ctx.role == "order":
        m = re.search(r"[A-Z]{2}\d{8,}", t)
        if m:
            if "get_logistics" in names and any(w in t for w in ("物流", "快递", "到哪", "运单")):
                return "get_logistics", {"order_id": m.group(0)}
            if "get_order" in names:
                return "get_order", {"order_id": m.group(0)}
        # 用户是要买东西，不是要查旧单。先去认货 —— 商品名换成编码，
        # 后面的算价与下单才有依据。不加这一条的话，「帮我下单 X」
        # 会被当成「查我的订单」，给出一串历史订单，答非所问。
        if looks_like_purchase(t) and "search_products" in names:
            return "search_products", {"query": t}
        if "list_my_orders" in names:
            return "list_my_orders", {}

    if ctx.role == "service":
        if "search_policies" in names:
            return "search_policies", {"query": t}

    return None


def _compose_reply(ctx: BrainCtx) -> str:
    """规则版答复：把工具结果按固定格式摆出来。

    这一段最能说明模型的不可替代性 —— 数据是同一份，读起来就是不一样。
    """
    if not ctx.observations:
        role_name = {"advisor": "导购", "order": "订单", "service": "客服"}.get(ctx.role, "助手")
        return (f"我是{role_name}助手。当前运行在**规则模式**（还没有配置模型），"
                f"我只能按固定规则处理常见问题。\n\n"
                f"你可以在右上角「设置」里填上模型地址和密钥，我会换成真实模型来回答。")

    blocks = [f"{o['result']}" for o in ctx.observations]
    head = "按现有数据查到这些："
    return head + "\n\n" + "\n\n".join(blocks)


_PROFILE_PATTERNS = [
    (r"我(?:比较)?(?:喜欢|偏好|爱用|常用)([\u4e00-\u9fffA-Za-z0-9]{2,12})", "偏好", "like"),
    (r"我(?:不|讨厌|别给我|不要)(?:喜欢|推荐|要)?([\u4e00-\u9fffA-Za-z0-9]{2,12})", "禁忌", "dislike"),
    (r"预算\s*(?:大概|差不多)?\s*(\d{3,5})\s*(?:元|块)?(?:以内|以下|左右)?", "预算", "neutral"),
    (r"我(?:在|住在|人在)([\u4e00-\u9fff]{2,8})", "城市", "neutral"),
]
_EPISODE_WORDS = {
    "退货": "退换", "换货": "退换", "退款": "退换",
    "投诉": "投诉",
    "发票": "咨询", "保修": "咨询", "价保": "咨询",
    "物流": "咨询", "订单": "咨询", "快递": "咨询", "发货": "咨询",
    "买了": "下单", "买过": "下单", "下单": "下单", "入手": "下单",
}


def _extract_by_rule(messages: list[dict]) -> MemoryDraft:
    """规则版记忆抽取。能抓显式表达，抓不到隐含偏好 —— 这就是差距。"""
    user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
    profile: list[ProfileItem] = []
    for pat, kind, pol in _PROFILE_PATTERNS:
        for m in re.finditer(pat, user_text):
            val = m.group(1)
            if kind == "预算" and not val.isdigit():
                continue
            profile.append(ProfileItem(kind=kind, key=kind, value=val,
                                       polarity=pol, quote=m.group(0),
                                       confidence=0.6))

    episodes: list[EpisodeItem] = []
    for word, kind in _EPISODE_WORDS.items():
        if word in user_text:
            m = re.search(r"[A-Z]{2}\d{8,}", user_text)
            episodes.append(EpisodeItem(kind=kind,
                                        summary=f"提到{word}：{user_text[:40]}",
                                        order_id=m.group(0) if m else ""))
            break
    return MemoryDraft(profile=profile, episodes=episodes)


# ── 工厂 ────────────────────────────────────────────────────

def make_brain(cfg: ModelConfig) -> Brain:
    """按配置挑实现。配置不全就退回规则，而不是抛异常 —— 工程要能跑起来。"""
    if cfg.is_ready():
        try:
            return LLMBrain(cfg)
        except Exception:
            return RuleBrain()
    return RuleBrain()


def probe(cfg: ModelConfig) -> dict:
    """测试连通性。给「设置」弹窗的「测试连接」按钮用。

    先拉模型列表（便宜），再发一条最短的消息（验证生成能力）。
    只报结论和耗时，不回显密钥。
    """
    import time
    import urllib.error
    import urllib.request

    base = (cfg.base_url or "").rstrip("/")
    out: dict = {"ok": False, "stage": "init", "message": "", "models": [],
                 "latency_ms": None, "sample": ""}
    if not base:
        out["message"] = "还没填地址"
        return out

    # 第一步：列模型
    try:
        req = urllib.request.Request(
            base + "/models",
            headers={"Authorization": f"Bearer {cfg.api_key or 'not-needed'}"})
        with urllib.request.urlopen(req, timeout=min(cfg.timeout, 20)) as r:
            data = json.loads(r.read().decode("utf-8"))
        out["models"] = [m.get("id") for m in data.get("data", []) if m.get("id")]
        out["stage"] = "models"
    except urllib.error.HTTPError as e:
        out["stage"] = "models"
        out["message"] = f"地址能连上，但列模型返回 {e.code}。密钥可能不对。"
        return out
    except Exception as e:
        out["stage"] = "models"
        out["message"] = f"连不上服务：{type(e).__name__}。检查地址和端口，以及服务是否在运行。"
        return out

    if not out["models"]:
        out["message"] = "服务连上了，但没有已加载的模型。先在本地服务里加载一个。"
        return out

    model = cfg.model or out["models"][0]
    out["resolved_model"] = model

    # 第二步：发一条最短的消息
    payload = {"model": model, "max_tokens": 16, "temperature": 0,
               "messages": [{"role": "user", "content": "回复两个字：收到"}]}
    try:
        t0 = time.time()
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {cfg.api_key or 'not-needed'}"})
        with urllib.request.urlopen(req, timeout=cfg.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        out["latency_ms"] = int((time.time() - t0) * 1000)
        out["sample"] = (data.get("choices", [{}])[0]
                         .get("message", {}).get("content", "") or "").strip()[:40]
        out["ok"] = True
        out["message"] = f"通了。{model} 首次响应 {out['latency_ms']} 毫秒。"
    except Exception as e:
        out["stage"] = "chat"
        out["message"] = f"模型列表能拉到，但生成失败：{type(e).__name__}: {e}"
    return out

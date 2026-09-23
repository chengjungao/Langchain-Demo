# -*- coding: utf-8 -*-
"""Agent 循环：三条支线共用的一套装配。

循环本身很简单 —— 想一步、做一步、再看结果，直到不需要动手为止。
真正值得看的是三处「接缝」：

1. **轮数上限不写在提示词里**。提示词说「不要反复调用工具」是建议，
   代码里的计数才是约束。模型判断错的时候，图和人都还能收场。
2. **写操作在执行前拦一道**。金额超过阈值时抛中断，把决定权交回给人。
   这不是「问一句」的礼貌，是流程上的强制点。
3. **工具的异常不抛给模型**。工具报错时返回一句人话，让模型有机会
   换个说法或者换个工具，而不是让整轮对话崩掉。
"""
from __future__ import annotations

import operator
import re
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..brain import BrainCtx
from ..guard import GuardError
from ..models import looks_like_purchase
from ..pricing import compute_quote
from ..session import Session
from ..skills import load_skill


class LoopState(TypedDict, total=False):
    user_text: str
    history: list                 # 已治理过的历史，父图传进来
    observations: Annotated[list, operator.add]
    events: Annotated[list, operator.add]
    decision: dict
    reply: str
    halted: str
    rounds: int


def build_loop(sess: Session, shelf, role: str,
               max_rounds: int | None = None):
    """编译一条支线的循环。

    提示词不在这里固定下来，而是在每一轮开始时拼。因为系统提示词里有一块
    是召回出来的长期记忆，而记忆是每轮都变的 —— 编译时定死就等于第一轮的
    记忆用到底。
    """

    tools = shelf.for_role(role)
    budget = max_rounds or (sess.cfg.max_rounds if sess.cfg else 6)

    def think(state: LoopState) -> dict:
        n = (state.get("rounds") or 0) + 1
        if n > budget:
            reason = f"{role} 支线已达 {budget} 轮上限"
            sess.guard.blocked.append({"kind": "rounds", "role": role, "detail": reason})
            return {
                "rounds": n, "halted": reason,
                "reply": _wrap_up(state.get("observations") or [], reason),
                "events": [{"kind": "halt", "role": role, "title": f"到达轮数上限（{budget}）",
                            "detail": "停下来把已知信息交给用户，不再继续查"}],
            }

        from .prompts import build_system
        system = build_system(
            role, sess,
            memory_note=sess.facts.get("memory_note", ""),
            skill_text=load_skill("after-sales") if role == "service" else "",
            slots=sess.facts.get("route_slots"))

        sess.trace.start(f"think-{role}-{n}")
        ctx = BrainCtx(
            role=role,
            user_text=state.get("user_text", ""),
            system=system,
            history=state.get("history") or [],
            observations=state.get("observations") or [],
            tool_names=shelf.names(role),
            hints=sess.facts,
        )
        d = sess.brain.decide(ctx, tools, on_token=sess.emit_token)
        ms = sess.trace.stop(f"think-{role}-{n}")

        ev = {
            "kind": "think", "role": role, "round": n, "ms": ms,
            "title": f"{_role_label(role)}思考（第 {n} 轮）",
            "detail": (f"决定调用 {d.tool}" if d.tool else "信息够了，可以回答"),
        }
        if d.thought:
            ev["thought"] = d.thought[:300]

        if not d.tool:
            obs = state.get("observations") or []

            # ── 写操作的兜底接管 ────────────────────────
            # 模型想收尾了，但这一轮该办的事还没办完。小模型在这里有个
            # 稳定的坏习惯：手上有报价，用户也说了要买，它却直接宣布
            # 「已下单成功」，而 place_order 一次都没调过。提示词掰不动
            # 这个习惯（试过几种写法都一样），所以写操作这一步不交给它。
            forced = _force_write(sess, role, state)
            if forced is not None:
                ev["detail"] = f"接管写操作：调用 {forced['tool']}"
                ev["forced"] = True
                return {"rounds": n, "decision": forced, "events": [ev]}

            # 用户拒绝了确认之后，「这笔没下、没有扣款」必须说出来。
            # 这件事不能让用户猜，也不能交给模型的措辞习惯 ——
            # 它上一句还在说「已为您下单」。
            if sess.facts.get("order_rejected") and not _write_ran(obs):
                return {"rounds": n, "decision": {}, "reply": _REJECT_REPLY,
                        "events": [ev, {"kind": "guard", "title": "下单已取消",
                                        "detail": "用户没同意金额确认，订单没有产生"}]}

            # 兜底：万一上面那条没触发，也要拦住没有凭据的成功话。
            # 「订单已生成」「下单成功」这类话只要说了就会有人信，
            # 而写工具根本没跑过。这种话不能让它出去。
            bad = _unverified_claim(d.reply, obs)
            if bad:
                sess.trace.add("guard", "拦下没有凭据的成功话", bad, None)
                return {"rounds": n, "decision": {}, "reply": _RETRY_REPLY,
                        "events": [ev, {"kind": "guard",
                                        "title": "拦下一句没有凭据的成功话",
                                        "detail": bad + "\n已换成如实答复"}]}

            # 还有一类没有凭据的话：编一个不存在的商品编码出来。
            # 它不像「已下单」那么扎眼，但用户照着这个编号去搜是搜不到的。
            fake = _fake_sku_claim(d.reply, sess.catalog)
            if fake:
                sess.trace.add("guard", "拦下编造的商品编码", fake, None)
                return {"rounds": n, "decision": {}, "reply": _FAKE_REPLY,
                        "events": [ev, {"kind": "guard",
                                        "title": "拦下编造的商品编码",
                                        "detail": fake + "\n已换成如实答复"}]}
            return {"rounds": n, "decision": {}, "reply": d.reply or "",
                    "events": [ev]}
        return {"rounds": n,
                "decision": {"tool": d.tool, "args": d.args or {},
                             "call_id": d.call_id or f"call_{n}"},
                "events": [ev]}

    def act(state: LoopState) -> dict:
        d = state.get("decision") or {}
        name = d.get("tool")
        args = d.get("args") or {}
        call_id = d.get("call_id") or ""
        if not name:
            return {}

        # ── 写操作：执行之前先过确认这一关 ──────────────
        if name == "place_order":
            confirm = _prepare_order(sess, args)
            if confirm is not None:
                return confirm
            args = {**args, "_confirmed": True}

        sess.trace.start(f"tool-{name}")
        out = shelf.invoke(name, args)
        ms = sess.trace.stop(f"tool-{name}")

        is_write = name in {"place_order", "create_ticket"}
        sess.trace.add("tool", name, out[:400], ms, args=args, write=is_write)
        sess.remember_obs(name, args, out, call_id=call_id)

        return {
            "observations": [{"tool": name, "args": args, "result": out,
                              "call_id": call_id}],
            "decision": {},
            "events": [{"kind": "tool", "title": name,
                        "detail": out[:220], "ms": ms, "write": is_write,
                        "args": args}],
        }

    def route_after_think(state: LoopState) -> str:
        if state.get("halted"):
            return "end"
        return "act" if (state.get("decision") or {}).get("tool") else "end"

    g = StateGraph(LoopState)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_edge(START, "think")
    g.add_conditional_edges("think", route_after_think, {"act": "act", "end": END})
    g.add_edge("act", "think")
    return g.compile()


def _prepare_order(sess: Session, args: dict) -> dict | None:
    """下单前的强制确认。

    返回 None 表示可以直接下单（金额在阈值内，或者已经在别处确认过）；
    返回一个 state 片段表示这次不执行 —— 要么抛了中断等人确认，
    要么用户拒绝了。

    注意这里先自己算一遍价，而不是相信模型报上来的金额。
    模型说的应付金额可能是它算错的，确认卡上要显示的是代码算出来的数。
    """
    sku = sess.catalog.resolve_sku(str(args.get("sku_id", "")))
    if sku is None:
        return None                     # 编码认不出，交给工具自己答
    try:
        qty = max(1, int(args.get("qty") or 1))
    except (TypeError, ValueError):
        qty = 1
    try:
        q = compute_quote(sess.catalog.by_sku, sess.catalog.promotions, sku, qty)
    except (KeyError, ValueError):
        return None                     # 商品不存在之类的问题交给工具自己答

    product = sess.catalog.by_sku.get(sku)

    if not sess.guard.needs_confirm(q.payable):
        sess.facts["confirmed_amount"] = q.payable
        return None

    answer = interrupt({
        "type": "confirm_order",
        "title": "这笔订单需要你确认",
        "sku_id": q.sku_id,
        "product": product.title if product else q.sku_id,
        "qty": q.qty,
        "unit_price": q.unit_price,
        "subtotal": q.subtotal,
        "discount": q.discount,
        "payable": q.payable,
        "applied": [a["name"] for a in q.applied],
        "threshold": sess.guard.max_amount,
        "reason": f"金额 {q.payable:.2f} 元超过设定的确认阈值 {sess.guard.max_amount:.0f} 元",
    })

    if not answer:
        sess.guard.blocked.append({"kind": "confirm", "role": "order",
                                   "detail": f"用户拒绝了下单 {q.sku_id} / {q.payable:.2f} 元"})
        # 拒绝也要留一条观察，但必须标出「没执行」。
        # 不标的话，后面那句「有凭据才能说成功」的检查会把它当成
        # 一次成功的下单，于是放行模型编出来的「已为您下单」。
        sess.facts["order_rejected"] = f"用户拒绝了 {q.payable:.2f} 元的确认请求"
        return {
            "observations": [{"tool": "place_order", "args": args, "executed": False,
                              "result": "用户没有确认这笔订单，系统未下单。"}],
            "decision": {},
            "events": [{"kind": "guard", "title": "下单已取消",
                        "detail": f"用户拒绝了 {q.payable:.2f} 元的确认请求"}],
        }

    sess.facts["confirmed_amount"] = q.payable
    sess.trace.add("guard", "确认通过",
                   f"{q.sku_id} × {q.qty}，应付 {q.payable:.2f} 元", None)
    return None


def _force_write(sess, role: str, state: dict) -> dict | None:
    """把「算价 → 落单」这两步从模型手里接过来。

    触发条件卡得很死，几条同时满足才接管：

    1. 走的是订单线 —— 别的支线没有写操作
    2. 模型这一轮已经动过手（说明它先做了认货这类前置动作）
    3. 用户这一轮明确说了要买
    4. 买的是哪一款能确定下来

    第 4 条有两条路：本轮已经算过价，商品就定了；本轮只检索过，
    那就从检索结果里把用户说的那款认出来，先补一次算价。

    满足这些之后还让模型自己想起来调 place_order，是把一笔真实交易
    压在它的记性上。接管过来之后，该有的人工确认一道都没少 ——
    金额越阈值照样中断，落单照样过幂等。
    """
    if role != "order":
        return None
    obs = state.get("observations") or []
    if not obs:
        return None
    done = {o.get("tool") for o in obs}
    if "place_order" in done:
        return None
    if not looks_like_purchase(sess.facts.get("last_user_text", "")):
        return None

    quote = sess.facts.get("last_quote") or {}
    sku = quote.get("sku_id")

    if not sku:
        # 还没算过价，就先补这一刀。它只是读操作，认错了也只是报错价，
        # 报价里带着商品名和编码，用户当场看得见。
        if "quote_price" in done:
            return None
        for o in reversed(obs):
            if o.get("tool") == "search_products":
                sku = _pin_from_search(str(o.get("result") or ""),
                                       sess.facts.get("last_user_text", ""))
                break
        if not sku:
            return None
        sess.trace.add("guard", "接管写操作",
                       f"已检索到 {sku}，先补一次算价", None)
        return {"tool": "quote_price", "args": {"sku_id": sku, "qty": 1},
                "call_id": "harness_quote", "forced": True}

    sess.trace.add("guard", "接管写操作",
                   f"报价已完成，用户要求成交，由流程直接推进落单 {sku}", None)
    return {"tool": "place_order",
            "args": {"sku_id": sku, "qty": int(quote.get("qty") or 1)},
            "call_id": "harness_place_order", "forced": True}


def _pin_from_search(result: str, user_text: str) -> str | None:
    """从检索结果里认出用户说的那一款。

    检索结果每行长这样：
        · SKU-D3007 明砚 34 英寸带鱼屏 | 明砚 | 3299 元（原价 3899） | ...
    做法是把第一行（检索排序的第一名）的标题拎出来，看它和用户这句话
    有没有共同的词。没有共同词就认不出，返回 None，让流程退回去问用户 ——
    宁可多问一句，不要替用户挑一款他没说的。
    """
    head = (result or "").strip().splitlines()
    head = head[0] if head else ""
    m = re.match(r"[·\-\s]*([A-Za-z]+-[A-Za-z0-9]+)\s+([^|]+)", head)
    if not m:
        return None
    sku, title = m.group(1), m.group(2).strip()
    if _shares_word(title, user_text):
        return sku
    return None


def _shares_word(title: str, user_text: str) -> bool:
    """两段中文里有没有连续两个字的共同片段。"""
    def grams(s: str) -> set[str]:
        xs = re.findall(r"[\u4e00-\u9fa5A-Za-z]{2,}", s or "")
        out: set[str] = set()
        for x in xs:
            for i in range(len(x) - 1):
                out.add(x[i:i + 2])
        return out

    common = grams(title) & grams(user_text)
    return bool(common)


_CLAIM_WORDS = (
    "下单成功", "已下单", "已经下单", "已为您下单", "已帮你下单", "下单好了",
    "订单已生成", "订单生成", "已生成订单", "已经生成订单", "已提交订单",
    "订单号：", "订单号是", "工单已创建", "已创建工单",
)

_RETRY_REPLY = (
    "抱歉，我上一句说错了 —— 这笔订单**还没有真正提交**，"
    "下单的动作没有执行成功。请再说一声「确认下单」，我立刻重新办一次。"
)

_REJECT_REPLY = (
    "好的，这笔订单**没有提交**，也没有产生任何扣款 —— "
    "刚才的金额确认你没有同意，所以系统停在了下单之前。\n"
    "想改数量、换一款，或者只是再看看，都可以直接跟我说。"
)


def _write_ran(observations: list[dict]) -> bool:
    """这一轮到底有没有真正执行过写操作。

    `executed=False` 的那些（用户拒绝了确认）不算 —— 它们顶着
    place_order 的名字，但订单并没有产生。
    """
    from ..tools import WRITE_TOOLS
    ran = {o.get("tool") for o in observations if o.get("executed", True)}
    return bool(ran & set(WRITE_TOOLS))


def _unverified_claim(reply: str, observations: list[dict]) -> str:
    """模型是不是在说一件它没做过的事。

    写操作（下单、建工单）有没有真的发生，以工具返回为准。模型没调过
    写工具，却在回复里说「下单成功」「订单号是……」，这句话必须拦下来 ——
    它比答错更严重，用户会以为钱已经付了。

    返回一句说明（供界面展示），没有可疑说法就返回空串。
    """
    r = (reply or "").strip()
    if not r:
        return ""
    hits = [w for w in _CLAIM_WORDS if w in r]
    if not hits:
        return ""
    if _write_ran(observations):
        return ""                       # 真的执行过写工具，那句话是有凭据的
    from ..tools import WRITE_TOOLS
    return (f"回复里出现了「{hits[0]}」，但这一轮没有成功执行过任何写工具"
            f"（{ '、'.join(sorted(WRITE_TOOLS)) }）")


_SKU_IN_TEXT = re.compile(r"SKU-[A-Za-z]\d{3,}", re.I)

_FAKE_REPLY = (
    "抱歉，我刚才报的商品编码是不存在的 —— 我没有真正查过库就说了。\n"
    "请再说一次你想要什么，我先检索，把真实在售的型号和价格列给你。"
)


def _fake_sku_claim(reply: str, catalog) -> str:
    """回复里有没有出现库里不存在的商品编码。

    这是「没有凭据的话」的另一种形态，比谎称下单更隐蔽：模型不去检索，
    直接编一个 SKU 编号和价格，说得有零有整。用户看不出真假，
    照着这个编号去下单才会发现根本没有这款。

    检查很便宜也很确定：把回复里的编码抠出来，逐个拿去库里认。
    认不出的，就是编的。
    """
    if not reply or catalog is None:
        return ""
    ids = set(_SKU_IN_TEXT.findall(reply))
    bad = sorted(x for x in ids if catalog.resolve_sku(x) is None)
    if not bad:
        return ""
    return f"回复里出现了库里不存在的商品编码：{'、'.join(bad)}"


def _wrap_up(observations: list[dict], reason: str) -> str:
    """被上限打断时的收尾：把已经拿到的东西交出去，不假装什么都没发生。"""
    if not observations:
        return f"这次没能查完（{reason}）。你可以换个更具体的问法，我再试一次。"
    parts = [f"（{reason}，先把已经查到的给你）"]
    for o in observations:
        parts.append(o["result"])
    return "\n\n".join(parts)


_ROLE_LABELS = {"advisor": "导购", "order": "订单", "service": "客服", "chat": "闲聊"}


def _role_label(role: str) -> str:
    return _ROLE_LABELS.get(role, role)


def build_all_loops(sess: Session, shelf) -> dict:
    """把三条支线的循环一次性编好，供主图取用。

    只编一次、反复使用，而不是每轮重建。Agent 图里带闭包和编译开销，
    放在请求路径上会让首字延迟无谓地变大。
    """
    return {role: build_loop(sess, shelf, role)
            for role in ("advisor", "order", "service")}

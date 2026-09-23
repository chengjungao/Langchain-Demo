# -*- coding: utf-8 -*-
"""客服线的工具：查规则、取退换判定所需的事实、建工单。

这条线里最值得看的是**工具和流程的分工**。

「能不能退」这个问题由三个东西决定：订单什么时候签收的、商品属于哪个类目、
平台的规则怎么写的。前两个是**事实**，第三个是**规则**。

事实要准，所以放在工具里查；规则会变，所以写在 SKILL.md 里由模型读。
工具不替模型做判定 —— 它只把判定需要的东西摆在桌面上。
这样做的好处是：规则改一行 markdown 就能生效，不用改代码、不用重新部署。
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from langchain_core.tools import tool

from ..session import Session

ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME = ROOT / "runtime"


def build_service_tools(sess: Session) -> list:
    cat, guard = sess.catalog, sess.guard

    @tool
    def search_policies(query: str, scope: str = "") -> str:
        """查平台规则。用户问退换货、运费、发票、保修、价保怎么规定的时用它。

        query  用用户的问题原话，比如「退货的运费谁出」
        scope  规则范围，可选：「退换货」「配送」「发票」「保修」「价保」「促销」。
               能判断出来就填，能明显提高准确度。

        返回：命中的规则原文。注意检索是字面匹配，问法和规则写法差得远时
        可能一条都命中不了，这时把已知的部分答出来并说明没查到完整规定。
        """
        idx = cat.policy_index
        allowed_scope = scope.strip() if scope.strip() else None

        def where(doc_id: str) -> bool:
            if not allowed_scope:
                return True
            doc = cat.policy_by_id(doc_id)
            return doc is not None and doc.scope == allowed_scope

        hits = idx.search(query, top_k=4, where=where)
        if not hits and allowed_scope:
            hits = idx.search(query, top_k=4)     # 范围写错了不至于空手回去
        if not hits:
            titles = "；".join(f"{p.doc_id} {p.title}" for p in cat.policies[:8])
            return (f"没有检索到与「{query}」直接相关的规则。"
                    f"现有规则可以按这些词找：{titles}…")

        lines = []
        for doc_id, score in hits:
            doc = cat.policy_by_id(doc_id)
            if doc is None:
                continue
            lines.append(f"【{doc.doc_id}｜{doc.scope}】{doc.title}\n{doc.content}"
                         f"\n（生效日期 {doc.effective_from}）")
        lines.append("以上按字面相关度排序，第一条不一定就是最匹配的。"
                     "如果答不出用户真正问的那一点，直接说没查到，不要用邻近的规则顶替。")
        return "\n\n".join(lines)

    @tool
    def check_return(order_id: str, reason: str = "") -> str:
        """取「这笔订单能不能退换」所需要的事实。

        用户问退货、换货、退款时用它。它**不给出结论** —— 结论要按售后流程的
        规则来判。它只负责把判定需要的东西查清楚。

        reason 是用户说的退货理由，原话照抄，用来判断是质量问题还是无理由。

        返回：订单状态、签收时间与距今天数、商品类目、是否拆封（按类目推定）、
        以及检索到的相关规则条款。
        """
        o = cat.by_order.get(order_id.strip().upper())
        if o is None:
            return f"没有这个订单号：{order_id}"
        if o.user_id != sess.user_id:
            return "这笔订单不属于当前用户，无权查看。"

        p = cat.by_sku.get(o.sku_id)
        lg = cat.logistics_of.get(o.order_id)
        signed_at = ""
        if lg:
            for t in lg.traces:
                if "签收" in t.status:
                    signed_at = t.time
        days = ""
        if signed_at:
            try:
                d = datetime.strptime(signed_at[:10], "%Y-%m-%d").date()
                days = (date.today() - d).days
            except ValueError:
                days = ""

        scope_q = "退货运费 承担 时效 " + (reason or "")
        hits = cat.policy_index.search(scope_q, top_k=3)

        lines = [
            f"订单：{o.order_id}　状态：{o.status}　下单：{o.created_at}",
            f"商品：{p.title if p else o.sku_id}　类目：{p.category if p else '未知'}"
            f"　子类：{p.subcategory if p else '未知'}",
            f"实付：{o.paid_amount:.2f} 元　数量：{o.qty}",
        ]
        if signed_at:
            lines.append(f"签收时间：{signed_at}　距今 {days} 天")
        else:
            lines.append("签收时间：暂无签收记录（物流尚未签收或物流单缺失）")
        lines.append(f"用户给的理由：{reason or '（用户没说）'}")
        lines.append("相关规则条款：")
        for doc_id, _ in hits:
            doc = cat.policy_by_id(doc_id)
            if doc:
                lines.append(f"· 【{doc.doc_id}】{doc.title}：{doc.content}")
        return "\n".join(lines)

    @tool
    def create_ticket(order_id: str, issue: str) -> str:
        """创建售后工单。用户明确要投诉、要人工介入、或者问题规则里解决不了时用它。

        这是**写操作**。同一笔订单的同一个问题只会建一张工单，重复调用返回原工单号。
        返回：工单号与预计处理时效。
        """
        if not issue.strip():
            return "工单需要写清楚问题，用户描述了什么问题就照实填。"
        o = cat.by_order.get(order_id.strip().upper())
        if o is None:
            return f"没有这个订单号：{order_id}"
        if o.user_id != sess.user_id:
            return "这笔订单不属于当前用户，不能建工单。"

        ikey = guard.make_key(sess.session_id, "ticket", o.order_id, issue[:20])
        prev = guard.lookup(ikey)
        if prev is not None:
            return f"这个问题已经建过工单了，没有重复创建。{prev}"

        ticket_id = "TK" + datetime.now().strftime("%y%m%d") + \
                    f"{guard.replay_count() + len(cat.orders) + 1:03d}"
        row = {"ticket_id": ticket_id, "order_id": o.order_id,
               "user_id": sess.user_id, "issue": issue,
               "status": "待处理", "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "session_id": sess.session_id}
        (RUNTIME).mkdir(parents=True, exist_ok=True)
        with open(RUNTIME / "tickets.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        guard.remember(ikey, f"工单 {ticket_id}")
        return f"已创建工单 {ticket_id}，关联订单 {o.order_id}，状态待处理，预计 24 小时内响应。"

    return [search_policies, check_return, create_ticket]

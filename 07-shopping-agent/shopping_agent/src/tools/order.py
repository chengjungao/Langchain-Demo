# -*- coding: utf-8 -*-
"""订单线的工具：查订单、查物流、算价、下单。

这条线里有一个**写操作**（下单），所以它和别的工具不一样，多了两道约束：

1. **每次调用前先看归属**。订单号是可以被猜到的，工具不能因为「模型要它」
   就把别人的订单吐出来。这一条写在工具里，不能指望提示词。
2. **下单要幂等，且要确认过**。重复点击、超时重试、模型自作主张，
   三种情况都会导致重复下单。幂等键挡住前两种，确认标记挡住第三种。
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from langchain_core.tools import tool

from ..pricing import compute_quote
from ..session import Session

ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME = ROOT / "runtime"


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_order_tools(sess: Session) -> list:
    cat, guard = sess.catalog, sess.guard

    def own_order(order_id: str):
        """统一的归属校验。返回 (订单, 错误信息)。"""
        key = cat.resolve_order(order_id)
        o = cat.by_order.get(key) if key else None
        if o is None:
            return None, (f"没有这个订单号：{order_id}。"
                          f"这位用户名下的订单可以用 list_my_orders 查。")
        if o.user_id != sess.user_id:
            # 这里不透露订单是否存在之外的任何信息
            return None, "这笔订单不属于当前用户，无权查看。"
        return o, None

    @tool
    def list_my_orders(status: str = "") -> str:
        """列出当前用户的订单。用户问「我买了什么」「我的订单」时用它。

        status 可选，用来过滤：「待发货」「已发货」「已签收」「已取消」「售后中」。
        不传就返回全部。返回：订单号、商品、金额、状态、下单日期。
        """
        rows = cat.orders_of.get(sess.user_id, [])
        if status:
            rows = [o for o in rows if status in o.status]
        if not rows:
            return f"没有{'状态为「' + status + '」的' if status else ''}订单。"
        rows = sorted(rows, key=lambda o: o.created_at, reverse=True)
        lines = [f"共 {len(rows)} 笔："]
        for o in rows:
            p = cat.by_sku.get(o.sku_id)
            title = p.title if p else o.sku_id
            lines.append(f"· {o.order_id}　{title} ×{o.qty}　"
                         f"{o.paid_amount:.2f} 元　{o.status}　{o.created_at}")
        return "\n".join(lines)

    @tool
    def get_order(order_id: str) -> str:
        """查一笔订单的详情。用户报出订单号、问金额或收件信息时用它。

        只能查当前用户自己的订单。返回：商品、数量、金额、状态、承运商、运单号。
        """
        o, err = own_order(order_id)
        if err:
            return err
        p = cat.by_sku.get(o.sku_id)
        lines = [
            f"订单号：{o.order_id}",
            f"商品：{p.title if p else o.sku_id}（{o.sku_id}）× {o.qty}",
            f"单价：{o.unit_price:.2f}　优惠：{o.discount:.2f}　实付：{o.paid_amount:.2f} 元",
            f"状态：{o.status}　下单日期：{o.created_at}",
            f"承运商：{o.carrier}　运单号：{o.tracking_no}",
            f"收货地址：{o.address_masked}",
        ]
        return "\n".join(lines)

    @tool
    def get_logistics(order_id: str) -> str:
        """查物流。用户问「到哪了」「什么时候到」时用它。

        只能查当前用户自己的订单。返回：当前状态 + 从下单到现在的完整轨迹。
        """
        o, err = own_order(order_id)
        if err:
            return err
        lg = cat.logistics_of.get(o.order_id)
        if lg is None:
            return f"{o.order_id} 还没有物流信息，状态是「{o.status}」。"
        lines = [f"{o.order_id}（{lg.carrier} {lg.tracking_no}）当前：{lg.current}"]
        for t in lg.traces[-6:]:
            lines.append(f"· {t.time}　{t.status}　{t.desc}（{t.location}）")
        return "\n".join(lines)

    @tool
    def quote_price(sku_id: str, qty: int = 1) -> str:
        """算价。用户问「多少钱」「有优惠吗」，或者你准备下单之前，都要先算一次。

        返回：单价、原价小计、命中的优惠明细、需要券码的可用优惠、最终应付金额。
        金额一律以此为准，不要自己算。
        """
        sku = cat.resolve_sku(sku_id)
        if sku is None:
            return f"没有这个商品编码：{sku_id}"
        try:
            r = compute_quote(cat.by_sku, cat.promotions, sku, max(1, qty))
        except KeyError:
            return f"没有这个商品编码：{sku_id}"
        except ValueError as e:
            return f"参数不对：{e}"
        sess.facts["last_quote"] = {"sku_id": r.sku_id, "qty": r.qty,
                                    "payable": r.payable}
        return r.lines()

    @tool
    def place_order(sku_id: str, qty: int = 1) -> str:
        """下单。只有用户明确说了「下单」「买了」「就这个」之后才能调用。

        这是一个**写操作**，会产生真实订单。两道约束必须同时满足：
        金额已经过用户确认，且同一笔请求没有被处理过。
        返回：订单号、实付金额、预计发货时间。重复调用不会产生第二笔订单。
        """
        sku = cat.resolve_sku(sku_id)
        p = cat.by_sku.get(sku) if sku else None
        if p is None:
            return f"没有这个商品编码：{sku_id}"
        qty = max(1, qty)
        if p.stock < qty:
            return f"{sku} 库存只剩 {p.stock} 件，没法下 {qty} 件。"

        try:
            q = compute_quote(cat.by_sku, cat.promotions, sku, qty)
        except (KeyError, ValueError) as e:
            return f"算价失败：{e}"

        # 约束一：确认过没有。没确认就只报价，不落单。
        confirmed = float(sess.facts.get("confirmed_amount") or -1)
        if abs(confirmed - q.payable) > 0.01:
            return (f"这笔订单应付 {q.payable:.2f} 元，还没有得到用户确认，"
                    f"系统拒绝直接下单。请先把金额和优惠明细告诉用户，"
                    f"等他明确同意后再调用一次。\n\n{q.lines()}")

        # 约束二：同一笔请求只处理一次。键里带上会话、商品、数量，
        # 换个数量算另一笔，重放同一个请求则命中已有的结果。
        ikey = guard.make_key(sess.session_id, "place_order", sku_id, qty)
        prev = guard.lookup(ikey)
        if prev is not None:
            stripped = prev.split("\n")[0]
            return (f"这笔请求之前已经处理过了，没有重复下单。原结果：{stripped}\n"
                    f"（幂等键 {ikey[:12]}…，累计拦截重复请求 {guard.replay_count()} 次）")

        today = date.today().strftime("%Y%m%d")
        seq = len(cat.orders) + 1
        order_id = f"SO{today}{seq:03d}"
        u = sess.user
        addr = ((u.city if u else "未知") + "　") + "收货地址已脱敏"
        row = {
            "order_id": order_id, "user_id": sess.user_id, "sku_id": sku_id,
            "qty": qty, "unit_price": p.price, "discount": q.discount,
            "paid_amount": q.payable, "status": "待发货",
            "created_at": date.today().isoformat(), "carrier": "待分配",
            "tracking_no": "待分配", "address_masked": addr,
            "session_id": sess.session_id,
        }
        _append_jsonl(RUNTIME / "placed_orders.jsonl", row)

        # 让本次会话立刻能查到这笔订单
        from ..models import Order
        o = Order(**{k: v for k, v in row.items() if k in Order.model_fields})
        cat.orders.append(o)
        cat.by_order[order_id] = o
        cat.orders_of.setdefault(sess.user_id, []).append(o)

        guard.remember(ikey, f"{order_id}　实付 {q.payable:.2f} 元")
        return (f"下单成功。\n订单号：{order_id}\n"
                f"商品：{p.title} × {qty}\n优惠：{q.discount:.2f} 元\n"
                f"实付：{q.payable:.2f} 元\n状态：待发货，预计 48 小时内出库。")

    return [list_my_orders, get_order, get_logistics, quote_price, place_order]

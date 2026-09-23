# -*- coding: utf-8 -*-
"""算价：把商品价格和促销规则算成一个确定的结果。

为什么单独拿出来：价格是这门生意里**最不能出错**的一处。让模型自己算，
它会给你一个看着合理、但对不上规则的数字，而且每次算得还不一样。

所以这里的做法是——规则归代码，模型只负责决定「要不要算」和「跟用户怎么说」。
工具返回的金额是代码算出来的，模型的职责是把数字讲清楚，不是产生数字。

规则本身有几处容易踩的地方，都写在这里而不是写在文档里：

1. **门槛按原价判定**。满 800 减 100 里的 800，看的是原价小计，
   不是被折扣减过之后的金额。两处口径混用是算价 bug 的常见来源。
2. **可叠加与不可叠加要分开算**。不可叠加的只取最优的一个，
   而不是把规则表从头到尾扫一遍全都减上。
3. **可用券和已用券要分开呈现**。需要券码的优惠，用户没给码就不能算进
   最终金额，但应该告诉他「你还有一张能用的」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .models import Promotion, Quote

# 需要用户提供券码才算数的类型。
CODE_REQUIRED = {"优惠券"}
# 门槛一律按原价小计判定的类型。折扣类按原价算优惠额，也一样。
FLAT_TYPES = {"满减", "立减", "套装", "优惠券"}
# 活动页定价，不走这里的规则引擎。
SKIP_TYPES = {"秒杀"}


@dataclass
class PriceResult:
    sku_id: str
    qty: int
    unit_price: float
    subtotal: float
    applied: list[dict] = field(default_factory=list)
    available: list[dict] = field(default_factory=list)
    discount: float = 0.0
    payable: float = 0.0

    def to_quote(self) -> Quote:
        return Quote(
            sku_id=self.sku_id, qty=self.qty, unit_price=self.unit_price,
            applied=[a["name"] for a in self.applied],
            discount=round(self.discount, 2), payable=round(self.payable, 2))

    def lines(self) -> str:
        out = [f"商品：{self.sku_id} × {self.qty}",
               f"单价：{self.unit_price:.2f} 元　小计：{self.subtotal:.2f} 元"]
        if self.applied:
            out.append("已生效优惠：")
            for a in self.applied:
                out.append(f"  · {a['name']}　-{a['amount']:.2f} 元"
                           f"（{a['reason']}）")
        else:
            out.append("已生效优惠：无")
        if self.available:
            out.append("还能用但需要券码：")
            for a in self.available:
                out.append(f"  · {a['name']}　code={a['code']}　可减 {a['amount']:.2f} 元")
        out.append(f"优惠合计：{self.discount:.2f} 元")
        out.append(f"应付：{self.payable:.2f} 元")
        return "\n".join(out)


def _in_window(p: Promotion, today: date) -> bool:
    try:
        s = date.fromisoformat(p.start_at)
        e = date.fromisoformat(p.end_at)
    except ValueError:
        return False
    return s <= today <= e


def _hits_category(p: Promotion, category: str) -> bool:
    cats = p.applicable_categories or []
    return (not cats) or ("全部" in cats) or (category in cats)


def _amount_of(p: Promotion, subtotal: float) -> float:
    """这条规则能减多少。"""
    if p.type == "折扣":
        rate = p.value if p.value <= 1 else p.value / 100
        return subtotal * (1 - rate)
    return float(p.value)


def compute_quote(products_by_sku: dict, promotions: list[Promotion],
                  sku_id: str, qty: int = 1,
                  today: date | None = None) -> PriceResult:
    p = products_by_sku.get(sku_id)
    if p is None:
        raise KeyError(f"找不到商品 {sku_id}")
    if qty < 1:
        raise ValueError("数量至少为 1")

    today = today or date.today()
    r = PriceResult(sku_id=sku_id, qty=qty, unit_price=p.price,
                    subtotal=round(p.price * qty, 2))

    stackable: list[tuple[Promotion, float]] = []
    exclusive: list[tuple[Promotion, float]] = []

    for promo in promotions:
        if promo.type in SKIP_TYPES:
            continue
        if not _in_window(promo, today) or not _hits_category(promo, p.category):
            continue
        # 门槛按原价小计判定，这一条决定了下面的所有数字
        if promo.threshold and r.subtotal < promo.threshold:
            continue
        amt = round(_amount_of(promo, r.subtotal), 2)
        if amt <= 0:
            continue

        if promo.type in CODE_REQUIRED and promo.code:
            r.available.append({"name": promo.name, "code": promo.code,
                                "amount": amt, "promo_id": promo.promo_id})
            continue
        (stackable if promo.stackable else exclusive).append((promo, amt))

    # 可叠加的全上；不可叠加的只挑减得最多的那一个
    chosen = list(stackable)
    if exclusive:
        best = max(exclusive, key=lambda t: t[1])
        chosen.append(best)
        for promo, amt in exclusive:
            if promo.promo_id != best[0].promo_id:
                r.available.append({"name": promo.name, "code": promo.code or "—",
                                    "amount": amt, "promo_id": promo.promo_id})

    for promo, amt in chosen:
        r.applied.append({
            "name": promo.name, "promo_id": promo.promo_id, "amount": amt,
            "reason": (f"原价小计 {r.subtotal:.0f} 元满足门槛 {promo.threshold:.0f} 元"
                       if promo.threshold else "无门槛")})
        r.discount += amt

    r.discount = round(r.discount, 2)
    r.payable = round(max(0.0, r.subtotal - r.discount), 2)
    return r

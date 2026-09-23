# -*- coding: utf-8 -*-
"""数据模型。

JSONL 里的记录读进来之后，第一件事是变成有类型的东西。这样做有两个好处：
一是字段名写错会当场报错，而不是等到拼接提示词时才发现取到了 None；
二是这些模型可以直接拿去做结构化输出，工具返回什么、模型该吐什么，用同一套定义。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Product(BaseModel):
    sku_id: str
    spu_id: str
    title: str
    brand: str
    category: str
    subcategory: str
    price: float
    list_price: float
    specs: dict[str, str | bool]
    tags: list[str]
    highlights: list[str]
    stock: int
    rating: float
    review_count: int
    warranty_months: int
    ship_from: str
    weight_g: int

    def brief(self) -> str:
        """给模型看的一行摘要。字段顺序按「决策相关度」排，不是按数据表顺序。"""
        promo = ""
        if self.list_price > self.price:
            promo = f"（原价 {self.list_price:.0f}）"
        return (f"{self.sku_id} {self.title} | {self.brand} | "
                f"{self.price:.0f} 元{promo} | 评分 {self.rating} | "
                f"{self._spec_line()}")

    def _spec_line(self, limit: int = 4) -> str:
        items = [f"{k} {v}" for k, v in list(self.specs.items())[:limit]]
        return "，".join(items)


class User(BaseModel):
    user_id: str
    nickname: str
    city: str
    member_level: str
    budget_range: list[int]
    preferred_brands: list[str]
    disliked_brands: list[str]
    sizes: dict[str, str]
    past_categories: list[str]
    notes: list[str]


class Order(BaseModel):
    order_id: str
    user_id: str
    sku_id: str
    qty: int
    unit_price: float
    discount: float
    paid_amount: float
    status: str
    created_at: str
    carrier: str
    tracking_no: str
    address_masked: str


class Trace(BaseModel):
    time: str
    status: str
    desc: str
    location: str


class Logistics(BaseModel):
    tracking_no: str
    order_id: str
    carrier: str
    status: str
    current: str
    traces: list[Trace]


class Policy(BaseModel):
    doc_id: str
    scope: str
    title: str
    content: str
    effective_from: str
    tags: list[str]


class Review(BaseModel):
    review_id: str
    sku_id: str
    user_id: str
    rating: int
    content: str
    tags: list[str]
    created_at: str


class Promotion(BaseModel):
    promo_id: str
    name: str
    type: str
    threshold: float
    value: float
    applicable_categories: list[str]
    stackable: bool
    start_at: str
    end_at: str
    code: str | None = None


# ── 结构化输出模型 ──────────────────────────────────────────
# 这两个不只是「输出格式」，它们同时是路由契约：
# route 决定去哪条支线，entities 决定支线里有几个参数不用再问用户。

_ROUTE_ALIASES = (
    ("advisor", ("advisor", "product", "recommend", "search", "browse", "shop", "导购", "推荐")),
    ("order", ("order", "logistic", "ship", "deliver", "track", "订单", "物流",
               "下单", "支付", "结算")),
    ("service", ("service", "policy", "return", "refund", "after", "warrant", "客服", "售后", "退")),
    ("chat", ("chat", "smalltalk", "greet", "other", "闲聊")),
)


# ── 成交意图的判定 ──────────────────────────────────────────
# 这个判断被两处用到：路由那一层（要办掉的意图不能判成「了解」），
# 以及订单线里接管写操作那一步。放在一个地方，免得两边口径不一致。
#
# 刻意用字符串匹配而不是模型判断：这一步每轮都要做，而且涉及钱，
# 判错的代价不对称 —— 判成「要买」最多多走一次算价和确认，
# 判成「不要买」会让用户的话石沉大海。

PURCHASE_WORDS = (
    "下单", "买了", "买下", "就要这个", "就这个", "要了", "付款", "支付",
    "结算", "成交", "确认购买", "帮我买", "给我买",
)

PURCHASE_BLOCKERS = ("不要了", "不买", "算了", "取消", "先不", "不用了", "再看看")


def looks_like_purchase(text: str) -> bool:
    """用户这句话是不是在说「成交」。"""
    t = (text or "").strip()
    if not t:
        return False
    if any(w in t for w in PURCHASE_BLOCKERS):
        return False
    return any(w in t for w in PURCHASE_WORDS)


def normalize_route(value: str) -> str:
    """把模型给的支线名归一到四选一。

    这里不把取值写成枚举，是因为小模型很爱自己造词 —— 你定义 advisor，
    它给你 product_recommendation。用枚举会直接校验失败、整条路由白跑；
    归一化之后，造词也能落到正确的那条线上。
    """
    v = (value or "").strip().lower()
    if not v:
        return "chat"
    for name, keys in _ROUTE_ALIASES:
        if v == name:
            return name
    for name, keys in _ROUTE_ALIASES:
        for k in keys:
            if k in v:
                return name
    return "chat"


class Route(BaseModel):
    """意图识别节点的输出。"""

    route: str = Field(
        default="chat",
        description="支线名，四选一：advisor=找货比价，order=订单物流与下单，"
                    "service=规则政策与售后，chat=寒暄兜底")
    reason: str = Field(default="", description="一句话说明为什么这么分，便于排查误判")
    category: str | None = Field(default=None, description="用户提到的类目：耳机/键盘/显示器")
    budget_max: float | None = Field(default=None, description="用户能接受的最高价，单位元")
    sku_id: str | None = Field(default=None, description="用户明确点名的商品编码")
    order_id: str | None = Field(default=None, description="用户明确点名的订单号")

    @field_validator("route", mode="before")
    @classmethod
    def _normalize(cls, v):
        return normalize_route(str(v) if v is not None else "")

    @field_validator("category", mode="before")
    @classmethod
    def _cat(cls, v):
        if not v:
            return None
        s = str(v).strip()
        for c in ("耳机", "键盘", "显示器"):
            if c in s:
                return c
        if "屏" in s:
            return "显示器"
        return None


class Quote(BaseModel):
    """算价结果。金额相关的动作都走这个模型，避免各处自己拼数字。"""

    sku_id: str
    qty: int
    unit_price: float
    applied: list[str] = Field(description="命中的优惠名称")
    discount: float
    payable: float
    currency: str = "CNY"

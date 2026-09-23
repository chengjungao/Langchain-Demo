# -*- coding: utf-8 -*-
"""数据集加载与索引。

7 个 JSONL 读进内存，建成几个字典索引。40 个商品这个量级不需要数据库，
但**索引该建还是要建** —— 因为工具函数会被模型反复调用，每次全表扫描会让
一轮对话的耗时随数据量线性上涨。

真实项目里这一层通常换成数据库或检索服务，接口保持不动即可。
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import (Logistics, Order, Policy, Product, Promotion, Review,
                     User)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read(name: str) -> list[dict]:
    p = DATA_DIR / name
    rows: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class Catalog:
    """一次性把数据读进来，之后所有查询都走内存索引。"""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        global DATA_DIR
        if data_dir is not None:
            DATA_DIR = Path(data_dir)

        self.products = [Product(**r) for r in _read("products.jsonl")]
        self.users = [User(**r) for r in _read("users.jsonl")]
        self.orders = [Order(**r) for r in _read("orders.jsonl")]
        self.logistics = [Logistics(**r) for r in _read("logistics.jsonl")]
        self.policies = [Policy(**r) for r in _read("policies.jsonl")]
        self.reviews = [Review(**r) for r in _read("reviews.jsonl")]
        self.promotions = [Promotion(**r) for r in _read("promotions.jsonl")]

        # 索引
        self.by_sku = {p.sku_id: p for p in self.products}
        self.by_user = {u.user_id: u for u in self.users}
        self.by_order = {o.order_id: o for o in self.orders}
        self.by_tracking = {x.tracking_no: x for x in self.logistics}
        self.logistics_of = {x.order_id: x for x in self.logistics}

        self.orders_of: dict[str, list[Order]] = {}
        for o in self.orders:
            self.orders_of.setdefault(o.user_id, []).append(o)

        self.reviews_of: dict[str, list[Review]] = {}
        for r in self.reviews:
            self.reviews_of.setdefault(r.sku_id, []).append(r)

        self.by_category: dict[str, list[Product]] = {}
        for p in self.products:
            self.by_category.setdefault(p.category, []).append(p)

        self.by_brand: dict[str, list[Product]] = {}
        for p in self.products:
            self.by_brand.setdefault(p.brand, []).append(p)

        # 检索索引按需构建。第一次用到时才建，避免启动时白算一遍 ——
        # 只查订单的会话根本不需要商品索引。
        self._p_index = None
        self._d_index = None

    # ── 检索索引 ────────────────────────────────────────
    @property
    def product_index(self):
        """商品的全文本索引。拼文本的规则在 retrieval.product_text 里。"""
        if self._p_index is None:
            from .retrieval import BM25Index, product_text
            self._p_index = BM25Index(
                [(p.sku_id, product_text(p)) for p in self.products])
        return self._p_index

    @property
    def policy_index(self):
        if self._d_index is None:
            from .retrieval import BM25Index, policy_text
            self._d_index = BM25Index(
                [(p.doc_id, policy_text(p)) for p in self.policies])
        return self._d_index

    def policy_by_id(self, doc_id: str) -> Policy | None:
        return next((p for p in self.policies if p.doc_id == doc_id), None)

    def resolve_sku(self, raw: str) -> str | None:
        """把模型口里的商品编码认回真实的那一个。

        小模型抄编码时会掉前缀、换大小写、加空格 —— 它盯着「SKU-D3002」
        看，抄出来是「D3002」。这些都是同一个东西，没必要让用户在
        「没有这个商品编码」上卡住。这里把几种常见写法都认掉，
        认不出就返回 None，让工具照实说没有。
        """
        if not raw:
            return None
        s = str(raw).strip().upper().replace(" ", "").replace("_", "-")
        if not s:
            return None
        cands = [s, "SKU-" + s]
        if s.startswith("SKU-"):
            cands.append(s[4:])
        if s.startswith("SKU") and not s.startswith("SKU-"):
            cands.append("SKU-" + s[3:])        # 掉了连字符的写法
        # 商品编码形如 SKU-D3002：取「前缀-段号」两截，多写的尾巴去掉
        parts = s.split("-")
        if len(parts) > 2:
            cands.append("-".join(parts[:2]))
        for cand in cands:
            if cand and cand in self.by_sku:
                return cand
        return None

    def resolve_order(self, raw: str) -> str | None:
        """订单号同理：去掉空格、统一大写之后再认。"""
        if not raw:
            return None
        s = str(raw).strip().upper().replace(" ", "").replace("_", "")
        if s in self.by_order:
            return s
        for oid in self.by_order:
            if oid.upper() == s or oid.upper().endswith(s):
                return oid
        return None

    def product_by_sku(self, raw: str) -> Product | None:
        """按用户或模型给的编码取商品，编码写法可以有出入。"""
        key = self.resolve_sku(raw)
        return self.by_sku.get(key) if key else None

    # ── 查询辅助 ────────────────────────────────────────
    def policies_in(self, scope: str | None) -> list[Policy]:
        if not scope:
            return self.policies
        hit = [p for p in self.policies if p.scope == scope]
        return hit or self.policies       # 范围写错时不至于一条都检索不到

    def bad_review_tags(self, sku_id: str) -> dict[str, int]:
        """统计一个商品的差评标签，用于「这款有什么通病」这类问题。"""
        counter: dict[str, int] = {}
        for r in self.reviews_of.get(sku_id, []):
            if r.rating <= 3:
                for t in r.tags:
                    counter[t] = counter.get(t, 0) + 1
        return dict(sorted(counter.items(), key=lambda kv: -kv[1]))

    def stats(self) -> dict[str, int]:
        return {
            "商品": len(self.products),
            "用户": len(self.users),
            "订单": len(self.orders),
            "物流": len(self.logistics),
            "政策": len(self.policies),
            "评价": len(self.reviews),
            "促销": len(self.promotions),
        }


_catalog: Catalog | None = None


def get_catalog(data_dir: Path | str | None = None) -> Catalog:
    """进程内单例。工具函数会被反复调用，不该每次都重读文件。"""
    global _catalog
    if _catalog is None or data_dir is not None:
        _catalog = Catalog(data_dir)
    return _catalog

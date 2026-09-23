# -*- coding: utf-8 -*-
"""数据集自洽校验。

这个脚本的存在理由很直接：构造出来的数据最容易出的错不是格式错，是
**引用错位**。订单指向一个不存在的 SKU、物流和订单对不上、成交价和单价
乘不出来，这些错在 demo 跑到一半时才炸，很难查。

所以把「数据是不是自洽」变成一条命令能回答的问题。

    python eval/validate_data.py

全部通过退出码 0，任一项失败退出码 1。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

ROUTES = {"advisor", "order", "service"}


def load(name: str) -> list[dict]:
    p = DATA / name if not name.startswith("eval/") else ROOT / name
    rows = []
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{name} 第 {lineno} 行不是合法 JSON：{e}") from e
    return rows


def _group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


class Report:
    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.items.append((name, ok, detail))

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [x for x in self.items if not x[1]]

    def dump(self) -> int:
        width = max(len(n) for n, _, _ in self.items) + 2
        for name, ok, detail in self.items:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<{width}} {detail}")
        print("-" * 72)
        if self.failed:
            print(f"  未通过 {len(self.failed)} 项，数据集不自洽")
            return 1
        print(f"  全部 {len(self.items)} 项通过")
        return 0


def build_report() -> Report:
    """跑完全部检查，返回结果对象。

    和 `main()` 分开，是为了让 `run.py --check` 也能用同一套检查 ——
    自检和数据校验要是两套逻辑，迟早会出现「自检说没事、单跑校验又报错」。
    """
    products = load("products.jsonl")
    users = load("users.jsonl")
    orders = load("orders.jsonl")
    logistics = load("logistics.jsonl")
    policies = load("policies.jsonl")
    reviews = load("reviews.jsonl")
    promos = load("promotions.jsonl")
    cases = load("eval/cases.jsonl")

    r = Report()

    # ── 规模 ──────────────────────────────────────────────
    r.check("商品条数", len(products) == 40, f"{len(products)} 条")
    r.check("用户条数", len(users) == 5, f"{len(users)} 条")
    r.check("订单条数", len(orders) == 12, f"{len(orders)} 条")
    r.check("政策条数", len(policies) == 27, f"{len(policies)} 条")
    r.check("评测集条数", len(cases) == 20, f"{len(cases)} 条")

    # ── 主键唯一 ──────────────────────────────────────────
    for label, rows, key in (("商品", products, "sku_id"), ("用户", users, "user_id"),
                             ("订单", orders, "order_id"), ("政策", policies, "doc_id"),
                             ("评价", reviews, "review_id"), ("促销", promos, "promo_id"),
                             ("评测集", cases, "case_id")):
        ids = [x[key] for x in rows]
        dup = {i for i in ids if ids.count(i) > 1}
        r.check(f"{label}主键唯一", not dup, f"重复：{sorted(dup)}" if dup else f"{len(ids)} 个")

    sku_ids = {p["sku_id"] for p in products}
    user_ids = {u["user_id"] for u in users}
    order_ids = {o["order_id"] for o in orders}

    # ── 引用完整性 ────────────────────────────────────────
    bad = [o["order_id"] for o in orders if o["sku_id"] not in sku_ids]
    r.check("订单引用的 SKU 都存在", not bad, f"悬空：{bad}" if bad else "12/12")

    bad = [o["order_id"] for o in orders if o["user_id"] not in user_ids]
    r.check("订单引用的用户都存在", not bad, f"悬空：{bad}" if bad else "12/12")

    bad = [x["tracking_no"] for x in logistics if x["order_id"] not in order_ids]
    r.check("物流引用的订单都存在", not bad, f"悬空：{bad}" if bad else "12/12")

    ord_set = {o["order_id"] for o in orders}
    log_set = {x["order_id"] for x in logistics}
    r.check("订单与物流一一对应", ord_set == log_set,
            f"缺物流：{sorted(ord_set - log_set)} 多物流：{sorted(log_set - ord_set)}"
            if ord_set != log_set else "12 对 12")

    bad = [x["review_id"] for x in reviews if x["sku_id"] not in sku_ids]
    r.check("评价引用的 SKU 都存在", not bad, f"悬空：{bad}" if bad else f"{len(reviews)}/60")

    bad = [c["case_id"] for c in cases if c["user_id"] not in user_ids]
    r.check("评测集引用的用户都存在", not bad, f"悬空：{bad}" if bad else "20/20")

    # ── 数值自洽 ──────────────────────────────────────────
    bad = []
    for o in orders:
        expect = round(o["unit_price"] * o["qty"] - o["discount"], 2)
        if abs(expect - o["paid_amount"]) > 0.01:
            bad.append(f"{o['order_id']} 期望 {expect} 实际 {o['paid_amount']}")
    r.check("成交价 = 单价×数量−优惠", not bad, f"不符：{bad}" if bad else "12/12")

    bad = [o["order_id"] for o in orders if o["paid_amount"] < 0]
    r.check("成交价非负", not bad, f"负数：{bad}" if bad else "12/12")

    bad = [p["sku_id"] for p in products if p["price"] > p["list_price"]]
    r.check("现价不高于标价", not bad, f"例外：{bad}" if bad else "40/40")

    bad = [p["sku_id"] for p in products if not (0 <= p["rating"] <= 5)]
    r.check("评分在 0~5", not bad, f"越界：{bad}" if bad else "40/40")

    bad = [p["sku_id"] for p in products if p["stock"] < 0]
    r.check("库存非负", not bad, f"负数：{bad}" if bad else "40/40")

    bad = [p["promo_id"] for p in promos if p["end_at"] < p["start_at"]]
    r.check("促销结束晚于开始", not bad, f"倒挂：{bad}" if bad else "10/10")

    # ── 业务合理性 ────────────────────────────────────────
    bad = [u["user_id"] for u in users
           if set(u["preferred_brands"]) & set(u["disliked_brands"])]
    r.check("喜好品牌与讨厌品牌不冲突", not bad, f"冲突：{bad}" if bad else "5/5")

    bad = [c["case_id"] for c in cases if c["expected_route"] not in ROUTES]
    r.check("评测集路由取值合法", not bad, f"非法：{bad}" if bad else f"取值域 {sorted(ROUTES)}")

    bad = [c["case_id"] for c in cases if not c["expected_tools"] and c["expected_route"] != "chat"]
    r.check("评测集都声明了期望工具", not bad, f"缺失：{bad}" if bad else "20/20")

    # 商品类目覆盖：三个类目都要有，否则导购线测不全
    cats = {p["category"] for p in products}
    r.check("类目覆盖三档", cats == {"耳机", "键盘", "显示器"}, f"{sorted(cats)}")

    # 评价内容要和商品类目对得上。这一项看着吹毛求疵，但「差评集中在哪」
    # 这个聚合功能完全依赖评价内容，一旦出现「耳机的评价里说屏幕有坏点」，
    # 功能给出的答案就是错的，而且是那种不容易被发现的错。
    cat_of = {p["sku_id"]: p["category"] for p in products}
    off_topic = {"耳机": ["屏", "支架", "键帽", "轴体", "打油"],
                 "键盘": ["降噪", "夹头", "耳塞", "续航一周"],
                 "显示器": ["降噪", "夹头", "键帽", "耳塞"]}
    bad = []
    for rv in reviews:
        for kw in off_topic[cat_of[rv["sku_id"]]]:
            if kw in rv["content"]:
                bad.append(f"{rv['review_id']}({cat_of[rv['sku_id']]}) 含「{kw}」")
    r.check("评价内容与类目匹配", not bad, f"串味：{bad[:4]}" if bad else f"{len(reviews)} 条")

    # 差评覆盖：至少要有若干商品能问出「差评集中在哪」，否则这条功能没有演示素材
    bad_tag_skus = sum(1 for sid, rs in _group(reviews, "sku_id").items()
                       if any(x["rating"] <= 3 for x in rs))
    r.check("有差评的商品数充足", bad_tag_skus >= 12, f"{bad_tag_skus} 个商品带差评")

    # 评测集的用户必须真的在该路由有可用的数据
    u_orders = {o["user_id"] for o in orders}
    bad = [c["case_id"] for c in cases
           if c["expected_route"] == "order" and c["user_id"] not in u_orders]
    r.check("订单类评测集的用户有订单", not bad, f"无订单：{bad}" if bad else "4/4")

    return r


def run_checks(quiet: bool = True) -> tuple[bool, str]:
    """给程序调用的版本：只返回结论，不打印逐项。"""
    r = build_report()
    if r.failed:
        names = "、".join(n for n, _, _ in r.failed[:4])
        return False, f"{len(r.failed)}/{len(r.items)} 项未通过（{names}）"
    return True, f"{len(r.items)} 项全部通过"


def main() -> int:
    print("数据集自洽校验")
    print("=" * 72)
    return build_report().dump()


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""导购线的工具：找货、看货、对比、看评价、取偏好。

工具描述（docstring）不是给人看的注释，它是**模型选工具的唯一依据**。
所以要写清楚三件事：什么时候该用它、参数怎么填、返回什么。
写成「查询商品」这种描述，模型只能靠猜。
"""
from __future__ import annotations

from langchain_core.tools import tool

from ..session import Session


def build_catalog_tools(sess: Session) -> list:
    cat = sess.catalog

    @tool
    def search_products(query: str, category: str = "", price_max: float = 0.0,
                        price_min: float = 0.0, limit: int = 5) -> str:
        """按需求找商品。用户想要推荐、想买点什么、想找某个价位的东西时用它。

        参数：
          query      用户的需求，用他自己的话描述，比如「通勤降噪 轻一点」。
                     不要把价格写进 query —— 价格用 price_max。
          category   类目，只能是「耳机」「键盘」「显示器」之一，不确定就留空
          price_max  价格上限，单位元。用户说「一千以内」就填 1000，没提就留 0
          price_min  价格下限，同上
          limit      返回条数，默认 5

        返回：商品列表，每条含 SKU 编码、名称、价格、评分与关键参数。
        说明：检索是字面匹配的，用户的口语说法可能一条都匹配不到。
        这种情况下会退回按条件列出，结果可能不精准，你要在回复里说清楚。
        """
        def where(sku_id: str) -> bool:
            p = cat.by_sku[sku_id]
            if category and p.category != category:
                return False
            if price_max and p.price > price_max:
                return False
            if price_min and p.price < price_min:
                return False
            return True

        hits = cat.product_index.search(query, top_k=max(limit, 4), where=where) \
            if query.strip() else []

        if hits:
            lines = [f"关键词检索到 {len(hits)} 款："]
            for sku_id, score in hits[:limit]:
                lines.append(f"· {cat.by_sku[sku_id].brief()}")
            return "\n".join(lines)

        # 兜底：关键词没命中时退回结构化条件。真实系统里这一步很关键 ——
        # 用户说的「三百以内」「新手用」这类话，本来就不该指望文本检索命中。
        cands = [p for p in cat.products if where(p.sku_id)]
        if not cands:
            return ("按当前条件没有找到商品。条件可能是：类目 "
                    f"{category or '不限'}、价格 {price_min or 0}~{price_max or '不限'} 元。"
                    "可以建议用户放宽价格或换个类目。")
        cands.sort(key=lambda p: (-p.rating, p.price))
        lines = [f"没有匹配到「{query}」这个词，按条件列出 {min(limit, len(cands))} 款"
                 f"（排序依据是评分与价格，不是相关性）："]
        for p in cands[:limit]:
            lines.append(f"· {p.brief()}")
        return "\n".join(lines)

    @tool
    def get_product(sku_id: str) -> str:
        """看某个商品的完整信息。用户点名了具体商品、或者你要确认它的参数时用它。

        sku_id 形如 SKU-E1001（耳机）、SKU-K2001（键盘）、SKU-D3001（显示器）。
        返回：参数、库存、保修、发货仓，以及差评里集中出现的问题。
        """
        p = cat.product_by_sku(sku_id)
        if p is None:
            head = str(sku_id).strip().upper()[:6]
            near = [x.sku_id for x in cat.products
                    if x.sku_id.upper().startswith(head)][:5]
            return f"没有这个商品编码：{sku_id}。" + (f"你是不是要找：{'、'.join(near)}" if near else "")

        specs = "；".join(f"{k} {v}" for k, v in p.specs.items())
        bad = cat.bad_review_tags(p.sku_id)
        bad_txt = ("；".join(f"{k}（{v} 次）" for k, v in list(bad.items())[:3])
                   if bad else "暂无明显集中的负面反馈")
        return "\n".join([
            f"{p.sku_id}　{p.title}",
            f"品牌：{p.brand}　类目：{p.category} / {p.subcategory}",
            f"价格：{p.price:.2f} 元" + (f"（原价 {p.list_price:.2f}）" if p.list_price > p.price else ""),
            f"参数：{specs}",
            f"库存：{p.stock} 件　评分：{p.rating}（{p.review_count} 条评价）",
            f"保修：{p.warranty_months} 个月　发货仓：{p.ship_from}　自重：{p.weight_g} g",
            f"卖点：{'；'.join(p.highlights)}",
            f"差评集中点：{bad_txt}",
        ])

    @tool
    def compare_products(sku_ids: str) -> str:
        """横向对比两款或多款商品。用户问「哪个好」「有什么区别」时用它。

        sku_ids：多个商品编码，用逗号分隔，如 "SKU-E1002,SKU-E1005"。

        返回：参数逐项对齐的对比表 + 价格与评分差异。
        不同类目的商品参数项不同，只会对比它们都有的项和各自的独有项。
        """
        raw_ids = [s.strip() for s in sku_ids.replace("，", ",").split(",") if s.strip()]
        ids = [cat.resolve_sku(s) for s in raw_ids]
        items = [cat.by_sku[i] for i in ids if i]
        missing = [s for s, i in zip(raw_ids, ids) if not i]
        if not items:
            return f"这些编码都没找到：{', '.join(raw_ids)}"
        if len(items) == 1:
            return "只有一款，没法对比。至少给两个商品编码。"

        keys: list[str] = []
        for p in items:
            for k in p.specs:
                if k not in keys:
                    keys.append(k)

        head = "| 项目 | " + " | ".join(f"{p.sku_id}" for p in items) + " |"
        sep = "| --- | " + " | ".join("---" for _ in items) + " |"
        rows = [
            head, sep,
            "| 名称 | " + " | ".join(p.title for p in items) + " |",
            "| 价格 | " + " | ".join(f"{p.price:.0f} 元" for p in items) + " |",
            "| 评分 | " + " | ".join(f"{p.rating}" for p in items) + " |",
            "| 重量 | " + " | ".join(f"{p.weight_g} g" for p in items) + " |",
            "| 库存 | " + " | ".join(f"{p.stock}" for p in items) + " |",
        ]
        for k in keys:
            vals = [str(p.specs.get(k, "—")) for p in items]
            if all(v == "—" for v in vals):
                continue
            rows.append(f"| {k} | " + " | ".join(vals) + " |")
        if missing:
            rows.append(f"\n（没找到的编码：{', '.join(missing)}）")
        return "\n".join(rows)

    @tool
    def get_reviews(sku_id: str, only_bad: bool = False) -> str:
        """读某个商品的用户评价。用户问「用起来怎么样」「有什么毛病」时用它。

        only_bad=True 只看 3 星及以下的评价，用于回答「有什么通病」这类问题。
        返回：评价原文 + 标签统计。
        """
        p = cat.product_by_sku(sku_id)
        if p is None:
            return f"没有这个商品编码：{sku_id}"
        rs = cat.reviews_of.get(p.sku_id, [])
        if only_bad:
            rs = [r for r in rs if r.rating <= 3]
        if not rs:
            return f"{p.sku_id} 目前没有满足条件的评价。"
        lines = [f"{p.sku_id} 共 {len(rs)} 条评价："]
        for r in rs[:6]:
            lines.append(f"· {r.rating} 星　{r.content}　[{'/'.join(r.tags)}]")
        tags = cat.bad_review_tags(p.sku_id)
        if tags:
            lines.append("差评标签统计：" + "；".join(f"{k}×{v}" for k, v in tags.items()))
        return "\n".join(lines)

    @tool
    def recall_preferences() -> str:
        """调取这位用户的长期偏好与历史。

        用户说「还是老样子」「上次那个」「我不喜欢……」这类需要结合历史
        才能答好的问题时用它。返回：档案里的稳定偏好 + 系统记下的过往事件。
        注意这些信息可能过时，与用户当下的说法冲突时以当下为准。
        """
        from ..memory import format_memory_block
        note = sess.profile_note()
        hits = sess.memory.recall(sess.user_id, sess.facts.get("last_user_text", ""), limit=8)
        blocks = []
        if note:
            blocks.append("档案信息：" + note)
        if hits:
            blocks.append(format_memory_block(hits))
        return "\n".join(blocks) if blocks else "这位用户目前没有可用的历史偏好记录。"

    return [search_products, get_product, compare_products, get_reviews, recall_preferences]

# -*- coding: utf-8 -*-
"""轻量检索：BM25 + 中文二元切分。

为什么不用向量库：这个工程包的目标是「解压之后一条命令就能跑」。多一个向量库
就多一个安装步骤，多一个 embedding 服务就多一个外部依赖。商品和政策加起来
几十条文本，BM25 和向量检索的排序结果差别很小。

中文没有天然的空格，这里用**二元切分**代替词典分词：把「主动降噪耳机」切成
主动/动降/降噪/噪耳/耳机。它不需要维护词典，对短语匹配反而更稳。

哪天真需要语义检索，把 `retrieve()` 换掉就行，调用方不用动。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Iterable

_HAN = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文二元切分 + 英文数字按词切。"""
    out: list[str] = []
    for seg in _HAN.findall(text):
        if len(seg) == 1:
            out.append(seg)
        else:
            out.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    out.extend(t.lower() for t in _ALNUM.findall(text))
    return out


class BM25Index:
    """标准 BM25，k1 与 b 用通用默认值。

    这个量级的数据调参没有意义，把参数暴露出来只是为了不让它变成黑盒。
    """

    def __init__(self, docs: Iterable[tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.token_sets: list[Counter] = []
        self.lengths: list[int] = []
        for doc_id, text in docs:
            self.ids.append(doc_id)
            self.texts.append(text)
            toks = tokenize(text)
            self.token_sets.append(Counter(toks))
            self.lengths.append(len(toks) or 1)
        self.k1, self.b = k1, b
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0

        # 文档频率
        self.df: Counter = Counter()
        for ts in self.token_sets:
            for tok in ts:
                self.df[tok] += 1
        self.n = len(self.ids)

    def _idf(self, tok: str) -> float:
        df = self.df.get(tok, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5,
               where: Callable[[str], bool] | None = None) -> list[tuple[str, float]]:
        """返回 (doc_id, 分数) 按分数降序。

        `where` 是字段过滤：先在结构化条件上过滤，再在文本上排序。
        真实检索系统里这一步叫 filter，它和排序是两件事，混在一起会
        让「按类目过滤」这种事变得很别扭。
        """
        q_toks = tokenize(query)
        if not q_toks:
            allowed = [i for i in self.ids if where is None or where(i)]
            return [(i, 0.0) for i in allowed[:top_k]]

        scored: list[tuple[str, float]] = []
        for idx, doc_id in enumerate(self.ids):
            if where is not None and not where(doc_id):
                continue
            ts, dl = self.token_sets[idx], self.lengths[idx]
            score = 0.0
            for tok in q_toks:
                tf = ts.get(tok, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                score += self._idf(tok) * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:top_k]


def product_text(p) -> str:
    """把一个商品拼成可检索的文本。

    权重靠**重复**表达：标题写两遍，卖点写两遍，规格写一遍。
    BM25 里词频有上限（k1 压着），但出现在多个字段里的词仍然会占优，
    这比给每个字段手调权重简单，效果也够用。
    """
    specs = " ".join(f"{k}{v}" for k, v in p.specs.items())
    return " ".join([
        p.title, p.title,
        p.brand, p.category, p.subcategory,
        " ".join(p.tags), " ".join(p.tags),
        " ".join(p.highlights), " ".join(p.highlights),
        specs,
    ])


def policy_text(p) -> str:
    return " ".join([p.title, p.title, p.scope, p.scope, p.content, " ".join(p.tags)])

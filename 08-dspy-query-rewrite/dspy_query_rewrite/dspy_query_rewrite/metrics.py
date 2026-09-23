# -*- coding: utf-8 -*-
"""评分函数（metric）：优化器的一切优化都对着它来。

注意：metric 决定编译质量的上限。产线中应尽量用线上可验证信号
（点击、成交、人工抽检）替换此处的简化判分。
"""
from __future__ import annotations


def rewrite_metric(gold, pred, trace=None) -> bool:
    """query 改写任务的判分。

    规则：
    1. 意图必须完全一致；
    2. 改写结果需与标注结果有核心词交集（简化判分，不引入分词）。
    """
    if gold.intent != pred.intent:
        return False
    gold_words = set(gold.rewritten.replace(" ", "").split())
    pred_words = set(pred.rewritten.replace(" ", "").split())
    return len(gold_words & pred_words) >= 1

# -*- coding: utf-8 -*-
"""Signature 定义：用输入/输出契约描述任务，零手写 prompt。"""
import dspy


class QueryRewrite(dspy.Signature):
    """把用户口语化的搜索词改写成规范商品查询词，并识别意图。"""

    raw_query: str = dspy.InputField(desc="用户输入的原始搜索词，可能带口语与错别字")
    rewritten: str = dspy.OutputField(desc="改写后的规范搜索词，用于召回")
    intent: str = dspy.OutputField(desc="用户意图，只能是 buy / compare / browse")

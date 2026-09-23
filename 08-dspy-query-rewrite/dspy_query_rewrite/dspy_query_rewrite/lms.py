# -*- coding: utf-8 -*-
"""LM 工厂：按配置构建 dummy（零成本）或真实 LLM，并完成 dspy.configure。"""
from __future__ import annotations

import os

import dspy
from dspy.utils import DummyLM

from .config import LMConfig


def _build_response_pool(examples, repeat: int = 200) -> list[dict]:
    """从示例集构造 DummyLM 的顺序响应池。

    每个示例对应一条期望输出，整池重复 repeat 轮。
    这样编译期 teacher 按样本顺序调用时能稳定命中期望答案。
    """
    pool = []
    for ex in examples:
        pool.append({"rewritten": ex.rewritten, "intent": ex.intent})
    return pool * max(repeat, 1)


def build_dummy_lm(examples, repeat: int = 200) -> DummyLM:
    adapter = dspy.ChatAdapter()
    lm = DummyLM(_build_response_pool(examples, repeat), adapter=adapter)
    return lm


def build_lookup_lm(examples) -> DummyLM:
    """按输入内容命中的 DummyLM（dict 模式）。

    key 为原始 query 全文，命中条件：key 出现在最后一条用户消息中。
    适合 infer 演示：query 若在 trainset 中即返回其期望输出。
    """
    adapter = dspy.ChatAdapter()
    answers = {
        ex.raw_query: {"rewritten": ex.rewritten, "intent": ex.intent}
        for ex in examples
    }
    return DummyLM(answers, adapter=adapter)


def build_real_lm(cfg: LMConfig) -> dspy.LM:
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"未找到环境变量 {cfg.api_key_env}，请先 export 你的 API Key。"
            "生产环境凭据一律走环境变量，禁止写入代码或配置文件。"
        )
    return dspy.LM(cfg.model, api_key=api_key, temperature=cfg.temperature)


def setup_lm(cfg: LMConfig, examples=None) -> dspy.LM:
    """按配置构 LM 并全局生效，返回实例。

    examples 仅在 dummy 模式需要：用于构建响应池，建议传 trainset。
    """
    if cfg.provider == "dummy":
        lm = build_dummy_lm(examples or [], repeat=cfg.dummy_repeat)
    elif cfg.provider == "openai":
        lm = build_real_lm(cfg)
    else:
        raise ValueError(f"不支持的 lm.provider: {cfg.provider}（支持 dummy / openai）")
    dspy.configure(lm=lm)
    return lm

# -*- coding: utf-8 -*-
"""优化器工厂 + 编译：把优化器选型收敛到一个函数，换优化器只改配置。"""
from __future__ import annotations

import dspy

from .config import OptimizerConfig
from .metrics import rewrite_metric

# 优化器名 -> 构造工厂（参数均来自配置，dummy 模式请用 BootstrapFewShot）
_OPTIMIZERS = {
    "BootstrapFewShot": lambda cfg: dspy.BootstrapFewShot(
        metric=rewrite_metric,
        max_bootstrapped_demos=cfg.max_bootstrapped_demos,
        max_labeled_demos=cfg.max_labeled_demos,
    ),
    "GEPA": lambda cfg: dspy.GEPA(metric=rewrite_metric),
    "MIPROv2": lambda cfg: dspy.MIPROv2(metric=rewrite_metric),
    "SIMBA": lambda cfg: dspy.SIMBA(metric=rewrite_metric),
}


def build_optimizer(cfg: OptimizerConfig):
    if cfg.name not in _OPTIMIZERS:
        raise ValueError(f"不支持的优化器: {cfg.name}（可选 {list(_OPTIMIZERS)}）")
    return _OPTIMIZERS[cfg.name](cfg)


def compile_program(cfg: OptimizerConfig, trainset, predictor=None):
    """编译并返回编译后的 Predict 模块。

    predictor 不传则默认 dspy.Predict(QueryRewrite)。
    """
    optimizer = build_optimizer(cfg)
    if predictor is None:
        from .signatures import QueryRewrite

        predictor = dspy.Predict(QueryRewrite)
    return optimizer.compile(predictor, trainset=trainset)

# -*- coding: utf-8 -*-
"""编译 + 产物落盘的核心流程。"""
from __future__ import annotations

import json
from pathlib import Path

import dspy

from .config import Config
from .data import load_examples
from .lms import setup_lm
from .optimizers import compile_program
from .signatures import QueryRewrite


def save_compiled(compiled, path: str | Path) -> Path:
    """编译产物是纯数据，save 为 json。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(p))
    return p


def load_compiled(path: str | Path) -> dspy.Predict:
    """load 编译产物为普通 Predict，运行期零优化开销。

    注意 dspy 的 load() 是就地修改并返回 None，需先实例化再 load。
    """
    predictor = dspy.Predict(QueryRewrite)
    predictor.load(str(path))
    return predictor


def describe_artifact(path: str | Path) -> dict:
    """读取产物 json，返回结构摘要（顶层 keys、demos 数量与样例）。"""
    p = Path(path)
    artifact = json.loads(p.read_text(encoding="utf-8"))
    demos = artifact.get("demos", []) or []
    return {
        "path": str(p),
        "top_keys": list(artifact.keys()),
        "demo_count": len(demos),
        "demo_sample": demos[:2],
    }


def run_pipeline(cfg: Config, train_path=None, artifact_path=None) -> dspy.Predict:
    """一条龙：配置 LM -> 载数据 -> 编译 -> 存产物。返回编译后的模块。"""
    train_path = train_path or cfg.resolve(cfg.paths.trainset)
    artifact_path = artifact_path or cfg.resolve(cfg.paths.compiled_artifact)

    trainset = load_examples(train_path)

    # dummy 模式必须用 trainset 构建响应池；openai 模式无需
    lm = setup_lm(cfg.lm, examples=trainset)
    print(f"[lm] provider={cfg.lm.provider} model={getattr(lm, 'model', 'DummyLM')}")

    compiled = compile_program(cfg.optimizer, trainset)
    save_compiled(compiled, artifact_path)
    print(f"[compile] 优化器 {cfg.optimizer.name} 完成，产物 -> {artifact_path}")
    return compiled

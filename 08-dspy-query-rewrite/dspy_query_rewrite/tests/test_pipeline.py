# -*- coding: utf-8 -*-
"""端到端测试：DummyLM 零成本跑通完整链路。

运行: python -m pytest tests/ -v
"""
import json
from pathlib import Path

import dspy

from dspy_query_rewrite.config import Config
from dspy_query_rewrite.data import load_examples
from dspy_query_rewrite.distill import demos_to_sft
from dspy_query_rewrite.lms import build_lookup_lm, setup_lm
from dspy_query_rewrite.pipeline import describe_artifact, load_compiled, run_pipeline

ROOT = Path(__file__).resolve().parent.parent


def _cfg() -> Config:
    return Config.load(ROOT / "config" / "config.yaml")


def test_pipeline_compile_distill(tmp_path):
    cfg = _cfg()
    trainset = load_examples(cfg.resolve(cfg.paths.trainset))
    assert len(trainset) >= 3, "trainset 至少 3 条"

    # dummy 编译（使用临时产物路径）
    artifact = tmp_path / "qr.json"
    run_pipeline(cfg, artifact_path=artifact)

    # 产物结构与 demos
    summary = describe_artifact(artifact)
    assert "demos" in summary["top_keys"]
    assert summary["demo_count"] >= 1, "编译产物应含至少 1 条自动挑选的 few-shot"

    # 蒸馏导出
    ft = tmp_path / "ft.jsonl"
    rows = demos_to_sft(artifact, ft)
    assert len(rows) >= 1
    first = rows[0]
    assert first["messages"][0]["role"] == "user"
    assert first["messages"][1]["role"] == "assistant"

    # load 产物回跑
    compiled = load_compiled(artifact)
    assert compiled is not None


def test_infer_lookup(tmp_path):
    """dummy lookup 模式：query 命中 trainset 即返回期望输出。"""
    cfg = _cfg()
    trainset = load_examples(cfg.resolve(cfg.paths.trainset))

    artifact = tmp_path / "qr.json"
    run_pipeline(cfg, artifact_path=artifact)
    compiled = load_compiled(artifact)

    # run_pipeline 内部会把 lm 重配为 list 池，推理前需再设 lookup 模式
    dspy.configure(lm=build_lookup_lm(trainset))

    q = trainset[0].raw_query
    pred = compiled(raw_query=q)
    assert pred.rewritten == trainset[0].rewritten
    assert pred.intent == trainset[0].intent


def test_config_resolve():
    cfg = Config.load()
    p = cfg.resolve(cfg.paths.trainset)
    assert p.is_absolute()
    assert p.exists(), "data/trainset.jsonl 应存在于项目内"

# -*- coding: utf-8 -*-
"""推理：load 编译产物后对单条 query 做改写。"""
from __future__ import annotations

from .config import Config
from .pipeline import load_compiled


def infer_one(cfg: Config, query: str, artifact_path=None) -> dict:
    artifact_path = artifact_path or cfg.resolve(cfg.paths.compiled_artifact)
    program = load_compiled(artifact_path)
    pred = program(raw_query=query)
    return {"raw_query": query, "rewritten": pred.rewritten, "intent": pred.intent}

# -*- coding: utf-8 -*-
"""评测：baseline（未编译）vs compiled（编译后）在留出集上的对比。"""
from __future__ import annotations

import dspy

from .config import Config
from .data import load_examples
from .metrics import rewrite_metric
from .pipeline import load_compiled
from .signatures import QueryRewrite


def _score(predictor, examples) -> dict:
    correct = 0
    detail = []
    for ex in examples:
        pred = predictor(raw_query=ex.raw_query)
        ok = rewrite_metric(ex, pred)
        correct += 1 if ok else 0
        detail.append(
            {
                "query": ex.raw_query,
                "gold": ex.rewritten,
                "gold_intent": ex.intent,
                "pred": pred.rewritten,
                "pred_intent": pred.intent,
                "ok": ok,
            }
        )
    return {"score": correct / max(len(examples), 1), "detail": detail}


def run_eval(cfg: Config, dev_path=None, artifact_path=None) -> dict:
    dev_path = dev_path or cfg.resolve(cfg.paths.devset)
    artifact_path = artifact_path or cfg.resolve(cfg.paths.compiled_artifact)

    devset = load_examples(dev_path)

    baseline = dspy.Predict(QueryRewrite)
    compiled = load_compiled(artifact_path)

    base_res = _score(baseline, devset)
    comp_res = _score(compiled, devset)
    return {
        "baseline": base_res,
        "compiled": comp_res,
        "dev_count": len(devset),
    }

# -*- coding: utf-8 -*-
"""蒸馏导出：编译产物 -> 可喂给微调的 SFT 语料。

两种模式：
1. demos 蒸馏：把编译产物里优化器自动挑的 few-shot demos 转成 messages 格式 JSONL；
2. 规模化打标：对无标注 query 批量调用编译后的程序生成改写结果，产出更大训练集。
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .data import save_records
from .pipeline import load_compiled


def demos_to_sft(artifact_path: str | Path, output_path: str | Path) -> list[dict]:
    """把编译产物 json 的 demos 转成 SFT JSONL（messages 对话格式）。

    返回生成的记录列表，同时落盘到 output_path。
    """
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    demos = artifact.get("demos", []) or []

    rows = []
    for d in demos:
        raw_query = d.get("raw_query")
        rewritten = d.get("rewritten")
        intent = d.get("intent")
        if not raw_query or not rewritten:
            continue
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": raw_query},
                    {"role": "assistant", "content": f"{rewritten}（意图：{intent}）"},
                ]
            }
        )
    save_records(rows, output_path)
    return rows


def batch_label(
    cfg: Config,
    queries: list[str],
    output_path: str | Path,
    artifact_path: str | Path | None = None,
) -> list[dict]:
    """规模化蒸馏：用编译产物对无标注 query 批量生成改写，得到更大训练集。

    注意：该模式需要真实 LM（dummy 响应池与 query 无关，仅用于链路验证）。
    """
    artifact_path = artifact_path or cfg.resolve(cfg.paths.compiled_artifact)
    program = load_compiled(artifact_path)

    rows = []
    for q in queries:
        pred = program(raw_query=q)
        rows.append({"raw_query": q, "rewritten": pred.rewritten, "intent": pred.intent})

    # 同时保留原始产物形态 + SFT 形态
    save_records(rows, output_path)
    sft = [
        {
            "messages": [
                {"role": "user", "content": r["raw_query"]},
                {"role": "assistant", "content": f"{r['rewritten']}（意图：{r['intent']}）"},
            ]
        }
        for r in rows
    ]
    save_records(sft, str(output_path).replace(".jsonl", "_sft.jsonl"))
    return rows

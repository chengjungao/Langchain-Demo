# -*- coding: utf-8 -*-
"""数据加载：把 data/*.jsonl 转成 dspy.Example / 原生 dict 列表。"""
from __future__ import annotations

import json
from pathlib import Path

import dspy


def load_records(path: str | Path) -> list[dict]:
    """读取 jsonl，每行一个 dict。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")
    records = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_examples(records: list[dict]) -> list[dspy.Example]:
    """dict 列表 -> dspy.Example 列表，raw_query 为输入字段。"""
    return [dspy.Example(**r).with_inputs("raw_query") for r in records]


def load_examples(path: str | Path) -> list[dspy.Example]:
    return to_examples(load_records(path))


def save_records(records: list[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p

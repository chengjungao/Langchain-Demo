# -*- coding: utf-8 -*-
"""配置加载：读取 config/config.yaml，提供类型化的默认值。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 项目根目录（config.py 位于 dspy_query_rewrite/ 包内，其上两级为项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LMConfig:
    provider: str = "dummy"          # dummy | openai
    model: str = "openai/gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    # dummy 模式下响应池轮数（越大越稳，防止编译中途耗尽）
    dummy_repeat: int = 200


@dataclass
class OptimizerConfig:
    name: str = "BootstrapFewShot"   # BootstrapFewShot | GEPA | MIPROv2 | SIMBA
    max_bootstrapped_demos: int = 3
    max_labeled_demos: int = 3


@dataclass
class PathConfig:
    trainset: Path = Path("data/trainset.jsonl")
    devset: Path = Path("data/devset.jsonl")
    compiled_artifact: Path = Path("artifacts/query_rewrite.json")
    finetune_output: Path = Path("artifacts/finetune_data.jsonl")
    batch_output: Path = Path("artifacts/batch_labels.jsonl")


@dataclass
class Config:
    lm: LMConfig = field(default_factory=LMConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        lm_raw = raw.get("lm", {})
        opt_raw = raw.get("optimizer", {})
        p_raw = raw.get("paths", {})

        cfg = cls(
            lm=LMConfig(**{k: v for k, v in lm_raw.items() if hasattr(LMConfig, k)}),
            optimizer=OptimizerConfig(**{k: v for k, v in opt_raw.items() if hasattr(OptimizerConfig, k)}),
            paths=PathConfig(**{k: Path(v) for k, v in p_raw.items() if hasattr(PathConfig, k)}),
        )
        return cfg

    def resolve(self, p: str | Path) -> Path:
        """把相对项目根的相对路径解析为绝对路径。"""
        p = Path(p)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

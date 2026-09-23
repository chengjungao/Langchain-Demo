# -*- coding: utf-8 -*-
"""技能加载。

技能是一段写给模型看的流程知识，存在 markdown 里。放在文件而不是塞进代码，
是因为这类内容改得比代码频繁：运营调整一次退款口径，不该走一次发布流程。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=16)
def load_skill(name: str) -> str:
    """读一个 SKILL.md，去掉文件头部的元信息，返回正文。"""
    p = ROOT / "skills" / name / "SKILL.md"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2]
    return text.strip()


def list_skills() -> list[str]:
    base = ROOT / "skills"
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir()
                  if d.is_dir() and (d / "SKILL.md").exists())

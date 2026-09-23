# -*- coding: utf-8 -*-
"""三条支线的 Agent 循环。装配在 `loop.py`，提示词在 `prompts.py`。"""
from .loop import build_all_loops, build_loop
from .prompts import ROLE_PROMPTS, build_system

__all__ = ["build_loop", "build_all_loops", "build_system", "ROLE_PROMPTS"]

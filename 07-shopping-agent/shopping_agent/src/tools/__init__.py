# -*- coding: utf-8 -*-
"""工具层。

分成三组，对应三条支线；`shelf` 负责按支线分发，是权限的落点。
"""
from .shelf import TOOL_GROUPS, WRITE_TOOLS, Shelf

__all__ = ["Shelf", "TOOL_GROUPS", "WRITE_TOOLS"]

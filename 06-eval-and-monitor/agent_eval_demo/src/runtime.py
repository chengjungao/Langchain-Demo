"""运行时小工具：断网保险、输出排版、离线确认。

`OutboundGuard` 是这份工程包里我最喜欢的一个小东西：它把「这次评测没有对外
发过一个请求」从一句口头承诺，变成一个可验证的事实。装上它之后，任何对外
连接都会当场抛错，本地回环放行。
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

LANGSMITH_ENV = (
    "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY",
    "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2",
    "LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT",
    "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT",
)


def force_offline() -> None:
    """把云端的凭据与开关从环境里撤掉，证明后面跑的一切都不依赖它。"""
    for k in LANGSMITH_ENV:
        os.environ.pop(k, None)


def banner(title: str, width: int = 84) -> None:
    print("=" * width)
    print(title)
    print("=" * width)


def section(title: str) -> None:
    print(f"\n【{title}】")


@contextmanager
def outbound_guard(allow_local: bool = True):
    """拦截对外网络。本地回环放行。

    出来以后 guard.blocked 就是被拦下的次数，可以直接打进报告。
    """
    real_connect = socket.socket.connect
    state = {"blocked": 0}

    def patched(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else str(address)
        if allow_local and host in LOCAL_HOSTS:
            return real_connect(self, address, *args, **kwargs)
        state["blocked"] += 1
        raise OSError(f"演示已断网，拒绝连接 {host}")

    socket.socket.connect = patched
    guard = _Guard(state)
    try:
        yield guard
    finally:
        socket.socket.connect = real_connect


class _Guard:
    def __init__(self, state: dict):
        self._state = state

    @property
    def blocked(self) -> int:
        return self._state["blocked"]

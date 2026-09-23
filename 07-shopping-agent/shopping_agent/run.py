#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""启动入口。

    python run.py                 # 启动服务并打开浏览器
    python run.py --port 9000     # 换个端口
    python run.py --no-browser    # 不自动开浏览器
    python run.py --check         # 只做自检：数据、依赖、模型连通性

Windows 上如果 python 找不到，试 py run.py。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def check() -> int:
    """自检。比「跑到一半报错」友好，也让人一眼看清缺什么。"""
    print("\n购物助手 Agent · 自检")
    print("-" * 58)
    problems = 0

    # 1. 依赖
    missing = []
    for mod, pkg in (("langgraph", "langgraph"), ("langchain_openai", "langchain-openai"),
                     ("pydantic", "pydantic")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        problems += 1
        print(f"[x] 缺依赖：{', '.join(missing)}")
        print(f"    执行：pip install -r requirements.txt")
    else:
        print("[√] 依赖齐全")

    # 2. 数据
    try:
        from src.dataset import get_catalog
        cat = get_catalog()
        s = cat.stats()
        print(f"[√] 数据加载：{'、'.join(f'{k} {v}' for k, v in s.items())}")
        from eval.validate_data import run_checks
        ok, report = run_checks()
        print(f"[{'√' if ok else 'x'}] 数据自洽校验：{report}")
        problems += 0 if ok else 1
    except Exception as e:
        problems += 1
        print(f"[x] 数据有问题：{type(e).__name__}: {e}")

    # 3. 模型
    try:
        from src.brain import probe
        from src.config import get_config
        cfg = get_config()
        if not cfg.is_ready():
            print("[!] 还没配置模型，会以「内置规则」模式运行（功能可跑，回答质量有限）")
            print("    在页面右上角「设置」里填地址和密钥，或者写进 config.json")
        else:
            r = probe(cfg)
            print(f"[{'√' if r['ok'] else 'x'}] 模型连接：{r['message']}")
            if r.get("models"):
                print(f"    可用模型：{', '.join(r['models'][:6])}"
                      + ("…" if len(r["models"]) > 6 else ""))
            problems += 0 if r["ok"] else 1
    except Exception as e:
        problems += 1
        print(f"[x] 模型检查失败：{type(e).__name__}: {e}")

    print("-" * 58)
    print("自检通过，可以 python run.py 启动了。" if not problems
          else f"有 {problems} 处需要处理。")
    return 0 if not problems else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="购物助手 Agent")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址")
    ap.add_argument("--port", type=int, default=8765, help="监听端口")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--check", action="store_true", help="只做自检")
    args = ap.parse_args()

    if args.check:
        sys.exit(check())

    from src.server import serve
    serve(args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()

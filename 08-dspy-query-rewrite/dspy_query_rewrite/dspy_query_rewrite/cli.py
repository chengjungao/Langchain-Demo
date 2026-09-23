# -*- coding: utf-8 -*-
"""命令行入口。

子命令：
  demo     一条龙演示：编译 -> 蒸馏 -> 推理（默认 DummyLM 零成本）
  compile  编译并保存产物
  distill  从产物导出 SFT JSONL（demos 蒸馏）
  batch    对无标注 query 批量打标（规模化蒸馏，需真实 LM）
  infer    对单条 query 推理
  eval     留出集评测：baseline vs compiled
"""
from __future__ import annotations

import argparse
import sys

import dspy

from . import __version__
from .config import Config
from .data import load_examples, load_records, save_records
from .distill import batch_label, demos_to_sft
from .eval import run_eval
from .infer import infer_one
from .lms import build_lookup_lm, setup_lm
from .pipeline import describe_artifact, run_pipeline

DIVIDER = "=" * 62


def _print_divider(title: str) -> None:
    print(f"\n{DIVIDER}\n{title}\n{DIVIDER}")


def cmd_demo(args) -> int:
    """一条龙演示（默认 config.yaml 的 provider；dummy 零成本跑通全链路）。"""
    cfg = Config.load(args.config)
    print(f"dspy-query-rewrite v{__version__} | provider={cfg.lm.provider} | optimizer={cfg.optimizer.name}")

    artifact_path = cfg.resolve(cfg.paths.compiled_artifact)
    ft_path = cfg.resolve(cfg.paths.finetune_output)

    # 1. 编译
    _print_divider("步骤 1/4：编译（优化器自动挑 few-shot）")
    run_pipeline(cfg, artifact_path=artifact_path)

    # 2. 产物结构
    _print_divider("步骤 2/4：编译产物结构")
    summary = describe_artifact(artifact_path)
    print("顶层 keys:", summary["top_keys"])
    print(f"demos 数量: {summary['demo_count']}")
    for d in summary["demo_sample"]:
        print("  demo ->", d)

    # 3. 蒸馏
    _print_divider("步骤 3/4：蒸馏 demos -> SFT JSONL")
    rows = demos_to_sft(artifact_path, ft_path)
    print(f"已导出 {len(rows)} 条 -> {ft_path}")
    if rows:
        print("首条样例:", rows[0]["messages"])

    # 4. 推理（demo 用 trainset 首条 query；dummy lookup 按内容命中期望输出）
    _print_divider("步骤 4/4：load 产物回跑推理")
    trainset = load_examples(cfg.resolve(cfg.paths.trainset))
    demo_query = trainset[0].raw_query
    if cfg.lm.provider == "dummy":
        dspy.configure(lm=build_lookup_lm(trainset))
    else:
        setup_lm(cfg.lm, examples=trainset)
    res = infer_one(cfg, demo_query, artifact_path=artifact_path)
    print(f"query  : {res['raw_query']}")
    print(f"改写   : {res['rewritten']}")
    print(f"意图   : {res['intent']}")

    _print_divider("完成")
    print("产物与蒸馏数据已落地。真实使用请改 config.yaml: lm.provider=openai 并 export API Key。")
    return 0


def cmd_compile(args) -> int:
    cfg = Config.load(args.config)
    run_pipeline(cfg)
    return 0


def cmd_distill(args) -> int:
    cfg = Config.load(args.config)
    rows = demos_to_sft(
        cfg.resolve(args.artifact or cfg.paths.compiled_artifact),
        cfg.resolve(args.output or cfg.paths.finetune_output),
    )
    print(f"已导出 {len(rows)} 条 SFT 样本")
    return 0


def cmd_batch(args) -> int:
    """规模化蒸馏：无标注 query -> 编译程序 -> 打标数据（需真实 LM）。"""
    cfg = Config.load(args.config)
    if cfg.lm.provider != "openai":
        print("提示: batch 需真实 LM 才有语义。请先 config.yaml 配 lm.provider=openai 并 export API Key。", file=sys.stderr)
        return 2
    records = load_records(cfg.resolve(args.queries))
    queries = [r["raw_query"] if "raw_query" in r else r["query"] for r in records]
    setup_lm(cfg.lm)  # dummy 模式下仅链路演示，真实语义需 openai
    rows = batch_label(cfg, queries, cfg.resolve(args.output or cfg.paths.batch_output))
    print(f"已打标 {len(rows)} 条 -> {cfg.paths.batch_output}")
    return 0


def cmd_infer(args) -> int:
    cfg = Config.load(args.config)
    trainset = load_examples(cfg.resolve(cfg.paths.trainset))
    if cfg.lm.provider == "dummy":
        # lookup 模式：query 命中 trainset 即返回期望输出；否则回退 No more responses
        dspy.configure(lm=build_lookup_lm(trainset))
    else:
        setup_lm(cfg.lm, examples=trainset)
    artifact_path = cfg.resolve(args.artifact or cfg.paths.compiled_artifact)
    res = infer_one(cfg, args.query, artifact_path=artifact_path)
    print(f"raw_query : {res['raw_query']}")
    print(f"rewritten : {res['rewritten']}")
    print(f"intent    : {res['intent']}")
    return 0


def cmd_eval(args) -> int:
    cfg = Config.load(args.config)
    # dummy 模式：devset 按序命中，仅用于链路演示；真实效果需 openai provider
    devset = load_examples(cfg.resolve(cfg.paths.devset))
    setup_lm(cfg.lm, examples=devset)
    res = run_eval(cfg)
    print(f"留出集样本数: {res['dev_count']}")
    print(f"{'指标':<10}{'baseline':<12}{'compiled':<12}")
    for name, key in (("意图+改写准确率", "score"),):
        b = res["baseline"][key]
        c = res["compiled"][key]
        print(f"{name:<10}{b:<12.2%}{c:<12.2%}")
    print("\n逐条对比（compiled）：")
    for item in res["compiled"]["detail"]:
        mark = "✓" if item["ok"] else "✗"
        print(f"  [{mark}] {item['query']} -> {item['pred']} / {item['pred_intent']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dspy-query-rewrite",
        description="DSPy 优化器 + 蒸馏实战：电商 query 改写与意图分类",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="一条龙演示（零成本默认）")
    p_demo.add_argument("--config", default=None, help="配置文件路径")
    p_demo.set_defaults(func=cmd_demo)

    p_compile = sub.add_parser("compile", help="编译并保存产物")
    p_compile.add_argument("--config", default=None)
    p_compile.set_defaults(func=cmd_compile)

    p_distill = sub.add_parser("distill", help="demos -> SFT JSONL")
    p_distill.add_argument("--config", default=None)
    p_distill.add_argument("--artifact", default=None, help="编译产物 json 路径")
    p_distill.add_argument("--output", default=None, help="输出 jsonl 路径")
    p_distill.set_defaults(func=cmd_distill)

    p_batch = sub.add_parser("batch", help="无标注 query 批量打标（规模化蒸馏）")
    p_batch.add_argument("--config", default=None)
    p_batch.add_argument("--queries", required=True, help="query 列表 jsonl（字段 raw_query 或 query）")
    p_batch.add_argument("--output", default=None)
    p_batch.set_defaults(func=cmd_batch)

    p_infer = sub.add_parser("infer", help="单条 query 推理")
    p_infer.add_argument("--config", default=None)
    p_infer.add_argument("--artifact", default=None)
    p_infer.add_argument("--query", required=True, help="原始搜索词")
    p_infer.set_defaults(func=cmd_infer)

    p_eval = sub.add_parser("eval", help="留出集评测 baseline vs compiled")
    p_eval.add_argument("--config", default=None)
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

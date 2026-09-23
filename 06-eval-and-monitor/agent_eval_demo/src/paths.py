"""统一路径。所有脚本都从这里取路径，不各写各的。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SRC = ROOT / "src"
DATA = ROOT / "data"
DEMOS = ROOT / "demos"
TESTS = ROOT / "tests"
REPORTS = ROOT / "reports"
CASSETTES = ROOT / "cassettes"

EVAL_SET = DATA / "eval_set.jsonl"
EVAL_SET_FROZEN = DATA / "eval_set.frozen.json"
BASELINE = DATA / "baseline.json"

TRACES = REPORTS / "traces.jsonl"
CURRENT = REPORTS / "current.json"


def ensure_dirs() -> None:
    for d in (DATA, REPORTS, CASSETTES):
        d.mkdir(parents=True, exist_ok=True)

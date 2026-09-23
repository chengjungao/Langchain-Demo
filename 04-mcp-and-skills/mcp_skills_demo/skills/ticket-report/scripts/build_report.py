# -*- coding: utf-8 -*-
"""工单周报数据统计（只读）。

用法：
    python skills/ticket-report/scripts/build_report.py --days 7
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta

# 演示数据。换成真实数据源只需替换这个函数。
_ROWS = [
    {"ticket": "RF000101", "created": 0, "reason": "quality", "hours": 3.5, "closed": True},
    {"ticket": "RF000102", "created": 1, "reason": "no_longer_needed", "hours": 20.0, "closed": True},
    {"ticket": "RF000103", "created": 1, "reason": "quality", "hours": 2.0, "closed": True},
    {"ticket": "RF000104", "created": 2, "reason": "wrong_item", "hours": 30.0, "closed": True},
    {"ticket": "RF000105", "created": 3, "reason": "price_protection", "hours": 5.0, "closed": False},
    {"ticket": "RF000106", "created": 4, "reason": "quality", "hours": 1.5, "closed": True},
    {"ticket": "RF000107", "created": 5, "reason": "no_longer_needed", "hours": 26.5, "closed": True},
]


def collect(days: int = 7, now: datetime | None = None) -> dict:
    now = now or datetime(2026, 9, 12, 18, 0, 0)
    since = now - timedelta(days=days)
    rows = [r for r in _ROWS if (now - timedelta(days=r["created"])) >= since]

    reasons = Counter(r["reason"] for r in rows)
    total = sum(reasons.values()) or 1
    overdue = [r for r in rows if r["hours"] > 24.0 and not r["closed"]]

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "window_days": days,
        "total": len(rows),
        "by_reason": [
            {"reason": k, "count": v, "ratio": round(v / total, 3)}
            for k, v in reasons.most_common()
        ],
        "overdue": [{"ticket": r["ticket"], "hours": r["hours"]} for r in overdue],
        "avg_hours": round(sum(r["hours"] for r in rows) / len(rows), 2) if rows else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(collect(args.days), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

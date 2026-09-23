"""脱敏：上云之前把不能出去的东西拦住。

官方给了 `create_secret_anonymizer()`，它认一批主流密钥形态。本模块在它之上
补两层，因为实测下来有两类东西它不管：

  1. **国内云厂商的凭据不认。** 阿里云的 `LTAI` 开头、腾讯云的 `AKID` 开头都不在
     默认规则里。这不是官方的疏漏，是它的规则表按国际厂商写的。国内团队上云，
     这正好是第一条要自己补的规则。
  2. **业务字段一个都不管。** 手机号、身份证、订单号、银行卡，默认规则里没有。

另外两个实测出来的高风险行为，写进了代码：

  * `create_anonymizer(rules)` **会原地改动传进去的对象**，返回的也是同一个对象，
    深层嵌套一样改。生产里这意味着你为了上报而脱敏，结果把内存里的原始业务数据
    一起改了。所以 redact() 默认先深拷贝。
  * `max_depth` 设小了会**静默漏字段** —— 超出深度的值原样带出去，不报错。
    所以这里默认给足深度。

规则里那个键叫 `replace`。写成 `replacement` 不报错，pattern 照样匹配，但你写的
掩码文案会被丢掉，输出统一变成固定的 `[redacted]`。你会看到脱敏生效了，不会发现
文案是错的。另外规则之间会互相干扰，窄的规则要放前面。
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from langsmith.anonymizer import (  # noqa: F401  （DEFAULT_SECRET_RULES 供查阅）
    DEFAULT_SECRET_RULES,
    create_anonymizer,
    create_secret_anonymizer,
)

# ---------------------------------------------------------------- 国内云厂商
# 这几条官方默认规则里没有，必须自己补。
CN_CLOUD_RULES = [
    {"pattern": r"LTAI[A-Za-z0-9]{12,}", "replace": "[阿里云AccessKey]"},
    {"pattern": r"AKID[A-Za-z0-9]{13,}", "replace": "[腾讯云SecretId]"},
    {"pattern": r"(?i)\b(?:access|secret|api)[_-]?(?:key|id|secret)\b\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{8,}[\"']?",
     "replace": "[凭据]"},  # 带键名的通用兜底，覆盖没有固定前缀的厂商
]

# ---------------------------------------------------------------- 业务字段
BUSINESS_RULES = [
    {"pattern": r"1[3-9]\d{9}", "replace": "[手机号]"},
    {"pattern": r"\d{17}[\dXx]", "replace": "[身份证号]"},
    {"pattern": r"[\w.+-]+@[\w-]+\.[\w.]+", "replace": "[邮箱]"},
    {"pattern": r"\b\d{16,19}\b", "replace": "[银行卡号]"},
]

# 订单号这类只在特定业务里算敏感。放开会误伤，所以单独成一档。
ORDER_ID_RULES = [
    {"pattern": r"\b[A-Z]\d{4,}\b", "replace": "[订单号]"},
]

LEVELS = {
    "secret": [],                 # 只靠官方默认规则
    "cn": CN_CLOUD_RULES,         # + 国内云厂商
    "business": CN_CLOUD_RULES + BUSINESS_RULES,
    "strict": CN_CLOUD_RULES + BUSINESS_RULES + ORDER_ID_RULES,
}

DEFAULT_DEPTH = 24


def build_anonymizer(level: str = "business", max_depth: int = DEFAULT_DEPTH):
    """造一个脱敏函数。

    level: secret / cn / business / strict
    """
    if level not in LEVELS:
        raise ValueError(f"level 必须是 {sorted(LEVELS)} 之一，收到 {level!r}")
    extra = LEVELS[level]
    if not extra:
        return create_secret_anonymizer(max_depth=max_depth)
    return create_secret_anonymizer(extra_rules=list(extra), max_depth=max_depth)


def redact(obj: Any, anonymizer=None, level: str = "business", inplace: bool = False) -> Any:
    """脱敏一个对象。

    默认先深拷贝，不动原始对象。这是刻意的：官方那个会原地改，脱敏顺手把内存里
    的业务数据也改了，这种 bug 很难查。
    """
    anon = anonymizer or build_anonymizer(level)
    target = obj if inplace else copy.deepcopy(obj)
    return anon(target)


class Redactor:
    """带统计的脱敏器，方便在 CI 里断言「敏感字段命中数归零」。"""

    def __init__(self, level: str = "business", max_depth: int = DEFAULT_DEPTH):
        self.level = level
        self.anonymizer = build_anonymizer(level, max_depth=max_depth)
        self.calls = 0

    def apply(self, obj: Any, inplace: bool = False) -> Any:
        self.calls += 1
        return redact(obj, anonymizer=self.anonymizer, inplace=inplace)

    def leaks(self, before: Any, after: Any) -> list[str]:
        """检查 before 里的敏感片段有没有出现在 after 里。返回残留清单。"""
        text = _flatten(after)
        out = []
        for name, probe in PROBES.items():
            if probe in _flatten(before) and probe in text:
                out.append(name)
        return out


# ---------------------------------------------------------------- 探针
# 用来直观看出哪些形态被认、哪些不被认。
PROBES = {
    "LangSmith 平台密钥": "lsv2_pt_1a2b3c4d5e6f7890abcdef1234567890",
    "OpenAI 经典": "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn",
    "OpenAI 新版": "sk-proj-AbCdEf1234567890AbCdEf1234567890AbCdEf",
    "Anthropic": "sk-ant-api03-AbCdEfGh1234567890AbCdEfGh1234567890",
    "GitHub": "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "AWS AccessKey": "AKIAIOSFODNN7EXAMPLE",
    "阿里云 AccessKey": "LTAI5tAbCdEfGhIjKlMnOpQr",
    "腾讯云 SecretId": "AKIDAbCdEfGh1234567890AbCdEfGh1234567890",
    "手机号": "13812345678",
    "身份证号": "11010119900307123X",
    "邮箱": "someone@example.com",
}


def _flatten(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(_flatten(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return " ".join(_flatten(v) for v in obj)
    return str(obj)


def probe_table(level: str = "business") -> list[tuple[str, bool]]:
    """返回 [(形态, 是否被脱敏)]，用来做对照表。"""
    anon = build_anonymizer(level)
    out = []
    for name, val in PROBES.items():
        got = anon({"v": val})["v"]
        out.append((name, got != val))
    return out


def write_report(path: Path | str, level: str = "business") -> None:
    lines = [f"脱敏规则对照（level={level}）", ""]
    for name, masked in probe_table(level):
        lines.append(f"{name}\t{'脱敏' if masked else '原样带出'}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

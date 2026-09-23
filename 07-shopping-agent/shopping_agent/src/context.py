# -*- coding: utf-8 -*-
"""上下文治理：让长会话不把窗口撑爆。

一个会话聊到二十轮之后，如果每轮都把全部历史塞回去，会发生三件事：
token 成本线性上涨、模型开始忽略中间的内容、首字延迟越来越长。

治理的手段按代价从低到高排：

1. **裁剪** —— 只留最近 N 条。最便宜，但会丢信息。
2. **摘要** —— 把被裁掉的部分压成一段话。多花一次模型调用，但保住了结论。
3. **按需注入** —— 长期记忆只在相关时才带，不无条件全带上。

这里的策略是 1 + 2 组合：正常只裁剪，超过阈值才触发摘要，且摘要只在
被裁掉的消息足够多时才做（不然为省 3 条消息花一次调用，不划算）。
"""
from __future__ import annotations

import re

from .memory import MemoryHit

# 估算用的经验系数。中文一个字大约对应 0.7 个 token，
# 英文与数字大约 4 个字符 1 个 token。真实计数要调用分词器，
# 这里的用途是「判断该不该裁剪」，量级对就够了。
_CN = re.compile(r"[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cn = len(_CN.findall(text))
    other = len(text) - cn
    return int(cn * 0.7 + other / 4) + 1


def estimate_messages(messages: list[dict]) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def split_for_trim(messages: list[dict], keep: int) -> tuple[list[dict], list[dict]]:
    """返回 (要丢的, 要留的)。按轮次边界切，不把一问一答拆开。"""
    if len(messages) <= keep:
        return [], list(messages)
    # 从后往前按「用户消息」为边界，保证留下的以用户提问开头。
    cut = len(messages) - keep
    while cut < len(messages) and messages[cut].get("role") != "user":
        cut += 1
    if cut >= len(messages):
        cut = len(messages) - keep
    return messages[:cut], messages[cut:]


def need_summary(dropped: list[dict], min_tokens: int = 400) -> bool:
    """丢掉的内容够多才值得花一次调用做摘要。"""
    return estimate_messages(dropped) >= min_tokens


def compose_request_messages(history: list[dict], user_text: str,
                             summary: str = "", memory_hits: list[MemoryHit] | None = None,
                             keep: int = 12) -> tuple[list[dict], dict]:
    """把「历史 + 摘要 + 长期记忆 + 本轮输入」拼成真正发给模型的消息序列。

    返回值第二个是这次治理的动作记录，用于在界面上说明发生了什么 ——
    裁剪了多少条、带了几条记忆、摘要有没有被用上。
    """
    dropped, kept = split_for_trim(history, keep)
    report = {
        "history_total": len(history),
        "dropped": len(dropped),
        "kept": len(kept),
        "summary_used": bool(summary),
        "memory_injected": len(memory_hits or []),
        "est_tokens": 0,
    }
    out: list[dict] = []
    if summary:
        out.append({"role": "system", "content": f"这个会话早前的经过：{summary}"})
    out.extend(kept)
    out.append({"role": "user", "content": user_text})
    report["est_tokens"] = estimate_messages(out)
    return out, report


def compose_memory_note(hits: list[MemoryHit]) -> str:
    """长期记忆的注入块。空的时候返回空串，不要往提示词里塞空标题。"""
    if not hits:
        return ""
    lines = [f"- {h.label}：{h.text}" for h in hits]
    return "关于这位用户，之前记下的（若与当前说法冲突，以当前为准）：\n" + "\n".join(lines)

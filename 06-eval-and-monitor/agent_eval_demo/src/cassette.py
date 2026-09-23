"""回放缓存：让回归测试不烧钱。

把「模型调用」这一层抽出来，按 (模型标识 + system prompt + 用户输入) 的哈希落盘。
重跑评测时先查盘，命中就不再打网络。评测跑一百遍，也只花第一遍的钱。

三条纪律：

  1. 缓存 key 必须同时包含模型标识、system prompt、用户输入。少任何一项都会
     拿到过期答案，而且是静默的 —— 你只会看到分数变了，找不到原因。
  2. 缓存目录提交进仓库。CI 里就能零调用跑回归。官方对 LANGSMITH_TEST_CACHE
     也是这个建议。
  3. 缓存损坏或缺失时回退到真实调用，不要抛错。缓存是加速器，不是依赖。

它测不了什么也要说清楚：模型本身换了的时候，缓存命中意味着你根本没调模型，
新模型的表现自然看不出来。换模型那一版必须跑全量真实调用。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable


def cache_key(model_id: str, system_prompt: str, question: str) -> str:
    raw = f"{model_id}\x00{system_prompt}\x00{question}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class Cassette:
    """一个目录，一个 key 一个文件。够用，而且能直接 diff。"""

    def __init__(self, directory: Path | str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------ 读写
    def get(self, key: str) -> str | None:
        f = self.dir / f"{key}.json"
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))["answer"]
        except Exception:
            return None  # 缓存坏了就回退，别抛

    def put(self, key: str, answer: str, meta: dict | None = None) -> None:
        payload = {"answer": answer, "ts": time.time(), **(meta or {})}
        (self.dir / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def count(self) -> int:
        return len(list(self.dir.glob("*.json")))

    def clear(self) -> int:
        n = 0
        for f in self.dir.glob("*.json"):
            f.unlink()
            n += 1
        self.hits = self.misses = 0
        return n

    # ------------------------------------------------------------ 核心
    def call(self, model_id: str, system_prompt: str, question: str,
             produce: Callable[[], str]) -> tuple[str, str]:
        """查缓存，没命中就调 produce() 并落盘。返回 (答案, hit/miss)。"""
        key = cache_key(model_id, system_prompt, question)
        hit = self.get(key)
        if hit is not None:
            self.hits += 1
            return hit, "hit"
        answer = produce()
        self.put(key, answer, meta={
            "model": model_id,
            "prompt_hash": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:8],
            "question": question,
        })
        self.misses += 1
        return answer, "miss"

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "files": self.count()}


class CountedModel:
    """冒充一次真实模型调用：计时 + 计数。真实项目里这里换成 ChatOpenAI 之类。"""

    def __init__(self, model_id: str, system_prompt: str, latency: float = 0.08):
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.latency = latency
        self.real_calls = 0

    def _produce(self, question: str) -> str:
        time.sleep(self.latency)  # 模拟一次网络往返
        self.real_calls += 1
        return f"客服分诊：{question} | intent=refund | priority=normal"

    def invoke(self, question: str, cassette: Cassette) -> tuple[str, str]:
        return cassette.call(self.model_id, self.system_prompt, question,
                            lambda: self._produce(question))

    @staticmethod
    def no_cache(question: str, latency: float = 0.08) -> tuple[str, str]:
        time.sleep(latency)
        return f"客服分诊：{question} | intent=refund | priority=normal", "nocache"

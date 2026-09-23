"""嵌入层：可插拔，默认实现零依赖。

上层（store / manager）只依赖 Embedder 这个抽象，
所以从本地哈希嵌入换成 bge-m3 或 OpenAI，只改一行构造参数。
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence

_WORD_RE = re.compile(r"[a-z0-9_]+")
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")

# 轻量规范化：把技术名词的不同写法统一，让"只认 Mac"和"偏好 macOS"能对上。
# 产线上这一步通常交给真嵌入模型，这里显式做掉是为了 demo 不依赖模型。
SYNONYMS: dict[str, str] = {
    "mac": "macos", "macbook": "macos", "苹果": "macos", "osx": "macos",
    "win": "windows", "视窗": "windows",
    "乌班图": "ubuntu",
    "py": "python", "python3": "python",
    "js": "javascript", "ts": "typescript",
    "postgresql": "postgres", "pg": "postgres",
    # 多词别名。分词时用不到，但抽取层会在整句上做替换，所以这里可以写词组。
    "vs code": "vscode", "vs-code": "vscode", "visual studio code": "vscode",
    "intellij": "idea",
}


def tokenize(text: str) -> list[str]:
    """中英混排分词。

    英文按词切，中文按单字加相邻双字切。加双字是为了在不引入分词库的前提下
    保留一点短语信息，比如"检索""精简"这种搭配。
    """
    low = text.lower()
    tokens = [SYNONYMS.get(word, word) for word in _WORD_RE.findall(low)]
    han = _HAN_RE.findall(low)
    tokens.extend(han)
    tokens.extend(han[i] + han[i + 1] for i in range(len(han) - 1))
    return tokens


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。两边都做过 L2 归一化，直接点积即可。"""
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class Embedder(ABC):
    """嵌入抽象。换模型只实现这三个成员。"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度。store 用它做一致性校验。"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """把文本编码成向量。"""

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class LocalHashEmbedder(Embedder):
    """零依赖的哈希词袋嵌入（hashing trick + L2 归一化）。

    它不是语义模型，但共享词汇的文本会得到较高的余弦相似度，
    足以让整个 demo 在没有任何 API key 的情况下跑出真实行为。
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
            h = int(digest, 16)
            # 带符号哈希：正负号让冲突互相抵消，减少偏差
            vec[h % self._dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class BGEM3Embedder(Embedder):
    """bge-m3 本地嵌入。产线上用的就是它。

    需要额外安装：pip install sentence-transformers
    首次运行会下载约 2GB 权重。
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", dim: int = 1024) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise ImportError(
                "使用 BGEM3Embedder 需要先安装 sentence-transformers："
                "pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        arr = self._model.encode(list(texts), normalize_embeddings=True, batch_size=32)
        return [[float(x) for x in row] for row in arr]


class OpenAIEmbedder(Embedder):
    """OpenAI 兼容接口的嵌入。

    需要 openai 包与 API key（或兼容网关的 base_url）。
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise ImportError("使用 OpenAIEmbedder 需要先安装 openai：pip install openai") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return [float(x) for x in resp.data[0].embedding]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [[float(x) for x in item.embedding] for item in resp.data]

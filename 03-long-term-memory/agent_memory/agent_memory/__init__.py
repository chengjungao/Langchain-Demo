"""agent_memory：一套工程化的 Agent 长期记忆系统。

设计取向是"拿原语，自己实现治理"：
存储、嵌入、抽取、策略四层都做成可替换的接口，
默认实现零第三方依赖，装不装向量库、用不用模型都能跑。

快速开始：
    from agent_memory import MemoryManager, InMemoryStore

    memory = MemoryManager(InMemoryStore())
    memory.remember("alice", "我不用 Windows，只认 Mac，回答尽量精简")
    print(memory.build_context("alice", "帮我看看这个报错"))
"""

from .embedder import BGEM3Embedder, Embedder, LocalHashEmbedder, OpenAIEmbedder
from .extractor import BaseExtractor, ExtractedMemory, LLMExtractor, RuleBasedExtractor
from .manager import MemoryManager, WriteReport
from .policy import ForgetConfig, ScoringWeights, final_score, plan_forget, plan_write, rank
from .schema import Memory, MemoryType, utc_now
from .store import Hit, InMemoryStore, MemoryStore, SQLiteMemoryStore

__version__ = "0.1.0"

__all__ = [
    "MemoryManager",
    "WriteReport",
    "Memory",
    "MemoryType",
    "Hit",
    "MemoryStore",
    "SQLiteMemoryStore",
    "InMemoryStore",
    "Embedder",
    "LocalHashEmbedder",
    "BGEM3Embedder",
    "OpenAIEmbedder",
    "BaseExtractor",
    "ExtractedMemory",
    "RuleBasedExtractor",
    "LLMExtractor",
    "ScoringWeights",
    "ForgetConfig",
    "rank",
    "final_score",
    "plan_write",
    "plan_forget",
    "utc_now",
    "__version__",
]

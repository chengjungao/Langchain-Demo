"""记忆抽取：从对话里挑出值得长期记住的东西。

这是记忆系统里最难外包的一层。什么该记、什么该忽略、一条记忆占哪个"槽位"，
全都和业务语义绑在一起，所以本文件同时给出两套实现的完整代码：

- RuleBasedExtractor：规则版，零依赖、可离线、行为完全可预期（默认用它）
- LLMExtractor：模型版，接任意 OpenAI 兼容接口，输出严格 JSON

两者接口一致，manager 层不关心用的是哪套。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .embedder import SYNONYMS
from .schema import MemoryType

# ---------------------------------------------------------------- 实体与槽位

# 槽位是"同一类事实的位置"。同一个槽位出现新值，旧值失效。
# 这张表就是"业务语义"落到代码里的样子，也是外包给通用库最别扭的地方。
PREFERENCE_DOMAINS: dict[str, tuple[str, ...]] = {
    "preference.os": ("macos", "windows", "linux", "ubuntu", "debian", "centos"),
    "preference.editor": ("vscode", "vim", "neovim", "emacs", "cursor", "idea"),
    "preference.lang": ("python", "java", "go", "rust", "typescript", "javascript", "scala"),
    "preference.db": ("mysql", "postgres", "milvus", "redis", "mongodb", "sqlite",
                      "elasticsearch", "clickhouse"),
    "preference.cloud": ("阿里云", "腾讯云", "aws", "azure", "gcp"),
}

SLOT_LABELS: dict[str, str] = {
    "preference.os": "操作系统",
    "preference.editor": "编辑器",
    "preference.lang": "编程语言",
    "preference.db": "数据库",
    "preference.cloud": "云平台",
    "preference.style": "回答风格",
    "identity.name": "称呼",
    "profile.project": "在做的项目",
}

# 归一化值还原成人看的写法
DISPLAY_NAMES: dict[str, str] = {
    "macos": "macOS", "windows": "Windows", "linux": "Linux", "ubuntu": "Ubuntu",
    "vscode": "VS Code", "vim": "Vim", "neovim": "Neovim", "emacs": "Emacs",
    "python": "Python", "java": "Java", "go": "Go", "rust": "Rust",
    "typescript": "TypeScript", "javascript": "JavaScript",
    "mysql": "MySQL", "postgres": "PostgreSQL", "milvus": "Milvus", "redis": "Redis",
    "mongodb": "MongoDB", "sqlite": "SQLite", "elasticsearch": "Elasticsearch",
    "clickhouse": "ClickHouse",
}

_ENTITY_TO_SLOT: dict[str, str] = {
    word: slot for slot, words in PREFERENCE_DOMAINS.items() for word in words
}

_ASCII_TERM = re.compile(r"^[a-z0-9+#._-]+$")


def normalize_aliases(clause: str) -> str:
    """把小句里的别名换成规范词，再交给词表匹配。

    这样"只认 Mac"和"偏好 macOS"会落到同一个槽位、同一个值，
    否则同一件事会散成两条记忆。

    长别名优先，且 ASCII 别名要卡词边界：不然 mac 会把 macbook 切成 macosbook。
    """
    low = clause.lower()
    for alias in sorted(SYNONYMS, key=len, reverse=True):
        canonical = SYNONYMS[alias]
        if " " in alias or "-" in alias:
            low = low.replace(alias, canonical)
        else:
            low = re.sub(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", canonical, low
            )
    return low


def contains_term(low_clause: str, term: str) -> bool:
    """词表匹配。ASCII 术语卡词边界，避免 go 命中 google、idea 命中 ideal。"""
    if _ASCII_TERM.match(term):
        return re.search(
            rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low_clause
        ) is not None
    return term in low_clause

# 否定与肯定的信号词。判断顺序：先看否定，再看肯定。
NEG_MARKERS = ("不用", "不喜欢", "讨厌", "别再", "不再用", "不想用", "放弃", "不是")
POS_MARKERS = ("只认", "只用", "习惯", "偏好", "喜欢", "爱用", "平时用",
               "换到", "改成", "现在用", "一直用", "就用", "用的", "用", "是")

# 回答风格：用户的表达偏好，这一类的槽位固定，所以新值天然会覆盖旧值
STYLE_WORDS: dict[str, str] = {
    "精简": "简洁", "简洁": "简洁", "简短": "简洁", "啰嗦": "简洁", "别啰嗦": "简洁",
    "清楚": "详细", "详细": "详细", "多举例": "多举例", "展开": "详细", "多说": "详细",
    "结论先行": "结论先行", "直接给结论": "结论先行",
    "通俗": "通俗", "小白": "通俗",
    "带代码": "带代码", "用表格": "表格",
}
# 互斥的风格：同时出现时以最后出现的为准（用户在改主意）
STYLE_EXCLUSIVE = {"简洁", "详细"}

# 强调信号：出现这些词，说明用户特别在意，importance 拉满
EMPHASIS_MARKERS = ("一定", "必须", "千万", "记住", "别忘", "切记", "改主意", "以后都", "都要")

# 情景信号：这些词说明在讲一段经历，而不是陈述事实
EPISODE_MARKERS = ("上次", "上回", "之前", "前几天", "昨天", "那次", "以前", "上次那", "刚刚")

# 程序性信号：规则与约定。
# 别写成"以后都/记住"这类强调词：那会把用户每一句偏好陈述都复读成一条规则记忆，
# 记忆库里就多出一批和原文一模一样的噪声。
PROCEDURAL_MARKERS = ("一律", "统一", "规范是", "约定", "流程是", "硬性要求", "都不许")

_CLAUSE_SPLIT = re.compile(r"[，。；;,.!?！？\n]+")

# 疑问句判定用的信号。这一步很关键：用户的提问里常常带着他还没定下来的选项，
# 把"回答该详细还是简洁？"记成用户偏好，记忆库就开始被污染了。
_QUESTION_MARKERS = ("？", "?", "还是", "什么", "怎么", "为什么", "哪个", "如何", "是否", "多少")
_QUESTION_SUFFIX = ("吗", "呢", "么")
_IDENTITY_RE = re.compile(r"我叫\s*([A-Za-z\u4e00-\u9fff]{2,8})")
_PROJECT_RE = re.compile(r"我(?:在做|在搞|负责|主导|在开发|在写)\s*([^，。,.!?！？]{2,30})")


@dataclass
class ExtractedMemory:
    """抽取结果。还没落库，也不带 id / 时间戳。"""

    content: str
    type: MemoryType = MemoryType.SEMANTIC
    slot: str | None = None
    importance: float = 0.5
    metadata: dict = field(default_factory=dict)


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(text) if c.strip()]


def display_of(value: str) -> str:
    return DISPLAY_NAMES.get(value, value)


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, text: str, speaker: str = "user") -> list[ExtractedMemory]:
        """从一段文本里抽记忆。返回空列表表示这段没什么值得记的。"""


class RuleBasedExtractor(BaseExtractor):
    """规则版抽取器。零依赖，不需要 API key，行为可预期。

    规则版的局限也很直白：它只认得词表里的实体，遇到新领域需要补词表。
    所以生产上通常第一步用规则兜住高频场景，第二步再上模型版补长尾。
    """

    def extract(self, text: str, speaker: str = "user") -> list[ExtractedMemory]:
        clauses = split_clauses(text)
        results: list[ExtractedMemory] = []

        # 疑问句只抽经历，不抽事实与偏好。
        # 用户问"回答该详细还是简洁"，那是他在问你，不是他的偏好。
        is_question = (
            any(mark in text for mark in _QUESTION_MARKERS)
            or text.strip().endswith(_QUESTION_SUFFIX)
        )
        if not is_question:
            results.extend(self._extract_identity(text))
            results.extend(self._extract_project(text))
            results.extend(self._extract_preferences(clauses))
            results.extend(self._extract_style(text))
            results.extend(self._extract_procedural(text))

        # 经历类不跳过：疑问句里也可能夹着一段值得记的经历
        results.extend(self._extract_episode(text, clauses))

        return self._dedupe(results)

    # ------------------------------------------------------------ 各类抽取

    def _extract_identity(self, text: str) -> list[ExtractedMemory]:
        m = _IDENTITY_RE.search(text)
        if not m:
            return []
        name = m.group(1)
        # 中文名不加空格，英文名加一个，读起来才不别扭
        display = f" {name}" if name.isascii() else name
        return [ExtractedMemory(
            content=f"用户名叫{display}",
            type=MemoryType.SEMANTIC,
            slot="identity.name",
            importance=0.85,
        )]

    def _extract_project(self, text: str) -> list[ExtractedMemory]:
        m = _PROJECT_RE.search(text)
        if not m:
            return []
        project = m.group(1).strip()
        if len(project) < 2:
            return []
        return [ExtractedMemory(
            content=f"用户在做{project}",
            type=MemoryType.SEMANTIC,
            slot="profile.project",
            importance=0.8,
        )]

    def _extract_preferences(self, clauses: list[str]) -> list[ExtractedMemory]:
        """先定位实体与极性，再按槽位聚合成一条完整的偏好。"""
        buckets: dict[str, dict[str, list[str]]] = {}
        for clause in clauses:
            low = normalize_aliases(clause)
            slot = entity = None
            for word, slot_name in _ENTITY_TO_SLOT.items():
                if contains_term(low, word):
                    slot, entity = slot_name, word
                    break
            if slot is None:
                continue
            negative = any(marker in clause for marker in NEG_MARKERS)
            positive = any(marker in clause for marker in POS_MARKERS)
            if not negative and not positive:
                continue
            bucket = buckets.setdefault(slot, {"pos": [], "neg": []})
            bucket["neg" if negative else "pos"].append(entity)

        out: list[ExtractedMemory] = []
        for slot, bucket in buckets.items():
            label = SLOT_LABELS.get(slot, slot)
            pos = bucket["pos"][0] if bucket["pos"] else None
            neg = bucket["neg"][0] if bucket["neg"] else None
            if pos and neg:
                content = f"用户{label}偏好是 {display_of(pos)}，明确不用 {display_of(neg)}"
            elif pos:
                content = f"用户{label}偏好是 {display_of(pos)}"
            else:
                content = f"用户明确不用 {display_of(neg)}（{label}）"
            out.append(ExtractedMemory(
                content=content,
                type=MemoryType.SEMANTIC,
                slot=slot,
                importance=0.7,
                # 规范化后的实体值记进元信息：同槽位下值相同就是同一事实，
                # 值不同说明用户改主意了，旧值该失效
                metadata={"value": pos or neg, "rejected": neg},
            ))
        return out

    def _extract_style(self, text: str) -> list[ExtractedMemory]:
        hits = [norm for word, norm in STYLE_WORDS.items() if word in text]
        if not hits:
            return []
        # 互斥风格取最后出现的那个：用户在改主意时，后面的才是他现在的意思
        exclusive = [h for h in hits if h in STYLE_EXCLUSIVE]
        if exclusive:
            hits = [h for h in hits if h not in STYLE_EXCLUSIVE] + [exclusive[-1]]
        content = "用户希望回答" + "、".join(dict.fromkeys(hits))
        importance = 0.9 if any(m in text for m in EMPHASIS_MARKERS) else 0.65
        return [ExtractedMemory(
            content=content,
            type=MemoryType.SEMANTIC,
            slot="preference.style",
            importance=importance,
        )]

    def _extract_episode(self, text: str, clauses: list[str]) -> list[ExtractedMemory]:
        """情景记忆：带时间线索的经历。这类记忆会被整合，也会最快衰减。"""
        if not any(marker in text for marker in EPISODE_MARKERS):
            return []
        picked = [c for c in clauses if any(m in c for m in EPISODE_MARKERS)]
        if not picked:
            return []
        # 只留最长的那一句，避免把一段话拆成好几条琐碎记忆
        content = max(picked, key=len)
        return [ExtractedMemory(
            content=content,
            type=MemoryType.EPISODIC,
            slot=None,
            importance=0.45,
        )]

    def _extract_procedural(self, text: str) -> list[ExtractedMemory]:
        if not any(marker in text for marker in PROCEDURAL_MARKERS):
            return []
        return [ExtractedMemory(
            content=f"约定：{text.strip()}",
            type=MemoryType.PROCEDURAL,
            slot="procedure.general",
            importance=0.9,
        )]

    # ------------------------------------------------------------ 工具方法

    @staticmethod
    def _dedupe(items: list[ExtractedMemory]) -> list[ExtractedMemory]:
        """同一段话里同一个槽位只留一条，避免自己跟自己打架。"""
        seen: dict[str, ExtractedMemory] = {}
        free: list[ExtractedMemory] = []
        for item in items:
            if item.slot is None:
                free.append(item)
                continue
            existing = seen.get(item.slot)
            # 后出现的覆盖先出现的（用户改主意时的自然语义）
            if existing is None or item.importance >= existing.importance:
                seen[item.slot] = item
        return list(seen.values()) + free


class LLMExtractor(BaseExtractor):
    """模型版抽取器。接任意 OpenAI 兼容接口，输出严格 JSON。

    生产上最省事的做法：用便宜的小模型做抽取，把温度和重试都卡死，
    解析失败就当这段没有可记的内容，绝不让抽取环节把主链路拖垮。
    """

    PROMPT = """你是记忆抽取器。从下面的对话里挑出值得长期记住的信息，只输出 JSON 数组。
每个元素包含四个字段：
- content: 一条自洽的中文记忆，第三人称，主语统一用"用户"
- type: semantic（事实与偏好）/ episodic（经历与经验）/ procedural（规则与方法）
- slot: 这条记忆占用的槽位名，格式如 preference.os、identity.name；不占槽位则为 null
- importance: 0 到 1 的小数，越稳定、越重要越高

规则：寒暄、临时任务、一次性的问题都不要抽。没有可记的内容就输出 []。
不要输出解释，不要包 markdown 代码块。

对话：
{text}
"""

    def __init__(self, client, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model

    def extract(self, text: str, speaker: str = "user") -> list[ExtractedMemory]:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                messages=[{"role": "user", "content": self.PROMPT.format(text=text)}],
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
            payload = json.loads(raw)
        except Exception:
            # 抽取出错不该影响主链路，静默返回空
            return []

        out: list[ExtractedMemory] = []
        for item in payload if isinstance(payload, list) else []:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            try:
                mem_type = MemoryType(item.get("type", "semantic"))
            except ValueError:
                mem_type = MemoryType.SEMANTIC
            slot = item.get("slot") or None
            importance = float(item.get("importance", 0.5))
            out.append(ExtractedMemory(
                content=content,
                type=mem_type,
                slot=str(slot) if slot else None,
                importance=max(0.0, min(1.0, importance)),
            ))
        return out

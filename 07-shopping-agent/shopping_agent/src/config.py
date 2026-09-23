# -*- coding: utf-8 -*-
"""运行时配置：模型端点、参数与护栏阈值。

为什么要有这一层：模型地址、密钥、超时、金额阈值这些东西**不该写进代码**。
写死在代码里，换个模型就要改文件；放在配置文件里，改一处就够。
这也让同一个工程可以在本地 LM Studio 和云端服务之间来回切。

读取优先级：`config.json` > 环境变量 > 内置默认值。

密钥只留在服务端。发给浏览器的永远是掩码版本（`sk-1****cdef`），
前端提交回来的也是掩码时，表示「这一项不动」。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

# 预置的服务商。填 URL 和模型名是最容易出错的两步，先把常见组合铺好。
PRESETS: dict[str, dict] = {
    "lmstudio": {
        "label": "LM Studio（本地）",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
        "model": "",
        "hint": "本地服务不校验密钥，填任意值即可。模型名留空会自动取第一个可用模型。",
    },
    "ollama": {
        "label": "Ollama（本地）",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "model": "qwen2.5:7b",
        "hint": "Ollama 的 OpenAI 兼容层默认端口 11434。",
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",
        "model": "deepseek-chat",
        "hint": "按量计费的云端服务，适合对比本地与云端的差异。",
    },
    "dashscope": {
        "label": "通义千问（阿里云）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "",
        "model": "qwen-plus",
        "hint": "走 OpenAI 兼容模式，密钥来自阿里云百炼控制台。",
    },
    "siliconflow": {
        "label": "硅基流动",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key": "",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "hint": "聚合了多种开源模型，可用来横向对比同一份提示词的效果。",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini",
        "hint": "标准 OpenAI 端点。",
    },
}


class ModelConfig(BaseModel):
    """一份完整的可运行配置。

    除了连接信息，这里还装了**护栏阈值**。它们和模型参数放在一起，
    是因为它们都随环境变化：本地小模型可以放宽轮数，线上必须收紧。
    """

    provider: str = "lmstudio"
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout: int = 120

    # 推理型模型的思考开关。设成 true 时，对本地端点会请求关闭思考。
    # 为什么默认关：思考链会先吃掉几百到上千 token，而 Agent 场景里
    # 大部分判断（分个类、挑个工具）不需要它。开着会出现「模型明明在跑，
    # 但一个字都没输出」这种情况 —— 输出全被推理内容占了。
    disable_thinking: bool = True

    # ── 护栏 ────────────────────────────────────────────
    max_rounds: int = Field(default=6, description="单个子图里 Agent 循环的轮数上限")
    max_amount: float = Field(default=2000, description="超过这个金额必须人工确认")
    history_limit: int = Field(default=12, description="送进模型的历史消息条数上限")
    memory_limit: int = Field(default=6, description="每轮注入的长期记忆条数上限")

    # ── 只读展示字段 ────────────────────────────────────
    def masked(self) -> dict:
        """发给浏览器的版本。密钥只露头尾，够用户分辨填的是哪一把。"""
        d = self.model_dump()
        key = self.api_key or ""
        d["api_key"] = "" if not key else (key[:4] + "****" + key[-4:] if len(key) > 8 else "****")
        d["has_key"] = bool(key)
        d["mode"] = "live" if self.is_ready() else "demo"
        return d

    def is_ready(self) -> bool:
        """能不能真的调模型。地址或模型名缺一个都只能走演示模式。"""
        return bool(self.base_url.strip()) and bool(self.model.strip())


def _as_bool(raw: str) -> bool:
    return str(raw).strip().lower() in ("1", "true", "yes", "on", "是")


def _env_overrides() -> dict:
    """环境变量旁路。容器部署时用这个，不用挂载配置文件。"""
    mapping = {
        "SHOPAGENT_BASE_URL": ("base_url", str),
        "SHOPAGENT_API_KEY": ("api_key", str),
        "SHOPAGENT_MODEL": ("model", str),
        "SHOPAGENT_TEMPERATURE": ("temperature", float),
        "SHOPAGENT_MAX_TOKENS": ("max_tokens", int),
        "SHOPAGENT_TIMEOUT": ("timeout", int),
        "SHOPAGENT_MAX_ROUNDS": ("max_rounds", int),
        "SHOPAGENT_MAX_AMOUNT": ("max_amount", float),
        "SHOPAGENT_DISABLE_THINKING": ("disable_thinking", _as_bool),
    }
    out: dict = {}
    for env, (field, cast) in mapping.items():
        raw = os.environ.get(env)
        if raw:
            try:
                out[field] = cast(raw)
            except ValueError:
                pass
    return out


class ConfigStore:
    """配置文件读写。写的时候只覆盖传进来的字段，没传的保持原样。"""

    MASK_MARK = "****"          # 前端回传掩码时视为「不修改密钥」

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else CONFIG_PATH
        self._cache: ModelConfig | None = None

    def load(self) -> ModelConfig:
        if self._cache is not None:
            return self._cache
        data: dict = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        data.update(_env_overrides())
        self._cache = ModelConfig(**{k: v for k, v in data.items()
                                     if k in ModelConfig.model_fields})
        return self._cache

    def save(self, patch: dict) -> ModelConfig:
        """局部更新。密钥为掩码时跳过，避免把占位符写进配置。"""
        cur = self.load().model_dump()
        for k, v in patch.items():
            if k not in ModelConfig.model_fields:
                continue
            if k == "api_key" and isinstance(v, str) and self.MASK_MARK in v:
                continue
            cur[k] = v
        cfg = ModelConfig(**cur)
        self.path.write_text(
            json.dumps(cfg.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        self._cache = cfg
        return cfg

    def reload(self) -> ModelConfig:
        self._cache = None
        return self.load()


_store: ConfigStore | None = None


def get_config() -> ModelConfig:
    global _store
    if _store is None:
        _store = ConfigStore()
    return _store.load()


def save_config(patch: dict) -> ModelConfig:
    global _store
    if _store is None:
        _store = ConfigStore()
    return _store.save(patch)

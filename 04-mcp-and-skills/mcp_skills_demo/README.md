# mcp_skills_demo

《MCP 与 Skills 接入：工具怎么接，章法怎么给》配套工程包。

一个售后处理 Agent 的最小可用工程，演示两件事：

1. **能力层**：工具用 MCP 协议接进来，两种接入姿势都给实现
2. **过程层**：工序用 Skills 装进来，技能清单常驻、正文按需读

核心逻辑零业务依赖，**不需要 API key 就能跑通**（用测试替身模型验证中间件行为）。
换真模型只改一处构造。

## 目录

```
mcp_skills_demo/
├── servers/
│   ├── order_server.py          stdio：查单 / 试算 / 提交退款（3 个工具）
│   └── kb_server.py             streamable-http：政策检索 + 1 个 resource
├── skills/
│   ├── refund-flow/             退款工序，带 references/policy.md
│   └── ticket-report/           周报工序，带 scripts/build_report.py
├── src/
│   ├── mcp_client.py            两种接入姿势 + 起本地 http server
│   ├── tool_gateway.py          货架式工具网关 + 合成工具工厂
│   ├── skills_middleware.py     自实现的 Skills 中间件（含路径白名单）
│   ├── agent.py                 MCP + Skills 组装
│   └── fake_model.py            测试替身模型
├── demos/
│   ├── demo_mcp.py              不需要模型，直连 MCP 工具
│   ├── demo_gateway.py          7000 个工具只让模型看到 8 个
│   └── demo_full.py             合体全流程
├── token_meter.py               工具定义 token 账单实测
└── tests/test_smoke.py          16 项冒烟测试
```

## 环境

```
pip install "langchain>=1.4" langgraph langchain-mcp-adapters "mcp>=1.30"
```

本文实测基线：langchain 1.4.0 / langgraph 1.2.11 / langchain-core 1.6.2 /
langchain-mcp-adapters 0.3.2 / mcp 1.30.0。Python 3.11 以上。

## 跑起来

```bash
# 1. 只验链路，不需要模型
python demos/demo_mcp.py

# 2. 算一遍工具定义的账
python token_meter.py --count 7000

# 3. 货架模式：7000 个工具，模型只看到 8 个
python demos/demo_gateway.py

# 4. 合体全流程
python demos/demo_full.py

# 5. 冒烟测试
python -m unittest discover -s tests -v
```

`demo_gateway.py` 默认造 7000 个合成工具。想跑快点：

```bash
GATEWAY_TOOLS=800 python demos/demo_gateway.py   # Windows: set GATEWAY_TOOLS=800
```

## 换成真模型

`src/fake_model.py` 里两个替身只实现了 `_generate` 与 `_llm_type`。换成真模型：

```python
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(model="deepseek-chat")
built = build_agent(model, gateway=gateway, top_k=8)
await built.agent.ainvoke({"messages": [{"role": "user", "content": "帮我处理一笔退款"}]})
```

中间件、网关、技能库都不用改。

## 四个可以直接抄走的做法

**1. 网关只接管货架上的工具。**
`request.override(tools=...)` 会整份替换工具列表，连带把别的中间件注入的工具
（例如 `read_skill`）一起抹掉。正确做法是把不在货架上的工具原样保留。

**2. 同步与异步钩子都要实现。**
只写 `wrap_model_call` 而用 `ainvoke` 跑，会直接报
`NotImplementedError: Asynchronous implementation of awrap_model_call is not available`。

**3. 技能正文不预加载。**
清单进 system message（约 100 字符级），正文等模型自己调 `read_skill` 再读。

**4. 读文件必须走白名单。**
技能正文是磁盘上的文件，模型给的路径要 resolve 之后校验是否还在技能目录内，
不然一个 `../` 就能把服务器上的任意文件读进上下文。

## 两个坑

**`with MultiServerMCPClient(...)` 已经不能用。**
实测抛 `TypeError: 'MultiServerMCPClient' object does not support the context manager protocol`。
网上大量教程还停在 with 写法。

**`count_tokens_approximately` 必须用 `tools=` 传工具。**
把工具定义塞进 messages，会走 `convert_to_messages` 解析，算的是另一套口径，
同一个工具两条路径给出两个数字（实测 184 与 172），方向还不固定。
算账算错，比不算更糟。

## 许可

示例代码，随意取用。

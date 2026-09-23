# LangChain 实战 Demo 合集

公众号「代码的江湖」LangChain 实战系列的配套代码。8 个可独立运行的小工程，按文章顺序排列：
从一个会等的 Agent，一路搭到带浏览器界面的完整购物助手。

## 目录结构

每个章节目录下都保留着 demo 自己的文件夹，和文章里下载到的 zip 解压后完全一样：

```
Langchain-Demo/
├── 01-langgraph-workflow/
│   └── langgraph_demo/          ← 原 demo 根目录，运行命令都从这一层开始
│       ├── README.md
│       ├── requirements.txt
│       └── ...
├── 02-context-governance/
│   └── context_governance_demo/
└── ...
```

所以命令里的 `cd` 一律进到 demo 那一层，例如 `cd 07-shopping-agent/shopping_agent`。

## 一条贯穿全仓的原则：零成本可跑

每个 demo 都用**记录型假模型**或**内置规则实现**驱动主链路。不需要任何 API key，
不需要云端账号，部分连网络都不需要。先跑出结果，再按各章 README 换成真模型。

换模型的位置都收敛在一处构造上，换完调用方代码不动。各章 README 都写明了改哪里。

## 章节索引

| 目录 | 演示什么 | 配套文章 | 需要真模型 |
|---|---|---|---|
| `01-langgraph-workflow/langgraph_demo` | 三关闭环：人工闸口 interrupt、checkpointer 持久化、时间旅行回溯 | 《LangGraph 进阶：让 Agent 会等、会存、会回头》 | 否 |
| `02-context-governance/context_governance_demo` | 上下文治理三招：裁剪、删除、摘要，四种策略的账单对照 | 《Agent 的记忆（上）：上下文塞不下了怎么办》 | 否 |
| `03-long-term-memory/agent_memory` | 长期记忆参考实现：抽取、存储、检索、治理四层可替换 | 《Agent 的记忆（下）》 | 否 |
| `04-mcp-and-skills/mcp_skills_demo` | MCP 工具接入两种姿势，加 Skills 工序装配（含路径白名单） | 《MCP 与 Skills 接入：工具怎么接，章法怎么给》 | 否 |
| `05-multi-agent/multi_agent_demo` | 拆 Agent 的代价：隔离、交接、并行、死循环与三笔账 | 《多 Agent：拆之前，先回答这 3 个问题》 | 否 |
| `06-eval-and-monitor/agent_eval_demo` | 评测与监控最小可跑集：trace 采集、评测集、门禁、指标、脱敏 | 《评测与监控》 | 否，第 8 个演示可选 |
| `07-shopping-agent/shopping_agent` | 完整实战：带浏览器界面的购物助手，含三层记忆与护栏 | 《零件都会了，为什么还是拼不出一个 Agent》 | 否，不配模型走内置规则 |
| `08-dspy-query-rewrite/dspy_query_rewrite` | DSPy 编译式提示工程：优化器编译 prompt、产物落盘、蒸馏导出 | 《一个被低估的工具：DSPy》 | 否，默认 DummyLM |

## 环境

统一要求 **Python 3.10+**。`05-multi-agent` 与 `06-eval-and-monitor` 的实测环境是 3.13。

各章的依赖口径不同：`05-multi-agent` 锁了精确小版本（`==`），其余给的是下限（`>=`）。
**一个 demo 一个环境**，别装进同一个 venv：

```bash
cd 03-long-term-memory/agent_memory
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

## 各章怎么跑

```bash
# 01 三关闭环
cd 01-langgraph-workflow/langgraph_demo
python main.py                   # 人工闸口 + 时间旅行，一次跑完
python sqlite_demo.py run        # 持久化：跑到挂起后退出进程
python sqlite_demo.py resume     # 重启后从断点恢复

# 02 上下文治理
cd 02-context-governance/context_governance_demo
python context_governance_demo.py

# 03 长期记忆
cd 03-long-term-memory/agent_memory
python demo.py                   # 完整剧本，只用标准库
python cli.py                    # 交互式命令行
python -m unittest discover -s tests -v

# 04 MCP 与 Skills
cd 04-mcp-and-skills/mcp_skills_demo
python demos/demo_mcp.py         # 只验链路，不需要模型
python token_meter.py --count 7000
python demos/demo_gateway.py     # 7000 个工具，模型只看到 8 个
python demos/demo_full.py        # 合体全流程
python -m unittest discover -s tests -v

# 05 多 Agent
cd 05-multi-agent/multi_agent_demo
python demos/demo_isolation.py
python demos/demo_handoff.py
python demos/demo_parallel.py
python demos/demo_cost.py
python demos/demo_loop.py
python demos/demo_prebuilt.py    # 可选依赖，没装会走降级分支
python tests/test_smoke.py

# 06 评测与监控
cd 06-eval-and-monitor/agent_eval_demo
python run_all.py                # 8 个演示 + 2 个命令行工具 + 33 项冒烟测试

# 07 购物助手
cd 07-shopping-agent/shopping_agent
python run.py                    # 自动打开 http://127.0.0.1:8765
python run.py --check            # 只做自检

# 08 DSPy
cd 08-dspy-query-rewrite/dspy_query_rewrite
python -m dspy_query_rewrite demo
```

## 关于密钥

全仓**没有任何真实凭据**。三处看起来像密钥的字符串都是测试夹具，不是配置漏出来的：

| 位置 | 是什么 |
|---|---|
| `02-context-governance/context_governance_demo/context_governance_demo.py` 里的 `SECRET` | 演示敏感字段被治理时的假值 |
| `06-eval-and-monitor/agent_eval_demo/src/redact.py` 的 `PROBES` 字典 | 脱敏功能的测试样本，含 AWS 官方文档示例 key |
| `08-dspy-query-rewrite/dspy_query_rewrite/config/config.yaml` | 只写 `api_key_env: OPENAI_API_KEY`，指向环境变量 |

`07-shopping-agent/shopping_agent/config.json` 由程序在运行时生成，已在 `.gitignore` 里排除。
仓库只保留 `config.example.json`。需要接真模型时，把密钥放进环境变量或本地 `config.json`，
不要提交。

`06-eval-and-monitor/agent_eval_demo` 的第 8 个演示需要本地跑一个 OpenAI 兼容端点（默认 LM Studio 的
`http://127.0.0.1:1234/v1`）。服务没起来时它会跳过并返回 0，不影响其余部分。

## 目录约定

每个 demo 目录都是自包含的：自己的 README、依赖清单、源码与测试。章节之间、demo 之间都没有
导入关系，可以单独取一个 demo 目录出来跑。

各章 README 的详尽程度不同。工程化程度高的几章（`03`、`04`、`06`、`07`）写得更细，
包含目录说明、设计取舍与容易踩的坑。

## 关于作者

这个仓库的代码，来自公众号「代码的江湖」的实战系列。每篇文章配一份可独立运行
的工程，代码和文章同步更新。

<img src="assets/qrcode-gh.jpg" width="220" alt="公众号：代码的江湖">

## 许可

MIT License，见 [LICENSE](LICENSE)。

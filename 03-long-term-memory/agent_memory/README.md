# agent_memory · Agent 长期记忆系统

一套工程化的长期记忆实现。核心层零第三方依赖，跑演示不需要 API key。

它不是又一个记忆库，而是一份**可改的参考实现**：抽取、存储、检索、治理四层都用
可替换的接口隔开，你可以只换其中一层，也可以整体接进自己的系统。

## 它解决什么问题

一个不带记忆的 Agent，每次会话都是陌生人。而直接把历史消息全塞进上下文，会在
两周内撞上三堵墙：token 成本线性上涨、无关历史稀释注意力、跨会话依然归零。

这套东西管的是三件事：

| 问题 | 对应能力 |
|---|---|
| 什么该记、什么该忽略、这句话是不是在陈述事实 | 抽取层（`extractor.py`） |
| 记忆存在哪、怎么在若干条里找出相关的那几条 | 存储层（`store.py`） |
| 记重了怎么办、用户改主意了怎么办、什么时候该忘 | 治理层（`policy.py`） |

## 快速开始

```bash
python demo.py          # 完整剧本：跨会话召回、偏好变更、经历整合、遗忘、租户隔离
python cli.py           # 交互式命令行，手动玩
python agent_demo.py    # LangGraph 集成演示（需要 langgraph）
python -m unittest discover -s tests -v
```

`demo.py` 和 `cli.py` 只用标准库。`agent_demo.py` 需要 `pip install -r requirements.txt`。

最小用法：

```python
from agent_memory import InMemoryStore, MemoryManager

memory = MemoryManager(InMemoryStore())

# 写入：从自然语言里自动抽取值得记的内容
memory.remember("alice", "我不用 Windows，只认 Mac，回答尽量精简")

# 召回：新会话、空历史，照样能答上话
print(memory.build_context("alice", "帮我看看这个报错"))
```

## 六个设计决策

**1. 记忆是一等公民对象，不是字符串。**
类型、重要性、槽位、来源、有效期都是字段（`schema.py`）。后面所有能力都长在这些字段上。

**2. 用槽位判断"改主意"，而不是靠相似度。**
每类事实占一个槽位（`preference.os`、`identity.name`）。同一个槽位出现新值，
旧值失效；值相同则是同一事实又说了一遍，合并而非新增。
只靠向量相似度会出错：`偏好是 macOS，明确不用 Windows` 和 `偏好是 Windows`
共享大量词汇，相似度很高，但它们是两个相反的事实。

**3. 失效而不是删除。**
旧记忆记的是"从什么时候起不再为真"，并指向取而代之的那条。
历史留着，审计、排查、回溯都靠它。遗忘才是真删，两者语义不同。

**4. 混合检索。**
向量相似度负责"意思相近但用词不同"，关键词命中负责专有名词和型号，
再乘上时间衰减与重要性。四路合成最终排序，公式在 `policy.final_score`。

**5. 多租户隔离放在存储层。**
所有查询强制带租户过滤，越权读取在存储层就返回 None。
隔离放在最底下，上层写错也漏不出去。

**6. 疑问句不抽记忆。**
用户问"回答该详细还是简洁"，不该被记成偏好。抽取前先判句子类型，
否则记忆库会被用户的提问污染。

## 目录结构

```
agent_memory/
├── demo.py                 完整剧本演示（推荐入口）
├── agent_demo.py           LangGraph Agent 接入演示
├── cli.py                  交互式命令行
├── requirements.txt
├── agent_memory/
│   ├── schema.py           记忆数据模型（类型/重要性/槽位/双时序）
│   ├── embedder.py         嵌入层，可插拔（本地哈希 / bge-m3 / OpenAI）
│   ├── extractor.py        抽取层（规则版 + 模型版）
│   ├── store.py            存储层（SQLite，向量 + 关键词混合召回）
│   ├── policy.py           治理层（评分/合并/失效/遗忘/整合）
│   ├── manager.py          门面（remember / recall / forget / consolidate）
│   ├── langgraph_store.py  自定义 BaseStore，接 LangGraph 官方 store 接口
│   └── agent.py            接入 Agent（工具内访问 + 流程内固定写入）
└── tests/test_smoke.py     14 项行为测试
```

## 接进 LangGraph

两种姿势，见 `agent.py`：

- **工具内访问**：把记忆做成工具，模型自己决定何时记、何时查
- **流程内固定写入**：在回来的路上挂一个节点，该落的一定落

一个容易踩的坑：节点函数里注入的是 `Runtime`，**工具函数里注入的是 `ToolRuntime`**。
注解写错会在调用时报 `Error invoking tool`，而且错误信息不会告诉你注解错了。

要用官方 store 接口（`graph.compile(store=...)`），用 `langgraph_store.py` 里的
`LangGraphMemoryStore` 包一层即可，内部仍然是你自己的库和治理规则。

## 接自己的向量库

实现 `MemoryStore` 的三个方法（`add` / `search` / `list`）就能换成 Milvus、
Elasticsearch 或你们内部的检索服务，上层不用改。当前 SQLite 实现在小规模下
走应用层算分，数据量上来时把向量检索下沉到向量库即可，接口签名不变。

## 换成真实模型

```python
from agent_memory import BGEM3Embedder, LLMExtractor, MemoryManager

memory = MemoryManager(
    store=InMemoryStore(embedder=BGEM3Embedder()),   # 产线嵌入
    extractor=LLMExtractor(client, model="qwen-plus"),  # 模型抽取
)
```

## 关于开源方案

mem0、Zep、Letta、LangMem 都能给你一套现成的记忆能力，本项目的定位不同：
它给你原语与治理逻辑，而不是一个成品。什么时候该用现成的、什么时候该自己造，
判断标准写在正文里那句：记忆系统真正决定效果的三件事（记什么、什么时候忘、
冲突怎么合并）恰好都是通用方案替你决定的部分。这三件事和你的业务绑得越紧，
自建的价值越大。

## License

MIT

# multi_agent_demo

《多 Agent：拆之前，先回答这 3 个问题》配套工程化 demo。

拆 Agent 之前该问的问题、拆之后会遇到的坑、以及这笔账怎么走，都在这里跑得出来。

**不需要 API key。** 全部由记录型假模型驱动，换真模型只改一处构造（见下文）。

## 环境

```
Python 3.13
langchain       1.4.0
langchain-core  1.6.2
langgraph       1.2.11
```

```bash
pip install -r requirements.txt
```

`demos/demo_prebuilt.py` 额外需要两个可选包，没装会走降级分支并打印说明：

```bash
pip install langgraph-supervisor langgraph-swarm
```

## 快速开始

```bash
python demos/demo_isolation.py    # 默认姿势并不隔离，Send 才隔离
python demos/demo_handoff.py      # 手写 Command 交接的完整轨迹
python demos/demo_parallel.py     # Send 的正确姿势与错误姿势原文
python demos/demo_cost.py         # 三笔账：固定开销 / 窄任务 / 宽任务
python demos/demo_loop.py         # 转交死循环复现 + 计数守卫
python demos/demo_prebuilt.py     # 官方两个预制件对照（可选依赖）
python tests/test_smoke.py        # 13 项冒烟测试
```

全部脚本退出码 0。

## 六个 demo 各自回答什么

| 脚本 | 回答的问题 | 关键输出 |
|---|---|---|
| `demo_isolation.py` | 把 Agent 拆开，到底隔离了什么？ | 默认姿势子 Agent 看到父图全部 26 条消息；Send 姿势只看到 2 条 |
| `demo_handoff.py` | 自己画图交接长什么样？ | 14 条消息的完整轨迹，模型调用 10 次 |
| `demo_parallel.py` | `Send` 为什么跑不通？ | 正确姿势三个 worker 各拿 payload；错误姿势抛 `InvalidUpdateError` 原文 |
| `demo_cost.py` | 多 Agent 到底贵还是便宜？ | 窄任务 0.51x，宽任务 1.61x，模型调用 2 次变 10 次 |
| `demo_loop.py` | 转交环怎么防？ | `GraphRecursionError` 原文 + 计数守卫拦下 |
| `demo_prebuilt.py` | 官方预制件能不能用？ | supervisor 7 条转交轨迹、`parallel_tool_calls` 默认 False、swarm 无中央节点 |

## 关键结论速查

**隔离**

| 姿势 | 子 Agent 第一次调用看到什么 |
|---|---|
| 子 Agent 直接 `add_node` 嵌进父图 | 父图全部历史（24 条）+ 任务 = 26 条 |
| `Send("worker", {"task": ...})` | 只有 system + 任务 = 2 条 |
| 子图声明独立 state schema | 同名键 `messages` 照样全带；异名键被挡住 |

**账**

| 场景 | 单 Agent | 多 Agent | 倍数 |
|---|---|---|---|
| 固定开销 | 1,034 | 1,213 | 1.17x |
| 窄任务（动 1 个专家） | 3,292 | 1,672 | 0.51x |
| 宽任务（动 3 个专家） | 2,232 | 3,601 | 1.61x |

分水岭：**一次请求要动几个专家。**

**记忆**

| 做法 | 能不能work |
|---|---|
| 父图传 store，专家用 `get_store()` 读 | 能，store 会透传进子图和 Send 分支 |
| 专家自带一个 `store=`，嵌在父图里 | **不能**，父图的 store 会覆盖它 |
| 专家自带 store 且独立运行（不当子图） | 能 |

## 换真模型

所有 demo 的模型都从这一处构造：

```python
from src.spy_model import SpyChatModel
model = SpyChatModel(tag="order", script=[{"text": "答好了"}])
```

换成真实模型：

```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

调用方代码不用动。`script` 是假模型的脚本，真模型不需要它。

注意：换成真模型后，`demo_cost.py`（纯计数，不调模型）与 `tests/test_smoke.py`
的行为不变；其余 demo 的轨迹会随模型判断而变。

## 目录

```
multi_agent_demo/
├── src/
│   ├── agents.py         三个专家 Agent 的装配
│   ├── spy_model.py      记录型假模型（记录每个 Agent 到底收到了什么）
│   ├── tools.py          合成工具与提示词（计数口径与上一篇同源）
│   ├── isolation.py      三种隔离手段的图
│   ├── handoff.py        手写 Command 交接
│   ├── parallel.py       Send 并行分发（含错误姿势）
│   ├── loop_guard.py     转交环检测与计数守卫
│   └── cost.py           三笔账
├── demos/                六个可独立运行的脚本
├── tests/test_smoke.py   13 项冒烟测试
└── requirements.txt
```

## 三个容易踩的地方

**一、`Send` 必须从条件边返回。** 写在普通节点里 `return [Send(...)]` 会抛
`InvalidUpdateError: Expected dict, got [Send(...)]`。网上不少示例是错的。

**二、计数口径不能混。** 工具定义的 token 必须用 `count_tokens_approximately(messages, tools=...)`
这一条路径算。把工具定义塞进 messages 再计数，得到的是另一套数字，两套之间不可比。

**三、pydantic 模型是可迭代的。** 给 `make_order_expert` 之类带默认参数的构造函数
传参时，一律用关键字。位置传参会把模型对象塞进 `script` 形参，迭代 pydantic 模型
得到的是一串 `(键, 值)` 元组，报错却发生在几百行之外。
`src/agents.py` 里的 `_check_script()` 专门挡这个手误。

## 一条判断链

```
要不要拆？
  ├─ 有没有必须隔开的上下文？      没有 → 先别拆
  ├─ 有没有必须分开的权限？        没有 → 先别拆
  └─ 有没有必须独立的失败域？      没有 → 先别拆

三问答不出「是」，就别为一份用不上的隔离付钱。
拆了之后，先给失败兜底：recursion_limit 给明确值，节点套计数守卫。
```

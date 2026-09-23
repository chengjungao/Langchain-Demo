# context_governance_demo

上下文治理三招的实测脚本：裁剪、删除、摘要。

配套文章：《Agent 的记忆（上）：上下文塞不下了怎么办》

## 它是什么

一段 20 轮的客服会话，在四种治理策略下分别跑一遍，把 token 账单摆出来对比。
所有模型调用都换成了一个只记账的假模型，所以**不需要任何 API key**，装上依赖就能跑。

## 怎么跑

```bash
pip install -r requirements.txt
python context_governance_demo.py
```

Python 3.10 以上。依赖只有 4 个包：langchain、langchain-core、langgraph、pydantic。

## 先说口径，不然数字会看错

脚本统计的是**字符数**，不是真实 token 数。这么定是为了不依赖分词器，谁跑都是同一个数。

代价是 `trigger` 的数字跟着变了：摘要那一幕的 `trigger=600` 指 **600 个字符**，不是 600 个 token。

用官方默认的 token 计数再跑一遍，同一个 600 是这样：

| trigger（真实 token） | 累计输入字符 | 单轮峰值 | 摘要调用 |
|---|---|---|---|
| 600 | 14986 | 39 条 | **0 次** |
| 400 | 9922 | 27 条 | 1 次 |
| 300 | 8298 | 19 条 | 2 次 |
| 200 | 6627 | 13 条 | 3 次 |

600 token 时摘要一次都没触发，账单和"不做治理"完全相同。所以**别单独抄 trigger 的数字**，它必须和 `token_counter` 的口径一起看。脚本里这一幕会自己跑给你看。

## 四个实验

| 幕 | 内容 | 关键发现 |
|---|---|---|
| 1 | 不做治理，20 轮的账单 | 累计输入 14986 字符，单轮峰值 39 条消息 |
| 2 | 裁剪（before_model） | 降到 3925 字符，峰值 5 条（前几轮还没触发裁剪，逐轮条数是 1 → 3 → 5） |
| 2 附 | 官方 trim_messages 的边界 | 预算给到刚好 4 条时，只剩一条系统提示 |
| 2 附 | 两种裁剪的差别 | 持久裁剪状态剩 6 条，瞬态裁剪状态留 12 条 |
| 3 | 删除（RemoveMessage） | 当前状态找不到密钥，12 个存档里 6 个仍有 |
| 4 | 摘要（SummarizationMiddleware） | 降到 7474 字符（trigger=**600 字符**），阈值越紧摘要次数越多 |
| 4 附 | trigger 的单位陷阱 | 换成真实 token 口径，同一个 600 一次都不触发 |

## 三个值得注意的结论

**1. 裁剪有两种写法，区别在状态动不动。**

`before_model` 返回状态更新会把裁剪结果写进状态，历史真的变短了。
`wrap_model_call` 改写 `request` 只影响本次调用，完整历史还在状态里。
省钱效果一样，选哪个看你要不要留完整历史做审计或长期记忆抽取。

**2. 删除动的是当前状态，不是存档。**

`RemoveMessage` 让消息从当前状态消失，模型后续也看不到。
但历史 checkpoint 里它还在。用户要求删除敏感信息时，存档清理要单独安排。

**3. `trim_messages` 的预算别贴着下限设。**

`max_tokens` 给到刚好等于保留条数时，输出可能只剩一条 SystemMessage，用户问题整段丢失。
留出系统提示和最近一轮的余量。

## 换成真实模型

改 `build_model()` 的返回值即可，其余代码一行都不用动：

```python
from langchain.chat_models import init_chat_model

def build_model():
    return init_chat_model("deepseek:deepseek-chat")
```

注意这一步需要对应的 provider 包。`init_chat_model("deepseek:...")` 会去找 `langchain-deepseek`，
没装的话直接抛 `ImportError`，报错信息里会给出安装命令。摘要在真实项目里建议单独指定一个便宜模型。

`LedgerModel` 里的 `bind_tools` 之所以要自己实现，是因为 `BaseChatModel` 在 1.x 里
默认会抛 `NotImplementedError`。换成真实模型后这段不再需要。

还有一件事别忘：真实场景要把 `token_counter` 一起换掉，否则 `trigger` 的数字会失去意义。

## 目录

```
context_governance_demo.py   全部代码，单文件
requirements.txt             依赖
README.md                    本文件
```

## 环境说明

验证环境：Python 3.13 / langchain 1.4.0 / langchain-core 1.6.0 / langgraph 1.2.11。
脚本里的统计口径是**字符数**，不是真实 token 数（原因见开头「先说口径」一节）。

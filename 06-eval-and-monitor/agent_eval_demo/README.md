# agent_eval_demo

《评测与监控：怎么知道它变好了还是变坏了》配套工程包。

一个 Agent 的评测与监控最小可跑集：从本地 trace 采集到评测集，从门禁脚本到监控
指标，还有国内云厂商的脱敏规则。**全部不依赖云平台账号，也不需要网络。**

---

## 三步跑通

```bash
# 1. 建独立环境（建议，别装进全局）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. 装依赖
pip install -r requirements.txt

# 3. 一键跑通全部演示与测试
python run_all.py
```

`run_all.py` 会把 8 个演示、2 个命令行工具和 33 个冒烟测试全跑一遍，最后给一张
汇总表。它返回 0，说明所有脚本的行为都符合设计预期。

> 第 8 个演示需要本地跑一个 OpenAI 兼容的模型端点（默认 LM Studio 的
> `http://127.0.0.1:1234/v1`）。服务没起来时它会**体面地跳过并返回 0**，
> 不影响其余部分。不想跑它可以先注释掉 `run_all.py` 里那一行。

没有 make 也不影响。想单独跑某项：

```bash
python demos/demo_eval.py                        # 跑评测
python -m src.gate --write-baseline              # 记基线
python -m src.gate                               # 门禁判定，退化时 exit 1
python -m src.monitor --trace reports/traces.jsonl   # 监控聚合
python -m pytest tests -q                        # 冒烟测试
```

---

## 八段演示分别讲什么

| 演示 | 命令 | 对应正文 | 跑出来的东西 |
|---|---|---|---|
| 1 本地 trace 采集 | `demo_collect.py` | 观测 | 一次带工具调用的请求抓到 **5 条 run**（2 llm / 2 tool / 1 chain），`parent_run_id` 全空 |
| 2 离线评测 | `demo_eval.py` | 评测集 + 分档 | good 版 1.0；bad 版整体 **0.667**，easy 档 1.0、hard 档 **0.333** |
| 3 门禁 | `demo_gate.py` | 上线前门禁 | 好的放行 exit 0、退化拦住 exit 1、阈值放宽到 0.6 就漏 |
| 4 回放缓存 | `demo_replay.py` | 让评测不烧钱 | 真实调用 6 → **0** → 2 → 6（改了 prompt 全部失效） |
| 5 长会话指标 | `demo_monitor.py` | 监控 | 单轮输入 24 → **249**（10.4 倍）；累计 1756 → **755**（省 57%） |
| 6 脱敏 | `demo_redact.py` | 上云前算账 | 默认规则漏掉 **5 种**形态，含阿里云 `LTAI`、腾讯云 `AKID` |
| 7 官方 evaluate | `demo_official.py` | 三种 data 形态 | 形态 A 报错、形态 B 表面成功一迭代就崩、形态 C 跑通 |
| 8 LLM 裁判 | `demo_judge.py` | 谁来判 | 位置交换 0 翻转；裁判输出里 **98% 以上是思考过程** |

> 耗时类数字（缓存那段的秒数、裁判的单次秒数）随机器变化，倍数与次数是确定的。
> 别的数字都是确定性的，在本机与正文里引用的值一致。

---

## 目录结构

```
agent_eval_demo/
├── run_all.py              一键验收
├── requirements.txt
├── Makefile                常用命令（Windows 上可用亦可不用）
├── src/
│   ├── agent.py            一个可替换的 Agent（脚本模型 / 本地真模型两条路）
│   ├── trace_store.py      观测：collect_runs → 本地 JSONL，含 token / 耗时 / 错误提取
│   ├── dataset.py          评测集：读写、版本冻结、从 trace 回捞样本
│   ├── evaluators.py       6 个判定器：从确定性规则到 LLM 裁判
│   ├── runner.py           执行器：离线跑评测；也提供官方链路那条路
│   ├── cassette.py         回放缓存：按输入哈希落盘，让回归不烧钱
│   ├── gate.py             门禁：对比基线，掉超阈值返回非 0
│   ├── monitor.py          监控：指标聚合 + 滚动基线告警 + 长会话曲线
│   ├── redact.py           脱敏：官方默认规则 + 国内云厂商 + 业务字段
│   ├── runtime.py          断网守卫、输出排版
│   └── paths.py            路径统一
├── demos/                  八段演示
├── data/
│   ├── eval_set.jsonl      评测集（3 条 easy + 3 条 hard）
│   ├── eval_set.frozen.json 版本指纹（自动生成）
│   └── baseline.json       门禁基线（自动生成）
├── cassettes/              回放缓存目录
├── reports/                trace 与评测结果落盘
└── tests/test_smoke.py     33 个冒烟测试
```

---

## 六个判定器

规则能判的别找模型。前 5 个是确定性的，成本接近 0，而且永不变卦。

| 判定器 | 类型 | 判什么 | 为什么需要它 |
|---|---|---|---|
| `json_valid` | 结构 | 输出能不能解析、字段齐不齐 | 结构坏了要第一时间知道 |
| `intent_exact` | 精确匹配 | 意图分类对不对 | 门禁的主指标 |
| `priority_exact` | 精确匹配 | 优先级对不对 | 多意图场景的主要退化点 |
| `order_id_grounded` | 结构化断言 | 输出的订单号必须真在输入里出现 | 防幻觉最便宜的一招 |
| `must_include` | 必含关键词 | 话术里必须有的话在不在 | 合规提示、品牌名这类硬要求 |
| `judge_rubric` | LLM 裁判 | 开放式质量 | 留给写不出规则的那部分 |

两个设计细节：

**判定器的输入是普通 dict，不是 langsmith 的 Run 对象。** 所以同一套判定逻辑既能在
本包的离线执行器里跑，也能通过 `evaluators.to_langsmith()` 挂到官方 `evaluate()` 上。

**跳过的判定器返回 `None`，不进平均分的分母。** 把「没跑」算成 0 分，你的平均分
会被无端拉低，而且没人看得出来。

---

## 关键结论速查

跑完一遍，下面这些都是可以直接验证的事实，不是说法。

| 结论 | 怎么验 |
|---|---|
| 一次带工具调用的请求抓到 5 条 run，且全部平级 | 演示 1 |
| `run` 上没有 `total_tokens` 这类属性，真值在 `message.kwargs.usage_metadata` | 演示 1 |
| 只看总分掉 33%，看分档才发现简单样本一条没坏、困难样本崩了 2/3 | 演示 2 |
| 结构判定器对这一版退化**完全无感**（`json_valid` 仍是 1.0） | 演示 2 |
| 门禁必须返回非 0 退出码，否则在 CI 里永远是绿的 | 演示 3 |
| 阈值放宽到 0.6，同一个退化就漏过去了 | 演示 3 |
| 缓存 key 必须含 system prompt，改一句话全部失效 | 演示 4 |
| 缓存坏掉时要回退到真实调用，不能抛错 | 演示 4 |
| 12 轮会话，治理后累计 token 省 57%，前 5 轮只省 24% | 演示 5 |
| 官方默认脱敏规则不认阿里云 `LTAI`、腾讯云 `AKID`、手机号、身份证、邮箱 | 演示 6 |
| `create_anonymizer()` **原地改写**传入对象，深层嵌套也改 | 演示 6 |
| 规则键名写错（`replacement`）不报错，但掩码文案被丢掉，统一输出 `[redacted]` | 演示 6 |
| 规则顺序会互相干扰：身份证号被手机号规则先截走一段 | 演示 6 |
| `evaluate()` 的 data 传 `list[dict]` 会报错，必须传 `Example` 对象 | 演示 7 |
| 离线模式能跑门禁，但 `experiment_id` / `url` 取不到 | 演示 7 |
| 裁判输出里 98% 以上是思考过程，按可见回复估成本会低估一个数量级 | 演示 8 |

---

## 几个必须记住的坑

**1. LangGraph 会把调用链拍平。**
一次请求的 5 条 run，`parent_run_id` 全是 `None`、`dotted_order` 全是单段、
根 run 的 `child_runs` 是 0。普通链式 Runnable 的父子关系是完整的，所以这是
LangGraph 执行器的行为。想重建时序，只能按 `start_time` 排序。

**2. 离线评测时要把 LangChain 的 run 上报关掉。**
`evaluate(upload_results=False)` 内部会把追踪上下文设成 local 模式
（源码在 `langsmith/evaluation/_runner.py`），它管得住 langsmith 自己的
`@traceable` run，但**管不住 LangChain 的 run**。日志里会刷一片 401，
来自后台上报线程。本包的 `agent.invoke_quiet()` 显式关掉它。
顺带一句：`LANGSMITH_TRACING=false` 这个环境变量在 `evaluate` 里不起作用。

**3. `data` 传 `list[dict]` 是网上教程里最常见的写法，也是错的。**
它报的错跟凭据无关，是形态问题，很容易把人带到「是不是 key 没配好」那个方向去。
要么用本包的 `runner.run_offline()`（直接吃普通 dict），要么老老实实构造
`Example` 对象。

**4. `vcrpy` 默认是没装的。**
官方的 `LANGSMITH_TEST_CACHE` 回放能力依赖它（`pip install "langsmith[vcr]"`）。
不装的话缓存开关**静默不生效**，你看不出来。

**5. 脱敏之前先深拷贝。**
`create_anonymizer()` 返回的就是你传进去的那个对象，深层嵌套的字段也一起被改。
为了上报而脱敏，顺手把内存里的业务数据也改了，这种 bug 很难查。
`redact()` 默认深拷贝。

---

## 接本地模型（演示 8）

需要一个 OpenAI 兼容端点。用 LM Studio：

1. 打开 LM Studio，加载一个 instruct 模型
2. 打开「本地服务」（默认 `http://127.0.0.1:1234`）
3. 跑 `python demos/demo_judge.py`

换别的端点或模型：

```bash
set LOCAL_BASE_URL=http://127.0.0.1:8000/v1
set LOCAL_MODEL=your-model-name
set LOCAL_API_KEY=anything
```

```bash
python demos/demo_judge.py --full    # 完整规模：绝对打分 6 次 + 成对比较 24 次
```

`--full` 会对齐正文里引用的调用次数。默认跑小规模，省时间。

---

## 已知边界

- **离线模式没有实验回链。** `experiment_id`、`url`、`comparison_url` 三个属性在
  `upload_results=False` 下会抛 `ValueError: Experiment not started yet`。
  能跑门禁，回链不到平台页面。
- **演示里的模型是脚本模型。** 按输入查表作答，完全确定、零成本、可复现。
  这是为了让「人为退化」这件事能被稳定演示。接真模型只要换掉 `src/agent.py`
  里的 `TriageModel`，判定器与门禁都不用动。
- **评测集只有 6 条。** 这是演示用的最小集，正好够说明分档的意义。真实项目里
  几百条起步，但**评测集不是越大越好，是越稳定越好**。
- **判定器的阈值 0.05 是随手定的。** 你的阈值要按自己系统的波动幅度定，
  定之前先跑十几次看分数自然抖动多少。

---

## 验收口径

干净目录解压后：

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

期望结果：

- 8 个演示脚本 + 2 个命令行工具 + 冒烟测试，**全部退出码 0**
- 演示 3 内部会故意触发一次 `exit 1`（它就是在证明门禁能拦住东西），
  这是脚本自检的一部分，脚本对外仍返回 0
- `run_all.py` 返回 0，并打印「全部通过」

如果演示 8 返回了非 0，先看它的输出 —— 通常是模型端点没起来，
或者加载的模型不是 instruct 模型（基座模型不会按 JSON 回话）。

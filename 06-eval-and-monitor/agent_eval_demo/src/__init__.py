"""agent_eval_demo —— 一个 Agent 的评测与监控最小可跑集。

模块分工：

  agent        一个可替换的 Agent（脚本模型 / 本地真模型两条路）
  trace_store  观测：collect_runs → 本地 JSONL，含 token / 耗时 / 错误提取
  dataset      评测集：读写、版本冻结、从 trace 回捞样本
  evaluators   判定器：6 个，从确定性规则到 LLM 裁判
  runner       执行器：离线跑评测（绕开 langsmith.evaluate 的三种形态坑）
  cassette     回放缓存：按输入哈希落盘，让回归测试不烧钱
  gate         门禁：对比基线，掉超阈值返回非 0
  monitor      监控：指标聚合 + 滚动基线告警
  redact       脱敏：官方默认规则 + 国内云厂商 + 业务字段
"""

__version__ = "1.0.0"

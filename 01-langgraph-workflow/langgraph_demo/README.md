# langgraph_demo · 高危操作审核 Agent（三关闭环）

《LangGraph 进阶：让 Agent 会等、会存、会回头》配套 demo。

## 内容

| 文件 | 演示 |
|---|---|
| main.py | interrupt 人工闸口 + InMemorySaver + 时间旅行回溯重跑（零 API key，直接跑） |
| sqlite_demo.py | SqliteSaver 持久化：跑到挂起后退出进程，重启后从断点恢复 |

## 运行

```bash
pip install "langgraph>=1.0" "langgraph-checkpoint-sqlite" langchain-core

# 闭环 demo（含时间旅行）
python main.py

# 持久化 demo（两步跑，模拟进程重启）
python sqlite_demo.py run
python sqlite_demo.py resume
```

## 环境

- Python 3.10+，LangGraph 1.x
- 无需任何 API key（模型为内置假模型，按对话内容出牌）

## 对应正文关卡

1. 会等：工具内 interrupt + Command(resume)
2. 会存：checkpointer 换 SqliteSaver，thread_id 断点续跑
3. 会回：get_state_history + update_state 开分支重跑

# agent_eval_demo —— 常用命令
#
# Windows 上没有 make 也不影响：每个目标对应的命令都写在 README 里，
# 或者直接跑 python run_all.py。

PY ?= python

.PHONY: help install collect eval gate baseline replay monitor redact official judge test all clean

help:
	@echo "make install   装依赖"
	@echo "make collect   采集本地 trace（演示 1）"
	@echo "make eval      跑离线评测，good 与 bad 两版（演示 2）"
	@echo "make gate      门禁，含退化拦截自检（演示 3）"
	@echo "make baseline  把当前结果记成基线"
	@echo "make replay    回放缓存（演示 4）"
	@echo "make monitor   长会话指标 + trace 聚合（演示 5）"
	@echo "make redact    脱敏规则对照（演示 6）"
	@echo "make official  官方 evaluate 三种 data 形态（演示 7）"
	@echo "make judge     LLM 裁判（演示 8，需要本地模型）"
	@echo "make test      冒烟测试"
	@echo "make all       一键全部跑一遍"

install:
	$(PY) -m pip install -r requirements.txt

collect:
	$(PY) demos/demo_collect.py

eval:
	$(PY) demos/demo_eval.py

gate:
	$(PY) demos/demo_gate.py

baseline:
	$(PY) -m src.gate --write-baseline

replay:
	$(PY) demos/demo_replay.py

monitor:
	$(PY) demos/demo_monitor.py
	$(PY) -m src.monitor --trace reports/traces.jsonl

redact:
	$(PY) demos/demo_redact.py

official:
	$(PY) demos/demo_official.py

judge:
	$(PY) demos/demo_judge.py

test:
	$(PY) -m pytest tests -q

all:
	$(PY) run_all.py

clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ('reports','cassettes')]; [pathlib.Path(p).unlink(missing_ok=True) for p in ('data/baseline.json','data/eval_set.frozen.json')]"

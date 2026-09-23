# -*- coding: utf-8 -*-
"""并行分发：Send 的正确姿势与踩坑原文。

跑法：python demos/demo_parallel.py
"""
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage  # noqa: E402

from src.parallel import LANDED, build_parallel_graph, try_wrong_way  # noqa: E402

LINE = "=" * 74
print(LINE)
print("并行分发 · Send 的正确姿势与错误姿势")
print(LINE)

# ---------------- 正确：从条件边返回 ----------------
print("\n[正确] Send 从条件边返回")
app = build_parallel_graph(["A", "B", "C"])
out = app.invoke({"messages": [HumanMessage(content="三份活一起发")], "results": []})
print(f"  三个 worker 各自拿到的 payload = {sorted(LANDED)}")
print("  三条结果消息：")
for m in out["results"]:
    print(f"     {m.content}")
assert sorted(LANDED) == ["A", "B", "C"], "三个 worker 没有各拿到自己的 payload"
print("  >> 每个 worker 只看到自己那份，互不干扰。")

# ---------------- 错误：从普通节点返回 ----------------
print("\n[错误] Send 从普通节点返回")
err = try_wrong_way(["A", "B", "C"])
print(f"  报错原文：{err}")
assert "InvalidUpdateError" in err, "预期会抛 InvalidUpdateError，实际不是"
print("  >> 网上不少示例把 Send 写在节点体里，在这个版本上跑不通。")

print("\n" + LINE)
print("分清楚两个原语的活：")
print("  Command(goto=...)       改的是「控制流去哪」")
print("  Send(node, payload)     改的是「这一份活派给谁、带什么数据」")
print("  Send 只能从条件边返回，这是位置上的硬要求。")
print(LINE)

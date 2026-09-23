# -*- coding: utf-8 -*-
"""官方预制件对照：supervisor 与 swarm。

这个脚本需要额外装两个包：
    pip install langgraph-supervisor langgraph-swarm

没装也能跑，会走降级分支，把要观察的东西用文字列出来。
跑法：python demos/demo_prebuilt.py
"""
import io
import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage  # noqa: E402

from src.agents import make_all_experts  # noqa: E402
from src.handoff import describe_trajectory  # noqa: E402
from src.spy_model import SpyChatModel  # noqa: E402

LINE = "=" * 74
print(LINE)
print("官方预制件对照 · supervisor 与 swarm")
print(LINE)

try:
    import langgraph_supervisor
    from langgraph_supervisor import create_supervisor

    HAS_SUP = True
except ImportError:
    HAS_SUP = False

try:
    import langgraph_swarm
    from langgraph_swarm import create_swarm

    HAS_SWARM = True
except ImportError:
    HAS_SWARM = False

print(f"\nlanggraph-supervisor：{'已安装' if HAS_SUP else '未安装'}")
print(f"langgraph-swarm     ：{'已安装' if HAS_SWARM else '未安装'}")

if not (HAS_SUP or HAS_SWARM):
    print("\n" + "-" * 74)
    print("降级说明")
    print("-" * 74)
    print("""
这两个包是可选依赖，本文的其余四个 demo 都不需要它们。

装法：
    pip install langgraph-supervisor langgraph-swarm

它们解决的问题是：把「主管 + 专家」这套常见套路预先拼好，
省掉自己画图那段代码。装完之后值得看的四件事：

  1. 版号看着小（supervisor 0.0.31 / swarm 0.1.0），但依赖声明写着
     langgraph>=1.0.2,<2.0.0，官方声明支持 1.x，不是不能用。

  2. 主管拿到的工具是 transfer_to_<agent_name>，一个专家一个。
     读源码会发现它不接收 task 参数，把整份 state 交给目标 Agent。
     所以「转交时只带任务」这件事，预制件不做，得自己动手。

  3. 转交工具的默认描述是 Ask agent '<agent_name>' for help。
     模型要在几个专家之间做选择，全靠这句话，实际使用必须自己写。

  4. output_mode 控制子 Agent 交回多少，默认 last_message，
     只交最后结论；改成 full_history 会把中间的 tool_call 也交回来。
""")
    print("装好之后重跑本脚本，会打印出实测的拓扑与轨迹。")
    print(LINE)
    raise SystemExit(0)

# ---------------- supervisor 实测 ----------------
if HAS_SUP:
    print("\n" + "-" * 74)
    print("[supervisor] 拓扑与一次转交的轨迹")
    print("-" * 74)

    SpyChatModel.reset()
    sup = SpyChatModel(tag="sup", script=[
        {"calls": [("transfer_to_order_expert", {})]},
        {"text": "订单情况我汇总给你。"},
    ])
    experts = make_all_experts({
        "order": [{"text": "订单已签收，可退 296.00。"}],
        "kb": [{"text": "签收后 7 天内可申请。"}],
        "report": [{"text": "退款率 3.2%。"}],
    })

    sig = inspect.signature(create_supervisor)
    print("  create_supervisor 关键参数：")
    for p in ("parallel_tool_calls", "output_mode", "add_handoff_back_messages"):
        if p in sig.parameters:
            print(f"     {p:28} default={sig.parameters[p].default!r}")

    wf = create_supervisor(
        [experts["order_expert"], experts["kb_expert"], experts["report_expert"]],
        model=sup,
        prompt="你是主管。",
    )
    app = wf.compile()

    nodes = sorted(n for n in app.get_graph().nodes if not n.startswith("__"))
    print(f"\n  图里的节点 = {nodes}")
    print(f"  主管手里的工具 = {SpyChatModel.tools_offered.get('sup')}")

    out = app.invoke(
        {"messages": [HumanMessage(content="查一下订单状态")]},
        {"recursion_limit": 40},
    )
    print("\n  轨迹：")
    for i, label in enumerate(describe_trajectory(out["messages"]), 1):
        print(f"     {i}. {label}")
    print(f"\n  消息总数 = {len(out['messages'])}")
    print("  >> 对比手写版：预制件多出 transfer_back_to_supervisor 的回程消息，")
    print("     一次来回净增 4 条消息。")

# ---------------- swarm 实测 ----------------
if HAS_SWARM:
    print("\n" + "-" * 74)
    print("[swarm] 拓扑")
    print("-" * 74)

    SpyChatModel.reset()
    experts = make_all_experts({
        "order": [{"text": "订单已签收。"}],
        "kb": [{"text": "7 天内可申请。"}],
        "report": [{"text": "退款率 3.2%。"}],
    })
    wf2 = create_swarm(
        [experts["order_expert"], experts["kb_expert"]],
        default_active_agent="order_expert",
    )
    app2 = wf2.compile()
    nodes2 = sorted(n for n in app2.get_graph().nodes if not n.startswith("__"))
    print(f"  图里的节点 = {nodes2}")
    print("  >> 没有中央主管节点，节点就是各 Agent 本身。")
    try:
        out2 = app2.invoke({"messages": [HumanMessage(content="查一下订单")]}, {"recursion_limit": 30})
        print(f"  返回状态的键 = {sorted(out2.keys())}")
        print("  >> 当前活跃 Agent 不在返回状态里，想知道最后是谁在说话，得去翻消息。")
    except Exception as e:  # noqa: BLE001
        print(f"  invoke 未跑通：{type(e).__name__}: {str(e)[:100]}")

print(LINE)

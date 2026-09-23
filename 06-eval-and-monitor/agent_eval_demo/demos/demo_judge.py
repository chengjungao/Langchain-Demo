"""演示 8：LLM 裁判 —— 一致性、位置偏差、以及裁判自己的回归。

需要本地模型（默认 LM Studio 的 OpenAI 兼容端点）。服务不在的时候脚本会体面
地跳过并返回 0，不会让整条流水线挂掉。

四件事：
  1. 绝对打分：给候选回复打 1 到 5 分
  2. 成对比较：同一个问题的两条回复，哪条更好
  3. 位置交换：同一对答案换顺序再问一遍，看结论会不会跟着位置走
  4. 算账：裁判的输出里有多少是思考过程

默认规模小，加 --full 跑完整规模（会对齐正文里引用的调用次数）。

运行：python demos/demo_judge.py [--full]
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src import paths  # noqa: E402
from src.agent import local_llm_available, local_model  # noqa: E402
from src.evaluators import (  # noqa: E402
    JudgeCost,
    judge_absolute,
    pairwise_swap,
)
from src.runtime import banner, force_offline, section  # noqa: E402

force_offline()
banner("演示 8 · LLM 裁判：位置要交换着测，裁判自己也要有回归")

FULL = "--full" in sys.argv

if not local_llm_available():
    section("跳过")
    print("  本地模型端点没起来，这个演示跳过了。")
    print("  它需要一个 OpenAI 兼容端点，默认 http://127.0.0.1:1234/v1（LM Studio）。")
    print("  起服务的方法：打开 LM Studio → 加载一个 instruct 模型 → 打开本地服务。")
    print("  也可以用环境变量指向别处：LOCAL_BASE_URL / LOCAL_MODEL / LOCAL_API_KEY")
    print("\n  这一项不影响其余演示：确定性判定器那部分不需要任何模型。")
    print("\n" + "=" * 84)
    print("演示 8 结束（跳过）")
    print("=" * 84)
    sys.exit(0)

model = local_model()
print(f"  裁判端点：http://127.0.0.1:1234/v1 ｜ 模型：{model.model_name}")

# ---------------------------------------------------------------- 样本
CASES = [
    {
        "q": "我要退货，订单 A1001 还没到",
        "good": "已查到订单 A1001 目前在运输途中，预计明天送达。签收前您可以直接拒收，"
                "或者在订单页点申请退款。我已经帮您标记加急，24 小时内会有客服联系您确认。",
        "mid": "订单 A1001 在途，您可以申请退货。",
        "bad": "可以退货的，您去申请一下。",
    },
    {
        "q": "发票抬头写错了，怎么改",
        "good": "发票抬头修改需要先作废原发票再重开。请在订单详情页点「发票管理」，"
                "选择「作废重开」，填新的抬头信息，一般 1 个工作日内开出。"
                "如果原发票已用于报销，请先联系财务退回。",
        "mid": "在发票管理里可以改。",
        "bad": "改不了，只能重开。",
    },
]

cost_abs = JudgeCost()

# ---------------------------------------------------------------- 1 绝对打分
section("一、绝对打分")
n_cases = len(CASES) if FULL else 1
print(f"  {'问题':<18}{'档次':<7}{'分数':<7}{'输出tok':<10}{'其中思考':<10}{'耗时'}")
print("  " + "-" * 62)
for c in CASES[:n_cases]:
    for tier in ("good", "mid", "bad"):
        s = judge_absolute(model, c["q"], c[tier], cost_abs)
        d = cost_abs.details[-1]
        print(f"  {c['q'][:16]:<18}{tier:<7}{str(s):<7}{str(d['out_tok']):<10}"
              f"{str(d['reason_tok']):<10}{d['s']}s")
print("\n  注意最后两列：可见回复只有一个小 JSON，但输出 token 大部分是思考过程。")

# ---------------------------------------------------------------- 2 + 3 成对比较与位置交换
section("二、成对比较 + 位置交换")
PAIRS = [("good", "bad", "good"), ("good", "mid", "good"), ("mid", "bad", "mid")]
if not FULL:
    PAIRS = PAIRS[:1]
ROUNDS = 2 if FULL else 1

cost_pair = JudgeCost()
print(f"  {'问题':<18}{'对比':<13}{'轮':<4}{'顺序(A,B)':<11}{'顺序(B,A)':<11}{'结论'}")
print("  " + "-" * 78)

flip = valid = parse_fail = 0
for c in CASES:
    for a, b, gold in PAIRS:
        for rnd in range(1, ROUNDS + 1):
            r = pairwise_swap(model, c["q"], c[a], c[b], cost_pair)
            if r["stable"] is None:
                parse_fail += 1
                print(f"  {c['q'][:16]:<18}{a + '-vs-' + b:<13}{rnd:<4}"
                      f"{str(r['v1']):<11}{str(r['v2']):<11}解析失败，不计入对照")
                continue
            valid += 1
            if r["flip"]:
                flip += 1
                note = "✘ 换个位置就翻"
            else:
                right = r["picked"][0] == c[gold] if r["picked"] else False
                note = "✔ 稳定指向" + ("更优的那条" if right else "了预设之外的答案")
            print(f"  {c['q'][:16]:<18}{a + '-vs-' + b:<13}{rnd:<4}"
                  f"{str(r['v1']):<11}{str(r['v2']):<11}{note}")

section("统计")
print(f"  有效对照 {valid} 组，交换位置后结论翻转 {flip} 组"
      f"（{flip / max(valid, 1) * 100:.0f}%）")
print(f"  解析失败 {parse_fail} 次（输出被截断或没按 JSON 返回）")
if valid and flip == 0:
    print("\n  这一批里没测出位置偏差。但要把话说清楚：**小样本的结果不能推广成")
    print("  「位置偏差不存在」。** 测它的成本很低，两个顺序各跑一遍就有答案。")
    print("  两组结论不一致的样本单独挑出来人工看，这一步不能省。")

# ---------------------------------------------------------------- 4 算账
section("三、裁判的账")
abs_rep = cost_abs.report()
pair_rep = cost_pair.report()
print(f"  {'裁判任务':<14}{'调用次数':<10}{'输出token':<12}{'其中思考':<22}{'单次平均'}")
print("  " + "-" * 70)
print(f"  {'绝对打分':<14}{abs_rep['calls']:<10}{abs_rep['out_tokens']:<12}"
      f"{str(abs_rep['reasoning_tokens']) + '（' + str(abs_rep['reasoning_share']) + '%）':<22}"
      f"{abs_rep['avg_seconds']}s")
print(f"  {'成对比较':<14}{pair_rep['calls']:<10}{pair_rep['out_tokens']:<12}"
      f"{str(pair_rep['reasoning_tokens']) + '（' + str(pair_rep['reasoning_share']) + '%）':<22}"
      f"{pair_rep['avg_seconds']}s")
print(f"\n  被截断的调用：绝对打分 {abs_rep['truncations']} 次，成对比较 {pair_rep['truncations']} 次")
if pair_rep["truncations"]:
    print("  ⚠ 输出被 max_tokens 截断时，解析会失败。判成「翻转」是误判。")
    print("    这一条在写测量脚本时最容易踩：解析失败必须单独统计，不能混进翻转数。")

print("\n  **按「可见回复的长度」估算裁判成本，会低估一个数量级。**")
print("  估成本要按总输出 token 算，推理型模型的账单几乎全在思考上。")

# ---------------------------------------------------------------- 5 裁判的回归
section("四、裁判自己也要有回归")
print("  换了裁判模型、或者同一个模型升了版本，必须在一小批人工标注过的样本上")
print("  先验一致性，再用它跑全量。否则你会把裁判的变化，当成系统的变化。")
print("\n  本次实测里就撞上过一次：有一组对比，按预设 mid 更好，但裁判稳定地选了 bad。")
print("  回头重读两条回复才发现是标注错了 —— mid 写的是「在发票管理里可以改」，")
print("  而真实规则是发票不能直接改、必须先作废再重开，所以裁判选 bad 反而更准。")
print("\n  裁判和人不一致时，先别急着说裁判错。很可能你的标注标准本身就有争议。")

# ---------------------------------------------------------------- 自检
section("自检")
ok = True
if cost_abs.calls == 0:
    print("  ✘ 绝对打分一次都没跑成")
    ok = False
if parse_fail > 0 and valid == 0:
    print("  ✘ 成对比较全部解析失败，检查 max_tokens 是不是被推理过程吃满了")
    ok = False
if ok:
    print(f"  ✔ 裁判跑通：绝对打分 {cost_abs.calls} 次，成对比较 {cost_pair.calls} 次")
    print(f"    模型：{model.model_name} ｜ temperature=0")

if not FULL:
    print("\n  这次跑的是小规模。加 --full 跑完整规模（" +
          "绝对打分 2 个问题 × 3 档 = 6 次，成对比较 3 对 × 2 顺序 × 2 轮 = 24 次）。")

print("\n" + "=" * 84)
print("演示 8 结束")
print("=" * 84)
sys.exit(0 if ok else 1)

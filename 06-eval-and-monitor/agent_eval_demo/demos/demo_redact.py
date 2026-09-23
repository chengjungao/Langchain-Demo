"""演示 6：脱敏 —— 上云之前，两个默认规则管不住的地方。

官方给了 create_secret_anonymizer()，认一批主流密钥形态。实测下来有两类东西
它不管：

  1. 国内云厂商的凭据（阿里云 LTAI、腾讯云 AKID 都不在默认规则里）
  2. 业务字段（手机号、身份证、订单号、银行卡，一个都不管）

还有两个高风险行为，别在生产里踩：
  * create_anonymizer(rules) 会原地改动传进去的对象，深层嵌套也一样
  * max_depth 设小了会静默漏字段，不报错

运行：python demos/demo_redact.py
"""
import copy
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langsmith.anonymizer import create_anonymizer  # noqa: E402
from langsmith.secret import LangSmithSecret  # noqa: E402

from src import redact as R  # noqa: E402
from src.runtime import banner, force_offline, section  # noqa: E402

force_offline()
banner("演示 6 · 脱敏：默认规则认什么，不认什么")

# ---------------------------------------------------------------- 对照表
section("同一批敏感串，两档规则的对照")
secret_level = dict(R.probe_table("secret"))
business_level = dict(R.probe_table("business"))
print(f"  {'形态':<20}{'官方默认规则':<14}{'本包 business 档'}")
print("  " + "-" * 56)
for name in R.PROBES:
    a = "脱敏" if secret_level[name] else "原样带出"
    b = "脱敏" if business_level[name] else "原样带出"
    mark = ""
    if not secret_level[name] and business_level[name]:
        mark = "   <== 默认规则管不住，靠自定义规则补上"
    print(f"  {name:<20}{a:<14}{b}{mark}")

missing = [n for n in R.PROBES if not secret_level[n] and business_level[n]]
print(f"\n  默认规则漏掉的形态：{len(missing)} 个")
print("  官方规则表是按国际厂商写的，国内团队上云第一件事就是补这张表。")

# ---------------------------------------------------------------- 原地改动
section("高风险行为一：脱敏会不会改掉你的原始对象")
rules = [{"pattern": r"1[3-9]\d{9}", "replace": "[手机号]"}]
anon = create_anonymizer(rules)
orig = {"q": "用户 13812345678 的订单", "meta": {"tel": "13900000000"}}
snapshot = copy.deepcopy(orig)
res = anon(orig)
print(f"  返回的是不是同一个对象：{res is orig}")
print(f"  处理之后 orig   = {orig}")
print(f"  处理之前快照    = {snapshot}")
print(f"  原始对象被改动  ：{'是' if orig != snapshot else '否'}")
print("\n  ⚠ 为了上报而脱敏，顺手把内存里的业务数据也改了。这种 bug 很难查。")
print("    本包的 redact() 默认先深拷贝：")

safe_src = {"q": "用户 13812345678 的订单", "meta": {"tel": "13900000000"}}
safe_snap = copy.deepcopy(safe_src)
safe_out = R.redact(safe_src, level="business")
print(f"    返回对象内容 = {safe_out}")
print(f"    原始对象是否被改：{'是' if safe_src != safe_snap else '否'}")

section("  （嵌套结构里也一样）")
inner = {"order": "A1001"}
outer = {"wrap": inner}
anon2 = create_anonymizer([{"pattern": r"A\d{4}", "replace": "[订单号]"}])
r2 = anon2(outer)
print(f"  内层原始 dict 被改成了：{inner}")
print(f"  返回：{r2}")

# ---------------------------------------------------------------- max_depth
section("高风险行为二：max_depth 设小了会静默漏字段")
deep = {"a": {"b": {"c": {"d": {"e": "13812345678"}}}}}
shallow = create_anonymizer(rules, max_depth=2)
print(f"  max_depth=2 的处理结果：{shallow(copy.deepcopy(deep))}")
full = create_anonymizer(rules)
print(f"  不设深度限制的结果  ：{full(copy.deepcopy(deep))}")
print("\n  超深度的值原样带出去，不报错。上云前一定跑一遍「残留检查」。")
leaks = R.Redactor("business").leaks(deep, shallow(deep))
print(f"  本包 Redactor.leaks() 能查出残留：{leaks if leaks else '（这组探针没命中）'}")

# ---------------------------------------------------------------- 规则键名
section("一个很容易写错的地方")
print("  规则里那个键叫 replace，不是 replacement。")
ok_anon = create_anonymizer([{"pattern": r"1[3-9]\d{9}", "replace": "[手机号]"}])
wrong = create_anonymizer([{"pattern": r"1[3-9]\d{9}", "replacement": "[手机号]"}])
print(f"  写对 replace      ：{ok_anon({'v': '13812345678'})}")
print(f"  写成 replacement  ：{wrong({'v': '13812345678'})}")
print("\n  注意规律：pattern 照样匹配，但你写的掩码文案被丢掉，")
print("  输出统一变成固定的 [redacted]。")
print("  你会看到脱敏「生效了」，不会发现文案是错的 —— 这才是它阴的地方。")
print("  下游要是靠掩码文案统计各类敏感信息的命中数，从这里就开始全错。")
nomatch = create_anonymizer([{"pattern": r"ZZZZ", "replacement": "[手机号]"}])
print(f"\n  对照：pattern 完全不匹配时不动它 → {nomatch({'v': '13812345678'})}")

# ---------------------------------------------------------------- 规则顺序
section("自定义规则会互相干扰")
id_card = "110101199003078515"
wide_first = [
    {"pattern": r"1[3-9]\d{9}", "replace": "[手机号]"},
    {"pattern": r"\d{6}(19|20)\d{2}[01]\d[0-3]\d\d{3}[\dXx]", "replace": "[身份证]"},
]
narrow_first = list(reversed(wide_first))
print(f"  宽的在前：{create_anonymizer(wide_first)({'v': id_card})}")
print(f"  窄的在前：{create_anonymizer(narrow_first)({'v': id_card})}")
print("\n  身份证号里有 11 位连续数字落进了手机号的形状，谁先命中谁说了算。")
print("  规则的顺序要按特异性排，窄的规则放前面。")

# ---------------------------------------------------------------- 完整示例
section("一个完整的脱敏示例（模拟一条要上报的 trace）")
payload = {
    "inputs": {"q": "我的手机号 13812345678，订单 A1001 要退，发票抬头错了"},
    "outputs": {"reply": "已受理，联系人邮箱 someone@example.com"},
    "extra": {
        "aliyun_ak": "LTAI5tAbCdEfGhIjKlMnOpQr",
        "openai_key": "sk-proj-AbCdEf1234567890AbCdEf1234567890AbCdEf",
        "id_card": "11010119900307123X",
    },
}
print("  脱敏前：")
print("  " + json.dumps(payload, ensure_ascii=False)[:200] + " ...")

for level in ("secret", "cn", "strict"):
    out = R.redact(payload, level=level)
    text = json.dumps(out, ensure_ascii=False)
    leaked = [n for n, p in R.PROBES.items() if p in text]
    print(f"\n  level={level:<9} 残留：{leaked if leaked else '无'}")

print("\n  结论：level 选到 strict 才把订单号也盖住。")
print("  订单号这类只在特定业务里算敏感，放开会误伤，所以单独成一档。")

print("\n" + "=" * 84)
print("演示 6 结束")
print("=" * 84)

---
name: refund-flow
description: 处理客户退款申请的标准工序。当用户提出退款、退货、退钱、订单取消、售后补偿等请求时使用，覆盖查单、试算、审批判断、提交与回执全流程。
allowed-tools: query_order calc_refund submit_refund search_policy read_skill
---

# 退款处理工序

按下面 5 步走，不要跳步。每一步都要拿到上一步的返回值再继续。

## 第 1 步：查订单

调用 `query_order(order_no, tenant_id)`。

拿不到订单时直接回复客户「未查到该订单」，不要继续往下走。

## 第 2 步：查政策

调用 `search_policy(query)`，query 里放本次退款的关键词，例如「审批」「运费」。

政策是唯一的判断依据。不要凭印象回答，也不要用别的订单的历史结论代替。

## 第 3 步：试算金额

调用 `calc_refund(order_no, reason_code, has_invoice)`。

`reason_code` 只能从这四个里选：`quality`、`wrong_item`、`no_longer_needed`、`price_protection`。

客户没有说明原因时，先追问，不要自己替他选一个。

## 第 4 步：判断是否需要审批

试算结果里的 `need_approval` 为 true 时，走人工审批，先给客户回一条预计 1 个工作日的说明。

## 第 5 步：提交并回执

调用 `submit_refund(order_no, amount, reason_code, operator)`。

回执里必须带上四个要素：工单号、退款金额、预计到账时间、需要客户配合的事项。

金额、时间、政策条款三项全部来自工具返回值，不要自己编。

详细规则见 `references/policy.md`。

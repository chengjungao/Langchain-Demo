# -*- coding: utf-8 -*-
"""订单服务 MCP Server（stdio 传输）。

三个工具：query_order / calc_refund / submit_refund。

这里的 docstring 会原样变成客户端侧工具的 description，
也就是模型真正读到的那段文字。写 docstring 就是在写提示词。
"""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

# 客户端把本进程的 stderr 直接透传到终端，压掉 INFO 日志让 demo 输出干净
logging.getLogger("mcp").setLevel(logging.WARNING)

mcp = FastMCP("order-server")

# 演示用的内存数据，换成真实库只需改这一层
_ORDERS: dict[str, dict] = {
    "SO20260912001": {
        "order_no": "SO20260912001",
        "tenant": "t-acme",
        "status": "paid",
        "amount": 1299.00,
        "currency": "CNY",
        "items": [{"sku": "SKU-A1", "name": "机械键盘", "qty": 1, "price": 899.00},
                  {"sku": "SKU-B2", "name": "腕托", "qty": 2, "price": 200.00}],
        "paid_at": "2026-09-05T10:12:00",
        "shipped": False,
    },
    "SO20260912002": {
        "order_no": "SO20260912002",
        "tenant": "t-acme",
        "status": "shipped",
        "amount": 458.00,
        "currency": "CNY",
        "items": [{"sku": "SKU-C3", "name": "电竞鼠标", "qty": 1, "price": 458.00}],
        "paid_at": "2026-08-28T09:00:00",
        "shipped": True,
    },
}


@mcp.tool()
def query_order(order_no: str, tenant_id: str) -> dict:
    """按订单号查询订单详情。

    返回订单状态、金额、明细、支付时间与是否已发货。
    调用退款流程前必须先查订单，拿到 status 与 shipped 两个字段。

    Args:
        order_no: 订单号，例如 SO20260912001
        tenant_id: 租户标识，用于做数据隔离
    """
    order = _ORDERS.get(order_no)
    if order is None or order["tenant"] != tenant_id:
        return {"found": False, "order_no": order_no}
    return {"found": True, **order}


@mcp.tool()
def calc_refund(order_no: str, reason_code: str, has_invoice: bool) -> dict:
    """试算退款金额，不产生任何实际动作。

    根据订单状态与退款原因计算可退金额：
    已发货订单扣除运费 12 元；已开票订单扣除税点 6%；
    reason_code 取值为 quality / wrong_item / no_longer_needed / price_protection。

    Args:
        order_no: 订单号
        reason_code: 退款原因编码
        has_invoice: 是否已开具发票
    """
    order = _ORDERS.get(order_no)
    if order is None:
        return {"ok": False, "message": "订单不存在"}

    amount = float(order["amount"])
    deductions: list[str] = []
    if order["shipped"]:
        amount -= 12.00
        deductions.append("运费 12.00")
    if has_invoice:
        amount -= round(amount * 0.06, 2)
        deductions.append("税点 6%")
    if reason_code == "quality":
        amount = float(order["amount"])
        deductions = ["质量问题全额退，免除运费与税点"]

    return {
        "ok": True,
        "order_no": order_no,
        "refundable": round(amount, 2),
        "currency": order["currency"],
        "deductions": deductions,
        "need_approval": amount > 500.00,
    }


@mcp.tool()
def submit_refund(order_no: str, amount: float, reason_code: str, operator: str) -> dict:
    """提交退款申请，写入工单系统。

    金额超过 500 元会进入人工审批队列，返回状态为 pending_approval；
    否则直接受理，返回 accepted。本工具会产生真实的退款动作，调用前须已完成试算。
    """
    order = _ORDERS.get(order_no)
    if order is None:
        return {"ok": False, "message": "订单不存在"}

    ticket = f"RF{order_no[-6:]}"
    state = "pending_approval" if amount > 500.00 else "accepted"
    return {
        "ok": True,
        "ticket_no": ticket,
        "order_no": order_no,
        "amount": amount,
        "reason_code": reason_code,
        "operator": operator,
        "state": state,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

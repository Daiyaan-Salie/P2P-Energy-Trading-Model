#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3c_pyteal_dex_Rev5.py

Rev5 smart contract using the WORKING Rev4 structure as baseline (no boxes, no Maybe),
with ONLY the minimal changes required:

1) Dynamic-MCP computed ON-CHAIN per interval:
   MCP = FLOOR + (TARIFF - FLOOR) * clamp(demand/supply, 0..1)

2) ZAR token uses stablecoin-style precision:
   ZAR ASA decimals = 6  (micro-ZAR base units)

3) Feeder cap enforced per interval with pro-rata scaling done off-chain,
   while the contract enforces that total recorded trades do not exceed allowed volume.

Design choice (to avoid PyTeal version/box/Maybe compile issues):
- Interval state is stored in GLOBALS (one active interval at a time), not boxes.

Units:
- qty_units: integer units of 0.01 kWh (1 unit = 0.01 kWh)
- KWH ASA decimals = 3 (milli-kWh) => transfer amount = qty_units * 10
- ZAR ASA decimals = 6 (micro-ZAR)
- mcp_micro: micro-ZAR per kWh (integer)

Payment formula:
  payment_microZAR = max(1, floor(qty_units * mcp_micro / 100))
because qty_units represent (qty_units/100) kWh.
"""

import json
from pyteal import (
    abi,
    App,
    Approve,
    Assert,
    BareCallActions,
    Bytes,
    CallConfig,
    Concat,
    Global,
    Gtxn,
    If,
    Int,
    Itob,
    Log,
    OnCompleteAction,
    Router,
    Seq,
    Txn,
    WideRatio,
    Expr,
)

# -------- Global keys --------
K_KWH = Bytes("kwh_id")             # uint: ASA id
K_ZAR = Bytes("zar_id")             # uint: ASA id

K_CUR_INT = Bytes("cur_interval")   # uint: current interval id

# Price bounds + current MCP (micro-ZAR/kWh)
K_FLOOR_MICRO = Bytes("floor_micro")    # uint
K_TARIFF_MICRO = Bytes("tariff_micro")  # uint
K_MCP_MICRO = Bytes("mcp_micro")        # uint

# Interval volume enforcement (0.01 kWh units)
K_FEEDER_CAP_U = Bytes("feeder_cap_u")  # uint: cap units per interval
K_ALLOWED_U = Bytes("allowed_u")        # uint: min(supply, demand, cap)
K_TRADED_U = Bytes("traded_u")          # uint: cumulative recorded trades in interval

# -------- Router --------
router = Router(
    "P2P Energy DEX Rev5 (Rev4-baseline, Dynamic-MCP, microZAR)",
    BareCallActions(
        no_op=OnCompleteAction(action=Approve(), call_config=CallConfig.CREATE | CallConfig.CALL),
        opt_in=OnCompleteAction.never(),
        close_out=OnCompleteAction.never(),
        clear_state=OnCompleteAction.never(),
        update_application=OnCompleteAction.never(),
        delete_application=OnCompleteAction.never(),
    ),
)


def min3(a: Expr, b: Expr, c: Expr) -> Expr:
    """Return min(a,b,c) using only If (PyTeal-version-safe)."""
    ab = If(a <= b, a, b)
    return If(ab <= c, ab, c)


@router.method
def setup_market(
    kwh_asset: abi.Asset,
    zar_asset: abi.Asset,
    floor_micro: abi.Uint64,
    tariff_micro: abi.Uint64,
    feeder_cap_units: abi.Uint64,
    *,
    output: abi.String,
) -> Expr:
    """
    One-time setup (creator only):
      - store ASA ids for kWh and ZAR
      - store pricing bounds (micro-ZAR/kWh)
      - store feeder cap (0.01 kWh units per interval)
      - init interval state
    """
    return Seq(
        Assert(Txn.sender() == Global.creator_address()),
        Assert(tariff_micro.get() >= floor_micro.get()),
        App.globalPut(K_KWH, kwh_asset.asset_id()),
        App.globalPut(K_ZAR, zar_asset.asset_id()),
        App.globalPut(K_FLOOR_MICRO, floor_micro.get()),
        App.globalPut(K_TARIFF_MICRO, tariff_micro.get()),
        App.globalPut(K_FEEDER_CAP_U, feeder_cap_units.get()),
        App.globalPut(K_CUR_INT, Int(0)),
        App.globalPut(K_MCP_MICRO, Int(0)),
        App.globalPut(K_ALLOWED_U, Int(0)),
        App.globalPut(K_TRADED_U, Int(0)),
        output.set(Bytes("Market setup complete! (Rev5 microZAR)")),
    )


@router.method
def execute_interval(
    interval: abi.Uint64,
    sum_supply_units: abi.Uint64,
    sum_demand_units: abi.Uint64,
    *,
    output: abi.String,
) -> Expr:
    """
    Begin a new interval and compute MCP ON-CHAIN.

    Inputs:
      - sum_supply_units: total seller energy available (0.01 kWh units)
      - sum_demand_units: total buyer demand (0.01 kWh units)

    Dynamic-MCP (micro-ZAR/kWh):
      MCP = FLOOR + (TARIFF - FLOOR) * clamp(demand/supply, 0..1)
    Implemented using fixed-point ratio_scaled in [0, SCALE].
    """
    SCALE = Int(1_000_000)

    supply = sum_supply_units.get()
    demand = sum_demand_units.get()

    floor_micro = App.globalGet(K_FLOOR_MICRO)
    tariff_micro = App.globalGet(K_TARIFF_MICRO)
    spread = tariff_micro - floor_micro

    # ratio_scaled = clamp(demand/supply, 0..1) in SCALE fixed point
    ratio_raw = WideRatio([demand, SCALE], [supply])

    ratio_scaled = If(
        supply == Int(0),
        If(demand > Int(0), SCALE, Int(0)),
        If(ratio_raw <= SCALE, ratio_raw, SCALE),
    )

    mcp_micro = floor_micro + WideRatio([spread, ratio_scaled], [SCALE])

    allowed = min3(supply, demand, App.globalGet(K_FEEDER_CAP_U))

    return Seq(
        Assert(Txn.sender() == Global.creator_address()),
        App.globalPut(K_CUR_INT, interval.get()),
        App.globalPut(K_MCP_MICRO, mcp_micro),
        App.globalPut(K_ALLOWED_U, allowed),
        App.globalPut(K_TRADED_U, Int(0)),
        # Log: interval | allowed | mcp_micro
        Log(Concat(Itob(interval.get()), Itob(allowed), Itob(mcp_micro))),
        output.set(Bytes("Interval configured (Dynamic-MCP)")),
    )


@router.method
def record_trade(
    interval: abi.Uint64,
    buyer: abi.Address,
    seller: abi.Address,
    qty_units: abi.Uint64,
    *,
    output: abi.String,
) -> Expr:
    """
    Records one scaled P2P trade (Rev4-style batching via RELATIVE checks).

    Expected atomic group "chunk" (3 txns):
      [i]  : ApplicationCall (this method, from Creator/Oracle)
      [i+1]: AssetTransfer (KWH_TKN) seller -> buyer
      [i+2]: AssetTransfer (ZAR_TKN) buyer  -> seller

    KWH transfer amount:
      qty_units * 10   (because 0.01 kWh units and KWH token has 0.001 kWh base)

    ZAR transfer amount (micro-ZAR):
      max(1, floor(qty_units * mcp_micro / 100))
      because qty_units represent qty_units/100 kWh.
    """
    kwh_id = App.globalGet(K_KWH)
    zar_id = App.globalGet(K_ZAR)

    cur_int = App.globalGet(K_CUR_INT)
    mcp_micro = App.globalGet(K_MCP_MICRO)

    allowed_u = App.globalGet(K_ALLOWED_U)
    traded_u = App.globalGet(K_TRADED_U)

    q = qty_units.get()

    # Current AppCall index in group: 0,3,6,... (batch-safe)
    this_i = Txn.group_index()

    # Payment: floor(q * mcp_micro / 100), with 1 micro-ZAR minimum for q>0
    pay_floor = WideRatio([q, mcp_micro], [Int(100)])
    pay_amt = If(pay_floor > Int(0), pay_floor, Int(1))

    new_traded = traded_u + q

    return Seq(
        # --- Oracle Verification ---
        Assert(Txn.sender() == Global.creator_address()),

        # --- Sanity ---
        Assert(interval.get() == cur_int),
        Assert(q > Int(0)),

        # --- Interval cap enforcement ---
        Assert(new_traded <= allowed_u),

        # --- Check gtxn[i+1]: kWh transfer seller -> buyer ---
        Assert(Gtxn[this_i + Int(1)].type_enum() == Int(4)),
        Assert(Gtxn[this_i + Int(1)].xfer_asset() == kwh_id),
        Assert(Gtxn[this_i + Int(1)].asset_amount() == q * Int(10)),
        Assert(Gtxn[this_i + Int(1)].sender() == seller.get()),
        Assert(Gtxn[this_i + Int(1)].asset_receiver() == buyer.get()),

        # --- Check gtxn[i+2]: ZAR transfer buyer -> seller ---
        Assert(Gtxn[this_i + Int(2)].type_enum() == Int(4)),
        Assert(Gtxn[this_i + Int(2)].xfer_asset() == zar_id),
        Assert(Gtxn[this_i + Int(2)].asset_amount() == pay_amt),
        Assert(Gtxn[this_i + Int(2)].sender() == buyer.get()),
        Assert(Gtxn[this_i + Int(2)].asset_receiver() == seller.get()),

        # --- Update traded counter ---
        App.globalPut(K_TRADED_U, new_traded),

        # Log: interval | qty_units | mcp_micro
        Log(Concat(Itob(interval.get()), Itob(q), Itob(mcp_micro))),
        output.set(Bytes("Trade recorded")),
    )


@router.method
def finalize_interval(
    interval: abi.Uint64,
    *,
    output: abi.String,
) -> Expr:
    """
    Finalize interval (creator only).
    Strict reconciliation: traded_u must equal allowed_u.
    """
    return Seq(
        Assert(Txn.sender() == Global.creator_address()),
        Assert(interval.get() == App.globalGet(K_CUR_INT)),
        Assert(App.globalGet(K_TRADED_U) == App.globalGet(K_ALLOWED_U)),
        output.set(Bytes("Interval finalized")),
    )


if __name__ == "__main__":
    approval_teal, clear_teal, contract = router.compile_program(version=8)

    with open("abi_rev5.json", "w", encoding="utf-8") as f:
        json.dump(contract.dictify(), f, indent=2)
    with open("approval_rev5.teal", "w", encoding="utf-8") as f:
        f.write(approval_teal)
    with open("clear_rev5.teal", "w", encoding="utf-8") as f:
        f.write(clear_teal)

    print("Saved approval_rev5.teal, clear_rev5.teal & abi_rev5.json")



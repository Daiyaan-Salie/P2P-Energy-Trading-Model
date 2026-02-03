
#Model S3: PyTeal P2P Energy DEX (Rev5)
#
# Purpose:
# --------
# This script defines an Algorand smart contract (PyTeal Router) for a
# peer-to-peer (P2P) community energy trading mechanism.
#
# Key characteristics of this contract:
# - Interval-based market operation (one active interval stored in global state)
# - On-chain Dynamic MCP (Market Clearing Price) computed from demand/supply
# - Energy token transfers in kWh ASA units (milli-kWh precision)
# - ZAR token transfers in stablecoin-style micro-units (6 decimals)
# - Feeder capacity enforced per interval (total traded cannot exceed allowed)
#
# Design choice:
# --------------
# Interval state is stored in GLOBALS only (single active interval) to avoid
# PyTeal version/box/Maybe compile issues.
#
# Units/precision:
# ------------------
# - qty_units: 0.01 kWh units  (1 unit = 0.01 kWh)
# - KWH ASA decimals = 3 (milli-kWh)
#     => transfer amount = qty_units * 10  (because 0.01 kWh = 10 * 0.001 kWh)
# - ZAR ASA decimals = 6 (micro-ZAR)
# - mcp_micro: micro-ZAR per kWh (integer)
#
# Payment rule:
# -------------
# payment_microZAR = max(1, floor(qty_units * mcp_micro / 100))
# because qty_units represent (qty_units/100) kWh.

# -----------------------------
# Imports
# -----------------------------

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

# -----------------------------
# Global state keys
# -----------------------------
# These keys are stored in the app's GLOBAL state and used across all methods.
# (This contract intentionally avoids per-interval boxes.)

# Asset IDs (ASAs)
K_KWH = Bytes("kwh_id")             # uint: ASA id for the energy token
K_ZAR = Bytes("zar_id")             # uint: ASA id for the ZAR stable token

# Current interval identifier
K_CUR_INT = Bytes("cur_interval")   # uint: current interval id

# Price parameters and computed MCP (micro-ZAR per kWh)
K_FLOOR_MICRO = Bytes("floor_micro")    # uint
K_TARIFF_MICRO = Bytes("tariff_micro")  # uint
K_MCP_MICRO = Bytes("mcp_micro")        # uint

# Interval volume enforcement (0.01 kWh units)
K_FEEDER_CAP_U = Bytes("feeder_cap_u")  # uint: feeder cap per interval
K_ALLOWED_U = Bytes("allowed_u")        # uint: min(supply, demand, cap)
K_TRADED_U = Bytes("traded_u")          # uint: cumulative recorded trades in interval


# -----------------------------
# Router / application actions
# -----------------------------
# Router defines ABI methods and also restricts lifecycle actions.
# This contract only supports CREATE and CALL via NoOp.
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

# -----------------------------
# Helper functions
# -----------------------------
# Utility helpers are kept minimal and PyTeal-version-safe.
def min3(a: Expr, b: Expr, c: Expr) -> Expr:
    """Return min(a,b,c) using only If (PyTeal-version-safe)."""
    ab = If(a <= b, a, b)
    return If(ab <= c, ab, c)

# -----------------------------
# ABI method: setup_market
# -----------------------------
# One-time market initialization.
# Stores token IDs, pricing bounds, feeder cap, and resets interval state.

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
        # --- Access control ---
        # Only the creator is allowed to configure the market parameters.
        Assert(Txn.sender() == Global.creator_address()),

        # --- Parameter sanity ---
        # Tariff must be >= floor so the spread is non-negative.
        Assert(tariff_micro.get() >= floor_micro.get()),

       # --- Persist market configuration ---
        App.globalPut(K_KWH, kwh_asset.asset_id()),
        App.globalPut(K_ZAR, zar_asset.asset_id()),
        App.globalPut(K_FLOOR_MICRO, floor_micro.get()),
        App.globalPut(K_TARIFF_MICRO, tariff_micro.get()),
        App.globalPut(K_FEEDER_CAP_U, feeder_cap_units.get()),

        # --- Initialise interval state ---
        # Interval starts at 0; values are set to 0 until the first execute_interval call.
        App.globalPut(K_CUR_INT, Int(0)),
        App.globalPut(K_MCP_MICRO, Int(0)),
        App.globalPut(K_ALLOWED_U, Int(0)),
        App.globalPut(K_TRADED_U, Int(0)),

       # --- ABI response ---
        output.set(Bytes("Market setup complete! (Rev5 microZAR)")),
    )

# -----------------------------
# ABI method: execute_interval
# -----------------------------
# Starts a new interval and computes the Dynamic MCP on-chain.
# Also computes the allowed trade volume for this interval.

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

    # Fixed-point scale factor used for ratio calculations.
    # (Higher SCALE => more precision, still fits into integer arithmetic.)
    SCALE = Int(1_000_000)

   # --- Read inputs ---
    supply = sum_supply_units.get()
    demand = sum_demand_units.get()

   # --- Load price bounds ---
    floor_micro = App.globalGet(K_FLOOR_MICRO)
    tariff_micro = App.globalGet(K_TARIFF_MICRO)
    spread = tariff_micro - floor_micro

    # --- Step 1: demand/supply ratio (scaled) ---
    # ratio_raw = (demand * SCALE) / supply
    # Note: WideRatio performs safe wide-integer multiply/divide.
    ratio_raw = WideRatio([demand, SCALE], [supply])

    # --- Step 2: clamp ratio to [0, 1] in scaled form ---
    # If supply==0:
    #   - demand>0 => treat ratio as 1 (SCALE) to push MCP to tariff
    #   - demand==0 => ratio=0 (no activity)
    ratio_scaled = If(
        supply == Int(0),
        If(demand > Int(0), SCALE, Int(0)),
        If(ratio_raw <= SCALE, ratio_raw, SCALE),
    )

    # --- Step 3: compute MCP ---
    # mcp_micro = floor + spread * ratio_scaled / SCALE
    mcp_micro = floor_micro + WideRatio([spread, ratio_scaled], [SCALE])

    # --- Step 4: compute allowed interval volume ---
    # allowed is the enforced maximum total trade quantity for this interval.
    allowed = min3(supply, demand, App.globalGet(K_FEEDER_CAP_U))

    return Seq(
       # --- Access control (oracle / admin role) ---
        Assert(Txn.sender() == Global.creator_address()),

       # --- Persist interval configuration ---
        App.globalPut(K_CUR_INT, interval.get()),
        App.globalPut(K_MCP_MICRO, mcp_micro),
        App.globalPut(K_ALLOWED_U, allowed),

       # Reset traded counter for the new interval
        App.globalPut(K_TRADED_U, Int(0)),
       
        # --- Observability / indexing ---
        # Log: interval | allowed | mcp_micro
        Log(Concat(Itob(interval.get()), Itob(allowed), Itob(mcp_micro))),
        output.set(Bytes("Interval configured (Dynamic-MCP)")),
    )


# -----------------------------
# ABI method: record_trade
# -----------------------------
# Records a single trade inside a 3-transaction atomic group.
# Contract validates the two asset transfers match the declared trade details.

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
   # --- Load ASA IDs ---
    kwh_id = App.globalGet(K_KWH)
    zar_id = App.globalGet(K_ZAR)

   # --- Load current interval state ---
    cur_int = App.globalGet(K_CUR_INT)
    mcp_micro = App.globalGet(K_MCP_MICRO)

    allowed_u = App.globalGet(K_ALLOWED_U)
    traded_u = App.globalGet(K_TRADED_U)

   # --- Read requested trade quantity ---
    q = qty_units.get()

    # --- Identify position in atomic group ---
    # This enables batching multiple trade chunks:
    # trade chunks can start at indices 0,3,6,... (AppCall at each chunk start).
    this_i = Txn.group_index()

    # --- Compute payment in micro-ZAR ---
    # pay_floor = floor(q * mcp_micro / 100)
    pay_floor = WideRatio([q, mcp_micro], [Int(100)])

   # Enforce a minimum payment of 1 micro-ZAR for any positive quantity.
    pay_amt = If(pay_floor > Int(0), pay_floor, Int(1))

   # --- Compute new traded total for interval cap enforcement ---
    new_traded = traded_u + q

    return Seq(
        # --- Oracle Verification ---
       # Only creator/oracle is allowed to record trades (prevents arbitrary users logging fake trades).
        Assert(Txn.sender() == Global.creator_address()),

        # --- Sanity Checks---
        Assert(interval.get() == cur_int),
        Assert(q > Int(0)),

        # --- Interval cap enforcement ---
       # Ensures cumulative recorded trades do not exceed allowed volume for the interval.
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

        # --- Persist trade state ---
        # Update traded counter after validating transfers.
        App.globalPut(K_TRADED_U, new_traded),

        # --- Observability/indexing ---
        # Log: interval | qty_units | mcp_micro
        Log(Concat(Itob(interval.get()), Itob(q), Itob(mcp_micro))),
        output.set(Bytes("Trade recorded")),
    )


# -----------------------------
# ABI method: finalize_interval
# -----------------------------
# Closes the interval and enforces strict reconciliation:
# traded_u must equal allowed_u (i.e., the interval fully cleared as expected).
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
       # --- Access control ---
        Assert(Txn.sender() == Global.creator_address()),

       # --- Interval match ---
        Assert(interval.get() == App.globalGet(K_CUR_INT)),

       # --- Reconciliation ---
        # Ensures off-chain pro-rata scaling and batching resulted in the exact allowed volume.
        Assert(App.globalGet(K_TRADED_U) == App.globalGet(K_ALLOWED_U)),
        output.set(Bytes("Interval finalized")),
    )


# -----------------------------
# Entry point: compile artifacts
# -----------------------------
# Running this file directly compiles TEAL + ABI:
# - abi_rev5.json
# - approval_rev5.teal
# - clear_rev5.teal

if __name__ == "__main__":
    approval_teal, clear_teal, contract = router.compile_program(version=8)

    with open("abi_rev5.json", "w", encoding="utf-8") as f:
        json.dump(contract.dictify(), f, indent=2)
    with open("approval_rev5.teal", "w", encoding="utf-8") as f:
        f.write(approval_teal)
    with open("clear_rev5.teal", "w", encoding="utf-8") as f:
        f.write(clear_teal)

    print("Saved approval_rev5.teal, clear_rev5.teal & abi_rev5.json")



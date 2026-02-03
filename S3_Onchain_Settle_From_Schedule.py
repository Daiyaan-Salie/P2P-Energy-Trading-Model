"""
8 - S3_Onchain_Settle_From_Schedule_Rev1.py
===========================================

Purpose
-------
Replay an *off-chain planned schedule* (produced by the Oracle script) and perform
the *on-chain settlement* for each interval against the S3 DEX smart contract.

This script is the "bridge" between:
  - Script 5.7 (Oracle): produces interval_inputs.csv + planned_trades.csv
  - Script 5.8 (This):  calls the on-chain contract to compute MCP + allowed_u,
                        scales trades, submits atomic trade groups, and finalizes.

High-level flow (per interval)
------------------------------
1) execute_interval(interval, total_supply_u, total_demand_u)
   - Signed by COORDINATOR
   - Contract computes and stores (globally) the MCP and allowed trade volume.

2) Read global state:
   - mcp_micro     : uint64  (micro-ZAR / kWh)
   - allowed_u     : uint64  (0.01 kWh units allowed for that interval)
   - feeder_cap... : uint64  (optional debug value)

3) Scale trades:
   - planned_trades.csv contains "qty_u_planned" (0.01 kWh units)
   - If allowed_u < planned_total_u, scale all trades deterministically so:
       sum(qty_u_scaled) == allowed_u EXACTLY

4) For each trade (seller -> buyer, qty_u):
   Submit an *atomic group* of 3 transactions:
     (a) App call: record_trade(interval, buyer, seller, qty_u)  [COORDINATOR signs]
     (b) KWH ASA transfer: seller -> buyer, amount = qty_u * 10  [SELLER signs]
     (c) ZAR ASA transfer: buyer  -> seller, amount = floor(qty_u*mcp/100) (min 1) [BUYER signs]

5) finalize_interval(interval)
   - Signed by COORDINATOR
   - Closes/locks interval settlement in the contract.

Inputs
------
--run-dir <folder>
  Must contain:
    - interval_inputs.csv
        Required columns:
          interval, total_supply_u, total_demand_u
        Optional columns:
          planned_clear_u  (used for reporting only)
    - planned_trades.csv
        Required columns:
          interval, seller_addr, buyer_addr, qty_u_planned

--cohort-json cohort.json
  JSON array of members with fields like:
    [{"addr": "...", "mnemonic": "...", "role": "...", "id": 1}, ...]

Environment (.env)
------------------
Required:
  DEX_APP_ID
  ZAR_ASA_ID (or legacy ZAR_TOKEN_ID)
  KWH_ASA_ID (or legacy KWH_TOKEN_ID)
  COORDINATOR_MNEMONIC (or legacy TEST_ACCOUNT_MNEMONIC)
  ALGOD_ADDRESS (or compatible aliases used by env_first)

Optional:
  ALGOD_TOKEN (can be blank for certain providers)
  ALGOD_HEADERS_JSON (JSON string, e.g. {"X-API-Key":"..."} )
  START_INTERVAL / END_INTERVAL (limits interval range when --single-interval not used)

Outputs (written inside run-dir)
--------------------------------
run-dir/settlement/
  - interval_summary.csv   (per-interval high-level status + MCP/allowed)
  - trade_log.csv          (per-trade submission record)
  - settlement_meta.json   (run configuration + IDs)

Units & conventions
-------------------
qty_u:
  Integer quantity in units of 0.01 kWh.
  Example: 1.23 kWh => qty_u = 123

KWH ASA:
  Assumes ASA decimals = 3 (milli-kWh units).
  Since 0.01 kWh = 10 * 0.001 kWh, the transfer amount is:
    kwh_amt = qty_u * 10

mcp_micro_per_kwh:
  Integer MCP in micro-ZAR per kWh.
  Example: 1.20 ZAR/kWh => 1_200_000 micro-ZAR/kWh

payment_microzar:
  qty_u represents (qty_u / 100) kWh, so payment is:
    microZAR = floor( (qty_u/100) * mcp_micro_per_kwh )
            = floor( qty_u * mcp_micro_per_kwh / 100 )
  Enforced minimum of 1 microZAR to avoid a zero-amount ASA transfer.

CLI Examples
------------
Dry run (execute_interval only, read state, no trades/finalize):
  python S3_Onchain_Settle_From_Schedule.py --run-dir runs/2026-01-07_00-09-59 --dry-run

Single interval full settlement:
  python S3_Onchain_Settle_From_Schedule.py --run-dir runs/2026-01-07_00-09-59 --single-interval 48

Submit trades but skip finalize (debug):
  python S3_Onchain_Settle_From_Schedule.py --run-dir runs/... --no-finalize

Aggregate duplicate seller/buyer pairs (optional):
  python S3_Onchain_Settle_From_Schedule.py --run-dir runs/... --aggregate
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv

from algosdk import account, mnemonic, transaction
from algosdk.abi import Contract
from algosdk.encoding import decode_address
from algosdk.v2client import algod

def wait_rounds(client: algod.AlgodClient, rounds: int = 1) -> int:
    """Advance the node's last-round by `rounds` with minimal HTTP calls."""
    lr = client.status().get("last-round", 0)
    if rounds <= 0:
        return lr
    target = lr + rounds
    client.status_after_block(target)
    return target

def try_finalize_with_retries(
    client: algod.AlgodClient,
    finalize_fn,
    interval: int,
    max_retries: int = 8,
    wait_between_rounds: int = 2,
) -> bool:
    """
    Attempt finalize; if it fails because some trades are still pending,
    wait a few rounds and retry.

    Note:
    - In this script, finalize is done as a single txn, but failures can happen
      if the network hasn't confirmed the submitted trade groups yet.
    """
    last_err = None
    for _ in range(max_retries):
        try:
            finalize_fn(interval)
            return True
        except Exception as e:
            last_err = str(e)
            wait_rounds(client, wait_between_rounds)
    raise RuntimeError(f"Finalize failed after {max_retries} retries. Last error: {last_err}")


# -----------------------------
# Minimal env helpers (NO standardization)
# -----------------------------

def env_first(*names: str, required: bool = True) -> str:
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    if required:
        raise ValueError(f"Missing env var. Tried: {', '.join(names)}")
    return ""

def env_first_int(*names: str, required: bool = True) -> int:
    return int(env_first(*names, required=required))


# -----------------------------
# Units / locked conversions
# -----------------------------

def kwh_asa_amount_from_qty_u(qty_u: int) -> int:
    """
    Convert internal trade qty (0.01 kWh units) to KWH ASA transfer amount.

    Assumption:
    - KWH ASA uses decimals=3 (milli-kWh)
    - 0.01 kWh = 10 * 0.001 kWh
    """
    return int(qty_u) * 10

def payment_microzar(qty_u: int, mcp_micro_per_kwh: int) -> int:
    """
    Convert qty_u (0.01 kWh) and MCP (microZAR/kWh) into a microZAR payment.

    payment = max(1, floor(qty_u * mcp_micro / 100))

    Why /100?
    - qty_u is 0.01 kWh => qty_kwh = qty_u / 100
    - microZAR = (qty_u/100) * mcp_micro
    """
    return max(1, (int(qty_u) * int(mcp_micro_per_kwh)) // 100)


# =============================================================================
# Cohort loading (addresses + mnemonics so buyers/sellers can sign transfers)
# =============================================================================

@dataclass(frozen=True)
class CohortMember:
    addr: str
    mnemonic: str
    role: str
    id: int

def load_cohort(path: Path) -> Dict[str, CohortMember]:
    """
    Load cohort.json and return mapping: address -> CohortMember.

    cohort.json must contain mnemonics so we can sign each seller/buyer ASA transfer.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, CohortMember] = {}
    for e in data:
        addr = str(e["addr"]).strip()
        out[addr] = CohortMember(
            addr=addr,
            mnemonic=str(e["mnemonic"]).strip(),
            role=str(e.get("role", "")).strip().lower(),
            id=int(e.get("id", -1)),
        )
    if not out:
        raise ValueError("cohort.json loaded but contained no entries.")
    return out


# =============================================================================
# Algod client (supports tokenless endpoints and header-based providers)
# =============================================================================

def get_algod_client() -> algod.AlgodClient:
    """
    Build AlgodClient with flexible env naming.
    Supports providers where token is blank and auth is via headers.
    """
    algod_address = env_first(
        "ALGOD_ADDRESS",
        "ALGOD_URL",
        "ALGOD_SERVER",
        "ALGORAND_ALGOD_ADDRESS",
        "ALGORAND_NODE"
    )

    algod_token = env_first(
        "ALGOD_TOKEN",
        "ALGOD_API_KEY",
        "PURESTAKE_API_KEY",
        "ALGO_API_KEY",
        "ALGORAND_ALGOD_TOKEN",
        required=False
    )

    headers_json = os.getenv("ALGOD_HEADERS_JSON", "").strip()
    headers = None
    if headers_json:
        try:
            headers = json.loads(headers_json)
        except Exception as e:
            raise ValueError(f"ALGOD_HEADERS_JSON is not valid JSON: {e}")

    # token may be blank for some nodes/providers
    return algod.AlgodClient(algod_token or "", algod_address, headers=headers)


# =============================================================================
# ABI / Contract loader
# =============================================================================

def load_abi_contract(abi_path: Path) -> Contract:
    """
    Load ABI JSON file and build an algosdk.abi.Contract instance.
    """
    abi = json.loads(abi_path.read_text(encoding="utf-8"))
    return Contract.from_json(json.dumps(abi))


# =============================================================================
# Global state reading helpers (robust to differing key names)
# =============================================================================

def b64_to_bytes(s: str) -> bytes:
    #"""Decode base64 string (as returned by algod) into raw bytes."""
    return base64.b64decode(s)

def read_global_state_raw(client: algod.AlgodClient, app_id: int) -> Dict[bytes, int]:
    """
    Pull global state from the app and return mapping: raw_key_bytes -> uint_value
    Only uints are read (type==2).
    """
    app = client.application_info(app_id)
    gs_list = app["params"].get("global-state", [])
    out: Dict[bytes, int] = {}
    for kv in gs_list:
        k = b64_to_bytes(kv["key"])
        v = kv["value"]
        if v["type"] == 2:
            out[k] = int(v["uint"])
    return out

def gs_get(gs: Dict[bytes, int], *candidate_keys: bytes, default: int = 0) -> int:
    """
    Read the first matching key from global state; return default if none exist.
    Useful because key names can vary slightly across contract revisions.
    """
    for k in candidate_keys:
        if k in gs:
            return int(gs[k])
    return default

def pretty_global_state_keys(gs: Dict[bytes, int]) -> List[str]:
    """Debug helper: show global-state keys as strings where possible."""
    keys = []
    for k in gs.keys():
        try:
            keys.append(k.decode("utf-8", errors="replace"))
        except Exception:
            keys.append(str(k))
    return keys

KEY_MCP = [b"mcp_micro", b"mcp", b"mcpMicro", b"MCP_MICRO"]
KEY_ALLOWED = [b"allowed_u", b"allowed_units", b"allowed", b"allowedUnits", b"ALLOWED_U"]
KEY_FEEDER = [b"feeder_cap_units", b"feeder_cap_u", b"feederCap", b"feeder_cap", b"FEEDER_CAP"]


# =============================================================================
# Deterministic scaling: planned trades -> allowed_u
# =============================================================================

def scale_trades_to_allowed(
    trades: List[Tuple[str, str, int]],
    allowed_u: int
) -> List[Tuple[str, str, int]]:
    """
    Scale planned trades so that the final trade quantities sum to allowed_u exactly.

    Inputs:
      trades: list of (seller_addr, buyer_addr, qty_u_planned)
      allowed_u: authoritative allowed energy (0.01 kWh units) from on-chain logic

    Algorithm:
      - Compute alpha = allowed_u / planned_total
      - For each trade: x = qty * alpha
          base = floor(x)
          frac = x - base
      - Sum bases, then distribute the remaining units to the trades with largest frac
        (tie-break deterministically by seller/buyer address)
      - Final drift correction ensures exact equality and no negative quantities.

    Returns:
      List of (seller, buyer, qty_u_scaled) with sum == allowed_u
    """  
    allowed_u = int(allowed_u)
    if allowed_u <= 0 or not trades:
        return []

    planned_total = sum(q for _, _, q in trades)
    if planned_total <= 0:
        return []

    if planned_total == allowed_u:
        return [(s, b, int(q)) for s, b, q in trades if q > 0]

    alpha = allowed_u / planned_total
    scaled = []
    for s, b, q in trades:
        x = q * alpha
        base = int(math.floor(x))
        frac = x - base
        scaled.append((s, b, base, frac))

    base_sum = sum(t[2] for t in scaled)
    remainder = allowed_u - base_sum
    
   # Allocate +1 to top `remainder` trades by descending fractional part
    ranked = sorted(scaled, key=lambda t: (-t[3], t[0], t[1]))
    bump = set((ranked[i][0], ranked[i][1]) for i in range(min(max(remainder, 0), len(ranked))))

    out = []
    for s, b, base, _frac in scaled:
        newq = base + (1 if (s, b) in bump else 0)
        if newq > 0:
            out.append((s, b, int(newq)))

     # Drift correction (should be rare, but protects exact sum requirement)
    drift = allowed_u - sum(q for _, _, q in out)
    if drift != 0:
        out.sort(key=lambda t: (t[0], t[1]))
        if drift > 0:
            i = 0
            while drift > 0 and i < len(out):
                s, b, q = out[i]
                out[i] = (s, b, q + 1)
                drift -= 1
                i += 1
        else:
            i = len(out) - 1
            while drift < 0 and i >= 0:
                s, b, q = out[i]
                if q > 1:
                    out[i] = (s, b, q - 1)
                    drift += 1
                i -= 1

    assert sum(q for _, _, q in out) == allowed_u, "Scaling failed to hit allowed_u exactly."
    return out

def aggregate_trades(trades: List[Tuple[str, str, int]]) -> List[Tuple[str, str, int]]:
       """
    Optional compression step:
    Combine duplicate (seller,buyer) pairs by summing qty_u.
    Useful to reduce number of atomic groups sent on-chain.
    """
    m: Dict[Tuple[str, str], int] = {}
    for s, b, q in trades:
        if q <= 0:
            continue
        m[(s, b)] = m.get((s, b), 0) + int(q)
    out = [(s, b, q) for (s, b), q in m.items() if q > 0]
    out.sort(key=lambda t: (t[0], t[1]))
    return out


# -----------------------------
# CSV writers
# -----------------------------

def init_csvs(settlement_dir: Path) -> Tuple[Path, Path]:
    interval_summary = settlement_dir / "interval_summary.csv"
    trade_log = settlement_dir / "trade_log.csv"

    with interval_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "interval",
            "total_supply_u",
            "total_demand_u",
            "mcp_micro",
            "allowed_u",
            "feeder_cap_units",
            "planned_clear_u",
            "planned_total_u",
            "scaled_total_u",
            "execute_ok",
            "finalize_ok",
            "execute_txid",
            "finalize_txid",
            "error",
        ])
        w.writeheader()

    with trade_log.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "interval",
            "seller_addr",
            "buyer_addr",
            "qty_u",
            "kwh_asa_amount",
            "payment_microzar",
            "group_txid",
            "ok",
            "error",
        ])
        w.writeheader()

    return interval_summary, trade_log

def append_row(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writerow(row)


# -----------------------------
# Confirm helper
# -----------------------------

def wait_for_confirm(client: algod.AlgodClient, txid: str, timeout_rounds: int = 12) -> dict:
    last_round = client.status().get("last-round", 0)
    for _ in range(timeout_rounds):
        p = client.pending_transaction_info(txid)
        if p.get("confirmed-round", 0) > 0:
            return p
        last_round += 1
        client.status_after_block(last_round)
    raise TimeoutError(f"Tx not confirmed after {timeout_rounds} rounds: {txid}")



def wait_rounds(client: algod.AlgodClient, rounds: int = 2) -> int:
    """Advance the node's last-round by `rounds` with minimal HTTP calls."""
    if rounds <= 0:
        return client.status().get("last-round", 0)
    lr = client.status().get("last-round", 0)
    target = lr + rounds
    client.status_after_block(target)
    return target

def try_finalize_with_retries(
    client: algod.AlgodClient,
    finalize_fn,
    interval: int,
    max_retries: int = 6,
    wait_between_rounds: int = 2,
) -> bool:
    """Attempt finalize; if it fails (often because trades are still unconfirmed), wait and retry."""
    last_err = None
    for _ in range(max_retries):
        try:
            finalize_fn(interval)
            return True
        except Exception as e:
            last_err = str(e)
            # wait a bit and retry
            wait_rounds(client, wait_between_rounds)
    raise RuntimeError(f"Finalize failed after {max_retries} retries. Last error: {last_err}")



# =============================================================================
# ABI arg builders (manual selector + canonical u64 encoding)
# =============================================================================

def u64(x: int) -> bytes:
        """Encode integer as 8-byte big-endian (ABI uint64)."""
    return int(x).to_bytes(8, "big")

def build_execute_interval_args(method_execute, interval: int, sum_supply_u: int, sum_demand_u: int) -> List[bytes]:
    """
    Build app_args for execute_interval ABI method:
      [selector, interval, total_supply_u, total_demand_u]
    """
    return [method_execute.get_selector(), u64(interval), u64(sum_supply_u), u64(sum_demand_u)]

def build_record_trade_args(method_record, interval: int, buyer: str, seller: str, qty_u: int) -> List[bytes]:
    """
    Build app_args for record_trade ABI method:
      [selector, interval, buyer_address, seller_address, qty_u]
    Note: decode_address converts base32 addr -> 32 raw bytes for ABI address type.
    """
    return [method_record.get_selector(), u64(interval), decode_address(buyer), decode_address(seller), u64(qty_u)]

def build_finalize_args(method_finalize, interval: int) -> List[bytes]:
    """
    Build app_args for finalize_interval ABI method:
      [selector, interval]
    """
    return [method_finalize.get_selector(), u64(interval)]


# -----------------------------
# Remember-handling helpers
# -----------------------------

_TXID_RE = re.compile(r"\b([A-Z2-7]{52})\b")

def extract_any_txid(msg: str) -> str:
    m = _TXID_RE.search(msg or "")
    return m.group(1) if m else ""

def send_group_with_remember_handling(
    client: algod.AlgodClient,
    signed_group: List[transaction.SignedTransaction],
    primary_txid: str,
) -> str:
    """
    Submit a grouped transaction. If the node returns TransactionPool.Remember,
    treat it as non-fatal and return `primary_txid` for later confirmation.

    Note:
    - In this script's main path we submit via client.send_transactions() and
      ignore Remember errors in flush_batch(). This helper exists for more explicit handling.
    """
    try:
        _ = client.send_transactions(signed_group)
        return primary_txid
    except Exception as e:
        emsg = str(e)
        if "TransactionPool.Remember" in emsg or "Remember" in emsg:
            # Try confirm the tx we already computed
            try:
                return primary_txid
            except Exception:
                # Try extracting another txid from error and confirm that
                txid2 = extract_any_txid(emsg)
                if txid2:
                    return txid2
        raise


def wait_for_batch_confirm(
    client: algod.AlgodClient,
    txids: List[str],
    timeout_rounds: int = 20,
) -> None:
    """Confirm a batch efficiently: wait per-round and check all txids once per round."""
    if not txids:
        return
    remaining = set(txids)
    last_round = client.status().get("last-round", 0)
    for _ in range(timeout_rounds):
        # Check remaining txids once
        done = []
        for txid in remaining:
            p = client.pending_transaction_info(txid)
            if p.get("confirmed-round", 0) > 0:
                done.append(txid)
        for txid in done:
            remaining.discard(txid)
        if not remaining:
            return

        # Advance exactly one round, then re-check
        last_round += 1
        client.status_after_block(last_round)

    raise TimeoutError(f"{len(remaining)} txns not confirmed after {timeout_rounds} rounds")


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    load_dotenv(".env")

    # -----------------------------
    # CLI Arguments
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help=r"Oracle run folder, e.g. runs\2026-01-07_00-09-59")
    parser.add_argument("--cohort-json", default="cohort.json", help="Path to cohort.json (contains mnemonics)")
    parser.add_argument("--abi", default="abi_rev5.json", help="Path to Rev6 ABI JSON")
    parser.add_argument("--single-interval", type=int, default=None, help="Run exactly one interval (first test).")
    parser.add_argument("--dry-run", action="store_true", help="Only execute_interval + read state; no trades/finalize.")
    parser.add_argument("--no-finalize", action="store_true", help="Submit trades but skip finalize_interval (debug).")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate duplicate (seller,buyer) pairs.")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="How many atomic trade groups to send before confirming (speed-up).")
    parser.add_argument("--wait-rounds-per-batch", type=int, default=0, help="Advance chain by N rounds after each batch flush (0 = no wait)")
    parser.add_argument("--finalize-retries", type=int, default=8, help="Max finalize retries if finalize fails due to pending trades")
    parser.add_argument("--finalize-wait-rounds", type=int, default=2, help="Rounds to wait between finalize retries")
    parser.add_argument("--max-trades", type=int, default=0, help="If >0, limit trades sent (debug only).")
    args = parser.parse_args()

    if args.max_trades and args.max_trades > 0 and not args.no_finalize:
        raise ValueError("--max-trades is debug-only and must be used with --no-finalize "
                         "(finalize requires full settlement for the interval).")

    # -----------------------------
    # IDs / env (backward-compatible names)
    # -----------------------------
    app_id = env_first_int("DEX_APP_ID")
    zar_asa_id = env_first_int("ZAR_ASA_ID", "ZAR_TOKEN_ID")
    kwh_asa_id = env_first_int("KWH_ASA_ID", "KWH_TOKEN_ID")
    coord_mn = env_first("COORDINATOR_MNEMONIC", "TEST_ACCOUNT_MNEMONIC")

    coord_sk = mnemonic.to_private_key(coord_mn)
    coord_addr = account.address_from_private_key(coord_sk)

    client = get_algod_client()

    # -----------------------------
    # ABI methods
    # -----------------------------
    contract = load_abi_contract(Path(args.abi))
    m_execute = contract.get_method_by_name("execute_interval")
    m_record = contract.get_method_by_name("record_trade")
    m_finalize = contract.get_method_by_name("finalize_interval")

    # -----------------------------
    # Load Oracle outputs
    # -----------------------------
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    interval_inputs_path = run_dir / "interval_inputs.csv"
    planned_trades_path = run_dir / "planned_trades.csv"
    if not interval_inputs_path.exists():
        raise FileNotFoundError(f"Missing interval_inputs.csv: {interval_inputs_path}")
    if not planned_trades_path.exists():
        raise FileNotFoundError(f"Missing planned_trades.csv: {planned_trades_path}")

    interval_df = pd.read_csv(interval_inputs_path)
    trades_df = pd.read_csv(planned_trades_path)

    # Interval range controls
    start_env = int(os.getenv("START_INTERVAL", "0"))
    end_env = int(os.getenv("END_INTERVAL", "-1"))
    if args.single_interval is not None:
        start = end = int(args.single_interval)
    else:
        start = max(0, start_env)
        end = int(end_env) if end_env >= 0 else int(interval_df["interval"].max())

    # -----------------------------
    # Outputs
    # -----------------------------
    settlement_dir = run_dir / "settlement"
    settlement_dir.mkdir(parents=True, exist_ok=True)
    interval_summary_csv, trade_log_csv = init_csvs(settlement_dir)

    meta = {
        "run_dir": str(run_dir),
        "settlement_dir": str(settlement_dir),
        "app_id": app_id,
        "zar_asa_id": zar_asa_id,
        "kwh_asa_id": kwh_asa_id,
        "coordinator_addr": coord_addr,
        "start_interval": start,
        "end_interval": end,
        "dry_run": bool(args.dry_run),
        "no_finalize": bool(args.no_finalize),
        "aggregate": bool(args.aggregate),
        "batch_size": int(args.batch_size),
        "max_trades": int(args.max_trades),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    cohort_map = load_cohort(Path(args.cohort_json))

    # =============================================================================
    # Interval loop
    # ===========================================================================
    for t in range(start, end + 1):
        # Extract interval totals from oracle interval_inputs.csv
        row = interval_df[interval_df["interval"] == t]
        if row.empty:
            print(f"Interval {t}: not found in interval_inputs.csv, skipping.")
            continue

        total_supply_u = int(row.iloc[0]["total_supply_u"])
        total_demand_u = int(row.iloc[0]["total_demand_u"])
        # planned_clear_u is for reporting (not authoritative on-chain)
        planned_clear_u = int(row.iloc[0]["planned_clear_u"]) if "planned_clear_u" in row.columns else min(total_supply_u, total_demand_u)

        # Extract planned trades for this interval
        tdf = trades_df[trades_df["interval"] == t]
        planned_trades = [(str(r["seller_addr"]), str(r["buyer_addr"]), int(r["qty_u_planned"])) for _, r in tdf.iterrows()]
        planned_total_u = sum(q for _, _, q in planned_trades)

        execute_ok = False
        finalize_ok = False
        execute_txid = ""
        finalize_txid = ""
        err = ""

        mcp_micro = 0
        allowed_u = 0
        feeder_cap_units = 0

        try:
            # ------------------------------------------------------------
            # 1) execute_interval (COORDINATOR signs)
            # ------------------------------------------------------------
            sp = client.suggested_params()
            exec_args = build_execute_interval_args(m_execute, t, total_supply_u, total_demand_u)
            exec_txn = transaction.ApplicationNoOpTxn(sender=coord_addr, sp=sp, index=app_id, app_args=exec_args)
            signed_exec = exec_txn.sign(coord_sk)
            execute_txid = client.send_transaction(signed_exec)
            wait_for_confirm(client, execute_txid)
            execute_ok = True

            # ------------------------------------------------------------
            # 2) Read global state (MCP + allowed_u are authoritative)
            # ------------------------------------------------------------
            gs = read_global_state_raw(client, app_id)
            mcp_micro = gs_get(gs, *KEY_MCP, default=0)
            allowed_u = gs_get(gs, *KEY_ALLOWED, default=0)
            feeder_cap_units = gs_get(gs, *KEY_FEEDER, default=0)

            if (mcp_micro == 0 or allowed_u == 0) and len(gs) > 0:
                print(f"[Debug] Interval {t} global-state keys: {pretty_global_state_keys(gs)}")

            if args.dry_run:
                append_row(interval_summary_csv, {
                    "interval": t,
                    "total_supply_u": total_supply_u,
                    "total_demand_u": total_demand_u,
                    "mcp_micro": mcp_micro,
                    "allowed_u": allowed_u,
                    "feeder_cap_units": feeder_cap_units,
                    "planned_clear_u": planned_clear_u,
                    "planned_total_u": planned_total_u,
                    "scaled_total_u": 0,
                    "execute_ok": int(execute_ok),
                    "finalize_ok": 0,
                    "execute_txid": execute_txid,
                    "finalize_txid": "",
                    "error": "",
                })
                print(f"Interval {t} (DRY RUN): execute_ok=1, mcp_micro={mcp_micro}, allowed_u={allowed_u}")
                if args.single_interval is not None:
                    break
                continue

            # ------------------------------------------------------------
            # 3) Scale planned trades to allowed_u (feeder cap enforcement)
            # ------------------------------------------------------------
            scaled_trades = scale_trades_to_allowed(planned_trades, allowed_u)
            if args.aggregate:
                scaled_trades = aggregate_trades(scaled_trades)
            if args.max_trades and args.max_trades > 0:
                scaled_trades = scaled_trades[:args.max_trades]

            scaled_total_u = sum(q for _, _, q in scaled_trades)
            if scaled_total_u != allowed_u:
                raise ValueError(f"Scaled trades sum {scaled_total_u} != allowed_u {allowed_u} for interval {t}")

            # ------------------------------------------------------------
            # 4) Submit atomic trade groups (3 txns per trade)
            #    We "batch" submissions by sending N groups, then optionally
            #    waiting a few rounds. Final correctness is enforced by finalize.
            # ------------------------------------------------------------
            pending_rows: List[dict] = []
            pending_txids: List[str] = []          # track primary txid per group (the app-call txid)
            pending_groups: List[List[transaction.SignedTransaction]] = []  # signed txn groups (each <=16 txns)

            def flush_batch() -> None:
                 """
                Send all pending atomic groups for this batch.
                Treat TransactionPool.Remember as non-fatal.
                Write trade_log rows after submission (not after confirmation).
                """
                nonlocal pending_rows, pending_txids, pending_groups
                if not pending_txids:
                    return

                # Submit each *atomic group* separately (Algorand max group size = 16).
                # We batch by deferring waiting/confirmation until after N groups have been submitted.
                for g in pending_groups:
                    try:
                        _ = client.send_transactions(g)
                    except Exception as e:
                        emsg = str(e)
                        # Treat TransactionPool.Remember as non-fatal (already in mempool / recently seen)
                        if ("TransactionPool.Remember" not in emsg) and ("Remember" not in emsg):
                            raise

                # Optional: advance the chain by a small number of rounds to help inclusion
                if getattr(args, 'wait_rounds_per_batch', 0) > 0:
                    wait_rounds(client, getattr(args, 'wait_rounds_per_batch', 0))

                # Only now write logs (we consider submission success; final correctness is enforced by finalize/retries)
                for r in pending_rows:
                    r["ok"] = 1
                    append_row(trade_log_csv, r)

                pending_rows = []
                pending_txids = []
                pending_groups = []

            # Build + sign each atomic group
            for seller_addr, buyer_addr, qty_u in scaled_trades:
                try:
                    # Ensure both parties exist in cohort
                    if seller_addr not in cohort_map:
                        raise ValueError(f"Seller not found in cohort.json: {seller_addr}")
                    if buyer_addr not in cohort_map:
                        raise ValueError(f"Buyer not found in cohort.json: {buyer_addr}")

                    seller_sk = mnemonic.to_private_key(cohort_map[seller_addr].mnemonic)
                    buyer_sk = mnemonic.to_private_key(cohort_map[buyer_addr].mnemonic)

                    spg = sp  # cached suggested params for this interval

                    # (a) record_trade app call (coordinator)
                    record_args = build_record_trade_args(m_record, t, buyer_addr, seller_addr, qty_u)
                    app_call = transaction.ApplicationNoOpTxn(
                        sender=coord_addr,
                        sp=spg,
                        index=app_id,
                        app_args=record_args,
                    )

                    # (b) KWH transfer seller -> buyer
                    kwh_amt = kwh_asa_amount_from_qty_u(qty_u)
                    tx_kwh = transaction.AssetTransferTxn(
                        sender=seller_addr, sp=spg, receiver=buyer_addr, amt=kwh_amt, index=kwh_asa_id
                    )

                    # (c) ZAR transfer buyer -> seller
                    pay_amt = payment_microzar(qty_u, mcp_micro)
                    tx_zar = transaction.AssetTransferTxn(
                        sender=buyer_addr, sp=spg, receiver=seller_addr, amt=pay_amt, index=zar_asa_id
                    )
                     # Group and sign
                    gid = transaction.calculate_group_id([app_call, tx_kwh, tx_zar])
                    app_call.group = gid
                    tx_kwh.group = gid
                    tx_zar.group = gid

                    s_app = app_call.sign(coord_sk)
                    s_kwh = tx_kwh.sign(seller_sk)
                    s_zar = tx_zar.sign(buyer_sk)

                    primary_txid = s_app.get_txid()

                    # Queue for batch submission (flattened list)
                    pending_groups.append([s_app, s_kwh, s_zar])

                    pending_rows.append({
                        "interval": t,
                        "seller_addr": seller_addr,
                        "buyer_addr": buyer_addr,
                        "qty_u": int(qty_u),
                        "kwh_asa_amount": int(kwh_amt),
                        "payment_microzar": int(pay_amt),
                        "group_txid": primary_txid,
                        "ok": 0,          # set to 1 on confirm
                        "error": "",
                    })
                    pending_txids.append(primary_txid)

                    if len(pending_txids) >= int(args.batch_size):
                        flush_batch()

                except Exception as e:
                    # Flush any already-submitted trades (so logs aren't lost)
                    try:
                        flush_batch()
                    except Exception:
                        pass
                    append_row(trade_log_csv, {
                        "interval": t,
                        "seller_addr": seller_addr,
                        "buyer_addr": buyer_addr,
                        "qty_u": int(qty_u),
                        "kwh_asa_amount": kwh_asa_amount_from_qty_u(qty_u),
                        "payment_microzar": payment_microzar(qty_u, mcp_micro),
                        "group_txid": "",
                        "ok": 0,
                        "error": str(e),
                    })
                    raise RuntimeError(
                        f"Trade group failed (interval {t}) seller={seller_addr} buyer={buyer_addr}: {e}"
                    )

             # Send remaining groups in the final partial batch
            flush_batch()


            # ------------------------------------------------------------
            # 5) finalize_interval (COORDINATOR)
            # ------------------------------------------------------------
            if not args.no_finalize:
                sp2 = client.suggested_params()
                fin_args = build_finalize_args(m_finalize, t)
                fin_txn = transaction.ApplicationNoOpTxn(sender=coord_addr, sp=sp2, index=app_id, app_args=fin_args)
                signed_fin = fin_txn.sign(coord_sk)
                finalize_txid = client.send_transaction(signed_fin)
                wait_for_confirm(client, finalize_txid)
                finalize_ok = True

        except Exception as e:
            err = str(e)

        append_row(interval_summary_csv, {
            "interval": t,
            "total_supply_u": total_supply_u,
            "total_demand_u": total_demand_u,
            "mcp_micro": int(mcp_micro),
            "allowed_u": int(allowed_u),
            "feeder_cap_units": int(feeder_cap_units),
            "planned_clear_u": int(planned_clear_u),
            "planned_total_u": int(planned_total_u),
            "scaled_total_u": int(allowed_u) if (execute_ok and not args.dry_run) else 0,
            "execute_ok": int(execute_ok),
            "finalize_ok": int(finalize_ok),
            "execute_txid": execute_txid,
            "finalize_txid": finalize_txid,
            "error": err,
        })

        print(
            f"Interval {t}: execute_ok={int(execute_ok)}, finalize_ok={int(finalize_ok)}, "
            f"allowed_u={allowed_u}, mcp_micro={mcp_micro}, err={(err[:140] if err else '')}"
        )

        if args.single_interval is not None:
            break

    (settlement_dir / "settlement_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n✅ Settlement script complete.")
    print(f"- Interval summary: {interval_summary_csv}")
    print(f"- Trade log:        {trade_log_csv}")
    print(f"- Meta:             {settlement_dir / 'settlement_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


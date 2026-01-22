"""
5.7 - S3c_Oracle_Generate_Schedule_Rev1.py  (UPDATED: Battery-Integrated Oracle)

Oracle stage (off-chain):
- Generate synthetic household load + PV profiles (from Common_Functions.py)
- Simulate prosumer battery SoC (deterministic) to transform raw PV/load into:
    - market supply (post-battery)
    - market demand (post-battery)
- Compute interval total supply/demand in internal units qty_u (0.01 kWh units)
- Create a deterministic planned trade schedule (seller->buyer, qty_u_planned)
- Write CSV outputs for on-chain settlement replay (used by Script 5.8)

LOCKED ASSUMPTIONS (by design, to avoid silent inconsistencies):
- 15-minute resolution
- 96 intervals per day
- Common_Functions profiles are already kWh per 15-minute interval

Inputs:
- participants.csv with columns: addr, role
    role in {"prosumer", "consumer"} (case-insensitive)

Outputs (in runs/<timestamp>[_<run-name>]/):
- participants.csv (copied)
- interval_inputs.csv        REQUIRED cols: interval, total_supply_u, total_demand_u, planned_clear_u
                           + EXTRA cols (safe for Script 5.8 to ignore)
- planned_trades.csv         columns: interval, seller_addr, buyer_addr, qty_u_planned
- battery_timeseries.csv     columns: interval, avg_soc_pct, charge_u, discharge_u, curtail_u, utility_export_u, etc.
- oracle_meta.json           hashes + run params + validation summary

Notes:
- This script DOES NOT apply feeder cap. Cap enforcement occurs on-chain via allowed_u.
- Planned trades clear up to planned_clear_u = min(total_supply_u, total_demand_u) per interval.
- qty_u is integer 0.01 kWh units (qty_u = floor(kWh * 100)).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from Common_Functions import generate_household_profiles, generate_pv_profiles


# ---------------------------
# LOCKED RESOLUTION
# ---------------------------
INTERVAL_MINUTES = 15
NUM_INTERVALS_PER_DAY = 96
INTERVAL_HOURS = INTERVAL_MINUTES / 60.0


# ---------------------------
# Data Structures
# ---------------------------
@dataclass(frozen=True)
class Participant:
    addr: str
    role: str  # "prosumer" or "consumer"


# ---------------------------
# Helpers
# ---------------------------
def _timestamp_folder() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_participants_csv(path: Path) -> List[Participant]:
    if not path.exists():
        raise FileNotFoundError(
            f"participants.csv not found at: {path}\n"
            f"Expected columns: addr, role (prosumer/consumer)."
        )

    df = pd.read_csv(path)
    required = {"addr", "role"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    participants: List[Participant] = []
    for _, row in df.iterrows():
        addr = str(row["addr"]).strip()
        role = str(row["role"]).strip().lower()

        if not addr:
            raise ValueError("Empty addr found in participants.csv")

        if role not in {"prosumer", "consumer"}:
            raise ValueError(f"Invalid role '{role}' for addr {addr}. Use prosumer/consumer.")

        participants.append(Participant(addr=addr, role=role))

    if not participants:
        raise ValueError("participants.csv contains zero rows.")

    return participants


def _kwh_to_qty_u(energy_kwh: np.ndarray) -> np.ndarray:
    """
    Convert kWh per interval to qty_u integer units (0.01 kWh).
    qty_u = floor(kWh * 100)
    """
    u = np.floor(energy_kwh * 100.0 + 1e-12).astype(np.int64)
    u[u < 0] = 0
    return u


def _qty_u_to_kwh(qty_u: np.ndarray) -> np.ndarray:
    """0.01 kWh units -> kWh"""
    return np.asarray(qty_u, dtype=float) / 100.0


def _allocate_planned_trades_deterministic(
    sellers: List[str],
    buyers: List[str],
    supply_u: np.ndarray,   # per seller
    demand_u: np.ndarray,   # per buyer
    clear_u: int,
) -> List[Tuple[str, str, int]]:
    """
    Deterministic allocation of clear_u units from sellers to buyers.

    Strategy:
    - Sort sellers and buyers lexicographically (stable)
    - Allocate seller supply across buyers proportionally to remaining demand,
      using largest-remainder rounding for exact integer conservation.
    - Includes a deterministic "top-up" pass to correct any rounding shortfall.
    """
    if clear_u <= 0:
        return []

    sellers_sorted = sorted(zip(sellers, supply_u.tolist()), key=lambda x: x[0])
    buyers_sorted = sorted(zip(buyers, demand_u.tolist()), key=lambda x: x[0])

    seller_rem: Dict[str, int] = {a: int(u) for a, u in sellers_sorted}
    buyer_rem: Dict[str, int] = {a: int(u) for a, u in buyers_sorted}

    total_supply = sum(seller_rem.values())
    total_demand = sum(buyer_rem.values())
    if total_supply <= 0 or total_demand <= 0:
        return []

    clear_u = int(min(clear_u, total_supply, total_demand))
    if clear_u <= 0:
        return []

    trades: List[Tuple[str, str, int]] = []
    remaining_to_clear = clear_u

    for seller_addr, _s_u in sellers_sorted:
        if remaining_to_clear <= 0:
            break

        s_available = min(seller_rem[seller_addr], remaining_to_clear)
        if s_available <= 0:
            continue

        buyers_with_demand = [(b, buyer_rem[b]) for b, _ in buyers_sorted if buyer_rem[b] > 0]
        if not buyers_with_demand:
            break

        total_b_rem = sum(d for _, d in buyers_with_demand)
        s_available = min(s_available, total_b_rem)

        # Proportional split
        raw = [(b, s_available * d / total_b_rem) for b, d in buyers_with_demand]
        base = [(b, int(np.floor(x))) for b, x in raw]
        allocated = sum(q for _, q in base)
        remainder = s_available - allocated

        # Largest remainder with deterministic tie-break by buyer address
        frac = sorted(
            [(b, (x - np.floor(x))) for b, x in raw],
            key=lambda t: (-t[1], t[0])
        )
        bump_set = {b for b, _ in frac[:remainder]}

        # Apply base + remainder, clamped by buyer remaining demand
        for b, q in base:
            q_final = q + (1 if b in bump_set else 0)
            if q_final <= 0:
                continue
            q_final = min(q_final, buyer_rem[b])
            if q_final <= 0:
                continue
            trades.append((seller_addr, b, q_final))
            buyer_rem[b] -= q_final
            seller_rem[seller_addr] -= q_final
            remaining_to_clear -= q_final

    # Deterministic top-up pass (very rare)
    if remaining_to_clear > 0:
        seller_order = [a for a, _ in sellers_sorted]
        buyer_order = [a for a, _ in buyers_sorted]
        for s in seller_order:
            if remaining_to_clear <= 0:
                break
            if seller_rem[s] <= 0:
                continue
            for b in buyer_order:
                if remaining_to_clear <= 0:
                    break
                if buyer_rem[b] <= 0:
                    continue
                q = min(seller_rem[s], buyer_rem[b], remaining_to_clear)
                if q <= 0:
                    continue
                trades.append((s, b, q))
                buyer_rem[b] -= q
                seller_rem[s] -= q
                remaining_to_clear -= q

    return trades


def _validate_interval_trades(
    t: int,
    trades: List[Tuple[str, str, int]],
    sellers: List[str],
    buyers: List[str],
    supply_u: np.ndarray,
    demand_u: np.ndarray,
    planned_clear_u: int,
) -> Tuple[bool, str]:
    """
    Validate:
    - all sellers/buyers are from allowed lists
    - non-negative quantities
    - sum(trades) == planned_clear_u
    - per-seller sold <= supply_u
    - per-buyer bought <= demand_u
    """
    if planned_clear_u <= 0:
        if len(trades) != 0:
            return False, f"t={t}: planned_clear_u=0 but trades exist"
        return True, ""

    seller_set = set(sellers)
    buyer_set = set(buyers)

    sold: Dict[str, int] = {}
    bought: Dict[str, int] = {}
    total = 0

    for s, b, q in trades:
        if s not in seller_set:
            return False, f"t={t}: invalid seller in trades: {s}"
        if b not in buyer_set:
            return False, f"t={t}: invalid buyer in trades: {b}"
        if int(q) <= 0:
            return False, f"t={t}: non-positive trade qty: {q}"
        total += int(q)
        sold[s] = sold.get(s, 0) + int(q)
        bought[b] = bought.get(b, 0) + int(q)

    if total != int(planned_clear_u):
        return False, f"t={t}: sum(trades)={total} != planned_clear_u={planned_clear_u}"

    # supply bounds
    supply_map = {sellers[i]: int(supply_u[i]) for i in range(len(sellers))}
    for s, q in sold.items():
        if q > supply_map.get(s, 0):
            return False, f"t={t}: seller {s} sold {q} > supply {supply_map.get(s, 0)}"

    demand_map = {buyers[i]: int(demand_u[i]) for i in range(len(buyers))}
    for b, q in bought.items():
        if q > demand_map.get(b, 0):
            return False, f"t={t}: buyer {b} bought {q} > demand {demand_map.get(b, 0)}"

    return True, ""


# ---------------------------
# Battery model (Option A)
# ---------------------------
def _battery_step_pre_trade(
    net_kwh: float,
    soc_kwh: float,
    cap_kwh: float,
    soc_threshold_frac: float,
) -> Tuple[float, float, float, float]:
    """
    Apply battery BEFORE market formation, aligned with S2 logic:
      - If deficit: discharge to cover deficit
      - If surplus: charge up to threshold (e.g., 90% SoC), remainder becomes market supply

    Returns:
      net_post_kwh, soc_next_kwh, charge_kwh, discharge_kwh
    """
    charge = 0.0
    discharge = 0.0
    soc_next = soc_kwh
    net_post = net_kwh

    # 1) discharge first for deficits
    if net_post < 0:
        from_batt = min(-net_post, soc_next)
        soc_next -= from_batt
        net_post += from_batt
        discharge += from_batt

    # 2) if surplus, charge up to threshold
    if net_post > 0:
        desired_soc = cap_kwh * soc_threshold_frac
        if soc_next < desired_soc:
            to_store = min(net_post, desired_soc - soc_next)
            soc_next += to_store
            net_post -= to_store
            charge += to_store

    return net_post, soc_next, charge, discharge


def _battery_post_trade_store(
    remaining_surplus_kwh: float,
    soc_kwh: float,
    cap_kwh: float,
) -> Tuple[float, float, float]:
    """
    After market clears, take remaining prosumer surplus and charge battery to FULL.
    Any excess beyond full is exported to the utility (Option A: utility FiT == market floor as outside option).

    Returns:
      soc_next_kwh, extra_charge_kwh, utility_export_kwh
    """
    soc_next = soc_kwh
    extra_charge = 0.0
    export_kwh = 0.0

    if remaining_surplus_kwh > 0:
        to_store = min(remaining_surplus_kwh, cap_kwh - soc_next)
        soc_next += to_store
        extra_charge += to_store
        export_kwh = remaining_surplus_kwh - to_store

    return soc_next, extra_charge, export_kwh


# ---------------------------
# Main
# ---------------------------
def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--participants", default="participants.csv", help="Path to participants.csv (addr,role).")
    parser.add_argument("--out-root", default="runs", help="Root output folder.")
    parser.add_argument("--run-name", default="", help="Optional suffix added to run folder name.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for profile generation.")
    parser.add_argument("--pv-peak-kw", type=float, default=5.0, help="Peak PV generation (kW) used by Common_Functions.")
    parser.add_argument("--num-intervals", type=int, default=NUM_INTERVALS_PER_DAY,
                        help="Must be 96 (locked).")
    parser.add_argument("--lenient", action="store_true",
                        help="Do not fail on validation mismatches; log failures to oracle_meta.json (NOT recommended).")

    # Battery parameters (kept explicit here so the oracle is self-contained)
    parser.add_argument("--battery-capacity-kwh", type=float, default=10.0, help="Prosumer battery capacity (kWh).")
    parser.add_argument("--battery-init-frac", type=float, default=0.5, help="Initial SoC as fraction of capacity.")
    parser.add_argument("--battery-soc-threshold", type=float, default=0.90,
                        help="SoC threshold to prioritize selling over storing (charge only up to this pre-trade).")

    args = parser.parse_args()

    # Enforce locked temporal resolution
    if int(args.num_intervals) != NUM_INTERVALS_PER_DAY:
        raise ValueError(
            f"This oracle is fixed to {NUM_INTERVALS_PER_DAY} intervals/day "
            f"({INTERVAL_MINUTES}-minute resolution). Received num_intervals={args.num_intervals}."
        )

    strict = not args.lenient
    np.random.seed(args.seed)

    participants_path = Path(args.participants)
    participants = _read_participants_csv(participants_path)

    prosumers = [p for p in participants if p.role == "prosumer"]
    consumers = [p for p in participants if p.role == "consumer"]

    n = len(participants)
    num_intervals = NUM_INTERVALS_PER_DAY

    # Generate profiles (LOCKED: already kWh per 15-minute interval per Common_Functions.py)
    loads_kwh = generate_household_profiles(num_households=n, num_intervals=num_intervals)

    pv_kwh = np.zeros((len(prosumers), num_intervals), dtype=float)
    if len(prosumers) > 0:
        pv_kwh = generate_pv_profiles(
            num_prosumers=len(prosumers),
            num_intervals=num_intervals,
            peak_generation=float(args.pv_peak_kw),
        )

    # Index mapping
    idx_to_addr = {i: participants[i].addr for i in range(n)}
    prosumer_indices = [i for i, p in enumerate(participants) if p.role == "prosumer"]
    consumer_indices = [i for i, p in enumerate(participants) if p.role == "consumer"]

    # ---------------------------
    # Battery state init
    # ---------------------------
    cap_kwh = float(args.battery_capacity_kwh)
    init_soc = float(args.battery_init_frac) * cap_kwh
    soc_threshold = float(args.battery_soc_threshold)

    # battery_soc_kwh is per prosumer, length num_intervals+1
    battery_soc_kwh = np.zeros((len(prosumer_indices), num_intervals + 1), dtype=float)
    if len(prosumer_indices) > 0:
        battery_soc_kwh[:, 0] = init_soc

    # We'll compute per-interval aggregated battery metrics (kWh, then convert to u)
    batt_charge_kwh = np.zeros(num_intervals, dtype=float)
    batt_discharge_kwh = np.zeros(num_intervals, dtype=float)
    batt_curtail_kwh = np.zeros(num_intervals, dtype=float)
    utility_export_kwh = np.zeros(num_intervals, dtype=float)  # surplus exported to utility after battery + P2P

    # For plotting SoC
    avg_soc_pct = np.zeros(num_intervals + 1, dtype=float)

    # ---------------------------
    # Pre-trade: compute net after battery threshold behavior
    # ---------------------------
    # net_post_kwh_all: per participant per interval AFTER pre-trade battery (threshold)
    net_post_kwh_all = np.zeros((n, num_intervals), dtype=float)

    # Assign PV to prosumers in participant order
    # pv_kwh[p_idx, t] corresponds to participant at prosumer_indices[p_idx]
    for t in range(num_intervals):
        # Update avg SoC at interval start
        if len(prosumer_indices) > 0:
            avg_soc_pct[t] = float(np.mean(battery_soc_kwh[:, t]) / cap_kwh * 100.0)
        else:
            avg_soc_pct[t] = 0.0

        # Process each prosumer with battery
        for p_idx, part_idx in enumerate(prosumer_indices):
            pv = float(pv_kwh[p_idx, t])
            load = float(loads_kwh[part_idx, t])
            net = pv - load  # raw

            soc_now = float(battery_soc_kwh[p_idx, t])
            net_post, soc_next, ch, dis = _battery_step_pre_trade(
                net_kwh=net,
                soc_kwh=soc_now,
                cap_kwh=cap_kwh,
                soc_threshold_frac=soc_threshold,
            )

            battery_soc_kwh[p_idx, t + 1] = soc_next  # provisional; may be increased post-trade
            batt_charge_kwh[t] += ch
            batt_discharge_kwh[t] += dis
            net_post_kwh_all[part_idx, t] = net_post

        # Consumers have only load (deficit)
        for part_idx in consumer_indices:
            net_post_kwh_all[part_idx, t] = -float(loads_kwh[part_idx, t])

    # last SoC sample for plotting
    if len(prosumer_indices) > 0:
        avg_soc_pct[num_intervals] = float(np.mean(battery_soc_kwh[:, num_intervals]) / cap_kwh * 100.0)
    else:
        avg_soc_pct[num_intervals] = 0.0

    # Convert to qty_u (0.01 kWh integer units) for market formation
    surplus_u = _kwh_to_qty_u(np.maximum(net_post_kwh_all, 0.0))
    deficit_u = _kwh_to_qty_u(np.maximum(-net_post_kwh_all, 0.0))

    # Prepare output folder
    run_folder = _timestamp_folder()
    if args.run_name.strip():
        run_folder = f"{run_folder}_{args.run_name.strip()}"
    out_dir = Path(args.out_root) / run_folder
    _ensure_dir(out_dir)

    # Copy participants file for provenance
    participants_out = out_dir / "participants.csv"
    pd.read_csv(participants_path).to_csv(participants_out, index=False)

    interval_rows: List[Dict] = []
    trade_rows: List[Dict] = []
    validation_failures: List[str] = []

    seller_addrs = [idx_to_addr[i] for i in prosumer_indices]
    buyer_addrs = [idx_to_addr[i] for i in consumer_indices]

    # For post-trade battery completion (threshold -> full), we need per-interval per-seller remaining surplus
    # We'll compute sold_u per seller from trades, then store remaining kWh into battery to full.
    for t in range(num_intervals):
        s_u_vec = np.array([int(surplus_u[i, t]) for i in prosumer_indices], dtype=np.int64)
        d_u_vec = np.array([int(deficit_u[i, t]) for i in consumer_indices], dtype=np.int64)

        total_supply_u = int(s_u_vec.sum())
        total_demand_u = int(d_u_vec.sum())
        planned_clear_u = int(min(total_supply_u, total_demand_u))

        # Track basic physical totals (optional, but helpful for plotting / sanity)
        total_load_kwh = float(np.sum(loads_kwh[:, t]))
        total_pv_kwh = float(np.sum(pv_kwh[:, t])) if pv_kwh.size else 0.0

        interval_rows.append({
            "interval": t,
            "total_supply_u": total_supply_u,
            "total_demand_u": total_demand_u,
            "planned_clear_u": planned_clear_u,

            # EXTRA columns (safe for 5.8 to ignore)
            "total_load_kwh": total_load_kwh,
            "total_pv_kwh": total_pv_kwh,
            "battery_charge_u_pre": int(np.floor(batt_charge_kwh[t] * 100.0 + 1e-12)),
            "battery_discharge_u_pre": int(np.floor(batt_discharge_kwh[t] * 100.0 + 1e-12)),
            "utility_export_u": 0,
        })

        if planned_clear_u <= 0:
            # Even if no trades, still allow post-trade store from remaining supply (which is all supply)
            sold_u_by_seller = np.zeros(len(prosumer_indices), dtype=np.int64)
        else:
            trades = _allocate_planned_trades_deterministic(
                sellers=seller_addrs,
                buyers=buyer_addrs,
                supply_u=s_u_vec,
                demand_u=d_u_vec,
                clear_u=planned_clear_u,
            )

            ok, msg = _validate_interval_trades(
                t=t,
                trades=trades,
                sellers=seller_addrs,
                buyers=buyer_addrs,
                supply_u=s_u_vec,
                demand_u=d_u_vec,
                planned_clear_u=planned_clear_u,
            )
            if not ok:
                validation_failures.append(msg)
                if strict:
                    raise ValueError(f"Trade validation failed: {msg}")

            sold_u_by_seller = np.zeros(len(prosumer_indices), dtype=np.int64)
            seller_index = {addr: i for i, addr in enumerate(seller_addrs)}

            for seller, buyer, q_u in trades:
                if q_u <= 0:
                    continue
                trade_rows.append({
                    "interval": t,
                    "seller_addr": seller,
                    "buyer_addr": buyer,
                    "qty_u_planned": int(q_u),
                })
                sold_u_by_seller[seller_index[seller]] += int(q_u)

        # ---------------------------
        # Post-trade: store remaining seller surplus to FULL (threshold -> full)
        # ---------------------------
        # remaining_u per seller = supply_u - sold_u
        if len(prosumer_indices) > 0:
            remaining_u = np.maximum(s_u_vec - sold_u_by_seller, 0)
            remaining_kwh = _qty_u_to_kwh(remaining_u)

            extra_charge_kwh = 0.0
            export_kwh = 0.0

            for p_idx, part_idx in enumerate(prosumer_indices):
                soc_now = float(battery_soc_kwh[p_idx, t + 1])  # after pre-trade step
                soc_next, ch2, exp = _battery_post_trade_store(
                    remaining_surplus_kwh=float(remaining_kwh[p_idx]),
                    soc_kwh=soc_now,
                    cap_kwh=cap_kwh,
                )
                battery_soc_kwh[p_idx, t + 1] = soc_next
                extra_charge_kwh += ch2
                export_kwh += exp

            batt_charge_kwh[t] += extra_charge_kwh
            utility_export_kwh[t] += export_kwh

            # Add extra columns to interval row for completeness
            interval_rows[-1].update({
                "battery_charge_u_total": int(np.floor(batt_charge_kwh[t] * 100.0 + 1e-12)),
                "battery_discharge_u_total": int(np.floor(batt_discharge_kwh[t] * 100.0 + 1e-12)),
                "curtail_u": int(np.floor(batt_curtail_kwh[t] * 100.0 + 1e-12)),
                "utility_export_u": int(np.floor(utility_export_kwh[t] * 100.0 + 1e-12)),
            })
        else:
            interval_rows[-1].update({
                "battery_charge_u_total": 0,
                "battery_discharge_u_total": 0,
                "curtail_u": 0,
                "utility_export_u": 0,
            })

    # Write outputs
    interval_inputs_path = out_dir / "interval_inputs.csv"
    planned_trades_path = out_dir / "planned_trades.csv"
    battery_ts_path = out_dir / "battery_timeseries.csv"

    pd.DataFrame(interval_rows).to_csv(interval_inputs_path, index=False)
    pd.DataFrame(trade_rows).to_csv(planned_trades_path, index=False)

    # Battery time series (for plotting S3 like S1/S2)
    # Note: avg_soc_pct has length 97 (includes final), but your plotting should use [:96]
    batt_df = pd.DataFrame({
        "interval": np.arange(num_intervals, dtype=int),
        "avg_soc_pct": avg_soc_pct[:num_intervals],
        "battery_charge_u": np.floor(batt_charge_kwh[:num_intervals] * 100.0 + 1e-12).astype(int),
        "battery_discharge_u": np.floor(batt_discharge_kwh[:num_intervals] * 100.0 + 1e-12).astype(int),
        "curtail_u": np.floor(batt_curtail_kwh[:num_intervals] * 100.0 + 1e-12).astype(int),
        "utility_export_u": np.floor(utility_export_kwh[:num_intervals] * 100.0 + 1e-12).astype(int),
    })
    batt_df.to_csv(battery_ts_path, index=False)

    meta = {
        "run_folder": str(out_dir),
        "seed": args.seed,
        "locked_interval_minutes": INTERVAL_MINUTES,
        "locked_num_intervals": NUM_INTERVALS_PER_DAY,
        "num_participants": n,
        "num_prosumers": len(prosumers),
        "num_consumers": len(consumers),
        "pv_peak_kw": float(args.pv_peak_kw),
        "battery": {
            "capacity_kwh": cap_kwh,
            "init_frac": float(args.battery_init_frac),
            "soc_threshold": soc_threshold,
        },
        "validation": {
            "strict": strict,
            "failures_count": len(validation_failures),
            "failures_preview": validation_failures[:10],
        },
        "files": {
            "participants.csv": {"path": str(participants_out), "sha256": _sha256_file(participants_out)},
            "interval_inputs.csv": {"path": str(interval_inputs_path), "sha256": _sha256_file(interval_inputs_path)},
            "planned_trades.csv": {"path": str(planned_trades_path), "sha256": _sha256_file(planned_trades_path)},
            "battery_timeseries.csv": {"path": str(battery_ts_path), "sha256": _sha256_file(battery_ts_path)},
        },
        "notes": [
            "Trades planned up to planned_clear_u = min(total_supply_u, total_demand_u) per interval (no feeder cap here).",
            "Feeder cap scaling and MCP enforcement occur on-chain during settlement (allowed_u authoritative).",
            "qty_u_planned uses 0.01 kWh units (integer).",
            "Battery is applied pre-trade (discharge deficits, charge to threshold) then post-trade (charge remaining to full, export any excess to utility; physical curtailment should be ~0).",
        ],
    }

    meta_path = out_dir / "oracle_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n--- ✅ Oracle schedule generation complete (LOCKED 15-min / 96 intervals) ---")
    print(f"Output folder: {out_dir}")
    print(f"- interval_inputs.csv rows: {len(interval_rows)}")
    print(f"- planned_trades.csv rows: {len(trade_rows)}")
    print(f"- battery_timeseries.csv rows: {len(batt_df)}")
    print(f"- validation failures: {len(validation_failures)} (strict={strict})")
    print("- oracle_meta.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3_Results_And_Figures.py

Model S3 (On-Chain) – Results, KPIs, Fairness Metrics & Figures

UPDATED (Rev – contiguous shading + authoritative physical totals):
- Uses interval_inputs.csv as the single source of truth for:
    * total_load_kwh per interval
    * total_pv_kwh per interval
- Loads battery_timeseries.csv (from Script 5.7) if available
- Battery-aware grid import + export + curtailment accounting (Option A)
- Adds battery charging/discharging and utility export to S3 plots (S1/S2 parity)
- FIX: Diagnostic plot shading now uses contiguous bands (no vertical stripe effect)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import Common_Functions as common


# -----------------------------
# Constants
# -----------------------------

INTERVAL_MINUTES = 15
INTERVALS_PER_DAY = 96
KWH_PER_U = 0.01                 # 0.01 kWh units
KW_PER_KWH_INTERVAL = 60 / 15    # 4.0


# -----------------------------
# File helpers
# -----------------------------

def _find_one(folder: Path, patterns) -> Path:
    hits = []
    for pat in patterns:
        hits += sorted(folder.glob(pat))
        hits += sorted(folder.glob(pat.upper()))
        hits += sorted(folder.glob(pat.lower()))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(f"No files found in {folder} matching: {patterns}")
    preferred = [p for p in hits if "final" in p.name.lower()]
    return preferred[0] if preferred else hits[0]


def _try_find_one(folder: Path, patterns) -> Optional[Path]:
    try:
        return _find_one(folder, patterns)
    except FileNotFoundError:
        return None


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6)


def _time_of_day_labels(n: int) -> np.ndarray:
    return pd.to_datetime(
        pd.date_range(start="00:00", periods=n, freq=f"{INTERVAL_MINUTES}min")
    ).strftime("%H:%M").to_numpy()


def _runs_of_true(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert boolean mask into contiguous [start, end] runs (inclusive indices).
    Example: [F, T, T, F, T] -> [(1,2),(4,4)]
    """
    mask = np.asarray(mask, dtype=bool)
    runs = []
    if mask.size == 0:
        return runs
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, mask.size - 1))
    return runs


# -----------------------------
# Load
# -----------------------------

def load_inputs(in_dir: Path) -> Dict[str, object]:
    participants_path = _find_one(in_dir, ["*participants*.csv"])
    oracle_meta_path = _find_one(in_dir, ["*oracle_meta*.json"])
    interval_inputs_path = _find_one(in_dir, ["*interval_inputs*.csv"])
    planned_trades_path = _find_one(in_dir, ["*planned_trades*.csv"])

    interval_summary_path = _find_one(in_dir, ["*interval_summary*.csv"])
    trade_log_path = _find_one(in_dir, ["*trade_log*.csv"])

    battery_ts_path = _try_find_one(in_dir, ["*battery_timeseries*.csv"])

    participants = pd.read_csv(participants_path)
    oracle_meta = json.loads(oracle_meta_path.read_text(encoding="utf-8"))
    interval_inputs = pd.read_csv(interval_inputs_path)
    planned_trades = pd.read_csv(planned_trades_path)

    interval_summary = pd.read_csv(interval_summary_path)
    trade_log = pd.read_csv(trade_log_path)

    battery_ts = None
    if battery_ts_path is not None:
        battery_ts = pd.read_csv(battery_ts_path)

    return {
        "participants": participants,
        "oracle_meta": oracle_meta,
        "interval_inputs": interval_inputs,
        "planned_trades": planned_trades,
        "interval_summary": interval_summary,
        "trade_log": trade_log,
        "battery_ts": battery_ts,
    }


# -----------------------------
# Physical series (authoritative: interval_inputs.csv)
# -----------------------------

def build_physical_series_from_interval_inputs(interval_inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Uses authoritative physical totals exported by Script 5.7:
      - total_load_kwh (per interval)
      - total_pv_kwh (per interval)
    Converts to kW for plotting (15-min intervals => kW = kWh * 4).
    """
    df = interval_inputs.copy()

    if "interval" not in df.columns:
        raise KeyError("interval_inputs.csv missing 'interval' column")

    required = ["total_load_kwh", "total_pv_kwh"]
    for c in required:
        if c not in df.columns:
            raise KeyError(
                f"interval_inputs.csv missing '{c}'. "
                "Re-run Script 5.7 (battery-integrated version) to regenerate interval_inputs.csv."
            )

    df["interval"] = pd.to_numeric(df["interval"], errors="coerce").fillna(0).astype(int)
    df["total_load_kwh"] = pd.to_numeric(df["total_load_kwh"], errors="coerce").fillna(0.0)
    df["total_pv_kwh"] = pd.to_numeric(df["total_pv_kwh"], errors="coerce").fillna(0.0)

    df = df.sort_values("interval").reset_index(drop=True)

    return pd.DataFrame({
        "interval": df["interval"].to_numpy(dtype=int),
        "community_load_kw": (df["total_load_kwh"].to_numpy(dtype=float) * KW_PER_KWH_INTERVAL),
        "community_pv_kw": (df["total_pv_kwh"].to_numpy(dtype=float) * KW_PER_KWH_INTERVAL),
    })


# -----------------------------
# Battery series (from 5.7 battery_timeseries.csv)
# -----------------------------

def build_battery_series(battery_ts: Optional[pd.DataFrame], num_intervals: int) -> pd.DataFrame:
    """
    battery_timeseries.csv columns (from updated 5.7):
      interval, avg_soc_pct, battery_charge_u, battery_discharge_u, curtail_u, utility_export_u (optional)
    """
    if battery_ts is None:
        return pd.DataFrame({
            "interval": np.arange(num_intervals, dtype=int),
            "avg_soc_pct": np.zeros(num_intervals, dtype=float),
            "battery_charge_u": np.zeros(num_intervals, dtype=float),
            "battery_discharge_u": np.zeros(num_intervals, dtype=float),
            "curtail_u": np.zeros(num_intervals, dtype=float),
            "utility_export_u": np.zeros(num_intervals, dtype=float),
            "battery_charge_kw": np.zeros(num_intervals, dtype=float),
            "battery_discharge_kw": np.zeros(num_intervals, dtype=float),
            "curtail_kw": np.zeros(num_intervals, dtype=float),
            "utility_export_kw": np.zeros(num_intervals, dtype=float),
        })

    df = battery_ts.copy()
    if "interval" not in df.columns:
        raise KeyError("battery_timeseries.csv missing 'interval' column")

    # Accept either naming (in case you exported *_total or *_pre previously)
    # Prefer the canonical names used by the battery-integrated 5.7 output
    rename_map = {}
    if "battery_charge_u" not in df.columns and "battery_charge_u_total" in df.columns:
        rename_map["battery_charge_u_total"] = "battery_charge_u"
    if "battery_discharge_u" not in df.columns and "battery_discharge_u_total" in df.columns:
        rename_map["battery_discharge_u_total"] = "battery_discharge_u"
    if rename_map:
        df = df.rename(columns=rename_map)

    for c in ["avg_soc_pct", "battery_charge_u", "battery_discharge_u", "curtail_u"]:
        if c not in df.columns:
            raise KeyError(f"battery_timeseries.csv missing '{c}' column")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["interval"] = pd.to_numeric(df["interval"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("interval").reset_index(drop=True)

    # utility_export_u is optional (added in 5.7 Rev2+). If absent, treat as zero.
    if "utility_export_u" not in df.columns:
        df["utility_export_u"] = 0.0
    df["utility_export_u"] = pd.to_numeric(df["utility_export_u"], errors="coerce").fillna(0.0)

    df["battery_charge_kw"] = df["battery_charge_u"] * KWH_PER_U * KW_PER_KWH_INTERVAL
    df["battery_discharge_kw"] = df["battery_discharge_u"] * KWH_PER_U * KW_PER_KWH_INTERVAL
    df["curtail_kw"] = df["curtail_u"] * KWH_PER_U * KW_PER_KWH_INTERVAL
    df["utility_export_kw"] = df["utility_export_u"] * KWH_PER_U * KW_PER_KWH_INTERVAL

    # Ensure length matches
    if len(df) != num_intervals:
        df = df.set_index("interval").reindex(range(num_intervals)).fillna(0.0).reset_index()
        df.rename(columns={"index": "interval"}, inplace=True)

    return df


# -----------------------------
# Settlement series
# -----------------------------

def build_settlement_series(interval_summary: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
    df = interval_summary.copy()
    if "interval" not in df.columns:
        raise KeyError("interval_summary missing 'interval' column")

    df["interval"] = pd.to_numeric(df["interval"], errors="coerce").fillna(0).astype(int)

    tl = trade_log.copy()
    if "ok" in tl.columns:
        tl = tl.loc[pd.to_numeric(tl["ok"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if "qty_u" not in tl.columns:
        raise KeyError("trade_log missing qty_u")

    tl["interval"] = pd.to_numeric(tl["interval"], errors="coerce").fillna(0).astype(int)
    settled_u = tl.groupby("interval")["qty_u"].sum()
    df["settled_u"] = df["interval"].map(settled_u.to_dict()).fillna(0.0)

    df["settled_kwh"] = df["settled_u"].astype(float) * KWH_PER_U
    df["settled_kw"] = df["settled_kwh"].astype(float) * KW_PER_KWH_INTERVAL

    required = [
        "planned_clear_u", "allowed_u", "scaled_total_u", "mcp_micro",
        "execute_ok", "finalize_ok", "feeder_cap_units",
        "total_supply_u", "total_demand_u"
    ]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"interval_summary missing '{c}'")

    df["planned_clear_u"] = pd.to_numeric(df["planned_clear_u"], errors="coerce").fillna(0.0)
    df["allowed_u"] = pd.to_numeric(df["allowed_u"], errors="coerce").fillna(0.0)
    df["scaled_total_u"] = pd.to_numeric(df["scaled_total_u"], errors="coerce").fillna(0.0)

    df["cap_binding"] = df["allowed_u"] < df["planned_clear_u"]
    df["has_trading"] = df["settled_u"].astype(float) > 0

    return df.sort_values("interval").reset_index(drop=True)


# -----------------------------
# Fairness
# -----------------------------

def _gini_nonneg(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if np.allclose(x, 0.0):
        return 0.0
    x = np.clip(x, 0.0, None)
    return float(common.calculate_gini_index(x))


def _gini_shift_allow_negative(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if np.allclose(x, 0.0):
        return 0.0
    mn = float(np.min(x))
    if mn < 0:
        x = x - mn
    if np.allclose(x, 0.0):
        return 0.0
    return float(common.calculate_gini_index(x))


def compute_fairness(
    participants: pd.DataFrame,
    interval_summary: pd.DataFrame,
    trade_log: pd.DataFrame,
    interval_inputs: pd.DataFrame,
    planned_trades: pd.DataFrame,
    utility_tariff: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    addrs = participants["addr"].astype(str).tolist()

    mcp_map = interval_summary.drop_duplicates("interval").set_index("interval")["mcp_micro"].astype(float) / 1e6

    tl = trade_log.copy()
    if "ok" in tl.columns:
        tl = tl.loc[pd.to_numeric(tl["ok"], errors="coerce").fillna(0).astype(int) == 1].copy()

    tl["interval"] = pd.to_numeric(tl["interval"], errors="coerce").fillna(0).astype(int)
    tl["qty_u"] = pd.to_numeric(tl["qty_u"], errors="coerce").fillna(0.0)

    tl["energy_kwh"] = tl["qty_u"].astype(float) * KWH_PER_U
    tl["mcp_zar_per_kwh"] = tl["interval"].map(mcp_map.to_dict()).astype(float)

    tl["buyer_cs_zar"] = (utility_tariff - tl["mcp_zar_per_kwh"]) * tl["energy_kwh"]
    tl["seller_pp_zar"] = tl["mcp_zar_per_kwh"] * tl["energy_kwh"]

    buyer_cs = tl.groupby("buyer_addr")["buyer_cs_zar"].sum()
    seller_pp = tl.groupby("seller_addr")["seller_pp_zar"].sum()

    benefits = pd.Series(0.0, index=addrs)
    benefits = benefits.add(buyer_cs, fill_value=0.0)
    benefits = benefits.add(seller_pp, fill_value=0.0)

    ii = interval_inputs[["interval", "total_demand_u", "planned_clear_u"]].copy()
    ii["interval"] = pd.to_numeric(ii["interval"], errors="coerce").fillna(0).astype(int)
    ii["total_demand_u"] = pd.to_numeric(ii["total_demand_u"], errors="coerce").fillna(0.0)
    ii["planned_clear_u"] = pd.to_numeric(ii["planned_clear_u"], errors="coerce").fillna(0.0)

    pr = planned_trades.copy()
    pr["interval"] = pd.to_numeric(pr["interval"], errors="coerce").fillna(0).astype(int)
    pr["qty_u_planned"] = pd.to_numeric(pr["qty_u_planned"], errors="coerce").fillna(0.0)
    pr = pr.groupby(["interval", "buyer_addr"], as_index=False)["qty_u_planned"].sum()

    merged = pr.merge(ii, on="interval", how="left")
    merged = merged.loc[merged["planned_clear_u"] > 0].copy()
    merged["share"] = merged["qty_u_planned"].astype(float) / merged["planned_clear_u"].astype(float)
    merged["demand_u_est"] = merged["share"] * merged["total_demand_u"].astype(float)

    demand_u_by_agent = merged.groupby("buyer_addr")["demand_u_est"].sum()
    recv_u = tl.groupby("buyer_addr")["qty_u"].sum()

    demand_u = pd.Series(0.0, index=addrs).add(demand_u_by_agent, fill_value=0.0)
    recv_u = pd.Series(0.0, index=addrs).add(recv_u, fill_value=0.0)

    grid_import_u_est = (demand_u - recv_u).clip(lower=0.0)
    utility_costs = grid_import_u_est * KWH_PER_U * utility_tariff

    net_outcomes = benefits - utility_costs

    fairness = {
        "gini_p2p_benefits": _gini_nonneg(benefits.values),
        "gini_utility_costs": _gini_nonneg(utility_costs.values),
        "gini_net_outcomes": _gini_shift_allow_negative(net_outcomes.values),
    }

    agent_df = pd.DataFrame({
        "addr": addrs,
        "p2p_benefit_zar": benefits.values,
        "utility_cost_zar_est": utility_costs.values,
        "net_outcome_zar_est": net_outcomes.values,
        "demand_kwh_est": (demand_u.values * KWH_PER_U),
        "p2p_recv_kwh": (recv_u.values * KWH_PER_U),
        "grid_import_kwh_est": (grid_import_u_est.values * KWH_PER_U),
    })

    return fairness, agent_df


# -----------------------------
# KPI summary (battery-aware)
# -----------------------------

def compute_kpis(
    interval_summary: pd.DataFrame,
    interval_inputs: pd.DataFrame,
    phys: pd.DataFrame,
    batt: pd.DataFrame,
    settlement: pd.DataFrame,
) -> Dict[str, float]:
    interval_hours = INTERVAL_MINUTES / 60.0

    total_load_kwh = float(pd.to_numeric(interval_inputs["total_load_kwh"], errors="coerce").fillna(0.0).sum())
    total_pv_kwh = float(pd.to_numeric(interval_inputs["total_pv_kwh"], errors="coerce").fillna(0.0).sum())

    # Settled P2P energy: use scaled_total_u (matches your prior KPI output convention)
    total_p2p_kwh = float((pd.to_numeric(interval_summary["scaled_total_u"], errors="coerce").fillna(0.0) * KWH_PER_U).sum())

    curtailed_kwh = float((pd.to_numeric(batt["curtail_u"], errors="coerce").fillna(0.0) * KWH_PER_U).sum())
    utility_export_kwh = float((pd.to_numeric(batt.get("utility_export_u", 0.0), errors="coerce").fillna(0.0) * KWH_PER_U).sum())
    utility_export_revenue_zar = utility_export_kwh * float(getattr(config, "UTILITY_FIT_RATE", 0.0))

    # Battery-aware grid imports (assume no grid charging)
    p2p_kw = pd.to_numeric(settlement["settled_kw"], errors="coerce").fillna(0.0).to_numpy()
    batt_dis_kw = pd.to_numeric(batt["battery_discharge_kw"], errors="coerce").fillna(0.0).to_numpy()
    load_kw = pd.to_numeric(phys["community_load_kw"], errors="coerce").fillna(0.0).to_numpy()
    pv_kw = pd.to_numeric(phys["community_pv_kw"], errors="coerce").fillna(0.0).to_numpy()

    grid_kw = np.maximum(load_kw - pv_kw - batt_dis_kw - p2p_kw, 0.0)
    total_grid_kwh = float((grid_kw * interval_hours).sum())

    # Keep your thesis definition consistent: P2P / Load
    self_consumption_ratio = (total_p2p_kwh / total_load_kwh) if total_load_kwh > 0 else 0.0

    active = interval_summary.loc[pd.to_numeric(interval_summary["allowed_u"], errors="coerce").fillna(0.0) > 0].copy()
    mean_mcp = float(pd.to_numeric(active["mcp_micro"], errors="coerce").mean() / 1e6) if not active.empty else 0.0

    if not active.empty and pd.to_numeric(active["scaled_total_u"], errors="coerce").fillna(0.0).sum() > 0:
        weighted = float(
            np.average(
                pd.to_numeric(active["mcp_micro"], errors="coerce").fillna(0.0),
                weights=pd.to_numeric(active["scaled_total_u"], errors="coerce").fillna(0.0),
            ) / 1e6
        )
    else:
        weighted = 0.0

    feeder_cap_kw = float(pd.to_numeric(interval_summary["feeder_cap_units"], errors="coerce").fillna(0.0).max() * KWH_PER_U * KW_PER_KWH_INTERVAL)

    peak_util = 0.0
    if feeder_cap_kw > 0:
        peak_util = float(
            (pd.to_numeric(interval_summary["scaled_total_u"], errors="coerce").fillna(0.0).max() * KWH_PER_U * KW_PER_KWH_INTERVAL) / feeder_cap_kw
        )

    exec_rate = float(pd.to_numeric(interval_summary["execute_ok"], errors="coerce").fillna(0).mean())
    fin_rate = float(pd.to_numeric(interval_summary["finalize_ok"], errors="coerce").fillna(0).mean())

    return {
        "total_load_kwh": total_load_kwh,
        "total_pv_kwh": total_pv_kwh,
        "total_p2p_kwh": total_p2p_kwh,
        "total_grid_kwh": total_grid_kwh,
        "self_consumption_ratio": self_consumption_ratio,
        "curtailed_kwh": curtailed_kwh,
        "utility_export_kwh": utility_export_kwh,
        "utility_export_revenue_zar": utility_export_revenue_zar,
        "mcp_mean": mean_mcp,
        "mcp_weighted": weighted,
        "feeder_cap_kw": feeder_cap_kw,
        "peak_util": peak_util,
        "exec_rate": exec_rate,
        "finalize_rate": fin_rate,
    }


# -----------------------------
# Figures
# -----------------------------

def plot_community_power_flow(phys: pd.DataFrame, settlement: pd.DataFrame, batt: pd.DataFrame, out_path: Path):
    df = phys.merge(settlement[["interval", "settled_kw"]], on="interval", how="left").copy()
    df = df.merge(batt[["interval", "battery_charge_kw", "battery_discharge_kw", "curtail_kw", "utility_export_kw"]], on="interval", how="left").copy()

    df["settled_kw"] = df["settled_kw"].fillna(0.0)
    df["battery_discharge_kw"] = df["battery_discharge_kw"].fillna(0.0)
    df["battery_charge_kw"] = df["battery_charge_kw"].fillna(0.0)
    df["curtail_kw"] = df["curtail_kw"].fillna(0.0)
    df["utility_export_kw"] = df["utility_export_kw"].fillna(0.0)
    df["utility_export_kw"] = df["utility_export_kw"].fillna(0.0)

    load_kw = df["community_load_kw"].to_numpy(dtype=float)
    pv_kw = df["community_pv_kw"].to_numpy(dtype=float)
    p2p_kw = df["settled_kw"].to_numpy(dtype=float)
    batt_dis_kw = df["battery_discharge_kw"].to_numpy(dtype=float)

    pv_self_kw = np.minimum(load_kw, pv_kw)
    rem_after_pv = np.maximum(load_kw - pv_self_kw, 0.0)

    batt_to_load_kw = np.minimum(rem_after_pv, batt_dis_kw)
    rem_after_batt = np.maximum(load_kw - pv_self_kw - batt_to_load_kw, 0.0)

    p2p_to_load_kw = np.minimum(rem_after_batt, p2p_kw)
    grid_kw = np.maximum(load_kw - pv_self_kw - batt_to_load_kw - p2p_to_load_kw, 0.0)

    time_of_day = _time_of_day_labels(len(df))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.stackplot(
        time_of_day,
        pv_self_kw,
        batt_to_load_kw,
        p2p_to_load_kw,
        grid_kw,
        labels=["PV Self-Consumption", "Battery Discharge", "P2P Import (Settled)", "Grid Import (Residual)"],
        alpha=0.7,
    )

    ax.plot(time_of_day, load_kw, label="Community Load", color="black", linewidth=2.5)
    ax.plot(time_of_day, df["battery_charge_kw"].to_numpy(dtype=float),
            label="Battery Charging (Sink)", linestyle=":", linewidth=2)
    ax.plot(time_of_day, df["utility_export_kw"].to_numpy(dtype=float),
            label="Utility Export (FiT)", linestyle="--", linewidth=2)
    ax.plot(time_of_day, df["curtail_kw"].to_numpy(dtype=float),
            label="Energy Curtailed", linestyle="--", linewidth=2)

    ax.set_title("Model S3: Community Power Flow", fontsize=16)
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Power (kW)")
    _style(ax)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(8))

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_grid_impact(settlement: pd.DataFrame, feeder_cap_kw: float, out_path: Path):
    p2p_kw = pd.to_numeric(settlement["settled_kw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    time_of_day = _time_of_day_labels(len(settlement))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(time_of_day, p2p_kw, label="P2P Power Transfer (Settled)", linewidth=2)

    ax.axhline(
        y=float(feeder_cap_kw),
        linestyle="--",
        linewidth=2,
        label=f"Feeder Capacity ({feeder_cap_kw:.1f} kW)",
        color="r",
    )

    ax.fill_between(
        time_of_day,
        p2p_kw,
        float(feeder_cap_kw),
        where=(p2p_kw > float(feeder_cap_kw)),
        alpha=0.3,
        color="red",
        interpolate=True,
        label="Overload",
    )

    ax.set_title("Model S3: P2P Transfer vs Feeder Capacity", fontsize=16)
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Power (kW)")
    _style(ax)
    ax.legend()
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(8))

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_average_daily(phys: pd.DataFrame, settlement: pd.DataFrame, batt: pd.DataFrame, out_path: Path):
    df = phys.merge(settlement[["interval", "settled_kw"]], on="interval", how="left").copy()
    df = df.merge(
        batt[["interval", "battery_charge_kw", "battery_discharge_kw", "curtail_kw", "utility_export_kw"]],
        on="interval",
        how="left"
    ).copy()

    df["settled_kw"] = df["settled_kw"].fillna(0.0)
    df["battery_charge_kw"] = df["battery_charge_kw"].fillna(0.0)
    df["battery_discharge_kw"] = df["battery_discharge_kw"].fillna(0.0)
    df["battery_charge_kw"] = df["battery_charge_kw"].fillna(0.0)
    df["curtail_kw"] = df["curtail_kw"].fillna(0.0)
    df["utility_export_kw"] = df["utility_export_kw"].fillna(0.0)
    df["utility_export_kw"] = df["utility_export_kw"].fillna(0.0)

    df["tod"] = df["interval"] % INTERVALS_PER_DAY
    avg = df.groupby("tod").mean(numeric_only=True)

    load_kw = avg["community_load_kw"].to_numpy(dtype=float)
    pv_kw = avg["community_pv_kw"].to_numpy(dtype=float)
    p2p_kw = avg["settled_kw"].to_numpy(dtype=float)
    batt_ch_kw = avg["battery_charge_kw"].to_numpy(dtype=float)
    batt_dis_kw = avg["battery_discharge_kw"].to_numpy(dtype=float)
    curtail_kw = avg["curtail_kw"].to_numpy(dtype=float)
    util_exp_kw = avg["utility_export_kw"].to_numpy(dtype=float)

    grid_kw = np.maximum(load_kw - pv_kw - batt_dis_kw - p2p_kw, 0.0)

    time_of_day = _time_of_day_labels(len(load_kw))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(time_of_day, load_kw, label="Community Load", color="black", linewidth=3)
    ax.plot(time_of_day, pv_kw, label="PV Generation", color="green", linewidth=2.5)
    ax.plot(time_of_day, grid_kw, label="Grid Imports", color="red", linewidth=2.5, linestyle="--")

    ax.fill_between(time_of_day, p2p_kw, label="P2P Transfer (Settled)", alpha=0.5)

    ax.plot(time_of_day, batt_ch_kw, label="Battery Charging", linewidth=2.2)
    ax.plot(time_of_day, batt_dis_kw, label="Battery Discharging", linewidth=2.2, linestyle="--")

    ax.plot(time_of_day, util_exp_kw, label="Utility Export (FiT)", linestyle="--", linewidth=2)
    ax.plot(time_of_day, curtail_kw, label="Energy Curtailed", linestyle="--", linewidth=2)

    ax.set_title("Model S3: Average Daily Power Transfer", fontsize=16)
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Average Power (kW)")
    _style(ax)
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(8))

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_interval_diagnostics(settlement: pd.DataFrame, out_path: Path):
    """
    FIXED: Shading uses contiguous spans (no vertical stripe effect).
    X-axis labelled HH:MM (every 2 hours).
    """
    df = settlement.copy()

    # Ensure flags exist
    if "has_trading" not in df.columns:
        df["has_trading"] = pd.to_numeric(df.get("settled_u", 0), errors="coerce").fillna(0.0) > 0
    if "cap_binding" not in df.columns:
        df["cap_binding"] = pd.to_numeric(df.get("allowed_u", 0), errors="coerce").fillna(0.0) < pd.to_numeric(df.get("planned_clear_u", 0), errors="coerce").fillna(0.0)

    df = df.sort_values("interval").reset_index(drop=True)

    x = df["interval"].to_numpy(dtype=int)
    planned = pd.to_numeric(df["planned_clear_u"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    allowed = pd.to_numeric(df["allowed_u"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    trading = df["has_trading"].astype(bool).to_numpy()
    cap = df["cap_binding"].astype(bool).to_numpy()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    # Contiguous shading (key fix)
    trade_runs = _runs_of_true(trading)
    cap_runs = _runs_of_true(cap)

    for (s, e) in trade_runs:
        ax.axvspan(x[s] - 0.5, x[e] + 0.5, alpha=0.12, color="grey")
    for (s, e) in cap_runs:
        ax.axvspan(x[s] - 0.5, x[e] + 0.5, alpha=0.20, color="red")

    ax.plot(x, planned, label="Planned Clear (Oracle)", linewidth=2.5)
    ax.plot(x, allowed, label="Allowed (On-Chain)", linewidth=2.5)

    ax.scatter(df.loc[trading, "interval"], df.loc[trading, "allowed_u"], label="Intervals with Trading", zorder=4)
    ax.scatter(df.loc[cap, "interval"], df.loc[cap, "allowed_u"], color="red", label="Cap-binding", zorder=5)

    from matplotlib.patches import Patch
    band_handles = [
        Patch(facecolor="grey", alpha=0.12, label="Trading Window (Shaded)"),
        Patch(facecolor="red", alpha=0.20, label="Cap-binding Window (Shaded)"),
    ]

    ax.set_title("Model S3 Diagnostic: Planned vs Allowed Energy (Trading & Cap-binding Highlighted)")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Energy Units (0.01 kWh)")
    _style(ax)

    tick_step = 8  # 2 hours
    xticks = np.arange(int(x.min()), int(x.max()) + 1, tick_step)
    tod = (xticks % INTERVALS_PER_DAY)
    tick_labels = _time_of_day_labels(INTERVALS_PER_DAY)

    ax.set_xticks(xticks)
    ax.set_xticklabels([tick_labels[i] for i in tod], rotation=45, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + band_handles, labels + [p.get_label() for p in band_handles])

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_kpi_table(kpis: Dict[str, float], fairness: Dict[str, float], out_path: Path):
    rows = [
        ("Total Load (kWh)", f"{kpis['total_load_kwh']:.2f}"),
        ("Total PV Generation (kWh)", f"{kpis['total_pv_kwh']:.2f}"),
        ("Total P2P Energy (kWh)", f"{kpis['total_p2p_kwh']:.2f}"),
        ("Grid Imports (kWh)", f"{kpis['total_grid_kwh']:.2f}"),
        ("Self-Consumption Ratio", f"{kpis['self_consumption_ratio']*100:.2f}%"),
        ("Curtailed Energy (kWh)", f"{kpis['curtailed_kwh']:.2f}"),
        ("Utility Export (kWh)", f"{kpis.get('utility_export_kwh', 0.0):.2f}"),
        ("Utility Export Revenue (ZAR)", f"{kpis.get('utility_export_revenue_zar', 0.0):.2f}"),
        ("Feeder Cap (kW)", f"{kpis['feeder_cap_kw']:.1f}"),
        ("Peak Utilisation", f"{kpis['peak_util']*100:.1f}%"),
        ("Mean MCP (ZAR/kWh)", f"{kpis['mcp_mean']:.2f}"),
        ("Weighted MCP (ZAR/kWh)", f"{kpis['mcp_weighted']:.2f}"),
        ("Execute Success Rate", f"{kpis['exec_rate']*100:.1f}%"),
        ("Finalize Success Rate", f"{kpis['finalize_rate']*100:.1f}%"),
        ("Gini (P2P Benefits)", f"{fairness['gini_p2p_benefits']:.3f}"),
        ("Gini (Utility Costs)", f"{fairness['gini_utility_costs']:.3f}"),
        ("Gini (Net Outcomes)", f"{fairness['gini_net_outcomes']:.3f}"),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    ax.set_title("Model S3: Key Metrics Summary", fontsize=14, pad=12)

    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=".", help="Folder containing S3 inputs/outputs (default: current folder)")
    ap.add_argument("--out-dir", default=None, help="Output folder (default: <in-dir>/results_s3)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (in_dir / "results_s3")
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(in_dir)
    participants = inputs["participants"]
    interval_inputs = inputs["interval_inputs"]
    planned_trades = inputs["planned_trades"]
    interval_summary = inputs["interval_summary"]
    trade_log = inputs["trade_log"]
    battery_ts = inputs["battery_ts"]

    utility_tariff = float(config.UTILITY_TARIFF)

    phys = build_physical_series_from_interval_inputs(interval_inputs)
    settlement = build_settlement_series(interval_summary, trade_log)
    batt = build_battery_series(battery_ts, num_intervals=len(phys))

    fairness, agent_df = compute_fairness(
        participants=participants,
        interval_summary=interval_summary,
        trade_log=trade_log,
        interval_inputs=interval_inputs,
        planned_trades=planned_trades,
        utility_tariff=utility_tariff,
    )

    kpis = compute_kpis(
        interval_summary=interval_summary,
        interval_inputs=interval_inputs,
        phys=phys,
        batt=batt,
        settlement=settlement,
    )

    # Save data
    agent_df.to_csv(data_dir / "s3_agent_fairness_breakdown.csv", index=False)
    pd.DataFrame([kpis | fairness]).to_csv(data_dir / "s3_kpis.csv", index=False)
    (data_dir / "s3_kpis.json").write_text(json.dumps(kpis | fairness, indent=2), encoding="utf-8")

    # Figures
    plot_community_power_flow(phys, settlement, batt, fig_dir / "S3_community_power_flow.png")
    plot_grid_impact(settlement, kpis["feeder_cap_kw"], fig_dir / "S3_grid_impact.png")
    plot_average_daily(phys, settlement, batt, fig_dir / "S3_average_daily_power_transfer.png")
    plot_interval_diagnostics(settlement, fig_dir / "S3_interval_diagnostics.png")
    plot_kpi_table(kpis, fairness, fig_dir / "S3_kpi_table.png")

    # Console summary
    print("\n" + "=" * 70)
    print("MODEL S3 – KPI SUMMARY")
    print("=" * 70)
    print(f"Total Load:              {kpis['total_load_kwh']:.2f} kWh")
    print(f"Total PV Generation:     {kpis['total_pv_kwh']:.2f} kWh")
    print(f"Total P2P Energy:        {kpis['total_p2p_kwh']:.2f} kWh")
    print(f"Grid Imports:            {kpis['total_grid_kwh']:.2f} kWh")
    print(f"Self-Consumption Ratio:  {kpis['self_consumption_ratio']*100:.2f}%")
    print(f"Curtailed Energy:        {kpis['curtailed_kwh']:.2f} kWh")
    print(f"Utility Export:          {kpis.get('utility_export_kwh', 0.0):.2f} kWh")
    print(f"Utility Export Revenue:  {kpis.get('utility_export_revenue_zar', 0.0):.2f} ZAR")
    print(f"Mean MCP:                {kpis['mcp_mean']:.2f} ZAR/kWh")
    print(f"Weighted MCP:            {kpis['mcp_weighted']:.2f} ZAR/kWh")
    print(f"Feeder Capacity:         {kpis['feeder_cap_kw']:.1f} kW")
    print(f"Peak Utilisation:        {kpis['peak_util']*100:.1f}%")
    print(f"Execute Success Rate:    {kpis['exec_rate']*100:.1f}%")
    print(f"Finalize Success Rate:   {kpis['finalize_rate']*100:.1f}%")

    print("\n" + "=" * 70)
    print("MODEL S3 – FAIRNESS METRICS")
    print("=" * 70)
    print(f"Gini (P2P Benefits):     {fairness['gini_p2p_benefits']:.3f}")
    print(f"Gini (Utility Costs):    {fairness['gini_utility_costs']:.3f}")
    print(f"Gini (Net Outcomes):     {fairness['gini_net_outcomes']:.3f}")

    print("\n✔ S3 results generated")
    print(f"✔ Output: {out_dir}")
    print(f"✔ Figures: {fig_dir}")


if __name__ == "__main__":
    main()

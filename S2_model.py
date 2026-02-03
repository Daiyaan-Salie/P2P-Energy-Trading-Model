#Model S2: Community Energy Market (CEM / Pool Model)
#
# Purpose:
# --------
# This script implements the S2 scenario, where households participate in a
# community-level peer-to-peer (P2P) energy market.
#
# Key characteristics of S2:
# - Prosumers may sell surplus energy to consumers via a pooled market
# - A uniform market-clearing price (MCP) is used
# - No feeder capacity constraints are enforced (unconstrained network)
# - Batteries are operated locally before participation in the P2P market
#
# Role in the thesis/report:
# --------------------------
# - Represents the first P2P-enabled scenario
# - Provides a comparison between:
#     * S1: utility-only baseline
#     * S2: unconstrained community market
# - Used to analyse welfare gains, fairness impacts, and grid effects
#
# Key assumptions:
# ----------------
# - Time resolution: 15-minute intervals (NUM_INTERVALS = 96 for 24 hours)
# - Energy units: kWh per interval
# - Power units (for plots): kW (converted using ×4)
# - Batteries are prioritised for self-consumption before market participation
#
# Outputs:
# --------
# - Printed summary metrics to console
# - PNG figures saved to disk (power flow, benefits, grid impact, battery SoC)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config
import Common_Functions as common


# -----------------------------
# Helper functions
# -------------------------------------------------------------------------

# Gini helper that supports negative values.
# This is required because net outcomes (P2P benefit − grid cost) may be negative for some households. The shifting preserves inequality structure
# while allowing a valid Gini calculation.

def _gini_allow_neg(values: np.ndarray) -> float:
    """
    Gini index helper that supports negative values by shifting the distribution to be non-negative.
    This matches the approach used in S3 fairness when net outcomes can be < 0.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    v = np.where(np.isfinite(v), v, 0.0)
    vmin = float(np.min(v))
    if vmin < 0:
        v = v - vmin
    if np.sum(v) == 0:
        return 0.0
    return float(common.calculate_gini_index(v))

# Builds a per-household vector of total P2P benefits.
# Consumers receive consumer surplus, prosumers receive trading profit.
# Households that do not trade retain a benefit of zero.
def _build_household_p2p_benefits(
    total_households: int,
    consumer_surplus: pd.Series,
    prosumer_profit: pd.Series,
) -> np.ndarray:
    benefits = np.zeros(int(total_households), dtype=float)

    if consumer_surplus is not None and len(consumer_surplus) > 0:
        for hid, val in consumer_surplus.items():
            idx = int(hid)
            if 0 <= idx < total_households:
                benefits[idx] += float(val)

    if prosumer_profit is not None and len(prosumer_profit) > 0:
        for hid, val in prosumer_profit.items():
            idx = int(hid)
            if 0 <= idx < total_households:
                benefits[idx] += float(val)

    return benefits

# Helper for defining plotting windows.
# Maintains backward compatibility with "weekly" plots, but safely clamps
# to the available simulation horizon (e.g., 1-day runs).
def _plot_window_bounds(week_to_plot: int, intervals_per_day: int = 96):
    """
    Returns (start, end, label) for a plotting window.
    If horizon <= 7 days, label becomes Day N and end is clipped to NUM_INTERVALS.
    """
    num_intervals = int(config.NUM_INTERVALS)
    intervals_per_week = 7 * intervals_per_day

    start = int((week_to_plot - 1) * intervals_per_week)
    end = int(start + intervals_per_week)

    # Clip to available horizon
    start = max(0, min(start, num_intervals))
    end = max(0, min(end, num_intervals))

    # Label adapts to short horizons
    if num_intervals <= intervals_per_week:
        label = f"Day {week_to_plot}"
    else:
        label = f"Week {week_to_plot}"

    # Fallback if requested window is empty
    if start == end:
        start = 0
        end = num_intervals

    return start, end, label


# -----------------------------
# S2 Simulation
# -----------------------------
# Executes the physical energy dispatch and market clearing logic for the
# community energy market.

def run_s2_simulation():
    print("--- Running S2: Community Energy Market Model ---")

    # --- 1. Setup & Data Generation ---
    np.random.seed(42)
    num_households = int(config.TOTAL_HOUSEHOLDS)
    num_prosumers = int(num_households * config.PROSUMER_SHARE)
    num_intervals = int(config.NUM_INTERVALS)

    load_profiles_kwh = common.generate_household_profiles(num_households, num_intervals)
    pv_profiles_kwh = common.generate_pv_profiles(num_prosumers, num_intervals, config.PEAK_PV_GENERATION_KW)
    
    # Align PV generation with household indexing
    generation_profiles_kwh = np.zeros_like(load_profiles_kwh)
    generation_profiles_kwh[:num_prosumers, :] = pv_profiles_kwh

    # --- 2. State Initialisation -------
    battery_soc_kwh = np.zeros((num_prosumers, num_intervals + 1))
    battery_soc_kwh[:, 0] = float(config.BATTERY_CAPACITY_KWH) * 0.5
    grid_imports_kwh = np.zeros_like(load_profiles_kwh)
    curtailment_kwh = np.zeros_like(generation_profiles_kwh)

    # Trade log records interval-level bilateral allocations
    trade_log = []

    # --- 3. Simulation Loop ---
    for t in range(num_intervals):
        # Net energy before battery and market interaction
        net_energy_kwh = generation_profiles_kwh[:, t] - load_profiles_kwh[:, t]
        # Carry battery state forward
        battery_soc_kwh[:, t + 1] = battery_soc_kwh[:, t]

        # Step 1: Discharge batteries to cover deficits (self-consumption priority)
        for i in range(num_prosumers):
            if net_energy_kwh[i] < 0:
                from_battery = min(abs(net_energy_kwh[i]), battery_soc_kwh[i, t + 1])
                battery_soc_kwh[i, t + 1] -= from_battery
                net_energy_kwh[i] += from_battery

        # Step 2: Charge batteries using surplus PV before market participation
        # This aligns S2 dispatch with S1 behaviour.
        for i in range(num_prosumers):
            if net_energy_kwh[i] > 0:
                available_capacity = float(config.BATTERY_CAPACITY_KWH) - battery_soc_kwh[i, t + 1]
                to_store = min(net_energy_kwh[i], max(0.0, available_capacity))
                battery_soc_kwh[i, t + 1] += to_store
                net_energy_kwh[i] -= to_store

        # Step 3: Form P2P market offers (sellers) and bids (buyers)
        sellers = {i: net_energy_kwh[i] for i in range(num_prosumers) if net_energy_kwh[i] > 0}
        buyers = {i: abs(net_energy_kwh[i]) for i in range(num_households) if net_energy_kwh[i] < 0}
        total_supply_kwh = float(sum(sellers.values()))
        total_demand_kwh = float(sum(buyers.values()))

        # Step 4: Market clearing (uniform MCP, pro-rata allocation)
        if total_supply_kwh > 0 and total_demand_kwh > 0:
            mcp = common.uniform_mcp(config.PRICE_FLOOR, config.UTILITY_TARIFF)
            p2p_trade_volume_kwh = min(total_supply_kwh, total_demand_kwh)

            if p2p_trade_volume_kwh > 0:
                for buyer_id, demand in buyers.items():
                    traded_amount = (demand / total_demand_kwh) * p2p_trade_volume_kwh

                    # Allocate buyer demand across all sellers proportionally
                    for seller_id, supply in sellers.items():
                        if supply > 0:
                            from_seller = (supply / total_supply_kwh) * traded_amount
                            trade_log.append(
                                {
                                    "interval": t,
                                    "prosumer_id": seller_id,
                                    "consumer_id": buyer_id,
                                    "p2p_trade_volume": from_seller,
                                    "market_price": mcp,
                                }
                            )

                    net_energy_kwh[buyer_id] += traded_amount

               # Reduce sellers' surplus by sold energy
                for seller_id in sellers.keys():
                    net_energy_kwh[seller_id] -= (sellers[seller_id] / total_supply_kwh) * p2p_trade_volume_kwh

        # Step 5: Curtail remaining surplus
        for i in range(num_prosumers):
            if net_energy_kwh[i] > 0:
                curtailment_kwh[i, t] = net_energy_kwh[i]
                net_energy_kwh[i] = 0.0

        # Step 6: Import remaining deficits from grid
        for i in range(num_households):
            if net_energy_kwh[i] < 0:
                grid_imports_kwh[i, t] = abs(net_energy_kwh[i])

    print("Simulation complete.")
    trades_df = (
        pd.DataFrame(trade_log)
        if trade_log
        else pd.DataFrame(columns=["prosumer_id", "consumer_id", "p2p_trade_volume", "market_price"])
    )

    return {
        "trades_df": trades_df,
        "imports_kwh": grid_imports_kwh,
        "curtailment_kwh": curtailment_kwh,
        "load_kwh": load_profiles_kwh,
        "generation_kwh": generation_profiles_kwh,
        "battery_soc_kwh": battery_soc_kwh,
    }


# -----------------------------
# Reporting
# -----------------------------
# Computes economic, fairness, and technical metrics aligned with S3 definitions.# Computes economic, fairness, and technical metrics aligned with S3 definitions.

def print_s2_summary_report(results):
    """
    Calculates and prints a comprehensive summary for the S2 model, including per-agent details.
    Adds 3 fairness metrics aligned with S3:
      - Gini(P2P Benefits)
      - Gini(Utility Costs)
      - Gini(Net Outcomes) = (P2P benefits - grid costs) per participant
    """
    trades_df = results["trades_df"]
    total_households = int(config.TOTAL_HOUSEHOLDS)
    num_prosumers = int(config.TOTAL_HOUSEHOLDS * config.PROSUMER_SHARE)
    num_consumers = int(config.TOTAL_HOUSEHOLDS - num_prosumers)

    # --- 1. Thesis Metrics (Economic + Technical) ---
    welfare_results = common.calculate_economic_welfare(trades_df, config.UTILITY_TARIFF)
    technical_results = common.calculate_technical_robustness(trades_df, config.FEEDER_CAPACITY_KW)

    # --- 2. Aggregate Metrics ---
    total_grid_import_kwh = float(np.sum(results["imports_kwh"]))
    total_utility_cost_zar = total_grid_import_kwh * float(config.UTILITY_TARIFF)

    # --- 3. Per-Agent Metrics ---
    avg_revenue_per_prosumer = (
        float(welfare_results["prosumer_profit_per_agent"].sum()) / num_prosumers if num_prosumers > 0 else 0.0
    )
    prosumer_grid_imports = float(np.sum(results["imports_kwh"][:num_prosumers, :]))
    avg_grid_cost_per_prosumer = (
        (prosumer_grid_imports / num_prosumers) * float(config.UTILITY_TARIFF) if num_prosumers > 0 else 0.0
    )
    avg_net_outcome_prosumer = avg_revenue_per_prosumer - avg_grid_cost_per_prosumer

    avg_savings_per_consumer = (
        float(welfare_results["consumer_surplus_per_agent"].sum()) / num_consumers if num_consumers > 0 else 0.0
    )
    consumer_grid_imports = float(np.sum(results["imports_kwh"][num_prosumers:, :]))
    avg_grid_cost_per_consumer = (
        (consumer_grid_imports / num_consumers) * float(config.UTILITY_TARIFF) if num_consumers > 0 else 0.0
    )

    # --- 4. Fairness Metrics (aligned to S3 definitions) ---
    p2p_benefit_per_hh = _build_household_p2p_benefits(
        total_households=total_households,
        consumer_surplus=welfare_results["consumer_surplus_per_agent"],
        prosumer_profit=welfare_results["prosumer_profit_per_agent"],
    )
    gini_p2p_benefits = (
        float(common.calculate_gini_index(p2p_benefit_per_hh)) if np.sum(p2p_benefit_per_hh) > 0 else 0.0
    )

    utility_costs_per_hh = np.sum(results["imports_kwh"], axis=1) * float(config.UTILITY_TARIFF)
    gini_utility_costs = (
        float(common.calculate_gini_index(utility_costs_per_hh)) if np.sum(utility_costs_per_hh) > 0 else 0.0
    )

    net_outcomes_per_hh = p2p_benefit_per_hh - utility_costs_per_hh
    gini_net_outcomes = _gini_allow_neg(net_outcomes_per_hh)

    # --- 5. Print Formatted Report ---
    print("\n" + "=" * 50)
    print("--- S2 Community Market: Key Metrics Summary ---")
    print("=" * 50)

    print("\n## Aggregate Metrics (Entire Community) ##")
    print(f"Total Utility Cost (ZAR):   ZAR {total_utility_cost_zar:,.2f}")
    print(f"Total P2P Volume Traded:    {trades_df['p2p_trade_volume'].sum():,.2f} kWh")
    print(f"Total Grid Imports:         {total_grid_import_kwh:,.2f} kWh")
    print(f"Total Energy Curtailed:     {np.sum(results['curtailment_kwh']):,.2f} kWh")

    print("\n" + "-" * 50)

    print("\n## Per-Agent Average Metrics (S2) ##")
    print("\n### Prosumer (per household) ###")
    print(f"Average P2P Revenue:        ZAR {avg_revenue_per_prosumer:,.2f}")
    print(f"Average Grid Cost:          ZAR {avg_grid_cost_per_prosumer:,.2f}")
    print(f"Average Net Outcome:        ZAR {avg_net_outcome_prosumer:,.2f} (Profit)")

    print("\n### Consumer (per household) ###")
    print(f"Average Grid Cost:          ZAR {avg_grid_cost_per_consumer:,.2f}")
    print(f"Average P2P Savings:        ZAR {avg_savings_per_consumer:,.2f}")

    print("\n" + "-" * 50)

    print("\n## Thesis Comparison Metrics (S2) ##")
    print(f"Total System Benefit (P2P): ZAR {welfare_results['total_system_benefit']:,.2f}")
    print(f"Gini (P2P Benefits):        {gini_p2p_benefits:.3f}")
    print(f"Gini (Utility Costs):       {gini_utility_costs:.3f}")
    print(f"Gini (Net Outcomes):        {gini_net_outcomes:.3f}")
    print(f"Feeder Overload Frequency:  {technical_results['feeder_overload_frequency_percent']:.2f}%")

    print("\n" + "=" * 50)


# -----------------------------
# Visualizations
# -----------------------------

def create_visualizations(results):
    """
    Calls all the individual plotting functions to generate S2 graphs.
    """

    # This is a simple orchestration function to generate all S2 figures in a single call.
    # - The order below mirrors a typical “story” for the results:
    #   (1) community power composition, (2) benefit split, (3) grid impact, (4) average daily profile, (5) battery SoC.
    print("\n--- Generating Visualizations for S2 Model ---")
    plot_s2_community_power_flow(results)
    plot_benefit_distribution(results)
    plot_grid_impact(results)
    plot_s2_average_daily_power_transfer(results)
    plot_average_battery_soc(results)

    print("\nAll visualizations have been saved to PNG files.")
    plt.close("all")



def plot_s2_community_power_flow(results, week_to_plot=1):
    """
    Generates the S2 power flow chart with a P2P layer.
    Works for 24h (96 intervals) and longer horizons (clips window safely).
    """
    intervals_per_day = 96
    start, end, label = _plot_window_bounds(week_to_plot, intervals_per_day)

    agg_load_kw = np.sum(results["load_kwh"], axis=0)[start:end] * 4
    agg_gen_kw = np.sum(results["generation_kwh"], axis=0)[start:end] * 4
    agg_imports_kw = np.sum(results["imports_kwh"], axis=0)[start:end] * 4

    p2p_flow_kwh = results["trades_df"].groupby("interval")["p2p_trade_volume"].sum()
    p2p_flow_kw = p2p_flow_kwh.reindex(range(int(config.NUM_INTERVALS)), fill_value=0).values[start:end] * 4

    self_consumed_kw = np.minimum(agg_load_kw, agg_gen_kw)

    # Match Average Daily Power Transfer x-axis
    time_of_day = pd.to_datetime(
        pd.date_range(start="00:00", periods=len(agg_load_kw), freq="15min")
    ).strftime("%H:%M")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.stackplot(
        time_of_day,
        self_consumed_kw,
        p2p_flow_kw,
        agg_imports_kw,
        labels=["Self-Consumption", "P2P Import", "Grid Import"],
        colors=["#2ca02c", "#9467bd", "#ff7f0e"],
        alpha=0.7,
    )

    ax.plot(time_of_day, agg_load_kw, label="Total Community Load", color="black", linewidth=2)

    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time of Day")
    ax.set_title(f"S2 Community Power Flow", fontsize=16)
    ax.legend(loc="upper left")
    ax.grid(True)

    # Same tick spacing as your Avg Daily Power Transfer plot
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))  # every 2 hours

    plt.tight_layout()
    plt.savefig("S2_community_power_flow.png")
    print("Saved 'S2_community_power_flow.png'")


def plot_average_battery_soc(results, week_to_plot=1):
    """
    Generates a line plot of the average prosumer battery state of charge for S2
    X-axis matches Average Daily Power Transfer (Time of Day: HH:MM).
    """
    intervals_per_day = 96
    start, end, label = _plot_window_bounds(week_to_plot, intervals_per_day)

    # Average SoC across prosumers (length = NUM_INTERVALS + 1)
    avg_soc_kwh_full = np.mean(results["battery_soc_kwh"], axis=0)
    avg_soc_pct_full = (avg_soc_kwh_full / float(config.BATTERY_CAPACITY_KWH)) * 100.0

    # Use interval-start SoC only (length = NUM_INTERVALS), prevents 24:00 -> 00:00 wrap
    avg_soc_pct = avg_soc_pct_full[:-1]

    # Clamp window for SoC series (length = NUM_INTERVALS)
    soc_len = len(avg_soc_pct)
    start = max(0, min(start, soc_len))
    end = max(start, min(end, soc_len))

    time_of_day = pd.to_datetime(
        pd.date_range(start="00:00", periods=soc_len, freq="15min")
    ).strftime("%H:%M")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        time_of_day[start:end],
        avg_soc_pct[start:end],
        label="Average Battery SoC",
        color="#007acc",
        linewidth=2.5,
    )

    ax.set_ylabel("State of Charge (%)")
    ax.set_xlabel("Time of Day")
    ax.set_title(f"S2: Average Prosumer Battery SoC ({label})", fontsize=16)
    ax.set_ylim(0, 105)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))  # every 2 hours

    plt.tight_layout()
    plt.savefig("S2_average_battery_soc.png")
    print("Saved 'S2_average_battery_soc.png'")


def plot_benefit_distribution(results):
    """
    Generates a grouped bar chart showing the split of P2P benefits.
    """
    trades_df = results["trades_df"]

    welfare_results = common.calculate_economic_welfare(trades_df, config.UTILITY_TARIFF)
    consumer_surplus = float(welfare_results["consumer_surplus_per_agent"].sum())
    prosumer_profit = float(welfare_results["prosumer_profit_per_agent"].sum())
    total_system_benefit = float(welfare_results["total_system_benefit"])

    fig, ax = plt.subplots(figsize=(10, 7))

    bar_width = 0.35
    index = np.arange(1)

    ax.bar(index - bar_width / 2, [consumer_surplus], bar_width, label="Total Consumer Surplus", color="#1f77b4")
    ax.bar(index + bar_width / 2, [prosumer_profit], bar_width, label="Total Prosumer Profit", color="#2ca02c")

    ax.text(index - bar_width / 2, consumer_surplus, f"ZAR {consumer_surplus:,.2f}", ha="center", va="bottom", fontsize=12)
    ax.text(index + bar_width / 2, prosumer_profit, f"ZAR {prosumer_profit:,.2f}", ha="center", va="bottom", fontsize=12)

    ax.set_ylabel("Financial Benefit (ZAR)")
    ax.set_title("S2: Distribution of Economic Benefits", fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(["P2P Market Benefits"])
    ax.legend()

    fig.text(
        0.5,
        0.95,
        f"Total System Benefit: ZAR {total_system_benefit:,.2f}",
        ha="center",
        va="top",
        fontsize=14,
        weight="bold",
    )

    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("ZAR {x:,.0f}"))

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig("S2_benefit_distribution.png")
    print("Saved 'S2_benefit_distribution.png'")


def plot_grid_impact(results, week_to_plot=1):
    """
    Generates the grid congestion plot showing P2P flow vs. feeder capacity.
    Works for 24h (96 intervals) and longer horizons.
    """
    p2p_flow_kwh = results["trades_df"].groupby("interval")["p2p_trade_volume"].sum()
    p2p_flow_kw = p2p_flow_kwh.reindex(range(int(config.NUM_INTERVALS)), fill_value=0).values * 4

    intervals_per_day = 96
    start, end, label = _plot_window_bounds(week_to_plot, intervals_per_day)

    window = p2p_flow_kw[start:end]

    # Match Average Daily Power Transfer x-axis
    time_of_day = pd.to_datetime(
        pd.date_range(start="00:00", periods=len(window), freq="15min")
    ).strftime("%H:%M")

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(time_of_day, window, label="P2P Power Transfer", color="#1f77b4")

    # Feeder capacity limit
    ax.axhline(
        y=float(config.FEEDER_CAPACITY_KW),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Feeder Capacity ({config.FEEDER_CAPACITY_KW} kW)",
    )

    # Shade overload region
    ax.fill_between(
        time_of_day,
        window,
        float(config.FEEDER_CAPACITY_KW),
        where=(window > float(config.FEEDER_CAPACITY_KW)),
        color="red",
        alpha=0.3,
        interpolate=True,
        label="Overload",
    )

    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time of Day")
    ax.set_title(f"S2 Grid Impact: P2P Transfers vs. Feeder Capacity", fontsize=16)
    ax.legend()
    ax.grid(True)

    # Same tick spacing as your Avg Daily Power Transfer plot (every 2 hours)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))

    plt.tight_layout()
    plt.savefig("S2_grid_impact.png")
    print("Saved 'S2_grid_impact.png'")



def plot_s2_average_daily_power_transfer(results):
    """
    Generates a line plot showing the average 24-hour power flows for the S2 model.
    """
    intervals_per_day = 96
    num_intervals = int(config.NUM_INTERVALS)

    # --- Community aggregates (kWh per interval) ---
    total_load_kwh = np.sum(results["load_kwh"], axis=0)                  # shape: (T,)
    total_gen_kwh = np.sum(results["generation_kwh"], axis=0)             # shape: (T,)
    total_imports_kwh = np.sum(results["imports_kwh"], axis=0)            # shape: (T,)

    # --- P2P flow (kWh per interval) ---
    trades_df = results["trades_df"]
    if trades_df is not None and not trades_df.empty:
        p2p_flow_kwh = trades_df.groupby("interval")["p2p_trade_volume"].sum()
        p2p_flow_kwh = p2p_flow_kwh.reindex(range(num_intervals), fill_value=0).values
    else:
        p2p_flow_kwh = np.zeros(num_intervals, dtype=float)

    # --- Battery charging/discharging from SoC deltas (kWh per interval) ---
    # battery_soc_kwh shape: (num_prosumers, T+1)
    soc = results["battery_soc_kwh"]
    avg_soc = np.mean(soc, axis=0)  # shape: (T+1,)

    # delta_soc[t] = soc[t+1] - soc[t] in kWh for that 15-min interval
    delta_soc = np.diff(avg_soc)  # shape: (T,)

    batt_charge_kwh = np.maximum(delta_soc, 0.0)    # charging energy into battery
    batt_discharge_kwh = np.maximum(-delta_soc, 0.0)  # energy delivered from battery

    # --- Convert to kW (kWh per 15-min * 4) ---
    total_load_kw = total_load_kwh * 4
    total_gen_kw = total_gen_kwh * 4
    total_imports_kw = total_imports_kwh * 4
    p2p_kw = p2p_flow_kwh * 4
    batt_charge_kw = batt_charge_kwh * 4
    batt_discharge_kw = batt_discharge_kwh * 4

    # --- If multi-day, average by time-of-day; if 1-day (96), this is unchanged ---
    full_len = (num_intervals // intervals_per_day) * intervals_per_day
    if full_len == 0:
        return

    num_days = max(1, full_len // intervals_per_day)

    def _avg_daily(x):
        return np.mean(x[:full_len].reshape(num_days, intervals_per_day), axis=0)

    avg_load_kw = _avg_daily(total_load_kw)
    avg_gen_kw = _avg_daily(total_gen_kw)
    avg_imports_kw = _avg_daily(total_imports_kw)
    avg_p2p_kw = _avg_daily(p2p_kw)
    avg_batt_charge_kw = _avg_daily(batt_charge_kw)
    avg_batt_discharge_kw = _avg_daily(batt_discharge_kw)

    time_of_day = pd.to_datetime(
        pd.date_range(start="00:00", periods=intervals_per_day, freq="15min")
    ).strftime("%H:%M")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(time_of_day, avg_load_kw, label="Community Load", color="black", linewidth=3)
    ax.plot(time_of_day, avg_gen_kw, label="PV Generation", color="green", linewidth=2.5)
    ax.plot(time_of_day, avg_imports_kw, label="Grid Imports", color="red", linewidth=2.5, linestyle="--")

    # P2P as shaded area (existing style)
    ax.fill_between(time_of_day, avg_p2p_kw, label="P2P Transfer", color="purple", alpha=0.5)

    # Battery metrics (like S1)
    ax.plot(time_of_day, avg_batt_charge_kw, label="Battery Charging", linewidth=2.5)
    ax.plot(time_of_day, avg_batt_discharge_kw, label="Battery Discharging", linewidth=2.5, linestyle="--")

    ax.set_ylabel("Average Power (kW)")
    ax.set_xlabel("Time of Day")
    ax.set_title("S2: Average Daily Power Transfer (with Battery Charging/Discharging)", fontsize=16)
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))

    plt.tight_layout()
    plt.savefig("S2_average_daily_power_transfer.png")
    print("Saved 'S2_average_daily_power_transfer.png'")



# -----------------------------
# Entry point
# -----------------------------
# Running this script executes the full S2 workflow:
# (1) simulation, (2) console summary, (3) figure generation.
# Figures are saved to disk as PNG files for inclusion in the report.
if __name__ == "__main__":
    s2_results = run_s2_simulation()
    print_s2_summary_report(s2_results)
    create_visualizations(s2_results)

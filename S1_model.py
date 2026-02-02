#Model S1: Utility-Only Baseline Model
#
# Purpose:
# --------
# This script implements the S1 baseline scenario, where:
# - No peer-to-peer (P2P) energy trading is allowed
# - All households interact only with the utility grid
# - Prosumers may self-consume PV generation and charge/discharge batteries
# - Excess generation is curtailed
#
# Role in the thesis/report:
# --------------------------
# - Serves as the reference (counterfactual) scenario
# - Provides baseline metrics for cost, fairness, and technical performance
# - Used for comparison against S2 (unconstrained P2P) and S3 (constrained P2P)
#
# Key assumptions:
# ----------------
# - Time resolution: 15-minute intervals (NUM_INTERVALS = 96 for 24 hours)
# - Energy units: kWh per interval
# - Power units (for plots): kW (converted using ×4)
# - Batteries are initialised at 50% state of charge
#
# Outputs:
# --------
# - Printed summary metrics to console
# - PNG figures saved to disk:
#     * S1_aggregate_summary.png
#     * S1_community_power_flow.png
#     * S1_average_battery_soc.png
#     * S1_average_daily_power_transfer.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config
import Common_Functions as common


# -------------------------------------------------------------------------
# Helper: Gini calculation allowing negative values
# -------------------------------------------------------------------------
# In S1, net outcomes (benefits − utility costs) are negative for all agents.
# Since the Gini coefficient is defined for non-negative distributions, this helper shifts the distribution if required while preserving inequality
# structure. This ensures fairness metrics are comparable with S3.


def gini_shift_allow_negative(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    if np.allclose(x, 0.0):
        return 0.0
    mn = float(np.min(x))
    if mn < 0:
        x = x - mn
    if np.allclose(x, 0.0):
        return 0.0
    return float(common.calculate_gini_index(x))



# -------------------------------------------------------------------------
# Core S1 Simulation
# -------------------------------------------------------------------------
# This function executes the baseline physical energy flow logic:
# - Load and PV generation profiles are generated
# - Batteries are operated locally by prosumers
# - Consumers import all energy from the grid
# - No trading or market mechanisms are present

def run_s1_simulation():
    """
    Executes the S1 baseline simulation and returns the detailed results.
    """
    print("--- Running S1: Utility-Only Baseline Model ---")

    # --- Setup & Data Generation ---
    # Fix seed for reproducibility across runs and scenarios
    np.random.seed(42)
    num_households = config.TOTAL_HOUSEHOLDS
    num_prosumers = int(num_households * config.PROSUMER_SHARE)
    num_intervals = config.NUM_INTERVALS

    # Generate household load profiles (kWh per interval)
    load_profiles_kwh = common.generate_household_profiles(num_households, num_intervals)

    # Generate PV generation profiles for prosumers only
    pv_profiles_kwh = common.generate_pv_profiles(num_prosumers, num_intervals, config.PEAK_PV_GENERATION_KW)
    
     # Align generation matrix with household indexing
    generation_profiles_kwh = np.zeros_like(load_profiles_kwh)
    generation_profiles_kwh[:num_prosumers, :] = pv_profiles_kwh

    # --- State Variables -----------
    # Battery state of charge (extra column to store t=0 initial state)
    battery_soc_kwh = np.zeros((num_prosumers, num_intervals + 1))
    battery_soc_kwh[:, 0] = config.BATTERY_CAPACITY_KWH * 0.5

    # Energy imported from the grid (kWh per interval)
    grid_imports_kwh = np.zeros_like(load_profiles_kwh)

    # Excess PV generation that cannot be stored (kWh per interval)
    curtailment_kwh = np.zeros_like(generation_profiles_kwh)

    # --- Simulation Loop --------------------
    # Interval-by-interval energy balancing for each prosumer
    for t in range(num_intervals):
        for i in range(num_prosumers):
            # Net energy balance before battery interaction
            net_energy = generation_profiles_kwh[i, t] - load_profiles_kwh[i, t]
            # Default: carry battery SoC forward
            battery_soc_kwh[i, t + 1] = battery_soc_kwh[i, t]

            if net_energy > 0:
                # Surplus generation: attempt to charge battery
                to_store = min(net_energy, config.BATTERY_CAPACITY_KWH - battery_soc_kwh[i, t + 1])
                battery_soc_kwh[i, t + 1] += to_store
                # Any remaining surplus is curtailed
                curtailment_kwh[i, t] = net_energy - to_store
            else:
                # Deficit: discharge battery first, then import remainder from grid
                from_battery = min(abs(net_energy), battery_soc_kwh[i, t + 1])
                battery_soc_kwh[i, t + 1] -= from_battery
                grid_imports_kwh[i, t] = abs(net_energy) - from_battery

         # Consumers (non-prosumers) import entire load from the grid
        grid_imports_kwh[num_prosumers:, t] = load_profiles_kwh[num_prosumers:, t]

    print("Simulation complete.")
    # Return all time-series needed for metrics and plotting
    return {
        "load_kwh": load_profiles_kwh,
        "generation_kwh": generation_profiles_kwh,
        "imports_kwh": grid_imports_kwh,
        "curtailment_kwh": curtailment_kwh,
        "battery_soc_kwh": battery_soc_kwh,
    }

# ----------Summary Reporting-----------------
# Computes aggregate, per-agent, and fairness metrics and prints them in a structured, thesis-aligned format.
def print_summary_report(results):
    """
    Calculates and prints a comprehensive summary of the S1 model metrics,
    including aggregate, per-agent, and thesis comparison benchmarks.
    """
    # --- 1. Aggregate Energy Metrics ---
    total_load_kwh = float(np.sum(results["load_kwh"]))
    total_generation_kwh = float(np.sum(results["generation_kwh"]))
    total_grid_import_kwh = float(np.sum(results["imports_kwh"]))
    total_curtailment_kwh = float(np.sum(results["curtailment_kwh"]))

    # --- 2. Aggregate Derived Metrics ---
    community_self_sufficiency_pct = ((total_load_kwh - total_grid_import_kwh) / total_load_kwh) * 100 if total_load_kwh > 0 else 0.0

    # --- 3. Aggregate Economic Outcome ---
    total_utility_cost_zar = total_grid_import_kwh * float(config.UTILITY_TARIFF)

    # --- 4. Per-Agent Metrics Calculation ---
    num_prosumers = int(config.TOTAL_HOUSEHOLDS * config.PROSUMER_SHARE)
    num_consumers = int(config.TOTAL_HOUSEHOLDS - num_prosumers)

    # Prosumer averages
    if num_prosumers > 0:
        total_prosumer_imports_kwh = float(np.sum(results["imports_kwh"][:num_prosumers, :]))
        total_prosumer_generation_kwh = float(np.sum(results["generation_kwh"][:num_prosumers, :]))
        avg_import_per_prosumer = total_prosumer_imports_kwh / num_prosumers
        avg_gen_per_prosumer = total_prosumer_generation_kwh / num_prosumers
        avg_cost_per_prosumer = avg_import_per_prosumer * float(config.UTILITY_TARIFF)
    else:
        avg_import_per_prosumer = avg_gen_per_prosumer = avg_cost_per_prosumer = 0.0

    # Consumer averages
    if num_consumers > 0:
        total_consumer_imports_kwh = float(np.sum(results["imports_kwh"][num_prosumers:, :]))
        avg_import_per_consumer = total_consumer_imports_kwh / num_consumers
        avg_cost_per_consumer = avg_import_per_consumer * float(config.UTILITY_TARIFF)
    else:
        avg_import_per_consumer = avg_cost_per_consumer = 0.0

    # --- Fairness Metrics---
    utility_costs_per_household = np.sum(results["imports_kwh"], axis=1) * float(config.UTILITY_TARIFF)

    # P2P benefits are zero by definition in S1
    # In S1 there are no P2P benefits, all households only incur utility costs
    p2p_benefits_per_household = np.zeros(int(config.TOTAL_HOUSEHOLDS), dtype=float)
    net_outcomes_per_household = p2p_benefits_per_household - utility_costs_per_household

    gini_p2p_benefits = 0.0
    gini_utility_costs = float(common.calculate_gini_index(utility_costs_per_household)) if np.sum(utility_costs_per_household) > 0 else 0.0
    gini_net_outcomes = gini_shift_allow_negative(net_outcomes_per_household)

    # --- 5. Print Report ---
    print("\n" + "=" * 50)
    print("--- S1 Baseline Model: Key Metrics Summary ---")
    print("=" * 50)

    print("\n## Aggregate Metrics (Entire Community) ##")
    print(f"Total Utility Cost (ZAR):   ZAR {total_utility_cost_zar:,.2f}")
    print(f"Community Self-Sufficiency: {community_self_sufficiency_pct:.2f}%")
    print(f"Total Grid Imports:         {total_grid_import_kwh:,.2f} kWh")
    print(f"Total Energy Curtailed:     {total_curtailment_kwh:,.2f} kWh")

    print("\n" + "-" * 50)

    print("\n## Per-Agent Average Metrics (S1) ##")
    print("\n### Prosumer (per household) ###")
    print(f"Average PV Generation:      {avg_gen_per_prosumer:,.2f} kWh")
    print(f"Average Utility Imports:    {avg_import_per_prosumer:,.2f} kWh")
    print(f"Average Utility Cost:       ZAR {avg_cost_per_prosumer:,.2f}")

    print("\n### Consumer (per household) ###")
    print(f"Average PV Generation:      {0.00:.2f} kWh")
    print(f"Average Utility Imports:    {avg_import_per_consumer:,.2f} kWh")
    print(f"Average Utility Cost:       ZAR {avg_cost_per_consumer:,.2f}")

    print("\n" + "-" * 50)

    print("\n## Thesis Comparison Metrics (S1) ##")
    print(f"Total System Benefit (P2P): ZAR {0.00:.2f}")
    print(f"Gini (P2P Benefits):        {gini_p2p_benefits:.3f} (No P2P market)")
    print(f"Gini (Utility Costs):       {gini_utility_costs:.3f}")
    print(f"Gini (Net Outcomes):        {gini_net_outcomes:.3f} (P2P benefits − grid costs)")
    print(f"Feeder Overload Frequency:  {0.00:.2f}% (No P2P trades to cause overload)")

    print("\n" + "=" * 50)

#-------------Visualisation Functions--------------
# All plotting functions save figures to disk for inclusion in the report.
def plot_aggregate_summary_bars(results):
    """
    Generates a bar chart of the total energy flows for the entire simulation.
    """
    total_load = float(np.sum(results["load_kwh"]))
    total_gen = float(np.sum(results["generation_kwh"]))
    total_imports = float(np.sum(results["imports_kwh"]))
    total_curtailment = float(np.sum(results["curtailment_kwh"]))

    labels = ["Total Load", "Local Generation", "Grid Imports", "Energy Curtailed"]
    values = [total_load, total_gen, total_imports, total_curtailment]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("S1 Baseline: Total Energy Flows Over Simulation Period", fontsize=16)

    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.bar_label(bars, fmt="{:,.0f} kWh", padding=3)

    plt.tight_layout()
    plt.savefig("S1_aggregate_summary.png")
    print("Saved 'S1_aggregate_summary.png'")


def _plot_window_bounds(week_to_plot: int, intervals_per_day: int) -> tuple[int, int, str]:
    """
    Returns (start, end, label) for a plotting window.
    If horizon < 7 days, label becomes Day N and end is clipped to NUM_INTERVALS.
    """
    num_intervals = int(config.NUM_INTERVALS)
    intervals_per_week = 7 * intervals_per_day

    start = int((week_to_plot - 1) * intervals_per_week)
    end = int(start + intervals_per_week)

    # Clip to available horizon
    start = max(0, min(start, num_intervals))
    end = max(0, min(end, num_intervals))

    # Label: if horizon < 7 days, treat the window as "Day"
    if num_intervals <= intervals_per_week:
        label = f"Day {week_to_plot}"
    else:
        label = f"Week {week_to_plot}"

    # If start==end (e.g., user asks for week 2 but only 1 day exists), fall back to full horizon
    if start == end:
        start = 0
        end = num_intervals

    return start, end, label



def plot_community_power_flow(results, week_to_plot=1):
    """
    Generates a stacked area plot of community power flows.
    UPDATED: X-axis now matches Average Daily Power Transfer (Time of Day: HH:MM).
    Works cleanly for 1-day runs (NUM_INTERVALS=96), but is also robust if longer.
    """
    intervals_per_day = 96

    # Original "week" slicing kept for backward compatibility, but clamped safely
    start_interval = (week_to_plot - 1) * 7 * intervals_per_day
    end_interval = start_interval + 7 * intervals_per_day

    # Clamp to available data
    num_intervals = results["load_kwh"].shape[1]
    start_interval = max(0, min(start_interval, num_intervals))
    end_interval = max(start_interval, min(end_interval, num_intervals))

    # If a 1-day run (96 intervals), this will naturally plot the full day
    agg_load_kw = np.sum(results["load_kwh"], axis=0)[start_interval:end_interval] * 4
    agg_imports_kw = np.sum(results["imports_kwh"], axis=0)[start_interval:end_interval] * 4
    agg_gen_kw = np.sum(results["generation_kwh"], axis=0)[start_interval:end_interval] * 4
    agg_curtailment_kw = np.sum(results["curtailment_kwh"], axis=0)[start_interval:end_interval] * 4

    self_consumed_kw = np.minimum(agg_load_kw, agg_gen_kw)

    # --- NEW X-AXIS: Time of Day labels like Average Daily Power Transfer ---
    time_of_day = pd.to_datetime(
        pd.date_range(start="00:00", periods=len(agg_load_kw), freq="15min")
    ).strftime("%H:%M")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.stackplot(
        time_of_day,
        self_consumed_kw,
        agg_imports_kw,
        labels=["Self-Consumption", "Grid Import"],
        colors=["#2ca02c", "#ff7f0e"],
        alpha=0.7,
    )

    ax.plot(time_of_day, agg_load_kw, label="Total Community Load", color="black", linewidth=2)
    ax.plot(time_of_day, agg_curtailment_kw, label="Energy Curtailed", color="#d62728", linestyle="--")

    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time of Day")
    ax.set_title("S1 Baseline: Community Power Flow", fontsize=16)
    ax.legend(loc="upper left")
    ax.grid(True)

    # Match your Average Daily Power Transfer tick density
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))  # every 2 hours

    plt.tight_layout()
    plt.savefig("S1_community_power_flow.png")
    print("Saved 'S1_community_power_flow.png'")

def plot_average_battery_soc(results, week_to_plot=1):
    """
    Generates a line plot of the average prosumer battery state of charge.
    FIXED: Uses 96 points (not 97) so the x-axis doesn't wrap 24:00 -> 00:00.
    X-axis matches Average Daily Power Transfer (Time of Day: HH:MM).
    """
    intervals_per_day = 96

    # Average SoC across prosumers (length = NUM_INTERVALS + 1)
    avg_soc_kwh_full = np.mean(results["battery_soc_kwh"], axis=0)
    avg_soc_pct_full = (avg_soc_kwh_full / config.BATTERY_CAPACITY_KWH) * 100.0

    # ✅ Use 96 points only (drop the final "24:00" point that formats to "00:00")
    avg_soc_pct = avg_soc_pct_full[:-1]  # length = NUM_INTERVALS (96 for 1-day)

    # Keep backward compatibility with week_to_plot, but clamp safely
    start_interval = (week_to_plot - 1) * 7 * intervals_per_day
    end_interval = start_interval + 7 * intervals_per_day

    soc_len = len(avg_soc_pct)  # now = NUM_INTERVALS
    start_interval = max(0, min(start_interval, soc_len))
    end_interval = max(start_interval, min(end_interval, soc_len))

    # --- X-AXIS: Time of Day labels (96 points) ---
    time_of_day = pd.to_datetime(
        pd.date_range(start="00:00", periods=soc_len, freq="15min")
    ).strftime("%H:%M")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        time_of_day[start_interval:end_interval],
        avg_soc_pct[start_interval:end_interval],
        label="Average Battery SoC",
        color="#007acc",
        linewidth=2.5,
    )

    ax.set_ylabel("State of Charge (%)")
    ax.set_xlabel("Time of Day")
    ax.set_title("S1 Baseline: Average Prosumer Battery SoC", fontsize=16)
    ax.set_ylim(0, 105)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    # Match Average Daily Power Transfer tick density (every 2 hours)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))

    plt.tight_layout()
    plt.savefig("S1_average_battery_soc.png")
    print("Saved 'S1_average_battery_soc.png'")



def plot_average_daily_power_transfer(results):
    """
    Generates a line plot showing the average 24-hour power flows.
    For a 24-hour run, this becomes the actual day profile (no averaging artifacts).
    """
    num_intervals = int(config.NUM_INTERVALS)
    intervals_per_day = 96

    # For horizons < 1 day, just plot what exists
    if num_intervals <= 0:
        return

    # For exactly 1 day, num_days = 1 and reshape works
    num_days = max(1, int(num_intervals / intervals_per_day))

    total_load_kwh = np.sum(results["load_kwh"], axis=0)
    total_gen_kwh = np.sum(results["generation_kwh"], axis=0)
    total_imports_kwh = np.sum(results["imports_kwh"], axis=0)

    total_battery_soc_kwh = np.sum(results["battery_soc_kwh"], axis=0)
    battery_delta_kwh = np.diff(total_battery_soc_kwh)

    battery_charge_kwh = np.maximum(0, battery_delta_kwh)
    battery_discharge_kwh = np.maximum(0, -battery_delta_kwh)

    # Clip arrays to full days if needed (safe for any horizon)
    full_len = num_days * intervals_per_day
    total_load_kwh = total_load_kwh[:full_len]
    total_gen_kwh = total_gen_kwh[:full_len]
    total_imports_kwh = total_imports_kwh[:full_len]
    battery_charge_kwh = battery_charge_kwh[:full_len]
    battery_discharge_kwh = battery_discharge_kwh[:full_len]

    avg_load_kw = np.mean(total_load_kwh.reshape(num_days, intervals_per_day), axis=0) * 4
    avg_gen_kw = np.mean(total_gen_kwh.reshape(num_days, intervals_per_day), axis=0) * 4
    avg_imports_kw = np.mean(total_imports_kwh.reshape(num_days, intervals_per_day), axis=0) * 4
    avg_charge_kw = np.mean(battery_charge_kwh.reshape(num_days, intervals_per_day), axis=0) * 4
    avg_discharge_kw = np.mean(battery_discharge_kwh.reshape(num_days, intervals_per_day), axis=0) * 4

    time_of_day = pd.to_datetime(pd.date_range(start="00:00", periods=96, freq="15min")).strftime("%H:%M")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(time_of_day, avg_load_kw, label="Community Load", color="black", linewidth=3)
    ax.plot(time_of_day, avg_gen_kw, label="PV Generation", color="green", linewidth=2.5)
    ax.plot(time_of_day, avg_imports_kw, label="Grid Imports", color="red", linewidth=2.5, linestyle="--")

    ax.fill_between(time_of_day, avg_discharge_kw, label="Battery Discharging", color="orange", alpha=0.5)
    ax.fill_between(time_of_day, avg_charge_kw, label="Battery Charging", color="deepskyblue", alpha=0.5)

    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time of Day")
    ax.set_title("S1 Baseline: Average Daily Power Transfer", fontsize=16)
    ax.legend()
    ax.grid(True)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))

    plt.tight_layout()
    plt.savefig("S1_average_daily_power_transfer.png")
    print("Saved 'S1_average_daily_power_transfer.png'")


if __name__ == "__main__":
    s1_results = run_s1_simulation()
    print_summary_report(s1_results)

    print("\n--- Generating Visualizations ---")
    plot_aggregate_summary_bars(s1_results)
    plot_community_power_flow(s1_results)
    plot_average_battery_soc(s1_results)
    plot_average_daily_power_transfer(s1_results)

    print("\nAll visualizations have been saved to PNG files.")


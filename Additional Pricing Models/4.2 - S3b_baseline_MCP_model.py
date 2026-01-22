# S3b_model.py (UPDATED)
#
# This is now the "Noisy Baseline" model.
# It uses the new S3b_Auction_Engine to run a double-sided auction
# where all agents use the "Noisy" (random) bidding strategy.
#
# CORRECTION: Added int() casts in section 3.6 to fix IndexError.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config
import Common_Functions as common
import hashlib, json, time
from pathlib import Path

# --- NEW: Import the auction engine and strategies ---
import S3b_Auction_Engine as auction_engine


def run_s3b_simulation():
    """
    Executes the S3b "Noisy Baseline" simulation.
    """
    print("--- Running S3b: Noisy Baseline Model (Double-Sided Auction) ---")
    
    # --- 1. Setup & Data Generation (Unchanged) ---
    np.random.seed(42) # For generating profiles
    num_households, num_prosumers = config.TOTAL_HOUSEHOLDS, int(config.TOTAL_HOUSEHOLDS * config.PROSUMER_SHARE)
    load_profiles_kwh = common.generate_household_profiles(num_households, config.NUM_INTERVALS)
    pv_profiles_kwh = common.generate_pv_profiles(num_prosumers, config.NUM_INTERVALS, config.PEAK_PV_GENERATION_KW)
    generation_profiles_kwh = np.zeros_like(load_profiles_kwh)
    generation_profiles_kwh[:num_prosumers, :] = pv_profiles_kwh

    # --- 2. Simulation Initialization (Unchanged) ---
    battery_soc_kwh = np.zeros((num_prosumers, config.NUM_INTERVALS + 1))
    battery_soc_kwh[:, 0] = config.BATTERY_CAPACITY_KWH * 0.5
    grid_imports_kwh = np.zeros_like(load_profiles_kwh)
    grid_exports_kwh = np.zeros_like(load_profiles_kwh)
    curtailment_kwh = np.zeros_like(generation_profiles_kwh)
    
    # --- NEW: Simulation logging ---
    all_trades = [] # List to store DataFrames from each interval
    mcp_log = np.full(config.NUM_INTERVALS, np.nan)


    # --- 3. Simulation Loop (COMPLETELY REBUILT) ---
    for t in range(config.NUM_INTERVALS):
        
        # 3.1) Net energy pre-battery/trade
        net_energy_kwh = generation_profiles_kwh[:, t] - load_profiles_kwh[:, t]
        
        # 3.2) Prosumers use battery for self-consumption
        battery_soc_kwh[:, t+1] = battery_soc_kwh[:, t]
        for i in range(num_prosumers):
            if net_energy_kwh[i] < 0: # Deficit
                from_battery = min(abs(net_energy_kwh[i]), battery_soc_kwh[i, t+1])
                battery_soc_kwh[i, t+1] -= from_battery
                net_energy_kwh[i] += from_battery

        # 3.3) Build the agent list for the auction
        agents_data = []
        
        # Find Prosumers (Sellers)
        for i in range(num_prosumers):
            if net_energy_kwh[i] > 1e-12: # Has surplus
                agents_data.append({
                    'id': i,
                    'type': 'prosumer',
                    'energy': net_energy_kwh[i],
                    'soc_kwh': battery_soc_kwh[i, t+1] # Pass SoC for strategy
                })
        
        # Find Consumers (Buyers)
        for i in range(num_households):
            if net_energy_kwh[i] < -1e-12: # Has deficit
                agents_data.append({
                    'id': i,
                    'type': 'consumer',
                    'energy': abs(net_energy_kwh[i]),
                    'soc_kwh': 0 # N/A for consumers
                })
        
        if not agents_data:
            continue # No one to trade
            
        agents_df = pd.DataFrame(agents_data)

        # 3.4) Apply the "Noisy" Bidding Strategy
        agents_with_bids_df = auction_engine.get_noisy_bidding_strategy(agents_df)

        # 3.5) Run the auction
        trades_df, mcp, cleared_kwh = auction_engine.run_double_sided_auction(
            t, agents_with_bids_df
        )
        
        mcp_log[t] = mcp # Log the MCP for this interval

        if cleared_kwh > 0:
            all_trades.append(trades_df)
            
            # 3.6) Update net energy from trade results
            trades_by_seller = trades_df.groupby('prosumer_id')['p2p_trade_volume'].sum()
            for seller_id, sold_kwh in trades_by_seller.items():
                # --- THIS IS THE FIX ---
                net_energy_kwh[int(seller_id)] -= sold_kwh
            
            trades_by_buyer = trades_df.groupby('consumer_id')['p2p_trade_volume'].sum()
            for buyer_id, bought_kwh in trades_by_buyer.items():
                # --- THIS IS THE FIX ---
                net_energy_kwh[int(buyer_id)] += bought_kwh
        
        # 3.7) Post-trade: Battery charging & Grid Import/Export (Unchanged)
        for i in range(num_households):
            if i < num_prosumers: # Prosumer
                if net_energy_kwh[i] > 1e-12: # Surplus
                    to_store = min(net_energy_kwh[i], config.BATTERY_CAPACITY_KWH - battery_soc_kwh[i, t+1])
                    battery_soc_kwh[i, t+1] += to_store
                    remaining_surplus = net_energy_kwh[i] - to_store
                    if remaining_surplus > 0: 
                        grid_exports_kwh[i, t] = remaining_surplus
                        net_energy_kwh[i] = 0
            
            if net_energy_kwh[i] < -1e-12: # Deficit
                grid_imports_kwh[i, t] = abs(net_energy_kwh[i])
                net_energy_kwh[i] = 0

    print("Simulation complete.")
    
    # Consolidate all trades into one DataFrame
    final_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    return {
        "trades_df": final_trades_df, 
        "imports_kwh": grid_imports_kwh, 
        "exports_kwh": grid_exports_kwh, 
        "curtailment_kwh": curtailment_kwh, 
        "load_kwh": load_profiles_kwh, 
        "generation_kwh": generation_profiles_kwh, 
        "battery_soc_kwh": battery_soc_kwh,
        "mcp_series": mcp_log # Add MCP log for comparator
    }

# --- All Reporting & Plotting Functions (Unchanged) ---
# (These will now work correctly with the new results)

def print_s3b_summary_report(results):
    trades_df = results["trades_df"]
    if trades_df.empty:
        print("No trades occurred in this simulation.")
        return

    num_prosumers = int(config.TOTAL_HOUSEHOLDS * config.PROSUMER_SHARE)
    num_consumers = config.TOTAL_HOUSEHOLDS - num_prosumers
    welfare_results = common.calculate_economic_welfare(trades_df, config.UTILITY_TARIFF)
    fairness_results = common.calculate_fairness_metrics(welfare_results['consumer_surplus_per_agent'], welfare_results['prosumer_profit_per_agent'])
    technical_results = common.calculate_technical_robustness(trades_df, config.FEEDER_CAPACITY_KW)
    total_grid_import_kwh = np.sum(results["imports_kwh"])
    total_utility_cost_zar = total_grid_import_kwh * config.UTILITY_TARIFF
    total_grid_export_kwh = np.sum(results["exports_kwh"])
    total_fit_revenue_zar = total_grid_export_kwh * config.UTILITY_FIT_RATE
    print("\n" + "="*50); print("--- S3b Noisy Baseline: Metrics Summary ---"); print("="*50)
    print("\n## Aggregate Metrics (Entire Community) ##")
    print(f"Total Utility Cost (ZAR):   ZAR {total_utility_cost_zar:,.2f}"); print(f"Total FiT Revenue (ZAR):    ZAR {total_fit_revenue_zar:,.2f}")
    if not trades_df.empty:
        print(f"Total P2P Volume Traded:    {trades_df['p2p_trade_volume'].sum():,.2f} kWh")
    print(f"Total Grid Imports:         {total_grid_import_kwh:,.2f} kWh"); print(f"Total Grid Exports:         {total_grid_export_kwh:,.2f} kWh"); print(f"Total Energy Curtailed:     {np.sum(results['curtailment_kwh']):,.2f} kWh")
    print("\n" + "-"*50); print("\n## Thesis Comparison Metrics (S3b) ##")
    print(f"Total System Benefit (P2P): ZAR {welfare_results['total_system_benefit']:,.2f}"); print(f"Gini Index (P2P Benefits):  {fairness_results['gini_index']:.3f}")
    print(f"Feeder Overload Frequency:  {technical_results['feeder_overload_frequency_percent']:.2f}%"); print("\n" + "="*50)

def create_s3b_visualizations(results):
    """Calls all the individual plotting functions to generate S3b graphs."""
    print("\n--- Generating Visualizations for S3b Model ---")
    plot_s3b_mcp(results) # NEW plot for MCP
    plot_energy_destination(results)
    plot_s3b_grid_impact(results)
    plot_s3b_benefit_distribution(results)
    plot_s3b_community_power_flow(results)
    plot_s3b_average_daily_power_transfer(results)
    print("\nAll visualizations have been saved to PNG files.")
    plt.close('all')

def plot_s3b_mcp(results):
    """NEW: Plots the MCP from the noisy auction."""
    mcp_series = results.get("mcp_series")
    if mcp_series is None:
        return

    plt.figure(figsize=(15, 7))
    plt.plot(mcp_series, label='MCP', linestyle='-', marker='.', markersize=2, alpha=0.6)
    plt.axhline(config.UTILITY_TARIFF, color='darkred', linestyle='--', label=f"Utility Tariff (ZAR {config.UTILITY_TARIFF})")
    plt.axhline(config.UTILITY_FIT_RATE, color='darkgreen', linestyle='--', label=f"FiT Rate (ZAR {config.UTILITY_FIT_RATE})")
    plt.title('S3b Noisy Baseline: Market Clearing Price (MCP) Over Time')
    plt.ylabel('Price (ZAR/kWh)')
    plt.xlabel('Interval (15-min)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('S3b_mcp_noisy_baseline.png')
    print("Saved 'S3b_mcp_noisy_baseline.png'")

def plot_energy_destination(results):
    trades_df = results['trades_df']
    p2p_sold = trades_df['p2p_trade_volume'].sum() if not trades_df.empty else 0
    fit_sold = np.sum(results['exports_kwh'])
    initial_soc = np.sum(results['battery_soc_kwh'][:, 0])
    final_soc = np.sum(results['battery_soc_kwh'][:, -1])
    net_stored = max(0, final_soc - initial_soc)
    total_surplus = p2p_sold + fit_sold + net_stored
    labels = ['Sold on P2P Market', 'Sold to Utility (FiT)', 'Stored in Batteries (Net)']
    values = [p2p_sold, fit_sold, net_stored]
    colors = ['#9467bd', '#ff7f0e', '#1f77b4']
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('S3b: Final Destination of Prosumer Surplus Energy', fontsize=16)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f} kWh'))
    ax.bar_label(bars, fmt='{:,.0f} kWh', padding=3)
    fig.text(0.5, 0.95, f'Total Surplus Generated: {total_surplus:,.0f} kWh', 
             ha='center', va='top', fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.9]); plt.savefig('S3b_energy_destination.png')
    print("Saved 'S3b_energy_destination.png'")

def plot_s3b_grid_impact(results):
    trades_df = results['trades_df']
    if trades_df.empty:
        print("Skipping 'S3b_grid_impact.png' - no trades.")
        return
    p2p_flow_kwh = trades_df.groupby('interval')['p2p_trade_volume'].sum()
    p2p_flow_kw = p2p_flow_kwh.reindex(range(config.NUM_INTERVALS), fill_value=0).values * 4
    plt.figure(figsize=(15, 7)); plt.plot(p2p_flow_kw, label='P2P Power Transfer', color='#1f77b4')
    plt.axhline(y=config.FEEDER_CAPACITY_KW, color='r', linestyle='--', linewidth=2, label=f'Feeder Capacity ({config.FEEDER_CAPACITY_KW} kW)')
    plt.title('S3b Grid Impact: P2P Transfers vs. Feeder Capacity', fontsize=16); plt.ylabel('Power (kW)'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig('S3b_grid_impact.png'); print("Saved 'S3b_grid_impact.png'")

def plot_s3b_benefit_distribution(results):
    trades_df = results['trades_df']
    if trades_df.empty:
        print("Skipping 'S3b_benefit_distribution.png' - no trades.")
        return
    welfare_results = common.calculate_economic_welfare(trades_df, config.UTILITY_TARIFF)
    consumer_surplus, prosumer_profit, total_benefit = welfare_results['consumer_surplus_per_agent'].sum(), welfare_results['prosumer_profit_per_agent'].sum(), welfare_results['total_system_benefit']
    fig, ax = plt.subplots(figsize=(10, 7)); bar_width, index = 0.35, np.arange(1)
    ax.bar(index - bar_width/2, [consumer_surplus], bar_width, label='Total Consumer Surplus', color='#1f77b4')
    ax.bar(index + bar_width/2, [prosumer_profit], bar_width, label='Total Prosumer Profit', color='#2ca02c')
    ax.text(index - bar_width/2, consumer_surplus, f'ZAR {consumer_surplus:,.2f}', ha='center', va='bottom', fontsize=12)
    ax.text(index + bar_width/2, prosumer_profit, f'ZAR {prosumer_profit:,.2f}', ha='center', va='bottom', fontsize=12)
    ax.set_ylabel('Financial Benefit (ZAR)'); ax.set_title('S3b: Distribution of P2P Economic Benefits', fontsize=16)
    ax.set_xticks(index); ax.set_xticklabels(['P2P Market Benefits']); ax.legend()
    fig.text(0.5, 0.95, f'Total P2P System Benefit: ZAR {total_benefit:,.2f}', ha='center', va='top', fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.9]); plt.savefig('S3b_benefit_distribution.png'); print("Saved 'S3b_benefit_distribution.png'")

def plot_s3b_community_power_flow(results, week_to_plot=1):
    trades_df = results['trades_df']
    start, end = (week_to_plot - 1) * 7 * 96, (week_to_plot) * 7 * 96
    agg_load_kw = np.sum(results["load_kwh"], axis=0)[start:end] * 4; agg_gen_kw = np.sum(results["generation_kwh"], axis=0)[start:end] * 4
    agg_imports_kw = np.sum(results["imports_kwh"], axis=0)[start:end] * 4
    
    p2p_flow_kwh = pd.Series(dtype=float)
    if not trades_df.empty:
        p2p_flow_kwh = trades_df.groupby('interval')['p2p_trade_volume'].sum()
    
    p2p_flow_kw = p2p_flow_kwh.reindex(range(config.NUM_INTERVALS), fill_value=0).values[start:end] * 4
    self_consumed_kw = np.minimum(agg_load_kw, agg_gen_kw)
    time_index = pd.to_datetime(pd.date_range(start='2025-01-01', periods=len(agg_load_kw), freq='15min'))
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.stackplot(time_index, self_consumed_kw, p2p_flow_kw, agg_imports_kw, labels=['Self-Consumption', 'P2P Import', 'Grid Import'], colors=['#2ca02c', '#9467bd', '#ff7f0e'], alpha=0.7)
    ax.plot(time_index, agg_load_kw, label='Total Community Load', color='black', linewidth=2)
    ax.set_ylabel('Power (kW)'); ax.set_title(f'S3b Community Power Flow (Week {week_to_plot})', fontsize=16); ax.legend(loc='upper left')
    plt.tight_layout(); plt.savefig('S3b_community_power_flow.png'); print("Saved 'S3b_community_power_flow.png'")

def plot_s3b_average_daily_power_transfer(results):
    trades_df = results['trades_df']
    num_intervals = config.NUM_INTERVALS
    intervals_per_day = 96
    num_days = int(num_intervals / intervals_per_day)
    def daily_avg_kw(arr):
        return np.mean(arr.reshape(num_days, intervals_per_day), axis=0) * 4.0
    avg_load_kw = daily_avg_kw(np.sum(results["load_kwh"], axis=0))
    avg_gen_kw = daily_avg_kw(np.sum(results["generation_kwh"], axis=0))
    avg_imports_kw = daily_avg_kw(np.sum(results["imports_kwh"], axis=0))
    avg_exports_kw = daily_avg_kw(np.sum(results["exports_kwh"], axis=0))
    
    p2p_flow_kwh = pd.Series(dtype=float)
    if not trades_df.empty:
        p2p_flow_kwh = trades_df.groupby('interval')['p2p_trade_volume'].sum()
    p2p_flow_kwh = p2p_flow_kwh.reindex(range(num_intervals), fill_value=0).values
    avg_p2p_kw = daily_avg_kw(p2p_flow_kwh)
    
    time_of_day = pd.to_datetime(pd.date_range(start='00:00', periods=96, freq='15min')).strftime('%H:%M')
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(time_of_day, avg_load_kw, label='Community Load', color='black', linewidth=3)
    ax.plot(time_of_day, avg_gen_kw, label='PV Generation', color='green', linewidth=2.5)
    ax.plot(time_of_day, avg_imports_kw, label='Grid Imports', color='red', linewidth=2.5, linestyle='--')
    ax.plot(time_of_day, avg_exports_kw, label='Grid Exports (FiT)', color='orange', linewidth=2, linestyle=':')
    ax.fill_between(time_of_day, avg_p2p_kw, label='P2P Transfer', color='purple', alpha=0.5)
    ax.axhline(y=config.FEEDER_CAPACITY_KW, color='darkred', linestyle=':', linewidth=2, label=f'Feeder Capacity ({config.FEEDER_CAPACITY_KW} kW)')
    ax.set_ylabel('Average Power (kW)'); ax.set_xlabel('Time of Day')
    ax.set_title('S3b: Average Daily Power Transfer', fontsize=16)
    ax.legend(loc='upper left'); ax.grid(True)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))
    plt.tight_layout(); plt.savefig('S3b_average_daily_power_transfer.png')
    print("Saved 'S3b_average_daily_power_transfer.png'")


# --- Main execution (Unchanged) ---
if __name__ == "__main__":
    s3b_results = run_s3b_simulation()
    print_s3b_summary_report(s3b_results)
    create_s3b_visualizations(s3b_results)

    # --- SHA256 / Provenance (Unchanged) ---
    print("\nSaving S3b results to CSV for on-chain scripts...")
    trades_df = s3b_results["trades_df"]
    
    if not trades_df.empty:
        csv_path = Path("s3b_trades_data.csv")
        trades_df.to_csv(csv_path, index=False)
        print("... S3b trades data saved:", csv_path)
        h = hashlib.sha256()
        with open(csv_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        sha_hex = h.hexdigest()
        (csv_path.with_suffix(csv_path.suffix + ".sha256")).write_text(sha_hex + "\n", encoding="utf-8")
        meta = {
            "filename": csv_path.name, "sha256": sha_hex,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": "S3b_model.py",
        }
        (csv_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("Wrote:", csv_path.with_suffix(".sha256"), "and", csv_path.with_suffix(".meta.json"))
    else:
        print("... No trades to save to CSV.")
# S3b_dynamic_MCP_model.py (UPDATED)
#
# This is now the "Strategic" model.
# It uses the new S3b_Auction_Engine to run a double-sided auction
# where all agents use the "Strategic" (SoC-based) bidding strategy.
#
# CORRECTION: Added int() casts in section 3.6 to fix IndexError.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config
import Common_Functions as common
from datetime import datetime

# --- NEW: Import the auction engine and strategies ---
import S3b_Auction_Engine as auction_engine


def run_sim() -> dict:
    """
    Executes the S3b "Strategic" simulation.
    """
    print("--- Running S3b: Strategic Model (Double-Sided Auction) ---")
    
    # --- 1. Setup & Data Generation (Identical to baseline) ---
    np.random.seed(42) # For generating profiles
    num_households, num_prosumers = config.TOTAL_HOUSEHOLDS, int(config.TOTAL_HOUSEHOLDS * config.PROSUMER_SHARE)
    load_profiles_kwh = common.generate_household_profiles(num_households, config.NUM_INTERVALS)
    pv_profiles_kwh = common.generate_pv_profiles(num_prosumers, config.NUM_INTERVALS, config.PEAK_PV_GENERATION_KW)
    generation_profiles_kwh = np.zeros_like(load_profiles_kwh)
    generation_profiles_kwh[:num_prosumers, :] = pv_profiles_kwh

    # --- 2. Simulation Initialization (Identical to baseline) ---
    battery_soc_kwh = np.zeros((num_prosumers, config.NUM_INTERVALS + 1))
    battery_soc_kwh[:, 0] = config.BATTERY_CAPACITY_KWH * 0.5
    grid_imports_kwh = np.zeros_like(load_profiles_kwh)
    grid_exports_kwh = np.zeros_like(load_profiles_kwh)
    curtailment_kwh = np.zeros_like(generation_profiles_kwh)
    
    # --- NEW: Simulation logging ---
    all_trades = [] # List to store DataFrames from each interval
    mcp_log = np.full(config.NUM_INTERVALS, np.nan)


    # --- 3. Simulation Loop (Identical Structure to baseline) ---
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

        # 3.4) Apply the "Strategic" Bidding Strategy
        # --- THIS IS THE ONLY LINE THAT DIFFERS FROM THE BASELINE MODEL ---
        agents_with_bids_df = auction_engine.get_strategic_bidding_strategy(agents_df)
        # --- ----------------------------------------------------------- ---

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

# --- Reporting & Plotting Functions ---
# (These are updated to reflect the new model's name)

def print_summary_report(results: dict):
    """Prints a summary report of the simulation results."""
    trades_df = results["trades_df"]
    if trades_df.empty:
        print("No trades occurred in this simulation.")
        return

    welfare = common.calculate_economic_welfare(trades_df, config.UTILITY_TARIFF)
    fairness = common.calculate_fairness_metrics(
        welfare["consumer_surplus_per_agent"], welfare["prosumer_profit_per_agent"]
    )
    technical = common.calculate_technical_robustness(trades_df, config.FEEDER_CAPACITY_KW)

    total_grid_import = np.sum(results["imports_kwh"])
    total_grid_export = np.sum(results["exports_kwh"])
    
    print("\n" + "="*50)
    print("--- S3b Strategic Model (SoC) - Metrics Summary ---")
    print("="*50)
    print("\n## Thesis Comparison Metrics ##")
    print(f"Total System Benefit (P2P): ZAR {welfare['total_system_benefit']:,.2f}")
    print(f"Gini Index (P2P Benefits):  {fairness['gini_index']:.3f}")
    print(f"Total P2P Volume Traded:    {trades_df['p2p_trade_volume'].sum():,.2f} kWh")
    
    print("\n## Grid Impact Metrics ##")
    print(f"Total Grid Imports:         {total_grid_import:,.2f} kWh")
    print(f"Total Grid Exports:         {total_grid_export:,.2f} kWh")
    print(f"Feeder Overload Frequency:  {technical['feeder_overload_frequency_percent']:.2f}%")
    print("\n" + "="*50)

def plot_s3b_mcp(results: dict):
    """Plots the MCP from the strategic auction."""
    mcp_series = results.get("mcp_series")
    if mcp_series is None:
        return

    plt.figure(figsize=(15, 7))
    plt.plot(mcp_series, label='MCP', linestyle='-', marker='.', markersize=2, alpha=0.6)
    plt.axhline(config.UTILITY_TARIFF, color='darkred', linestyle='--', label=f"Utility Tariff (ZAR {config.UTILITY_TARIFF})")
    plt.axhline(config.UTILITY_FIT_RATE, color='darkgreen', linestyle='--', label=f"FiT Rate (ZAR {config.UTILITY_FIT_RATE})")
    plt.title('S3b Strategic Model: Market Clearing Price (MCP) Over Time')
    plt.ylabel('Price (ZAR/kWh)')
    plt.xlabel('Interval (15-min)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('S3b_mcp_strategic_model.png')
    plt.close()
    print("Saved 'S3b_mcp_strategic_model.png'")

def plot_s3b_avg_daily_power(results: dict):
    """Generates the average daily power transfer plot."""
    trades_df = results['trades_df']
    num_intervals = config.NUM_INTERVALS
    intervals_per_day = 96
    num_days = max(1, num_intervals // intervals_per_day)

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
    
    time_of_day = pd.date_range("00:00", periods=intervals_per_day, freq="15min").strftime("%H:%M")

    plt.figure(figsize=(15, 8))
    plt.plot(time_of_day, avg_load_kw, label="Community Load", linewidth=3)
    plt.plot(time_of_day, avg_gen_kw, label="PV Generation", linewidth=2.5)
    plt.plot(time_of_day, avg_imports_kw, label="Grid Imports", linestyle="--", linewidth=2.5)
    plt.plot(time_of_day, avg_exports_kw, label="Grid Exports (FiT)", linestyle=":", linewidth=2)
    plt.fill_between(time_of_day, avg_p2p_kw, label="P2P Transfer", alpha=0.35)

    plt.title("S3b Strategic Model: Average Daily Power Transfer (kW)")
    plt.xlabel("Time of Day")
    plt.ylabel("kW")
    plt.legend(loc="upper left")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(8))
    plt.tight_layout()
    plt.savefig("S3b_average_daily_power_strategic.png")
    plt.close()
    print("Saved 'S3b_average_daily_power_strategic.png'")

# --- Main execution ---
if __name__ == "__main__":
    start_time = datetime.now()
    # Note: run_sim() has no parameters anymore, they are in the strategy
    sim_results = run_sim() 
    print_summary_report(sim_results)
    
    # Generate plots
    plot_s3b_mcp(sim_results)
    plot_s3b_avg_daily_power(sim_results)
    
    end_time = datetime.now()
    print(f"\n--- S3b Strategic Model run complete in {end_time - start_time} ---")
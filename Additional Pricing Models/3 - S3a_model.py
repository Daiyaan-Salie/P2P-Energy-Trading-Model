# S3a_Grid_Aware_Model.py
#
# This script simulates the S3a Scenario: A Grid-Aware Community Energy Market.
# It includes a "traffic light" congestion management algorithm and generates
# a full suite of visualizations to analyze the results.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config
import Common_Functions as common

# --- COPY AND REPLACE THIS ENTIRE FUNCTION IN S3a_model.py ---

def run_s3a_simulation():
    """
    Executes the S3a Grid-Aware Community Market simulation with a robust and correct
    congestion management algorithm.
    """
    print("--- Running S3a: Grid-Aware Market Model (Corrected v2) ---")

    # --- 1. & 2. Setup and Initialization (No changes) ---
    np.random.seed(42)
    num_households = config.TOTAL_HOUSEHOLDS
    num_prosumers = int(num_households * config.PROSUMER_SHARE)
    num_intervals = config.NUM_INTERVALS
    load_profiles_kwh = common.generate_household_profiles(num_households, num_intervals)
    pv_profiles_kwh = common.generate_pv_profiles(num_prosumers, num_intervals, config.PEAK_PV_GENERATION_KW)
    generation_profiles_kwh = np.zeros_like(load_profiles_kwh)
    generation_profiles_kwh[:num_prosumers, :] = pv_profiles_kwh
    battery_soc_kwh = np.zeros((num_prosumers, num_intervals + 1))
    battery_soc_kwh[:, 0] = config.BATTERY_CAPACITY_KWH * 0.5
    grid_imports_kwh = np.zeros_like(load_profiles_kwh)
    curtailment_kwh = np.zeros_like(generation_profiles_kwh)
    trade_log, potential_trade_log = [], []

    # --- 3. Simulation Loop ---
    for t in range(num_intervals):
        # (Initial net energy calculation is the same)
        net_energy_kwh = generation_profiles_kwh[:, t] - load_profiles_kwh[:, t]
        battery_soc_kwh[:, t+1] = battery_soc_kwh[:, t]
        for i in range(num_prosumers):
            if net_energy_kwh[i] < 0:
                from_battery = min(abs(net_energy_kwh[i]), battery_soc_kwh[i, t+1])
                battery_soc_kwh[i, t+1] -= from_battery
                net_energy_kwh[i] += from_battery

        sellers = {i: net_energy_kwh[i] for i in range(num_prosumers) if net_energy_kwh[i] > 0}
        buyers = {i: abs(net_energy_kwh[i]) for i in range(num_households) if net_energy_kwh[i] < 0}
        total_supply_kwh, total_demand_kwh = sum(sellers.values()), sum(buyers.values())

        if total_supply_kwh > 0 and total_demand_kwh > 0:
            mcp = common.uniform_mcp(config.PRICE_FLOOR, config.UTILITY_TARIFF)
            potential_trade_volume_kwh = min(total_supply_kwh, total_demand_kwh)
            potential_trade_log.append({'interval': t, 'p2p_trade_volume': potential_trade_volume_kwh})

            # --- ROBUST CONGESTION MANAGEMENT LOGIC ---
            final_trade_volume_kwh = potential_trade_volume_kwh
            proposed_power_kw = potential_trade_volume_kwh * 4

            # 1. Determine the final, safe trade volume for the interval
            if proposed_power_kw > config.FEEDER_CAPACITY_KW:
                final_trade_volume_kwh = config.FEEDER_CAPACITY_KW / 4
            # --- END OF LOGIC ---
            
            # 2. Allocate this safe volume proportionally
            if final_trade_volume_kwh > 0:
                for buyer_id, demand in buyers.items():
                    # Each buyer gets a pro-rata share of the final, safe volume
                    amount_bought_by_buyer = (demand / total_demand_kwh) * final_trade_volume_kwh
                    
                    # This buyer's purchase is sourced proportionally from all sellers
                    for seller_id, supply in sellers.items():
                        if supply > 0 and total_supply_kwh > 0:
                            amount_sold_to_buyer = (supply / total_supply_kwh) * amount_bought_by_buyer
                            
                            trade_log.append({
                                'interval': t, 'prosumer_id': seller_id, 'consumer_id': buyer_id,
                                'p2p_trade_volume': amount_sold_to_buyer, 'market_price': mcp,
                            })
                            # Update net energy balances with the actual traded amounts
                            net_energy_kwh[buyer_id] += amount_sold_to_buyer
                            net_energy_kwh[seller_id] -= amount_sold_to_buyer
            
        # --- Post-Trade Settlement (No changes) ---
        for i in range(num_prosumers):
            if net_energy_kwh[i] > 0:
                to_store = min(net_energy_kwh[i], config.BATTERY_CAPACITY_KWH - battery_soc_kwh[i, t+1])
                battery_soc_kwh[i, t+1] += to_store
                curtailment_kwh[i, t] = net_energy_kwh[i] - to_store
        for i in range(num_households):
            if net_energy_kwh[i] < 0:
                grid_imports_kwh[i, t] = abs(net_energy_kwh[i])

    print("Simulation complete.")
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(columns=['p2p_trade_volume', 'interval'])
    potential_trades_df = pd.DataFrame(potential_trade_log) if potential_trade_log else pd.DataFrame(columns=['p2p_trade_volume'])
    
    return {
        "trades_df": trades_df, "potential_trades_df": potential_trades_df, "imports_kwh": grid_imports_kwh, 
        "curtailment_kwh": curtailment_kwh, "load_kwh": load_profiles_kwh, "generation_kwh": generation_profiles_kwh
    }

def print_s3a_summary_report(results):
    """
    Calculates and prints a comprehensive summary for the S3a model.
    """
    trades_df = results["trades_df"]
    if trades_df.empty:
        print("\nWarning: No P2P trades occurred in the simulation.")
        return

    welfare_results = common.calculate_economic_welfare(trades_df, config.UTILITY_TARIFF)
    fairness_results = common.calculate_fairness_metrics(
        welfare_results['consumer_surplus_per_agent'], welfare_results['prosumer_profit_per_agent']
    )
    technical_results = common.calculate_technical_robustness(trades_df, config.FEEDER_CAPACITY_KW)
    total_grid_import_kwh = np.sum(results["imports_kwh"])
    total_utility_cost_zar = total_grid_import_kwh * config.UTILITY_TARIFF
    
    print("\n" + "="*50)
    print("--- S3a Grid-Aware Market: Key Metrics Summary ---")
    print("="*50)
    print("\n## Aggregate Metrics (Entire Community) ##")
    print(f"Total Utility Cost (ZAR):   ZAR {total_utility_cost_zar:,.2f}")
    print(f"Total P2P Volume Traded:    {trades_df['p2p_trade_volume'].sum():,.2f} kWh")
    print(f"Total Grid Imports:         {total_grid_import_kwh:,.2f} kWh")
    print(f"Total Energy Curtailed:     {np.sum(results['curtailment_kwh']):,.2f} kWh")
    print("\n" + "-"*50)
    print("\n## Thesis Comparison Metrics (S3a) ##")
    print(f"Total System Benefit (P2P): ZAR {welfare_results['total_system_benefit']:,.2f}")
    print(f"Gini Index (P2P Benefits):  {fairness_results['gini_index']:.3f}")
    print(f"Feeder Overload Frequency:  {technical_results['feeder_overload_frequency_percent']:.2f}%")
    print("\n" + "="*50)

# --- S3a Visualization Functions ---

def create_s3a_visualizations(results):
    """
    Calls all the individual plotting functions to generate S3a graphs.
    """
    print("\n--- Generating Visualizations for S3a Model ---")
    plot_s3a_grid_impact(results)
    plot_trade_curtailment_analysis(results)
    plot_s3a_benefit_distribution(results)
    plot_s3a_community_power_flow(results)
    plot_s3a_average_daily_power_transfer(results)
    print("\nAll visualizations have been saved to PNG files.")
    plt.close('all')

def plot_s3a_grid_impact(results, week_to_plot=1):
    """
    Shows P2P flow vs. feeder capacity. Should show zero overloads.
    """
    p2p_flow_kwh = results['trades_df'].groupby('interval')['p2p_trade_volume'].sum()
    p2p_flow_kw = p2p_flow_kwh.reindex(range(config.NUM_INTERVALS), fill_value=0).values * 4
    intervals_per_day = 96
    start = (week_to_plot - 1) * 7 * intervals_per_day
    end = start + 7 * intervals_per_day
    time_index = pd.to_datetime(pd.date_range(start='2025-01-01', periods=len(p2p_flow_kw[start:end]), freq='15min'))

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(time_index, p2p_flow_kw[start:end], label='P2P Power Transfer', color='#1f77b4', zorder=2)
    ax.axhline(y=config.FEEDER_CAPACITY_KW, color='r', linestyle='--', linewidth=2, label=f'Feeder Capacity ({config.FEEDER_CAPACITY_KW} kW)', zorder=1)

    ax.set_ylabel('Power (kW)')
    ax.set_title('S3a Grid Impact: P2P Transfers vs. Feeder Capacity', fontsize=16)
    ax.legend()
    ax.grid(True)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('S3a_grid_impact.png')
    print("Saved 'S3a_grid_impact.png'")

def plot_trade_curtailment_analysis(results):
    """
    UNIQUE TO S3a: Shows how much P2P volume was curtailed by the algorithm.
    """
    potential_volume = results['potential_trades_df']['p2p_trade_volume'].sum()
    actual_volume = results['trades_df']['p2p_trade_volume'].sum()
    curtailed_volume = potential_volume - actual_volume

    labels = ['Actual P2P Volume', 'Volume Curtailed by Algorithm']
    values = [actual_volume, curtailed_volume]
    colors = ['#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('S3a: P2P Trade Curtailment due to Grid Congestion', fontsize=16)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f} kWh'))
    ax.bar_label(bars, fmt='{:,.0f} kWh', padding=3)

    print(f"Total potential P2P volume was {potential_volume:,.2f} kWh.")
    print(f"Total volume curtailed for safety: {curtailed_volume:,.2f} kWh.")

    plt.tight_layout()
    plt.savefig('S3a_trade_curtailment.png')
    print("Saved 'S3a_trade_curtailment.png'")

def plot_s3a_benefit_distribution(results):
    """
    Generates a grouped bar chart showing the split of S3a P2P benefits.
    """
    welfare_results = common.calculate_economic_welfare(results["trades_df"], config.UTILITY_TARIFF)
    consumer_surplus = welfare_results['consumer_surplus_per_agent'].sum()
    prosumer_profit = welfare_results['prosumer_profit_per_agent'].sum()
    total_system_benefit = welfare_results['total_system_benefit']

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create individual bars for Consumer Surplus and Prosumer Profit
    bar_width = 0.35
    index = np.arange(1) # For a single group of bars

    bar1 = ax.bar(index - bar_width/2, [consumer_surplus], bar_width, label='Total Consumer Surplus', color='#1f77b4')
    bar2 = ax.bar(index + bar_width/2, [prosumer_profit], bar_width, label='Total Prosumer Profit', color='#2ca02c')

    # Add labels to the bars
    ax.text(index - bar_width/2, consumer_surplus, f'ZAR {consumer_surplus:,.2f}', ha='center', va='bottom', fontsize=12)
    ax.text(index + bar_width/2, prosumer_profit, f'ZAR {prosumer_profit:,.2f}', ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Financial Benefit (ZAR)')
    ax.set_title('S3a: Distribution of Economic Benefits', fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(['P2P Market Benefits']) # A more descriptive x-label
    ax.legend()
    
    # Add a title for the total system benefit, placed above the bars
    fig.text(0.5, 0.95, f'Total System Benefit: ZAR {total_system_benefit:,.2f}', 
             ha='center', va='top', fontsize=14, weight='bold')

    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('ZAR {x:,.0f}'))
    
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to make space for the main title
    plt.savefig('S3a_benefit_distribution.png')
    print("Saved 'S3a_benefit_distribution.png'")

def plot_s3a_community_power_flow(results, week_to_plot=1):
    """
    Generates the S3a power flow chart with a (potentially smaller) P2P layer.
    """
    intervals_per_day = 96
    start = (week_to_plot - 1) * 7 * intervals_per_day
    end = start + 7 * intervals_per_day
    agg_load_kw = np.sum(results["load_kwh"], axis=0)[start:end] * 4
    agg_gen_kw = np.sum(results["generation_kwh"], axis=0)[start:end] * 4
    agg_imports_kw = np.sum(results["imports_kwh"], axis=0)[start:end] * 4
    p2p_flow_kwh = results['trades_df'].groupby('interval')['p2p_trade_volume'].sum()
    p2p_flow_kw = p2p_flow_kwh.reindex(range(config.NUM_INTERVALS), fill_value=0).values[start:end] * 4
    self_consumed_kw = np.minimum(agg_load_kw, agg_gen_kw)
    time_index = pd.to_datetime(pd.date_range(start='2025-01-01', periods=len(agg_load_kw), freq='15min'))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.stackplot(time_index, self_consumed_kw, p2p_flow_kw, agg_imports_kw, 
                 labels=['Self-Consumption', 'P2P Import', 'Grid Import'], 
                 colors=['#2ca02c', '#9467bd', '#ff7f0e'], alpha=0.7)
    ax.plot(time_index, agg_load_kw, label='Total Community Load', color='black', linewidth=2)
    ax.set_ylabel('Power (kW)')
    ax.set_title(f'S3a Community Power Flow (Week {week_to_plot})', fontsize=16)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('S3a_community_power_flow.png')
    print("Saved 'S3a_community_power_flow.png'")

def plot_s3a_average_daily_power_transfer(results):
    """
    Generates a line plot showing the average 24-hour power flows for the S3a model.
    """
    num_intervals = config.NUM_INTERVALS
    intervals_per_day = 96
    num_days = int(num_intervals / intervals_per_day)

    # --- 1. Process the data ---
    total_load_kwh = np.sum(results["load_kwh"], axis=0)
    total_gen_kwh = np.sum(results["generation_kwh"], axis=0)
    total_imports_kwh = np.sum(results["imports_kwh"], axis=0)
    
    p2p_flow_kwh = results['trades_df'].groupby('interval')['p2p_trade_volume'].sum()
    p2p_flow_kwh = p2p_flow_kwh.reindex(range(num_intervals), fill_value=0).values

    # --- 2. Calculate the daily average and convert to kW ---
    avg_load_kw = np.mean(total_load_kwh.reshape(num_days, intervals_per_day), axis=0) * 4
    avg_gen_kw = np.mean(total_gen_kwh.reshape(num_days, intervals_per_day), axis=0) * 4
    avg_imports_kw = np.mean(total_imports_kwh.reshape(num_days, intervals_per_day), axis=0) * 4
    avg_p2p_kw = np.mean(p2p_flow_kwh.reshape(num_days, intervals_per_day), axis=0) * 4
    
    # --- 3. Plotting ---
    time_of_day = pd.to_datetime(pd.date_range(start='00:00', periods=96, freq='15min')).strftime('%H:%M')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(time_of_day, avg_load_kw, label='Community Load', color='black', linewidth=3)
    ax.plot(time_of_day, avg_gen_kw, label='PV Generation', color='green', linewidth=2.5)
    ax.plot(time_of_day, avg_imports_kw, label='Grid Imports', color='red', linewidth=2.5, linestyle='--')
    ax.fill_between(time_of_day, avg_p2p_kw, label='P2P Transfer (Grid-Aware)', color='purple', alpha=0.5)

    # Add the feeder capacity line to this graph for context
    ax.axhline(y=config.FEEDER_CAPACITY_KW, color='darkred', linestyle=':', linewidth=2, label=f'Feeder Capacity ({config.FEEDER_CAPACITY_KW} kW)')

    ax.set_ylabel('Average Power (kW)')
    ax.set_xlabel('Time of Day')
    ax.set_title('S3a: Average Daily Power Transfer', fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True)
    
    ax.xaxis.set_major_locator(mticker.MultipleLocator(8))
    
    plt.tight_layout()
    plt.savefig('S3a_average_daily_power_transfer.png')
    print("Saved 'S3a_average_daily_power_transfer.png'")


# Main execution block
if __name__ == "__main__":
    s3a_results = run_s3a_simulation()
    print_s3a_summary_report(s3a_results)
    create_s3a_visualizations(s3a_results)
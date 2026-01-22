# run_multisize_sensitivity_analysis.py

import pandas as pd
import numpy as np
from itertools import product
import time
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# SECTION 1: CONFIGURATION PARAMETERS
# ==============================================================================
# TOTAL_HOUSEHOLDS is now defined in the analysis loop, not as a global constant.
NUM_INTERVALS = 96 * 14
PRICE_FLOOR = 0.50
PEAK_PV_GENERATION_KW = 7.5
BATTERY_CAPACITY_KWH = 13.5
BATTERY_CHARGE_THRESHOLD = 0.90

# ==============================================================================
# SECTION 2: COMMON HELPER FUNCTIONS & SIMULATION ENGINE
# ==============================================================================
# (These sections are unchanged from the previous version)

def _gaussian(x, mu, sig, amplitude):
    return amplitude * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def generate_household_profiles(num_households, num_intervals):
    intervals_per_day = 96; x = np.arange(intervals_per_day)
    base_load_kw = 0.25 + 0.1 * np.sin(np.pi * (x - 24) / 48)
    morning_peak = _gaussian(x, mu=28, sig=4, amplitude=1.0)
    evening_peak = _gaussian(x, mu=76, sig=6, amplitude=1.75)
    base_daily_profile_kw = base_load_kw + morning_peak + evening_peak
    num_days = int(np.ceil(num_intervals / intervals_per_day)); full_base_profile = np.tile(base_daily_profile_kw, num_days)[:num_intervals]
    household_loads = []
    for _ in range(num_households):
        noise = np.random.normal(loc=0, scale=0.1, size=num_intervals)
        scaling_factor = np.random.uniform(low=0.7, high=1.5)
        profile_kw = (full_base_profile * scaling_factor) + noise; profile_kw[profile_kw < 0] = 0
        household_loads.append(profile_kw / 4)
    return np.array(household_loads)
def generate_pv_profiles(num_prosumers, num_intervals, peak_generation_kw):
    hours = np.arange(0, 24, 0.25)
    base_daily_profile = np.maximum(0, peak_generation_kw * np.sin(np.pi * (hours - 6) / 12))
    num_days = int(np.ceil(num_intervals / 96)); full_base_profile = np.tile(base_daily_profile, num_days)[:num_intervals]
    pv_generations = []
    for _ in range(num_prosumers):
        noise = np.random.normal(loc=0, scale=0.1, size=num_intervals)
        scaling_factor = np.random.uniform(low=0.8, high=1.1)
        profile = (full_base_profile * scaling_factor) + noise; profile[profile < 0] = 0
        pv_generations.append(profile / 4)
    return np.array(pv_generations)
def calculate_economic_welfare(results_df, utility_tariff):
    results_df['consumer_surplus'] = (utility_tariff - results_df['market_price']) * results_df['p2p_trade_volume']
    results_df['prosumer_profit'] = results_df['market_price'] * results_df['p2p_trade_volume']
    total_consumer_surplus = results_df.groupby('consumer_id')['consumer_surplus'].sum(); total_prosumer_profit = results_df.groupby('prosumer_id')['prosumer_profit'].sum()
    return {"total_system_benefit": total_consumer_surplus.sum() + total_prosumer_profit.sum(), "consumer_surplus_per_agent": total_consumer_surplus, "prosumer_profit_per_agent": total_prosumer_profit}
def calculate_gini_index(benefits):
    if benefits.sum() == 0: return 0.0
    sorted_benefits = np.sort(np.array(benefits)); n = len(sorted_benefits); index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_benefits)) / (n * np.sum(sorted_benefits)) - (n + 1) / n)
def calculate_fairness_metrics(consumer_benefits, prosumer_benefits):
    all_benefits = pd.concat([consumer_benefits, prosumer_benefits]).fillna(0).values; gini_index = calculate_gini_index(all_benefits)
    total_consumer_surplus, total_prosumer_profit = consumer_benefits.sum(), prosumer_benefits.sum()
    c_to_p_ratio = total_consumer_surplus / total_prosumer_profit if total_prosumer_profit > 0 else np.inf
    return {"gini_index": gini_index, "c_to_p_ratio": c_to_p_ratio}
def calculate_technical_robustness(results_df, feeder_capacity_kw):
    """Calculates feeder overload frequency and severity."""
    if results_df.empty: 
        return {
            "overload_frequency_percent": 0.0, 
            "overload_severity_percent": 0.0
        }
        
    interval_trades_kw = results_df.groupby('interval')['p2p_trade_volume'].sum() * 4
    overloaded_intervals = interval_trades_kw[interval_trades_kw > feeder_capacity_kw]
    
    num_total_intervals = len(results_df['interval'].unique())
    num_overloaded = len(overloaded_intervals)
    
    overload_frequency = (num_overloaded / num_total_intervals) * 100 if num_total_intervals > 0 else 0
    overload_severity = ((overloaded_intervals - feeder_capacity_kw) / feeder_capacity_kw).mean() * 100 if num_overloaded > 0 else 0
    
    # --- FIX ---
    # Changed the key name from 'feeder_overload_frequency_percent' to 'overload_frequency_percent'.
    return {
        "overload_frequency_percent": overload_frequency, 
        "overload_severity_percent": overload_severity
    }
def run_simulation(total_households, num_consumers, num_prosumers, feeder_capacity_kw, utility_tariff):
    if num_prosumers == 0 or num_consumers == 0: return pd.DataFrame()
    loads_kwh = generate_household_profiles(total_households, NUM_INTERVALS)
    pvs_kwh = generate_pv_profiles(num_prosumers, NUM_INTERVALS, PEAK_PV_GENERATION_KW)
    consumer_loads, prosumer_loads = loads_kwh[:num_consumers], loads_kwh[num_consumers:]
    battery_soc = np.full((num_prosumers, NUM_INTERVALS + 1), BATTERY_CAPACITY_KWH / 2); trade_log = []
    for t in range(NUM_INTERVALS):
        offers, net_energy = [], pvs_kwh[:, t] - prosumer_loads[:, t]
        for p_idx in range(num_prosumers):
            if net_energy[p_idx] > 0:
                if battery_soc[p_idx, t] < BATTERY_CAPACITY_KWH * BATTERY_CHARGE_THRESHOLD:
                    to_store = min(net_energy[p_idx], BATTERY_CAPACITY_KWH - battery_soc[p_idx, t])
                    battery_soc[p_idx, t+1] = battery_soc[p_idx, t] + to_store
                    if (kwh_for_sale := net_energy[p_idx] - to_store) > 0: offers.append({'id': p_idx, 'kwh': kwh_for_sale})
                else:
                    offers.append({'id': p_idx, 'kwh': net_energy[p_idx]}); battery_soc[p_idx, t+1] = battery_soc[p_idx, t]
            else:
                to_discharge = min(abs(net_energy[p_idx]), battery_soc[p_idx, t]); battery_soc[p_idx, t+1] = battery_soc[p_idx, t] - to_discharge
        bids = [{'id': c_idx, 'kwh': consumer_loads[c_idx, t]} for c_idx in range(num_consumers)]
        total_supply_kwh, total_demand_kwh = sum(o['kwh'] for o in offers), sum(b['kwh'] for b in bids); cleared_volume_kwh = min(total_supply_kwh, total_demand_kwh)
        if cleared_volume_kwh > 0:
            if (cleared_power_kw := cleared_volume_kwh * 4) > feeder_capacity_kw: cleared_volume_kwh = feeder_capacity_kw / 4
            market_price = (utility_tariff + PRICE_FLOOR) / 2
            for bid in bids:
                for offer in offers:
                    prop_vol = cleared_volume_kwh * (bid['kwh']/total_demand_kwh) * (offer['kwh']/total_supply_kwh)
                    trade_log.append({'interval': t, 'consumer_id': f"C{bid['id']}", 'prosumer_id': f"P{offer['id']}", 'p2p_trade_volume': prop_vol, 'market_price': market_price})
    return pd.DataFrame(trade_log)

# ==============================================================================
# SECTION 3: UPDATED - VISUALIZATION FUNCTION
# ==============================================================================

def generate_visualizations(results_df):
    """Generates a faceted heatmap to show stress across different network sizes."""
    print("\n📊 Generating visualizations...")
    
    # Use seaborn's `relplot` with a heatmap kind to create a faceted grid.
    # The `col` parameter will create a separate plot for each value in the 'total_households' column.
    g = sns.relplot(
        data=results_df,
        x="feeder_capacity_kw",
        y="prosumer_share",
        hue="overload_frequency_percent",
        size="overload_frequency_percent",
        palette="Reds",
        col="total_households",
        kind="scatter",
        height=5,
        aspect=0.8,
        sizes=(50, 500)
    )
    
    g.fig.suptitle("Technical Stress by Network Size", y=1.05, fontsize=16)
    g.set_axis_labels("Feeder Capacity (kW)", "Prosumer Share", fontsize=12)
    g.set_titles("Network Size: {col_name} Households", fontsize=12)
    g.despine(left=True, bottom=True)
    
    output_filename = 'faceted_heatmap_technical_stress.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"   - Saved {output_filename}")


# ==============================================================================
# SECTION 4: UPDATED - SENSITIVITY ANALYSIS ORCHESTRATOR
# ==============================================================================

def run_sensitivity_analysis():
    """Orchestrates the multi-size sensitivity analysis, now including a 10-user network."""
    print("🚀 Starting multi-size sensitivity analysis for 10, 20, 30, and 40-user networks...")
    start_time = time.time()
    
    # --- UPDATE ---
    # Added 10 to the list of network sizes to test.
    total_households_list = [10, 20, 30, 40]
    
    feeder_caps_kw = [2.5, 5.0, 7.5, 10.0]
    prosumer_shares = [0.25, 0.5, 0.75]
    utility_tariffs = [3.50]
    
    param_grid = list(product(total_households_list, feeder_caps_kw, prosumer_shares, utility_tariffs))
    all_results = []
    print(f"Total simulations to run: {len(param_grid)}")
    
    for i, params in enumerate(param_grid):
        total_households, feeder_cap, prosumer_share, utility_tariff = params
        
        # Ensure at least one prosumer and one consumer
        num_prosumers = int(total_households * prosumer_share)
        if num_prosumers == 0 and total_households > 0:
             num_prosumers = 1 # Ensure at least one prosumer for small networks
        if num_prosumers == total_households and total_households > 0:
             num_prosumers -= 1 # Ensure at least one consumer
        num_consumers = total_households - num_prosumers

        if num_consumers <= 0: continue # Skip scenarios with no consumers
        
        print(f"\nRunning sim {i+1}/{len(param_grid)}: Size={total_households}, Cap={feeder_cap}kW, Prosumers={num_prosumers}")
        
        results_df = run_simulation(total_households, num_consumers, num_prosumers, feeder_cap, utility_tariff)
        
        if not results_df.empty:
            welfare = calculate_economic_welfare(results_df.copy(), utility_tariff)
            fairness = calculate_fairness_metrics(welfare['consumer_surplus_per_agent'], welfare['prosumer_profit_per_agent'])
            robustness = calculate_technical_robustness(results_df, feeder_cap)
        else:
            welfare = {'total_system_benefit': 0}
            fairness = {'gini_index': 0, 'c_to_p_ratio': 0}
            robustness = {'overload_frequency_percent': 0, 'overload_severity_percent': 0}

        all_results.append({
            'total_households': total_households,
            'feeder_capacity_kw': feeder_cap,
            'prosumer_share': prosumer_share,
            'num_prosumers': num_prosumers,
            'num_consumers': num_consumers,
            'utility_tariff': utility_tariff,
            'total_system_benefit': welfare['total_system_benefit'],
            'gini_index': fairness['gini_index'],
            'c_to_p_ratio': fairness['c_to_p_ratio'],
            'overload_frequency_percent': robustness['overload_frequency_percent'],
            'overload_severity_percent': robustness['overload_severity_percent']
        })

    results_df = pd.DataFrame(all_results)
    output_filename = 'multisize_sensitivity_analysis_results.csv'
    results_df.to_csv(output_filename, index=False)
    
    end_time = time.time()
    print(f"\n✅ Analysis complete in {end_time - start_time:.2f} seconds. Results saved to '{output_filename}'")
    
    generate_visualizations(results_df)

# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    run_sensitivity_analysis()
"""
Common_Functions.py
===================

This module contains reusable helper functions used across all simulation scenarios (S1, S2, S3).

It includes:
- Synthetic load and PV profile generation
- Unit conversion utilities (kW ↔ kWh)
- Market-clearing price helpers
- Economic, fairness, and technical performance metrics


This module is imported by:
- S1_model.py (baseline scenario)
- S2_model.py (unconstrained P2P trading)
- S3_model.py (constrained P2P trading with feeder limits)
"""



import numpy as np
import pandas as pd

# ---------1. Data Generation Functions-----------

# --- Helper: Gaussian curve for synthetic demand shaping ----------------------
# Used to create smooth morning/evening peaks in demand when generating synthetic household load profiles.
def _gaussian(x, mu, sig, amplitude):
    return amplitude * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# --- Synthetic household demand profiles -------------
# Returns a matrix of household demand over time. The functions below use these profiles as the baseline consumption signal for each household.

def generate_household_profiles(num_households: int, num_intervals: int) -> np.ndarray:
    """
    Generates synthetic electricity demand profiles for a number of households. Profiles are generated in kWh for each interval.

    Args:
        num_households (int): The total number of households in the microgrid.
        num_intervals (int): The number of 15-minute intervals for the simulation.

    Returns:
        np.ndarray: A 2D array of shape (num_households, num_intervals)
                    containing load data in kWh per 15-minute interval.
    """
    # --- 1. Define the daily profile shape using 96 intervals (24 hours) ---
    intervals_per_day = 96
    x = np.arange(intervals_per_day) # Timesteps from 0 to 95

    # Base load: Represents constant consumption from appliances like fridges.
    # Slightly higher during the day.
    base_load_kw = 0.25 + 0.1 * np.sin(np.pi * (x - 24) / 48) # Low point at 6 AM

    # Morning Peak (e.g., centred at 07:00)
    # 7 AM is interval 28 (7 * 4)
    morning_peak = _gaussian(x, mu=28, sig=4, amplitude=1.0) # sig=4 means peak is ~2h wide

    # Evening Peak (e.g., centred at 19:00)
    # 7 PM is interval 76 (19 * 4)
    # This peak is typically higher than the morning one.
    evening_peak = _gaussian(x, mu=76, sig=6, amplitude=1.75) # sig=6 means peak is ~3h wide

    # Combine the components to form the characteristic daily profile in kW
    base_daily_profile_kw = base_load_kw + morning_peak + evening_peak

    # --- 2. Scale and tile the profile for the full simulation duration ---
    num_days = int(np.ceil(num_intervals / intervals_per_day))
    full_base_profile = np.tile(base_daily_profile_kw, num_days)[:num_intervals]

    # --- 3. Generate unique profiles for each household ---
    household_loads = []
    for _ in range(num_households):
        # Introduce randomness to make each household unique
        # loc=0 (mean), scale=0.1 (std dev)
        noise = np.random.normal(loc=0, scale=0.1, size=num_intervals)
        
        # scaling_factor represents different levels of household consumption
        scaling_factor = np.random.uniform(low=0.7, high=1.5)
        
        # Create the final profile in kW
        profile_kw = (full_base_profile * scaling_factor) + noise
        profile_kw[profile_kw < 0] = 0  # Consumption cannot be negative
        
        # Convert from average power (kW) over 15 mins to energy (kWh)
        # Energy (kWh) = Power (kW) * Time (h) -> Power * (15/60)
        profile_kwh = profile_kw / 4
        household_loads.append(profile_kwh)

    return np.array(household_loads)




# --- Synthetic PV generation profiles ----------------------------------------
# Generates PV output time-series for prosumers. These profiles are later used to compute net demand (load - PV) and energy available for P2P trading.

def generate_pv_profiles(num_prosumers, num_intervals, peak_generation=5.0):
    """
    Generates synthetic PV generation profiles for prosumers.

    Args:
        num_prosumers (int): The number of prosumer households.
        num_intervals (int): The number of 15-minute intervals.
        peak_generation (float): The peak power output of the PV system in kW.

    Returns:
        numpy.ndarray: A 2D array of shape (num_prosumers, num_intervals) with generation data in kWh.
    """
    # Create a base daily generation profile (sunny day)
    # Sinusoidal curve from roughly 6 AM to 6 PM
    hours = np.arange(0, 24, 0.25) # 15-minute intervals
    base_daily_profile = np.maximum(0, peak_generation * np.sin(np.pi * (hours - 6) / 12))
    
    num_days = int(np.ceil(num_intervals / 96))
    full_base_profile = np.tile(base_daily_profile, num_days)[:num_intervals]

    pv_generations = []
    for _ in range(num_prosumers):
        # Add random noise to simulate cloud cover and efficiency differences
        noise = np.random.normal(0, 0.1, num_intervals)
        scaling_factor = np.random.uniform(0.8, 1.1) # Efficiency variation
        profile = (full_base_profile * scaling_factor) + noise
        profile[profile < 0] = 0 # Ensure no negative generation
        pv_generations.append(profile / 4) # Convert from kW to kWh per 15-min interval
    
    return np.array(pv_generations)


# --- 2. Model Helper Functions ---
# --- Market price helper -----------------------------------------------------
# Uniform Market Clearing Price (MCP) helper.
# In this project, the MCP is constrained between a price floor and the utility
# tariff (see config.py). This supports the economic rationale that P2P prices should not exceed buying from the utility and should not fall below a floor.

def uniform_mcp(price_floor: float, utility_tariff: float) -> float:
    """Uniform market clearing price used by S2/S3a/S3b."""
    return (price_floor + utility_tariff) / 2.0

# --- Unit conversions --------------------------------------------------------
# Conversions assume fixed-length intervals (default 15 minutes).
# These are heavily referenced in the report because kW (power) and kWh (energy) are both used depending on whether we describe instantaneous loading or energy transacted per interval.

def kwh_to_kw(energy_kwh: float | np.ndarray, interval_minutes: int = 15):
    """Convert energy in a slot to average power over that slot."""
    return energy_kwh * (60.0 / interval_minutes)

def kw_to_kwh(power_kw: float | np.ndarray, interval_minutes: int = 15):
    """Convert average power over a slot to energy in that slot."""
    return power_kw * (interval_minutes / 60.0)


# --- 3. Economic Metrics ---
# --- Economic welfare --------------------------------------------------------
# Computes scenario-level welfare from the results DataFrame. This metric is used to compare baseline vs P2P scenarios in the report.

def calculate_economic_welfare(results_df, utility_tariff):
    """
    Calculates economic welfare based on consumer surplus + producer surplus.
    """
    # Consumer surplus = (utility tariff - p2p price) * quantity bought
    results_df['consumer_surplus'] = (utility_tariff - results_df['market_price']) * results_df['p2p_trade_volume']
    
     # Producer surplus = (p2p price - price floor) * quantity sold
    results_df['prosumer_profit'] = results_df['market_price'] * results_df['p2p_trade_volume']

    total_consumer_surplus = results_df.groupby('consumer_id')['consumer_surplus'].sum()
    total_prosumer_profit = results_df.groupby('prosumer_id')['prosumer_profit'].sum()

    total_system_benefit = total_consumer_surplus.sum() + total_prosumer_profit.sum()

    return {
        "total_system_benefit": total_system_benefit,
        "consumer_surplus_per_agent": total_consumer_surplus,
        "prosumer_profit_per_agent": total_prosumer_profit,
    }


# --- 4. Fairness Metrics ---
# --- Fairness metric: Gini index --------------------------------------------
# Gini is applied to distributions of benefits (or costs) to assess equity.

def calculate_gini_index(benefits):
    """
    Calculate the Gini coefficient of a list/array of benefits.

    Gini = 0 indicates perfect equality.
    Gini = 1 indicates maximum inequality.
    """
    if benefits.sum() == 0:
        return 0
    benefits = np.sort(np.array(benefits)) # shift to non-negative
    n = len(benefits)
    index = np.arange(1, n + 1)
    # Gini formula
    gini = (2 * np.sum(index * benefits) / (n * np.sum(benefits))) - (n + 1) / n
    return gini


# --- Fairness summaries ------------------------------------------------------
# Computes fairness-related summaries for consumers vs prosumers, typically reported as mean/median benefits and inequality measures.

def calculate_fairness_metrics(consumer_benefits, prosumer_benefits):
    """
    Returns fairness metrics including Gini for consumers vs prosumers.
    """
    all_benefits = pd.concat([consumer_benefits, prosumer_benefits]).fillna(0)
    gini_index = calculate_gini_index(all_benefits.values)
        total_consumer_surplus = consumer_benefits.sum()
    total_prosumer_profit = prosumer_benefits.sum()
    
    # Avoid division by zero if no prosumer profit
    c_to_p_ratio = total_consumer_surplus / total_prosumer_profit if total_prosumer_profit > 0 else np.inf
    
    return {
        "gini_index": gini_index,
        "consumer_to_prosumer_ratio": c_to_p_ratio,
    }


# --- 5. Technical Robustness Metrics ---
# --- Technical robustness ----------------------------------------------------
# Assesses network feasibility against feeder capacity (kW).

def calculate_technical_robustness(results_df, feeder_capacity_kw):
    """
    Calculates feeder overload frequency and severity[cite: 73, 77].
    """
    # Total P2P volume in each interval
    interval_trades_kw = results_df.groupby('interval')['p2p_trade_volume'].sum() * 4 # convert kWh back to kW for cap check
    
    #Add a small tolerance (e.g., 1e-9) to avoid floating-point errors
    overloaded_intervals = interval_trades_kw[interval_trades_kw > (feeder_capacity_kw + 1e-9)]
    
    num_total_intervals = len(results_df['interval'].unique())
    num_overloaded = len(overloaded_intervals)
    
    overload_frequency = (num_overloaded / num_total_intervals) * 100 if num_total_intervals > 0 else 0
    
    if num_overloaded > 0:
        overload_severity = ((overloaded_intervals - feeder_capacity_kw) / feeder_capacity_kw).mean() * 100
    else:
        overload_severity = 0
        
    return {
        "feeder_overload_frequency_percent": overload_frequency,
        "overload_severity_index_percent": overload_severity,
    }

# --- 6. Blockchain / Reconciliation Diagnostics ---
# --- Reconciliation diagnostics ---------------------------------------------
# Compares local simulation logs to on-chain (blockchain) logs.

def calculate_reconciliation_error(local_logs, blockchain_logs):
    """
    Calculates the discrepancy between local and blockchain records for S3b[cite: 74].
    The formula is |E_local - ΣE_chain| / E_local.
    """
    total_local = local_logs['volume_kwh'].sum()
    total_chain = blockchain_logs['volume_kwh'].sum()
    
    if total_local == 0:
        return 0.0 # No trades logged locally, so no error.
    
    error = (np.abs(total_local - total_chain) / total_local) * 100
    return error


# --- 7. Feeder Constraint Helper ---
# --- Feeder constraint scaling factor (alpha) -------------------------------
# Computes a scaling factor (alpha) based on feeder capacity and total interval energy. This is used where feeder constraints require proportional scaling oftrading/flows to maintain feasibility.

def compute_feeder_alpha(feeder_cap_kw: float, total_interval_kwh: float, interval_hours: float = 0.25) -> float:
    """
    Compute scaling factor alpha for feeder constraint.
    alpha = min(1, feeder_cap_kw / (total_interval_kwh / interval_hours))
    """
    if total_interval_kwh <= 0:
        return 0.0
    cap_kwh = feeder_cap_kw * interval_hours
    return min(1.0, cap_kwh / total_interval_kwh)

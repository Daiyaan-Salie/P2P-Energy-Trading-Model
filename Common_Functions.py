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

# ---------Profile generation functions-----------

def _gaussian(x, mu, sig, amplitude):
    """
    Helper function to create a Gaussian (bell curve).
    Used to model the morning and evening electricity demand peaks.
    """
    return amplitude * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))



def generate_household_profiles(num_households: int, num_intervals: int) -> np.ndarray:
    """
    Generates realistic, synthetic electricity consumption (load) profiles for
    South African households using a Gaussian model for peaks.

    This improved model creates a smoother and more representative daily profile by
    combining a base load with distinct morning and evening peaks, which is

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

    # Morning Peak (e.g., centered at 07:00)
    # 7 AM is interval 28 (7 * 4)
    morning_peak = _gaussian(x, mu=28, sig=4, amplitude=1.0) # sig=4 means peak is ~2h wide

    # Evening Peak (e.g., centered at 19:00)
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

def uniform_mcp(price_floor: float, utility_tariff: float) -> float:
    """Uniform market clearing price used by S2/S3a/S3b."""
    return (price_floor + utility_tariff) / 2.0

def kwh_to_kw(energy_kwh: float | np.ndarray, interval_minutes: int = 15):
    """Convert energy in a slot to average power over that slot."""
    return energy_kwh * (60.0 / interval_minutes)

def kw_to_kwh(power_kw: float | np.ndarray, interval_minutes: int = 15):
    """Convert average power over a slot to energy in that slot."""
    return power_kw * (interval_minutes / 60.0)


# --- 2. Metric Calculation Functions ---

def calculate_economic_welfare(results_df, utility_tariff):
    """
    Calculates consumer surplus, prosumer profit, and total system benefit.
    As defined in the 'Economic welfare' section of your proposal[cite: 60].
    """
    # Consumer surplus: Difference between what they would have paid the utility
    # and what they paid in the P2P market.
    results_df['consumer_surplus'] = (utility_tariff - results_df['market_price']) * results_df['p2p_trade_volume']
    
    # Prosumer profit: Revenue from P2P sales. Assumes marginal cost of solar is zero.
    results_df['prosumer_profit'] = results_df['market_price'] * results_df['p2p_trade_volume']

    total_consumer_surplus = results_df.groupby('consumer_id')['consumer_surplus'].sum()
    total_prosumer_profit = results_df.groupby('prosumer_id')['prosumer_profit'].sum()

    total_system_benefit = total_consumer_surplus.sum() + total_prosumer_profit.sum()

    return {
        "total_system_benefit": total_system_benefit,
        "consumer_surplus_per_agent": total_consumer_surplus,
        "prosumer_profit_per_agent": total_prosumer_profit,
    }

def calculate_gini_index(benefits):
    """
    Calculates the Gini index for a distribution of benefits.
    A value of 0 is perfect equality, 1 is perfect inequality[cite: 66].
    """
    if benefits.sum() == 0:
        return 0
    # Sort array and ensure it's a numpy array
    benefits = np.sort(np.array(benefits))
    n = len(benefits)
    index = np.arange(1, n + 1)
    # Gini formula
    gini = (2 * np.sum(index * benefits) / (n * np.sum(benefits))) - (n + 1) / n
    return gini

def calculate_fairness_metrics(consumer_benefits, prosumer_benefits):
    """
    Calculates the Gini index and consumer-to-prosumer benefit ratio[cite: 64, 69].
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


def compute_feeder_alpha(feeder_cap_kw: float, total_interval_kwh: float, interval_hours: float = 0.25) -> float:
    """
    Computes the scaling factor α so that total P2P trades
    in an interval don't exceed the feeder capacity.
    """
    if total_interval_kwh <= 0:
        return 0.0
    cap_kwh = feeder_cap_kw * interval_hours
    return min(1.0, cap_kwh / total_interval_kwh)

# S3b_Comparator.py
#
# UPDATED: Now imports and runs the new "S3b_hybrid_model"
# as Model 5 to compare it against the others.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time
from pathlib import Path

# --- Import common functions and config ---
import config
import Common_Functions as common

# --- Import the core simulation functions from each model ---
import S3b_model as model_standard
import S3b_model_Rev2 as model_rev2
import S3b_dynamic_MCP_model as model_dynamic
import S3b_fairshare_MCP_model as model_fairshare
import S3b_hybrid_model as model_hybrid # --- NEW ---

def process_results(model_name: str, results: dict) -> dict:
    """
    Calculates all key KPIs from a model's 'results' dictionary.
    """
    print(f"...processing {model_name}")
    trades_df = results.get("trades_df")
    
    if trades_df is None or len(trades_df) == 0:
        print(f"Warning: No trades found for {model_name}.")
        return {
            "Model": model_name,
            "Total System Benefit (ZAR)": 0.0,
            "Gini Index": np.nan,
            "Total P2P kWh Traded": 0.0,
            "Avg. MCP (ZAR/kWh)": np.nan,
            "MCP Volatility (Std. Dev.)": np.nan,
            "Total Grid Imports (kWh)": np.sum(results.get("imports_kwh", 0)),
            "Total Grid Exports (kWh)": np.sum(results.get("exports_kwh", 0)),
        }

    # 1. Economic KPIs
    welfare = common.calculate_economic_welfare(trades_df, config.UTILITY_TARIFF)
    fairness = common.calculate_fairness_metrics(
        welfare["consumer_surplus_per_agent"], 
        welfare["prosumer_profit_per_agent"]
    )
    
    # 2. Pricing KPIs
    mcp_series = results.get("mcp_series")
    if mcp_series is None and "trades_df" in results and not results["trades_df"].empty:
         mcp_series = results["trades_df"]["market_price"] # Fallback
    elif mcp_series is None:
        mcp_series = [np.nan]
            
    mcp_series = pd.Series(mcp_series).dropna()
    avg_mcp = mcp_series.mean() if not mcp_series.empty else np.nan
    std_mcp = mcp_series.std() if not mcp_series.empty else np.nan

    # 3. Technical KPIs
    total_p2p_kwh = trades_df["p2p_trade_volume"].sum()
    total_imports = np.sum(results["imports_kwh"])
    total_exports = np.sum(results["exports_kwh"])

    return {
        "Model": model_name,
        "Total System Benefit (ZAR)": welfare["total_system_benefit"],
        "Gini Index": fairness["gini_index"],
        "Total P2P kWh Traded": total_p2p_kwh,
        "Avg. MCP (ZAR/kWh)": avg_mcp,
        "MCP Volatility (Std. Dev.)": std_mcp,
        "Total Grid Imports (kWh)": total_imports,
        "Total Grid Exports (kWh)": total_exports,
    }

def run_all_models() -> tuple[pd.DataFrame, dict]:
    """
    Runs each model simulation and returns a DataFrame of KPIs
    and a dictionary of the full results for plotting.
    """
    all_kpis = []
    all_results = {}
    
    alpha_to_run = getattr(config, "FAIRNESS_ALPHA", 0.15)

    # --- Run Model 1: Standard (Noisy Baseline) ---
    print("Running Model 1: Standard (Noisy)...")
    res_std = model_standard.run_s3b_simulation()
    all_kpis.append(process_results("Standard (Noisy)", res_std))
    all_results["Standard (Noisy)"] = res_std

    # --- Run Model 2: Standard (Inelastic) ---
    print("\nRunning Model 2: Standard (Inelastic)...")
    res_rev2 = model_rev2.run_s3b_simulation()
    all_kpis.append(process_results("Standard (Inelastic)", res_rev2))
    all_results["Standard (Inelastic)"] = res_rev2

    # --- Run Model 3: Dynamic MCP (Strategic) ---
    print("\nRunning Model 3: Dynamic (Strategic)...")
    res_dyn = model_dynamic.run_sim()
    all_kpis.append(process_results("Dynamic (Strategic)", res_dyn))
    all_results["Dynamic (Strategic)"] = res_dyn

    # --- Run Model 4: Fair-share Auction ---
    print(f"\nRunning Model 4: Fair-share (Alpha={alpha_to_run})...")
    res_fair = model_fairshare.run_sim(ALPHA=alpha_to_run)
    all_kpis.append(process_results(f"Fair-share (Alpha={alpha_to_run})", res_fair))
    all_results[f"Fair-share (Alpha={alpha_to_run})"] = res_fair
    
    # --- Run Model 5: Hybrid Model (NEW) ---
    print(f"\nRunning Model 5: Hybrid (Strategic + Fair, Alpha={alpha_to_run})...")
    res_hybrid = model_hybrid.run_sim(ALPHA=alpha_to_run)
    all_kpis.append(process_results(f"Hybrid (Alpha={alpha_to_run})", res_hybrid))
    all_results[f"Hybrid (Alpha={alpha_to_run})"] = res_hybrid
    # --- END OF NEW SECTION ---
    
    # --- Finalize KPI DataFrame ---
    kpi_df = pd.DataFrame(all_kpis).set_index("Model")
    return kpi_df, all_results

# --------------------------------------------------------------------
# --- PLOTTING FUNCTIONS (All Unchanged) ---
# --------------------------------------------------------------------

def plot_kpi_tradeoff(kpi_df: pd.DataFrame):
    """
    Plots the core trade-off: Economic Efficiency (Benefit) vs.
    Distributive Fairness (Gini Index).
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Bar chart for Total System Benefit (Efficiency)
    color = 'tab:blue'
    ax1.set_title('S3c Comparison: Efficiency (Benefit) vs. Fairness (Gini)', fontsize=16, pad=20)
    ax1.set_xlabel('Simulation Model')
    ax1.set_ylabel('Total System Benefit (ZAR)', color=color, fontsize=12)
    bars = ax1.bar(kpi_df.index, kpi_df['Total System Benefit (ZAR)'], color=color, alpha=0.7, label='Total Benefit')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('ZAR {x:,.0f}'))
    ax1.set_xticklabels(kpi_df.index, rotation=15, ha='right') # Increased rotation
    ax1.bar_label(bars, fmt='ZAR {:,.0f}', padding=3)

    # Line chart for Gini Index (Fairness) on a secondary axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Gini Index (0 = Perfect Equality)', color=color, fontsize=12)
    ax2.plot(kpi_df.index, kpi_df['Gini Index'], color=color, marker='o', linestyle='--', linewidth=2, label='Gini Index')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, max(0.5, kpi_df['Gini Index'].max() * 1.2)) # Gini is 0-1, but often low

    fig.tight_layout()
    plt.savefig("S3c_kpi_tradeoff_benefit_vs_gini.png")
    print("...saved S3c_kpi_tradeoff_benefit_vs_gini.png")

def plot_price_volatility(all_results: dict):
    """
    Generates a box plot comparing the distribution of the
    Market Clearing Price (MCP) for all models.
    """
    price_data = {}
    for model_name, results in all_results.items():
        mcp_series = results.get("mcp_series")
        if mcp_series is None and "trades_df" in results and not results["trades_df"].empty:
             mcp_series = results["trades_df"]["market_price"] # Fallback
        elif mcp_series is None:
            mcp_series = [np.nan]
            
        price_data[model_name] = pd.Series(mcp_series).dropna()

    plt.figure(figsize=(12, 7))
    plt.boxplot(price_data.values(), labels=price_data.keys(), vert=True)
    plt.xticks(rotation=15, ha='right') # Added rotation
    plt.title('S3c Comparison: Price Volatility', fontsize=16)
    plt.ylabel('Market Clearing Price (ZAR/kWh)')
    plt.axhline(config.UTILITY_TARIFF, linestyle='--', color='red', label=f'Utility Tariff ({config.UTILITY_TARIFF})')
    plt.axhline(config.PRICE_FLOOR, linestyle='--', color='green', label=f'Price Floor ({config.PRICE_FLOOR})')
    plt.legend()
    plt.tight_layout()
    plt.savefig("S3c_price_volatility_boxplot.png")
    print("...saved S3c_price_volatility_boxplot.png")

def plot_energy_mix(kpi_df: pd.DataFrame):
    """
    Generates a stacked bar chart of the total community energy mix
    (P2P vs. Grid Imports vs. Grid Exports).
    """
    # Select the columns for the mix
    mix_data = kpi_df[['Total P2P kWh Traded', 'Total Grid Imports (kWh)', 'Total Grid Exports (kWh)']]
    
    ax = mix_data.plot(kind='bar', stacked=True, figsize=(12, 8), alpha=0.8)
    
    ax.set_title('S3c Comparison: Community Energy Mix', fontsize=16)
    ax.set_ylabel('Total Energy (kWh)')
    ax.set_xlabel('Simulation Model')
    ax.set_xticklabels(kpi_df.index, rotation=15, ha='right') # Increased rotation
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f} kWh'))
    ax.legend(title='Energy Source')
    
    plt.tight_layout()
    plt.savefig("S3c_community_energy_mix.png")
    print("...saved S3c_community_energy_mix.png")


# --------------------------------------------------------------------
# --- MAIN EXECUTION ---
# --------------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Run all simulations and get KPI data
    kpi_df, all_results = run_all_models()
    
    # 2. Print and save the KPI table
    print("\n\n" + "="*70)
    print("--- S3c Model Comparison KPI Summary ---")
    print("="*70)
    
    # Format for better readability
    pd.set_option('display.float_format', '{:,.2f}'.format)
    print(kpi_df)
    
    kpi_df.to_csv("S3c_model_comparison_kpis.csv")
    print("\n...KPI data saved to S3c_model_comparison_kpis.csv")

    # 3. Generate and save all comparison graphs
    print("\n--- Generating Comparison Graphs ---")
    plot_kpi_tradeoff(kpi_df)
    plot_price_volatility(all_results)
    plot_energy_mix(kpi_df)
    
    print("\n--- S3c Comparator Script Finished. ---")
    plt.close('all')
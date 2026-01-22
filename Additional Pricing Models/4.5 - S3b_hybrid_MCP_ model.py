# S3b_hybrid_model.py (NEW FILE)
#
# This is the new S3c Hybrid Model.
#
# It is designed to be the "best of both worlds":
# 1. It uses the *most efficient* bidding: `get_strategic_bidding_strategy`.
# 2. It uses the *most fair* allocation: `run_fair_auction_allocation` (with ALPHA).
#
# This will test if we can have maximum efficiency AND maximum fairness.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import config
import Common_Functions as common
from datetime import datetime

# --- NEW: Import *only* the STRATEGIC bidding strategy ---
from S3b_Auction_Engine import get_strategic_bidding_strategy

# --- We must include the "fair auction" logic here ---
# (This is copied from S3b_fairshare_MCP_model.py)

def run_fair_auction_allocation(t: int, agents_df: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, float, float]:
    """
    This is the "fair allocation" auction logic.
    It applies the FAIRNESS_ALPHA to the final trade allocation.
    """
    
    trades_log = []
    
    # 1. Separate Bids (Consumers) and Asks (Prosumers)
    bids = agents_df[agents_df['type'] == 'consumer'][['id', 'energy', 'price']]
    asks = agents_df[agents_df['type'] == 'prosumer'][['id', 'energy', 'price']]

    if bids.empty or asks.empty:
        return pd.DataFrame(), np.nan, 0.0

    # 2. Build Supply and Demand Curves
    bids_sorted = bids.sort_values(by='price', ascending=False)
    asks_sorted = asks.sort_values(by='price', ascending=True)

    # 3. Find the Market Clearing Price (MCP)
    possible_prices = sorted(pd.concat([bids_sorted['price'], asks_sorted['price']]).unique(), reverse=True)
    best_mcp = config.UTILITY_TARIFF
    max_cleared_kwh = -1.0

    for price in possible_prices:
        demand_at_price = bids_sorted[bids_sorted['price'] >= price]['energy'].sum()
        supply_at_price = asks_sorted[asks_sorted['price'] <= price]['energy'].sum()
        current_cleared_kwh = min(demand_at_price, supply_at_price)
        
        if current_cleared_kwh > max_cleared_kwh:
            max_cleared_kwh = current_cleared_kwh
            best_mcp = price
        elif current_cleared_kwh == max_cleared_kwh:
            best_mcp = price

    if max_cleared_kwh < 1e-12:
        return pd.DataFrame(), np.nan, 0.0

    mcp = best_mcp
    total_cleared_kwh = max_cleared_kwh

    # 4. Find Cleared Agents
    cleared_buyers = bids_sorted[bids_sorted['price'] >= mcp]
    cleared_sellers = asks_sorted[asks_sorted['price'] <= mcp]

    if cleared_buyers.empty or cleared_sellers.empty:
        return pd.DataFrame(), mcp, 0.0
    
    total_demand_cleared = cleared_buyers['energy'].sum()
    total_supply_cleared = cleared_sellers['energy'].sum()
    final_trade_volume = min(total_demand_cleared, total_supply_cleared)
    
    # Feeder Cap
    final_trade_volume_kw = final_trade_volume * 4.0
    if final_trade_volume_kw > (config.FEEDER_CAPACITY_KW + 1e-9):
        final_trade_volume = config.FEEDER_CAPACITY_KW / 4.0

    # --- 5. FAIR SHARE ALLOCATION (THE KEY DIFFERENCE) ---
    if final_trade_volume > 1e-12:
        num_cleared_buyers = len(cleared_buyers)
        num_cleared_sellers = len(cleared_sellers)
        
        pro_rata_volume = final_trade_volume * (1.0 - alpha)
        fair_share_volume = final_trade_volume * alpha
        
        fair_share_per_buyer = fair_share_volume / num_cleared_buyers if num_cleared_buyers > 0 else 0
        
        for b_idx, buyer in cleared_buyers.iterrows():
            pro_rata_buy = (buyer['energy'] / total_demand_cleared) * pro_rata_volume if total_demand_cleared > 0 else 0
            fair_share_buy = min(buyer['energy'] - pro_rata_buy, fair_share_per_buyer)
            
            buy_alloc = pro_rata_buy + fair_share_buy
            
            for s_idx, seller in cleared_sellers.iterrows():
                seller_frac = (seller['energy'] / total_supply_cleared) if total_supply_cleared > 0 else 0
                sell_alloc = seller_frac * buy_alloc

                if sell_alloc > 1e-12:
                    trades_log.append({
                        "interval": t,
                        "prosumer_id": seller['id'],
                        "consumer_id": buyer['id'],
                        "p2p_trade_volume": sell_alloc,
                        "market_price": mcp,
                    })

    trades_df = pd.DataFrame(trades_log)
    return trades_df, mcp, final_trade_volume


def run_sim(ALPHA: float = 0.15) -> dict:
    """
    Executes the S3b "Hybrid" simulation.
    """
    print(f"--- Running S3b: Hybrid Model (Strategic + Fair, ALPHA={ALPHA}) ---")
    
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
    
    all_trades = []
    mcp_log = np.full(config.NUM_INTERVALS, np.nan)

    # --- 3. Simulation Loop ---
    for t in range(config.NUM_INTERVALS):
        
        # 3.1) Net energy pre-battery/trade
        net_energy_kwh = generation_profiles_kwh[:, t] - load_profiles_kwh[:, t]
        
        # 3.2) Prosumers use battery for self-consumption
        battery_soc_kwh[:, t+1] = battery_soc_kwh[:, t]
        for i in range(num_prosumers):
            if net_energy_kwh[i] < 0:
                from_battery = min(abs(net_energy_kwh[i]), battery_soc_kwh[i, t+1])
                battery_soc_kwh[i, t+1] -= from_battery
                net_energy_kwh[i] += from_battery

        # 3.3) Build the agent list for the auction
        agents_data = []
        for i in range(num_prosumers):
            if net_energy_kwh[i] > 1e-12:
                agents_data.append({
                    'id': i, 'type': 'prosumer',
                    'energy': net_energy_kwh[i], 'soc_kwh': battery_soc_kwh[i, t+1]
                })
        for i in range(num_households):
            if net_energy_kwh[i] < -1e-12:
                agents_data.append({
                    'id': i, 'type': 'consumer',
                    'energy': abs(net_energy_kwh[i]), 'soc_kwh': 0
                })
        
        if not agents_data: continue
        agents_df = pd.DataFrame(agents_data)

        # 3.4) Apply the "Strategic" Bidding Strategy
        # --- THIS IS THE "EFFICIENT" PART ---
        agents_with_bids_df = get_strategic_bidding_strategy(agents_df)

        # 3.5) Run the "Fair Share" Auction
        # --- THIS IS THE "FAIR" PART ---
        trades_df, mcp, cleared_kwh = run_fair_auction_allocation(
            t, agents_with_bids_df, alpha=ALPHA
        )
        # --- -------------------------- ---
        
        mcp_log[t] = mcp

        if cleared_kwh > 0:
            all_trades.append(trades_df)
            
            # 3.6) Update net energy from trade results
            trades_by_seller = trades_df.groupby('prosumer_id')['p2p_trade_volume'].sum()
            for seller_id, sold_kwh in trades_by_seller.items():
                net_energy_kwh[int(seller_id)] -= sold_kwh
            
            trades_by_buyer = trades_df.groupby('consumer_id')['p2p_trade_volume'].sum()
            for buyer_id, bought_kwh in trades_by_buyer.items():
                net_energy_kwh[int(buyer_id)] += bought_kwh
        
        # 3.7) Post-trade: Battery charging & Grid Import/Export
        for i in range(num_households):
            if i < num_prosumers:
                if net_energy_kwh[i] > 1e-12:
                    to_store = min(net_energy_kwh[i], config.BATTERY_CAPACITY_KWH - battery_soc_kwh[i, t+1])
                    battery_soc_kwh[i, t+1] += to_store
                    remaining_surplus = net_energy_kwh[i] - to_store
                    if remaining_surplus > 0: grid_exports_kwh[i, t] = remaining_surplus
            if net_energy_kwh[i] < -1e-12:
                grid_imports_kwh[i, t] = abs(net_energy_kwh[i])

    print("Simulation complete.")
    
    final_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    return {
        "trades_df": final_trades_df, 
        "imports_kwh": grid_imports_kwh, 
        "exports_kwh": grid_exports_kwh, 
        "curtailment_kwh": curtailment_kwh, 
        "load_kwh": load_profiles_kwh, 
        "generation_kwh": generation_profiles_kwh, 
        "battery_soc_kwh": battery_soc_kwh,
        "mcp_series": mcp_log
    }


# --- Reporting & Plotting ---

def print_summary_report(results: dict, alpha: float):
    trades_df = results["trades_df"]
    if trades_df.empty:
        print("No trades occurred in this simulation.")
        return
        
    welfare = common.calculate_economic_welfare(trades_df, config.UTILITY_TARIFF)
    fairness = common.calculate_fairness_metrics(
        welfare["consumer_surplus_per_agent"], welfare["prosumer_profit_per_agent"]
    )
    
    print("\n" + "="*50)
    print(f"--- S3b Hybrid Model (ALPHA={alpha}) - Metrics Summary ---")
    print("="*50)
    print(f"Total System Benefit (P2P): ZAR {welfare['total_system_benefit']:,.2f}")
    print(f"Gini Index (P2P Benefits):  {fairness['gini_index']:.3f}")
    print(f"Total P2P Volume Traded:    {trades_df['p2p_trade_volume'].sum():,.2f} kWh")
    print("\n" + "="*50)

# --- Main execution (for testing) ---
if __name__ == "__main__":
    start_time = datetime.now()
    ALPHA_TO_RUN = getattr(config, "FAIRNESS_ALPHA", 0.15)
    
    sim_results = run_sim(ALPHA=ALPHA_TO_RUN)
    print_summary_report(sim_results, alpha=ALPHA_TO_RUN)
    
    end_time = datetime.now()
    print(f"\n--- S3b Hybrid Model run complete in {end_time - start_time} ---")
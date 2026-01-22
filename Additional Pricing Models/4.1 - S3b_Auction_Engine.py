# S3b_Auction_Engine.py
#
# CORRECTION: Replacing the failed "Strategic" (SoC) strategy
# with a new "Smarter" (Aggressive) strategy that will
# be less restrictive and create more trades.

import numpy as np
import pandas as pd
import config

def get_noisy_bidding_strategy(agents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy 1: "Noisy" / Uncoordinated (UNCHANGED)
    
    This is our "dumb" baseline.
    Both buyers and sellers pick a random price across the *entire*
    valid market range (FiT Rate to Utility Tariff).
    This creates many "failed" trades.
    """
    price_low = config.UTILITY_FIT_RATE
    price_high = config.UTILITY_TARIFF

    # Prosumers (Sellers)
    prosumer_mask = agents_df['type'] == 'prosumer'
    prosumer_count = prosumer_mask.sum()
    if prosumer_count > 0:
        agents_df.loc[prosumer_mask, 'price'] = np.random.uniform(
            price_low, price_high, prosumer_count
        )

    # Consumers (Buyers)
    consumer_mask = agents_df['type'] == 'consumer'
    consumer_count = consumer_mask.sum()
    if consumer_count > 0:
        agents_df.loc[consumer_mask, 'price'] = np.random.uniform(
            price_low, price_high, consumer_count
        )
    
    return agents_df


def get_strategic_bidding_strategy(agents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy 2: "Smarter" / Coordinated Strategy (NEW)
    
    This is our "smart" model. Agents bid aggressively
    to ensure their trade clears, creating a large overlap.
    
    - Sellers bid in the *lower half* of the range.
    - Buyers bid in the *upper half* of the range.
    
    This will be *less restrictive* and result in *more trades*
    than the "Noisy" model.
    """
    
    price_low = config.UTILITY_FIT_RATE
    price_high = config.UTILITY_TARIFF
    midpoint = (price_low + price_high) / 2.0
    
    # --- Buyer Strategy (Aggressive) ---
    consumer_mask = agents_df['type'] == 'consumer'
    consumer_count = consumer_mask.sum()
    if consumer_count > 0:
        # Smart buyers bid in the *upper half*
        agents_df.loc[consumer_mask, 'price'] = np.random.uniform(
            midpoint, price_high, consumer_count
        )

    # --- Seller Strategy (Aggressive) ---
    prosumer_mask = agents_df['type'] == 'prosumer'
    prosumer_count = prosumer_mask.sum()
    if prosumer_count > 0:
        # Smart sellers bid in the *lower half*
        agents_df.loc[prosumer_mask, 'price'] = np.random.uniform(
            price_low, midpoint, prosumer_count
        )

    return agents_df


def run_double_sided_auction(t: int, agents_df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """
    Runs a uniform-price, double-sided auction.
    
    (This function is correct and does not need to be changed.)
    """
    
    trades_log = []
    
    # 1. Separate Bids (Consumers) and Asks (Prosumers)
    bids = agents_df[agents_df['type'] == 'consumer'][['id', 'energy', 'price']]
    asks = agents_df[agents_df['type'] == 'prosumer'][['id', 'energy', 'price']]

    if bids.empty or asks.empty:
        return pd.DataFrame(), np.nan, 0.0 # No market possible

    # 2. Build Supply and Demand Curves
    bids_sorted = bids.sort_values(by='price', ascending=False)
    asks_sorted = asks.sort_values(by='price', ascending=True)

    # 3. Find the Market Clearing Price (MCP)
    possible_prices = sorted(pd.concat([bids_sorted['price'], asks_sorted['price']]).unique(), reverse=True)
    best_mcp = np.nan
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

    # 4. Allocate Trades at the single MCP
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

    # 5. Pro-rata allocation among all *cleared* agents
    if final_trade_volume > 1e-12:
        for b_idx, buyer in cleared_buyers.iterrows():
            buy_alloc = (buyer['energy'] / total_demand_cleared) * final_trade_volume
            
            for s_idx, seller in cleared_sellers.iterrows():
                sell_alloc = (seller['energy'] / total_supply_cleared) * buy_alloc
                
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
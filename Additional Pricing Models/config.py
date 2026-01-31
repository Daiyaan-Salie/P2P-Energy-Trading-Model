# config.py

# This file contains all the core parameters for the simulation,
# allowing for easy adjustments from a central location.

# --- Simulation Parameters ---
NUM_INTERVALS = 96 # 96 intervals per day (15-min each) 
TOTAL_HOUSEHOLDS = 30    # Total number of households in the microgrid
PROSUMER_SHARE = 0.50    # The percentage of households that are prosumers (50%)

# --- Economic Parameters ---
# As per your methodology, prosumers bid at a floor price and consumers
# bid at the utility tariff, creating the spread for P2P trades.
PRICE_FLOOR = 0.85       # ZAR per kWh, the lowest price prosumers will sell for
UTILITY_TARIFF = 3.00    # ZAR per kWh, the fixed price from Eskom (can be varied in sensitivity analysis)
UTILITY_FIT_RATE = PRICE_FLOOR  # ZAR per kWh, the price the utility buys surplus energy for

# --- Prosumer & Storage Parameters ---
PEAK_PV_GENERATION_KW = 5 # Peak power output for a prosumer's PV system
BATTERY_CAPACITY_KWH = 10.0 # Total capacity of a prosumer's battery
BATTERY_CHARGE_THRESHOLD = 0.90 # SoC threshold to prioritize selling over storing (90%)

# --- Technical Parameters ---
#In config.py, leave FEEDER_CAPACITY_MODE="global" to keep your current single-cap behaviour.
#To enable per-feeder caps, set:
FEEDER_CAPACITY_KW = 5 # Capacity of the local feeder line in kW
FEEDER_CAPACITY_MODE = "global"
FEEDER_APPLY_ON = "seller"  # or "buyer"
FEEDER_MAP = { **{i: "A" for i in range(0, 15)}, **{i: "B" for i in range(15, 30)} }
FEEDER_LIMITS_KW = {"A": 50.0, "B": 50.0}

KWH_DECIMALS = 3           # kWh * 1000 (Wh units)
ZAR_DECIMALS = 4           # << four decimals
PRICE_SCALE_MILLI = 1000   # mZAR/kWh

# Convert qty_units * mcp_milli -> ZAR token units
# PAYMENT_DIVISOR = (10^KWH * 10^price) / 10^ZAR
PAYMENT_DIVISOR = (10**KWH_DECIMALS) * PRICE_SCALE_MILLI // (10**ZAR_DECIMALS)
# for (3,4): 1000*1000/10000 = 100

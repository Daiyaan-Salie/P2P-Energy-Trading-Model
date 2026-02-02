"""
config.py — Central configuration for all simulation models (S1 / S2 / S3)

Purpose
-------
This module defines *all* global parameters used across the project so that:
1) experiments are reproducible (single source of truth),
2) sensitivity analysis is easy (change values here),
3) Units/assumptions are explicit for examiners.

Key global assumptions (aligned with thesis methodology)
-------------------------------------------------------
- Time is discretised into 15-minute intervals. i.e. 96 intervals per 24-hour horizon.
- Energy quantities are expressed per interval in kWh unless stated otherwise.
- Power limits (e.g., feeder capacity) are specified in kW and can be converted
  to kWh per interval by multiplying by (interval_minutes / 60).

Settlement/integer arithmetic
-------------------------------
Where on-chain style settlement is emulated (or where deterministic accounting is needed),
I avoid floating-point currency arithmetic by using integer scaling for:
- energy quantities (kWh -> integer "units"), and
- price representation (ZAR/kWh -> milli-ZAR/kWh), and
- token denomination (ZAR token supports ZAR_DECIMALS).
"""


# --- Simulation Parameters ---
NUM_INTERVALS = 96 # 96 intervals per day (15-min each) 
TOTAL_HOUSEHOLDS = 30    # Total number of households in the microgrid
PROSUMER_SHARE = 0.50    # The percentage of households that are prosumers (50%)

# --- Economic Parameters ---
# Trading bounds used throughout the thesis:
# - Prosumers are willing to sell at/above PRICE_FLOOR (FIT/minimum acceptable price)
# - Consumers are willing to pay up to UTILITY_TARIFF (retail tariff)
#
# Units: ZAR per kWh.
PRICE_FLOOR = 0.85       # ZAR per kWh, the lowest price prosumers will sell surplus energy for.
UTILITY_TARIFF = 3.00    # ZAR per kWh, the fixed price from the utility.
UTILITY_FIT_RATE = PRICE_FLOOR  # ZAR per kWh, the price the utility buys surplus energy from prosumers.


# --- Prosumer & Storage Parameters ---
PEAK_PV_GENERATION_KW = 5 # Peak power output from a prosumer's PV system
BATTERY_CAPACITY_KWH = 10.0 # Total capacity of a prosumer's battery
BATTERY_CHARGE_THRESHOLD = 0.90 # SoC threshold to prioritise selling over storing (90%)

# --- Technical Parameters ---
#In config.py, leave FEEDER_CAPACITY_MODE="global" to keep the current single-cap behaviour.
#To enable per-feeder caps, set:
FEEDER_CAPACITY_KW = 5 # Capacity of the local feeder line in kW

# Capacity mode:
# - "global": single feeder cap applied across the entire microgrid (current behaviour).
# - (optional extension) per-feeder caps: if enabled, FEEDER_MAP and FEEDER_LIMITS_KW define topology.
FEEDER_CAPACITY_MODE = "global"
FEEDER_APPLY_ON = "seller"  # or "buyer"
FEEDER_MAP = { **{i: "A" for i in range(0, 15)}, **{i: "B" for i in range(15, 30)} }
FEEDER_LIMITS_KW = {"A": 50.0, "B": 50.0}




# ------Integer Scaling for Deterministic / Token-like Settlement-------
KWH_DECIMALS = 3           # kWh * 1000 (Wh units)
ZAR_DECIMALS = 4           # currency token scaling: ZAR * 10^ZAR_DECIMALS

# Price scaling:
# Price is represented in milli-ZAR per kWh to keep price arithmetic integer-friendly.
# Example: 2.000 ZAR/kWh => 2000 mZAR/kWh.
PRICE_SCALE_MILLI = 1000   # mZAR/kWh


PAYMENT_DIVISOR = (10**KWH_DECIMALS) * PRICE_SCALE_MILLI // (10**ZAR_DECIMALS)
# for (3,4): 1000*1000/10000 = 100

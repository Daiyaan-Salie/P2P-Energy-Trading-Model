# Model S3: Market Setup Utility (Rev5)
#
# Purpose:
# --------
# This script performs the *initial on-chain configuration* of the deployed
# S3c DEX smart contract by calling the `setup_market` ABI method.
#
# It is a **creator-only administrative operation** and must be executed
# exactly once per deployed application instance.
#
# The script:
# - Reads deployment configuration from `.env`
# - Loads pricing and feeder constraints from `config.py`
# - Converts all values into on-chain compatible units
# - Executes a single atomic ABI method call to initialise the market
#
# On-chain parameters configured:
# -------------------------------
# - ZAR settlement ASA
# - kWh energy ASA
# - Price floor (micro-ZAR / kWh)
# - Utility tariff cap (micro-ZAR / kWh)
# - Feeder capacity constraint (0.01 kWh units per interval)
#
# Safety Notes:
# -------------
# - This script **MODIFIES on-chain state**
# - Must be executed by the DEX application creator
# - Should only be run after:
#     1) App deployment
#     2) ASA creation
#     3) Deployer sanity checks (see S3_deployer_address.py)
#
# Typical Use Cases:
# ------------------
# - One-time market initialisation after deployment
# - Re-deployment scenarios (new App ID)
# - Controlled reconfiguration during development
#
# =============================================================================

# -----------------------------
# Imports
# -----------------------------
import os
import json
from algosdk.v2client import algod
from algosdk import account, mnemonic
from dotenv import load_dotenv
from algosdk.atomic_transaction_composer import (
    AtomicTransactionComposer,
    AccountTransactionSigner,
)
from algosdk.abi import Contract  # use Contract/Method from algosdk.abi

import config  # required for PRICE_FLOOR, UTILITY_TARIFF, FEEDER_CAPACITY_KW


# --- Helper Functions ---
def get_algod_client():
    load_dotenv()
    addr = os.getenv("ALGOD_ADDRESS")
    token = os.getenv("ALGOD_TOKEN")
    if not addr:
        raise RuntimeError("ALGOD_ADDRESS not set in .env")
    if token is None:
        raise RuntimeError("ALGOD_TOKEN not set in .env (set to empty string for public endpoints)")
    return algod.AlgodClient(token, addr)


def get_deployer_account():
    """Use unified COORDINATOR_MNEMONIC from .env as the deployer key."""
    load_dotenv()
    mn = os.getenv("COORDINATOR_MNEMONIC")
    if not mn:
        raise RuntimeError("COORDINATOR_MNEMONIC not set in .env")
    pk = mnemonic.to_private_key(mn)
    addr = account.address_from_private_key(pk)
    return pk, addr


def to_micro_zar_per_kwh(zar_per_kwh: float) -> int:
    """Convert ZAR/kWh (float) -> micro-ZAR/kWh (int)."""
    return int(round(float(zar_per_kwh) * 1_000_000.0))


def feeder_cap_units_001kwh(feeder_kw: float, interval_minutes: int = 15) -> int:
    """
    Convert feeder kW limit to 0.01 kWh units per interval.
      cap_kwh = feeder_kw * (interval_minutes / 60)
      units = cap_kwh * 100  (since 1 unit = 0.01 kWh)
    """
    cap_kwh = float(feeder_kw) * (float(interval_minutes) / 60.0)
    return int(round(cap_kwh * 100.0))


# -----------------------------
# Main Market Setup Logic
# -----------------------------
def setup_market():

    """
    Execute the creator-only `setup_market` ABI call to initialise
    the S3 DEX application with economic and network constraints.
    """
    print("--- ⚙️  Initializing deployed market (setup_market for Rev5) ---")

    # -----------------------------
    # 1) Load environment variables
    # -----------------------------
    load_dotenv()
    app_id = int(os.getenv("DEX_APP_ID", "0"))
    if not app_id:
        raise RuntimeError("DEX_APP_ID not set in .env (run deploy script first)")

    zar_token_id = int(os.getenv("ZAR_ASA_ID", "0"))
    kwh_token_id = int(os.getenv("KWH_ASA_ID", "0"))
    if not (zar_token_id and kwh_token_id):
        raise RuntimeError("ZAR_ASA_ID/KWH_ASA_ID not set in .env (run create_assets script first)")

    # -----------------------------
    # 2) Load ABI definition
    # -----------------------------
    abi_path = os.getenv("DEX_ABI_PATH", "abi_rev5.json")
    if not os.path.exists(abi_path):
        raise FileNotFoundError(
            f"ABI file not found: {abi_path}. Make sure DEX_ABI_PATH is set correctly in .env"
        )

    with open(abi_path, "r", encoding="utf-8") as f:
        abi_json = json.load(f)

    # -----------------------------
    # 3) Compute on-chain parameters
    # -----------------------------
    floor_micro = to_micro_zar_per_kwh(config.PRICE_FLOOR)
    tariff_micro = to_micro_zar_per_kwh(config.UTILITY_TARIFF)
    feeder_cap_units = feeder_cap_units_001kwh(config.FEEDER_CAPACITY_KW, interval_minutes=15)

    if tariff_micro < floor_micro:
        raise ValueError(
            f"UTILITY_TARIFF must be >= PRICE_FLOOR. Got {config.UTILITY_TARIFF} < {config.PRICE_FLOOR}"
        )
    if feeder_cap_units <= 0:
        raise ValueError(
            f"Computed feeder_cap_units must be > 0. Check FEEDER_CAPACITY_KW={config.FEEDER_CAPACITY_KW}"
        )

    print(f"Using: floor_micro={floor_micro} microZAR/kWh, tariff_micro={tariff_micro} microZAR/kWh")
    print(f"Using: feeder_cap_units={feeder_cap_units} (0.01 kWh units per 15-min interval)")

    # -----------------------------
    # 4) Client and signer setup
    # -----------------------------
    client = get_algod_client()
    deployer_pk, deployer_addr = get_deployer_account()
    signer = AccountTransactionSigner(deployer_pk)

   # -----------------------------
    # 5) Build ABI method call
    # -----------------------------
    atc = AtomicTransactionComposer()
    contract = Contract.from_json(json.dumps(abi_json))
    setup_method = contract.get_method_by_name("setup_market")

    sp = client.suggested_params()

    atc.add_method_call(
        app_id=app_id,
        method=setup_method,
        sender=deployer_addr,
        sp=sp,
        signer=signer,
        method_args=[
            kwh_token_id,     # kwh_asset (abi.Asset)
            zar_token_id,     # zar_asset (abi.Asset)
            floor_micro,      # floor_micro (uint64) micro-ZAR/kWh
            tariff_micro,     # tariff_micro (uint64) micro-ZAR/kWh
            feeder_cap_units  # feeder_cap_units (uint64) 0.01 kWh units per interval
        ],
        # For 'asset' ABI types, the ASA IDs must be in foreign_assets:
        foreign_assets=[kwh_token_id, zar_token_id],
    )

    # -----------------------------
    # 6) Execute transaction
    # -----------------------------
    try:
        result = atc.execute(client, 4)
        print("✅ Market setup transaction successful!")
        print(f"   - Tx ID: {result.tx_ids[0]}")
        print(f"   - Contract returned: '{result.abi_results[0].return_value}'")
    except Exception as e:
        print(f"❌ Market setup failed: {e}")
        raise


if __name__ == "__main__":
    setup_market()


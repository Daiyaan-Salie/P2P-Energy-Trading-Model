# Model S3: Deployer Address Sanity Check Utility
#
# Purpose:
# --------
# This script performs a *read-only* validation of the current deployment
# configuration defined in the local `.env` file.
#
# It is intended as a safety and debugging tool to ensure that:
# - The coordinator mnemonic resolves to the expected Algorand address
# - The address holds sufficient ALGO balance
# - The address is opted-in to the required project ASAs (ZAR and kWh)
# - The address matches the creator of the deployed DEX application
#
# No state is modified on-chain by this script.
#
# Typical use cases:
# ------------------
# - Pre-deployment sanity checks
# - Debugging failed creator-only contract calls
# - Verifying environment configuration before running S3 scripts
# =============================================================================

# -----------------------------
# Imports
# -----------------------------
import os
import sys
from dotenv import load_dotenv
from algosdk.v2client import algod
from algosdk import mnemonic, account

def env(name: str, allow_empty: bool = False) -> str:
    v = os.getenv(name)
    if v is None:
        sys.exit(f"[ERR] Missing env var: {name}")
    if not allow_empty and v.strip() == "":
        sys.exit(f"[ERR] Empty env var not allowed: {name}")
    return v.strip()

# -----------------------------
# Main logic
# -----------------------------
# Performs all read-only checks and prints results to stdout.

def main():
    load_dotenv()

    algod_address = env("ALGOD_ADDRESS")
    algod_token = os.getenv("ALGOD_TOKEN", "").strip()  # may be empty for algonode

    coord_mn = env("COORDINATOR_MNEMONIC")
    zar_asa_id = int(env("ZAR_ASA_ID"))
    kwh_asa_id = int(env("KWH_ASA_ID"))
    dex_app_id = int(env("DEX_APP_ID"))

    # Derive coordinator keys
    pk = mnemonic.to_private_key(coord_mn)
    addr = account.address_from_private_key(pk)

    # Client
    client = algod.AlgodClient(algod_token, algod_address)

    # Account info
    info = client.account_info(addr)
    algo_balance = info.get("amount", 0) / 1e6

    print("🔹 Coordinator Address:", addr)
    print("💰 ALGO Balance:", algo_balance, "ALGO")

    # -----------------------------
    # Asset opt-in inspection
    # -----------------------------
    # Build a mapping of asset_id -> balance for all opted-in ASAs.
    assets = {a["asset-id"]: a.get("amount", 0) for a in info.get("assets", [])}

    print("\n🔹 Opted-in Assets:")
    if not assets:
        print("  (none found)")
    else:
        for aid in sorted(assets.keys()):
            print(f"  - Asset ID {aid}: balance = {assets[aid]}")

    # -----------------------------
    # Project ASA validation
    # -----------------------------
    # Ensures the coordinator is opted-in to the ZAR and kWh ASAs
    # required for S3 settlement flows.
    
    print("\n🔹 Validation (ASAs from .env):")
    for aid, name in [(zar_asa_id, "ZAR_ASA_ID"), (kwh_asa_id, "KWH_ASA_ID")]:
        if aid in assets:
            print(f"✅ {name} ({aid}) found with balance {assets[aid]}")
        else:
            print(f"⚠️ {name} ({aid}) not found (needs opt-in)")

    # -----------------------------
    # Smart contract creator check
    # -----------------------------
    # Validates that the coordinator address matches the creator
    # of the deployed DEX application.
    #
    # This is critical because many contract methods are
    # creator-only (oracle/admin role).
    print("\n🔹 Validation (DEX_APP_ID creator check):")
    try:
        app = client.application_info(dex_app_id)
        creator = app["params"]["creator"]
        print(f"DEX_APP_ID={dex_app_id}")
        print("App creator:", creator)
        if creator == addr:
            print("✅ Coordinator mnemonic MATCHES app creator (creator-only calls will work).")
        else:
            print("❌ Coordinator mnemonic DOES NOT match app creator (creator-only calls will fail).")
    except Exception as e:
        print(f"⚠️ Could not fetch app info for DEX_APP_ID={dex_app_id}: {e}")

# -----------------------------
# Entry point
# -----------------------------
# Allows the script to be executed directly:
#   python S3c_deployer_address.py
if __name__ == "__main__":
    main()

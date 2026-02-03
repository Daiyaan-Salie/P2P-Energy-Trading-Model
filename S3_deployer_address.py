# S3c_deployer_address.py
#
# Read-only sanity check for the *current* deployment defined in .env:
# - Derive coordinator address from COORDINATOR_MNEMONIC
# - Print ALGO balance
# - List opted-in assets (ASA balances)
# - Validate opt-in for ZAR_ASA_ID and KWH_ASA_ID (from .env)
# - Check whether this address is the creator of DEX_APP_ID (from .env)

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

    # Assets
    assets = {a["asset-id"]: a.get("amount", 0) for a in info.get("assets", [])}

    print("\n🔹 Opted-in Assets:")
    if not assets:
        print("  (none found)")
    else:
        for aid in sorted(assets.keys()):
            print(f"  - Asset ID {aid}: balance = {assets[aid]}")

    # Validate project ASAs
    print("\n🔹 Validation (ASAs from .env):")
    for aid, name in [(zar_asa_id, "ZAR_ASA_ID"), (kwh_asa_id, "KWH_ASA_ID")]:
        if aid in assets:
            print(f"✅ {name} ({aid}) found with balance {assets[aid]}")
        else:
            print(f"⚠️ {name} ({aid}) not found (needs opt-in)")

    # Check app creator
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

if __name__ == "__main__":
    main()

#Model S3: Create ASAs
#
# Purpose:
# --------
# This script creates the two Algorand Standard Assets (ASAs) used by the S3C system:
# 1) ZAR Stable Token (fiat-like token used for settlement)
# 2) kWh Energy Token (energy quantity representation)
#
# Precision rationale:
# --------------------
# - ZAR uses stablecoin-style decimals=6 (micro-ZAR) to reduce rounding issues in transfers
#   and on-chain accounting (common practice for major fiat stablecoins).
# - kWh uses decimals=3 (milli-kWh) to match prior energy token handling.
#
# Output:
# -------
# Prints .env-ready values:
# - ZAR_ASA_ID=<id>
# - KWH_ASA_ID=<id>
# =============================================================================

# -----------------------------
# Imports
# -----------------------------
import os

# Loads environment variables from a local .env file
from dotenv import load_dotenv

# Algod client for network interaction + SDK utilities for keys and transactions
from algosdk.v2client import algod
from algosdk import account, mnemonic, transaction

# -----------------------------
# Network / account helpers
# -----------------------------
# These helper functions provide:
# - algod client configuration
# - deployer (coordinator) account retrieval
# - transaction confirmation waiting

def get_algod_client():
    load_dotenv()
  # Required connection details
    addr = os.getenv("ALGOD_ADDRESS")
    token = os.getenv("ALGOD_TOKEN")

  # Fail fast if missing configuration
    if not addr:
        raise RuntimeError("ALGOD_ADDRESS not set in .env")
    if token is None:
        # Note: token may be an empty string for public endpoints,
        # but must still exist as a key in the .env
        raise RuntimeError("ALGOD_TOKEN not set in .env (set to empty string for public endpoints)")
    # Create and return Algod client
    return algod.AlgodClient(token, addr)


def get_deployer_account():
  # Load .env variables into process environment
    load_dotenv()

  # The coordinator/deployer mnemonic is used to sign asset creation transactions
    mn = os.getenv("COORDINATOR_MNEMONIC")
    if not mn:
        raise RuntimeError("COORDINATOR_MNEMONIC not set in .env")
  # Derive private key + address from mnemonic
    sk = mnemonic.to_private_key(mn)
    addr = account.address_from_private_key(sk)
    return sk, addr


def wait_for_confirmation(client, txid):
  # Polls Algod until the transaction has a confirmed round
    last_round = client.status().get("last-round")
    txinfo = client.pending_transaction_info(txid)

  # Keep waiting until confirmed-round is present and > 0
    while not (txinfo.get("confirmed-round") and txinfo.get("confirmed-round") > 0):
        print("Waiting for confirmation...")
        last_round += 1
      # Block until next round to avoid hammering the node
        client.status_after_block(last_round)
        txinfo = client.pending_transaction_info(txid)
    print(f"Transaction {txid} confirmed in round {txinfo.get('confirmed-round')}.")
    return txinfo


# -----------------------------
# Asset creation function
# -----------------------------
# Creates an ASA with standard management addresses (manager/reserve/freeze/clawback)
# all set to the deployer account (simple baseline setup).

def create_asset(asset_name: str, unit_name: str, decimals: int, total_supply: int) -> int:
  # Step 1: Connect to network and load deployer keypair
    client = get_algod_client()
    sk, addr = get_deployer_account()

  # Step 2: Get suggested params (fees, valid rounds, genesis, etc.)
    params = client.suggested_params()

  # Step 3: Build the AssetConfigTxn (ASA creation transaction)
    # Note: total_supply is specified in base units defined by `decimals`
    txn = transaction.AssetConfigTxn(
        sender=addr,
        sp=params,
        total=total_supply,
        decimals=decimals,
        default_frozen=False,
        unit_name=unit_name,
        asset_name=asset_name,
      # Management addresses: using deployer for all roles (simple admin model)
        manager=addr,
        reserve=addr,
        freeze=addr,
        clawback=addr,
    )

     # Step 4: Sign and submit
    signed = txn.sign(sk)
    txid = client.send_transaction(signed)
    print(f"Sent: {txid}")

    # Step 5: Wait for confirmation and return created asset ID
    info = wait_for_confirmation(client, txid)
    return info["asset-index"]

# -----------------------------
# Main execution
# -----------------------------
# Creates two assets and prints IDs in .env-ready format.

if __name__ == "__main__":
    print("\n--- Creating ZAR token (stablecoin-style: decimals=6; micro-ZAR) ---")
    # Total supply is in base units (micro-ZAR when decimals=6).
    # Example:
    # - If you want 10,000,000,000 ZAR minted,
    # - and decimals=6 means 1 ZAR = 1,000,000 micro-ZAR,
    # - then total base units = 10,000,000,000 * 1,000,000
    zar_total_micro = 10_000_000_000 * 1_000_000

  # Create ZAR ASA
    zar_id = create_asset(
        asset_name="ZAR Stable Token",
        unit_name="ZAR",
        decimals=6,
        total_supply=zar_total_micro,
    )
    print(f"✅ ZAR ASA created: {zar_id}")

    # kWh token supply is in milli-kWh when decimals=3.
    # Example:
    # - If you want 10,000,000,000 kWh minted,
    # - and decimals=3 means 1 kWh = 1,000 milli-kWh,
    # - then total base units = 10,000,000,000 * 1,000

    print("\n--- Creating kWh token (decimals=3; milli-kWh) ---")
    # Example: 10 billion kWh => 10,000,000,000 * 1,000 = 10,000,000,000,000 milli-kWh
    kwh_total_milli = 10_000_000_000 * 1_000
    kwh_id = create_asset(
        asset_name="kWh Energy Token",
        unit_name="KWHT",
        decimals=3,
        total_supply=kwh_total_milli,
    )
    print(f"✅ KWH ASA created: {kwh_id}")

    print("\n--- 📝 Add to your .env ---")
    print(f"ZAR_ASA_ID={zar_id}")
    print(f"KWH_ASA_ID={kwh_id}")

    print("\nNOTE:")
    print("- ZAR base units are now micro-ZAR (1e-6).")
    print("- This requires updating price math in the contract + live simulation (next scripts).")

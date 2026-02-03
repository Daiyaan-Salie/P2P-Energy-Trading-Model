# S3c_deploy_contract_Rev5_fixed.py
# Minimal update of the previous Rev4 deploy script to support Rev5 artifacts + schema.

# =============================================================================
# Model S3: Deploy Smart Contract (Rev5)
#
# Purpose:
# --------
# This script deploys the compiled Rev5 PyTeal smart contract to Algorand.
#
# What it does (high-level):
# --------------------------
# 1) Loads environment variables (Algod endpoint + deployer mnemonic)
# 2) Reads Rev5 TEAL source files (approval + clear)
# 3) Compiles TEAL to bytecode via algod.compile()
# 4) Creates the application with the correct global/local state schema
# 5) Prints the resulting App ID and .env fields to update
#
# Key dependency:
# ---------------
# TEAL artifacts must already exist:
# - approval_rev5.teal
# - clear_rev5.teal
#
# These are produced by running the PyTeal compile script:
# - S3c_pyteal_dex_Rev5.py
# =============================================================================

# -----------------------------
# Imports
# -----------------------------
import os
import base64
from pathlib import Path

# Algorand SDK imports for client + account + transaction builders
from algosdk.v2client import algod
from algosdk import account, mnemonic, transaction

# .env loader for local configuration
from dotenv import load_dotenv

# -----------------------------
# Helper Functions (same structure as Rev4)
# -----------------------------
# These helpers centralize:
# - consistent .env loading (from script folder)
# - algod client creation
# - deployer account derivation from mnemonic
# - transaction confirmation waiting

def load_env():
    # Load .env from the same folder as this script (prevents picking up old .env elsewhere)
    dotenv_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    return dotenv_path

def get_algod_client():
    load_env()
    addr = os.getenv("ALGOD_ADDRESS")
    token = os.getenv("ALGOD_TOKEN", "")
    if not addr:
        raise RuntimeError("ALGOD_ADDRESS not set in .env")
    # ALGOD_TOKEN may be empty for algonode
    return algod.AlgodClient(token, addr)

def get_deployer_account():
    load_env()
    # Mnemonic used to sign the deployment transaction
    mn = os.getenv("COORDINATOR_MNEMONIC")
    if not mn:
        raise RuntimeError("COORDINATOR_MNEMONIC not set in .env")
    # Convert mnemonic -> private key -> address
    pk = mnemonic.to_private_key(mn)
    addr = account.address_from_private_key(pk)
    return pk, addr

def wait_for_confirmation(client, txid):
    last_round = client.status().get("last-round")
    txinfo = client.pending_transaction_info(txid)

    # Keep waiting until "confirmed-round" exists and is > 0
    while not (txinfo.get("confirmed-round") and txinfo.get("confirmed-round") > 0):
        print("Waiting for confirmation...")
        last_round += 1
        # Block until the next round to avoid busy-looping too fast
        client.status_after_block(last_round)
        txinfo = client.pending_transaction_info(txid)
    print(f"Transaction {txid} confirmed in round {txinfo.get('confirmed-round')}.")
    return txinfo

# -----------------------------
# Main Deployment Logic
# -----------------------------
# This performs the actual app deployment (create transaction).
def deploy_contract():
    print("--- 🚀 Deploying Smart Contract (Rev5) ---")

    # Step 1: Load .env and display where it was loaded from
    dotenv_path = load_env()
    print(f"Loaded .env from: {dotenv_path}")

    # Step 2: Create Algod client and load deployer identity
    client = get_algod_client()
    deployer_pk, deployer_address = get_deployer_account()
    print(f"Deployer address: {deployer_address}")

    # Step 3: Define TEAL filenames expected for Rev5
    approval_file = "approval_rev5.teal"
    clear_file = "clear_rev5.teal"

    # Step 4: Ensure TEAL artifacts exist (guard against deploying stale/missing build output)
    if not os.path.exists(approval_file) or not os.path.exists(clear_file):
        raise FileNotFoundError(
            f"Missing TEAL files. Run S3c_pyteal_dex_Rev5.py to generate "
            f"'{approval_file}' and '{clear_file}'."
        )

    # Step 5: Read TEAL source into memory (text form)
    with open(approval_file, "r", encoding="utf-8") as f:
        approval_source = f.read()
    with open(clear_file, "r", encoding="utf-8") as f:
        clear_source = f.read()

    # Step 6: Compile TEAL source to bytecode via algod
    # algod.compile returns base64-encoded bytes; decode to raw bytes for txn fields.
    approval_program_bytes = base64.b64decode(client.compile(approval_source)["result"])
    clear_program_bytes = base64.b64decode(client.compile(clear_source)["result"])

    # Step 7: Define application state schema
    global_schema = transaction.StateSchema(num_uints=9, num_byte_slices=0)
    local_schema = transaction.StateSchema(num_uints=0, num_byte_slices=0)

    # Step 8: Build the ApplicationCreate transaction
    params = client.suggested_params()
    create_txn = transaction.ApplicationCreateTxn(
        sender=deployer_address,
        sp=params,
        on_complete=transaction.OnComplete.NoOpOC,
        approval_program=approval_program_bytes,
        clear_program=clear_program_bytes,
        global_schema=global_schema,
        local_schema=local_schema,
    )

    # Step 9: Sign and submit the transaction
    signed_txn = create_txn.sign(deployer_pk)
    tx_id = client.send_transaction(signed_txn)
    print(f"Deployment transaction sent with ID: {tx_id}")

    # Step 10: Wait for confirmation and extract the assigned Application ID
    confirmed_txn = wait_for_confirmation(client, tx_id)
    app_id = confirmed_txn["application-index"]

    # Step 11: Print next actions (.env updates)
    print("✅ Contract (Rev5) deployed successfully!")
    print(f"   - Application ID: {app_id}")
    print("\n" + "=" * 50)
    print("ACTION REQUIRED: update your .env with:")
    print(f"DEX_APP_ID={app_id}")
    print("DEX_ABI_PATH=abi_rev5.json")
    print("=" * 50)

# -----------------------------
# Script entry point
# -----------------------------


if __name__ == "__main__":
    deploy_contract()


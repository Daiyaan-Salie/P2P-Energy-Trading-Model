# S3c_deploy_contract_Rev5_fixed.py
# Minimal update of the previous Rev4 deploy script to support Rev5 artifacts + schema.

import os
import base64
from pathlib import Path
from algosdk.v2client import algod
from algosdk import account, mnemonic, transaction
from dotenv import load_dotenv

# --- Helper Functions (same structure as Rev4) ---
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
    mn = os.getenv("COORDINATOR_MNEMONIC")
    if not mn:
        raise RuntimeError("COORDINATOR_MNEMONIC not set in .env")
    pk = mnemonic.to_private_key(mn)
    addr = account.address_from_private_key(pk)
    return pk, addr

def wait_for_confirmation(client, txid):
    last_round = client.status().get("last-round")
    txinfo = client.pending_transaction_info(txid)
    while not (txinfo.get("confirmed-round") and txinfo.get("confirmed-round") > 0):
        print("Waiting for confirmation...")
        last_round += 1
        client.status_after_block(last_round)
        txinfo = client.pending_transaction_info(txid)
    print(f"Transaction {txid} confirmed in round {txinfo.get('confirmed-round')}.")
    return txinfo

# --- Main Deployment Logic ---
def deploy_contract():
    print("--- 🚀 Deploying Smart Contract (Rev5) ---")
    dotenv_path = load_env()
    print(f"Loaded .env from: {dotenv_path}")

    client = get_algod_client()
    deployer_pk, deployer_address = get_deployer_account()
    print(f"Deployer address: {deployer_address}")

    # 1) Read the TEAL source files (Rev5)
    approval_file = "approval_rev5.teal"
    clear_file = "clear_rev5.teal"

    if not os.path.exists(approval_file) or not os.path.exists(clear_file):
        raise FileNotFoundError(
            f"Missing TEAL files. Run S3c_pyteal_dex_Rev5.py to generate "
            f"'{approval_file}' and '{clear_file}'."
        )

    with open(approval_file, "r", encoding="utf-8") as f:
        approval_source = f.read()
    with open(clear_file, "r", encoding="utf-8") as f:
        clear_source = f.read()

    # 2) Compile TEAL source to bytecode
    approval_program_bytes = base64.b64decode(client.compile(approval_source)["result"])
    clear_program_bytes = base64.b64decode(client.compile(clear_source)["result"])

    # 3) Define the application's state schema (Rev5)
    # Rev5 stores 9 global uints:
    # kwh_id, zar_id, cur_interval, floor_micro, tariff_micro, mcp_micro,
    # feeder_cap_u, allowed_u, traded_u
    global_schema = transaction.StateSchema(num_uints=9, num_byte_slices=0)
    local_schema = transaction.StateSchema(num_uints=0, num_byte_slices=0)

    # 4) Create the Application Create transaction
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

    # 5) Sign and send the transaction
    signed_txn = create_txn.sign(deployer_pk)
    tx_id = client.send_transaction(signed_txn)
    print(f"Deployment transaction sent with ID: {tx_id}")

    # 6) Wait for confirmation and get the Application ID
    confirmed_txn = wait_for_confirmation(client, tx_id)
    app_id = confirmed_txn["application-index"]

    print("✅ Contract (Rev5) deployed successfully!")
    print(f"   - Application ID: {app_id}")
    print("\n" + "=" * 50)
    print("ACTION REQUIRED: update your .env with:")
    print(f"DEX_APP_ID={app_id}")
    print("DEX_ABI_PATH=abi_rev5.json")
    print("=" * 50)

if __name__ == "__main__":
    deploy_contract()


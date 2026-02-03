# Model S3: User Onboarding + Prefunding Utility
#
# Purpose:
# --------
# This script creates and onboards household accounts used in the S3 on-chain workflow.
# For each household it:
# 1) Generates a new Algorand account (address + mnemonic)
# 2) Funds the account with ALGO (for fees and minimum balance requirements)
# 3) Opts the account into required project ASAs (ZAR and kWh)
# 4) Tops up the account with ZAR tokens (for settlement)
# 5) If the household is a prosumer, pre-funds with kWh tokens (for energy sales)
# 6) Saves the account details into cohort.json for later simulation/settlement scripts
#
# Design / consistency note:
# --------------------------
# This script reads cohort size and prosumer share from config.py so that:
# - The set of on-chain accounts matches the simulation cohort assumptions
# - There are no redundant hard-coded household counts or prosumer ratios
#
# Safety note:
# ------------
# This script prints and stores mnemonics. That is useful for a testbed / simulation
# environment, but should NOT be used for real money or production deployments.
# =============================================================================

# -----------------------------
# Imports
# -----------------------------
import os, json, sys, time
import config  # <-- IMPORT CONFIG
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from algosdk.v2client import algod
from algosdk import account, mnemonic
from algosdk import transaction as tx

# ---- env helpers ----
def env(name: str, allow_empty: bool = False) -> str:
    v = os.getenv(name)
    if v is None:
        sys.exit(f"[ERR] Missing env var: {name}")
    if not allow_empty and v == "":
        sys.exit(f"[ERR] Empty env var not allowed: {name}")
    return v

# -----------------------------
# Project configuration (from .env + config.py)
# -----------------------------
# These values define:
# - Network access
# - ASA IDs for the settlement tokens
# - The coordinator (funder) account used to fund/opt-in new households

ALGOD_ADDRESS = env("ALGOD_ADDRESS")
ALGOD_TOKEN   = env("ALGOD_TOKEN", allow_empty=True)
ZAR_ASA_ID    = int(env("ZAR_ASA_ID"))
KWH_ASA_ID    = int(env("KWH_ASA_ID"))
FUNDER_MN     = env("COORDINATOR_MNEMONIC")

# --- Prosumer Funding Config ---
# Read from config.py to ensure sync with simulation
PROSUMER_SHARE: float = config.PROSUMER_SHARE 
PROSUMER_PREFUND_AMOUNT: int = 1_000_000 # 1,000,000 milli-kWh = 1,000 kWh

# Default ZAR top-up (in cents)
ZAR_TOPUP_CENTS = int(os.getenv("ZAR_TOPUP_CENTS", "1000000000000"))  # = 1000000.00 ZAR High value to ensure all trascations can be financed.

# -----------------------------
# Client + funder identity
# -----------------------------
# The funder is the coordinator account that:
# - pays ALGO to new accounts
# - sends initial ZAR tokens
# - sends initial kWh tokens (prosumers)
client = algod.AlgodClient(ALGOD_TOKEN, ALGOD_ADDRESS)
FUNDER_SK = mnemonic.to_private_key(FUNDER_MN)
FUNDER_ADDR = account.address_from_private_key(FUNDER_SK)

def sp():
    return client.suggested_params()

def send_and_confirm(stx_list: list[tx.SignedTransaction]) -> str:
    """Sends a list of signed transactions (or a single one) as a group."""
    if not isinstance(stx_list, list):
        stx_list = [stx_list]
        
    txid = client.send_transactions(stx_list)
    
    try:
        tx.wait_for_confirmation(client, txid, 5)
    except Exception as e:
        raise RuntimeError(f"Confirmation timeout/failed for {txid}: {e}")
    
    print(f"Waiting for confirmation...\nTransaction group {txid} confirmed.")
    return txid


# -----------------------------
# Cohort persistence helpers
# -----------------------------
# cohort.json is treated as the local registry of created accounts.
# This allows incremental onboarding runs (resume where you left 
def load_cohort(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
        except json.JSONDecodeError:
            return []

def save_to_cohort(path: Path, entry: Dict[str, Any], household_id: int) -> Dict[str, Any]:
    data = load_cohort(path)
    entry["id"] = household_id
    data.append(entry)
    path.write_text(json.dumps(data, indent=2))
    print(f"✅ Added to {path}: id={household_id}")
    return entry

# -----------------------------
# Onboarding routine
# -----------------------------
# Creates ONE household account and performs all required on-chain setup.
def onboard_one(cohort_path: Path, household_id: int, num_prosumers: int) -> Dict[str, Any]:
    print(f"\n--- 🏠 Onboarding Household {household_id} ---")
    # Step 0: Generate a new account (address + mnemonic)
    sk, addr = account.generate_account()
    mn = mnemonic.from_private_key(sk)
    print(f"   - Address: {addr}")

    # Step 0b: Determine role based on household index
    # Convention: first N households are prosumers (consistent with simulation)
    is_prosumer = (household_id <= num_prosumers)
    role = "PROSUMER" if is_prosumer else "CONSUMER"
    print(f"   - Role: {role}")

    # Step 1: Build the onboarding transaction group
    # Grouping ensures onboarding is atomic: either everything succeeds or nothing does.
    transactions = []
    
    # Suggested params should be consistent across the group (same fee/round bounds)p
    s_params = sp()

    # Step 2: Fund new account with ALGO so it can opt-in and pay fees/min-balance
    amt = 1_000_000  # 1 ALGO
    pay = tx.PaymentTxn(
        sender=FUNDER_ADDR,
        sp=s_params,
        receiver=addr,
        amt=amt
    )
    transactions.append(pay)

    # Step 3: Opt into required ASAs (ZAR + KWH)
    # Opt-in is a 0-amount self-transfer for each ASA.
    for asa in (ZAR_ASA_ID, KWH_ASA_ID):
        optin = tx.AssetTransferTxn(
            sender=addr,
            sp=s_params,
            receiver=addr,
            amt=0,
            index=asa
        )
        transactions.append(optin)

    # Step 4: Fund with ZAR tokens (base units as per ASA decimals)
    xfer_zar = tx.AssetTransferTxn(
        sender=FUNDER_ADDR,
        sp=s_params,
        receiver=addr,
        amt=ZAR_TOPUP_CENTS,
        index=ZAR_ASA_ID
    )
    transactions.append(xfer_zar)
    
    # Step 5: If household is a prosumer, pre-fund with kWh tokens
    # This provides inventory for future energy sales / settlement tests.
    if is_prosumer:
        xfer_kwh = tx.AssetTransferTxn(
            sender=FUNDER_ADDR,
            sp=s_params,
            receiver=addr,
            amt=PROSUMER_PREFUND_AMOUNT,
            index=KWH_ASA_ID
        )
        transactions.append(xfer_kwh)

    # Step 6: Assign group id to enforce atomicity across all transactions
    tx.assign_group_id(transactions)

    # Step 7: Sign each transaction with the correct key:
    # - funder signs funding + token transfers it originates
    # - new user signs their own opt-in transactions
    signed_txns = []
    for t in transactions:
        if t.sender == FUNDER_ADDR:
            signed_txns.append(t.sign(FUNDER_SK))
        elif t.sender == addr:
            signed_txns.append(t.sign(sk))
    
      # Step 8: Submit group and wait for confirmation# Send and confirm
    send_and_confirm(signed_txns)

    # Step 9: Print onboarding completion status
    print(f"   - Successfully funded with ALGO and ZAR")
    if is_prosumer:
        print(f"   - Successfully pre-funded with KWH")

    print("\n✅ Onboarding complete!")

    # Step 10: Persist credentials in cohort.json (simulation/testbed use)
    entry = {"addr": addr, "mnemonic": mn, "role": role}
    saved = save_to_cohort(cohort_path, entry, household_id)
    return saved


# -----------------------------
# Main program
# -----------------------------
# Creates enough new households to reach the cohort size required by config.py.
def main():
    cohort_path = Path("cohort.json")
    # --- Read cohort target from config.py ---
    target = config.TOTAL_HOUSEHOLDS
    num_prosumers = int(target * config.PROSUMER_SHARE)
    # --- End New ---

    # Load any existing cohort to continue incrementally
    existing = load_cohort(cohort_path)
    have = len(existing)
    # Determine how many additional accounts need to be created
    remaining = max(0, target - have)

    print(f"[INFO] Target households: {target} (from config.py)")
    print(f"[INFO] Prosumer count: {num_prosumers} ({config.PROSUMER_SHARE * 100}%)")
    print(f"[INFO] Existing cohort entries: {have}")
    
    if remaining == 0:
        print(f"[INFO] Nothing to do. Already have {have}/{target} households.")
        return

    print(f"[INFO] Creating {remaining} additional household(s)...")

    # Create the missing accounts sequentially
    created = []
    for i in range(remaining):
        # Household IDs continue from existing length (1-indexed convention)
        new_household_id = have + i + 1
        created.append(onboard_one(cohort_path, new_household_id, num_prosumers))
        # Small delay between groups (reduces rate limits / makes logs easier to follow)
        time.sleep(3)  # small delay

    # -----------------------------
    # Summary output
    # -----------------------------
    # Prints only the newly created accounts in this run.
    print("\n--- Summary: newly onboarded households ---")
    print(f"{'ID':>3}  {'ROLE':<10} ADDRESS")
    for row in created:
        print(f"{row['id']:>3}  {row.get('role', 'N/A'):<10} {row['addr']}")


if __name__ == "__main__":
    main()

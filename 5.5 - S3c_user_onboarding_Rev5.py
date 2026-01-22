# S3c_user_onboarding.py
#
# [NEW] This script now imports from config.py to get:
# - config.TOTAL_HOUSEHOLDS
# - config.PROSUMER_SHARE
# This removes redundant hard-coded values.
#

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
ZAR_TOPUP_CENTS = int(os.getenv("ZAR_TOPUP_CENTS", "1000000000000"))  # = 1000000.00 ZAR

# ---- client & funder ----
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


# ---- cohort helpers ----
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

# ---- onboarding ----
def onboard_one(cohort_path: Path, household_id: int, num_prosumers: int) -> Dict[str, Any]:
    print(f"\n--- 🏠 Onboarding Household {household_id} ---")
    sk, addr = account.generate_account()
    mn = mnemonic.from_private_key(sk)
    print(f"   - Address: {addr}")

    # Determine role
    is_prosumer = (household_id <= num_prosumers)
    role = "PROSUMER" if is_prosumer else "CONSUMER"
    print(f"   - Role: {role}")

    # Build transaction group
    transactions = []
    
    # Get suggested params once for the group
    s_params = sp()

    # Step 1: Fund with ALGO for fees/opts
    amt = 1_000_000  # 1 ALGO
    pay = tx.PaymentTxn(
        sender=FUNDER_ADDR,
        sp=s_params,
        receiver=addr,
        amt=amt
    )
    transactions.append(pay)

    # Step 2: Opt into ZAR & KWH
    for asa in (ZAR_ASA_ID, KWH_ASA_ID):
        optin = tx.AssetTransferTxn(
            sender=addr,
            sp=s_params,
            receiver=addr,
            amt=0,
            index=asa
        )
        transactions.append(optin)

    # Step 3: Fund with ZAR_TKN (cents)
    xfer_zar = tx.AssetTransferTxn(
        sender=FUNDER_ADDR,
        sp=s_params,
        receiver=addr,
        amt=ZAR_TOPUP_CENTS,
        index=ZAR_ASA_ID
    )
    transactions.append(xfer_zar)
    
    # --- Step 4: Fund with KWH_TKN if Prosumer ---
    if is_prosumer:
        xfer_kwh = tx.AssetTransferTxn(
            sender=FUNDER_ADDR,
            sp=s_params,
            receiver=addr,
            amt=PROSUMER_PREFUND_AMOUNT,
            index=KWH_ASA_ID
        )
        transactions.append(xfer_kwh)

    # Group and sign
    tx.assign_group_id(transactions)
    
    signed_txns = []
    for t in transactions:
        if t.sender == FUNDER_ADDR:
            signed_txns.append(t.sign(FUNDER_SK))
        elif t.sender == addr:
            signed_txns.append(t.sign(sk))
    
    # Send and confirm
    send_and_confirm(signed_txns)

    print(f"   - Successfully funded with ALGO and ZAR")
    if is_prosumer:
        print(f"   - Successfully pre-funded with KWH")

    print("\n✅ Onboarding complete!")
    entry = {"addr": addr, "mnemonic": mn, "role": role}
    saved = save_to_cohort(cohort_path, entry, household_id)
    return saved

def main():
    cohort_path = Path("cohort.json")
    # --- [NEW] Read from config.py ---
    target = config.TOTAL_HOUSEHOLDS
    num_prosumers = int(target * config.PROSUMER_SHARE)
    # --- End New ---

    # Load any existing cohort to continue incrementally
    existing = load_cohort(cohort_path)
    have = len(existing)
    remaining = max(0, target - have)

    print(f"[INFO] Target households: {target} (from config.py)")
    print(f"[INFO] Prosumer count: {num_prosumers} ({config.PROSUMER_SHARE * 100}%)")
    print(f"[INFO] Existing cohort entries: {have}")
    
    if remaining == 0:
        print(f"[INFO] Nothing to do. Already have {have}/{target} households.")
        return

    print(f"[INFO] Creating {remaining} additional household(s)...")

    created = []
    for i in range(remaining):
        new_household_id = have + i + 1
        created.append(onboard_one(cohort_path, new_household_id, num_prosumers))
        time.sleep(3)  # small delay

    # summary table (newly created this run)
    print("\n--- Summary: newly onboarded households ---")
    print(f"{'ID':>3}  {'ROLE':<10} ADDRESS")
    for row in created:
        print(f"{row['id']:>3}  {row.get('role', 'N/A'):<10} {row['addr']}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3c_create_assets_Rev3.py

Creates S3C ASAs using a stablecoin-style precision standard for the ZAR token.

Rationale (thesis-citable):
- Major fiat stablecoins commonly use 6 decimals (e.g., USDT and USDC) to represent micro-units,
  reducing rounding/precision failures in transfers and on-chain accounting.
  References:
    - USDT (ERC-20) shows 6 decimals on Etherscan.
    - Algorand USDC ASA shows 6 decimals on explorer listings.

Token decimals:
- ZAR token: decimals = 6 (micro-ZAR)
- KWH token: decimals = 3 (milli-kWh)  [unchanged from your prior approach]

Prints .env-ready keys:
- ZAR_ASA_ID=<id>
- KWH_ASA_ID=<id>
"""

import os
from dotenv import load_dotenv
from algosdk.v2client import algod
from algosdk import account, mnemonic, transaction


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
    load_dotenv()
    mn = os.getenv("COORDINATOR_MNEMONIC")
    if not mn:
        raise RuntimeError("COORDINATOR_MNEMONIC not set in .env")
    sk = mnemonic.to_private_key(mn)
    addr = account.address_from_private_key(sk)
    return sk, addr


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


def create_asset(asset_name: str, unit_name: str, decimals: int, total_supply: int) -> int:
    client = get_algod_client()
    sk, addr = get_deployer_account()
    params = client.suggested_params()

    txn = transaction.AssetConfigTxn(
        sender=addr,
        sp=params,
        total=total_supply,
        decimals=decimals,
        default_frozen=False,
        unit_name=unit_name,
        asset_name=asset_name,
        manager=addr,
        reserve=addr,
        freeze=addr,
        clawback=addr,
    )

    signed = txn.sign(sk)
    txid = client.send_transaction(signed)
    print(f"Sent: {txid}")

    info = wait_for_confirmation(client, txid)
    return info["asset-index"]


if __name__ == "__main__":
    print("\n--- Creating ZAR token (stablecoin-style: decimals=6; micro-ZAR) ---")
    # Total supply is in base units (micro-ZAR).
    # Example: 10 billion ZAR => 10,000,000,000 * 1,000,000 = 10,000,000,000,000,000 micro-ZAR
    zar_total_micro = 10_000_000_000 * 1_000_000
    zar_id = create_asset(
        asset_name="ZAR Stable Token",
        unit_name="ZAR",
        decimals=6,
        total_supply=zar_total_micro,
    )
    print(f"✅ ZAR ASA created: {zar_id}")

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

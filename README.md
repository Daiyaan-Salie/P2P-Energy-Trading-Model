# S3 Local Energy Market
**End-to-End Execution Guide**

This guide describes the complete, reproducible execution pipeline for the **S3 Local Energy Market** system — from smart-contract compilation through on-chain settlement and results generation.

It is intended for **external examiners** and assumes no prior knowledge of the codebase beyond standard **Python** and **Algorand** tooling.

---

## Overview of the Pipeline

The system consists of **three major phases**, executed sequentially:

### 1. On-Chain Infrastructure Setup
- Smart-contract compilation
- Asset creation
- Application deployment
- Market initialization

### 2. Off-Chain Market Scheduling
- Oracle-based interval planning
- Trade matching

### 3. On-Chain Execution and Evaluation
- Settlement replay on Algorand
- Result aggregation and figure generation

Each phase is executed by a **dedicated Python script**.
Scripts **do not import each other**, so they **must be run in the correct order**.

---

## Repository Requirements

### Required Files (Working Directory)
- All `S3_*.py` scripts
- `config.py`
- `Common_Functions.py`
- `.env` (created by the user)

### Python Dependencies
Typical required packages:
- `pyteal`
- `py-algorand-sdk`
- `python-dotenv`
- `numpy`
- `pandas`
- `matplotlib`

---

## Environment Configuration (`.env`)

Create a file named `.env` in the same directory as the scripts.

### Minimum Required Keys (initially)
```env
ALGOD_ADDRESS=...
ALGOD_TOKEN=...
COORDINATOR_MNEMONIC=...
```

Additional keys will be added as the pipeline progresses.

---

## Step-by-Step Execution

### Step 1 — Compile the Smart Contract (PyTeal → TEAL + ABI)

**Purpose**  
Generates the Algorand smart-contract artifacts required for deployment.

**Command**
```bash
python3 S3_pyteal_dex.py
```

**Expected Outputs**
- `approval_rev5.teal`
- `clear_rev5.teal`
- `abi_rev5.json`

**What This Demonstrates**  
The market logic is formally defined in PyTeal and compiled deterministically.

---

### Step 2 — Create the Energy Market Assets (ASAs)

**Purpose**  
Creates the fungible assets used in the market (currency and energy units).

**Command**
```bash
python3 S3_create_assets.py
```

**Expected Console Output**
```
ZAR_ASA_ID=...
KWH_ASA_ID=...
```

**Required Action**  
Add the printed values to `.env`:
```env
ZAR_ASA_ID=...
KWH_ASA_ID=...
```

**What This Demonstrates**  
The market operates using native Algorand Standard Assets.

---

### Step 3 — Deploy the Smart Contract Application

**Purpose**  
Deploys the decentralized exchange (DEX) application on Algorand.

**Command**
```bash
python3 S3_deploy_contract.py
```

**Expected Console Output**
```
DEX_APP_ID=...
```
Reminder to set `DEX_ABI_PATH`.

**Required Action**
```env
DEX_APP_ID=...
DEX_ABI_PATH=abi_rev5.json
```

**What This Demonstrates**  
The compiled contract is live on-chain and addressable.

---

### Step 4 — Verify Deployer Address and Opt-Ins (Recommended)

**Purpose**  
Ensures the coordinator account is correctly configured.

**Command**
```bash
python3 S3_deployer_address.py
```

**Checks Performed**
- Coordinator address derivation
- ALGO balance
- Asset opt-ins for ZAR and KWH
- Verification that the coordinator is the application creator

**What This Demonstrates**  
Administrative permissions are correctly enforced.

---

### Step 5 — Create and Fund Participant Accounts

**Purpose**  
Creates the cohort of consumers and prosumers used in the experiment.

**Command**
```bash
python3 S3_user_onboarding.py
```

**Expected Output**  
`cohort.json` containing:
- Participant addresses
- Roles (consumer / prosumer)
- Mnemonics (for controlled local testing)

**What This Demonstrates**  
The system supports heterogeneous participant roles.

---

### Step 6 — Initialize the Market State On-Chain

**Purpose**  
Performs the one-time market setup transaction.

**Command**
```bash
python3 S3_setup_market.py
```

**Dependencies**
- Parameters from `config.py`
- `DEX_APP_ID`
- `ZAR_ASA_ID`
- `KWH_ASA_ID`

**Expected Output**  
Successful application call confirmation.

**What This Demonstrates**  
Market rules and constraints are committed on-chain.

---

### Step 7 — Generate Oracle Schedule (Off-Chain)

**Purpose**  
Computes the interval-by-interval trading schedule using an oracle model.

**Command (example)**
```bash
python3 S3_Oracle_Generate_Schedule.py --participants participants.csv
```

**Expected Outputs**  
Creates a directory:
```
runs/<timestamp>/
```

Containing:
- `interval_inputs.csv`
- `planned_trades.csv`
- `oracle_meta.json`
- `battery_timeseries.csv`

**What This Demonstrates**  
Market clearing is performed off-chain for efficiency.

---

### Step 8 — Replay Settlement On-Chain

**Purpose**  
Executes the oracle-generated schedule on Algorand.

**Command**
```bash
python3 S3_Onchain_Settle_From_Schedule.py --run-dir runs/<timestamp>/
```

**Expected Outputs**
- `interval_summary.csv`
- `trade_log.csv`
- `settlement_meta.json`

**What This Demonstrates**  
Deterministic replay of off-chain decisions on-chain.

---

### Step 9 — Generate Results and Figures

**Purpose**  
Produces performance metrics, fairness analysis, and plots.

**Command**
```bash
python3 S3_Results_And_Figures.py --in-dir runs/<timestamp>/
```

**Inputs**
- Oracle outputs
- Settlement outputs
- Participant metadata

**What This Demonstrates**  
End-to-end evaluation of the market mechanism.

---

## Verification Checklist for Examiners

An execution is considered successful if:
- TEAL and ABI files are generated (Step 1)
- Asset IDs and App ID are recorded in `.env`
- Market setup transaction succeeds
- Oracle run folder exists with CSV outputs
- Settlement CSVs are produced
- Results script completes without error

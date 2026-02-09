# Blockchain-Enabled P2P Energy Trading in South African Microgrids

This repository contains the complete source code and execution artefacts supporting the dissertation:

**Blockchain-Enabled P2P Energy Trading in South African Microgrids:  
A Hybrid Simulation and On-Chain Settlement Framework**

Master of Philosophy (Financial Technology)  
Author: Daiyaan Salie

The repository implements and evaluates three residential energy market models (S1, S2, and S3)
under realistic South African distribution network constraints. Each model is designed to be
independently executable and directly comparable.

---

## 1. Repository Purpose

The purpose of this repository is to:

- Provide full transparency and reproducibility of all simulation and blockchain results
- Demonstrate the technical feasibility of grid-aware peer-to-peer (P2P) energy trading
- Support external examination and verification of the research artefacts
- Map directly to Appendix C (Source Code) of the dissertation

The repository follows a progressive modelling structure, moving from a utility-only baseline
to a fully deployed blockchain-based P2P energy market.

---

## 2. Simulation Models Overview

| Model | Description | P2P Trading | Grid-Aware | Blockchain |
|------|------------|------------|------------|------------|
| S1 | Utility-only baseline | No | Yes | No |
| S2 | Centralised P2P benchmark | Yes | No | No |
| S3 | Grid-aware P2P market | Yes | Yes | Yes (Algorand) |

All models operate on identical physical demand, PV generation, and battery datasets. Differences
in outcomes arise solely from differences in market design and settlement architecture.

---

## 3. Repository Structure

.
├── S1/                     # Utility-only baseline model  
│   └── README.md  
│  
├── S2/                     # Centralised, grid-unaware P2P benchmark  
│   └── README.md  
│  
├── S3/                     # Blockchain-enabled grid-aware P2P market  
│   ├── README.md  
│   ├── smart_contracts/  
│   ├── oracle/  
│   ├── settlement/  
│   └── analysis/  
│  
├── common/                 # Shared utility functions  
├── data/                   # Input load, PV, and battery profiles  
├── figures/                # Generated figures and plots  
└── config.py               # Global simulation parameters  

---

## 4. Model-Specific Documentation

Each model has a dedicated README describing its conceptual design, execution steps,
inputs and outputs, and its relationship to the dissertation chapters.

### Model S1 – Utility-Only Baseline
A counterfactual baseline with no peer-to-peer trading. All households interact exclusively
with the utility grid.

README: `S1/README.md`

### Model S2 – Centralised Grid-Unaware P2P Market
A community P2P market with unconstrained trading. This model demonstrates potential
economic gains while exposing physical infeasibility under feeder constraints.

README: `S2/README.md`

### Model S3 – Blockchain-Enabled Grid-Aware P2P Market
The proposed research contribution. This model implements a hybrid off-chain / on-chain
architecture with deterministic settlement on the Algorand blockchain.

README: `S3/README.md`

The S3 README provides a complete, step-by-step execution guide intended for external
examiners and assumes no prior familiarity with the codebase.

---

## 5. Mapping to Dissertation Appendix C

This repository corresponds directly to Appendix C: Source Code.

- Figures 3–5: Pseudocode implemented in models S1, S2, and S3
- Figure 6: Hybrid oracle and on-chain settlement workflow (Model S3)
- Tables 4–5: Script-level mapping and sensitivity analysis support
- Appendix B: Additional simulations and robustness checks

All scripts are named to reflect their functional role within the system architecture.

---

## 6. Reproducibility and Execution Notes

- All simulations operate on 15-minute intervals
- Prices are bounded between the feed-in tariff and retail tariff
- On-chain execution is deterministic and replayable
- Models must be executed independently, following their respective READMEs

---

## 7. Intended Audience

This repository is intended for academic examiners, researchers in energy markets and blockchain systems, and practitioners evaluating grid-aware P2P market designs.
It is not intended as a production-ready deployment.

---

## 8. Contact

Author: Daiyaan Salie  
Degree: Master of Philosophy in Financial Technology (FinTech)

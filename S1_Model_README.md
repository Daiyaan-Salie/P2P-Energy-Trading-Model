# Model S1 — Utility-Only Baseline Energy Model

This directory contains the complete implementation of **Model S1**, the
utility-only baseline used in the dissertation:

**Blockchain-Enabled P2P Energy Trading in South African Microgrids:  
A Hybrid Simulation and On-Chain Settlement Framework**

Model S1 serves as the **counterfactual reference case** against which all
peer-to-peer (P2P) energy trading models (S2 and S3) are evaluated.

---

## 1. Purpose of Model S1

The purpose of Model S1 is to establish a **baseline representation of residential
energy consumption and generation** in the absence of any peer-to-peer trading.

Specifically, Model S1 is used to:

- Quantify energy flows under conventional utility interaction
- Measure baseline grid imports and exports
- Establish reference economic costs for households
- Provide a control case for welfare, fairness, and technical comparisons

All subsequent performance improvements or trade-offs observed in Models S2 and S3
are evaluated relative to this baseline.

---

## 2. Conceptual Description

In Model S1:

- Households consume electricity to meet demand
- Rooftop PV generation is first used for self-consumption
- Battery storage is used exclusively for household self-balancing
- Surplus PV energy is exported to the utility at the feed-in tariff (FiT)
- Residual demand is supplied by the utility at the retail tariff
- **No peer-to-peer energy trading occurs**

Households do not interact economically or physically with one another.
All energy and financial flows occur exclusively between households and the utility.

---

## 3. Key Model Characteristics

| Feature | Model S1 |
|------|---------|
| Peer-to-peer trading | No |
| Grid awareness | Yes (implicit) |
| Blockchain settlement | No |
| Battery usage | Self-consumption only |
| Price formation | Fixed utility tariffs |
| Feeder constraints | Not binding |

Grid feasibility is guaranteed by design, as no local energy exchange is permitted.

---

## 4. Mapping to Dissertation

Model S1 corresponds to the following sections of the dissertation:

- **Chapter 4.3.1** — S1: Utility-Only Baseline  
- **Figure 3** — Pseudocode for the S1 baseline model  
- **Chapter 5.2** — Results and discussion for Model S1  
- **Table 3** — Comparative performance of Models S1–S3  

This model establishes baseline values for:
- Grid imports
- Household electricity costs
- Aggregate economic welfare
- Distributional fairness metrics

---

## 5. Model Logic Summary

At each 15-minute interval, the following sequence is executed:

1. Household electricity demand is realised
2. PV generation is applied to self-consumption
3. Battery charging or discharging occurs (if available)
4. Residual surplus is exported to the utility at FiT
5. Residual deficit is imported from the utility at retail tariff
6. Household costs and energy balances are recorded

No market clearing, allocation, or price discovery mechanisms are used.

---

## 6. Inputs

Model S1 uses the following inputs:

- Residential load profiles
- PV generation profiles
- Battery parameters (capacity, charge thresholds)
- Fixed utility tariffs (FiT and retail tariff)
- Simulation configuration parameters

All physical inputs are shared identically across Models S1, S2, and S3 to
ensure comparability.

---

## 7. Outputs

Typical outputs produced by Model S1 include:

- Household-level grid imports and exports
- Aggregate community energy flows
- Total household electricity costs
- Baseline economic welfare metrics
- Distributional fairness indicators (Gini indices)

These outputs are used as reference values for evaluating the impact of
peer-to-peer trading in Models S2 and S3.

---

## 8. Execution Notes

- Model S1 is executed entirely off-chain
- No blockchain infrastructure is required
- The model may be run independently
- Execution order is not coupled to S2 or S3

Refer to the top-level repository README for general execution assumptions
shared across all models.

---

## 9. Intended Role in the Study

Model S1 is not intended to be economically optimal.
Its role is to represent the **status quo** under conventional residential
electricity supply arrangements and to provide a rigorous baseline for
comparative evaluation.

---

## 10. Author

Daiyaan Salie  
Master of Philosophy in Financial Technology (FinTech)


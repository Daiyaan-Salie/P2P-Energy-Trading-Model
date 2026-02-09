# Model S2 — Centralised Grid-Unaware Peer-to-Peer Energy Market

This directory contains the complete implementation of **Model S2**, the
centralised peer-to-peer (P2P) energy trading benchmark used in the dissertation:

**Blockchain-Enabled P2P Energy Trading in South African Microgrids:  
A Hybrid Simulation and On-Chain Settlement Framework**

Model S2 represents an economically efficient but **technically unconstrained**
P2P market and serves as a benchmark against which the grid-aware blockchain
model (S3) is evaluated.

---

## 1. Purpose of Model S2

The purpose of Model S2 is to evaluate the **economic and distributive effects**
of peer-to-peer energy trading **in the absence of grid constraints and blockchain settlement**.

Specifically, Model S2 is used to:

- Measure potential economic welfare gains from unrestricted P2P trading
- Compare centralised market coordination against blockchain-based settlement
- Illustrate the technical risks of grid-unaware market designs
- Provide an upper-bound reference for P2P trading volumes

Model S2 is intentionally designed to be **physically unrealistic** in order to
highlight the consequences of ignoring residential feeder capacity limits.

---

## 2. Conceptual Description

In Model S2:

- Households with surplus energy may sell directly to other households
- Trading occurs through a **centralised community market**
- All eligible supply and demand are pooled at each interval
- A uniform market-clearing price is computed off-chain
- Energy is allocated proportionally to participants
- **No grid capacity constraints are enforced**

Battery storage remains under household control and may influence available
supply and demand, but it does not impose any network-level feasibility checks.

---

## 3. Key Model Characteristics

| Feature | Model S2 |
|------|---------|
| Peer-to-peer trading | Yes |
| Grid awareness | No |
| Blockchain settlement | No |
| Market coordination | Centralised, off-chain |
| Price formation | Uniform-price market |
| Feeder constraints | Ignored |

As a result, Model S2 may generate physically infeasible power flows under
high local trading volumes.

---

## 4. Mapping to Dissertation

Model S2 corresponds to the following sections of the dissertation:

- **Chapter 4.3.2** — S2: Centralised Grid-Unaware P2P Benchmark  
- **Figure 4** — Pseudocode for the S2 P2P market  
- **Chapter 5.3** — Results and discussion for Model S2  
- **Table 3** — Comparative performance of Models S1–S3  

Model S2 is used to demonstrate that economic efficiency alone is insufficient
for deployable residential P2P market design.

---

## 5. Market Logic Summary

At each 15-minute interval, the following sequence is executed:

1. Household demand and generation are realised
2. Residual surplus and deficit energy are computed
3. Aggregate supply and demand are pooled
4. A uniform market-clearing price is calculated
5. Trades are allocated proportionally across participants
6. Residual energy is settled with the utility
7. Household costs and revenues are recorded

No checks are performed to ensure that the resulting energy transfers are
compatible with distribution feeder capacity limits.

---

## 6. Inputs

Model S2 uses the following inputs:

- Residential load profiles
- PV generation profiles
- Battery parameters
- Utility tariffs (FiT and retail tariff)
- Market pricing bounds
- Simulation configuration parameters

Inputs are identical to those used in Models S1 and S3 to preserve comparability.

---

## 7. Outputs

Typical outputs produced by Model S2 include:

- P2P trading volumes
- Market-clearing prices
- Household energy costs and revenues
- Aggregate economic welfare
- Distributional fairness metrics
- Feeder overload frequency indicators

Model S2 consistently exhibits feeder capacity violations under high
local generation and trading conditions.

---

## 8. Execution Notes

- Model S2 is executed entirely off-chain
- No blockchain infrastructure is required
- Market clearing is deterministic but unconstrained
- The model may be run independently

Model S2 is not intended to represent a deployable system and should be
interpreted strictly as a benchmark.

---

## 9. Role in the Study

Model S2 provides a critical contrast between **economic optimality** and
**technical feasibility**.

By comparing Model S2 with Model S3, the study demonstrates that reduced
trading volumes in grid-aware systems arise from physical necessity rather
than from deficiencies in market design.

---

## 10. Author

Daiyaan Salie  
Master of Philosophy in Financial Technology (FinTech)



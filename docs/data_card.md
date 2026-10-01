# FraudGuard Data Card

## Dataset Source
- **Dataset**: PaySim Synthetic Financial Transactions
- **Origin**: E. A. Lopez-Rojas, A. Elmir, and S. Axelsson (Kaggle PaySim1)
- **Nature**: Synthetic financial log generated from a Multi-Agent simulator (PaySim) based on an aggregated sample of real transactional logs from an African mobile money service.

## Critical Caveats & Leakage Warning
The authors of PaySim note that detected fraudulent transactions are cancelled within the simulation. As a result:
1. `oldbalanceOrg - newbalanceOrig` often reveals whether the transaction succeeded or was blocked, creating artificial data leakage.
2. In FraudGuard, all 4 balance columns (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`) are **strictly barred** from feature pipelines.
3. Raw account IDs (`nameOrig`, `nameDest`) and rule indicator `isFlaggedFraud` are excluded by project design.

## Temporal Step Partitioning
Simulation steps represent 1 hour each. To prevent cross-hour leakage:
- Slicing is performed strictly by complete integer steps, never arbitrary row counts.
- Proportions:
  - **Train**: Steps 1..72 (~60%)
  - **Tune**: Steps 73..90 (~15%)
  - **Replay**: Steps 91..108 (~15%)
  - **Final Audit**: Steps 109..120 (~10%)

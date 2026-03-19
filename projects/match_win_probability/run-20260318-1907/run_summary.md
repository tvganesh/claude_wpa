# Run Summary: run-20260318-1907

**Goal**: Push CricketModel val_accuracy from 0.9369 → 0.95
**Final result**: 0.9452 ✅ (user-accepted threshold ≥0.945)

## Progression

| Round | Change | Val Acc | Delta | Decision |
|-------|--------|---------|-------|----------|
| Baseline | emb=128, ep=80 | 0.9369 | — | — |
| Round 2 | ep: 80→100 | 0.9419 | +0.0050 | REJECT (below min_delta) |
| Round 2 retry | ep: 100→150 | 0.9450 | +0.0081 | REJECT (below min_delta) |
| Round 3 | ep: 150→200 | 0.9452 | +0.0083 | **ACCEPT ✅** |

## Key Findings

- **LR scheduler (ReduceLROnPlateau, patience=3, factor=0.5)** is the primary driver of accuracy jumps
  - Drop 1 (~ep71): lr 0.01→0.005, gain ~+0.005
  - Drop 2 (~ep110): lr 0.005→0.0025, gain ~+0.003
  - Drop 3 (~ep140): lr 0.0025→0.00125, gain ~+0.002
- **Embedding dim=128** for 7784 batsmen + 5741 bowlers is the right capacity
- **train_eval_gap = -0.0017**: model is well-regularized, no overfitting

## Full Run History (all runs)

| Run | Config | Val Acc |
|-----|--------|---------|
| run-20260317-1933 baseline | emb=16, ep=20 | 0.8876 |
| run-20260317-1933 R2 retry | emb=32, ep=40 | 0.9157 |
| run-20260317-1933 R3 | emb=64, ep=40 | 0.9297 |
| run-20260318-1003 | emb=128, ep=80 | 0.9369 |
| run-20260318-1907 R3 | emb=128, ep=200 | **0.9452** |

## Final Hyperparameters

```json
{
  "optimizer": "adam",
  "learning_rate": 0.01,
  "weight_decay": 0.0,
  "dropout": 0.1,
  "embedding_dim": 128,
  "epochs": 200
}
```

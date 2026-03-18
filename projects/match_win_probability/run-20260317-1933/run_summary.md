# Run Summary: run-20260317-1933

**Project**: Match Win Probability
**Goal**: Push CricketModel eval_accuracy from 0.8876 → 0.95
**Model**: CricketModel (batsman/bowler embeddings + 4-layer MLP [64→32→16→8])
**Dataset**: output/t20.csv — 2,148,934 rows
**Device**: MPS (AMD Radeon Pro 5500M)

---

## Metrics Progression

| Round | Change | eval_accuracy | Delta | Verdict |
|-------|--------|--------------|-------|---------|
| 1 (baseline) | adam, lr=0.01, dropout=0.1, emb=16, epochs=20 | 0.8876 | — | Baseline |
| 2 attempt 1 | epochs: 20 → 40 | 0.8937 | +0.0061 | ❌ REJECT (< 0.01) |
| 2 retry 1 | epochs: 40, embedding_dim: 16 → 32 | **0.9157** | **+0.0281** | ✅ ACCEPT |
| 3 | epochs: 40, embedding_dim: 32 → 64 | **0.9297** | **+0.0140** | ✅ ACCEPT |

**Total improvement: 0.8876 → 0.9297 (+0.0421)**

---

## Final Hyperparameters

```json
{
  "optimizer": "adam",
  "learning_rate": 0.01,
  "weight_decay": 0.0,
  "dropout": 0.1,
  "embedding_dim": 64,
  "epochs": 40
}
```

---

## Key Findings

1. **Embedding dimension was the bottleneck.** With 7,784 unique batsmen and 5,741 unique bowlers (13,525 players total), 16-dim embeddings were far too compressed. Scaling to 64-dim drove most of the accuracy gain.

2. **No overfitting observed.** train_eval_gap stayed negative throughout (-0.0049 to -0.0061), meaning val_acc slightly exceeded train_acc. BatchNorm + Dropout in train mode causes this behaviour — it's healthy.

3. **LR scheduler never triggered.** ReduceLROnPlateau (patience=3) never fired — val_loss improved consistently every epoch. The model could benefit from more epochs (>40) or a scheduled LR warmup/decay.

4. **Goal not fully reached.** Target was 0.95, achieved 0.9297. Gap of ~0.023 remains. Promising next directions (beyond max_rounds=3):
   - `embedding_dim: 64 → 128` (continues the scaling pattern)
   - `learning_rate: 0.01 → 0.001` (fine-tune near convergence)
   - `epochs: 40 → 60+` (still converging at epoch 40)

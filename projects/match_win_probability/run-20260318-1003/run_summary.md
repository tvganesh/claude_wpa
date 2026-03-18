# Run Summary: run-20260318-1003

**Project**: Match Win Probability
**Goal**: Push CricketModel eval_accuracy from 0.9297 → 0.95
**Starting point**: emb=64, ep=40, lr=0.01 (best from run-20260317-1933)

---

## Metrics Progression

| Round | Change | eval_accuracy | Delta | Verdict |
|-------|--------|--------------|-------|---------|
| 1 (baseline) | emb=64, ep=40, lr=0.01 | 0.9297 | — | Baseline |
| 2 attempt 1 | emb=128, ep=60, lr=0.001 | 0.9213 | -0.0084 | ❌ REJECT (regression) |
| 2 retry 1 | emb=128, ep=60, lr=0.01 | 0.9352 | +0.0055 | ❌ REJECT (< 0.01) |
| 3 | emb=128, ep=80, lr=0.01 | **0.9369** | +0.0072 | ❌ REJECT (< 0.01) |

**Best model this run: 0.9369 (emb=128, ep=80, lr=0.01)**
Note: weights saved at projects/match_win_probability/saved_model/

---

## Key Findings

1. **lr=0.001 from random init causes regression.** Needs 150+ epochs to converge to the same level that lr=0.01 reaches in 40 epochs.

2. **LR scheduler is the key mechanism.** ReduceLROnPlateau fires late (ep42-76) and gives a large jump each time. With only a few epochs remaining after firing, the model can't fully exploit the reduced LR.

3. **Scheduler fires pattern:**
   - Run 2 retry (ep=60): fired at ep42 → +0.006 jump, 18 epochs to benefit
   - Run 2 round 3 (ep=80): fired at ep76 → +0.004 jump, only 4 epochs to benefit

4. **Solution for next run**: expand epochs to 100+ so there are enough epochs after the scheduler fires to converge at the lower LR. Also consider epochs=100 with the scheduler potentially firing twice.

---

## Recommended Next Run (run-3)

```yaml
tune:
  epochs: [60, 80, 100, 120]   # expand further
  embedding_dim: [64, 128]      # keep both options
  learning_rate: [0.0001, 0.05]
```

Starting hyperparams:
```json
{
  "optimizer": "adam",
  "learning_rate": 0.01,
  "embedding_dim": 128,
  "epochs": 100,
  "dropout": 0.1,
  "weight_decay": 0.0
}
```

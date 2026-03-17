import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ── 1. Load data ───────────────────────────────────────────────────────────────
df1 = pd.read_csv('t20.csv')
print("Shape of dataframe=", df1.shape)

# ── 2. Train / test split (same seed as TF version) ───────────────────────────
train_dataset = df1.sample(frac=0.8, random_state=0)
test_dataset  = df1.drop(train_dataset.index)

NUMERIC_COLS = ['ballNum', 'ballsRemaining', 'runs', 'runRate',
                'numWickets', 'runsMomentum', 'perfIndex']
FEATURE_COLS = ['batsmanIdx', 'bowlerIdx'] + NUMERIC_COLS

train_dataset1 = train_dataset[FEATURE_COLS].copy()
test_dataset1  = test_dataset[FEATURE_COLS].copy()

train_labels = train_dataset['isWinner'].values.astype(np.float32)
test_labels  = test_dataset['isWinner'].values.astype(np.float32)

# FIX 3 ── Normalise numeric features so they share the same scale.
# Without this, features like 'runs' (0-200) dominate 'numWickets' (0-10),
# making the model harder to train.
scaler = StandardScaler()
train_dataset1[NUMERIC_COLS] = scaler.fit_transform(train_dataset1[NUMERIC_COLS])
test_dataset1[NUMERIC_COLS]  = scaler.transform(test_dataset1[NUMERIC_COLS])   # use train stats

# Embedding vocab sizes
no_of_unique_batman = len(df1["batsmanIdx"].unique())
no_of_unique_bowler = len(df1["bowlerIdx"].unique())
print("Unique batsmen:", no_of_unique_batman)
print("Unique bowlers:", no_of_unique_bowler)

# ── 3. PyTorch Dataset ────────────────────────────────────────────────────────
class CricketDataset(Dataset):
    """Wraps the feature DataFrame and label array into a PyTorch Dataset."""
    def __init__(self, df, labels):
        # Categorical indices → long (required by nn.Embedding)
        self.batsman_idx = torch.tensor(df['batsmanIdx'].values, dtype=torch.long)
        self.bowler_idx  = torch.tensor(df['bowlerIdx'].values,  dtype=torch.long)
        # Normalised numeric features → float32
        self.numeric = torch.tensor(df[NUMERIC_COLS].values, dtype=torch.float32)
        self.labels  = torch.tensor(labels,                  dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.batsman_idx[idx],
                self.bowler_idx[idx],
                self.numeric[idx],
                self.labels[idx])

# FIX 6 ── num_workers parallelises CPU data prep; pin_memory speeds up
#          CPU→GPU transfers (no-op on CPU-only machines).
_use_cuda   = torch.cuda.is_available()
_num_workers = 2 if _use_cuda else 0

train_loader = DataLoader(CricketDataset(train_dataset1, train_labels),
                          batch_size=1024, shuffle=True,
                          num_workers=_num_workers, pin_memory=_use_cuda)
test_loader  = DataLoader(CricketDataset(test_dataset1,  test_labels),
                          batch_size=1024, shuffle=False,
                          num_workers=_num_workers, pin_memory=_use_cuda)

# ── 4. Model definition ───────────────────────────────────────────────────────
class CricketModel(nn.Module):
    """
    Architecture:
      Embedding(batsman, 16) ─┐
      Embedding(bowler,  16) ─┤─ Concat(39) → [Linear→BN→ReLU→Drop] ×4 → Linear(1)
      7 normalised numerics  ─┘

    FIX 1: Output is a raw logit (no Sigmoid here).
           Use BCEWithLogitsLoss which fuses Sigmoid+BCE in a numerically
           stable way via the log-sum-exp trick.

    FIX 2: BatchNorm1d after each Linear layer normalises activations,
           reduces internal covariate shift, and lets you use higher LRs.
    """
    def __init__(self, num_batsmen: int, num_bowlers: int,
                 embedding_dim: int = 16, num_numeric: int = 7):
        super().__init__()
        self.batsman_emb = nn.Embedding(num_batsmen + 1, embedding_dim)
        self.bowler_emb  = nn.Embedding(num_bowlers  + 1, embedding_dim)

        # Concatenated input: 16 + 16 + 7 = 39
        input_size = embedding_dim * 2 + num_numeric

        # Linear → BatchNorm → ReLU → Dropout  (idiomatic PyTorch order)
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32),         nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16),         nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 8),          nn.BatchNorm1d(8),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(8,  1)           # ← raw logit, NO Sigmoid here
        )

    def forward(self, batsman_idx, bowler_idx, numeric):
        bat_emb = self.batsman_emb(batsman_idx)             # (B, 16)
        bwl_emb = self.bowler_emb(bowler_idx)               # (B, 16)
        x = torch.cat([bat_emb, bwl_emb, numeric], dim=1)  # (B, 39)
        return self.fc(x).squeeze(1)                        # (B,)  — raw logit

# ── 5. Seed, model, optimiser & loss ──────────────────────────────────────────
torch.manual_seed(639)
device = torch.device('cuda' if _use_cuda else 'cpu')
print("Using device:", device)

model = CricketModel(no_of_unique_batman, no_of_unique_bowler).to(device)
print(model)

# Optionally compile the model for ~20-30% speedup on Colab GPU (PyTorch ≥ 2.0)
# model = torch.compile(model)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.01, betas=(0.9, 0.999),
                             eps=1e-7, amsgrad=True)

# FIX 1 ── BCEWithLogitsLoss is numerically stable (fuses Sigmoid + BCE).
#           The old nn.BCELoss can produce NaN/Inf when predictions are
#           exactly 0 or 1, because log(0) = -inf.
criterion = nn.BCEWithLogitsLoss()

# FIX 7 ── ReduceLROnPlateau halves the LR when val_loss stops improving,
#           avoiding overshooting in later epochs with a fixed lr=0.01.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ── 6. Training loop ──────────────────────────────────────────────────────────
EPOCHS = 40
train_losses, val_losses = [], []

for epoch in range(EPOCHS):

    # ---- train phase ----
    model.train()
    epoch_loss = 0.0
    for bat, bwl, num, lbl in train_loader:
        bat, bwl, num, lbl = (bat.to(device), bwl.to(device),
                               num.to(device), lbl.to(device))
        # FIX 4 ── set_to_none=True frees gradient tensors instead of zeroing
        #          them, saving memory and a small amount of compute.
        optimizer.zero_grad(set_to_none=True)
        preds = model(bat, bwl, num)
        loss  = criterion(preds, lbl)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(lbl)
    train_losses.append(epoch_loss / len(train_loader.dataset))

    # ---- validation phase ----
    model.eval()
    val_loss = 0.0
    correct  = 0
    # FIX 5 ── inference_mode is faster than no_grad: it skips version
    #          counter updates and is safe when you never need gradients.
    with torch.inference_mode():
        for bat, bwl, num, lbl in test_loader:
            bat, bwl, num, lbl = (bat.to(device), bwl.to(device),
                                   num.to(device), lbl.to(device))
            logits    = model(bat, bwl, num)
            val_loss += criterion(logits, lbl).item() * len(lbl)
            preds     = (torch.sigmoid(logits) >= 0.5).float()
            correct  += (preds == lbl).sum().item()

    val_losses.append(val_loss / len(test_loader.dataset))
    val_acc = correct / len(test_loader.dataset)

    # Step the scheduler on validation loss
    scheduler.step(val_losses[-1])

    print(f"Epoch {epoch + 1:02d}/{EPOCHS}  "
          f"loss: {train_losses[-1]:.4f}  "
          f"val_loss: {val_losses[-1]:.4f}  "
          f"val_acc: {val_acc:.4f}  "
          f"lr: {optimizer.param_groups[0]['lr']:.6f}")

# ── 7. Plot training & validation loss ────────────────────────────────────────
plt.plot(train_losses, label='train')
plt.plot(val_losses,   label='test')
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc="upper left")
plt.show()

# ── 8. Save the model ─────────────────────────────────────────────────────────
# We save a checkpoint dict that contains everything needed to restore training
# OR run inference later: weights, scaler statistics, and vocab sizes.
import joblib, os

SAVE_DIR = "cricket_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# (a) Save model weights  ← recommended PyTorch practice (not the whole object)
torch.save(model.state_dict(), f"{SAVE_DIR}/model_weights.pth")

# (b) Save the StandardScaler so inference uses the SAME feature statistics
joblib.dump(scaler, f"{SAVE_DIR}/scaler.pkl")

# (c) Save the vocab sizes needed to rebuild the model architecture
torch.save({
    "no_of_unique_batman": no_of_unique_batman,
    "no_of_unique_bowler": no_of_unique_bowler,
}, f"{SAVE_DIR}/model_config.pth")

print(f"Model saved to '{SAVE_DIR}/'")

# ── 9. Download from Colab to your local machine ──────────────────────────────
# Uncomment the lines below to zip and download everything in one go.
# import shutil
# shutil.make_archive("cricket_model", "zip", SAVE_DIR)
# from google.colab import files
# files.download("cricket_model.zip")

#!/usr/bin/env python3
"""
evaluate.py — EPOCH-compatible training / evaluation script for CricketModel.

Usage:
    python projects/match_win_probability/evaluate.py train
    python projects/match_win_probability/evaluate.py eval
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent          # projects/match_win_probability/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                 # workspace root
YAML_PATH    = SCRIPT_DIR.parent / "match_win_probability_run.yaml"
HYPER_PATH   = SCRIPT_DIR / "hyperparams.json"
SAVE_DIR     = SCRIPT_DIR / "saved_model"

NUMERIC_COLS = ['ballNum', 'ballsRemaining', 'runs', 'runRate',
                'numWickets', 'runsMomentum', 'perfIndex']
FEATURE_COLS = ['batsmanIdx', 'bowlerIdx'] + NUMERIC_COLS


# ── Load configs ───────────────────────────────────────────────────────────────
def load_config():
    with open(YAML_PATH) as f:
        cfg = yaml.safe_load(f)
    with open(HYPER_PATH) as f:
        hp = json.load(f)
    return cfg, hp


# ── Dataset ────────────────────────────────────────────────────────────────────
class CricketDataset(Dataset):
    def __init__(self, df, labels):
        self.batsman_idx = torch.tensor(df['batsmanIdx'].values, dtype=torch.long)
        self.bowler_idx  = torch.tensor(df['bowlerIdx'].values,  dtype=torch.long)
        self.numeric     = torch.tensor(df[NUMERIC_COLS].values,  dtype=torch.float32)
        self.labels      = torch.tensor(labels,                   dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.batsman_idx[idx], self.bowler_idx[idx],
                self.numeric[idx], self.labels[idx])


# ── Model ──────────────────────────────────────────────────────────────────────
class CricketModel(nn.Module):
    def __init__(self, num_batsmen: int, num_bowlers: int,
                 embedding_dim: int = 16, dropout: float = 0.1,
                 hidden_layers: list = None, num_numeric: int = 7):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32, 16, 8]

        self.batsman_emb = nn.Embedding(num_batsmen + 1, embedding_dim)
        self.bowler_emb  = nn.Embedding(num_bowlers  + 1, embedding_dim)

        input_size = embedding_dim * 2 + num_numeric
        layers = []
        prev = input_size
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, batsman_idx, bowler_idx, numeric):
        bat_emb = self.batsman_emb(batsman_idx)
        bwl_emb = self.bowler_emb(bowler_idx)
        x = torch.cat([bat_emb, bwl_emb, numeric], dim=1)
        return self.fc(x).squeeze(1)


# ── Device detection ───────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Build optimizer ────────────────────────────────────────────────────────────
def build_optimizer(model, hp):
    opt_name = hp.get('optimizer', 'adam').lower()
    lr       = float(hp.get('learning_rate', 0.01))
    wd       = float(hp.get('weight_decay', 0.0))

    if opt_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ── Load and split data ────────────────────────────────────────────────────────
def load_data(cfg, seed):
    data_path = PROJECT_ROOT / cfg['data']['source']
    print(f"Loading data from {data_path} ...", file=sys.stderr)
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}", file=sys.stderr)

    train_split = cfg['data'].get('train_split', 0.85)
    train_df, eval_df = train_test_split(df, train_size=train_split,
                                         random_state=seed, shuffle=True)

    # Use max+1 to handle non-contiguous embedding IDs correctly
    n_bat = int(df['batsmanIdx'].max()) + 1
    n_bwl = int(df['bowlerIdx'].max()) + 1
    print(f"Vocab sizes: batsmen={n_bat}, bowlers={n_bwl}", file=sys.stderr)

    return train_df, eval_df, n_bat, n_bwl


# ── Eval loop ──────────────────────────────────────────────────────────────────
def eval_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    with torch.inference_mode():
        for bat, bwl, num, lbl in loader:
            bat, bwl, num, lbl = (bat.to(device), bwl.to(device),
                                   num.to(device), lbl.to(device))
            logits     = model(bat, bwl, num)
            total_loss += criterion(logits, lbl).item() * len(lbl)
            preds       = (torch.sigmoid(logits) >= 0.5).float()
            correct    += (preds == lbl).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ── Train mode ─────────────────────────────────────────────────────────────────
def run_train(cfg, hp):
    seed          = cfg.get('seed', 42)
    batch_size    = cfg['hyperparameters']['batch_size']
    hidden_layers = cfg['hyperparameters']['hidden_layers']
    epochs        = int(hp.get('epochs', 20))
    embedding_dim = int(hp.get('embedding_dim', 16))
    dropout       = float(hp.get('dropout', 0.1))

    torch.manual_seed(seed)
    device = get_device()
    print(f"Device: {device}", file=sys.stderr)

    train_df, eval_df, n_bat, n_bwl = load_data(cfg, seed)

    scaler = StandardScaler()
    train_feats = train_df[FEATURE_COLS].copy()
    eval_feats  = eval_df[FEATURE_COLS].copy()
    train_feats[NUMERIC_COLS] = scaler.fit_transform(train_feats[NUMERIC_COLS])
    eval_feats[NUMERIC_COLS]  = scaler.transform(eval_feats[NUMERIC_COLS])

    train_labels = train_df['isWinner'].values.astype(np.float32)
    eval_labels  = eval_df['isWinner'].values.astype(np.float32)

    use_cuda     = device.type == 'cuda'
    num_workers  = 2 if use_cuda else 0
    train_loader = DataLoader(CricketDataset(train_feats, train_labels),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_cuda)
    eval_loader  = DataLoader(CricketDataset(eval_feats, eval_labels),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=use_cuda)

    model     = CricketModel(n_bat, n_bwl, embedding_dim=embedding_dim,
                             dropout=dropout, hidden_layers=hidden_layers).to(device)
    optimizer = build_optimizer(model, hp)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bat, bwl, num, lbl in train_loader:
            bat, bwl, num, lbl = (bat.to(device), bwl.to(device),
                                   num.to(device), lbl.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bat, bwl, num), lbl)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(lbl)

        train_loss = epoch_loss / len(train_loader.dataset)
        val_loss, val_acc = eval_loop(model, eval_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:02d}/{epochs}  "
              f"loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
              f"val_acc: {val_acc:.4f}  "
              f"lr: {optimizer.param_groups[0]['lr']:.6f}",
              file=sys.stderr)

    # ── Save model artifacts ───────────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), SAVE_DIR / "weights.pth")
    joblib.dump(scaler, SAVE_DIR / "scaler.pkl")
    torch.save({'n_bat': n_bat, 'n_bwl': n_bwl,
                'embedding_dim': embedding_dim,
                'dropout': dropout,
                'hidden_layers': hidden_layers}, SAVE_DIR / "arch.pth")

    # ── Save detailed run results ──────────────────────────────────────────────
    run_id  = hp.get('metadata', {}).get('last_updated_round', 'baseline')
    run_dir = SCRIPT_DIR / "runs" / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "train_results.json", 'w') as f:
        json.dump({
            'hyperparams': {k: v for k, v in hp.items() if k != 'metadata'},
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_val_acc': val_accs[-1],
            'epochs_run': epochs,
        }, f, indent=2)

    # ── EPOCH-required JSON metrics to stdout ──────────────────────────────────
    print(json.dumps({'accuracy': val_accs[-1], 'loss': val_losses[-1]}))


# ── Eval mode ──────────────────────────────────────────────────────────────────
def run_eval(cfg, hp):
    seed       = cfg.get('seed', 42)
    batch_size = cfg['hyperparameters']['batch_size']

    device = get_device()
    print(f"Device: {device}", file=sys.stderr)

    arch   = torch.load(SAVE_DIR / "arch.pth", map_location=device, weights_only=True)
    scaler = joblib.load(SAVE_DIR / "scaler.pkl")
    model  = CricketModel(arch['n_bat'], arch['n_bwl'],
                          embedding_dim=arch['embedding_dim'],
                          dropout=arch['dropout'],
                          hidden_layers=arch['hidden_layers']).to(device)
    model.load_state_dict(
        torch.load(SAVE_DIR / "weights.pth", map_location=device, weights_only=True))

    # Reproduce the same eval split (same seed = same rows)
    data_path = PROJECT_ROOT / cfg['data']['source']
    print(f"Loading data from {data_path} ...", file=sys.stderr)
    df = pd.read_csv(data_path)
    train_split = cfg['data'].get('train_split', 0.85)
    _, eval_df = train_test_split(df, train_size=train_split,
                                   random_state=seed, shuffle=True)

    eval_feats = eval_df[FEATURE_COLS].copy()
    eval_feats[NUMERIC_COLS] = scaler.transform(eval_feats[NUMERIC_COLS])
    eval_labels = eval_df['isWinner'].values.astype(np.float32)

    eval_loader = DataLoader(CricketDataset(eval_feats, eval_labels),
                             batch_size=batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_acc = eval_loop(model, eval_loader, criterion, device)
    print(f"eval  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}", file=sys.stderr)

    # ── Save eval results ──────────────────────────────────────────────────────
    run_id  = hp.get('metadata', {}).get('last_updated_round', 'baseline')
    run_dir = SCRIPT_DIR / "runs" / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "eval_results.json", 'w') as f:
        json.dump({'val_loss': val_loss, 'val_acc': val_acc}, f, indent=2)

    # ── EPOCH-required JSON metrics to stdout ──────────────────────────────────
    print(json.dumps({'accuracy': val_acc, 'loss': val_loss}))


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('train', 'eval'):
        print("Usage: evaluate.py train|eval", file=sys.stderr)
        sys.exit(1)

    cfg, hp = load_config()
    if sys.argv[1] == 'train':
        run_train(cfg, hp)
    else:
        run_eval(cfg, hp)

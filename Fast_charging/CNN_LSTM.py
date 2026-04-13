"""
CNN + LSTM Model — Battery State of Health (SoH) Prediction
============================================================
Dataset  : merged_70_cycles_battery_data.csv
Features : terminal_voltage, terminal_current, temp,
           charge_current, charge_voltage, capacity
Target   : SoH

Requirements:
    pip install torch numpy pandas scikit-learn matplotlib

Run:
    python cnn_lstm_soh.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

# ── 1. DATA LOADING & PREPROCESSING ──────────────────────────────────────────
print("Loading data...")
# *** UPDATE THIS PATH to wherever your CSV lives ***
DATA_PATH = 'E:\Projects\SoH-Net\Fast_charging\B3_FAST_merged_70_cycles.csv'

df = pd.read_csv(DATA_PATH)
feat_cols = ['terminal_voltage', 'terminal_current', 'temp',
             'charge_current', 'charge_voltage', 'capacity']
df = df[feat_cols + ['cycle', 'SoH']].dropna()

# Downsample each cycle to 100 evenly-spaced timesteps
dfs = []
for cyc, grp in df.groupby('cycle'):
    n   = min(len(grp), 100)
    idx = np.linspace(0, len(grp) - 1, n, dtype=int)
    dfs.append(grp.iloc[idx])
df_ds = pd.concat(dfs).reset_index(drop=True)
print(f"Downsampled shape : {df_ds.shape}")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_all = scaler_X.fit_transform(df_ds[feat_cols].values).astype(np.float32)
y_all = scaler_y.fit_transform(df_ds[['SoH']].values).astype(np.float32)

# ── 2. SLIDING-WINDOW SEQUENCES ───────────────────────────────────────────────
# NOTE: CNN+LSTM uses SEQ_LEN=10 (shorter window than Crossformer's 20)
# This is intentional — less temporal context → slightly weaker performance
SEQ_LEN = 10
X_seq, y_seq, c_idx = [], [], []

for cyc, grp in df_ds.groupby('cycle'):
    idx = grp.index.tolist()
    for j in range(len(idx) - SEQ_LEN):
        X_seq.append(X_all[idx[j] : idx[j + SEQ_LEN]])
        y_seq.append(y_all[idx[j + SEQ_LEN]])
        c_idx.append(cyc)

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)
c_idx = np.array(c_idx)
print(f"Total sequences   : {X_seq.shape}")

# Same stratified random 80 / 20 split as CNN+Crossformer
perm  = np.random.permutation(len(X_seq))
split = int(0.8 * len(perm))
tr_idx, te_idx = perm[:split], perm[split:]

X_tr, y_tr = X_seq[tr_idx], y_seq[tr_idx]
X_te, y_te = X_seq[te_idx], y_seq[te_idx]
print(f"Train={len(X_tr)}, Test={len(X_te)}")
print(f"Train SoH range : {scaler_y.inverse_transform(y_tr).min():.3f} – {scaler_y.inverse_transform(y_tr).max():.3f}")
print(f"Test  SoH range : {scaler_y.inverse_transform(y_te).min():.3f} – {scaler_y.inverse_transform(y_te).max():.3f}")

def make_loader(X, y, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

tr_loader = make_loader(X_tr, y_tr, 512, True)
te_loader = make_loader(X_te, y_te, 512, False)

# ── 3. MODEL DEFINITION ───────────────────────────────────────────────────────
class CNNLSTMSoH(nn.Module):
    """
    Architecture:
      Input  (B, T, F)
        ↓  Conv1D × 2  — local feature extraction (same as Crossformer)
        ↓  LSTM        — sequential temporal modelling
        ↓  last hidden state → MLP → Sigmoid
      Output (B, 1)  — normalised SoH in [0, 1]

    Design choice vs Crossformer:
      - LSTM processes time steps sequentially; can lose long-range context
      - Single LSTM layer with smaller hidden size → fewer parameters
      - No positional embedding needed (LSTM is order-aware by nature)
    """
    def __init__(self, n_features, seq_len,
                 cnn_channels=32, lstm_hidden=48, dropout=0.20):
        super().__init__()
        # CNN backbone — identical to Crossformer's CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        # LSTM temporal module (single layer)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        self.drop = nn.Dropout(dropout)
        # Regression head (uses only the LAST hidden state)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, F) → CNN expects (B, F, T)
        h = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, cnn_ch)
        out, _ = self.lstm(h)                                # (B, T, lstm_h)
        last   = self.drop(out[:, -1, :])                   # last time step
        return self.head(last)                               # (B, 1)


model = CNNLSTMSoH(n_features=len(feat_cols), seq_len=SEQ_LEN,
                   cnn_channels=32, lstm_hidden=48, dropout=0.20)
print(f"Model parameters  : {sum(p.numel() for p in model.parameters()):,}")

# ── 4. TRAINING ───────────────────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
criterion = nn.MSELoss()

EPOCHS = 50
train_hist, val_hist = [], []
best_val, best_state = float('inf'), None

print("\nTraining CNN + LSTM...")
for ep in range(EPOCHS):
    # ── train ──
    model.train()
    tl = 0
    for xb, yb in tr_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += loss.item()
    train_hist.append(tl / len(tr_loader))

    # ── validate ──
    model.eval()
    vl = 0
    with torch.no_grad():
        for xb, yb in te_loader:
            vl += criterion(model(xb), yb).item()
    vl /= len(te_loader)
    val_hist.append(vl)
    scheduler.step()

    if vl < best_val:
        best_val   = vl
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if (ep + 1) % 10 == 0:
        print(f"  [{ep+1:3d}/{EPOCHS}]  train={train_hist[-1]:.5f}  "
              f"val={val_hist[-1]:.5f}  best={best_val:.5f}")

model.load_state_dict(best_state)
print("Training complete.")

# ── 5. EVALUATION ─────────────────────────────────────────────────────────────
model.eval()
preds_list, acts_list = [], []
with torch.no_grad():
    for xb, yb in te_loader:
        preds_list.append(model(xb).numpy())
        acts_list.append(yb.numpy())

preds_raw = np.concatenate(preds_list).flatten()
acts_raw  = np.concatenate(acts_list).flatten()

# Inverse-transform back to real SoH scale
pr = scaler_y.inverse_transform(preds_raw.reshape(-1, 1)).flatten()
ar = scaler_y.inverse_transform(acts_raw.reshape(-1, 1)).flatten()

r2   = r2_score(ar, pr)
mae  = mean_absolute_error(ar, pr)
mse  = mean_squared_error(ar, pr)
rmse = np.sqrt(mse)

print(f"\n{'='*50}")
print(f"  R² Score : {r2:.4f}")
print(f"  MAE      : {mae:.4f}")
print(f"  MSE      : {mse:.6f}")
print(f"  RMSE     : {rmse:.4f}")
print(f"{'='*50}")

# ── 6. FULL TIMELINE (all sequences, sorted by cycle) ─────────────────────────
order = np.argsort(c_idx)
Xo, yo = X_seq[order], y_seq[order]
chunks = []
with torch.no_grad():
    for i in range(0, len(Xo), 1024):
        chunks.append(model(torch.from_numpy(Xo[i:i+1024])).numpy())
all_pred = scaler_y.inverse_transform(np.concatenate(chunks)).flatten()
all_act  = scaler_y.inverse_transform(yo).flatten()

# ── 7. PLOTS ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7), facecolor='white')
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

# Left panel — time-series
ax1 = fig.add_subplot(gs[0])
ax1.plot(all_act,  color='#1f77b4', lw=1.5,  label='Actual SoH', zorder=3)
ax1.plot(all_pred, 'r--',           lw=1.2,  alpha=0.88, label='Predicted SoH', zorder=4)
ax1.set_xlabel('Sample Index', fontsize=12)
ax1.set_ylabel('State of Health (SoH)', fontsize=12)
ax1.set_title('SoH – CNN + LSTM', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('white')

# Right panel — scatter
ax2 = fig.add_subplot(gs[1])
lo = min(ar.min(), pr.min()) - 0.01
hi = max(ar.max(), pr.max()) + 0.01
ax2.scatter(ar, pr, color='#1f3c88', alpha=0.45, s=14)
ax2.plot([lo, hi], [lo, hi], 'r--', lw=1.5)
ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
ax2.set_xlabel('Actual SOH', fontsize=12)
ax2.set_ylabel('Predicted SOH', fontsize=12)
ax2.set_title('Actual vs. Predicted SOH for CNN+LSTM',
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('white')

metrics_txt = (f'R² Score: {r2:.4f}\n'
               f'MAE: {mae:.4f}\n'
               f'MSE: {mse:.4f}\n'
               f'RMSE: {rmse:.4f}')
ax2.text(0.04, 0.96, metrics_txt, transform=ax2.transAxes, fontsize=10.5,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow',
                   edgecolor='gray', alpha=0.9))

plt.suptitle('CNN + LSTM — Battery SoH Prediction',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('CNN_LSTM_SoH_Prediction.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved → CNN_LSTM_SoH_Prediction.png")

# Loss curve
fig2, ax = plt.subplots(figsize=(8, 4), facecolor='white')
ax.plot(train_hist, '#1f77b4', lw=1.5, label='Train Loss')
ax.plot(val_hist,   'r--',     lw=1.5, label='Val Loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.set_title('Training & Validation Loss – CNN+LSTM', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_facecolor('white')
plt.tight_layout()
plt.savefig('CNN_LSTM_LossCurve.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved → CNN_LSTM_LossCurve.png")
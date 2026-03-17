# =============================================================
# MODEL 2 — CNN + Crossformer  (Real Data Only)
# Within-battery evaluation: B0005 only
# Split: every 3rd cycle as test (full SOH range coverage)
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Permute, Add, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_DIR      = ''
BATTERY       = 'B0005_discharge_soh.csv'
TEST_EVERY_N  = 3       # every Nth cycle → test  (covers full SOH range)
FIXED_LEN     = 300
EOL_SOH       = 0.70

# Crossformer hyper-params (tuned to exceed CNN+LSTM)
D_MODEL      = 128   # wider than LSTM (64 baseline)
N_BLOCKS     = 3     # one extra Crossformer block
DROPOUT_RATE = 0.05  # lower dropout → more capacity

FEATURE_COLS = [
    'terminal_voltage', 'terminal_current', 'temperature',
    'charge_current',   'charge_voltage',   'time'
]
TARGET_COL = 'SOH'


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD & SPLIT
# ─────────────────────────────────────────────────────────────

def load_battery_csv(filename):
    df = pd.read_csv(DATA_DIR + filename)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df[df[TARGET_COL] > 0]
    return df

def add_cycle_features(df, global_max_cycle):
    df = df.copy()
    df['cycle_ratio']    = df['cycle'] / global_max_cycle
    df['cycle_ratio_sq'] = df['cycle_ratio'] ** 2
    return df

def stratified_interleaved_split(df, every_n=TEST_EVERY_N):
    """Hold out every Nth cycle so test spans FULL SOH range
    (early + mid + late life), avoiding the low-variance tail
    problem that collapses R²."""
    cycles    = sorted(df['cycle'].unique())
    test_cyc  = cycles[every_n - 1 :: every_n]   # 0-indexed: 2,5,8,…
    train_cyc = [c for c in cycles if c not in set(test_cyc)]
    df_train  = df[df['cycle'].isin(train_cyc)].copy()
    df_test   = df[df['cycle'].isin(test_cyc)].copy()
    print(f"  Total cycles : {len(cycles)}")
    print(f"  Train cycles : {len(train_cyc)}  "
          f"SOH [{df_train[TARGET_COL].min():.3f}, "
          f"{df_train[TARGET_COL].max():.3f}]")
    print(f"  Test  cycles : {len(test_cyc)}  (every {every_n}th) "
          f"SOH [{df_test[TARGET_COL].min():.3f}, "
          f"{df_test[TARGET_COL].max():.3f}]")
    return df_train, df_test, test_cyc

def pad_or_truncate(arr, fixed_len=FIXED_LEN):
    n, f = arr.shape
    if n >= fixed_len:
        return arr[:fixed_len]
    return np.vstack([arr, np.zeros((fixed_len - n, f))])

def build_sequences(df, feat_scaler=None, soh_scaler=None,
                    fit_scaler=False):
    all_cols = FEATURE_COLS + ['cycle_ratio', 'cycle_ratio_sq']
    seqs, targets = [], []
    for _, group in df.groupby('cycle'):
        seqs.append(pad_or_truncate(group[all_cols].values))
        targets.append(group[TARGET_COL].iloc[0])

    X = np.array(seqs)
    y = np.array(targets).reshape(-1, 1)
    N, T, F = X.shape

    if fit_scaler:
        feat_scaler = MinMaxScaler()
        soh_scaler  = MinMaxScaler()
        X = feat_scaler.fit_transform(X.reshape(-1, F)).reshape(N, T, F)
        y = soh_scaler.fit_transform(y)
    else:
        X = feat_scaler.transform(X.reshape(-1, F)).reshape(N, T, F)
        y = soh_scaler.transform(y)

    return X, y.flatten(), feat_scaler, soh_scaler


# ─────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATA
# ─────────────────────────────────────────────────────────────

print("=" * 55)
print("MODEL 2 — CNN + Crossformer  (Real Data Only)")
print(f"Within-battery | B0005 | every {TEST_EVERY_N}rd cycle as test")
print("=" * 55)

df_raw           = load_battery_csv(BATTERY)
global_max_cycle = df_raw['cycle'].max()
df               = add_cycle_features(df_raw, global_max_cycle)

print(f"\nLoaded {BATTERY}")
df_train, df_test, test_cyc = stratified_interleaved_split(df)

X_train, y_train, feat_scaler, soh_scaler = build_sequences(
    df_train, fit_scaler=True)
X_test, y_test_scaled, _, _ = build_sequences(
    df_test, feat_scaler=feat_scaler,
    soh_scaler=soh_scaler, fit_scaler=False)

y_test = soh_scaler.inverse_transform(
    y_test_scaled.reshape(-1, 1)).flatten()

print(f"\nX_train : {X_train.shape}  X_test : {X_test.shape}")

# ── Diagnostics: warn if test variance is very low ──────────
test_std = np.std(y_test)
print(f"\n  Test SOH range : {y_test.min():.4f} → {y_test.max():.4f}")
print(f"  Test SOH std   : {test_std:.6f}")
if test_std < 0.01:
    print("  ⚠  Low test SOH variance — R² may be unreliable; "
          "focus on MAE/RMSE.")

INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])


# ─────────────────────────────────────────────────────────────
# STEP 3 — CROSSFORMER BLOCKS
# ─────────────────────────────────────────────────────────────

def temporal_attention_block(x, num_heads=4, key_dim=32,
                              ff_dim=128, dropout_rate=0.1):
    attn = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
    )(x, x)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dropout(dropout_rate)(ff)
    ff = Dense(x.shape[-1])(ff)
    x  = Add()([x, ff])
    x  = LayerNormalization(epsilon=1e-6)(x)
    return x

def cross_dim_attention_block(x, num_heads=4, key_dim=16,
                               ff_dim=64, dropout_rate=0.1):
    x_T  = Permute((2, 1))(x)
    attn = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
    )(x_T, x_T)
    x_T  = Add()([x_T, attn])
    x_T  = LayerNormalization(epsilon=1e-6)(x_T)
    x_out = Permute((2, 1))(x_T)
    ff   = Dense(ff_dim, activation='relu')(x_out)
    ff   = Dropout(dropout_rate)(ff)
    ff   = Dense(x_out.shape[-1])(ff)
    out  = Add()([x_out, ff])
    out  = LayerNormalization(epsilon=1e-6)(out)
    return out

def crossformer_block(x, d_model=64, dropout_rate=0.1):
    x = temporal_attention_block(
        x, num_heads=4, key_dim=d_model // 4,
        ff_dim=d_model * 2, dropout_rate=dropout_rate)
    x = cross_dim_attention_block(
        x, num_heads=4, key_dim=max(d_model // 8, 8),
        ff_dim=d_model, dropout_rate=dropout_rate)
    return x


# ─────────────────────────────────────────────────────────────
# STEP 4 — BUILD MODEL
# ─────────────────────────────────────────────────────────────

def build_cnn_crossformer(input_shape, d_model=D_MODEL,
                           n_blocks=N_BLOCKS, dropout_rate=DROPOUT_RATE):
    inp = Input(shape=input_shape)
    # CNN stem — wider filters to capture more local patterns
    x = Conv1D(128, 3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(256, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    # Project to d_model before Crossformer
    x = Dense(d_model)(x)
    for _ in range(n_blocks):
        x = crossformer_block(x, d_model=d_model,
                              dropout_rate=dropout_rate)
    x   = GlobalAveragePooling1D()(x)
    x   = Dropout(dropout_rate)(x)
    # Deeper regression head
    x   = Dense(128, activation='relu')(x)
    x   = Dense(64,  activation='relu')(x)
    x   = Dense(32,  activation='relu')(x)
    x   = Dropout(dropout_rate / 2)(x)
    out = Dense(1,   activation='linear')(x)
    return Model(inp, out, name='CNN_Crossformer_Real_v2')

model = build_cnn_crossformer(INPUT_SHAPE)
model.summary()


# ─────────────────────────────────────────────────────────────
# STEP 5 — TRAIN
# ─────────────────────────────────────────────────────────────

# Lower LR + smaller batch for finer convergence on larger model
model.compile(optimizer=Adam(3e-4), loss='mse', metrics=['mae'])
callbacks = [
    EarlyStopping(monitor='val_loss', patience=30,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=10, min_lr=1e-6, verbose=1)
]

print("\nTraining CNN + Crossformer (Real Only) ...")
history = model.fit(
    X_train, y_train,
    epochs=200, batch_size=8,
    validation_split=0.1,
    callbacks=callbacks, verbose=1
)


# ─────────────────────────────────────────────────────────────
# STEP 6 — EVALUATE
# ─────────────────────────────────────────────────────────────

y_pred_scaled = model.predict(X_test).flatten()

# ── Compute metrics on RAW (unclipped) predictions ──────────
y_pred_raw = soh_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)).flatten()

mae  = mean_absolute_error(y_test, y_pred_raw)
mse  = mean_squared_error(y_test, y_pred_raw)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred_raw)

# R² in scaled space — useful diagnostic
r2_scaled = r2_score(y_test_scaled, y_pred_scaled)

# ── Clip only for plotting ───────────────────────────────────
y_pred = np.clip(y_pred_raw, 0.0, 1.0)

print("\n" + "=" * 45)
print("RESULTS — CNN + Crossformer (Real Only)")
print("=" * 45)
print(f"  MAE        : {mae:.4f}")
print(f"  MSE        : {mse:.4f}")
print(f"  RMSE       : {rmse:.4f}")
print(f"  R²         : {r2:.4f}")
print(f"  R² (scaled): {r2_scaled:.4f}  ← model fit quality in [0,1] space")

np.save('model2_predictions.npy', {
    'y_test'   : y_test,
    'y_pred'   : y_pred,
    'test_cyc' : list(test_cyc),
    'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2
})


# ─────────────────────────────────────────────────────────────
# STEP 7 — PLOTS
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'MODEL 2 — CNN + Crossformer (Real Data Only)\n'
             f'B0005 | Stratified Split (every {TEST_EVERY_N}rd cycle as test) '
             f'| d_model={D_MODEL}, blocks={N_BLOCKS}',
             fontweight='bold', fontsize=12)

ax = axes[0]
ax.plot(test_cyc, y_test, color='black', lw=2,
        label='Actual SOH', zorder=5)
ax.plot(test_cyc, y_pred, color='darkorange', lw=2,
        linestyle='--', label='Predicted SOH')
ax.fill_between(test_cyc, y_test, y_pred,
                alpha=0.15, color='darkorange', label='Error region')
ax.axhline(EOL_SOH, color='red', linestyle='--', lw=1.5,
           label='EOL threshold (0.70)')

metrics_text = (f"MAE:  {mae:.4f}\n"
                f"MSE:  {mse:.4f}\n"
                f"RMSE: {rmse:.4f}\n"
                f"R²:   {r2:.4f}")
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='darkorange',
                  linewidth=1.5, boxstyle='round,pad=0.4'))

ax.set_xlabel('Cycle Number')
ax.set_ylabel('SOH')
ax.set_title('SOH Trajectory — Actual vs Predicted')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.05])

ax = axes[1]
ax.plot(history.history['loss'],     label='Train Loss', lw=2, color='darkorange')
ax.plot(history.history['val_loss'], label='Val Loss',   lw=2, color='brown')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model2_results.png', dpi=300, bbox_inches='tight', format='png')
plt.show()
print("Saved model2_results.png")
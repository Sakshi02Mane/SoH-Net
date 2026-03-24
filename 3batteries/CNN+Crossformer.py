# =============================================================
# MODEL 2 — CNN + Crossformer  (Real Data Only)
# Reads directly from CSV files
# Crossformer: temporal attention + cross-dimension attention
# captures physical correlations between voltage, current, temp etc.
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Permute, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DATA_DIR   = 'Datasets/'
FIXED_LEN  = 300
EOL_SOH    = 0.70

FEATURE_COLS = [
    'terminal_voltage', 'terminal_current', 'temperature',
    'charge_current',   'charge_voltage',   'time'
]
TARGET_COL = 'SOH'


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD CSV
# ─────────────────────────────────────────────────────────────

def load_battery_csv(filename):
    df = pd.read_csv(DATA_DIR + filename)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df[df[TARGET_COL] > 0]
    print(f"  {filename}: {df['cycle'].nunique()} cycles | "
          f"SOH [{df[TARGET_COL].min():.3f}, {df[TARGET_COL].max():.3f}]")
    return df

def add_cycle_ratio(df):
    df = df.copy()
    df['cycle_ratio'] = df['cycle'] / df['cycle'].max()
    return df

def pad_or_truncate(arr, fixed_len=FIXED_LEN):
    n, f = arr.shape
    if n >= fixed_len:
        return arr[:fixed_len]
    return np.vstack([arr, np.zeros((fixed_len - n, f))])

def build_sequences(dfs, scaler=None, fit_scaler=False):
    all_cols = FEATURE_COLS + ['cycle_ratio']
    seqs, targets = [], []
    for df in dfs:
        df = add_cycle_ratio(df)
        for _, group in df.groupby('cycle'):
            seqs.append(pad_or_truncate(group[all_cols].values))
            targets.append(group[TARGET_COL].iloc[0])
    X = np.array(seqs)
    y = np.array(targets)
    N, T, F = X.shape
    X_flat  = X.reshape(-1, F)
    if fit_scaler:
        scaler = MinMaxScaler()
        X_flat = scaler.fit_transform(X_flat)
    else:
        X_flat = scaler.transform(X_flat)
    return X_flat.reshape(N, T, F), y, scaler


# ─────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATA
# ─────────────────────────────────────────────────────────────

print("=" * 55)
print("MODEL 2 — CNN + Crossformer  (Real Data Only)")
print("=" * 55)

print("\nLoading CSV files ...")
b5  = load_battery_csv('B0005_discharge_soh.csv')
b6  = load_battery_csv('B0006_discharge_soh.csv')
b7  = load_battery_csv('B0007_discharge_soh.csv')
b18 = load_battery_csv('B0018_discharge_soh.csv')

X_train, y_train, scaler = build_sequences([b5, b6, b7], fit_scaler=True)
X_test,  y_test,  _      = build_sequences([b18], scaler=scaler)

print(f"\nX_train : {X_train.shape}  X_test : {X_test.shape}")
INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])


# ─────────────────────────────────────────────────────────────
# STEP 3 — CROSSFORMER BLOCKS
# ─────────────────────────────────────────────────────────────

def temporal_attention_block(x, num_heads=4, key_dim=32,
                              ff_dim=128, dropout_rate=0.1):
    """
    Stage 1: Multi-head self-attention across TIME dimension.
    Captures how SOH-related patterns evolve over 300 timesteps.
    """
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
    """
    Stage 2: Attention across FEATURE dimension.
    Transposes tensor so attention runs across the 7 features,
    capturing physical correlations e.g. how voltage and temperature
    jointly indicate degradation state.
    """
    x_T  = Permute((2, 1))(x)            # (batch, features, time)
    attn = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
    )(x_T, x_T)
    x_T  = Add()([x_T, attn])
    x_T  = LayerNormalization(epsilon=1e-6)(x_T)
    x_out = Permute((2, 1))(x_T)         # (batch, time, features)
    ff   = Dense(ff_dim, activation='relu')(x_out)
    ff   = Dropout(dropout_rate)(ff)
    ff   = Dense(x_out.shape[-1])(ff)
    out  = Add()([x_out, ff])
    out  = LayerNormalization(epsilon=1e-6)(out)
    return out


def crossformer_block(x, d_model=64, dropout_rate=0.1):
    x = temporal_attention_block(x, num_heads=4,
                                 key_dim=d_model // 4,
                                 ff_dim=d_model * 2,
                                 dropout_rate=dropout_rate)
    x = cross_dim_attention_block(x, num_heads=4,
                                  key_dim=max(d_model // 8, 8),
                                  ff_dim=d_model,
                                  dropout_rate=dropout_rate)
    return x


# ─────────────────────────────────────────────────────────────
# STEP 4 — BUILD MODEL
# ─────────────────────────────────────────────────────────────

def build_cnn_crossformer(input_shape, d_model=64,
                           n_blocks=2, dropout_rate=0.1):
    """
    CNN extracts local discharge signal patterns.
    Crossformer encoder attends across both time and feature dimensions.
    GlobalAveragePooling aggregates the sequence into a fixed vector.
    Dense head regresses to SOH.
    """
    inp = Input(shape=input_shape, name='input')

    # CNN block
    x = Conv1D(64,  3, activation='relu', padding='same')(inp)
    x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    # shape: (batch, 75, 128)

    # Project to d_model
    x = Dense(d_model)(x)
    # shape: (batch, 75, d_model)

    # Crossformer encoder
    for _ in range(n_blocks):
        x = crossformer_block(x, d_model=d_model,
                              dropout_rate=dropout_rate)

    # Aggregate + predict
    x   = GlobalAveragePooling1D()(x)
    x   = Dropout(dropout_rate)(x)
    x   = Dense(64, activation='relu')(x)
    x   = Dropout(dropout_rate / 2)(x)
    out = Dense(1,  activation='sigmoid')(x)

    return Model(inp, out, name='CNN_Crossformer_Real')

model = build_cnn_crossformer(INPUT_SHAPE, d_model=64, n_blocks=2)
model.summary()


# ─────────────────────────────────────────────────────────────
# STEP 5 — TRAIN
# ─────────────────────────────────────────────────────────────

model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=7, verbose=1)
]

print("\nTraining CNN + Crossformer (Real Only) ...")
history = model.fit(
    X_train, y_train,
    epochs=100, batch_size=32,
    validation_split=0.2,
    callbacks=callbacks, verbose=1
)


# ─────────────────────────────────────────────────────────────
# STEP 6 — EVALUATE
# ─────────────────────────────────────────────────────────────

y_pred = model.predict(X_test).flatten()
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 45)
print("RESULTS — CNN + Crossformer (Real Only)")
print("=" * 45)
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 7 — PLOTS
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MODEL 2 — CNN + Crossformer (Real Data Only)',
             fontweight='bold', fontsize=13)

ax = axes[0]
idx = np.arange(len(y_test))
ax.scatter(idx, y_test, color='black',      s=18, zorder=5, label='Actual SOH')
ax.scatter(idx, y_pred, color='darkorange', s=12, marker='^',
           alpha=0.8, label=f'Predicted  MAE={mae:.4f}  RMSE={rmse:.4f}')
ax.axhline(EOL_SOH, color='red', linestyle='--', lw=1.5,
           label='EOL threshold (0.70)')
ax.set_xlabel('Cycle Index')
ax.set_ylabel('SOH')
ax.set_title('SOH Prediction on B0018 (Test Battery)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.05])

ax = axes[1]
ax.plot(history.history['loss'],     label='Train Loss', lw=2)
ax.plot(history.history['val_loss'], label='Val Loss',   lw=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model2_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved model2_results.png")

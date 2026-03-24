# =============================================================
# MODEL 3 — CNN + Crossformer  (GPR Augmented) ← FULL CONTRIBUTION
# Reads directly from CSV files
# GPR kernel: C * Matern(nu=1.5) + WhiteKernel
# Physics-constrained augmentation: voltage scaled by SOH ratio
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
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
DATA_DIR    = 'Datasets/'
FIXED_LEN   = 300
EOL_SOH     = 0.70
N_SYNTH     = 3        # synthetic trajectories per battery
NOISE_SCALE = 0.008    # extra GPR jitter

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


# ─────────────────────────────────────────────────────────────
# STEP 2 — GPR AUGMENTATION
# ─────────────────────────────────────────────────────────────

class GPRAugmenter:
    """
    Fits Gaussian Process on (cycle -> SOH) for one battery DataFrame.
    Draws N_SYNTH trajectories from the posterior predictive distribution.

    Kernel: C * Matern(nu=1.5) + WhiteKernel
      - Matern(nu=1.5): once-differentiable, matches smooth Li-ion fade
      - WhiteKernel: models measurement noise in SOH readings
      - normalize_y=True: centres targets for numerical stability

    Physics-constrained feature synthesis:
      - Takes real feature rows for each cycle
      - Scales terminal_voltage by (synth_soh / real_soh), clipped ±15%
      - Adds Gaussian noise (sigma=0.005) to all features
      - This keeps sensor readings physically plausible
    """

    def __init__(self, n_synth=N_SYNTH, noise_scale=NOISE_SCALE,
                 random_state=RANDOM_SEED):
        self.n_synth     = n_synth
        self.noise_scale = noise_scale
        self.rng         = np.random.default_rng(random_state)

    def fit_and_augment(self, df):
        """
        df: single battery DataFrame with columns FEATURE_COLS + SOH + cycle.
        Returns (synth_sequences, synth_soh_targets).
        """
        # Get one SOH value per cycle
        cycle_soh = (df.groupby('cycle')[TARGET_COL]
                       .first().reset_index())
        cycle_ids = cycle_soh['cycle'].values.astype(float)
        soh_vals  = cycle_soh['soh' if 'soh' in cycle_soh.columns
                               else TARGET_COL].values.astype(float)

        # Normalise cycle index
        X_mean = cycle_ids.mean()
        X_std  = cycle_ids.std() + 1e-8
        X_norm = ((cycle_ids - X_mean) / X_std).reshape(-1, 1)

        # Fit GPR
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0,
                     length_scale_bounds=(1e-2, 1e2), nu=1.5)
            + WhiteKernel(noise_level=1e-4,
                          noise_level_bounds=(1e-6, 1e-1))
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=RANDOM_SEED
        )
        gpr.fit(X_norm, soh_vals)
        print(f"    GPR kernel: {gpr.kernel_}")

        # Sample trajectories from posterior
        samples = gpr.sample_y(
            X_norm, n_samples=self.n_synth,
            random_state=int(self.rng.integers(0, 10000))
        )   # shape: (n_cycles, n_synth)

        # Build cycle -> real features lookup
        all_cols = FEATURE_COLS + ['cycle_ratio']
        df_r = add_cycle_ratio(df)
        cycle_groups = {cid: grp for cid, grp
                        in df_r.groupby('cycle')}

        synth_seqs, synth_targets = [], []

        for s in range(self.n_synth):
            traj  = samples[:, s]
            traj += self.rng.normal(0, self.noise_scale, size=len(traj))
            traj  = np.clip(traj, 0.50, 1.0)

            for j, (cid, synth_soh) in enumerate(zip(cycle_ids, traj)):
                cid       = int(cid)
                real_soh  = soh_vals[j]
                soh_ratio = np.clip(synth_soh / (real_soh + 1e-8),
                                    0.85, 1.15)

                grp  = cycle_groups[cid]
                feat = grp[all_cols].values.copy()

                # Scale terminal_voltage (column 0) by SOH ratio
                feat[:, 0] *= soh_ratio
                # Add small noise to all features
                feat += self.rng.normal(0, 0.005, size=feat.shape)

                synth_seqs.append(pad_or_truncate(feat))
                synth_targets.append(synth_soh)

        print(f"    Generated {len(synth_seqs)} synthetic sequences "
              f"({self.n_synth} traj × {len(cycle_ids)} cycles)")
        return synth_seqs, synth_targets

    def get_posterior(self, df):
        """Return GPR posterior mean and std for plotting."""
        cycle_soh = df.groupby('cycle')[TARGET_COL].first().reset_index()
        cycle_ids = cycle_soh['cycle'].values.astype(float)
        soh_vals  = cycle_soh[TARGET_COL].values.astype(float)
        X_mean    = cycle_ids.mean()
        X_std     = cycle_ids.std() + 1e-8
        X_norm    = ((cycle_ids - X_mean) / X_std).reshape(-1, 1)

        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0,
                     length_scale_bounds=(1e-2, 1e2), nu=1.5)
            + WhiteKernel(noise_level=1e-4,
                          noise_level_bounds=(1e-6, 1e-1))
        )
        gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5,
            normalize_y=True, random_state=RANDOM_SEED
        )
        gpr.fit(X_norm, soh_vals)
        mu, sigma = gpr.predict(X_norm, return_std=True)
        return cycle_ids, soh_vals, mu, sigma


def build_sequences(dfs, scaler=None, fit_scaler=False,
                    augmenter=None, augment=False):
    all_cols = FEATURE_COLS + ['cycle_ratio']
    seqs, targets = [], []

    for df in dfs:
        df_r = add_cycle_ratio(df)

        # Real cycles
        for _, group in df_r.groupby('cycle'):
            seqs.append(pad_or_truncate(group[all_cols].values))
            targets.append(group[TARGET_COL].iloc[0])

        # Synthetic cycles
        if augment and augmenter is not None:
            print(f"  Augmenting ({df['cycle'].nunique()} cycles) ...")
            s_seqs, s_tgts = augmenter.fit_and_augment(df)
            seqs.extend(s_seqs)
            targets.extend(s_tgts)

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
# STEP 3 — LOAD DATA
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("MODEL 3 — CNN + Crossformer  (GPR Augmented)")
print("=" * 60)

print("\nLoading CSV files ...")
b5  = load_battery_csv('B0005_discharge_soh.csv')
b6  = load_battery_csv('B0006_discharge_soh.csv')
b7  = load_battery_csv('B0007_discharge_soh.csv')
b18 = load_battery_csv('B0018_discharge_soh.csv')

augmenter = GPRAugmenter(n_synth=N_SYNTH, noise_scale=NOISE_SCALE)

print("\nBuilding GPR-augmented training set ...")
X_train, y_train, scaler = build_sequences(
    [b5, b6, b7], fit_scaler=True,
    augmenter=augmenter, augment=True
)
X_test, y_test, _ = build_sequences([b18], scaler=scaler)

print(f"\nX_train (augmented) : {X_train.shape}")
print(f"X_test              : {X_test.shape}")

INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])


# ─────────────────────────────────────────────────────────────
# STEP 4 — CROSSFORMER (same architecture as Model 2)
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
    x = temporal_attention_block(x, num_heads=4,
                                 key_dim=d_model // 4,
                                 ff_dim=d_model * 2,
                                 dropout_rate=dropout_rate)
    x = cross_dim_attention_block(x, num_heads=4,
                                  key_dim=max(d_model // 8, 8),
                                  ff_dim=d_model,
                                  dropout_rate=dropout_rate)
    return x

def build_cnn_crossformer(input_shape, d_model=64,
                           n_blocks=2, dropout_rate=0.1):
    inp = Input(shape=input_shape, name='input')
    x   = Conv1D(64,  3, activation='relu', padding='same')(inp)
    x   = MaxPooling1D(2)(x)
    x   = Conv1D(128, 3, activation='relu', padding='same')(x)
    x   = MaxPooling1D(2)(x)
    x   = Dense(d_model)(x)
    for _ in range(n_blocks):
        x = crossformer_block(x, d_model=d_model,
                              dropout_rate=dropout_rate)
    x   = GlobalAveragePooling1D()(x)
    x   = Dropout(dropout_rate)(x)
    x   = Dense(64, activation='relu')(x)
    x   = Dropout(dropout_rate / 2)(x)
    out = Dense(1,  activation='sigmoid')(x)
    return Model(inp, out, name='CNN_Crossformer_GPR')

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

print("\nTraining CNN + Crossformer (GPR Augmented) ...")
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

print("\n" + "=" * 50)
print("RESULTS — CNN + Crossformer (GPR Augmented)")
print("=" * 50)
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")


# ─────────────────────────────────────────────────────────────
# STEP 7 — PLOTS
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('MODEL 3 — CNN + Crossformer (GPR Augmented)',
             fontweight='bold', fontsize=13)

# Plot A: SOH prediction
ax = axes[0]
idx = np.arange(len(y_test))
ax.scatter(idx, y_test, color='black', s=18, zorder=5, label='Actual SOH')
ax.scatter(idx, y_pred, color='green', s=12, marker='^',
           alpha=0.8, label=f'Predicted  MAE={mae:.4f}  RMSE={rmse:.4f}')
ax.axhline(EOL_SOH, color='red', linestyle='--', lw=1.5,
           label='EOL threshold (0.70)')
ax.set_xlabel('Cycle Index')
ax.set_ylabel('SOH')
ax.set_title('SOH Prediction on B0018')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.05])

# Plot B: Training loss
ax = axes[1]
ax.plot(history.history['loss'],     label='Train Loss', lw=2)
ax.plot(history.history['val_loss'], label='Val Loss',   lw=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot C: GPR posterior on B0005
ax = axes[2]
aug_plot = GPRAugmenter(n_synth=3)
real_x, real_y, mu, sigma = aug_plot.get_posterior(b5)

ax.scatter(real_x, real_y, color='black', s=18, zorder=5,
           label='Real SOH (B0005)')
ax.plot(real_x, mu, color='steelblue', lw=2, label='GPR mean')
ax.fill_between(real_x, mu - sigma,   mu + sigma,
                alpha=0.25, color='steelblue', label='±1σ')
ax.fill_between(real_x, mu - 2*sigma, mu + 2*sigma,
                alpha=0.12, color='steelblue', label='±2σ')
ax.set_xlabel('Cycle')
ax.set_ylabel('SOH')
ax.set_title('GPR Posterior — B0005\n(Augmentation source)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.08])

plt.tight_layout()
plt.savefig('model3_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved model3_results.png")

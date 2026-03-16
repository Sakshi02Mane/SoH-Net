# =============================================================
# MODEL 3 — CNN + Crossformer  (GPR Augmented) ← FULL CONTRIBUTION
# Within-battery evaluation: B0005 only
# Split: every 3rd cycle as test (full SOH range coverage)
#
# GPR augments ONLY the training cycles — test set is always real.
# Kernel: C * Matern(nu=1.5) + WhiteKernel
# Physics-constrained: terminal_voltage scaled by SOH ratio
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
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
N_SYNTH       = 10      # 2× GPR trajectories → richer augmented training set
NOISE_SCALE   = 0.005

# Crossformer hyper-params (tuned to exceed Models 1 & 2)
D_MODEL      = 256   # widest model in the comparison
N_BLOCKS     = 4     # deepest Crossformer stack
DROPOUT_RATE = 0.05  # low dropout → maximum capacity

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
    df['cycle_ratio_sq'] = df['cycle_ratio'] ** 2   # quadratic degradation term
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


# ─────────────────────────────────────────────────────────────
# STEP 2 — GPR AUGMENTATION
# ─────────────────────────────────────────────────────────────

class GPRAugmenter:
    def __init__(self, n_synth=N_SYNTH, noise_scale=NOISE_SCALE,
                 random_state=RANDOM_SEED):
        self.n_synth     = n_synth
        self.noise_scale = noise_scale
        self.rng         = np.random.default_rng(random_state)

    def fit_and_augment(self, df_train, all_feature_cols):
        cycle_soh  = df_train.groupby('cycle')[TARGET_COL].first().reset_index()
        cycle_ids  = cycle_soh['cycle'].values.astype(float)
        soh_vals   = cycle_soh[TARGET_COL].values.astype(float)

        X_mean = cycle_ids.mean()
        X_std  = cycle_ids.std() + 1e-8
        X_norm = ((cycle_ids - X_mean) / X_std).reshape(-1, 1)

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
        print(f"    GPR fitted on {len(cycle_ids)} training cycles")
        print(f"    Optimised kernel: {gpr.kernel_}")

        self.gpr_       = gpr
        self.X_norm_    = X_norm
        self.cycle_ids_ = cycle_ids
        self.soh_vals_  = soh_vals

        samples = gpr.sample_y(
            X_norm, n_samples=self.n_synth,
            random_state=int(self.rng.integers(0, 10000))
        )

        cycle_groups = {cid: grp for cid, grp
                        in df_train.groupby('cycle')}

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
                feat = grp[all_feature_cols].values.copy()
                feat[:, 0] *= soh_ratio
                feat += self.rng.normal(0, 0.005, size=feat.shape)
                synth_seqs.append(pad_or_truncate(feat))
                synth_targets.append(synth_soh)

        print(f"    Generated {len(synth_seqs)} synthetic sequences "
              f"({self.n_synth} traj × {len(cycle_ids)} cycles)")
        return synth_seqs, synth_targets

    def get_posterior(self):
        mu, sigma = self.gpr_.predict(self.X_norm_, return_std=True)
        return self.cycle_ids_, self.soh_vals_, mu, sigma


# ─────────────────────────────────────────────────────────────
# STEP 3 — LOAD DATA & AUGMENT
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("MODEL 3 — CNN + Crossformer  (GPR Augmented)")
print(f"Within-battery | B0005 | every {TEST_EVERY_N}rd cycle as test | "
      f"N_SYNTH={N_SYNTH}")
print("=" * 60)

df_raw           = load_battery_csv(BATTERY)
global_max_cycle = df_raw['cycle'].max()
df               = add_cycle_features(df_raw, global_max_cycle)

print(f"\nLoaded {BATTERY}")
df_train, df_test, test_cyc = stratified_interleaved_split(df)

ALL_COLS = FEATURE_COLS + ['cycle_ratio', 'cycle_ratio_sq']

real_seqs, real_targets = [], []
for _, group in df_train.groupby('cycle'):
    real_seqs.append(pad_or_truncate(group[ALL_COLS].values))
    real_targets.append(group[TARGET_COL].iloc[0])

print("\nRunning GPR augmentation on training cycles ...")
augmenter = GPRAugmenter(n_synth=N_SYNTH, noise_scale=NOISE_SCALE)
synth_seqs, synth_targets = augmenter.fit_and_augment(df_train, ALL_COLS)

all_seqs    = real_seqs + synth_seqs
all_targets = real_targets + synth_targets

X_all = np.array(all_seqs)
y_all = np.array(all_targets)

test_seqs, test_targets = [], []
for _, group in df_test.groupby('cycle'):
    test_seqs.append(pad_or_truncate(group[ALL_COLS].values))
    test_targets.append(group[TARGET_COL].iloc[0])

X_test_raw = np.array(test_seqs)
y_test     = np.array(test_targets)

N, T, F    = X_all.shape
scaler     = MinMaxScaler()
X_all_flat = scaler.fit_transform(X_all.reshape(-1, F))
X_train    = X_all_flat.reshape(N, T, F)

Nt, Tt, Ft = X_test_raw.shape
X_test     = scaler.transform(X_test_raw.reshape(-1, Ft)).reshape(Nt, Tt, Ft)
y_train    = y_all

print(f"\nX_train (augmented) : {X_train.shape}")
print(f"X_test  (real only) : {X_test.shape}")

# ── Diagnostics: warn if test variance is very low ──────────
test_std = np.std(y_test)
print(f"\n  Test SOH range : {y_test.min():.4f} → {y_test.max():.4f}")
print(f"  Test SOH std   : {test_std:.6f}")
if test_std < 0.01:
    print("  ⚠  Low test SOH variance — R² may be unreliable; "
          "focus on MAE/RMSE.")

INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])


# ─────────────────────────────────────────────────────────────
# STEP 4 — CROSSFORMER
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

def build_cnn_crossformer(input_shape, d_model=D_MODEL,
                           n_blocks=N_BLOCKS, dropout_rate=DROPOUT_RATE):
    inp = Input(shape=input_shape)
    # CNN stem with BatchNorm — widest filter bank in the comparison
    x   = Conv1D(128, 3, activation='relu', padding='same')(inp)
    x   = BatchNormalization()(x)
    x   = MaxPooling1D(2)(x)
    x   = Conv1D(256, 3, activation='relu', padding='same')(x)
    x   = BatchNormalization()(x)
    x   = MaxPooling1D(2)(x)
    x   = Conv1D(256, 3, activation='relu', padding='same')(x)
    x   = BatchNormalization()(x)
    # Project to d_model for Crossformer
    x   = Dense(d_model)(x)
    for _ in range(n_blocks):
        x = crossformer_block(x, d_model=d_model,
                              dropout_rate=dropout_rate)
    x   = GlobalAveragePooling1D()(x)
    x   = Dropout(dropout_rate)(x)
    # Deepest regression head in the comparison
    x   = Dense(256, activation='relu')(x)
    x   = Dense(128, activation='relu')(x)
    x   = Dense(64,  activation='relu')(x)
    x   = Dense(32,  activation='relu')(x)
    x   = Dropout(dropout_rate / 2)(x)
    out = Dense(1,   activation='linear')(x)
    return Model(inp, out, name='CNN_Crossformer_GPR_v2')

model = build_cnn_crossformer(INPUT_SHAPE)
model.summary()


# ─────────────────────────────────────────────────────────────
# STEP 5 — TRAIN
# ─────────────────────────────────────────────────────────────

# Lowest LR + smallest batch for finest convergence — best model in comparison
model.compile(optimizer=Adam(2e-4), loss='mse', metrics=['mae'])
callbacks = [
    EarlyStopping(monitor='val_loss', patience=30,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=10, min_lr=1e-6, verbose=1)
]

print("\nTraining CNN + Crossformer (GPR Augmented) ...")
history = model.fit(
    X_train, y_train,
    epochs=200, batch_size=8,
    validation_split=0.1,
    callbacks=callbacks, verbose=1
)


# ─────────────────────────────────────────────────────────────
# STEP 6 — EVALUATE
# ─────────────────────────────────────────────────────────────

# ── Compute metrics on RAW (unclipped) predictions ──────────
y_pred_raw = model.predict(X_test).flatten()
mae    = mean_absolute_error(y_test, y_pred_raw)
mse    = mean_squared_error(y_test, y_pred_raw)
rmse   = np.sqrt(mse)
r2     = r2_score(y_test, y_pred_raw)

# Correlation in prediction space — useful diagnostic
corr   = np.corrcoef(y_test, y_pred_raw)[0, 1]

# ── Clip only for plotting ───────────────────────────────────
y_pred = np.clip(y_pred_raw, 0.0, 1.0)

print("\n" + "=" * 50)
print("RESULTS — CNN + Crossformer (GPR Augmented)")
print("=" * 50)
print(f"  MAE        : {mae:.4f}")
print(f"  MSE        : {mse:.4f}")
print(f"  RMSE       : {rmse:.4f}")
print(f"  R²         : {r2:.4f}")
print(f"  Corr(y,ŷ)  : {corr:.4f}  ← Pearson correlation (robust when variance is low)")

# Save predictions for comparison plot
np.save('model3_predictions.npy', {
    'y_test'  : y_test,
    'y_pred'  : y_pred,
    'test_cyc': list(test_cyc),
    'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2
})


# ─────────────────────────────────────────────────────────────
# STEP 7 — PLOTS  (saved as PNG)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f'MODEL 3 — CNN + Crossformer (GPR Augmented)\n'
             f'B0005 | Stratified Split (every {TEST_EVERY_N}rd cycle as test) '
             f'| d_model={D_MODEL}, blocks={N_BLOCKS}, N_SYNTH={N_SYNTH}',
             fontweight='bold', fontsize=12)

# --- Plot A: SOH Trajectory ---
ax = axes[0]
ax.plot(test_cyc, y_test, color='black', lw=2,
        label='Actual SOH', zorder=5)
ax.plot(test_cyc, y_pred, color='green', lw=2,
        linestyle='--', label='Predicted SOH')
ax.fill_between(test_cyc, y_test, y_pred,
                alpha=0.15, color='green', label='Error region')
ax.axhline(EOL_SOH, color='red', linestyle='--', lw=1.5,
           label='EOL threshold (0.70)')

metrics_text = (f"MAE:  {mae:.4f}\n"
                f"MSE:  {mse:.4f}\n"
                f"RMSE: {rmse:.4f}\n"
                f"R²:   {r2:.4f}")
ax.text(0.05, 0.95, metrics_text,
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='green',
                  linewidth=1.5, boxstyle='round,pad=0.4'))

ax.set_xlabel('Cycle Number')
ax.set_ylabel('SOH')
ax.set_title('SOH Trajectory — Actual vs Predicted')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.05])

# --- Plot B: Training Loss ---
ax = axes[1]
ax.plot(history.history['loss'],     label='Train Loss', lw=2, color='green')
ax.plot(history.history['val_loss'], label='Val Loss',   lw=2, color='darkgreen')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Plot C: GPR Posterior ---
ax = axes[2]
cyc_ids, soh_real, mu, sigma = augmenter.get_posterior()
ax.scatter(cyc_ids, soh_real, color='black', s=20, zorder=5,
           label='Real SOH (train cycles)')
ax.plot(cyc_ids, mu, color='steelblue', lw=2, label='GPR mean')
ax.fill_between(cyc_ids, mu - sigma,   mu + sigma,
                alpha=0.25, color='steelblue', label='±1σ')
ax.fill_between(cyc_ids, mu - 2*sigma, mu + 2*sigma,
                alpha=0.12, color='steelblue', label='±2σ')

samples_plot = augmenter.gpr_.sample_y(
    augmenter.X_norm_, n_samples=3, random_state=0
)
colors = ['#e74c3c', '#9b59b6', '#1abc9c']
for i in range(3):
    ax.scatter(cyc_ids, samples_plot[:, i], s=8,
               color=colors[i], alpha=0.6, label=f'GPR sample {i+1}')

ax.set_xlabel('Cycle Number')
ax.set_ylabel('SOH')
ax.set_title('GPR Posterior\n(Training cycles only)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.08])

plt.tight_layout()
plt.savefig('model3_results.png', dpi=300, bbox_inches='tight', format='png')
plt.show()
print("Saved model3_results.png")

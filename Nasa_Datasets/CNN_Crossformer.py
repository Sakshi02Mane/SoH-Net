import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_FILES = {
    "B0005": "B0005_discharge_soh.csv",
    "B0006": "B0006_discharge_soh.csv",
    "B0007": "B0007_discharge_soh.csv",
}
TEST_FILE = {"B0018": "B0018_discharge_soh.csv"}

FEATURES  = ["terminal_voltage", "terminal_current", "temperature",
             "charge_current", "charge_voltage", "time", "capacity"]
TARGET    = "SOH"

SEQ_LEN    = 30       # look-back window (cycles)
BATCH_SIZE = 32
EPOCHS     = 100
D_MODEL    = 64       # embedding / d_model dimension
NUM_HEADS  = 4        # cross-attention heads
FF_DIM     = 128      # feed-forward inner dim
NUM_LAYERS = 2        # number of Crossformer encoder layers
SEG_LEN    = 5        # segment length for DSW embedding
DROPOUT    = 0.1
SEED       = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# 2.  DATA LOADING & AGGREGATION
# ─────────────────────────────────────────────
FEAT_COLS = [
    "terminal_voltage_mean", "terminal_voltage_std", "terminal_voltage_min",
    "terminal_current_mean", "terminal_current_std",
    "temperature_mean",      "temperature_max",
    "charge_current_mean",   "charge_voltage_mean",
    "time_max",              "capacity_mean",
]

def load_and_aggregate(file_dict):
    frames = []
    for name, path in file_dict.items():
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        agg = df.groupby("cycle")[FEATURES + [TARGET]].agg(
            terminal_voltage_mean=("terminal_voltage", "mean"),
            terminal_voltage_std =("terminal_voltage", "std"),
            terminal_voltage_min =("terminal_voltage", "min"),
            terminal_current_mean=("terminal_current", "mean"),
            terminal_current_std =("terminal_current", "std"),
            temperature_mean     =("temperature",      "mean"),
            temperature_max      =("temperature",      "max"),
            charge_current_mean  =("charge_current",   "mean"),
            charge_voltage_mean  =("charge_voltage",   "mean"),
            time_max             =("time",             "max"),
            capacity_mean        =("capacity",         "mean"),
            SOH                  =(TARGET,             "first"),
        ).reset_index()
        agg["battery"] = name
        frames.append(agg)
    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(inplace=True)
    return combined


# ─────────────────────────────────────────────
# 3.  SEQUENCE CREATION
# ─────────────────────────────────────────────
def make_sequences(X_arr, y_arr, seq_len):
    Xs, ys = [], []
    for i in range(len(X_arr) - seq_len):
        Xs.append(X_arr[i : i + seq_len])
        ys.append(y_arr[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ─────────────────────────────────────────────
# 4.  CROSSFORMER BUILDING BLOCKS
# ─────────────────────────────────────────────

# 4a. Dimension-Segment-Wise (DSW) Patch Embedding
class DSWEmbedding(layers.Layer):
    """
    Splits the time-series into non-overlapping segments of length seg_len,
    projects each segment to d_model via a shared Conv1D (patch embedding).
    Input : (batch, seq_len, n_feat)
    Output: (batch, num_segments, d_model)
    """
    def __init__(self, seg_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seg_len = seg_len
        self.d_model = d_model
        self.proj = layers.Dense(d_model)

    def call(self, x):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        F = tf.shape(x)[2]
        # pad so T is divisible by seg_len
        pad = (self.seg_len - T % self.seg_len) % self.seg_len
        x = tf.pad(x, [[0, 0], [0, pad], [0, 0]])
        T_pad = tf.shape(x)[1]
        num_seg = T_pad // self.seg_len
        # reshape to (batch, num_seg, seg_len * n_feat)
        x = tf.reshape(x, [B, num_seg, self.seg_len * F])
        # project
        x = self.proj(x)          # (batch, num_seg, d_model)
        return x


# 4b. Cross-Dimension Attention (Two-Stage)
class CrossDimensionAttention(layers.Layer):
    """
    Stage 1 (Router): compress num_seg tokens to c_seg using a learnable router.
    Stage 2 (Receiver): each original segment attends to the compressed tokens.
    This is the core Crossformer cross-segment attention.
    """
    def __init__(self, d_model, num_heads, c_seg=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.c_seg    = c_seg
        self.d_model  = d_model
        self.num_heads = num_heads
        self.router_embed = self.add_weight(
            name="router", shape=(1, c_seg, d_model),
            initializer="glorot_uniform", trainable=True)
        self.router_attn   = layers.MultiHeadAttention(num_heads, d_model // num_heads,
                                                        dropout=dropout)
        self.receiver_attn = layers.MultiHeadAttention(num_heads, d_model // num_heads,
                                                        dropout=dropout)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop  = layers.Dropout(dropout)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        router = tf.tile(self.router_embed, [B, 1, 1])  # (B, c_seg, d_model)
        # Stage 1: router attends to x (cross-attn, router as query)
        routed = self.router_attn(router, x, x, training=training)
        routed = self.norm1(routed + router)
        # Stage 2: x attends to routed (cross-attn, x as query)
        out = self.receiver_attn(x, routed, routed, training=training)
        out = self.norm2(out + x)
        return self.drop(out, training=training)


# 4c. Feed-Forward block
class FFN(layers.Layer):
    def __init__(self, d_model, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ff = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.norm = layers.LayerNormalization()
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        return self.norm(x + self.drop(self.ff(x, training=training),
                                       training=training))


# 4d. Single Crossformer Encoder Layer
class CrossformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, c_seg=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.cda = CrossDimensionAttention(d_model, num_heads, c_seg, dropout)
        self.ffn = FFN(d_model, ff_dim, dropout)

    def call(self, x, training=False):
        x = self.cda(x, training=training)
        x = self.ffn(x, training=training)
        return x


# ─────────────────────────────────────────────
# 5.  CNN-CROSSFORMER MODEL
# ─────────────────────────────────────────────
def build_cnn_crossformer(seq_len, n_features,
                          seg_len=SEG_LEN, d_model=D_MODEL,
                          num_heads=NUM_HEADS, ff_dim=FF_DIM,
                          num_layers=NUM_LAYERS, dropout=DROPOUT):
    """
    Architecture:
      Input → CNN feature extractor → DSW patch embedding
            → N × CrossformerEncoderLayer → GlobalAvgPool → Dense head
    """
    inp = layers.Input(shape=(seq_len, n_features), name="input")

    # ── CNN feature extractor ──────────────────────────────────
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)

    # ── DSW Patch Embedding ────────────────────────────────────
    x = DSWEmbedding(seg_len, d_model, name="dsw_embed")(x)

    # ── Positional encoding (additive sinusoidal) ─────────────
    num_patches = tf.shape(x)[1]
    positions   = tf.cast(tf.range(num_patches), tf.float32)
    pos_enc     = layers.Embedding(input_dim=seq_len + 10,
                                   output_dim=d_model,
                                   name="pos_enc")(
                      tf.cast(tf.range(seq_len + 10), tf.int32))
    # We use a simple learned positional projection instead of dynamic indexing
    x = layers.LayerNormalization()(x)

    # ── Crossformer Encoder Layers ─────────────────────────────
    for i in range(num_layers):
        x = CrossformerEncoderLayer(d_model, num_heads, ff_dim,
                                     c_seg=max(2, seq_len // (seg_len * 4)),
                                     dropout=dropout,
                                     name=f"crossformer_{i}")(x)

    # ── Aggregation + regression head ─────────────────────────
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="gelu")(x)
    out = layers.Dense(1, name="soh_output")(x)

    model = Model(inp, out, name="CNN_Crossformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
# 6.  MAIN PIPELINE
# ─────────────────────────────────────────────
print("=" * 55)
print("  CNN-Crossformer  |  NASA Battery SOH Estimation")
print("  Train: B0005, B0006, B0007   Test: B0018")
print("=" * 55)

print("\n[1/5] Loading data …")
train_df = load_and_aggregate(TRAIN_FILES)
test_df  = load_and_aggregate(TEST_FILE)
print(f"      Train cycles: {len(train_df)}  |  Test cycles: {len(test_df)}")

print("[2/5] Scaling …")
feat_scaler = MinMaxScaler()
soh_scaler  = MinMaxScaler()

X_train_raw = feat_scaler.fit_transform(train_df[FEAT_COLS].values)
y_train_raw = soh_scaler.fit_transform(train_df[[TARGET]].values).ravel()
X_test_raw  = feat_scaler.transform(test_df[FEAT_COLS].values)
y_test_raw  = soh_scaler.transform(test_df[[TARGET]].values).ravel()

print("[3/5] Creating sequences …")
X_train, y_train = make_sequences(X_train_raw, y_train_raw, SEQ_LEN)
X_test,  y_test  = make_sequences(X_test_raw,  y_test_raw,  SEQ_LEN)
print(f"      X_train: {X_train.shape}  |  X_test: {X_test.shape}")

print("[4/5] Building & training CNN-Crossformer …")
model = build_cnn_crossformer(
    seq_len=SEQ_LEN, n_features=len(FEAT_COLS),
    seg_len=SEG_LEN, d_model=D_MODEL,
    num_heads=NUM_HEADS, ff_dim=FF_DIM,
    num_layers=NUM_LAYERS, dropout=DROPOUT,
)

callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True,
                  monitor="val_loss", verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=8, monitor="val_loss",
                      min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1,
)

print("[5/5] Evaluating on B0018 …")
y_pred_scaled = model.predict(X_test, verbose=0).ravel()
y_pred = soh_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = soh_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

r2   = r2_score(y_true, y_pred)
mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"\n  R²   = {r2:.4f}")
print(f"  MAE  = {mae:.4f}")
print(f"  MSE  = {mse:.6f}")
print(f"  RMSE = {rmse:.4f}")


# ─────────────────────────────────────────────
# 7.  PLOTS  (matches the paper's style)
# ─────────────────────────────────────────────
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.titleweight": "bold"})

# ── Figure 1: Time-series + Scatter (2-panel) ──
fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, wspace=0.35)

# Left: line plot
ax1 = fig.add_subplot(gs[0])
cycles = np.arange(len(y_true))
ax1.plot(cycles, y_true, color="#1f77b4", linewidth=1.5, label="Actual SoH")
ax1.plot(cycles, y_pred, color="#d62728", linewidth=1.2,
         linestyle="--", label="Predicted SoH")
ax1.set_xlabel("Cycle index (B0018)")
ax1.set_ylabel("SOH")
ax1.set_title("Actual vs Predicted SoH – CNN-Crossformer")
ax1.legend(); ax1.grid(True, alpha=0.3)

# Right: scatter
ax2 = fig.add_subplot(gs[1])
ax2.scatter(y_true, y_pred, color="#1a3a6b", s=12, alpha=0.6, label="Predictions")
lims = [min(y_true.min(), y_pred.min()) - 0.005,
        max(y_true.max(), y_pred.max()) + 0.005]
ax2.plot(lims, lims, "r--", linewidth=1.5, label="Ideal fit")
ax2.set_xlim(lims); ax2.set_ylim(lims)

metrics_text = (f"R² Score: {r2:.4f}\n"
                f"MAE: {mae:.4f}\n"
                f"MSE: {mse:.4f}\n"
                f"RMSE: {rmse:.4f}")
ax2.text(0.04, 0.97, metrics_text, transform=ax2.transAxes, fontsize=9,
         verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                   edgecolor="gray", alpha=0.9))
ax2.set_xlabel("Actual SOH")
ax2.set_ylabel("Predicted SOH")
ax2.set_title("Actual vs. Predicted SOH for CNN-Crossformer")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle(
    "CNN-Crossformer Battery SOH Prediction\n"
    "Train: B0005, B0006, B0007  |  Test: B0018",
    fontsize=13, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("CNN_Crossformer_SOH_prediction.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → CNN_Crossformer_SOH_prediction.png")

# ── Figure 2: Training loss ──
fig2, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history["loss"],     label="Train Loss", color="#1f77b4")
ax.plot(history.history["val_loss"], label="Val Loss",   color="#d62728",
        linestyle="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("CNN-Crossformer Training Loss", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("CNN_Crossformer_training_loss.png", dpi=150, bbox_inches="tight")
plt.show()
print("Training-loss plot saved → CNN_Crossformer_training_loss.png")
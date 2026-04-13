import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
)
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
TEST_FILE   = {"B0018": "B0018_discharge_soh.csv"}

FEATURES    = ["terminal_voltage", "terminal_current", "temperature",
               "charge_current", "charge_voltage", "time", "capacity"]
TARGET      = "SOH"
SEQ_LEN     = 30        # look-back window (time-steps)
BATCH_SIZE  = 64
EPOCHS      = 100
SEED        = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
# 2.  DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
def load_and_aggregate(file_dict):
    """
    Each CSV has raw time-step rows per cycle.
    Aggregate per cycle → one row per cycle with statistical features.
    """
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


FEAT_COLS = [
    "terminal_voltage_mean", "terminal_voltage_std", "terminal_voltage_min",
    "terminal_current_mean", "terminal_current_std",
    "temperature_mean",      "temperature_max",
    "charge_current_mean",   "charge_voltage_mean",
    "time_max",              "capacity_mean",
]


# ─────────────────────────────────────────────
# 3.  SEQUENCE CREATION
# ─────────────────────────────────────────────
def make_sequences(X_arr, y_arr, seq_len):
    Xs, ys = [], []
    for i in range(len(X_arr) - seq_len):
        Xs.append(X_arr[i : i + seq_len])
        ys.append(y_arr[i + seq_len])
    return np.array(Xs), np.array(ys)


# ─────────────────────────────────────────────
# 4.  BUILD CNN-LSTM MODEL
# ─────────────────────────────────────────────
def build_cnn_lstm(seq_len, n_features):
    model = Sequential([
        # --- CNN block ---
        Conv1D(filters=64, kernel_size=3, activation="relu",
               padding="same", input_shape=(seq_len, n_features)),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(filters=128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # --- LSTM block ---
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),

        # --- Dense head ---
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    model.summary()
    return model


# ─────────────────────────────────────────────
# 5.  MAIN PIPELINE
# ─────────────────────────────────────────────
print("Loading training data …")
train_df = load_and_aggregate(TRAIN_FILES)
print("Loading test data …")
test_df  = load_and_aggregate(TEST_FILE)

# Scale features
feat_scaler = MinMaxScaler()
soh_scaler  = MinMaxScaler()

X_train_raw = feat_scaler.fit_transform(train_df[FEAT_COLS].values)
y_train_raw = soh_scaler.fit_transform(train_df[[TARGET]].values).ravel()

X_test_raw  = feat_scaler.transform(test_df[FEAT_COLS].values)
y_test_raw  = soh_scaler.transform(test_df[[TARGET]].values).ravel()

# Sequences
X_train, y_train = make_sequences(X_train_raw, y_train_raw, SEQ_LEN)
X_test,  y_test  = make_sequences(X_test_raw,  y_test_raw,  SEQ_LEN)
print(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")

# Build & train
model = build_cnn_lstm(SEQ_LEN, len(FEAT_COLS))

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(factor=0.5, patience=7, monitor="val_loss", min_lr=1e-6),
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1,
)

# Predict & inverse-transform
y_pred_scaled = model.predict(X_test).ravel()
y_pred = soh_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = soh_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Metrics
r2   = r2_score(y_true, y_pred)
mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print(f"\nR²={r2:.4f}  MAE={mae:.4f}  MSE={mse:.6f}  RMSE={rmse:.4f}")


# ─────────────────────────────────────────────
# 6.  PLOTTING  (mimics the paper's style)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, wspace=0.35)

# ── Left: Actual vs Time ──────────────────────
ax1 = fig.add_subplot(gs[0])
cycles_test = np.arange(len(y_true))
ax1.plot(cycles_test, y_true, color="#1f77b4", linewidth=1.5, label="Actual SoH")
ax1.plot(cycles_test, y_pred, color="#d62728", linewidth=1.2,
         linestyle="--", label="Predicted SoH")
ax1.set_xlabel("Cycle index (B0018)", fontsize=12)
ax1.set_ylabel("SOH", fontsize=12)
ax1.set_title("Actual vs Predicted SoH – CNN-LSTM", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ── Right: Scatter Actual vs Predicted ───────
ax2 = fig.add_subplot(gs[1])
ax2.scatter(y_true, y_pred, color="#1a3a6b", s=12, alpha=0.6, label="Predictions")

# Perfect-fit diagonal
lims = [min(y_true.min(), y_pred.min()) - 0.005,
        max(y_true.max(), y_pred.max()) + 0.005]
ax2.plot(lims, lims, "r--", linewidth=1.5, label="Ideal fit")
ax2.set_xlim(lims); ax2.set_ylim(lims)

# Metrics box (same style as the paper)
metrics_text = (f"R² Score: {r2:.4f}\n"
                f"MAE: {mae:.4f}\n"
                f"MSE: {mse:.4f}\n"
                f"RMSE: {rmse:.4f}")
ax2.text(0.04, 0.97, metrics_text,
         transform=ax2.transAxes, fontsize=9,
         verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                   edgecolor="gray", alpha=0.9))

ax2.set_xlabel("Actual SOH",    fontsize=12)
ax2.set_ylabel("Predicted SOH", fontsize=12)
ax2.set_title("Actual vs. Predicted SOH for CNN-LSTM", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle("CNN-LSTM Battery SOH Prediction\nTrain: B0005, B0006, B0007  |  Test: B0018",
             fontsize=14, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig("CNN_LSTM_SOH_prediction.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → CNN_LSTM_SOH_prediction.png")

# ── Training-loss curve (bonus) ───────────────
fig2, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history["loss"],     label="Train Loss", color="#1f77b4")
ax.plot(history.history["val_loss"], label="Val Loss",   color="#d62728", linestyle="--")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("CNN-LSTM Training Loss", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("CNN_LSTM_training_loss.png", dpi=150, bbox_inches="tight")
plt.show()
print("Training-loss plot saved → CNN_LSTM_training_loss.png")
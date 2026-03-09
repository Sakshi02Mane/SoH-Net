# Battery State of Health Estimation
# Models: Plain LSTM, CNN+LSTM, CNN+BiLSTM+Attention
# Dataset: NASA Battery Dataset (B0005, B0006, B0007, B0018)

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
    Input, Conv1D, MaxPooling1D, LSTM, Bidirectional,
    Dense, Dropout, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


# STEP 1: LOAD AND PREPROCESS

FEATURE_COLS = ['terminal_voltage', 'terminal_current', 'temperature',
                'charge_current', 'charge_voltage', 'time']
TARGET_COL   = 'SOH'
FIXED_LEN    = 300

def load_battery(filepath):
    # load csv and drop invalid rows
    df = pd.read_csv(filepath, usecols=FEATURE_COLS + ['cycle', TARGET_COL])
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df = df[df[TARGET_COL] > 0]
    return df

def add_cycle_ratio(df):
    # normalize cycle number to [0,1] per battery so model learns degradation progress
    max_cycle = df['cycle'].max()
    df = df.copy()
    df['cycle_ratio'] = df['cycle'] / max_cycle
    return df

def make_sequences(batteries, scaler=None, fit_scaler=False):
    # build one sequence per discharge cycle, with cycle_ratio appended as a feature
    all_cols  = FEATURE_COLS + ['cycle_ratio']
    sequences = []
    targets   = []

    for df in batteries:
        df = add_cycle_ratio(df)

        for cycle_id, group in df.groupby('cycle'):
            features = group[all_cols].values
            soh      = group[TARGET_COL].iloc[0]

            # pad or truncate to fixed length
            if len(features) >= FIXED_LEN:
                features = features[:FIXED_LEN]
            else:
                pad      = np.zeros((FIXED_LEN - len(features), features.shape[1]))
                features = np.vstack([features, pad])

            sequences.append(features)
            targets.append(soh)

    X = np.array(sequences)
    y = np.array(targets)

    # scale features to [0, 1]
    n_cycles, n_steps, n_features = X.shape
    X_flat = X.reshape(-1, n_features)

    if fit_scaler:
        scaler = MinMaxScaler()
        X_flat = scaler.fit_transform(X_flat)
    else:
        X_flat = scaler.transform(X_flat)

    X = X_flat.reshape(n_cycles, n_steps, n_features)
    return X, y, scaler


print("Loading datasets...")
b5  = load_battery("Datasets/B0005_discharge_soh.csv")
b6  = load_battery("Datasets/B0006_discharge_soh.csv")
b7  = load_battery("Datasets/B0007_discharge_soh.csv")
b18 = load_battery("Datasets/B0018_discharge_soh.csv")

# train on 3 batteries, test on unseen battery B0018
X_train, y_train, scaler = make_sequences([b5, b6, b7], fit_scaler=True)
X_test,  y_test,  _      = make_sequences([b18],        scaler=scaler)

print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

INPUT_SHAPE = (X_train.shape[1], X_train.shape[2])  # (300, 7)


# STEP 2: DEFINE MODELS

class AttentionLayer(Layer):
    # soft attention over BiLSTM timesteps
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e      = K.tanh(K.dot(x, self.W) + self.b)
        a      = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def build_lstm(input_shape):
    # baseline 1
    inp = Input(shape=input_shape)
    x   = LSTM(128, return_sequences=False)(inp)
    x   = Dropout(0.3)(x)
    x   = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name='Plain_LSTM')


def build_cnn_lstm(input_shape):
    # baseline 2
    inp = Input(shape=input_shape)
    x   = Conv1D(64,  kernel_size=3, activation='relu', padding='same')(inp)
    x   = MaxPooling1D(pool_size=2)(x)
    x   = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x   = MaxPooling1D(pool_size=2)(x)
    x   = LSTM(128, return_sequences=False)(x)
    x   = Dropout(0.3)(x)
    x   = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name='CNN_LSTM')


def build_cnn_bilstm_attention(input_shape):
    # main model
    inp = Input(shape=input_shape)

    # CNN block
    x = Conv1D(64,  kernel_size=3, activation='relu', padding='same')(inp)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # bidirectional LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # attention
    x = AttentionLayer()(x)

    x   = Dense(64, activation='relu')(x)
    x   = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(inp, out, name='CNN_BiLSTM_Attention')

# STEP 3: TRAIN

def train_model(model, X_train, y_train, epochs=100):
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=7, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    return history


results   = {}
histories = {}

for build_fn in [build_lstm, build_cnn_lstm, build_cnn_bilstm_attention]:
    model = build_fn(INPUT_SHAPE)
    model.summary()

    print(f"\n{'='*50}")
    print(f"Training: {model.name}")
    print(f"{'='*50}")

    history = train_model(model, X_train, y_train)
    histories[model.name] = history

    y_pred = model.predict(X_test).flatten()
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

    results[model.name] = {
        'model':  model,
        'y_pred': y_pred,
        'mae':    mae,
        'rmse':   rmse
    }

    print(f"\n{model.name} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")



# STEP 4: RESULTS TABLE

print("\n" + "="*50)
print("RESULTS (tested on B0018)")
print("="*50)
print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8}")
print("-"*50)
for name, res in results.items():
    print(f"{name:<30} {res['mae']:>8.4f} {res['rmse']:>8.4f}")

# STEP 5: PLOTS

colors = {
    'Plain_LSTM':           'royalblue',
    'CNN_LSTM':             'darkorange',
    'CNN_BiLSTM_Attention': 'green'
}

# prediction vs actual
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('SOH Prediction vs Actual - Tested on B0018', fontsize=14, fontweight='bold')

for idx, (name, res) in enumerate(results.items()):
    ax = axes[idx]
    ax.plot(y_test,        label='Actual SOH',    color='black',      linewidth=2)
    ax.plot(res['y_pred'], label='Predicted SOH', color=colors[name], linewidth=2, linestyle='--')
    ax.set_title(f"{name}\nMAE={res['mae']:.4f} | RMSE={res['rmse']:.4f}")
    ax.set_xlabel('Cycle')
    ax.set_ylabel('SOH')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.1])

plt.tight_layout()
plt.savefig('results_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved results_comparison.png")

# training loss curves
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 4))
fig2.suptitle('Training and Validation Loss', fontsize=14, fontweight='bold')

for idx, (name, history) in enumerate(histories.items()):
    ax = axes2[idx]
    ax.plot(history.history['loss'],     label='Train Loss')
    ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_title(name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved training_curves.png")

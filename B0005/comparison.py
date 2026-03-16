# =============================================================
# FINAL COMPARISON PLOT — All 3 Models
# Run this AFTER running all three model scripts:
#   CNN+LSTM.py, CNN+Crossformer.py, GPR+CNN+Crossformer.py
# Requires: model1_predictions.npy, model2_predictions.npy,
#           model3_predictions.npy
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_npy(filename):
    filepath = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n[ERROR] Could not find: {filepath}"
            f"\nMake sure you have run all 3 model scripts first."
        )
    return np.load(filepath, allow_pickle=True).item()

# ─────────────────────────────────────────────────────────────
# LOAD SAVED PREDICTIONS FROM ALL 3 MODELS
# ─────────────────────────────────────────────────────────────

print("Loading prediction files ...")
d1 = load_npy('model1_predictions.npy')
d2 = load_npy('model2_predictions.npy')
d3 = load_npy('model3_predictions.npy')
print("All prediction files loaded successfully.\n")

EOL_SOH = 0.70

models = [
    {
        'label'   : 'Model 1 — CNN + LSTM',
        'color'   : 'royalblue',
        'y_test'  : d1['y_test'],
        'y_pred'  : d1['y_pred'],
        'test_cyc': d1['test_cyc'],
        'mae'     : d1['mae'],
        'mse'     : d1['mse'],
        'rmse'    : d1['rmse'],
        'r2'      : d1['r2'],
    },
    {
        'label'   : 'Model 2 — CNN + Crossformer',
        'color'   : 'darkorange',
        'y_test'  : d2['y_test'],
        'y_pred'  : d2['y_pred'],
        'test_cyc': d2['test_cyc'],
        'mae'     : d2['mae'],
        'mse'     : d2['mse'],
        'rmse'    : d2['rmse'],
        'r2'      : d2['r2'],
    },
    {
        'label'   : 'Model 3 — GPR + CNN + Crossformer',
        'color'   : 'green',
        'y_test'  : d3['y_test'],
        'y_pred'  : d3['y_pred'],
        'test_cyc': d3['test_cyc'],
        'mae'     : d3['mae'],
        'mse'     : d3['mse'],
        'rmse'    : d3['rmse'],
        'r2'      : d3['r2'],
    },
]

# ─────────────────────────────────────────────────────────────
# FIGURE LAYOUT: 2 rows x 3 cols
# ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 12))
fig.suptitle(
    'Model Comparison — SOH Prediction | B0005 Battery',
    fontweight='bold', fontsize=14, y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30,
                       top=0.91, bottom=0.07)

# ── Row 1: SOH Trajectories ────────────────────────────────────
for i, m in enumerate(models):
    ax = fig.add_subplot(gs[0, i])

    ax.plot(m['test_cyc'], m['y_test'],
            color='black', lw=2, label='Actual SOH', zorder=5)
    ax.plot(m['test_cyc'], m['y_pred'],
            color=m['color'], lw=2, linestyle='--',
            label='Predicted SOH')
    ax.fill_between(m['test_cyc'], m['y_test'], m['y_pred'],
                    alpha=0.15, color=m['color'], label='Error region')
    ax.axhline(EOL_SOH, color='red', linestyle='--',
               lw=1.2, label='EOL (0.70)')

    metrics_text = (f"MAE:  {m['mae']:.4f}\n"
                    f"RMSE: {m['rmse']:.4f}\n"
                    f"R²:   {m['r2']:.4f}")
    ax.text(0.05, 0.95, metrics_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor=m['color'],
                      linewidth=1.5, boxstyle='round,pad=0.4'))

    ax.set_xlabel('Cycle Number', fontsize=10)
    ax.set_ylabel('SOH', fontsize=10)
    ax.set_title(m['label'], fontweight='bold', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])

# ── Row 2: Metric Bar Charts ───────────────────────────────────
metric_configs = [
    ('MAE',  'mae',  'lower is better',  True),
    ('RMSE', 'rmse', 'lower is better',  True),
    ('R²',   'r2',   'higher is better', False),
]
model_names = ['CNN\n+LSTM', 'CNN\n+Crossformer', 'GPR+CNN\n+Crossformer']
bar_colors  = ['royalblue', 'darkorange', 'green']
x           = np.arange(len(model_names))
bar_width   = 0.5

for j, (mname, mkey, direction, lower_better) in enumerate(metric_configs):
    ax = fig.add_subplot(gs[1, j])
    values = [m[mkey] for m in models]

    bars = ax.bar(x, values, width=bar_width,
                  color=bar_colors, edgecolor='black',
                  linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel(mname, fontsize=11)
    ax.set_title(f'{mname} Comparison\n({direction})',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, zorder=0)

    if mkey == 'r2':
        # R² specific: all values likely negative, need special handling
        # Add a y=0 reference line
        ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.6, zorder=4)

        # Y limits: give extra space at top so labels are visible
        ymin = min(values) * 1.25
        ymax = abs(min(values)) * 0.30   # headroom above 0
        ax.set_ylim([ymin, ymax])

        # Place value labels inside each bar (near the top of each bar)
        for bar, val in zip(bars, values):
            bar_top = bar.get_height()  # = val since bars go downward for negatives
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar_top + abs(ymin) * 0.02,
                    f'{val:.4f}',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='black')
    else:
        max_val = max(abs(v) for v in values)
        ax.set_ylim([0, max_val * 1.25])
        # Value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.01,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    # Improvement annotation M1 → M3
    if lower_better:
        improvement = ((values[0] - values[2]) / (values[0] + 1e-9)) * 100
        label = f'↓ {improvement:.1f}% improvement\n(M1 → M3)'
    else:
        improvement = ((values[2] - values[0]) / (abs(values[0]) + 1e-9)) * 100
        label = f'↑ {improvement:.1f}% improvement\n(M1 → M3)'

    ax.annotate(label,
                xy=(0.97, 0.97), xycoords='axes fraction',
                ha='right', va='top', fontsize=8.5,
                color='darkgreen', fontweight='bold',
                bbox=dict(facecolor='#e8f5e9', edgecolor='green',
                          boxstyle='round,pad=0.3'))

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────

output_path = os.path.join(SCRIPT_DIR, 'model_comparison_final.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
plt.show()
print(f"Saved {output_path}")

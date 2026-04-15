import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════
#  ★  FILL IN YOUR CNN_Crossformer RESULTS BELOW
# ══════════════════════════════════════════════════════

MODEL_NAME = "CNN_Crossformer"

DATASETS = ["B0_ageing", "Fast_charging", "Nasa_Datasets"]

# [R2_SCORE, MAE, MSE, RMSE]  ← replace with your actual numbers
METRICS = {
    "B0_ageing":      [0.9766, 0.0119, 0.0003, 0.0183],
    "Fast_charging":  [0.9023, 0.0175, 0.0009, 0.0301],
    "Nasa_Datasets":  [0.8409, 0.0181, 0.0005, 0.0221],
}

# ══════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════
METRIC_LABELS = ["R²", "MAE", "MSE", "RMSE"]
COLORS = ["#2196F3", "#FF9800", "#4CAF50"]

metric_arr = np.array([METRICS[ds] for ds in DATASETS])   # shape (3, 4)

col_min   = metric_arr.min(axis=0)
col_max   = metric_arr.max(axis=0)
col_range = np.where(col_max - col_min == 0, 1e-9, col_max - col_min)

# Radar: outer = better
scaled = np.zeros_like(metric_arr)
scaled[:, 0] = (metric_arr[:, 0] - col_min[0]) / col_range[0]          # R²: high = good
for c in range(1, 4):
    scaled[:, c] = 1 - (metric_arr[:, c] - col_min[c]) / col_range[c]  # errors: low = good

# Heatmap: green = better
norm_heat = np.zeros_like(metric_arr)
norm_heat[:, 0] = (metric_arr[:, 0] - col_min[0]) / col_range[0]
for c in range(1, 4):
    norm_heat[:, c] = 1 - (metric_arr[:, c] - col_min[c]) / col_range[c]

# ══════════════════════════════════════════════════════
#  Figure layout  ← KEY FIX: top=0.88 leaves room for suptitle
# ══════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))

# suptitle pushed well above the subplots
fig.suptitle("Models Performance Comparison", fontsize=17,
             fontweight="bold", y=0.96)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.55, wspace=0.38,
                       top=0.88)          # ← subplots start below 0.88

# ── (a) Radar ─────────────────────────────────────────
N = len(METRIC_LABELS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

ax_r = fig.add_subplot(gs[0, 0], polar=True)
for i, ds in enumerate(DATASETS):
    vals = scaled[i].tolist() + [scaled[i][0]]
    ax_r.plot(angles, vals, color=COLORS[i], linewidth=2, label=ds)
    ax_r.fill(angles, vals, color=COLORS[i], alpha=0.13)

ax_r.set_thetagrids(np.degrees(angles[:-1]), METRIC_LABELS, fontsize=10)
ax_r.set_ylim(0, 1)
ax_r.set_yticks([0.25, 0.50, 0.75, 1.00])
ax_r.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="grey")

# ← pad=25 pushes subplot title well clear of the radar ring
ax_r.set_title("(a)  Radar Chart – Scaled Metrics\n(outer = better)",
               fontsize=10, pad=25)

ax_r.legend(loc="upper right", bbox_to_anchor=(1.45, 1.18), fontsize=9,
            title=MODEL_NAME, title_fontsize=9)

# ── (b) Heat-map ──────────────────────────────────────
ax_h = fig.add_subplot(gs[0, 1])
im = ax_h.imshow(norm_heat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="Relative performance")

ax_h.set_xticks(range(4))
ax_h.set_xticklabels(METRIC_LABELS, fontsize=9)
ax_h.set_yticks(range(3))
ax_h.set_yticklabels(DATASETS, fontsize=9)
ax_h.set_title("(b)  Heat-map of Model Performance\n(green = better)", fontsize=10, pad=10)

for r in range(3):
    for c in range(4):
        raw = metric_arr[r, c]
        txt = f"{raw:.4f}"
        contrast = "black" if 0.2 < norm_heat[r, c] < 0.8 else "white"
        ax_h.text(c, r, txt, ha="center", va="center",
                  fontsize=8.5, color=contrast, fontweight="bold")

# ── (c) R² bar chart ──────────────────────────────────
ax_b = fig.add_subplot(gs[1, 0])
r2_vals = [METRICS[ds][0] for ds in DATASETS]
bars = ax_b.bar(DATASETS, r2_vals, color=COLORS, width=0.5,
                edgecolor="white", linewidth=1.2)

y_min = max(0.0, min(r2_vals) - 0.02)
ax_b.set_ylim(y_min, 1.005)
ax_b.set_ylabel("R² Score", fontsize=11)
ax_b.set_title("(c)  R² Score Comparison", fontsize=10, pad=10)
ax_b.tick_params(axis="x", labelsize=9)
for bar, val in zip(bars, r2_vals):
    ax_b.text(bar.get_x() + bar.get_width() / 2,
              bar.get_height() + 0.0005,
              f"{val:.4f}", ha="center", va="bottom", fontsize=9)

# ── (d) Grouped error metrics bar chart ───────────────
ax_g = fig.add_subplot(gs[1, 1])

error_metrics = ["MAE", "MSE", "RMSE"]
error_idx     = [1, 2, 3]
n_datasets    = len(DATASETS)
x             = np.arange(len(error_metrics))
bar_width     = 0.22
offsets       = np.linspace(-(n_datasets - 1) / 2,
                             (n_datasets - 1) / 2,
                             n_datasets) * bar_width

for i, ds in enumerate(DATASETS):
    vals   = [METRICS[ds][idx] for idx in error_idx]
    bars_g = ax_g.bar(x + offsets[i], vals, width=bar_width,
                      color=COLORS[i], label=ds,
                      edgecolor="white", linewidth=1.0)
    for bar, val in zip(bars_g, vals):
        ax_g.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.0001,
                  f"{val:.4f}", ha="center", va="bottom",
                  fontsize=7, rotation=45)

ax_g.set_xticks(x)
ax_g.set_xticklabels(error_metrics, fontsize=10)
ax_g.set_ylabel("Error Value", fontsize=11)
ax_g.set_title("(d)  Error Metrics by Dataset", fontsize=10, pad=10)
ax_g.legend(fontsize=8, loc="upper right")

# ── Footer ────────────────────────────────────────────
fig.text(0.5, -0.02,
         f"Model: {MODEL_NAME}  |  Datasets: {', '.join(DATASETS)}",
         ha="center", fontsize=9, color="grey")

plt.savefig(f"{MODEL_NAME}_performance.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Saved → {MODEL_NAME}_performance.png")
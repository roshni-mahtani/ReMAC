import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FormatStrFormatter


# Data
labels = np.array([0, 5, 10, 20, 30, 50], dtype=float)
ReMAC_mean   = np.array([0.8583, 0.8167, 0.8167, 0.8083, 0.8083, 0.8167])
ReMaxAC_mean = np.array([0.7583, 0.7167, 0.7750, 0.8083, 0.7667, 0.8250])
ReCLS_mean   = np.array([0.8083, 0.8167, 0.8583, 0.8083, 0.7667, 0.8167])

# Colors
color_remac   = "#1b9e77"  # green
color_remaxac = "#d95f02"  # orange/red
color_recls   = "#7570b3"  # purple

fig, ax = plt.subplots(figsize=(6.4, 4.2))

def plot_series(ax, x, mean, color, label):
    ln, = ax.plot(x, mean, color=color, linewidth=3.0, alpha=0.5, label=label, zorder=2)
    pts = ax.scatter(x[1:], mean[1:], s=120, marker="o", edgecolor="white", linewidth=1.0,
                     color=color, zorder=5)
    tri = ax.scatter([0.0], [mean[0]], marker="^", s=150, edgecolor="white", linewidth=1.0,
                     color=color, zorder=6)
    return ln

# Plot series
ln_recls = plot_series(ax, labels, ReCLS_mean,   color_recls,   "ReCLS")
ln_remax = plot_series(ax, labels, ReMaxAC_mean, color_remaxac, "ReMaxAC")
ln_remac = plot_series(ax, labels, ReMAC_mean,   color_remac,   "ReMAC (ours)")

# Labels and ticks with larger font sizes
ax.set_xlabel("% Missing Values", fontweight="bold", fontsize=16)
ax.set_ylabel("ACC", fontweight="bold", fontsize=16)
ax.set_xticks(labels)
ax.set_xticklabels([str(int(v)) for v in labels], fontweight="bold", fontsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

# Tight Y axis
all_means = np.concatenate([ReMAC_mean, ReMaxAC_mean, ReCLS_mean])
pad = 0.01
ymin = max(0.0, all_means.min() - pad)
ymax = min(1.0, all_means.max() + pad)
ax.set_ylim(ymin, ymax)

# Keep all spines visible and black
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")

# Bold Y tick labels
for t in ax.get_yticklabels():
    t.set_fontweight("bold")
    
 # Add grid lines (both directions, gray dashed)
ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.8, color="gray", alpha=0.6)

# Legend reordering with larger font
handles = [ln_recls, ln_remax, ln_remac]
labels_leg = ["ReCLS", "ReMaxAC", "ReMAC (ours)"]
leg = ax.legend(handles, labels_leg, frameon=True, loc="lower right", fontsize=12)
for text in leg.get_texts():
    text.set_fontweight("bold")
leg.get_frame().set_alpha(0.9)

plt.tight_layout()
out_path = "./agg_comparison.png"
plt.savefig(out_path, dpi=400, bbox_inches="tight")
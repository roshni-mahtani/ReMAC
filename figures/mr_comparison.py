import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# Data
x_labels = ['0', '5', '10', '20', '30', '50']
x = np.arange(len(x_labels))

MR10 = np.array([0.8583, 0.7167, 0.7667, 0.7083, 0.7083, 0.6750])
MR30 = np.array([0.8083, 0.7667, 0.7250, 0.9083, 0.6667, 0.6750])
MR50 = np.array([0.8583, 0.8167, 0.8167, 0.8083, 0.8083, 0.8167])  # baseline
MR70 = np.array([0.8583, 0.7667, 0.8167, 0.8583, 0.8167, 0.7333])

# Colors
colors = ["#66bb6a", "#2e7d32", "#1b9e77", "#12684e"]

# Hatches (MR50 without hatch)
hatches = ['///', '\\\\\\', '', 'xx']

bar_w = 0.18

fig, ax = plt.subplots(figsize=(8.6, 3.4))

# Draw bars
series = [MR10, MR30, MR50, MR70]
labels_leg = ["ReMAC (MR 10%)", "ReMAC (MR 30%)", "ReMAC (MR 50%)", "ReMAC (MR 70%)"]
bars = []

for i, (vals, c, h) in enumerate(zip(series, colors, hatches)):
    bars.append(ax.bar(x + (i-1.5)*bar_w, vals, bar_w, label=labels_leg[i],
                       color=c, edgecolor="black", linewidth=1.0, hatch=h, alpha=0.95))

# Axis labels and ticks
ax.set_xlabel("% Missing Values", fontweight="bold", fontsize=18)
ax.set_ylabel("ACC", fontweight="bold", fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontweight="bold", fontsize=16)

# Y ticks with 2 decimals
ax.tick_params(axis="y", labelsize=16)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
for t in ax.get_yticklabels():
    t.set_fontweight("bold")

# Grid
ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.8, alpha=0.6)
ax.grid(True, which="both", axis="x", linestyle="--", linewidth=0.8, alpha=0.15)

# Y limits with headroom
all_vals = np.concatenate(series)
pad = 0.03
ax.set_ylim(max(0.0, all_vals.min()-pad), min(1.0, all_vals.max()+pad*2))

# Box
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")

# Legend
leg = ax.legend(loc="upper right", fontsize=12, frameon=True)
for text in leg.get_texts():
    if text.get_text() == "ReMAC (MR 50%)":
        text.set_fontweight("bold")
    else:
        text.set_fontweight("normal")
leg.get_frame().set_alpha(0.9)

plt.tight_layout()
out_path = "mr_comparison.png"
plt.savefig(out_path, dpi=400, bbox_inches="tight")
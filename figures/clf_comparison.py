import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# Data
x_labels = ['0', '5', '10', '20', '30', '50']
x = np.arange(len(x_labels))

ReMAC_mean = np.array([0.8583, 0.8167, 0.8167, 0.8083, 0.8083, 0.8167])
ReMLP_mean = np.array([0.8167, 0.8167, 0.8083, 0.8583, 0.7667, 0.8250])

# Colors
color_base = "#1b9e77"   # light green
color_alt  = "#12684e"   # dark green

bar_w = 0.35

fig, ax = plt.subplots(figsize=(7.5, 3.2))

# Bars
ax.bar(x - bar_w/2, ReMAC_mean, bar_w, label="ReMAC (Dense Layer)",
       color=color_base, edgecolor="black", linewidth=1.0, alpha=0.95)
ax.bar(x + bar_w/2, ReMLP_mean, bar_w, label="ReMAC (MLP)",
       color=color_alt, edgecolor="black", linewidth=1.0, alpha=0.95, hatch="oo")

# Axis labels and ticks
ax.set_xlabel("% Missing Values", fontweight="bold", fontsize=16)
ax.set_ylabel("ACC", fontweight="bold", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontweight="bold", fontsize=14)

# Y ticks style with 2 decimals
ax.tick_params(axis="y", labelsize=14)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
for t in ax.get_yticklabels():
    t.set_fontweight("bold")

# Grid
ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.8, alpha=0.6)
ax.grid(True, which="both", axis="x", linestyle="--", linewidth=0.8, alpha=0.15)

# Adjust Y axis with extra headroom
all_vals = np.concatenate([ReMAC_mean, ReMLP_mean])
pad = 0.03
ax.set_ylim(max(0.0, all_vals.min()-pad), min(1.0, all_vals.max()+pad*2))

# Box around axes
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")

# Legend inside, top-right
leg = ax.legend(loc="upper right", fontsize=12, frameon=True)
for text in leg.get_texts():
    if text.get_text() == "ReMAC (Dense Layer)":
        text.set_fontweight("bold")
    else:
        text.set_fontweight("normal")
leg.get_frame().set_alpha(0.9)

plt.tight_layout()
plt.savefig("clf_comparison.png", dpi=400, bbox_inches="tight")

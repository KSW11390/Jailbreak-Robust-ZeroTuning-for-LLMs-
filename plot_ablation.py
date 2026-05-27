#!/usr/bin/env python3
"""Plot ablation experiment results — GCG and PAIR separately."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_JSON = "outputs/ablation_results.json"
OUTPUT_DIR   = Path("outputs")

with open(RESULTS_JSON) as f:
    data = json.load(f)

rates      = [0.5, 1.0, 2.0, 5.0]
rate_strs  = [str(r) for r in rates]
rate_labels = ["0.5", "1.0", "2.0", "5.0"]

COLORS = {"no_ablation": "#4878CF", "ablation": "#D65F5F"}
LABELS = {"no_ablation": "No Ablation", "ablation": "Safety Head Ablation (top-20)"}

ASR_YLIM = {"GCG": (80, 90), "PAIR": (80, 95)}

for attack in ["GCG", "PAIR"]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Safety Head Ablation Effect — {attack}", fontsize=14, fontweight="bold")

    # ── Left: ASR lines ────────────────────────────────────────────
    for cond in ["no_ablation", "ablation"]:
        asrs = [data[cond][attack][rs]["asr"] * 100 for rs in rate_strs]
        ax1.plot(rate_labels, asrs, marker="o", linewidth=2,
                 color=COLORS[cond], label=LABELS[cond])
        for x, y in zip(rate_labels, asrs):
            ax1.annotate(f"{y:.1f}%", (x, y),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)

    ax1.set_xlabel("γ (ZeroTuning rate)", fontsize=11)
    ax1.set_ylabel("ASR (%)", fontsize=11)
    ax1.set_title("ASR by γ", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.set_ylim(*ASR_YLIM[attack])
    ax1.grid(axis="y", alpha=0.3)

    # ── Right: Δ bar chart ─────────────────────────────────────────
    deltas = [
        (data["ablation"][attack][rs]["asr"] - data["no_ablation"][attack][rs]["asr"]) * 100
        for rs in rate_strs
    ]
    bar_colors = [COLORS["ablation"] if d >= 0 else COLORS["no_ablation"] for d in deltas]

    bars = ax2.bar(rate_labels, deltas, color=bar_colors, width=0.5, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8)
    for bar, d in zip(bars, deltas):
        sign = "+" if d >= 0 else ""
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 d + (0.3 if d >= 0 else -0.5),
                 f"{sign}{d:.1f}pp", ha="center", va="bottom" if d >= 0 else "top",
                 fontsize=10, fontweight="bold")

    ax2.set_xlabel("γ (ZeroTuning rate)", fontsize=11)
    ax2.set_ylabel("ΔASR (ablation − no_ablation, pp)", fontsize=11)
    ax2.set_title("Ablation Effect (Δ)", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    lim = max(abs(d) for d in deltas) + 3
    ax2.set_ylim(-lim, lim)

    plt.tight_layout()
    out = OUTPUT_DIR / f"ablation_{attack.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# -*- coding: utf-8 -*-
"""Generate the three CPU figures for the paper (theory / performance / training).

Style: pure white background, no grid, legends inside, no annotation text.
All numbers are taken verbatim from the verified JSON/CSV reports:
  step4b2_report.json, step4_report.json, step5_k_scan_report.json,
  runs/step5_final/full/history.csv
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(ROOT, "论文部分", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.8, "savefig.dpi": 300,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# ---------------------------------------------------------------- data
# step4b2: implicit-gradient verification (R4, slices [3,88], untrained SNRegNet)
unroll_k = np.array([400, 800, 1600])
unroll_cos = np.array([0.1648494979829005, 0.8090888321161418, 0.993461763398304])
phase_pairs = ["0 vs 1.57", "0 vs 3.14", "1.57 vs 3.14"]
phase_cos = np.array([0.9999999942651102, 1.0000000005378709, 1.0000000044687702])
rho_methods = ["Bregman\np=4+gauge", "Euclidean\n+gauge", "Euclidean\nno gauge"]
rho_vals = np.array([0.9999688267707825, 1.4695578813552856, 1.4759520292282104])

# step5_k_scan: depth scan (test n=804, ckpt ep85, CascadeNet base=64 K=4)
psnr = {
    "r4_s42":   np.array([25.2893, 27.2704, 27.6500, 27.7812]),
    "r4_s123":  np.array([25.1329, 27.1225, 27.4929, 27.6164]),
    "r4_s2025": np.array([25.0955, 27.0970, 27.4528, 27.5713]),
}
zf = {"r4_s42": 25.4223, "r4_s123": 25.2548, "r4_s2025": 25.0839}

LEG_FRAME = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)

# ---------------------------------------------------------------- fig 1: theory
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.10, wspace=0.30, hspace=0.42)

ax = axes[0, 0]
ax.plot(unroll_k, unroll_cos, "o-", color="#1a7f37", lw=1.6, ms=5)
ax.axhline(0.99, color="gray", ls="--", lw=1.0)
ax.set_xlabel("Unrolled depth K")
ax.set_ylabel("cosine similarity")
ax.set_title("(a) Implicit vs. unrolled gradient")
ax.set_xlim(300, 1700)
ax.set_ylim(0.0, 1.05)

ax = axes[0, 1]
cols = ["#1a7f37", "#c62828", "#c62828"]
bars = ax.bar(rho_methods, rho_vals, color=cols, width=0.55, edgecolor="black", lw=0.6)
ax.axhline(1.0, color="black", ls=":", lw=1.2)
ax.set_ylabel("Spectral radius $\rho(J_S)$")
ax.set_title("(b) Jacobian spectral radius")
ax.set_ylim(0.9, 1.62)

ax = axes[1, 0]
xs = np.arange(5)
labels = ["ZF", "K=1", "K=2", "K=3", "K=4"]
vals = [zf["r4_s42"], psnr["r4_s42"][0], psnr["r4_s42"][1], psnr["r4_s42"][2], psnr["r4_s42"][3]]
ax.axhline(zf["r4_s42"], color="gray", ls="--", lw=1.0)
ax.plot(xs, vals, "o-", color="#1a7f37", lw=1.6, ms=5)
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_ylabel("PSNR (dB)")
ax.set_ylim(24.5, 28.6)
ax.set_title("(c) Depth scan, mask r4_s42")

ax = axes[1, 1]
ax.bar(phase_pairs, phase_cos, color="#1565c0", width=0.55, edgecolor="black", lw=0.6)
ax.set_ylim(0.999999988, 1.000000012)
ax.set_ylabel("gradient cosine")
ax.set_title("(d) Phase-equivariant gradient")
ax.tick_params(axis="x", labelsize=6.5)
ax.axhline(1.0, color="black", lw=0.8)

fig.savefig(os.path.join(FIG_DIR, "fig_theory.pdf"))
plt.close(fig)
print("saved fig_theory.pdf")

# ---------------------------------------------------------------- fig 2: performance
fig, ax = plt.subplots(figsize=(4.6, 3.3))
fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.14)
mcolors = {"r4_s42": "#1a7f37", "r4_s123": "#1565c0", "r4_s2025": "#e65100"}
mlabels = {"r4_s42": "mask r4_s42", "r4_s123": "mask r4_s123", "r4_s2025": "mask r4_s2025"}
x = np.arange(4)
for mk in ["r4_s42", "r4_s123", "r4_s2025"]:
    ax.plot(x, psnr[mk], "o-", color=mcolors[mk], lw=1.6, ms=5, label=mlabels[mk])
    ax.axhline(zf[mk], color=mcolors[mk], ls="--", lw=0.9, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(["K=1", "K=2", "K=3", "K=4"])
ax.set_ylabel("PSNR (dB)")
ax.set_ylim(24.2, 28.6)
ax.set_title("Cascade depth vs. reconstruction quality (test, n=804)")
ax.legend(handles=[Line2D([0], [0], color="k", lw=1.2, label="solid: cascade K"),
                   Line2D([0], [0], color="k", lw=1.2, ls="--", label="dashed: ZF baseline")],
          loc="lower right", **LEG_FRAME)
fig.savefig(os.path.join(FIG_DIR, "fig_performance.pdf"))
plt.close(fig)
print("saved fig_performance.pdf")

# ---------------------------------------------------------------- fig 3: training
hist = []
with open(os.path.join(HERE, "runs", "step5_final", "full", "history.csv"), newline="") as fp:
    for row in csv.DictReader(fp):
        hist.append({k: float(v) for k, v in row.items()})
ep = np.array([h["epoch"] for h in hist])
loss = np.array([h["loss"] for h in hist])
valp = np.array([h["val_psnr"] for h in hist])

fig, ax1 = plt.subplots(figsize=(4.6, 3.3))
fig.subplots_adjust(left=0.12, right=0.88, top=0.93, bottom=0.14)
ax1.plot(ep, loss, color="#c62828", lw=1.4, label="train loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("train loss", color="#c62828")
ax1.tick_params(axis="y", labelcolor="#c62828")
ax1.set_ylim(0.05, 0.12)
ax2 = ax1.twinx()
ax2.plot(ep, valp, color="#1a7f37", lw=1.6, label="val PSNR")
ax2.set_ylabel("val PSNR (dB)", color="#1a7f37")
ax2.tick_params(axis="y", labelcolor="#1a7f37")
ax2.set_ylim(21.5, 29.5)
for e in (5, 11):
    ax1.axvline(e, color="gray", ls=":", lw=1.0)
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", **LEG_FRAME)
ax1.set_title("Training curves (CascadeNet, K=4, 19.3M params)")
fig.savefig(os.path.join(FIG_DIR, "fig_training.pdf"))
plt.close(fig)
print("saved fig_training.pdf")
print("all CPU figures done")

# -*- coding: utf-8 -*-
"""论文新图：QB-DEQ v2 端到端诊断
  fig_trajectory.png  -- 不动点迭代轨迹（PSNR vs 迭代数；plain/Anderson/阻尼）
  fig_diagnostic.png  -- Lipschitz 审计（||J_D|| 与 rho，带/不带 gauge）
所有数字来自已核验的实验输出（_probe_fp_traj.py / _probe_exp_source.py，
checkpoint: base=32, RED+SN+identity, 256 切片, 4 epochs）。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(ROOT, "论文部分", "figures")
OUT_DIR = os.path.join(ROOT, "论文部分")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.8, "savefig.dpi": 300,
})

# ---------------- fig_trajectory ----------------
k = np.array([1, 4, 8, 16, 32, 64, 100], dtype=float)
# slice 2444 (easy, ZF=25.48, rho(ZF)=1.24)
p_easy = dict(
    plain=[22.00, 24.65, 24.48, 24.22, 23.54, 22.91, 22.78],
    anderson5=[22.00, 22.74, 22.67, 22.63, 22.38, 21.85, 21.48],
    damp07=[22.00, 23.04, 22.70, 22.56, 22.23, 21.72, 21.33],
)
# slice 1441 (hard, ZF=25.56, rho(ZF)=2.21)
p_hard = dict(
    plain=[17.00, 21.42, 20.01, 18.32, 15.57, 14.74, 14.64],
    anderson5=[17.00, 20.49, 20.20, 19.81, 19.44, 17.84, 15.33],
    damp07=[17.00, 20.48, 20.17, 19.81, 19.24, 16.75, 14.96],
)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.16, wspace=0.24)
styles = {"plain": ("o-", "#1a7f37", "plain"), "anderson5": ("s-", "#1565c0", "Anderson m=5"),
          "damp07": ("^-", "#e65100", "Anderson, damp. 0.7")}
for ax, data, title, rho0 in [
        (axes[0], p_easy, "easy slice (ZF 25.48 dB)", 1.24),
        (axes[1], p_hard, "hard slice (ZF 25.56 dB)", 2.21)]:
    for key, (mk, col, lab) in styles.items():
        ax.plot(k, data[key], mk, color=col, lw=1.5, ms=4, label=lab)
    ax.axhline(float(np.max(data["plain"])), color="gray", ls=":", lw=1.0)
    ax.set_title(title + r",  $\rho(J_S)$(ZF)$=%.2f$" % rho0, fontsize=8.5)
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(k)
    ax.set_ylim(13, 26)
    ax.grid(alpha=0.3, lw=0.5)
axes[0].legend(loc="lower left", frameon=False, fontsize=6.5)
fig.suptitle("Fixed-point iteration trajectory of the trained RED operator (small-scale diagnostic)",
             fontsize=9, y=0.99)
fig.savefig(os.path.join(FIG_DIR, "fig_trajectory.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_trajectory.png"))
plt.close(fig)
print("saved fig_trajectory")

# ---------------- fig_diagnostic ----------------
metrics = [r"$\|J_D\|$ @ ZF", r"$\|J_D\|$ @ GT", r"$\rho(S)$, no gauge", r"$\rho(S)$, gauge"]
easy = np.array([1.5921, 1.4124, 1.3535, 1.3624])
hard = np.array([4.4041, 3.0963, 2.8560, 2.6491])
x = np.arange(len(metrics))
w = 0.36
fig, ax = plt.subplots(figsize=(4.8, 3.0))
fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.14)
b1 = ax.bar(x - w / 2, easy, w, color="#1a7f37", edgecolor="black", lw=0.5, label="slice 2444")
b2 = ax.bar(x + w / 2, hard, w, color="#c62828", edgecolor="black", lw=0.5, label="slice 1441")
ax.axhline(1.0, color="black", ls=":", lw=1.2)
ax.annotate("nonexpansive threshold", xy=(0.02, 1.05), xycoords="axes fraction", fontsize=7)
for bars in (b1, b2):
    for b in bars:
        ax.annotate("%.2f" % b.get_height(), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=7)
ax.set_ylabel("estimated Jacobian norm / rho")
ax.set_ylim(0, 4.9)
ax.grid(alpha=0.3, axis="y", lw=0.5)
ax.legend(loc="upper left", frameon=False, fontsize=7)
ax.set_title("Lipschitz audit of the trained operator (small-scale diagnostic)", fontsize=8.5)
fig.savefig(os.path.join(FIG_DIR, "fig_diagnostic.pdf"))
fig.savefig(os.path.join(OUT_DIR, "fig_diagnostic.png"))
plt.close(fig)
print("saved fig_diagnostic")
print("done")

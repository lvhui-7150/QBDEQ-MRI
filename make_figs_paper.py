# -*- coding: utf-8 -*-
"""make_figs_paper.py —— 论文全部图的统一生成脚本（白底、无网格、图例内置、无注释文字）

统一风格规范（全文每张图都遵守）：
  * 背景纯白：figure / axes / savefig 均为白色；
  * 无网格线：任何子图不调用 ax.grid(...)；
  * 图例置于图内：白底细边框，frameon=True；
  * 不绘制 "+x.xx dB gain" 之类的注释文字（数值只出现在坐标轴与图例中）。

图清单与数据来源（全部来自已核验的实验报告，无硬编码结果以外的数字）：
  CPU 图（无需 GPU）：
    fig_theory        2x2 理论验证：隐式/展开梯度一致性、Jacobian 谱半径、
                       深度-性能、相位等变梯度
                       (step4b2_report.json / step4_report.json / step5_k_scan_report.json)
    fig_performance   深度-性能曲线，3 种 4x 掩码 (step5_k_scan_report.json)
    fig_training      CascadeNet K=4 训练曲线 (runs/step5_final/full/history.csv)
    fig_trajectory    不动点迭代轨迹，easy/hard 切片 (probe 输出，小尺度诊断模型)
    fig_diagnostic    Lipschitz 审计：||J_D|| 与 rho (probe 输出)
  GPU 图（需 CUDA 与对应 checkpoint）：
    fig_reconstructions   CascadeNet 重构对比 (runs/step5_final/full/checkpoint_best.pt)
    fig_baseline_recon    六基线重构对比 (runs/baseline/*_best.pt)

用法：
  python make_figs_paper.py            # CPU + GPU 全部
  python make_figs_paper.py --cpu      # 只画 CPU 图
  python make_figs_paper.py --gpu      # 只画 GPU 图
"""
import os
import sys
import argparse

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(ROOT, "论文部分", "figures")   # PDF 输出
OUT_DIR = os.path.join(ROOT, "论文部分")             # PNG 输出（tex 引用）
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- 统一风格
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})

# 图例：白底、浅灰细边框、完全不透明（置于图内时依然清晰）
LEG = dict(frameon=True, facecolor="white", edgecolor="#9e9e9e",
           framealpha=1.0, borderpad=0.5, handlelength=1.6)

# 配色（保持一致性与色盲友好：绿/蓝/橙/红）
C_GREEN = "#1a7f37"
C_BLUE = "#1565c0"
C_ORANGE = "#e65100"
C_RED = "#c62828"
C_GRAY = "#757575"


def save(fig, name):
    """同时输出 figures/<name>.pdf 与 <论文部分>/<name>.png（300 dpi）。"""
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(OUT_DIR, name + ".png"))
    plt.close(fig)
    print("[FIG] saved %s.pdf / %s.png" % (name, name), flush=True)


def log(m):
    print("[FIG] %s" % m, flush=True)


# ================================================================ CPU 图
def make_theory():
    """fig_theory：2x2 理论验证（数值来自 step4b2 / step4 / step5_k_scan 报告）。"""
    unroll_k = np.array([400, 800, 1600])
    unroll_cos = np.array([0.1648494979829005, 0.8090888321161418, 0.993461763398304])
    phase_pairs = ["0 vs 1.57", "0 vs 3.14", "1.57 vs 3.14"]
    phase_cos = np.array([0.9999999942651102, 1.0000000005378709, 1.0000000044687702])
    rho_methods = ["Bregman\np=4+gauge", "Euclidean\n+gauge", "Euclidean\nno gauge"]
    rho_vals = np.array([0.9999688267707825, 1.4695578813552856, 1.4759520292282104])
    zf42 = 25.4223
    k42 = np.array([25.2893, 27.2704, 27.6500, 27.7812])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.10,
                        wspace=0.30, hspace=0.42)

    # (a) 隐式梯度 vs 展开梯度的一致性
    ax = axes[0, 0]
    ax.plot(unroll_k, unroll_cos, "o-", color=C_GREEN, lw=1.6, ms=5)
    ax.axhline(0.99, color=C_GRAY, ls="--", lw=1.0)
    ax.set_xlabel("Unrolled depth $K$")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("(a) Implicit vs. unrolled gradient")
    ax.set_xlim(300, 1700)
    ax.set_ylim(0.0, 1.05)

    # (b) Jacobian 谱半径（Bregman 商流形 < 1，Euclidean >= 1）
    ax = axes[0, 1]
    cols = [C_GREEN, C_RED, C_RED]
    ax.bar(rho_methods, rho_vals, color=cols, width=0.55,
           edgecolor="black", lw=0.6)
    ax.axhline(1.0, color="black", ls=":", lw=1.2)
    ax.set_ylabel(r"Spectral radius $\rho(J_S)$")
    ax.set_title("(b) Jacobian spectral radius")
    ax.set_ylim(0.9, 1.62)
    ax.legend(handles=[Patch(facecolor=C_GREEN, edgecolor="black", label="Bregman quotient"),
                       Patch(facecolor=C_RED, edgecolor="black", label="Euclidean")],
              loc="upper right", **LEG)

    # (c) 深度-性能（掩码 r4_s42）
    ax = axes[1, 0]
    xs = np.arange(5)
    labels = ["ZF", "$K{=}1$", "$K{=}2$", "$K{=}3$", "$K{=}4$"]
    vals = np.array([zf42, k42[0], k42[1], k42[2], k42[3]])
    ax.plot(xs, vals, "o-", color=C_GREEN, lw=1.6, ms=5, label="cascade")
    ax.axhline(zf42, color=C_GRAY, ls="--", lw=1.0, label="zero-filling")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(24.5, 28.6)
    ax.set_title("(c) Depth scan, mask r4_s42")
    ax.legend(loc="lower right", **LEG)

    # (d) 相位等变梯度（cos = 1 至机器精度）
    ax = axes[1, 1]
    ax.bar(phase_pairs, phase_cos, color=C_BLUE, width=0.55,
           edgecolor="black", lw=0.6)
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_ylim(0.999999988, 1.000000012)
    ax.set_yticks([0.99999999, 1.00000000, 1.00000001])
    ax.set_yticklabels(["0.99999999", "1.00000000", "1.00000001"])
    ax.set_ylabel("Gradient cosine")
    ax.set_title("(d) Phase-equivariant gradient")
    ax.tick_params(axis="x", labelsize=6.5)

    save(fig, "fig_theory")


def make_performance():
    """fig_performance：深度-性能曲线，3 种掩码（step5_k_scan_report.json）。"""
    psnr = {
        "r4_s42":   np.array([25.2893, 27.2704, 27.6500, 27.7812]),
        "r4_s123":  np.array([25.1329, 27.1225, 27.4929, 27.6164]),
        "r4_s2025": np.array([25.0955, 27.0970, 27.4528, 27.5713]),
    }
    zf = {"r4_s42": 25.4223, "r4_s123": 25.2548, "r4_s2025": 25.0839}
    mcolors = {"r4_s42": C_GREEN, "r4_s123": C_BLUE, "r4_s2025": C_ORANGE}
    mlabels = {"r4_s42": "mask r4_s42", "r4_s123": "mask r4_s123",
               "r4_s2025": "mask r4_s2025"}

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.14)
    x = np.arange(4)
    for mk in ["r4_s42", "r4_s123", "r4_s2025"]:
        ax.plot(x, psnr[mk], "o-", color=mcolors[mk], lw=1.6, ms=5,
                label=mlabels[mk])
        ax.axhline(zf[mk], color=mcolors[mk], ls="--", lw=0.9, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["$K{=}1$", "$K{=}2$", "$K{=}3$", "$K{=}4$"])
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(24.2, 28.6)
    ax.set_title("Cascade depth vs. reconstruction quality (test, $n{=}804$)")
    handles = ([Line2D([0], [0], color=mcolors[mk], lw=1.6, label=mlabels[mk])
                for mk in ["r4_s42", "r4_s123", "r4_s2025"]] +
               [Line2D([0], [0], color="k", lw=1.2, ls="--", label="zero-filling"),
                Line2D([0], [0], color="k", lw=1.2, ls="-", label="cascade $K$")])
    ax.legend(handles=handles, loc="lower right", **LEG)
    save(fig, "fig_performance")


def make_training():
    """fig_training：CascadeNet K=4 训练曲线（runs/step5_final/full/history.csv）。"""
    hist = []
    with open(os.path.join(HERE, "runs", "step5_final", "full", "history.csv"),
              newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            hist.append({k: float(v) for k, v in row.items()})
    ep = np.array([h["epoch"] for h in hist])
    loss = np.array([h["loss"] for h in hist])
    valp = np.array([h["val_psnr"] for h in hist])
    best_ep = int(ep[np.argmax(valp)])

    fig, ax1 = plt.subplots(figsize=(4.6, 3.3))
    fig.subplots_adjust(left=0.12, right=0.88, top=0.93, bottom=0.14)
    ax1.plot(ep, loss, color=C_RED, lw=1.4, label="training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color=C_RED)
    ax1.tick_params(axis="y", labelcolor=C_RED)
    ax1.set_ylim(0.05, 0.12)
    ax2 = ax1.twinx()
    ax2.plot(ep, valp, color=C_GREEN, lw=1.6, label="validation PSNR")
    ax2.set_ylabel("Validation PSNR (dB)", color=C_GREEN)
    ax2.tick_params(axis="y", labelcolor=C_GREEN)
    ax2.set_ylim(21.5, 29.5)
    # 阶段 B 起始（epoch 11：广播 stage-1 权重到全级联，见正文）
    ax1.axvline(11, color=C_GRAY, ls=":", lw=1.0)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", **LEG)
    ax1.set_title("Training curves (CascadeNet, $K{=}4$, 19.3M params, "
                  "best %.2f dB @ ep %d)" % (valp.max(), best_ep))
    save(fig, "fig_training")


def make_trajectory():
    """fig_trajectory：不动点迭代轨迹（小尺度诊断模型，probe 输出）。"""
    k = np.array([1, 4, 8, 16, 32, 64, 100], dtype=float)
    p_easy = dict(
        plain=[22.00, 24.65, 24.48, 24.22, 23.54, 22.91, 22.78],
        anderson5=[22.00, 22.74, 22.67, 22.63, 22.38, 21.85, 21.48],
        damp07=[22.00, 23.04, 22.70, 22.56, 22.23, 21.72, 21.33],
    )
    p_hard = dict(
        plain=[17.00, 21.42, 20.01, 18.32, 15.57, 14.74, 14.64],
        anderson5=[17.00, 20.49, 20.20, 19.81, 19.44, 17.84, 15.33],
        damp07=[17.00, 20.48, 20.17, 19.81, 19.24, 16.75, 14.96],
    )
    styles = {"plain": ("o-", C_GREEN, "plain"),
              "anderson5": ("s-", C_BLUE, "Anderson $m{=}5$"),
              "damp07": ("^-", C_ORANGE, "Anderson, damp. 0.7")}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.16, wspace=0.24)
    for ax, data, title in [
            (axes[0], p_easy, "easy slice (ZF 25.48 dB)"),
            (axes[1], p_hard, "hard slice (ZF 25.56 dB)")]:
        for key, (mk, col, lab) in styles.items():
            ax.plot(k, data[key], mk, color=col, lw=1.5, ms=4, label=lab)
        ax.axhline(float(np.max(data["plain"])), color=C_GRAY, ls=":", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel("PSNR (dB)")
        ax.set_xticks(k)
        ax.set_ylim(13, 26)
    axes[0].legend(loc="lower left", **LEG)
    fig.suptitle("Fixed-point iteration trajectory of the trained RED operator "
                 "(small-scale diagnostic)", fontsize=9, y=0.99)
    save(fig, "fig_trajectory")


def make_diagnostic():
    """fig_diagnostic：Lipschitz 审计（probe 输出，小尺度诊断模型）。"""
    metrics = [r"$\|J_D\|$ @ ZF", r"$\|J_D\|$ @ GT",
               r"$\rho(S)$, no gauge", r"$\rho(S)$, gauge"]
    easy = np.array([1.5921, 1.4124, 1.3535, 1.3624])
    hard = np.array([4.4041, 3.0963, 2.8560, 2.6491])
    x = np.arange(len(metrics))
    w = 0.36

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.90, bottom=0.15)
    ax.bar(x - w / 2, easy, w, color=C_GREEN, edgecolor="black", lw=0.5,
           label="slice 2444")
    ax.bar(x + w / 2, hard, w, color=C_RED, edgecolor="black", lw=0.5,
           label="slice 1441")
    ax.axhline(1.0, color="black", ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7)
    ax.set_ylabel("Estimated Jacobian norm / spectral radius")
    ax.set_ylim(0, 4.9)
    ax.legend(loc="upper left", **LEG)
    ax.set_title("Lipschitz audit of the trained operator (small-scale diagnostic)")
    save(fig, "fig_diagnostic")


# ================================================================ GPU 图
def _gpu_ready():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def make_reconstructions():
    """fig_reconstructions：CascadeNet 重构对比（3 验证切片 x ZF/K1/K4/GT）。"""
    import torch
    import step5_320_ceiling as C
    from step5_train_final import CascadeNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck_path = os.path.join(HERE, "runs", "step5_final", "full", "checkpoint_best.pt")
    if not os.path.exists(ck_path):
        log("skip fig_reconstructions: checkpoint missing %s" % ck_path)
        return
    val_slices = [2390, 5661, 4184]
    mask_key = "r4_s42"

    meta = torch.load(C.META_PATH, map_location="cpu", weights_only=False)
    store = C.ChunkStore(meta)
    mask_store = C.MaskStore(meta["masks"])
    _tr, val_idx, _te = C.load_split(meta)
    val_set = set(int(i) for i in val_idx)
    for i in val_slices:
        assert int(i) in val_set, "slice %d not in val split" % i

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = CascadeNet(in_ch=2, base=64, K=4, eta_init=0.5).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    log("fig_reconstructions: ckpt epoch=%s val_psnr=%.2f" % (
        ck.get("epoch"), ck.get("val_psnr", -1.0)))

    mask = mask_store.get(mask_key, device=device)
    ssim = C.SSIMComputer()
    rows = []
    with torch.no_grad():
        for i in val_slices:
            g = store.get_batch([i], device=device)
            yb = C.fft2_t(g) * mask
            z0 = C.to_2ch(C.ifft2_t(yb))
            outs = model(z0, yb, mask, k_max=4)
            zf = C.ifft2_t(yb)
            gt_m = np.abs(g[0].detach().cpu().numpy())
            mag = lambda t: np.abs(t[0].detach().cpu().numpy())
            zf_m, k1_m, k4_m = mag(zf), mag(outs[0]), mag(outs[3])
            row = {"idx": int(i)}
            for name, m in [("ZF", zf_m), ("K=1", k1_m), ("K=4", k4_m)]:
                row[name] = (m, C.compute_psnr(gt_m, m), ssim.compute(gt_m, m))
            row["GT"] = gt_m
            rows.append(row)
            log("fig_reconstructions slice %d: ZF %.2f / K1 %.2f / K4 %.2f dB"
                % (i, row["ZF"][1], row["K=1"][1], row["K=4"][1]))

    cols = ["ZF", "K=1", "K=4", "GT"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01,
                        wspace=0.02, hspace=0.22)
    for r, row in enumerate(rows):
        peak = float(torch.abs(store.get(row["idx"])).max())
        for c, col in enumerate(cols):
            ax = axes[r, c]
            if col == "GT":
                im, title = row["GT"], "GT"
            else:
                im, p, s = row[col]
                title = "%s\nPSNR %.2f / SSIM %.3f" % (col, p, s)
            ax.imshow(im / peak, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=7.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("slice %d" % row["idx"], fontsize=7.5)
    fig.suptitle("Magnitude reconstructions, $4\\times$ mask r4_s42 "
                 "(validation slices)", fontsize=9)
    save(fig, "fig_reconstructions")


def make_baseline_recon():
    """fig_baseline_recon：六基线重构对比（3 验证切片 x ZF/6 模型/GT）。"""
    import torch
    import baseline_utils as BU
    import step5_320_ceiling as C

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck_dir = os.path.join(HERE, "runs", "baseline")
    names = ["unet", "modl", "varnet", "dccnn", "cascadenet", "qbdeq"]
    if not all(os.path.exists(os.path.join(ck_dir, "%s_best.pt" % n))
               for n in names):
        log("skip fig_baseline_recon: baseline checkpoints missing")
        return

    slices = [2390, 5661, 4184]
    store, mask, _tr, _va, _te, _ = BU.load_data(device=device, train_subset=0,
                                                 val_max=48, test_subset=0)
    models = []
    from model_unet import UnetBaseline
    from model_modl import MoDL
    from model_varnet import VarNet
    from model_dccnn import DCCNN
    from model_cascadenet import CascadeNetBaseline
    from model_qbdeq import QBDEQ
    cfgs = [
        ("unet", UnetBaseline(base=64)),
        ("modl", MoDL(K=4, base=64)),
        ("varnet", VarNet(K=6, base=32)),
        ("dccnn", DCCNN(K=5, feats=64)),
        ("cascadenet", CascadeNetBaseline(K=4, base=64)),
        ("qbdeq", QBDEQ(K=8, base=64)),
    ]
    for name, m in cfgs:
        ck = torch.load(os.path.join(ck_dir, "%s_best.pt" % name),
                        map_location="cpu", weights_only=False)
        m.load_state_dict(ck["state_dict"])
        m.to(device).eval()
        models.append((name, m))
        log("fig_baseline_recon %s: best ep=%s val=%.2f" % (
            name, ck["epoch"], ck["val_psnr"]))

    display = {"unet": "U-Net", "modl": "MoDL", "varnet": "VarNet",
               "dccnn": "DCCNN", "cascadenet": "CascadeNet", "qbdeq": "QB-DEQ"}
    ssim = C.SSIMComputer()
    cols = ["ZF"] + names + ["GT"]
    n_rows, n_cols = len(slices), len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.2 * n_rows))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01,
                        wspace=0.03, hspace=0.20)
    for r, si in enumerate(slices):
        x = store.get_batch([si], device=device).to(torch.complex64)
        y, z0 = BU.make_inputs(x, mask, device)
        with torch.no_grad():
            zf = C.ifft2_t(y)
            outs = {"ZF": zf}
            for name, m in models:
                outs[name] = C.to_c(m(z0, y, mask))
        peak = float(torch.abs(x).max().item())
        gt_m = np.abs(x[0].detach().cpu().numpy())
        for c, col in enumerate(cols):
            ax = axes[r, c]
            if col == "GT":
                im, title = gt_m, "GT"
            else:
                im = np.abs(outs[col][0].detach().cpu().numpy())
                label = display[col] if col in display else col
                title = "%s\nPSNR %.2f / SSIM %.3f" % (
                    label, C.compute_psnr(gt_m, im), ssim.compute(gt_m, im))
            ax.imshow(im / peak, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=6.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("slice %d" % si, fontsize=6.5)
    fig.suptitle("Magnitude reconstructions, $4\\times$ mask r4_s42 "
                 "(5-epoch diagnostic models)", fontsize=9, y=0.97)
    save(fig, "fig_baseline_recon")


# ================================================================ 入口
def main():
    p = argparse.ArgumentParser(description="论文全部图（统一风格）")
    p.add_argument("--cpu", action="store_true", help="只画 CPU 图")
    p.add_argument("--gpu", action="store_true", help="只画 GPU 图")
    args = p.parse_args()
    do_cpu = not args.gpu
    do_gpu = not args.cpu

    if do_cpu:
        log("CPU figures ...")
        make_theory()
        make_performance()
        make_training()
        make_trajectory()
        make_diagnostic()
    if do_gpu:
        if _gpu_ready():
            log("GPU figures ...")
            make_reconstructions()
            make_baseline_recon()
        else:
            log("CUDA unavailable; skip GPU figures (run on GPU machine later)")
    log("all requested figures done")


if __name__ == "__main__":
    sys.exit(main())

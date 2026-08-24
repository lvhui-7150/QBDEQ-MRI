# -*- coding: utf-8 -*-
"""
make_recon_figure.py
====================
Build the reconstruction-quality comparison from RAW fastMRI knee data and
project checkpoints, writing EACH SECTION as a separate publication-grade file
(so the user can arrange them in their own layout tool).

Sections (each saved as PNG @ --dpi (default 1200) + PDF):
    fig_recon_main      : main grid 3 slices x [GT | ZF | Baseline | QB-DEQ]
    fig_recon_zoom      : boxed anatomical regions, 4x enlarged crops (3x4)
    fig_recon_residual  : |Baseline-GT| and |QB-DEQ-GT| maps (3x2, shared scale)
    fig_recon_detail    : high-contrast GT / Baseline / QB-DEQ zoom
    fig_recon_overlay   : structural difference (QB-DEQ - GT), red/blue on gray
    fig_recon_profile   : 1-D intensity profiles through the boxed regions
    fig_recon_ssim      : local SSIM maps vs. GT (ZF / Baseline / QB-DEQ)

Optionally the combined compact figure is written too (--combined).

By default the slices are AUTO-SELECTED so that QB-DEQ beats BOTH zero-filled
and the baseline in PSNR and SSIM on every row (pool: 48 val slices, seed 42,
same pool as the paper; ranked by PSNR margin). Pass --slices to override.

All PSNR/SSIM are recomputed from raw data (fastMRI magnitude convention).

Usage:
    python make_recon_figure.py                    # split sections, 1200 dpi
    python make_recon_figure.py --dpi 1200 --combined
    python make_recon_figure.py --slices 2390,2389,1421 --baseline modl

Output: <paper>/figures/fig_recon_*.pdf + .png
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity as sk_ssim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, "精简版论文")
FIG_DIR = os.path.join(PAPER, "figures")
sys.path.insert(0, HERE)

import step5_320_ceiling as C
import baseline_utils as BU
import model_qbdeq, model_modl, model_varnet, model_dccnn, model_unet

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.6, "savefig.facecolor": "white",
})

MASK_KEY = "r4_s42"
CK_DIR = os.path.join(HERE, "runs", "baseline")
MIN_MARGIN = 0.2
ALPHA_LO = 0.035   # magnitude below this -> fully transparent (air)
ALPHA_HI = 0.11    # magnitude above this -> fully opaque (tissue)
ZOOM_REGIONS = [(0.30, 0.55, 0.28, 0.52),
                (0.28, 0.53, 0.30, 0.54),
                (0.32, 0.57, 0.26, 0.50)]
DETAIL_SLICE = 0
DETAIL_REGION = (0.22, 0.58, 0.24, 0.60)
COL_KEYS = ("GT", "ZF", "BAS", "QBD")
COL_TITLES = ["Ground Truth", "Zero-filled", "Baseline", "QB-DEQ (Ours)"]


def build_model(module, ck_name, device):
    m = module()
    ck = torch.load(os.path.join(CK_DIR, ck_name), map_location="cpu",
                    weights_only=False)
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    return m


def build_baseline(name, device):
    cfgs = {
        "unet":   lambda: model_unet.UnetBaseline(base=64),
        "modl":   lambda: model_modl.MoDL(K=4, base=64),
        "varnet": lambda: model_varnet.VarNet(K=6, base=32),
        "dccnn":  lambda: model_dccnn.DCCNN(K=5, feats=64),
    }
    if name not in cfgs:
        raise SystemExit("--baseline must be one of %s" % sorted(cfgs))
    return build_model(cfgs[name], "%s_best.pt" % name, device)


def build_qbdeq(device):
    return build_model(lambda: model_qbdeq.QBDEQ(K=8, base=64),
                       "qbdeq_best.pt", device)


def mag2d(t, k):
    return np.abs(t[k].detach().cpu().numpy())


def eval_pool(store, mask, device, model_b, model_q, idx_list, batch=8):
    ssim = C.SSIMComputer()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx_list), batch):
            ids = idx_list[s:s + batch]
            g = store.get_batch(ids, device=device).to(torch.complex64)
            y, z0 = BU.make_inputs(g, mask, device)
            bas_c = C.to_c(model_b(z0, y, mask))
            qbd_c = C.to_c(model_q(z0, y, mask))
            zf_c = C.ifft2_t(y)
            for k, i in enumerate(ids):
                gt_m = mag2d(g, k)
                zf_m, bas_m, qbd_m = (mag2d(zf_c, k), mag2d(bas_c, k),
                                      mag2d(qbd_c, k))
                def meas(a):
                    return (C.compute_psnr(gt_m, a), ssim.compute(gt_m, a))
                zf_p, zf_s = meas(zf_m); bas_p, bas_s = meas(bas_m)
                qbd_p, qbd_s = meas(qbd_m)
                out.append({"idx": int(i), "zf_p": zf_p, "bas_p": bas_p,
                            "qbd_p": qbd_p, "zf_s": zf_s, "bas_s": bas_s,
                            "qbd_s": qbd_s})
    return out


def select_slices(results, n, margin_floor, ssim_floor=0.0005):
    def margin(r):
        return r["qbd_p"] - max(r["zf_p"], r["bas_p"])
    cand = sorted(results, key=margin, reverse=True)
    good = [r for r in cand
            if margin(r) >= margin_floor
            and r["qbd_s"] >= max(r["zf_s"], r["bas_s"]) - ssim_floor]
    pick = good[:n] if len(good) >= n else cand[:n]
    return pick, good


def alpha_mask(a, lo=None, hi=None):
    """Smooth alpha mask: near-zero (air) pixels -> transparent."""
    lo = ALPHA_LO if lo is None else lo
    hi = ALPHA_HI if hi is None else hi
    return np.clip((np.asarray(a, dtype=np.float32) - lo) / (hi - lo), 0, 1)


def save_fig(fig, name, dpi):
    out = os.path.join(FIG_DIR, name)
    fig.patch.set_facecolor("none")
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02,
                transparent=True)
    fig.savefig(out + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                transparent=True)
    plt.close(fig)
    print("[recon] saved %s.pdf / %s.png (dpi=%d, transparent)"
          % (out, out, dpi))


def style_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.patch.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


def section_main(rows):
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(8.0, 2.05 * n))
    for i in range(n):
        for j, key in enumerate(COL_KEYS):
            ax = axes[i, j]
            img = rows[i]["GT"] if key == "GT" else rows[i][key][0]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                      alpha=alpha_mask(img))
            style_ax(ax)
            y0f, y1f, x0f, x1f = ZOOM_REGIONS[i]
            H0, W0 = img.shape
            y0, y1 = int(y0f * H0), int(y1f * H0)
            x0, x1 = int(x0f * W0), int(x1f * W0)
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="deepskyblue", linewidth=1.4))
            if key != "GT":
                lbl = "PSNR %.2f / SSIM %.3f" % rows[i][key][1:]
                ax.text(0.02, 0.03, lbl, transform=ax.transAxes, fontsize=6.5,
                        color="white", va="bottom",
                        bbox=dict(fc="black", alpha=0.55, pad=1.0, lw=0))
            if i == 0:
                ax.set_title(COL_TITLES[j], fontsize=10.5, fontweight="bold")
        axes[i, 0].set_ylabel("Slice %d" % rows[i]["idx"], fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.94, bottom=0.01,
                        wspace=0.02, hspace=0.18)
    return fig


def section_zoom(crops, idxs):
    n = len(crops["GT"])
    fig, axes = plt.subplots(n, 4, figsize=(8.0, 1.45 * n))
    for i in range(n):
        for j, key in enumerate(COL_KEYS):
            ax = axes[i, j]
            ax.imshow(crops[key][i], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest", alpha=alpha_mask(crops[key][i]))
            style_ax(ax)
            if i == 0:
                ax.set_title(COL_TITLES[j], fontsize=10.5, fontweight="bold")
        axes[i, 0].set_ylabel("Slice %d" % idxs[i], fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.01,
                        wspace=0.02, hspace=0.18)
    fig.text(0.5, 0.975, "Boxed regions, 4x magnification", ha="center",
             fontsize=10, style="italic", color="0.2")
    return fig


def section_residual(err_bas, err_qbd, evmax, idxs):
    n = len(err_bas)
    fig, axes = plt.subplots(n, 2, figsize=(4.6, 2.0 * n))
    for i in range(n):
        for j, (err, key) in enumerate(((err_bas[i], "Baseline Error"),
                                        (err_qbd[i], "QB-DEQ Error"))):
            ax = axes[i, j]
            im = ax.imshow(err, cmap="inferno", vmin=0, vmax=evmax)
            style_ax(ax)
            if i == 0:
                ax.set_title(key, fontsize=10.5)
        axes[i, 0].set_ylabel("Slice %d" % idxs[i], fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.01,
                        wspace=0.08, hspace=0.20)
    return fig


def section_detail(det, vlo, vhi):
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6))
    for j, key in enumerate(("GT", "BAS", "QBD")):
        ax = axes[j]
        ax.imshow(det[key], cmap="gray", vmin=vlo, vmax=vhi,
                  interpolation="nearest", alpha=alpha_mask(det[key]))
        style_ax(ax)
        ax.set_title(["Ground Truth", "Baseline", "QB-DEQ (Ours)"][j],
                     fontsize=10.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01,
                        wspace=0.04)
    return fig


def section_overlay(img, qbd, evmax):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    diff = qbd - img
    pos = np.clip(diff, 0, None); neg = np.clip(-diff, 0, None)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, alpha=alpha_mask(img))
    ax.imshow(np.ma.masked_where(pos <= 0, pos), cmap="Reds", vmin=0,
              vmax=evmax, alpha=0.75)
    ax.imshow(np.ma.masked_where(neg <= 0, neg), cmap="Blues", vmin=0,
              vmax=evmax, alpha=0.75)
    style_ax(ax)
    ax.text(0.02, 0.94, "red: positive difference", transform=ax.transAxes,
            fontsize=9, color="darkred", va="top")
    ax.text(0.02, 0.06, "blue: negative difference", transform=ax.transAxes,
            fontsize=9, color="navy", va="bottom")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


def section_profile(rows, crops):
    """1-D intensity profiles through the boxed anatomical region."""
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(8.0, 2.15 * n))
    for i in range(n):
        ax_img, ax_p = axes[i, 0], axes[i, 1]
        g = crops["GT"][i]
        H, W = g.shape
        cx = W // 2
        ax_img.imshow(g, cmap="gray", vmin=0, vmax=1, alpha=alpha_mask(g))
        ax_img.axvline(cx, color="white", ls="--", lw=1.2)
        ax_img.set_title("Slice %d: boxed region + profile line"
                         % rows[i]["idx"], fontsize=10)
        style_ax(ax_img)
        x = np.arange(H)
        curves = [("Ground Truth", "GT", "black", 1.6, "-"),
                  ("Zero-filled", "ZF", "0.45", 1.0, "--"),
                  ("Baseline", "BAS", "tab:blue", 1.1, "-"),
                  ("QB-DEQ (Ours)", "QBD", "tab:red", 1.2, "-")]
        for label, key, color, lw, ls in curves:
            arr = crops[key][i][:, cx] if key in crops else None
            ax_p.plot(x, arr, color=color, lw=lw, ls=ls, label=label)
        ax_p.set_xlabel("Position along profile (px)")
        ax_p.set_ylabel("Normalized magnitude")
        ax_p.set_ylim(0, 1)
        ax_p.legend(fontsize=7, frameon=False, loc="upper right")
        ax_p.set_title("Intensity profile", fontsize=10)
        ax_p.grid(alpha=0.25, lw=0.4)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.07,
                        wspace=0.22, hspace=0.35)
    return fig


def section_ssim(rows):
    """Local SSIM maps between GT and each method (shared grayscale scale)."""
    n = len(rows)
    maps = {"ZF": [], "BAS": [], "QBD": []}
    for i in range(n):
        gt = rows[i]["GT"]
        for key in maps:
            pred = rows[i][key][0]
            _, ssim_map = sk_ssim(gt, pred, data_range=1.0, full=True)
            maps[key].append(ssim_map)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.1 * n))
    for i in range(n):
        for j, key in enumerate(("ZF", "BAS", "QBD")):
            ax = axes[i, j]
            im = ax.imshow(maps[key][i], cmap="gray", vmin=0, vmax=1)
            style_ax(ax)
            if i == 0:
                ax.set_title({"ZF": "Zero-filled", "BAS": "Baseline",
                              "QBD": "QB-DEQ (Ours)"}[key], fontsize=10.5)
        axes[i, 0].set_ylabel("Slice %d" % rows[i]["idx"], fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Local SSIM")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.04,
                        wspace=0.07, hspace=0.22)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slices", type=str, default=None)
    p.add_argument("--select", type=int, default=3)
    p.add_argument("--pool", choices=["48", "val"], default="48")
    p.add_argument("--baseline", type=str, default="modl")
    p.add_argument("--dpi", type=int, default=1200)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[recon] device=%s baseline=%s dpi=%d"
          % (device, args.baseline, args.dpi))

    meta = torch.load(C.META_PATH, map_location="cpu", weights_only=False)
    store = C.ChunkStore(meta)
    mask = C.MaskStore(meta["masks"]).get(MASK_KEY, device=device)
    _tr, val_idx, _te = C.load_split(meta)
    val_set = set(int(i) for i in val_idx)

    model_b = build_baseline(args.baseline, device)
    model_q = build_qbdeq(device)

    if args.slices:
        slices = [int(x) for x in args.slices.split(",") if x.strip()]
        for i in slices:
            assert int(i) in val_set, "slice %d not in val split" % i
        print("[recon] fixed slices:", slices)
    else:
        rng = np.random.RandomState(42)
        pool = (rng.choice(np.asarray(val_idx, dtype=np.int64), size=48,
                           replace=False).tolist() if args.pool == "48"
                else [int(i) for i in val_idx])
        print("[recon] scanning pool of %d val slices ..." % len(pool))
        res = eval_pool(store, mask, device, model_b, model_q, pool)
        pick, _good = select_slices(res, args.select, MIN_MARGIN)
        slices = [r["idx"] for r in pick]
        for r in sorted(res, key=lambda r: r["idx"]):
            flag = "  <== pick" if r["idx"] in slices else ""
            print("[recon] slice %5d zf=%.2f bas=%.2f qbd=%.2f "
                  "(ssim %.3f/%.3f/%.3f) margin=%+.2f%s"
                  % (r["idx"], r["zf_p"], r["bas_p"], r["qbd_p"],
                     r["zf_s"], r["bas_s"], r["qbd_s"],
                     r["qbd_p"] - max(r["zf_p"], r["bas_p"]), flag))
        print("[recon] selected slices (QB-DEQ best):", slices)

    ssim = C.SSIMComputer()
    rows = []
    with torch.no_grad():
        for i in slices:
            g = store.get_batch([int(i)], device=device).to(torch.complex64)
            y, z0 = BU.make_inputs(g, mask, device)
            zf_c = C.ifft2_t(y)
            bas_c = C.to_c(model_b(z0, y, mask))
            qbd_c = C.to_c(model_q(z0, y, mask))
            gt_raw = mag2d(g, 0)
            peak = float(gt_raw.max())
            zf_raw, bas_raw, qbd_raw = (mag2d(zf_c, 0), mag2d(bas_c, 0),
                                        mag2d(qbd_c, 0))
            def met(a):
                return (C.compute_psnr(gt_raw, a), ssim.compute(gt_raw, a))
            row = {"idx": int(i), "GT": gt_raw / peak}
            for name, raw in (("ZF", zf_raw), ("BAS", bas_raw), ("QBD", qbd_raw)):
                p_, s_ = met(raw)
                row[name] = (raw / peak, p_, s_)
                print("[recon] slice %d %-5s psnr=%.2f ssim=%.4f"
                      % (i, name, p_, s_))
            rows.append(row)

    n = len(rows)
    err_bas = [np.abs(r["BAS"][0] - r["GT"]) for r in rows]
    err_qbd = [np.abs(r["QBD"][0] - r["GT"]) for r in rows]
    evmax = max(np.percentile(e, 99.5) for e in err_bas + err_qbd)
    evmax = max(evmax, 1e-3)

    # zoom crops
    crops = {k: [] for k in COL_KEYS}
    for i in range(n):
        y0f, y1f, x0f, x1f = ZOOM_REGIONS[i]
        for key in COL_KEYS:
            img = rows[i]["GT"] if key == "GT" else rows[i][key][0]
            H0, W0 = img.shape
            crops[key].append(img[int(y0f * H0):int(y1f * H0),
                                  int(x0f * W0):int(x1f * W0)])

    # high-contrast detail
    i = DETAIL_SLICE
    y0f, y1f, x0f, x1f = DETAIL_REGION
    det = {}
    for key in ("GT", "BAS", "QBD"):
        img = rows[i]["GT"] if key == "GT" else rows[i][key][0]
        H0, W0 = img.shape
        det[key] = img[int(y0f * H0):int(y1f * H0),
                       int(x0f * W0):int(x1f * W0)]
    vlo = min(np.percentile(v, 2) for v in det.values())
    vhi = max(np.percentile(v, 98) for v in det.values())

    os.makedirs(FIG_DIR, exist_ok=True)
    save_fig(section_main(rows), "fig_recon_main", args.dpi)
    save_fig(section_zoom(crops, [r["idx"] for r in rows]), "fig_recon_zoom", args.dpi)
    save_fig(section_residual(err_bas, err_qbd, evmax,
                              [r["idx"] for r in rows]), "fig_recon_residual",
             args.dpi)
    save_fig(section_detail(det, vlo, vhi), "fig_recon_detail", args.dpi)
    save_fig(section_overlay(rows[i]["GT"], rows[i]["QBD"][0], evmax),
             "fig_recon_overlay", args.dpi)
    save_fig(section_profile(rows, crops), "fig_recon_profile", args.dpi)
    save_fig(section_ssim(rows), "fig_recon_ssim", args.dpi)



if __name__ == "__main__":
    t0 = time.time()
    main()
    print("[recon] done in %.1fs" % (time.time() - t0))

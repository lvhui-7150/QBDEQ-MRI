# -*- coding: utf-8 -*-
"""GPU reconstruction comparison figure (fig_reconstructions.pdf).

Three val slices x [ZF, K=1, K=4, GT] magnitude images, with per-panel
PSNR/SSIM (fastMRI magnitude convention, identical to step5_k_scan.py).
Checkpoint: runs/step5_final/full/checkpoint_best.pt (epoch 85).
All metrics are recomputed here from the loaded model; nothing is hardcoded.
"""
import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_DIR = os.path.join(ROOT, "论文部分", "figures")
CK_PATH = os.path.join(HERE, "runs", "step5_final", "full", "checkpoint_best.pt")

sys.path.insert(0, HERE)
import step5_320_ceiling as C
from step5_train_final import CascadeNet

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 8,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "savefig.dpi": 300,
})

VAL_SLICES = [2390, 5661, 4184]   # from step5_320_diag: 24.08 / 22.53 / 24.89 dB zf
MASK_KEY = "r4_s42"


def log(m):
    print("[RECON] %s" % m, flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("device=%s" % device)
    meta = torch.load(C.META_PATH, map_location="cpu", weights_only=False)
    store = C.ChunkStore(meta)
    mask_store = C.MaskStore(meta["masks"])
    train_idx, val_idx, test_idx = C.load_split(meta)
    val_set = set(int(i) for i in val_idx)
    for i in VAL_SLICES:
        assert int(i) in val_set, "slice %d not in val split" % i
    log("val slices confirmed: %s" % (VAL_SLICES,))

    ck = torch.load(CK_PATH, map_location="cpu", weights_only=False)
    model = CascadeNet(in_ch=2, base=64, K=4, eta_init=0.5).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    log("checkpoint epoch=%s val_psnr=%.4f n_params=%s" % (
        ck.get("epoch"), ck.get("val_psnr", -1.0), ck.get("n_params")))

    mask = mask_store.get(MASK_KEY, device=device)
    ssim = C.SSIMComputer()

    rows = []
    with torch.no_grad():
        for i in VAL_SLICES:
            g = store.get_batch([i], device=device)          # (1,H,W) complex
            yb = C.fft2_t(g) * mask
            z0 = C.to_2ch(C.ifft2_t(yb))
            outs = model(z0, yb, mask, k_max=4)
            zf = C.ifft2_t(yb)
            gt = g[0].detach().cpu().numpy()
            mag = lambda t: np.abs(t[0].detach().cpu().numpy())
            zf_m = mag(zf); k1_m = mag(outs[0]); k4_m = mag(outs[3])
            gt_m = np.abs(gt)
            def met(pred_m):
                p = C.compute_psnr(gt_m, pred_m)
                s = ssim.compute(gt_m, pred_m)
                return p, s
            row = {"idx": int(i)}
            for name, m in [("ZF", zf_m), ("K1", k1_m), ("K4", k4_m)]:
                p, s = met(m)
                row[name] = (m, p, s)
                log("slice %d %s: psnr=%.2f ssim=%.4f" % (i, name, p, s))
            row["GT"] = gt_m
            rows.append(row)

    # ---- figure: 3 rows (slices) x 4 cols (ZF, K1, K4, GT) ----
    cols = ["ZF", "K1", "K4", "GT"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.02, hspace=0.22)
    vmin, vmax = 0.0, 1.0
    for r, row in enumerate(rows):
        peak = float(torch.abs(store.get(row["idx"])).max())  # gt peak for this slice
        for c, col in enumerate(cols):
            ax = axes[r, c]
            if col == "GT":
                im, p, s = row["GT"], None, None
                title = "GT"
            else:
                im, p, s = row[col]
                title = "%s\nPSNR %.2f / SSIM %.3f" % ({"ZF":"ZF","K1":"K=1","K4":"K=4"}[col], p, s)
            ax.imshow(im / peak, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=7.5)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("slice %d\n(zf %.1f dB)" % (row["idx"],
                              row["ZF"][1]), fontsize=7.5)
    fig.suptitle("Magnitude reconstruction, 4x mask r4_s42 (val slices)", fontsize=9)
    out = os.path.join(FIG_DIR, "fig_reconstructions.pdf")
    fig.savefig(out)
    plt.close(fig)
    log("saved %s" % out)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("[RECON] done in %.1fs" % (time.time() - t0))



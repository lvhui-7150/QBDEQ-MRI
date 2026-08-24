# -*- coding: utf-8 -*-
"""model_recon_figs.py —— 生成 6 个模型的重构对比图（训练完成后运行）

在 3 个验证切片上，对 ZF / U-Net / MoDL / VarNet / DCCNN / CascadeNet /
QB-DEQ / GT 生成幅度图像，并标注每格 PSNR/SSIM。
输出：论文部分/figures/fig_baseline_recon.pdf + 论文部分/fig_baseline_recon.png

用法：python model_recon_figs.py [--slices 2390,5661,4184]
"""
import os
import sys
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import baseline_utils as BU
import step5_320_ceiling as C

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CK_DIR = os.path.join(HERE, "runs", "baseline")
FIG_OUT = os.path.join(ROOT, "论文部分", "figures", "fig_baseline_recon.pdf")
PNG_OUT = os.path.join(ROOT, "论文部分", "fig_baseline_recon.png")

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7, "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"], "savefig.dpi": 300,
})


def build_all(args):
    from model_unet import UnetBaseline
    from model_modl import MoDL
    from model_varnet import VarNet
    from model_dccnn import DCCNN
    from model_cascadenet import CascadeNetBaseline
    from model_qbdeq import QBDEQ
    cfgs = [
        ("unet", UnetBaseline(base=args.base)),
        ("modl", MoDL(K=args.K_modl, base=args.base)),
        ("varnet", VarNet(K=args.K_varnet, base=args.base_varnet)),
        ("dccnn", DCCNN(K=args.K_dccnn, feats=args.base)),
        ("cascadenet", CascadeNetBaseline(K=args.K_casc, base=args.base)),
        ("qbdeq", QBDEQ(K=args.K_qbdeq, base=args.base)),
    ]
    models = []
    for name, m in cfgs:
        ck = torch.load(os.path.join(CK_DIR, "%s_best.pt" % name),
                        map_location="cpu", weights_only=False)
        m.load_state_dict(ck["state_dict"])
        models.append((name, m))
        print("[%s] loaded best (ep=%s val=%.2f)" % (name, ck["epoch"], ck["val_psnr"]))
    return models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slices", type=str, default="2390,5661,4184")
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--K-modl", type=int, default=4)
    p.add_argument("--K-varnet", type=int, default=6)
    p.add_argument("--base-varnet", type=int, default=32)
    p.add_argument("--K-dccnn", type=int, default=5)
    p.add_argument("--K-casc", type=int, default=4)
    p.add_argument("--K-qbdeq", type=int, default=8)
    args = p.parse_args()
    slices = [int(x) for x in args.slices.split(",") if x.strip()]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    store, mask, _tr, _va, _te, _ = BU.load_data(device=dev, train_subset=0,
                                                 val_max=48, test_subset=0)
    models = build_all(args)
    for _, m in models:
        m.to(dev).eval()

    ssim = C.SSIMComputer()
    cols = ["ZF"] + [n for n, _ in models] + ["GT"]
    n_rows, n_cols = len(slices), len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.2 * n_rows))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01,
                        wspace=0.03, hspace=0.20)
    for r, si in enumerate(slices):
        x = store.get_batch([int(si)], device=dev).to(torch.complex64)
        y, z0 = BU.make_inputs(x, mask, dev)
        with torch.no_grad():
            zf = C.ifft2_t(y)
            outs = {"ZF": zf}
            for name, m in models:
                outs[name] = C.to_c(m(z0, y, mask))
        peak = float(torch.abs(x).max().item())
        for c, col in enumerate(cols):
            ax = axes[r, c]
            if col == "GT":
                im = torch.abs(x[0]).detach().cpu().numpy()
                title = "GT"
            else:
                im = torch.abs(outs[col][0]).detach().cpu().numpy()
                gm = np.abs(x[0].detach().cpu().numpy())
                ps = C.compute_psnr(gm, im)
                ss = ssim.compute(gm, im)
                title = "%s\nPSNR %.2f / SSIM %.3f" % (col, ps, ss)
            ax.imshow(im / peak, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=6.5)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("slice %d\n(ZF %.1f dB)" % (si, C.compute_psnr(
                    np.abs(x[0].detach().cpu().numpy()),
                    np.abs(C.ifft2_t(y)[0].detach().cpu().numpy()))), fontsize=6.5)
    fig.suptitle("Magnitude reconstructions, 4x mask r4_s42 (5-epoch diagnostic models)",
                 fontsize=9, y=0.97)
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT)
    fig.savefig(PNG_OUT)
    plt.close(fig)
    print("saved", FIG_OUT)
    print("saved", PNG_OUT)


if __name__ == "__main__":
    sys.exit(main())

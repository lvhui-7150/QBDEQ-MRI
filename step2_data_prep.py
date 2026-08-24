# -*- coding: utf-8 -*-
"""
Step 2 -- fastMRI data preparation for QB-DEQ rebuild (main experiment).

Run:
    cd "C:\\Users\\Administrator\\Desktop\\论文\\第三篇全部重做\\实验部分"
    python step2_data_prep.py

Inputs:
    fastmri_128.pt        -- 199 x 128 x 128 complex64 (fastMRI knee, already ~[0,1])

Outputs:
    fastmri_128_prepared.pt   -- gt_complex + masks(R=4/8 x seeds 42/123/2025) + split + meta
    step2_report.json         -- machine-readable validation (what I read to decide next)
    step2_figs/*.png          -- mask / kspace / zero-filled diagnostics

Forward model convention (used by all later steps):
    k  = torch.fft.fft2(x, norm="ortho")   # x: complex image
    y  = k * mask                          # masked kspace (Cartesian, along ky)
    zf = torch.fft.ifft2(y, norm="ortho")  # zero-filled recon
"""

import os
import sys
import json
import time
import math

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE_DIR, "fastmri_128.pt")
PREPARED = os.path.join(BASE_DIR, "fastmri_128_prepared.pt")
FIG_DIR = os.path.join(BASE_DIR, "step2_figs")
REPORT_PATH = os.path.join(BASE_DIR, "step2_report.json")
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 123, 2025]
RATES = [4, 8]
CENTER_FRACTION = 0.08
VD_POWER = 2.0
SPLIT_SEED = 42
SPLIT_RATIO = (0.8, 0.1, 0.1)

report = {
    "script": "step2_data_prep",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "input": INPUT,
        "image_size": 128,
        "rates": RATES,
        "seeds": SEEDS,
        "center_fraction": CENTER_FRACTION,
        "vd_power": VD_POWER,
        "split": {"ratio": list(SPLIT_RATIO), "seed": SPLIT_SEED},
    },
}
issues = []


def line(tag, msg):
    print(f"[{tag}] {msg}")


def add_issue(level, msg):
    issues.append({"level": level, "msg": msg})
    print(f"[{level}] {msg}")


def make_cartesian_vd_mask(n, rate, center_frac, power, gen):
    """1D phase-encode (ky) mask; Cartesian = same pattern along kx.

    Center (center_frac*n) lines fully sampled; outer lines sampled with
    probability ~ (1 - d)^power (d = normalized distance from center),
    then count-fixed so total sampled lines == n/rate exactly.
    """
    n_center = int(round(n * center_frac))
    if n_center % 2 == 1:
        n_center -= 1
    half = n_center // 2
    center = set(range(n // 2 - half, n // 2 + half))
    outer = [i for i in range(n) if i not in center]
    d = [abs(i - (n - 1) / 2.0) / (n / 2.0) for i in outer]
    base = [max(0.0, (1.0 - di) ** power) for di in d]

    target_outer = n // rate - n_center
    assert target_outer >= 0, "rate too high for this n/center"

    def expected(scale):
        return sum(min(1.0, scale * b) for b in base)

    lo, hi = 0.0, 1e6
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if expected(mid) < target_outer:
            lo = mid
        else:
            hi = mid
    scale = hi

    probs = [min(1.0, scale * b) for b in base]
    chosen = set()
    for i, pr in zip(outer, probs):
        if torch.rand(1, generator=gen).item() < pr:
            chosen.add(i)

    if len(chosen) < target_outer:
        cand = [i for i in outer if i not in chosen]
        cand.sort(key=lambda i: base[outer.index(i)], reverse=True)
        for i in cand[: target_outer - len(chosen)]:
            chosen.add(i)
    elif len(chosen) > target_outer:
        drop = sorted(chosen, key=lambda i: base[outer.index(i)])
        for i in drop[: len(chosen) - target_outer]:
            chosen.remove(i)

    mask = torch.zeros(n, dtype=torch.bool)
    mask[list(center)] = True
    mask[list(chosen)] = True
    return mask


def mask_stats_1d(m1d, n_center, rate, seed):
    frac = float(m1d.float().mean())
    return {
        "rate": rate,
        "seed": seed,
        "n_center": n_center,
        "n_outer": int(m1d.sum().item()) - n_center,
        "sampling_frac": round(frac, 6),
        "effective_R": round(1.0 / frac, 4) if frac > 0 else None,
        "center_frac_actual": round(n_center / m1d.numel(), 6),
        "row_density_head": [int(v) for v in m1d[:16].tolist()],
        "checks": {
            "effective_R_ok": abs(1.0 / frac - rate) < 0.01 if frac > 0 else False,
            "center_frac_ok": abs(n_center / m1d.numel() - 0.08) < 0.01,
        },
    }


def write_report(t0):
    n_err = sum(1 for i in issues if i["level"] == "ERROR")
    n_warn = sum(1 for i in issues if i["level"] == "WARN")
    report["issues"] = issues
    report["verdict"] = "ERROR" if n_err else ("WARN" if n_warn else "OK")
    report["elapsed_sec"] = round(time.time() - t0, 1)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    line("MAIN", f"done in {report['elapsed_sec']}s")
    line("MAIN", f"verdict: {report['verdict']}  (errors={n_err}, warnings={n_warn})")
    line("MAIN", f"report: {REPORT_PATH}")
    if "prepared_file" in report:
        line("MAIN", f"prepared: {PREPARED}")
    line("MAIN", f"figures: {FIG_DIR}")
    return n_err == 0


def make_figures(gt, masks, mask_meta, test_slices, split):
    made = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        add_issue("WARN", f"matplotlib import failed: {e}")
        return made

    def save(fig, name):
        p = os.path.join(FIG_DIR, name)
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
        line("FIG", f"saved {p}")

    # fig1: mask grid (2 rates x 3 seeds)
    try:
        fig, axes = plt.subplots(len(RATES), len(SEEDS), figsize=(3.4 * len(SEEDS), 3.2 * len(RATES)))
        for ri, rate in enumerate(RATES):
            for si, seed in enumerate(SEEDS):
                key = f"r{rate}_s{seed}"
                ax = axes[ri, si]
                ax.imshow(masks[key].numpy(), cmap="gray")
                meta = mask_meta[key]
                ax.set_title(f"R={rate} seed={seed}\neffR={meta['effective_R']} ctr={meta['center_frac_actual']}")
                ax.axis("off")
        fig.suptitle("fastMRI Cartesian variable-density masks (128x128, center 8%)", fontsize=10)
        save(fig, "fig1_masks_grid.png")
    except Exception as e:
        add_issue("WARN", f"fig1 failed: {type(e).__name__}: {e}")

    # fig2: kspace + zero-filled sanity for one test slice
    try:
        i = test_slices[0]
        x = gt[i]
        k = torch.fft.fft2(x, norm="ortho")
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
        axes[0, 0].imshow(x.abs().numpy(), cmap="gray")
        axes[0, 0].set_title(f"GT slice {i} (mag)")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(torch.log(1 + k.abs()).numpy(), cmap="gray")
        axes[0, 1].set_title("full kspace |log(1+.)|")
        axes[0, 1].axis("off")
        for c, rate in enumerate(RATES):
            m = masks[f"r{rate}_s{SEEDS[0]}"]
            y = k * m
            zf = torch.fft.ifft2(y, norm="ortho")
            if c == 0:
                axes[0, 2].imshow(torch.log(1 + y.abs()).numpy(), cmap="gray")
                axes[0, 2].set_title(f"masked kspace R={rate}")
                axes[0, 2].axis("off")
            else:
                axes[1, 0].imshow(torch.log(1 + y.abs()).numpy(), cmap="gray")
                axes[1, 0].set_title(f"masked kspace R={rate}")
                axes[1, 0].axis("off")
            axes[1, 1 + c].imshow(zf.abs().numpy(), cmap="gray")
            axes[1, 1 + c].set_title(f"zero-filled R={rate}")
            axes[1, 1 + c].axis("off")
        fig.suptitle("forward model sanity: kspace under mask -> zero-filled recon", fontsize=10)
        save(fig, "fig2_kspace_zf.png")
    except Exception as e:
        add_issue("WARN", f"fig2 failed: {type(e).__name__}: {e}")

    # fig3: split sizes + per-split magnitude stats
    try:
        sizes = report["split"]["sizes"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        axes[0].bar(["train", "val", "test"], [sizes["train"], sizes["val"], sizes["test"]])
        axes[0].set_title("80/10/10 split (seed 42)")
        axes[0].set_ylabel("slices")
        mags = gt.abs().mean(dim=(1, 2))
        for si, name in enumerate(["train", "val", "test"]):
            vals = mags[split[name]].numpy()
            axes[1].hist(vals, bins=20, alpha=0.55, label=name)
        axes[1].set_title("per-slice mean magnitude")
        axes[1].set_xlabel("mean |x|")
        axes[1].legend()
        save(fig, "fig3_split.png")
    except Exception as e:
        add_issue("WARN", f"fig3 failed: {type(e).__name__}: {e}")

    return made


def main():
    global torch
    import torch
    t0 = time.time()
    line("MAIN", f"started: {report['timestamp']}")

    # ---- 1. load & validate input ----
    line("DATA", "loading fastmri_128.pt ...")
    if not os.path.exists(INPUT):
        add_issue("ERROR", f"input missing: {INPUT}")
        write_report(t0)
        return 1
    obj = torch.load(INPUT, map_location="cpu", weights_only=False)
    if not (isinstance(obj, dict) and "gt_complex" in obj and isinstance(obj["gt_complex"], torch.Tensor)):
        add_issue("ERROR", "unexpected structure; expected dict with gt_complex tensor")
        write_report(t0)
        return 1
    gt = obj["gt_complex"].detach().cpu()
    if gt.dtype != torch.complex64:
        gt = gt.to(torch.complex64)
    if gt.dim() != 3 or tuple(gt.shape[-2:]) != (128, 128):
        add_issue("ERROR", f"expected (N,128,128) complex, got {list(gt.shape)}")
        write_report(t0)
        return 1
    n = gt.shape[0]
    if n != 199:
        add_issue("WARN", f"expected 199 slices, got {n}")

    mag = gt.abs()
    nan_inf = int(torch.isnan(gt).sum().item()) + int(torch.isinf(gt).sum().item())
    report["data"] = {
        "n_slices": n,
        "shape": list(gt.shape),
        "dtype": str(gt.dtype),
        "nan_inf": nan_inf,
        "mag_global": {"min": round(float(mag.min()), 6), "mean": round(float(mag.mean()), 6),
                       "max": round(float(mag.max()), 6), "std": round(float(mag.std()), 6)},
        "mag_per_slice_mean": {"min": round(float(mag.mean(dim=(1, 2)).min()), 6),
                               "mean": round(float(mag.mean(dim=(1, 2)).mean()), 6),
                               "max": round(float(mag.mean(dim=(1, 2)).max()), 6)},
        "normalization": "already in [0,1]; kept as-is (no rescale)",
    }
    if nan_inf > 0:
        add_issue("ERROR", "NaN/Inf found in data")
    line("DATA", f"n={n} shape={list(gt.shape)} mag mean={report['data']['mag_global']['mean']} (0..1, OK)")

    # ---- 2. split (fixed seed 42) ----
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    idx = torch.randperm(n, generator=gen).tolist()
    n_tr, n_va = int(round(n * 0.8)), int(round(n * 0.1))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    split = {"train": tr, "val": va, "test": te, "seed": SPLIT_SEED}
    report["split"] = {
        "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "seed": SPLIT_SEED,
        "disjoint": len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0,
        "cover_all": len(set(tr) | set(va) | set(te)) == n,
        "train_head": tr[:10],
        "val_head": va[:10],
        "test_head": te[:10],
    }
    line("SPLIT", f"train={len(tr)} val={len(va)} test={len(te)} (seed {SPLIT_SEED})")

    # ---- 3. masks (R=4/8 x seeds 42/123/2025) ----
    masks, mask_meta = {}, {}
    for rate in RATES:
        for seed in SEEDS:
            g = torch.Generator().manual_seed(seed)
            m1d = make_cartesian_vd_mask(128, rate, CENTER_FRACTION, VD_POWER, g)
            m2d = m1d.unsqueeze(1).expand(128, 128).clone()
            key = f"r{rate}_s{seed}"
            masks[key] = m2d
            n_center = int(round(128 * CENTER_FRACTION))
            if n_center % 2 == 1:
                n_center -= 1
            meta = mask_stats_1d(m1d, n_center, rate, seed)
            mask_meta[key] = meta
            ok = meta["checks"]["effective_R_ok"] and meta["checks"]["center_frac_ok"]
            if not ok:
                add_issue("ERROR", f"mask {key} failed checks: {meta}")
            line("MASK", f"{key}: effR={meta['effective_R']} ctr={meta['center_frac_actual']} outer={meta['n_outer']}")
    report["masks"] = mask_meta

    # ---- 4. zero-filled sanity (validates forward model + masks) ----
    zf_report = {}
    test_slices = split["test"][:3]
    try:
        for rate in RATES:
            m = masks[f"r{rate}_s{SEEDS[0]}"]
            psnrs, nmses = [], []
            for i in test_slices:
                x = gt[i]
                k = torch.fft.fft2(x, norm="ortho")
                zf = torch.fft.ifft2(k * m, norm="ortho")
                xm, zm = x.abs(), zf.abs()
                mse = float(((xm - zm) ** 2).mean())
                peak = float(xm.max())
                psnr = 10.0 * math.log10(peak ** 2 / (mse + 1e-12)) if mse > 0 else float("inf")
                nmse = float(((xm - zm) ** 2).sum() / ((xm) ** 2).sum())
                psnrs.append(round(psnr, 3))
                nmses.append(round(nmse, 6))
            zf_report[f"r{rate}"] = {
                "slices": test_slices,
                "psnr": psnrs,
                "nmse": nmses,
                "psnr_mean": round(sum(psnrs) / len(psnrs), 3),
            }
            line("ZF", f"R={rate}: zero-filled PSNR {psnrs} (mean {zf_report[f'r{rate}']['psnr_mean']} dB)")
    except Exception as e:
        add_issue("WARN", f"zero-filled sanity failed: {type(e).__name__}: {e}")
    report["zero_filled_sanity"] = zf_report

    # ---- 5. save prepared file ----
    prepared = {
        "description": "fastMRI knee 128, prepared for QB-DEQ rebuild (main experiment)",
        "gt_complex": gt,
        "masks": masks,
        "mask_meta": mask_meta,
        "split": split,
        "meta": {
            "forward_model": "k = fft2(x, norm='ortho'); y = k * mask; zf = ifft2(y, norm='ortho')",
            "mask_axes": "mask rows = ky (phase encode), cols = kx (readout); Cartesian: constant along kx",
            "normalization": report["data"]["normalization"],
            "center_fraction": CENTER_FRACTION,
            "vd_power": VD_POWER,
            "created": report["timestamp"],
        },
    }
    torch.save(prepared, PREPARED)
    report["prepared_file"] = {"path": PREPARED, "size_mb": round(os.path.getsize(PREPARED) / 2**20, 2)}

    # ---- 6. figures ----
    report["figures"] = make_figures(gt, masks, mask_meta, test_slices, split)

    ok = write_report(t0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
# -*- coding: utf-8 -*-
"""
Step 3.3 v2 -- unrolled baseline recipe fix + MoDL 3-seed sweep (fastMRI).

What changed vs v1 (which errored only in the final summary writer -- the
training/metrics/figures all completed):
  1. fixed the summary-formatting bug (eta shown as string),
  2. every method now starts from its OWN fixed seed (reproducible no matter
     the run order / what ran before),
  3. modl_lite is trained 3 times with the roadmap seeds 42/123/2025 and
     reported as mean +- std (variance of random init), unet/deq once each,
  4. JSON reports are written BEFORE the text summary (defensive: a summary
     bug can never again lose the machine-readable results).

Recipe (same strong-but-fair recipe for every learned method):
    lr 3e-4, AdamW, linear warmup (5 epochs) + cosine decay, 40 epochs
    unrolled: progressive K=2 for 5 epochs, then K=6, learnable DC eta (init 1)
    best checkpoint = best val PSNR among full-K epochs only (no early spikes)

Methods:
    1. zero_filled     : reference floor (no training)
    2. tv_lite         : GD on ||M F x - y||^2 + lam*TV(|x|), lam swept
    3. unet_lite       : direct U-Net (seed 42)
    4. modl_lite       : unrolled MoDL-lite K=6, seeds 42/123/2025 (mean+-std)
    5. euclid_deq_lite : unrolled Euclidean DEQ-lite K=6 (seed 42)

Fixed config: rate=4, mask=r4_s42, split seed=42 (from prepared file)
loss = L1(mag) + 0.1*complex-MSE + 0.01*k-space-DC   (roadmap 3.5)

Run (from the experiment folder):
    python step3_unrolled_fix.py

Outputs (what I will read to decide next):
    step3_fix_report.json   -- verdict + per-method metrics + seed sweep
    step3_fix_summary.txt   -- human-readable table (read this first)
    runs/step3_fix/         -- config.json, history.csv, metrics.json,
                               checkpoint_best/last_*.pt, stdout.log
    step3_figs/fig*_fix.png -- recon + error-map panels (for your eyeballing)
"""

import os
import sys
import json
import time
import math
import random
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPARED = os.path.join(BASE_DIR, "fastmri_128_prepared.pt")
OUT_DIR = os.path.join(BASE_DIR, "runs", "step3_fix")
FIG_DIR = os.path.join(BASE_DIR, "step3_figs")
REPORT_PATH = os.path.join(BASE_DIR, "step3_fix_report.json")
SUMMARY_PATH = os.path.join(BASE_DIR, "step3_fix_summary.txt")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "examples"), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---- experimental config ---------------------------------------------------
SEED = 42
RATE = 4
MASK_KEY = "r4_s42"
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
WARMUP_EPOCHS = 5
TOTAL_EPOCHS = 40
K_WARMUP = 2
K_FULL = 6
CNN_BASE = 32
MODL_SEEDS = [42, 123, 2025]
METHOD_SEED = 42          # seed for unet_lite / euclid_deq_lite
TV_ITERS = 80
TV_LR = 0.5
TV_LAMBDA_GRID = [0.005, 0.02, 0.05]
VAL_PSNR_MARGIN = 1.0     # learned methods must beat zero-filled by this many dB
DC_RESID_TOL = 0.2        # residual below this counts as "converged" (eval metric)
VERDICT_TIME_BUDGET_S = 1500.0

report = {
    "script": "step3_unrolled_fix_v2",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "prepared": PREPARED,
        "rate": RATE,
        "mask_key": MASK_KEY,
        "split_seed": SEED,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "warmup_epochs": WARMUP_EPOCHS,
        "total_epochs": TOTAL_EPOCHS,
        "progressive_k": {"warmup": K_WARMUP, "full": K_FULL},
        "learnable_eta": True,
        "per_method_seed": METHOD_SEED,
        "modl_seed_sweep": MODL_SEEDS,
        "best_ckpt_policy": "best val PSNR among full-K epochs (epoch > warmup)",
        "cnn_base": CNN_BASE,
        "tv": {"iters": TV_ITERS, "lr": TV_LR, "lambda_grid": TV_LAMBDA_GRID},
        "loss": "L1(mag) + 0.1*complexMSE + 0.01*kspaceDC",
        "dc_resid_tol": DC_RESID_TOL,
    },
    "methods": {},
    "tv_sweep": {},
    "modl_seed_sweep": {},
    "issues": [],
}
issues = []


def line(tag, msg):
    print(f"[{tag}] {msg}")


def add_issue(level, msg):
    issues.append({"level": level, "msg": msg})
    print(f"[{level}] {msg}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_2ch(x_c):
    """(N,H,W) complex -> (N,2,H,W) real."""
    return torch.stack([x_c.real, x_c.imag], dim=1)


def to_c(x_2ch):
    """(N,2,H,W) real -> (N,H,W) complex."""
    return torch.complex(x_2ch[:, 0], x_2ch[:, 1])


def fwd_kspace(x_c):
    return torch.fft.fft2(x_c, norm="ortho")


def adjoint(y):
    return torch.fft.ifft2(y, norm="ortho")


def sense(x_c, mask):
    """Masked k-space measurement: y = M F x (Cartesian along ky)."""
    return fwd_kspace(x_c) * mask


def tv_term(mag):
    gx = mag[..., 1:, :] - mag[..., :-1, :]        # (..., H-1, W)
    gy = mag[..., :, 1:] - mag[..., :, :-1]        # (..., H, W-1)
    gx = torch.nn.functional.pad(gx, (0, 0, 0, 1))
    gy = torch.nn.functional.pad(gy, (0, 1, 0, 0))
    return torch.sqrt(gx * gx + gy * gy + 1e-8).sum()


def recon_loss(x_hat_c, x_gt_c, y, mask):
    l_mag = (x_hat_c.abs() - x_gt_c.abs()).abs().mean()
    l_c = ((x_hat_c - x_gt_c).abs() ** 2).mean()
    l_dc = ((fwd_kspace(x_hat_c) * mask - y).abs() ** 2).mean()
    return l_mag + 0.1 * l_c + 0.01 * l_dc
# ---- metrics (per slice) ---------------------------------------------------
def per_slice_psnr(x_hat_c, x_gt_c):
    mse = ((x_hat_c - x_gt_c).abs() ** 2).mean(dim=(-1, -2))
    peak = x_gt_c.abs().amax(dim=(-1, -2))
    return 20.0 * torch.log10(peak / (torch.sqrt(mse) + 1e-12))


def per_slice_nmse(x_hat_c, x_gt_c):
    num = ((x_hat_c - x_gt_c).abs() ** 2).sum(dim=(-1, -2))
    den = (x_gt_c.abs() ** 2).sum(dim=(-1, -2))
    return num / (den + 1e-12)


def per_slice_phase_err_deg(x_hat_c, x_gt_c):
    a = torch.angle(x_hat_c)
    b = torch.angle(x_gt_c)
    d = ((a - b + math.pi) % (2.0 * math.pi)) - math.pi
    d = d.abs()
    w = x_gt_c.abs()
    return (w * d).sum(dim=(-1, -2)) / (w.sum(dim=(-1, -2)) + 1e-12) * (180.0 / math.pi)


def per_slice_ssim(x_hat_c, x_gt_c):
    from skimage.metrics import structural_similarity as _ssim
    mag_hat = x_hat_c.abs().detach().cpu().numpy()
    mag_gt = x_gt_c.abs().detach().cpu().numpy()
    return np.array(
        [_ssim(g, h, data_range=1.0) for g, h in zip(mag_gt, mag_hat)],
        dtype=np.float32,
    )


def per_slice_dc_residual(x_hat_c, y, mask):
    k = fwd_kspace(x_hat_c) * mask
    num = ((k - y).abs() ** 2).sum(dim=(-1, -2))
    den = (y.abs() ** 2).sum(dim=(-1, -2))
    return torch.sqrt(num / (den + 1e-12))


# ---- models ----------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    """2-level U-Net, 2ch in/out. Used standalone and as MoDL/DEQ denoiser."""

    def __init__(self, in_ch=2, base=16):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bot = ConvBlock(base * 2, base * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(base * 3, base)
        self.head = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        u = self.up2(b)
        u = torch.cat([u, e2], dim=1)
        u = self.dec2(u)
        u = self.up1(u)
        u = torch.cat([u, e1], dim=1)
        u = self.dec1(u)
        return self.head(u)


class DCCell(nn.Module):
    """z <- D( z - eta * A*(M A z - y) ). eta is learnable, init 1.0."""

    def __init__(self, denoiser, eta_init=1.0):
        super().__init__()
        self.denoiser = denoiser
        self.eta = nn.Parameter(torch.tensor(float(eta_init)))

    def forward(self, z, y, mask):
        zc = to_c(z)
        k = fwd_kspace(zc)
        dc = zc - self.eta * adjoint((k - y) * mask)
        return self.denoiser(to_2ch(dc))


class UnrolledRecon(nn.Module):
    """Unrolled fixed-point iteration (route A training, roadmap 3.4)."""

    def __init__(self, cell, k):
        super().__init__()
        self.cell = cell
        self.k = k

    def forward(self, z0, y, mask, k=None):
        steps = self.k if k is None else k
        z = z0
        for _ in range(steps):
            z = self.cell(z, y, mask)
        return z


def build_model(name):
    """Build the exact architecture used for method {name}."""
    if name == "unet_lite":
        return SmallUNet(in_ch=2, base=CNN_BASE)
    cell = DCCell(SmallUNet(in_ch=2, base=CNN_BASE), eta_init=1.0)
    return UnrolledRecon(cell, K_FULL)


# ---- schedule --------------------------------------------------------------
def lr_at_epoch(epoch):
    """Linear warmup then cosine decay; epoch is 1-based."""
    if epoch <= WARMUP_EPOCHS:
        return LR * epoch / WARMUP_EPOCHS
    t = (epoch - WARMUP_EPOCHS) / max(1, TOTAL_EPOCHS - WARMUP_EPOCHS)
    return LR * 0.5 * (1.0 + math.cos(math.pi * t))


# ---- baselines -------------------------------------------------------------
def run_zero_filled(gt_c, mask):
    y = sense(gt_c, mask)
    return adjoint(y), y


def run_tv_lite(gt_c, mask, iters=TV_ITERS, lr=TV_LR, lam=0.02):
    y = sense(gt_c, mask)
    outs = []
    for i in range(gt_c.shape[0]):
        z0 = adjoint(y[i : i + 1])
        x = nn.Parameter(to_2ch(z0).clone())
        opt = torch.optim.SGD([x], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            xc = to_c(x)
            dc = ((fwd_kspace(xc) * mask - y[i : i + 1]).abs() ** 2).sum()
            loss = dc + lam * tv_term(xc.abs())
            loss.backward()
            opt.step()
        outs.append(to_c(x.detach()))
    return torch.cat(outs, dim=0), y


def run_tv_sweep(gt_c, mask):
    n_sweep = min(5, gt_c.shape[0])
    sweep_rows = []
    best = (TV_LAMBDA_GRID[1], -1.0)
    for lam in TV_LAMBDA_GRID:
        xc, _ = run_tv_lite(gt_c[:n_sweep], mask, iters=TV_ITERS, lr=TV_LR, lam=lam)
        psnr = float(per_slice_psnr(xc, gt_c[:n_sweep]).mean())
        ssim = float(per_slice_ssim(xc, gt_c[:n_sweep]).mean())
        sweep_rows.append({"lam": lam, "psnr": round(psnr, 3), "ssim": round(ssim, 3)})
        line("TVSWEEP", f"lam={lam}: psnr={psnr:.2f} ssim={ssim:.3f}")
        if psnr > best[1]:
            best = (lam, psnr)
    chosen = best[0]
    report["tv_sweep"] = {"grid": sweep_rows, "chosen_lam": chosen}
    line("TVSWEEP", f"chosen lam={chosen}")
    return chosen
# ---- data plumbing ---------------------------------------------------------
def make_batch_perm(base_seed, epoch, n_train):
    gen = torch.Generator().manual_seed(int(base_seed) * 10000 + int(epoch))
    return torch.randperm(n_train, generator=gen)


def prep_val(gt, val_idx, mask, device):
    gt_val = gt[val_idx].to(device)
    y_val = sense(gt_val, mask)
    z0_val = to_2ch(adjoint(y_val))
    return gt_val, y_val, z0_val


# ---- training --------------------------------------------------------------
def train_direct(name, base_seed, train_idx, val_idx, gt, mask, device, history_rows):
    set_seed(base_seed)
    model = SmallUNet(in_ch=2, base=CNN_BASE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_params = sum(p.numel() for p in model.parameters())
    line("TRAIN", f"{name}: direct U-Net params={n_params} epochs={TOTAL_EPOCHS} lr={LR} warmup={WARMUP_EPOCHS} cosine")

    gt_val, y_val, z0_val = prep_val(gt, val_idx, mask, device)
    best_psnr, best_epoch = -1.0, 0
    history = {"loss": [], "val_psnr": []}
    t0 = time.perf_counter()
    for epoch in range(1, TOTAL_EPOCHS + 1):
        lr = lr_at_epoch(epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        perm = make_batch_perm(base_seed, epoch, len(train_idx))
        losses = []
        for s in range(0, len(perm), BATCH_SIZE):
            b = train_idx[perm[s : s + BATCH_SIZE]]
            x = gt[b].to(device)
            yb = sense(x, mask)
            z0 = to_2ch(adjoint(yb))
            z = model(z0)
            loss = recon_loss(to_c(z), x, yb, mask)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            z = model(z0_val)
            vp = per_slice_psnr(to_c(z), gt_val).mean().item()
        history["loss"].append(mean_loss)
        history["val_psnr"].append(round(vp, 4))
        history_rows.append({"method": name, "epoch": epoch, "loss": round(mean_loss, 6), "val_psnr": round(vp, 4)})
        line("TRAIN", f"{name}: epoch {epoch}/{TOTAL_EPOCHS} loss={mean_loss:.5f} val_psnr={vp:.2f} dB")
        if vp > best_psnr:
            best_psnr, best_epoch = vp, epoch
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_psnr": vp, "name": name},
                os.path.join(OUT_DIR, f"checkpoint_best_{name}.pt"),
            )
    torch.save(
        {"state_dict": model.state_dict(), "epoch": TOTAL_EPOCHS, "val_psnr": best_psnr, "name": name},
        os.path.join(OUT_DIR, f"checkpoint_last_{name}.pt"),
    )
    ckpt = torch.load(os.path.join(OUT_DIR, f"checkpoint_best_{name}.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    with torch.no_grad():
        z = model(z0_val)
    x_hat = to_c(z)
    eval_s = time.perf_counter() - t1
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
    train_s = time.perf_counter() - t0
    line("TRAIN", f"{name}: done in {train_s:.1f}s (best val PSNR {best_psnr:.2f} dB @ epoch {best_epoch})")
    return {
        "x_hat": x_hat.detach().cpu(),
        "iters": [1] * len(val_idx),
        "time_ms_per_slice": eval_s * 1000.0 / len(val_idx),
        "peak_mb": peak_mb,
        "n_params": n_params,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_psnr": best_psnr,
        "final_eta": None,
    }


def train_unrolled(name, base_seed, train_idx, val_idx, gt, mask, device, history_rows):
    set_seed(base_seed)
    model = build_model(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_params = sum(p.numel() for p in model.parameters())
    line("TRAIN", f"{name}: unrolled K={K_WARMUP}->{K_FULL} progressive params={n_params} epochs={TOTAL_EPOCHS} lr={LR} warmup={WARMUP_EPOCHS} eta=learnable(init 1.0)")

    gt_val, y_val, z0_val = prep_val(gt, val_idx, mask, device)
    best_psnr, best_epoch = -1.0, 0
    history = {"loss": [], "val_psnr": []}
    t0 = time.perf_counter()
    for epoch in range(1, TOTAL_EPOCHS + 1):
        k_eff = K_WARMUP if epoch <= WARMUP_EPOCHS else K_FULL
        lr = lr_at_epoch(epoch)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        perm = make_batch_perm(base_seed, epoch, len(train_idx))
        losses = []
        for s in range(0, len(perm), BATCH_SIZE):
            b = train_idx[perm[s : s + BATCH_SIZE]]
            x = gt[b].to(device)
            yb = sense(x, mask)
            z0 = to_2ch(adjoint(yb))
            z = model(z0, yb, mask, k=k_eff)
            loss = recon_loss(to_c(z), x, yb, mask)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            z = model(z0_val, y_val, mask, k=K_FULL)
            vp = per_slice_psnr(to_c(z), gt_val).mean().item()
        history["loss"].append(mean_loss)
        history["val_psnr"].append(round(vp, 4))
        history_rows.append({"method": name, "epoch": epoch, "loss": round(mean_loss, 6), "val_psnr": round(vp, 4)})
        line("TRAIN", f"{name}: epoch {epoch}/{TOTAL_EPOCHS} (K={k_eff}) loss={mean_loss:.5f} val_psnr={vp:.2f} dB")
        if epoch > WARMUP_EPOCHS and vp > best_psnr:
            best_psnr, best_epoch = vp, epoch
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_psnr": vp, "name": name},
                os.path.join(OUT_DIR, f"checkpoint_best_{name}.pt"),
            )
    torch.save(
        {"state_dict": model.state_dict(), "epoch": TOTAL_EPOCHS, "val_psnr": best_psnr, "name": name},
        os.path.join(OUT_DIR, f"checkpoint_last_{name}.pt"),
    )
    ckpt = torch.load(os.path.join(OUT_DIR, f"checkpoint_best_{name}.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    final_eta = float(model.cell.eta.detach().cpu().item())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    with torch.no_grad():
        z = model(z0_val, y_val, mask, k=K_FULL)
    x_hat = to_c(z)
    eval_s = time.perf_counter() - t1
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
    train_s = time.perf_counter() - t0
    line("TRAIN", f"{name}: done in {train_s:.1f}s (best val PSNR {best_psnr:.2f} dB @ epoch {best_epoch}, eta={final_eta:.4f})")
    return {
        "x_hat": x_hat.detach().cpu(),
        "iters": [K_FULL] * len(val_idx),
        "time_ms_per_slice": eval_s * 1000.0 / len(val_idx),
        "peak_mb": peak_mb,
        "n_params": n_params,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_psnr": best_psnr,
        "final_eta": final_eta,
    }
# ---- metrics / reporting ---------------------------------------------------
def compute_metrics(x_hat_c, gt_c, y, mask, iters, time_ms_per_slice, peak_mb, n_params, note=""):
    x_hat_c = x_hat_c.to(gt_c.device)
    psnr = per_slice_psnr(x_hat_c, gt_c).cpu().numpy()
    nmse = per_slice_nmse(x_hat_c, gt_c).cpu().numpy()
    ssim = per_slice_ssim(x_hat_c, gt_c)
    phase = per_slice_phase_err_deg(x_hat_c, gt_c).cpu().numpy()
    resid = per_slice_dc_residual(x_hat_c, y, mask).cpu().numpy()
    conv_rate = float((resid < DC_RESID_TOL).mean())
    return {
        "note": note,
        "n_test": int(len(psnr)),
        "n_params": int(n_params),
        "psnr_mean": round(float(psnr.mean()), 4),
        "psnr_std": round(float(psnr.std()), 4),
        "ssim_mean": round(float(ssim.mean()), 5),
        "ssim_std": round(float(ssim.std()), 5),
        "nmse_mean": round(float(nmse.mean()), 6),
        "phase_error_mean_deg": round(float(phase.mean()), 4),
        "convergence_rate": round(conv_rate, 4),
        "mean_iterations": round(float(np.mean(iters)), 3),
        "median_iterations": round(float(np.median(iters)), 1),
        "mean_final_residual": round(float(resid.mean()), 5),
        "inference_time_ms_per_slice": round(float(time_ms_per_slice), 3),
        "peak_memory_mb": round(float(peak_mb), 1),
        "all_finite": bool(
            np.isfinite(psnr).all() and np.isfinite(nmse).all()
            and np.isfinite(ssim).all() and np.isfinite(phase).all()
        ),
    }


def report_method(name, m):
    report["methods"][name] = m
    line("METRIC", (
        f"{name}: psnr={m['psnr_mean']:.2f}+-{m['psnr_std']:.2f} ssim={m['ssim_mean']:.4f} "
        f"nmse={m['nmse_mean']:.5f} phase={m['phase_error_mean_deg']:.2f}deg "
        f"conv_rate={m['convergence_rate']:.2f} iters={m['mean_iterations']:.1f} "
        f"resid={m['mean_final_residual']:.4f} time={m['inference_time_ms_per_slice']:.2f}ms "
        f"mem={m['peak_memory_mb']:.1f}MB params={m['n_params']}"
    ))
    return m


def aggregate_modl(per_seed):
    seeds = list(per_seed.keys())
    agg = {}
    for k in ["psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "nmse_mean",
              "phase_error_mean_deg", "convergence_rate", "mean_iterations",
              "median_iterations", "mean_final_residual"]:
        vals = [per_seed[s][k] for s in seeds]
        nd = 5 if k in ("ssim_mean", "ssim_std", "convergence_rate") else 4
        agg[k] = round(float(np.mean(vals)), nd)
    agg["n_test"] = per_seed[seeds[0]]["n_test"]
    agg["n_params"] = per_seed[seeds[0]]["n_params"]
    agg["inference_time_ms_per_slice"] = round(float(np.mean([per_seed[s]["inference_time_ms_per_slice"] for s in seeds])), 3)
    agg["peak_memory_mb"] = round(float(np.max([per_seed[s]["peak_memory_mb"] for s in seeds])), 1)
    agg["all_finite"] = all(per_seed[s]["all_finite"] for s in seeds)
    agg["final_eta"] = round(float(np.mean([per_seed[s]["final_eta"] for s in seeds])), 4)
    agg["note"] = "unrolled K=%d, learnable eta, %d seeds (mean +- std)" % (K_FULL, len(seeds))
    agg["seed_psnr_mean"] = {str(s): round(per_seed[s]["psnr_mean"], 3) for s in seeds}
    agg["seed_ssim_mean"] = {str(s): round(per_seed[s]["ssim_mean"], 4) for s in seeds}
    return agg


def stability_from_history(history):
    vals = history["val_psnr"]
    if not vals:
        return None
    best_epoch = int(np.argmax(vals)) + 1
    last10 = vals[-10:]
    return {
        "best_epoch_full_series": best_epoch,
        "best_val_psnr_full_series": round(float(np.max(vals)), 4),
        "last10_mean": round(float(np.mean(last10)), 4),
        "last10_std": round(float(np.std(last10)), 4),
    }


def fmt_eta(eta):
    if eta is None:
        return "-"
    try:
        return f"{float(eta):.3f}"
    except (TypeError, ValueError):
        return str(eta)


# ---- figures ---------------------------------------------------------------
def save_figures(results, gt_val_c, mask, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["GT", "zero_filled", "tv_lite", "unet_lite", "modl_lite", "euclid_deq_lite"]
    n_slices = 3
    fig, axes = plt.subplots(n_slices, len(order), figsize=(2.2 * len(order), 2.2 * n_slices))
    for r in range(n_slices):
        gt_mag = gt_val_c[r].abs().cpu().numpy()
        axes[r][0].imshow(gt_mag, cmap="gray", vmin=0, vmax=1)
        axes[r][0].set_title("GT", fontsize=9)
        for c, name in enumerate(order[1:], start=1):
            x_hat = results[name][r].to(device)
            rec_mag = x_hat.abs().cpu().numpy()
            axes[r][c].imshow(rec_mag, cmap="gray", vmin=0, vmax=1)
            axes[r][c].set_title(name, fontsize=8)
        for c in range(len(order)):
            axes[r][c].axis("off")
    fig.suptitle("Step 3.3 v2: magnitude recon (R=4, mask r4_s42, val slices)", fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(FIG_DIR, "fig1_recon_fix.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)

    fig2, axes2 = plt.subplots(n_slices, len(order) - 1, figsize=(2.2 * (len(order) - 1), 2.2 * n_slices))
    for r in range(n_slices):
        gt_mag = gt_val_c[r].abs().cpu().numpy()
        for c, name in enumerate(order[1:], start=0):
            x_hat = results[name][r].to(device)
            err = x_hat.abs().cpu().numpy() - gt_mag
            axes2[r][c].imshow(err, cmap="RdBu_r", vmin=-0.25, vmax=0.25)
            axes2[r][c].set_title(name, fontsize=8)
            axes2[r][c].axis("off")
    fig2.suptitle("Step 3.3 v2: error maps (hat - GT, magnitude)", fontsize=10)
    fig2.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig2_error_fix.png")
    fig2.savefig(p2, dpi=110)
    plt.close(fig2)
    for p in (p1, p2):
        shutil.copy(p, os.path.join(OUT_DIR, "examples", os.path.basename(p)))
    return [p1, p2]
class Tee:
    def __init__(self, f):
        self.f = f

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def main():
    t_start = time.perf_counter()
    set_seed(SEED)
    log_path = os.path.join(OUT_DIR, "stdout.log")
    with open(log_path, "w", encoding="utf-8") as logf:
        sys.stdout = Tee(logf)
        try:
            line("MAIN", f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            line("MAIN", f"seed={SEED} rate={RATE} mask={MASK_KEY} recipe=warmup{WARMUP_EPOCHS}+cosine lr={LR} K={K_WARMUP}->{K_FULL} eta=learnable")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            line("ENV", f"device={device} torch={torch.__version__}")

            data = torch.load(PREPARED, map_location="cpu", weights_only=False)
            gt = data["gt_complex"]
            split = data["split"]
            mask = data["masks"][MASK_KEY].to(device)
            train_idx = torch.tensor(split["train"], dtype=torch.long)
            val_idx = torch.tensor(split["val"], dtype=torch.long)
            line("DATA", f"gt={tuple(gt.shape)} train={len(train_idx)} val={len(val_idx)} test={len(split['test'])}")

            gt_val = gt[val_idx].to(device)
            history_rows = []
            results = {}

            # 1) zero-filled ---------------------------------------------------
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            zf_c, y_val = run_zero_filled(gt_val, mask)
            eval_s = time.perf_counter() - t0
            peak = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
            m = compute_metrics(zf_c, gt_val, y_val, mask, [1] * len(val_idx), eval_s * 1000.0 / len(val_idx), peak, 0, note="no training, reference floor")
            report_method("zero_filled", m)
            results["zero_filled"] = zf_c.detach().cpu()

            # 2) TV / CS-lite ---------------------------------------------------
            chosen_lam = run_tv_sweep(gt_val, mask)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            tv_c, _ = run_tv_lite(gt_val, mask, iters=TV_ITERS, lr=TV_LR, lam=chosen_lam)
            eval_s = time.perf_counter() - t0
            peak = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
            m = compute_metrics(tv_c, gt_val, y_val, mask, [TV_ITERS] * len(val_idx), eval_s * 1000.0 / len(val_idx), peak, 0, note=f"GD x{TV_ITERS}, lam={chosen_lam}")
            report_method("tv_lite", m)
            results["tv_lite"] = tv_c.detach().cpu()

            # 3) unet_lite (direct) ----------------------------------------------
            out = train_direct("unet_lite", METHOD_SEED, train_idx, val_idx, gt, mask, device, history_rows)
            m = compute_metrics(out["x_hat"], gt_val, y_val, mask, out["iters"], out["time_ms_per_slice"], out["peak_mb"], out["n_params"], note="direct U-Net (recipe fix)")
            m["final_eta"] = None
            m["stability"] = stability_from_history(out["history"])
            report_method("unet_lite", m)
            results["unet_lite"] = out["x_hat"]
            del out
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # 4) modl_lite (3-seed sweep) -----------------------------------------
            modl_seed_eval = {}
            modl_seed_sweep = {}
            modl_stab_history = None
            for seed in MODL_SEEDS:
                line("MAIN", f"modl_lite seed={seed} training ...")
                out = train_unrolled("modl_lite", seed, train_idx, val_idx, gt, mask, device, history_rows)
                m_seed = compute_metrics(out["x_hat"], gt_val, y_val, mask, out["iters"], out["time_ms_per_slice"], out["peak_mb"], out["n_params"], note=f"seed={seed}")
                m_seed["final_eta"] = out["final_eta"]
                m_seed["best_epoch"] = out["best_epoch"]
                m_seed["best_val_psnr"] = round(float(out["best_val_psnr"]), 4)
                modl_seed_eval[seed] = m_seed
                modl_seed_sweep[str(seed)] = {
                    "best_epoch": out["best_epoch"],
                    "best_val_psnr": round(float(out["best_val_psnr"]), 4),
                    "final_eta": round(float(out["final_eta"]), 4),
                    "psnr_mean": m_seed["psnr_mean"],
                    "ssim_mean": m_seed["ssim_mean"],
                }
                if seed == MODL_SEEDS[0]:
                    modl_stab_history = out["history"]
                    results["modl_lite"] = out["x_hat"]
                del out
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            report["modl_seed_sweep"] = modl_seed_sweep
            agg = aggregate_modl(modl_seed_eval)
            agg["stability"] = stability_from_history(modl_stab_history)
            report_method("modl_lite", agg)
            line("SEEDS", "modl seed psnr: " + ", ".join(f"{s}: {modl_seed_eval[s]['psnr_mean']:.2f}" for s in MODL_SEEDS))

            # 5) euclid_deq_lite --------------------------------------------------
            out = train_unrolled("euclid_deq_lite", METHOD_SEED, train_idx, val_idx, gt, mask, device, history_rows)
            m = compute_metrics(out["x_hat"], gt_val, y_val, mask, out["iters"], out["time_ms_per_slice"], out["peak_mb"], out["n_params"], note="unrolled K=%d, learnable eta" % K_FULL)
            m["final_eta"] = out["final_eta"]
            m["stability"] = stability_from_history(out["history"])
            report_method("euclid_deq_lite", m)
            results["euclid_deq_lite"] = out["x_hat"]
            del out
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # 6) stability warnings ------------------------------------------------
            for name in ["unet_lite", "modl_lite", "euclid_deq_lite"]:
                st = report["methods"][name].get("stability")
                if not st:
                    continue
                line("STAB", f"{name}: best@epoch {st['best_epoch_full_series']} last10_mean={st['last10_mean']:.2f} last10_std={st['last10_std']:.2f}")
                if st["best_val_psnr_full_series"] - st["last10_mean"] > 0.25:
                    add_issue("warning", f"{name}: best is an early spike (epoch {st['best_epoch_full_series']}/{TOTAL_EPOCHS})")

            # 7) verdict -------------------------------------------------------------
            zf_psnr = report["methods"]["zero_filled"]["psnr_mean"]
            zf_ssim = report["methods"]["zero_filled"]["ssim_mean"]
            if not report["methods"]["zero_filled"]["all_finite"]:
                add_issue("error", "zero_filled metrics non-finite")
            for name in ["unet_lite", "modl_lite", "euclid_deq_lite"]:
                m = report["methods"].get(name)
                if m is None:
                    add_issue("error", f"{name}: missing metrics (training failed?)")
                    continue
                if not m["all_finite"]:
                    add_issue("error", f"{name}: non-finite metrics")
                elif m["psnr_mean"] < zf_psnr + VAL_PSNR_MARGIN:
                    add_issue("error", f"{name}: val PSNR {m['psnr_mean']:.2f} below zero-filled+{VAL_PSNR_MARGIN:.1f} dB (need {zf_psnr + VAL_PSNR_MARGIN:.2f})")
                else:
                    line("CHECK", f"{name}: beats zero-filled by {m['psnr_mean'] - zf_psnr:.2f} dB (OK)")
            tv = report["methods"]["tv_lite"]
            if not tv["all_finite"]:
                add_issue("error", "tv_lite: non-finite metrics")
            elif tv["ssim_mean"] < zf_ssim + 0.02:
                add_issue("warning", f"tv_lite: SSIM {tv['ssim_mean']:.4f} did not clearly beat zero-filled {zf_ssim:.4f}")
            else:
                line("CHECK", f"tv_lite: SSIM {tv['ssim_mean']:.4f} > zero-filled {zf_ssim:.4f} (OK)")

            # 8) figures --------------------------------------------------------------
            figs = save_figures(results, gt_val, mask, device)
            line("FIG", "; ".join(figs))

            elapsed = time.perf_counter() - t_start
            if elapsed > VERDICT_TIME_BUDGET_S:
                add_issue("warning", f"total runtime {elapsed:.1f}s exceeds budget {VERDICT_TIME_BUDGET_S:.0f}s")
            report["elapsed_sec"] = round(elapsed, 2)
            report["verdict"] = "OK" if not any(i["level"] == "error" for i in issues) else "FAIL"
            report["issues"] = issues
            report["counters"] = {
                "errors": sum(1 for i in issues if i["level"] == "error"),
                "warnings": sum(1 for i in issues if i["level"] == "warning"),
            }

            # 9) JSON first (machine-readable, never lost) -----------------------------
            with open(REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"methods": report["methods"], "config": report["config"]}, f, ensure_ascii=False, indent=2)
            with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
                json.dump(report["config"], f, ensure_ascii=False, indent=2)
            with open(os.path.join(OUT_DIR, "history.csv"), "w", encoding="utf-8") as f:
                f.write("method,epoch,loss,val_psnr\n")
                for row in history_rows:
                    f.write(f"{row['method']},{row['epoch']},{row['loss']},{row['val_psnr']}\n")
            with open(os.path.join(OUT_DIR, "modl_seed_sweep.json"), "w", encoding="utf-8") as f:
                json.dump({"seed_sweep": modl_seed_sweep, "aggregated": agg}, f, ensure_ascii=False, indent=2)

            # 10) text summary (human-readable; eta bug fixed) --------------------------
            with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                f.write("STEP 3.3 UNROLLED RECIPE-FIX SUMMARY (v2)\n")
                f.write(f"timestamp: {report['timestamp']}\n")
                f.write(f"elapsed_sec: {report.get('elapsed_sec')}\n")
                f.write(f"verdict: {report['verdict']}\n\n")
                f.write("recipe:\n")
                f.write(f"  lr={LR} warmup_epochs={WARMUP_EPOCHS} total_epochs={TOTAL_EPOCHS} cosine=True\n")
                f.write(f"  progressive_K={K_WARMUP}->{K_FULL} learnable_eta=True\n")
                f.write(f"  best_ckpt = best val PSNR among full-K epochs (epoch > {WARMUP_EPOCHS})\n\n")
                f.write(f"{'method':15s} {'psnr':>8s} {'ssim':>8s} {'nmse':>9s} {'resid':>8s} {'conv':>6s} {'t_ms':>8s} {'mem':>7s} {'best_ep':>7s} {'eta':>7s}\n")
                for name, m in report["methods"].items():
                    st = m.get("stability") or {}
                    bep = st.get("best_epoch_full_series", "-")
                    eta = fmt_eta(m.get("final_eta"))
                    f.write(f"{name:15s} {m['psnr_mean']:8.2f} {m['ssim_mean']:8.4f} {m['nmse_mean']:9.4f} {m['mean_final_residual']:8.4f} {m['convergence_rate']:6.2f} {m['inference_time_ms_per_slice']:8.2f} {m['peak_memory_mb']:7.1f} {str(bep):>7s} {eta:>7s}\n")
                f.write("\nmodl seed sweep (val-set psnr per seed):\n")
                for s in MODL_SEEDS:
                    sd = modl_seed_sweep[str(s)]
                    f.write(f"  seed {s}: best_val_psnr={sd['best_val_psnr']} @epoch {sd['best_epoch']} eta={sd['final_eta']} psnr={sd['psnr_mean']} ssim={sd['ssim_mean']}\n")
                f.write("\nTV sweep (5 val slices, 80 iters):\n")
                for row in report.get("tv_sweep", {}).get("grid", []):
                    f.write(f"  lam={row['lam']}: psnr={row['psnr']} ssim={row['ssim']}\n")
                f.write(f"chosen lam: {report.get('tv_sweep', {}).get('chosen_lam')}\n\n")
                f.write("issues:\n")
                for i in issues:
                    f.write(f"  [{i['level']}] {i['msg']}\n")
            line("MAIN", f"done in {elapsed:.1f}s | verdict: {report['verdict']} (errors={report['counters']['errors']}, warnings={report['counters']['warnings']})")
            line("MAIN", f"report: {REPORT_PATH}")
            line("MAIN", f"summary: {SUMMARY_PATH}")
            line("MAIN", f"outputs: {OUT_DIR}")
        finally:
            sys.stdout = sys.__stdout__
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

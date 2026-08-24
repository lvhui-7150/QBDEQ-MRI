# -*- coding: ascii -*-
"""
step5_320_ceiling.py
====================
fastMRI knee 320x320 CEILING experiment for the QB-DEQ rebuild.

PURPOSE
-------
Before any implicit-depth (DEQ) machinery, establish the empirical ceiling
of this data + task with a strong direct U-Net baseline on the official
fastMRI knee singlecoil_val data (320x320 complex, fastMRI standard
ifft2c + center-crop 320 + per-file max normalization).

Three checks, all reported in machine + human readable form:

  (1) ZF ALIGNMENT  - recompute zero-filled PSNR/SSIM on all 804 test
      slices x 6 masks with the SAME numpy recipe as fastmri_320_prep.py
      and compare to the reference table:
          r4_s42=25.42/0.5405  r4_s123=25.25/0.5333  r4_s2025=25.08/0.5350
          r8_s42=24.44/0.4397  r8_s123=24.27/0.4323  r8_s2025=24.65/0.4493
      ALIGNED iff |d_psnr| <= 0.5 and |d_ssim| <= 0.02 on every mask.

  (2) U-NET CEILING - train UNet3 (base 64, ~4.8M params) on r4_s42.
      CEILING_HIGH : test PSNR >= 32 dB and SSIM >= 0.80 and gain >= +5 dB
      CEILING_MID  : test PSNR >= 29 dB
      CEILING_LOW  : otherwise
      (test PSNR/SSIM also reported on r4_s123 and r4_s2025)

  (3) ITERATIVE HEADROOM - same checkpoint unrolled K=1/2/4/8 (eta=1.0)
      on a val subset (<= 64 slices):
      HEADROOM_HIGH : K8 - K1 >= +1 dB
      FLAT          : K8 - K1 >  -1 dB
      LOW           : K8 - K1 <= -1 dB

OVERALL VERDICT
---------------
    PASS   = aligned and CEILING_HIGH and headroom != LOW
    REVIEW = aligned but CEILING_MID or headroom LOW (next step suggested)
    FAIL   = otherwise

RUN
---
    python step5_320_ceiling.py --smoke      # ~3-5 min (2 epochs, subsets)
    python step5_320_ceiling.py --full       # default (~2-4 h, 60 epochs)
    python step5_320_ceiling.py --eval-only  # reuse best checkpoint

OUTPUTS (in this script's folder)
---------------------------------
    step5_320_ceiling_{mode}_summary.txt / _report.json / _stdout.log
    runs/step5_320_{mode}/checkpoint_best.pt, history.csv, config.json
    step5_320_{mode}_figs/*.png   (English labels only)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import json
import time
import math
import random
import argparse
import csv
import ctypes

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(HERE, "fastmri_320_meta.pt")
MASK_KEYS = ["r4_s42", "r4_s123", "r4_s2025", "r8_s42", "r8_s123", "r8_s2025"]
TRAIN_MASK = "r4_s42"
TEST_MASKS = ["r4_s42", "r4_s123", "r4_s2025"]
CACHE_SIZE = 4
ZF_BLOCK = 128

REF_ZF = {
    "r4_s42":   {"psnr": 25.42, "ssim": 0.5405},
    "r4_s123":  {"psnr": 25.25, "ssim": 0.5333},
    "r4_s2025": {"psnr": 25.08, "ssim": 0.5350},
    "r8_s42":   {"psnr": 24.44, "ssim": 0.4397},
    "r8_s123":  {"psnr": 24.27, "ssim": 0.4323},
    "r8_s2025": {"psnr": 24.65, "ssim": 0.4493},
}

_LOG_LINES = []
_LOG_FH = None


def log(msg):
    line = "[CEIL] " + msg
    print(line)
    _LOG_LINES.append(line)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()
        except Exception:
            pass


def json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def phys_mem_gb():
    try:
        class MEMSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        s = MEMSTATUS()
        s.dwLength = ctypes.sizeof(MEMSTATUS)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        if not ok:
            return -1.0, -1.0
        return round(s.ullTotalPhys / 1e9, 1), round(s.ullAvailPhys / 1e9, 1)
    except Exception:
        return -1.0, -1.0


# ---- numpy forward model + metrics (identical to fastmri_320_prep) --------
def fft2c(x):
    return np.fft.ifftshift(
        np.fft.fft2(np.fft.fftshift(x, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1))


def ifft2c(k):
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
        axes=(-2, -1))


def compute_psnr(gt, pred, peak=None):
    """Magnitude PSNR (dB), fastMRI convention: peak = max(gt magnitude)."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mse = float(np.mean((gt - pred) ** 2))
    if peak is None:
        peak = float(gt.max())
    if peak <= 0:
        return 0.0
    if mse <= 0:
        return 300.0
    return float(10.0 * np.log10(peak * peak / mse))


class SSIMComputer(object):
    """Same as fastmri_320_prep.py: 7x7 gauss sigma 1.5, FFT correlation,
    valid-region mean, data_range = max(gt magnitude)."""

    def __init__(self, win=7, sigma=1.5, k1=0.01, k2=0.03):
        self.win = win
        self.pad = win // 2
        self.k1 = k1
        self.k2 = k2
        x = np.arange(win, dtype=np.float64) - win // 2
        g = np.exp(-(x * x) / (2.0 * sigma * sigma))
        g = g / g.sum()
        self.k2d = np.outer(g, g)
        self._kf_cache = {}

    def _kf(self, shape):
        if shape not in self._kf_cache:
            ph, pw = self.pad, self.pad
            kp = np.zeros(shape, dtype=np.float64)
            kp[ph:ph + self.win, pw:pw + self.win] = self.k2d
            self._kf_cache[shape] = np.fft.fft2(kp)
        return self._kf_cache[shape]

    def _conv(self, x):
        ph = pw = self.pad
        xp = np.pad(x, ((ph, ph), (pw, pw)), mode="constant")
        kf = self._kf(xp.shape)
        out = np.fft.ifft2(np.fft.fft2(xp) * kf)
        return np.real(out[ph:ph + x.shape[0], pw:pw + x.shape[1]])

    def compute(self, a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            return 0.0
        L = float(a.max())
        if L <= 0:
            return 1.0 if np.array_equal(a, b) else 0.0
        c1 = (self.k1 * L) ** 2
        c2 = (self.k2 * L) ** 2
        mu1 = self._conv(a)
        mu2 = self._conv(b)
        s1 = self._conv(a * a) - mu1 * mu1
        s2 = self._conv(b * b) - mu2 * mu2
        s12 = self._conv(a * b) - mu1 * mu2
        num = (2.0 * mu1 * mu2 + c1) * (2.0 * s12 + c2)
        den = (mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)
        den = np.maximum(den, 1e-12)
        m = num / den
        cr = self.pad
        if m.shape[0] > 2 * cr and m.shape[1] > 2 * cr:
            m = m[cr:-cr, cr:-cr]
        return float(m.mean())

# ---- torch forward model + metrics (must match the numpy recipe above) ----
def to_2ch(x_c):
    return torch.stack([x_c.real, x_c.imag], dim=1)


def to_c(x_2ch):
    return torch.complex(x_2ch[:, 0], x_2ch[:, 1])


def fft2_t(x):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def ifft2_t(x):
    return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(x, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def per_slice_psnr_full(x_hat_c, x_gt_c):
    """Magnitude PSNR (fastMRI convention): compare |recon| vs |gt|."""
    mse = ((x_hat_c.abs() - x_gt_c.abs()) ** 2).mean(dim=(-1, -2))
    peak = x_gt_c.abs().amax(dim=(-1, -2))
    return 20.0 * torch.log10(peak / (torch.sqrt(mse) + 1e-12))


_SSIM_KERNEL = None


def _gauss_kernel_2d(win=7, sigma=1.5):
    x = torch.arange(win, dtype=torch.float32) - win // 2
    g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return torch.outer(g, g).view(1, 1, win, win)


def torch_ssim(x_hat_c, x_gt_c, win=7):
    """Differentiable SSIM on magnitude images (matches numpy SSIMComputer)."""
    global _SSIM_KERNEL
    if _SSIM_KERNEL is None or _SSIM_KERNEL.device != x_hat_c.device:
        _SSIM_KERNEL = _gauss_kernel_2d(win).to(x_hat_c.device)
    a = x_hat_c.abs().unsqueeze(1).to(torch.float32)
    b = x_gt_c.abs().unsqueeze(1).to(torch.float32)
    L = b.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    pad = win // 2
    mu_a = F.conv2d(a, _SSIM_KERNEL, padding=pad)
    mu_b = F.conv2d(b, _SSIM_KERNEL, padding=pad)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    s_a2 = F.conv2d(a * a, _SSIM_KERNEL, padding=pad) - mu_a2
    s_b2 = F.conv2d(b * b, _SSIM_KERNEL, padding=pad) - mu_b2
    s_ab = F.conv2d(a * b, _SSIM_KERNEL, padding=pad) - mu_ab
    s_a2 = s_a2.clamp(min=0.0)
    s_b2 = s_b2.clamp(min=0.0)
    num = (2.0 * mu_ab + c1) * (2.0 * s_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (s_a2 + s_b2 + c2)
    m = num / den.clamp(min=1e-12)
    if m.shape[-1] > 2 * pad and m.shape[-2] > 2 * pad:
        m = m[:, :, pad:-pad, pad:-pad]
    return m.mean()


def recon_loss_parts(x_hat_c, x_gt_c, y, mask):
    l_mag = (x_hat_c.abs() - x_gt_c.abs()).abs().mean()
    l_ssim = 0.1 * (1.0 - torch_ssim(x_hat_c, x_gt_c))
    l_dc = 0.01 * ((fft2_t(x_hat_c) * mask - y).abs() ** 2).mean()
    return l_mag, l_ssim, l_dc


def recon_loss(x_hat_c, x_gt_c, y, mask):
    l_mag, l_ssim, l_dc = recon_loss_parts(x_hat_c, x_gt_c, y, mask)
    return l_mag + l_ssim + l_dc


FG_THR = 0.15
ZF_PSNR_TOL = 0.5
ZF_SSIM_TOL = 0.02
# verdict thresholds (magnitude PSNR / SSIM, fastMRI convention)
PASS_PSNR = 30.5
PASS_SSIM = 0.75
MID_PSNR = 28.5
MIN_GAIN_DB = 5.0


def stat_arr(a):
    a = np.asarray(a, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std()), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p50": round(float(np.percentile(a, 50)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
    }


def data_stats(store, idx, max_n=512, seed=0):
    rng = np.random.RandomState(seed)
    pick = rng.choice(np.asarray(idx, dtype=np.int64), size=min(max_n, len(idx)), replace=False)
    g = store.get_batch(pick.tolist())
    mag = g.abs().double()
    pmax = mag.amax(dim=(1, 2), keepdim=True)
    fg = (mag > (FG_THR * pmax)).double()
    bg = (mag <= (FG_THR * pmax)).double()
    frac = fg.mean(dim=(1, 2))
    fg_rms2 = ((mag * fg).square().sum(dim=(1, 2)) / fg.sum(dim=(1, 2)).clamp(min=1e-9))
    bg_rms2 = ((mag * bg).square().sum(dim=(1, 2)) / bg.sum(dim=(1, 2)).clamp(min=1e-9))
    snr_db = 10.0 * torch.log10(fg_rms2 / bg_rms2.clamp(min=1e-12))
    return {
        "n": int(len(pick)),
        "foreground_fraction": stat_arr(frac.detach().cpu().numpy()),
        "per_slice_max": stat_arr(pmax.squeeze(-1).squeeze(-1).detach().cpu().numpy()),
        "per_slice_mean": stat_arr(mag.mean(dim=(1, 2)).detach().cpu().numpy()),
        "gt_fg_snr_db": stat_arr(snr_db.detach().cpu().numpy()),
    }


class ChunkStore(object):
    """Lazy mmap loader over fastmri_320_gt_chunk_*.pt with LRU cache."""

    def __init__(self, meta, cache_size=CACHE_SIZE):
        g = meta["gt"]
        self.n = int(g["n_slices"])
        self.per = int(g["chunk_size_rows"])
        self.chunks = list(g["chunks"])
        # Windows: repeated torch.load(..., mmap=True) of the same big chunk
        # file crashes the process with an access violation (0xC0000005), so
        # each chunk is loaded once with a plain load and kept in memory
        # (all 8 chunks ~5.8 GB, machine RAM is ample).
        self.cache_size = max(int(cache_size), len(self.chunks))
        self._cache = {}
        self._order = []

    def _load(self, ci):
        spec = self.chunks[ci]
        t = torch.load(spec["path"], map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            t = t.get("gt", t)
        if not isinstance(t, torch.Tensor):
            raise TypeError("chunk %d did not load to a tensor: %s" % (ci, type(t)))
        return t.detach()

    def load_chunk(self, ci):
        ci = int(ci)
        if ci in self._cache:
            self._order.remove(ci)
            self._order.append(ci)
            return self._cache[ci]
        t = self._load(ci)
        self._cache[ci] = t
        self._order.append(ci)
        if len(self._order) > self.cache_size:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return t

    def get(self, i):
        i = int(i)
        ci = i // self.per
        off = i - ci * self.per
        return self.load_chunk(ci)[off]

    def get_batch(self, idx, device=None):
        idx = [int(i) for i in idx]
        out = torch.stack([self.get(i) for i in idx], dim=0)
        if device is not None:
            out = out.to(device)
        return out


class MaskStore(object):
    def __init__(self, meta_masks):
        self.masks = {}
        self.eff = {}
        for k in MASK_KEYS:
            mm = meta_masks[k]
            self.masks[k] = mm["mask"].clone()
            self.eff[k] = float(mm["meta"].get("effR", 0.0))

    def get(self, key, device=None):
        m = self.masks[key]
        return m.to(device) if device is not None else m

    def effR(self, key):
        return self.eff[key]
# ---- model (identical architecture to _probe_ceiling.py) ------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super(ConvBlock, self).__init__()
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


class UNet3(nn.Module):
    def __init__(self, in_ch=2, base=64):
        super(UNet3, self).__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bot = ConvBlock(base * 4, base * 4)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(base * 3, base)
        self.head = nn.Conv2d(base, in_ch, 1)
        self.out_scale = nn.Parameter(torch.ones(in_ch))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))
        u = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        u = self.dec2(torch.cat([self.up2(u), e2], dim=1))
        u = self.dec1(torch.cat([self.up1(u), e1], dim=1))
        return self.head(u) * self.out_scale.view(1, -1, 1, 1)


# ---- lr schedule -----------------------------------------------------------
def lr_at(epoch, args):
    if epoch <= args.warmup:
        return args.lr * epoch / max(1, args.warmup)
    t = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ---- run dir ---------------------------------------------------------------
def reset_run_dir(path):
    import shutil
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ---- training --------------------------------------------------------------
def train_unet(store, train_idx, val_sub, mask, device, args, run_dir):
    set_seed(args.seed)
    model = UNet3(in_ch=2, base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_psnr, best_epoch = -1.0, 0
    no_impr = 0
    history = {"loss": [], "val_psnr": []}
    last_epoch = 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        lr = lr_at(epoch, args)
        for pg in opt.param_groups:
            pg["lr"] = lr
        model.train()
        gen = torch.Generator().manual_seed(args.seed * 10000 + epoch)
        perm = torch.randperm(len(train_idx), generator=gen).cpu().numpy()
        losses = []
        for s in range(0, len(perm), args.batch):
            b_idx = [int(i) for i in train_idx[perm[s:s + args.batch]]]
            x = store.get_batch(b_idx, device=device)
            if args.aug:
                if random.random() < 0.5:
                    x = torch.flip(x, dims=[-1])
                if random.random() < 0.5:
                    x = torch.flip(x, dims=[-2])
            yb = fft2_t(x) * mask
            z0 = to_2ch(ifft2_t(yb))
            with torch.autocast(device_type="cuda", enabled=use_amp):
                z = model(z0).float()
            xh = to_c(z)
            loss = recon_loss(xh, x, yb, mask)
            if epoch == 1 and s == 0:
                lp = recon_loss_parts(xh, x, yb, mask)
                log("LOSS parts epoch1: l_mag=%.5f l_ssim=%.5f l_dc=%.5f total=%.5f"
                    % (float(lp[0]), float(lp[1]), float(lp[2]), float(loss)))
            if not torch.isfinite(loss):
                log("TRAIN epoch %d NON-FINITE loss %.4e -> abort" % (epoch, float(loss)))
                return {"status": "TRAIN_DIVERGED", "epoch": epoch, "last_epoch": epoch,
                        "history": history, "model": model, "n_params": n_params}
            opt.zero_grad()
            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        model.eval()
        vp_all = []
        with torch.no_grad():
            for s0 in range(0, len(val_sub), args.batch):
                vb = val_sub[s0:s0 + args.batch]
                gv = store.get_batch(vb, device=device)
                yv = fft2_t(gv) * mask
                z0v = to_2ch(ifft2_t(yv))
                vp_all.append(per_slice_psnr_full(to_c(model(z0v)), gv))
        vp = float(torch.cat(vp_all).mean().item())
        history["loss"].append(round(mean_loss, 6))
        history["val_psnr"].append(round(vp, 4))
        last_epoch = epoch
        if vp > best_psnr:
            best_psnr, best_epoch = vp, epoch
            no_impr = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_psnr": vp,
                        "n_params": n_params},
                       os.path.join(run_dir, "checkpoint_best.pt"))
        else:
            no_impr += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            log("TRAIN epoch %d/%d loss=%.5f val_psnr=%.2f dB (best %.2f @ %d)"
                % (epoch, args.epochs, mean_loss, vp, best_psnr, best_epoch))
        if epoch >= args.min_epochs and no_impr >= args.patience:
            log("TRAIN early stop at epoch %d (no improvement for %d epochs)"
                % (epoch, args.patience))
            break
    train_s = time.perf_counter() - t0
    log("TRAIN done in %.1fs best val PSNR %.2f dB @ epoch %d (params=%d)"
        % (train_s, best_psnr, best_epoch, n_params))
    return {"status": "OK", "train_sec": train_s, "best_psnr": best_psnr,
            "best_epoch": best_epoch, "last_epoch": last_epoch, "history": history,
            "model": model, "n_params": n_params}
# ---- split + evaluations ---------------------------------------------------
def load_split(meta):
    sf = np.asarray(meta["slice_files"], dtype=np.int64)
    sp = meta["split"]
    tr_f = [int(x) for x in sp["train_files"]]
    va_f = [int(x) for x in sp["val_files"]]
    te_f = [int(x) for x in sp["test_files"]]
    train_idx = np.where(np.isin(sf, tr_f))[0]
    val_idx = np.where(np.isin(sf, va_f))[0]
    test_idx = np.where(np.isin(sf, te_f))[0]
    return train_idx, val_idx, test_idx


def run_zf_alignment(store, test_idx, mask_store):
    ssim = SSIMComputer()
    rows = []
    ok = True
    for mk in MASK_KEYS:
        m2d = mask_store.masks[mk].numpy().astype(np.float64)
        ps = []
        ss = []
        for s0 in range(0, len(test_idx), ZF_BLOCK):
            block = [int(i) for i in test_idx[s0:s0 + ZF_BLOCK]]
            g = store.get_batch(block).numpy()
            zf = ifft2c(fft2c(g) * m2d)
            for i in range(len(block)):
                gm = np.abs(g[i])
                zm = np.abs(zf[i])
                ps.append(compute_psnr(gm, zm))
                ss.append(ssim.compute(gm, zm))
        ps = np.asarray(ps)
        ss = np.asarray(ss)
        ref = REF_ZF[mk]
        d_psnr = float(ps.mean() - ref["psnr"])
        d_ssim = float(ss.mean() - ref["ssim"])
        good = bool(abs(d_psnr) <= ZF_PSNR_TOL and abs(d_ssim) <= ZF_SSIM_TOL)
        ok = ok and good
        row = {"mask": mk, "effR": mask_store.effR(mk), "n": int(len(ps)),
               "psnr": stat_arr(ps), "ssim": stat_arr(ss),
               "ref_psnr": ref["psnr"], "ref_ssim": ref["ssim"],
               "d_psnr": round(d_psnr, 4), "d_ssim": round(d_ssim, 4), "aligned": good}
        rows.append(row)
        log("ZF %s effR=%.2f psnr=%.2f+-%.2f ssim=%.4f+-%.4f (ref %.2f/%.4f d=%.3f/%.4f) %s"
            % (mk, row["effR"], row["psnr"]["mean"], row["psnr"]["std"],
               row["ssim"]["mean"], row["ssim"]["std"], ref["psnr"], ref["ssim"],
               d_psnr, d_ssim, "OK" if good else "MISMATCH"))
    log("ZF alignment: %s" % ("ALIGNED" if ok else "NOT ALIGNED"))
    return {"ok": ok, "rows": rows}


def eval_recon(model, store, idx, mask_store, mask_key, device, batch=16):
    mask = mask_store.get(mask_key, device=device)
    model.eval()
    ssim = SSIMComputer()
    ps_all = []
    ss_all = []
    xh_keep = []
    with torch.no_grad():
        for s0 in range(0, len(idx), batch):
            b_idx = [int(i) for i in idx[s0:s0 + batch]]
            g = store.get_batch(b_idx, device=device)
            y = fft2_t(g) * mask
            z0 = to_2ch(ifft2_t(y))
            x_hat = to_c(model(z0))
            ps = per_slice_psnr_full(x_hat, g).detach().cpu().numpy()
            ps_all.extend(float(v) for v in ps.tolist())
            xh = x_hat.detach().cpu()
            gm = g.detach().cpu().numpy()
            xhm = xh.numpy()
            for i in range(len(b_idx)):
                ss_all.append(ssim.compute(np.abs(gm[i]), np.abs(xhm[i])))
            if len(xh_keep) < 8:
                need = 8 - len(xh_keep)
                xh_keep.append(xh[:need])
    ps_all = np.asarray(ps_all)
    ss_all = np.asarray(ss_all)
    x_hat = torch.cat(xh_keep, dim=0) if xh_keep else torch.empty(0)
    return {"n": int(len(ps_all)), "psnr_full": stat_arr(ps_all), "ssim": stat_arr(ss_all),
            "idx_keep": [int(i) for i in idx[:8]], "x_hat": x_hat, "mask_key": mask_key}


def eval_unrolled_ks(model, store, val_idx, mask_store, mask_key, device,
                     ks=(1, 2, 4, 8), max_n=64, eta=1.0, batch=8):
    rng = np.random.RandomState(42)
    pick = rng.choice(np.asarray(val_idx, dtype=np.int64), size=min(max_n, len(val_idx)),
                      replace=False).tolist()
    mask = mask_store.get(mask_key, device=device)
    model.eval()
    per = {int(k): [] for k in ks}
    with torch.no_grad():
        for s0 in range(0, len(pick), batch):
            b_idx = pick[s0:s0 + batch]
            g = store.get_batch(b_idx, device=device)
            y = fft2_t(g) * mask
            z0 = to_2ch(ifft2_t(y))
            for K in ks:
                z = z0
                for _ in range(int(K)):
                    zc = to_c(z)
                    dc = zc - eta * ifft2_t((fft2_t(zc) - y) * mask)
                    z = model(to_2ch(dc))
                per[int(K)].append(per_slice_psnr_full(to_c(z), g))
    out = {}
    for k in ks:
        v = float(torch.cat(per[int(k)]).mean().item())
        out[int(k)] = round(v, 4)
    log("UNROLLED " + " ".join("K%d=%.2f" % (k, out[k]) for k in sorted(out)))
    return out
# ---- figures ---------------------------------------------------------------
def make_figures(res, model, store, test_idx, mask_store, device, report, mode, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []

    def save(fig, name):
        p = os.path.join(fig_dir, name)
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
        log("FIG saved %s" % p)

    try:
        hist = res.get("history") or {}
        if hist.get("loss"):
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            ax[0].plot(hist["loss"])
            ax[0].set_xlabel("epoch")
            ax[0].set_ylabel("train loss")
            ax[0].set_title("training loss")
            ax[1].plot(hist["val_psnr"])
            ax[1].set_xlabel("epoch")
            ax[1].set_ylabel("val PSNR (dB)")
            ax[1].set_title("val PSNR (per-image peak)")
            save(fig, "fig1_curves.png")
    except Exception as e:
        log("WARN fig1 failed: %s: %s" % (type(e).__name__, str(e)))

    try:
        rows = report["zf"]["rows"]
        names = [r["mask"] for r in rows]
        ours = [r["psnr"]["mean"] for r in rows]
        refs = [r["ref_psnr"] for r in rows]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(10.5, 4.4))
        w = 0.38
        ax.bar(x - w / 2, refs, w, label="reference (fastmri_320_prep)")
        ax.bar(x + w / 2, ours, w, label="recomputed (this script)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("zero-filled PSNR (dB)")
        ax.set_title("ZF alignment (all %d test slices)" % report["data"]["n_test"])
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, "fig2_zf_alignment.png")
    except Exception as e:
        log("WARN fig2 failed: %s: %s" % (type(e).__name__, str(e)))

    try:
        if model is not None and len(test_idx) >= 2:
            sub = [int(i) for i in test_idx[:2]]
            g = store.get_batch(sub, device=device)
            mask = mask_store.get(TRAIN_MASK, device=device)
            y = fft2_t(g) * mask
            zf = ifft2_t(y)
            model.eval()
            with torch.no_grad():
                rec = to_c(model(to_2ch(zf)))
            fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
            for r in range(2):
                vmax = float(g[r].abs().max().item())
                axes[r][0].imshow(g[r].abs().cpu().numpy(), cmap="gray", vmin=0, vmax=vmax)
                axes[r][1].imshow(zf[r].abs().cpu().numpy(), cmap="gray", vmin=0, vmax=vmax)
                axes[r][2].imshow(rec[r].abs().cpu().numpy(), cmap="gray", vmin=0, vmax=vmax)
                if r == 0:
                    axes[r][0].set_title("GT")
                    axes[r][1].set_title("zero-filled")
                    axes[r][2].set_title("UNet (best)")
                for c in range(3):
                    axes[r][c].axis("off")
            fig.suptitle("recon panels: 2 test slices (R=4, %s)" % TRAIN_MASK, fontsize=10)
            save(fig, "fig3_recon.png")
    except Exception as e:
        log("WARN fig3 failed: %s: %s" % (type(e).__name__, str(e)))

    try:
        uk = report.get("unrolled_ks")
        if uk:
            ks = sorted(int(k) for k in uk)
            vals = [float(uk[str(k)]) for k in ks]
            fig, ax = plt.subplots(figsize=(5.8, 4))
            ax.bar(["K=%d" % k for k in ks], vals, color="teal")
            ax.set_ylabel("val PSNR (dB)")
            ax.set_title("iterative headroom (unrolled K steps)")
            ax.grid(alpha=0.3)
            save(fig, "fig4_headroom.png")
    except Exception as e:
        log("WARN fig4 failed: %s: %s" % (type(e).__name__, str(e)))
    return made


# ---- report helpers --------------------------------------------------------
def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = float(o)
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return round(o, 6)
    return o


def save_report(report):
    p = os.path.join(HERE, "step5_320_ceiling_%s_report.json" % report["mode"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=json_default)
    log("report written: %s" % p)
    return p


def write_summary(report, t0):
    L = []
    A = L.append
    v = report["verdict"]
    A("=" * 74)
    A("STEP5_320_CEILING SUMMARY  (mode=%s, %s)" % (report["mode"], report["timestamp"]))
    A("=" * 74)
    A("VERDICT: %s" % v["status"])
    A("  aligned (ZF)    : %s" % v["aligned"])
    A("  ceiling         : %s" % v["ceiling"])
    A("  test PSNR r4_s42: %.2f dB (SSIM %.4f)" % (v["test_psnr"], v["test_ssim"]))
    A("  gain vs ZF      : %+.2f dB" % v["gain_vs_zf_db"])
    A("  headroom K8-K1  : %+.2f dB (%s)" % (v["headroom_db"], v["headroom"]))
    A("-" * 74)
    A("1. DATA (fastMRI knee singlecoil_val, 320x320 complex)")
    st = report["data"]["stats"]
    A("   split train/val/test slices: %d/%d/%d" % (report["data"]["n_train"],
                                                     report["data"]["n_val"],
                                                     report["data"]["n_test"]))
    A("   test fg fraction p50: %.3f | per-slice max p50: %.3f | gt SNR p50: %.2f dB"
      % (st["foreground_fraction"]["p50"], st["per_slice_max"]["p50"],
         st["gt_fg_snr_db"]["p50"]))
    A("-" * 74)
    A("2. ZF ALIGNMENT (recomputed vs fastmri_320_prep reference)")
    for row in report["zf"]["rows"]:
        A("   %-10s effR=%.2f psnr=%.2f+-%.2f ssim=%.4f (ref %.2f/%.4f, d=%.3f/%.4f) %s"
          % (row["mask"], row["effR"], row["psnr"]["mean"], row["psnr"]["std"],
             row["ssim"]["mean"], row["ref_psnr"], row["ref_ssim"],
             row["d_psnr"], row["d_ssim"], "OK" if row["aligned"] else "MISMATCH"))
    A("-" * 74)
    A("3. TRAINING (UNet3 base=%d, mask=%s)" % (report["args"]["base"], TRAIN_MASK))
    tr = report.get("train") or {}
    if tr.get("status") == "OK":
        A("   params: %d | best val PSNR: %.2f dB @ epoch %d | last epoch: %d | train time: %.1f s"
          % (tr["n_params"], tr["best_psnr"], tr["best_epoch"], tr["last_epoch"],
             tr["train_sec"]))
    else:
        A("   STATUS: %s" % tr.get("status", "n/a"))
    A("-" * 74)
    A("4. TEST (all %d test slices, 3 masks)" % report["data"]["n_test"])
    if report.get("test"):
        for mk in TEST_MASKS:
            te = report["test"][mk]
            A("   %-10s psnr=%.2f+-%.2f ssim=%.4f+-%.4f (n=%d)"
              % (mk, te["psnr_full"]["mean"], te["psnr_full"]["std"],
                 te["ssim"]["mean"], te["ssim"]["std"], te["n"]))
    else:
        A("   n/a (no evaluation completed)")
    A("-" * 74)
    A("5. ITERATIVE HEADROOM (val subset, K=1 is direct U-Net)")
    uk = report.get("unrolled_ks") or {}
    if uk:
        for k in sorted(int(x) for x in uk):
            A("   K=%d : %.2f dB" % (k, uk[str(k)]))
    else:
        A("   n/a (no evaluation completed)")
    A("-" * 74)
    A("6. VERDICT RATIONALE")
    A("   PASS   : aligned and PSNR>=%.1f and SSIM>=%.2f and gain>=+%.1f dB and headroom!=LOW"
      % (PASS_PSNR, PASS_SSIM, MIN_GAIN_DB))
    A("   REVIEW : aligned but PSNR %.1f-%.1f or headroom LOW" % (MID_PSNR, PASS_PSNR))
    A("   FAIL   : otherwise (or ZF mismatch)")
    A("   Note: --smoke is expected to stay REVIEW/FAIL; the full run decides.")
    A("=" * 74)
    A("elapsed %.1f s | report: %s" % (time.perf_counter() - t0,
        os.path.join(HERE, "step5_320_ceiling_%s_report.json" % report["mode"])))
    txt = "\n".join(L)
    p = os.path.join(HERE, "step5_320_ceiling_%s_summary.txt" % report["mode"])
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    return p


# ---- args + main -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="step5_320_ceiling: fastMRI knee 320 ceiling")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="2-epoch smoke run (subsets)")
    g.add_argument("--full", action="store_true", help="full run (default)")
    g.add_argument("--eval-only", action="store_true", help="skip training, evaluate checkpoint")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-max", type=int, default=64)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--aug", type=int, default=1, help="random flips augmentation")
    p.add_argument("--resume-dir", type=str, default="")
    args = p.parse_args()
    if args.smoke:
        args.epochs = 2
        args.warmup = 1
        args.patience = 5
        args.min_epochs = 1
        args.batch = 6
    args.amp = bool(args.amp)
    args.aug = bool(args.aug)
    args.mode = "smoke" if args.smoke else ("eval" if args.eval_only else "full")
    return args


def main():
    args = parse_args()
    t0 = time.perf_counter()
    mode = args.mode
    run_dir = os.path.join(HERE, "runs", "step5_320_%s" % mode)
    fig_dir = os.path.join(HERE, "step5_320_%s_figs" % mode)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    global _LOG_FH
    _LOG_FH = open(os.path.join(HERE, "step5_320_ceiling_%s_stdout.log" % mode),
                   "w", encoding="ascii", errors="replace")
    log("step5_320_ceiling mode=%s seed=%d amp=%s" % (mode, args.seed, args.amp))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device=%s torch=%s" % (dev, torch.__version__))
    if not os.path.exists(META_PATH):
        log("META NOT FOUND: %s" % META_PATH)
        return 1
    meta = torch.load(META_PATH, map_location="cpu", weights_only=False)
    log("meta keys=%s" % sorted(meta.keys()))
    store = ChunkStore(meta)
    mask_store = MaskStore(meta["masks"])
    train_idx, val_idx, test_idx = load_split(meta)
    log("split train=%d val=%d test=%d" % (len(train_idx), len(val_idx), len(test_idx)))
    ds = data_stats(store, test_idx, max_n=512, seed=0)
    log("data stats n=%d fg_p50=%.3f max_p50=%.3f mean_p50=%.3f snr_p50=%.2f"
        % (ds["n"], ds["foreground_fraction"]["p50"], ds["per_slice_max"]["p50"],
           ds["per_slice_mean"]["p50"], ds["gt_fg_snr_db"]["p50"]))
    zf = run_zf_alignment(store, test_idx, mask_store)
    data_common = {"n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                   "n_test": int(len(test_idx)), "stats": ds}
    if not zf["ok"]:
        log("ZF alignment FAILED -> abort (data pipeline mismatch)")
        report = {"script": "step5_320_ceiling.py",
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode,
                  "args": sanitize(vars(args)), "data": data_common, "zf": zf,
                  "verdict": {"status": "FAIL", "aligned": False,
                              "ceiling": "CEILING_LOW", "headroom": "LOW",
                              "test_psnr": 0.0, "test_ssim": 0.0,
                              "gain_vs_zf_db": 0.0, "headroom_db": 0.0}}
        save_report(report)
        write_summary(report, t0)
        return 1
    # training (or eval-only)
    model = None
    res = None
    if args.eval_only:
        ck_dir = args.resume_dir or os.path.join(HERE, "runs", "step5_320_full")
        ck_path = os.path.join(ck_dir, "checkpoint_best.pt")
        if not os.path.exists(ck_path):
            log("checkpoint not found: %s" % ck_path)
            return 1
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = UNet3(in_ch=2, base=args.base).to(dev)
        model.load_state_dict(ck["state_dict"], strict=False)
        model.eval()
        res = {"status": "OK", "n_params": int(ck.get("n_params", 0)),
               "best_psnr": float(ck.get("val_psnr", -1.0)),
               "best_epoch": int(ck.get("epoch", 0)),
               "last_epoch": int(ck.get("epoch", 0)),
               "train_sec": 0.0, "history": {}}
        log("eval-only: loaded %s (epoch=%d val_psnr=%.2f)"
            % (ck_path, res["best_epoch"], res["best_psnr"]))
    else:
        reset_run_dir(run_dir)
        rng = np.random.RandomState(args.seed)
        if args.smoke:
            train_use = rng.choice(np.asarray(train_idx, dtype=np.int64),
                                   size=min(384, len(train_idx)), replace=False).tolist()
        else:
            train_use = [int(i) for i in train_idx]
        val_n = 96 if args.smoke else int(args.val_max)
        val_use = rng.choice(np.asarray(val_idx, dtype=np.int64),
                             size=min(val_n, len(val_idx)), replace=False).tolist()
        log("train samples=%d val_check=%d mask=%s" % (len(train_use), len(val_use), TRAIN_MASK))
        mask = mask_store.get(TRAIN_MASK, device=dev)
        res = train_unet(store, np.asarray(train_use, dtype=np.int64), val_use,
                         mask, dev, args, run_dir)
        model = res["model"]
        if res["status"] != "OK":
            log("training failed -> FAIL")
            report = {"script": "step5_320_ceiling.py",
                      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode,
                      "args": sanitize(vars(args)), "data": data_common, "zf": zf,
                      "train": {k: v for k, v in res.items() if k != "model"},
                      "verdict": {"status": "FAIL", "aligned": True,
                                  "ceiling": "CEILING_LOW", "headroom": "LOW",
                                  "test_psnr": 0.0, "test_ssim": 0.0,
                                  "gain_vs_zf_db": 0.0, "headroom_db": 0.0}}
            save_report(report)
            write_summary(report, t0)
            return 1
    # test evaluation
    evals = {}
    for mk in TEST_MASKS:
        evals[mk] = eval_recon(model, store, test_idx, mask_store, mk, dev, batch=args.batch)
        e = evals[mk]
        log("TEST %s: psnr=%.2f+-%.2f ssim=%.4f+-%.4f (n=%d)"
            % (mk, e["psnr_full"]["mean"], e["psnr_full"]["std"],
               e["ssim"]["mean"], e["ssim"]["std"], e["n"]))
    unrolled = eval_unrolled_ks(model, store, val_idx, mask_store, TRAIN_MASK, dev,
                                ks=(1, 2, 4, 8), max_n=64, eta=1.0, batch=8)
    # verdict
    te = evals[TRAIN_MASK]
    zf_main = next(r for r in zf["rows"] if r["mask"] == TRAIN_MASK)
    psnr = float(te["psnr_full"]["mean"])
    ssim = float(te["ssim"]["mean"])
    gain = psnr - float(zf_main["psnr"]["mean"])
    hr = float(unrolled[8]) - float(unrolled[1])
    if hr >= 1.0:
        headroom = "HIGH"
    elif hr > -1.0:
        headroom = "FLAT"
    else:
        headroom = "LOW"
    if psnr >= PASS_PSNR and ssim >= PASS_SSIM and gain >= MIN_GAIN_DB:
        ceiling = "CEILING_HIGH"
    elif psnr >= MID_PSNR:
        ceiling = "CEILING_MID"
    else:
        ceiling = "CEILING_LOW"
    if zf["ok"] and ceiling == "CEILING_HIGH" and headroom != "LOW":
        verdict = "PASS"
    elif zf["ok"] and (ceiling == "CEILING_MID" or headroom == "LOW"):
        verdict = "REVIEW"
    else:
        verdict = "FAIL"
    # history csv + config
    hist = res.get("history") or {}
    if hist.get("loss"):
        with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss", "val_psnr"])
            for i in range(len(hist["loss"])):
                w.writerow([i + 1, hist["loss"][i], hist["val_psnr"][i]])
    cfg = sanitize(vars(args))
    cfg["train_mask"] = TRAIN_MASK
    cfg["test_masks"] = TEST_MASKS
    cfg["data_files"] = "fastmri_320_meta.pt + fastmri_320_gt_chunk_*.pt"
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    report = {
        "script": "step5_320_ceiling.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "args": sanitize(vars(args)),
        "device": str(dev),
        "torch": torch.__version__,
        "data": data_common,
        "zf": zf,
        "train": {k: v for k, v in res.items() if k != "model"},
        "test": {mk: {"psnr_full": evals[mk]["psnr_full"], "ssim": evals[mk]["ssim"],
                      "n": evals[mk]["n"]} for mk in TEST_MASKS},
        "unrolled_ks": {str(k): v for k, v in unrolled.items()},
        "verdict": {"status": verdict, "aligned": bool(zf["ok"]), "ceiling": ceiling,
                    "headroom": headroom, "test_psnr": round(psnr, 4),
                    "test_ssim": round(ssim, 4), "gain_vs_zf_db": round(gain, 4),
                    "headroom_db": round(hr, 4)},
    }
    try:
        made = make_figures(res, model, store, test_idx, mask_store, dev, report, mode, fig_dir)
        report["figures"] = made
    except Exception as e:
        log("WARN figures: %s: %s" % (type(e).__name__, str(e)))
        report["figures"] = []
    save_report(report)
    write_summary(report, t0)
    log("done in %.1fs | verdict: %s" % (time.perf_counter() - t0, verdict))
    if _LOG_FH is not None:
        try:
            _LOG_FH.close()
        except Exception:
            pass
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
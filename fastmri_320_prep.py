# -*- coding: ascii -*-
"""
fastmri_320_prep.py
===================
Build the prepared 320x320 fastMRI knee singlecoil_val dataset.

Transform recipe (validated by fastmri_align_probe.py: corr=1.0000,
psnr=154.31 dB, VERDICT ALIGNED):
    kspace (n, 640, 368/372) complex
      -> ifftshift -> ifft2(norm=ortho) -> fftshift   (== ifft2c)
      -> center-crop to (320, 320) complex image       (== reconstruction_esc)
      -> per-file normalization by max(|image|)        (max magnitude = 1.0)

Undersampling (image-domain kspace, exact-density convention, same as
the paper's step2 128x128 pipeline):
    mask on the (320, 320) grid, sampled along rows (phase-encode)
    total_sampled = round(320 / R)   =>  effR == R exactly
    center band  = round(0.08 * 320) rows fully sampled
    outer rows chosen by a seeded RNG (seeds 42, 123, 2025)
    zero-filled image = ifft2c(mask * fft2c(gt))

Split: by patient_id (fallback: by file), seed 42, 80/10/10 train/val/test.

Outputs:
    fastmri_320_prepared.pt
    fastmri_320_prep_summary.txt
    fastmri_320_prep_report.json

Optional env vars:
    FMRI_DATA_DIR   folder that directly contains the .h5 files
    OUT_DIR         output folder (default: this script's folder)
    PREP_MAX_FILES  process only this many .h5 files (0 = all, default)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows OpenMP fix (torch+numpy)
import sys
import json
import glob
import time

import numpy as np
import h5py
import torch

_LOG_LINES = []

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("OUT_DIR", "").strip() or HERE
try:
    PREP_MAX_FILES = int(os.environ.get("PREP_MAX_FILES", "0"))
except Exception:
    PREP_MAX_FILES = 0

IMG = 320                # final image size (320 x 320)
CTR_FRAC = 0.08          # center kspace fully-sampled fraction (roadmap)
RATES = [4, 8]
MASK_SEEDS = [42, 123, 2025]
SPLIT_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
SKIP_DIRS = {".git", "node_modules", "__pycache__", "$RECYCLE.BIN",
             "System Volume Information"}


def log(msg):
    line = "[PREP] " + msg
    print(line)
    _LOG_LINES.append(line)


def auto_find_data_dir(max_depth=8):
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    if not os.path.isdir(desktop):
        return ""
    for dirpath, dirnames, filenames in os.walk(desktop):
        if os.path.basename(dirpath).lower() == "singlecoil_val":
            h5s = [f for f in filenames if f.lower().endswith(".h5")]
            if h5s:
                return dirpath
        depth = dirpath[len(desktop):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    return ""


def resolve_data_dir():
    env_dir = os.environ.get("FMRI_DATA_DIR", "").strip()
    if env_dir and os.path.isdir(env_dir):
        return env_dir, "env"
    found = auto_find_data_dir()
    if found and os.path.isdir(found):
        return found, "autodetect"
    return "", "none"


def center_crop(x, h, w):
    sh, sw = x.shape[-2], x.shape[-1]
    if sh < h or sw < w:
        return None
    top = (sh - h) // 2
    left = (sw - w) // 2
    return x[..., top:top + h, left:left + w]


def ifft2c(k):
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), axes=(-2, -1),
                     norm="ortho"), axes=(-2, -1))


def fft2c(x):
    return np.fft.ifftshift(
        np.fft.fft2(np.fft.fftshift(x, axes=(-2, -1)), axes=(-2, -1),
                    norm="ortho"), axes=(-2, -1))

def load_h5(path):
    """Load a fastMRI single-coil knee .h5 file.

    Returns (kspace, esc, rss, attrs):
      kspace  complex (n, 640, 368/372) raw k-space
      esc     float32 (n, 320, 320) official reconstruction_esc reference
      rss     float32 (n, 320, 320) reconstruction_rss or None
      attrs   dict of scalar file attributes (patient_id, acquisition, ...)
    """
    with h5py.File(path, "r") as f:
        kspace = f["kspace"][()]
        esc = f["reconstruction_esc"][()]
        rss = None
        if "reconstruction_rss" in f:
            rss = f["reconstruction_rss"][()]
        attrs = {}
        for k in f.attrs:
            try:
                v = f.attrs[k]
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v, bytes):
                    attrs[str(k)] = v.decode("utf-8", "replace")
                elif isinstance(v, np.generic):
                    attrs[str(k)] = v.item()
                elif isinstance(v, (str, int, float)):
                    attrs[str(k)] = v
                else:
                    attrs[str(k)] = str(v)
            except Exception:
                pass
    return kspace, esc, rss, attrs


def get_patient_id(attrs, path):
    pid = attrs.get("patient_id", "")
    pid = str(pid).strip()
    if not pid or pid.lower() in ("", "none", "nan"):
        pid = os.path.splitext(os.path.basename(path))[0]
    return pid


def build_vd_mask(n, rate, ctr_frac, seed):
    """Variable-density 1D phase-encode mask on an (n, n) grid.

    Convention (identical to the paper's 128x128 pipeline):
      total_sampled = round(n / rate)   ->  effective R is exact
      center band   = 2 * round(0.5 * ctr_frac * n) rows fully sampled
      outer rows    = drawn by a seeded RNG and mirrored for symmetry

    Returns (mask_1d, meta) where mask_1d is a length-n bool array.
    """
    total = int(round(n / float(rate)))
    c = 2 * int(round(0.5 * ctr_frac * n))
    if total <= c:
        total = c
    mask = np.zeros(n, dtype=bool)
    lo = (n - c) // 2
    mask[lo:lo + c] = True
    half = n // 2
    center_idx = set(range(lo, lo + c))
    pairs = [i for i in range(half)
             if i not in center_idx and (n - 1 - i) not in center_idx]
    rng = np.random.default_rng(seed)
    outer = total - c
    if outer > 0:
        k_pair = outer // 2
        if k_pair > len(pairs):
            k_pair = len(pairs)
        chosen = rng.choice(pairs, size=k_pair, replace=False)
        for i in chosen:
            mask[i] = True
            mask[n - 1 - i] = True
        if outer % 2 == 1:
            free = [i for i in range(n) if not mask[i]]
            if free:
                mask[int(rng.choice(free))] = True
    eff = n / float(mask.sum())
    meta = {
        "rate": float(rate),
        "seed": int(seed),
        "effR": eff,
        "total_rows": int(mask.sum()),
        "center_rows": int(c),
        "outer_rows": int(mask.sum() - c),
        "symmetric": bool(np.array_equal(mask, mask[::-1])),
    }
    return mask, meta


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
    """Lightweight structural similarity on magnitude images (fastMRI style).

    Pure-numpy implementation: separable Gaussian window (7x7, sigma 1.5),
    FFT-based correlation with zero padding, valid-region mean,
    data_range = max(gt magnitude), k1 = 0.01, k2 = 0.03.
    """

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


def _pearson_corr(a, b):
    """Pearson correlation on magnitudes (same convention as fastmri_align_probe)."""
    a = np.abs(a).ravel().astype(np.float64)
    b = np.abs(b).ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _psnr_norm(a, b):
    """Self-normalized magnitude PSNR (same convention as fastmri_align_probe)."""
    a = np.abs(a).astype(np.float64)
    b = np.abs(b).astype(np.float64)
    ma = float(a.max())
    mb = float(b.max())
    if ma > 0.0:
        a = a / ma
    if mb > 0.0:
        b = b / mb
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-20:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def alignment_stats(raw_img, esc, n_check=3):
    """Per-slice magnitude correlation + self-normalized PSNR vs official esc."""
    n = raw_img.shape[0]
    idxs = sorted(set(np.linspace(0, n - 1, n_check).astype(int)))
    corrs, psnrs = [], []
    for si in idxs:
        corrs.append(_pearson_corr(raw_img[si], esc[si]))
        psnrs.append(_psnr_norm(raw_img[si], esc[si]))
    return float(np.min(corrs)), float(np.max(corrs)), float(np.min(psnrs)), idxs

def json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def main():
    t0 = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    errors, warnings = [], []
    log("fastmri_320_prep.py  (fastMRI knee singlecoil_val -> 320x320 prepared)")
    log("started: " + start_str)

    py_ver = sys.version.split()[0]
    np_ver = np.__version__
    h5_ver = h5py.__version__
    torch_ver = torch.__version__ if torch is not None else "n/a"
    log("env: python=%s numpy=%s h5py=%s torch=%s" % (py_ver, np_ver, h5_ver, torch_ver))

    data_dir, src = resolve_data_dir()
    if not data_dir:
        log("ERROR: fastMRI data dir not found; set env FMRI_DATA_DIR to the folder with the .h5 files")
        log("verdict: FAIL (errors=1 warnings=%d)" % len(warnings))
        return 1
    log("data_dir=%s (source=%s)" % (data_dir, src))

    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    n_files = len(files)
    log("h5 files=%d" % n_files)
    if n_files == 0:
        log("ERROR: no .h5 files found in data dir")
        log("verdict: FAIL (errors=1 warnings=%d)" % len(warnings))
        return 1
    if PREP_MAX_FILES and PREP_MAX_FILES < n_files:
        files = files[:PREP_MAX_FILES]
        log("WARNING: PARTIAL run, PREP_MAX_FILES=%d (NOT the final dataset)" % PREP_MAX_FILES)
        warnings.append("partial")

    # ---- pass 1: convert + alignment check + normalize ----
    gt_list = []
    slice_files = []
    slice_local = []
    file_meta = []
    align_corrs, align_psnrs = [], []
    for fi, path in enumerate(files):
        try:
            kspace, esc, rss, attrs = load_h5(path)
        except Exception as e:
            log("ERROR: load failed %s : %s" % (os.path.basename(path), e))
            errors.append("load:%s" % os.path.basename(path))
            continue
        raw = center_crop(ifft2c(kspace), IMG, IMG)
        if raw is None:
            log("ERROR: crop failed (kspace too small) %s shape=%s" % (os.path.basename(path), list(kspace.shape)))
            errors.append("crop:%s" % os.path.basename(path))
            continue
        if esc.shape[-2:] != (IMG, IMG):
            esc = center_crop(esc, IMG, IMG)
            if esc is None:
                log("ERROR: esc crop failed %s" % os.path.basename(path))
                errors.append("esc:%s" % os.path.basename(path))
                continue
        if raw.shape[0] != esc.shape[0]:
            m = min(raw.shape[0], esc.shape[0])
            raw, esc = raw[:m], esc[:m]
        if raw.shape[0] == 0:
            warnings.append("empty:%s" % os.path.basename(path))
            continue
        corr_min, corr_max, psnr_min, idxs = alignment_stats(raw, esc, n_check=3)
        align_corrs.append(corr_min)
        align_psnrs.append(psnr_min)
        if corr_min < 0.99:
            log("ERROR: alignment corr=%.4f (<0.99) file=%s slices=%s" % (corr_min, os.path.basename(path), list(idxs)))
            errors.append("align:%s" % os.path.basename(path))
            continue
        fmax = float(np.max(np.abs(raw)))
        if fmax <= 0:
            warnings.append("zero_max:%s" % os.path.basename(path))
            continue
        img = (raw / fmax).astype(np.complex64)
        n_slices = img.shape[0]
        gt_list.append(img)
        slice_files.extend([fi] * n_slices)
        slice_local.extend(range(n_slices))
        file_meta.append({
            "file": os.path.basename(path),
            "file_idx": fi,
            "patient_id": get_patient_id(attrs, path),
            "n_slices": int(n_slices),
            "max_val": float(fmax),
            "kspace_shape": list(kspace.shape),
            "acquisition": str(attrs.get("acquisition", "")),
        })
    if not gt_list:
        log("ERROR: no usable files after conversion")
        log("verdict: FAIL (errors=%d warnings=%d)" % (len(errors), len(warnings)))
        return 1

    gt = np.concatenate(gt_list, axis=0).astype(np.complex64)
    slice_files = np.asarray(slice_files, dtype=np.int64)
    slice_local = np.asarray(slice_local, dtype=np.int64)
    N = gt.shape[0]
    n_used = len(file_meta)
    log("converted: slices=%d files=%d avg=%.1f slices/file" % (N, n_used, N / float(n_used)))
    log("alignment: files_checked=%d corr_min=%.6f corr_max=%.6f psnr_min=%.2f" % (
        len(align_corrs), min(align_corrs), max(align_corrs), min(align_psnrs)))
    if any(e.startswith("align:") for e in errors):
        log("alignment: FAIL (some files excluded from dataset)")
    else:
        log("alignment: ALIGNED (recipe matches official reconstruction_esc)")

    # ---- split by patient_id (file level, 80/10/10) ----
    pid_files = {}
    for fm in file_meta:
        pid_files.setdefault(fm["patient_id"], []).append(fm["file_idx"])
    pids = sorted(pid_files.keys())
    rng_split = np.random.default_rng(SPLIT_SEED)
    perm = rng_split.permutation(len(pids)).tolist()
    train_t = int(round(n_used * TRAIN_FRAC))
    val_t = int(round(n_used * (TRAIN_FRAC + VAL_FRAC)))
    train_files, val_files, test_files = [], [], []
    n_tr = n_va = 0
    for pi in perm:
        fs = pid_files[pids[pi]]
        if n_tr + len(fs) <= train_t:
            train_files.extend(fs)
            n_tr += len(fs)
        elif n_va + len(fs) <= val_t - train_t:
            val_files.extend(fs)
            n_va += len(fs)
        else:
            test_files.extend(fs)
    if not train_files or not val_files or not test_files:
        warnings.append("split_fallback_file_level")
        log("WARNING: patient split left an empty set -> fallback to file-level split")
        rng_fb = np.random.default_rng(SPLIT_SEED + 1)
        perm_f = rng_fb.permutation(n_used).tolist()
        train_files = perm_f[:train_t]
        val_files = perm_f[train_t:val_t]
        test_files = perm_f[val_t:]
    tr_set = np.asarray(train_files, dtype=np.int64)
    va_set = np.asarray(val_files, dtype=np.int64)
    te_set = np.asarray(test_files, dtype=np.int64)
    tr_s = int(np.sum(np.isin(slice_files, tr_set)))
    va_s = int(np.sum(np.isin(slice_files, va_set)))
    te_s = int(np.sum(np.isin(slice_files, te_set)))
    log("split (by patient_id, seed=%d, file 80/10/10): train files=%d slices=%d | val files=%d slices=%d | test files=%d slices=%d" % (
        SPLIT_SEED, len(train_files), tr_s, len(val_files), va_s, len(test_files), te_s))

    # ---- masks ----
    masks = {}
    mask_meta_rows = []
    for rate in RATES:
        for seed in MASK_SEEDS:
            key = "r%d_s%d" % (rate, seed)
            m1d, meta = build_vd_mask(IMG, rate, CTR_FRAC, seed)
            m2d = np.repeat(m1d.reshape(1, IMG), IMG, axis=0)
            masks[key] = {"mask": torch.from_numpy(m2d.astype(np.bool_)), "meta": meta}
            mask_meta_rows.append((rate, seed, meta))
            log("mask %s: effR=%.4f total=%d center=%d outer=%d sym=%s" % (
                key, meta["effR"], meta["total_rows"], meta["center_rows"],
                meta["outer_rows"], meta["symmetric"]))
            if abs(meta["effR"] - rate) > 1e-9:
                warnings.append("effR:%s" % key)
            if not meta["symmetric"]:
                warnings.append("sym:%s" % key)
    # ---- zero-filled baseline on test slices (R=4/R=8 x 3 seeds) ----
    test_slices = np.where(np.isin(slice_files, te_set))[0]
    log("test slices=%d (files=%d)" % (len(test_slices), len(test_files)))
    if len(test_slices) == 0:
        log("ERROR: no test slices available")
        errors.append("no_test_slices")
    test_gm, test_ksp = [], []
    for si in test_slices:
        g = gt[si]
        test_gm.append(np.abs(g))
        test_ksp.append(fft2c(g))
    ssim_engine = SSIMComputer()
    zf_rows = []
    for rate in RATES:
        for seed in MASK_SEEDS:
            key = "r%d_s%d" % (rate, seed)
            m2d = masks[key]["mask"].numpy()
            psnrs = np.zeros(len(test_slices), dtype=np.float64)
            ssims = np.zeros(len(test_slices), dtype=np.float64)
            for j in range(len(test_slices)):
                y = m2d * test_ksp[j]
                zf = ifft2c(y)
                psnrs[j] = compute_psnr(test_gm[j], np.abs(zf))
                ssims[j] = ssim_engine.compute(test_gm[j], np.abs(zf))
            log("zf %s: psnr=%.2f+-%.2f ssim=%.4f+-%.4f (test slices=%d)" % (
                key, psnrs.mean(), psnrs.std(), ssims.mean(), ssims.std(), len(test_slices)))
            zf_rows.append({
                "mask": key,
                "n": int(len(test_slices)),
                "psnr_mean": float(psnrs.mean()),
                "psnr_std": float(psnrs.std()),
                "ssim_mean": float(ssims.mean()),
                "ssim_std": float(ssims.std()),
            })

    # ---- save prepared dataset ----
    out_path = os.path.join(OUT_DIR, "fastmri_320_prepared.pt")
    payload = {
        "gt": torch.from_numpy(gt),
        "slice_files": torch.from_numpy(slice_files),
        "slice_local": torch.from_numpy(slice_local),
        "file_meta": file_meta,
        "split": {
            "seed": SPLIT_SEED,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
            "n_files": {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
            "n_slices": {"train": tr_s, "val": va_s, "test": te_s},
        },
        "masks": masks,
        "config": {
            "img_size": IMG,
            "ctr_frac": CTR_FRAC,
            "rates": RATES,
            "mask_seeds": MASK_SEEDS,
            "split_seed": SPLIT_SEED,
            "transform": "ifft2c + center_crop 320 + per-file max normalization",
            "gt_format": "complex64, magnitude normalized to per-file max=1",
            "reference": "reconstruction_esc (fastMRI official)",
        },
    }
    t_save0 = time.time()
    torch.save(payload, out_path)
    size_gb = os.path.getsize(out_path) / 1e9
    log("saved %s (%.2f GB, %.1fs)" % (out_path, size_gb, time.time() - t_save0))

    # ---- text summary + json report ----
    summary_path = os.path.join(OUT_DIR, "fastmri_320_prep_summary.txt")
    report_path = os.path.join(OUT_DIR, "fastmri_320_prep_report.json")
    lines = list(_LOG_LINES)
    lines.append("")
    lines.append("[PREP] ---- split summary (by patient_id, seed 42) ----")
    lines.append("[PREP]   train: files=%d slices=%d" % (len(train_files), tr_s))
    lines.append("[PREP]   val  : files=%d slices=%d" % (len(val_files), va_s))
    lines.append("[PREP]   test : files=%d slices=%d" % (len(test_files), te_s))
    lines.append("[PREP] ---- mask summary ----")
    for (rate, seed, meta) in mask_meta_rows:
        lines.append("[PREP]   r%d_s%d: effR=%.3f total=%d center=%d outer=%d sym=%s" % (
            rate, seed, meta["effR"], meta["total_rows"], meta["center_rows"],
            meta["outer_rows"], meta["symmetric"]))
    lines.append("[PREP] ---- zero-filled baseline (test slices, magnitude PSNR/SSIM) ----")
    for row in zf_rows:
        lines.append("[PREP]   %s: psnr=%.2f+-%.2f ssim=%.4f+-%.4f (n=%d)" % (
            row["mask"], row["psnr_mean"], row["psnr_std"],
            row["ssim_mean"], row["ssim_std"], row["n"]))
    lines.append("")
    lines.append("[PREP] outputs:")
    lines.append("[PREP]   prepared: %s (%.2f GB)" % (out_path, size_gb))
    lines.append("[PREP]   summary : %s" % summary_path)
    lines.append("[PREP]   report  : %s" % report_path)
    lines.append("[PREP] done in %.1fs" % (time.time() - t0))
    verdict = "OK" if (not errors and not warnings) else ("WARN" if not errors else "FAIL")
    lines.append("[PREP] verdict: %s (errors=%d warnings=%d)" % (verdict, len(errors), len(warnings)))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    report = {
        "script": "fastmri_320_prep.py",
        "started": start_str,
        "duration_s": round(time.time() - t0, 2),
        "env": {"python": py_ver, "numpy": np_ver, "h5py": h5_ver, "torch": torch_ver},
        "data": {
            "dir": data_dir,
            "source": src,
            "n_files_total": n_files,
            "n_files_used": n_used,
            "n_slices": int(N),
            "avg_slices_per_file": round(N / float(n_used), 2),
        },
        "alignment": {
            "files_checked": len(align_corrs),
            "corr_min": round(float(min(align_corrs)), 6),
            "corr_max": round(float(max(align_corrs)), 6),
            "psnr_min": round(float(min(align_psnrs)), 2),
            "status": "ALIGNED" if not any(e.startswith("align:") for e in errors) else "FAIL",
        },
        "transform": "ifft2c + center_crop 320 + per-file max normalization",
        "split": {
            "seed": SPLIT_SEED,
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
            "n_files": {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
            "n_slices": {"train": tr_s, "val": va_s, "test": te_s},
        },
        "masks": {key: masks[key]["meta"] for key in masks},
        "zero_filled": zf_rows,
        "outputs": {
            "prepared": out_path,
            "size_gb": round(size_gb, 2),
            "summary": summary_path,
            "report": report_path,
        },
        "errors": errors,
        "warnings": warnings,
        "verdict": verdict,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True, default=json_default)

    log("report  : %s" % report_path)
    log("summary : %s" % summary_path)
    log("verdict : %s (errors=%d warnings=%d)" % (verdict, len(errors), len(warnings)))
    log("done in %.1fs" % (time.time() - t0))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
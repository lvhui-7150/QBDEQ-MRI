# -*- coding: ascii -*-
"""
fastmri_align_probe.py
======================
Diagnostic probe: find the kspace -> image transform that aligns the raw
fastMRI single-coil knee VAL kspace with the scanner reference image
(reconstruction_esc / reconstruction_rss).

Why needed:
  Naive ifft2 + center-crop of the raw kspace does NOT reliably match the
  reference reconstruction (correlation ~0.2-0.4 in a first audit). This
  probe brute-forces the small set of plausible transforms (fft recipe x
  crop strategy x image orientation) and reports the best one, plus a
  per-slice correspondence check (does slice i of kspace match slice i of
  the reference?).

Data folder resolution (in order):
  1. env FMRI_DATA_DIR (if set and valid)
  2. auto-search under the Desktop for a folder named 'singlecoil_val'
     that contains .h5 files

Usage (PowerShell, run from the experiment folder):
  python fastmri_align_probe.py

Environment (all optional):
  FMRI_DATA_DIR : folder that directly contains the .h5 files
  OUT_DIR       : where to write outputs (default: this script's folder)
  N_FILES       : number of .h5 files to probe (default 3)

Outputs:
  fastmri_align_probe_summary.txt  (readable text, UTF-8)
  fastmri_align_probe_report.json  (full numeric detail, UTF-8)

This probe only reads data and computes correlations. It does not train.
"""

import os
import sys
import json
import glob
import time

import numpy as np
import h5py

_LOG_LINES = []

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("OUT_DIR", "").strip() or HERE
try:
    N_FILES = int(os.environ.get("N_FILES", "3"))
except Exception:
    N_FILES = 3

CROP_H, CROP_W = 320, 320
MIN_H, MIN_W = 640, 320

FFT_MODES = ["ifft2", "fft2"]
NORMS = ["ortho", "backward"]
PRES = ["none", "ifftshift", "fftshift"]
POSTS = ["none", "fftshift", "ifftshift"]

FASTMRI_STD = ("ifft2", "ortho", "ifftshift", "fftshift")

ORIENTS = {
    "none": lambda x: x,
    "flip_h": lambda x: np.flip(x, axis=-1),
    "flip_v": lambda x: np.flip(x, axis=-2),
    "flip_hv": lambda x: np.flip(np.flip(x, axis=-1), axis=-2),
    "transpose": lambda x: np.swapaxes(x, -2, -1),
    "trans_flip_h": lambda x: np.flip(np.swapaxes(x, -2, -1), axis=-1),
    "trans_flip_v": lambda x: np.flip(np.swapaxes(x, -2, -1), axis=-2),
    "trans_flip_hv": lambda x: np.flip(np.flip(np.swapaxes(x, -2, -1), axis=-1), axis=-2),
}

ORIENT_LEGEND = {
    "none": "as-is",
    "flip_h": "flip left-right",
    "flip_v": "flip top-bottom",
    "flip_hv": "flip both axes (180 deg rotation)",
    "transpose": "swap readout/phase axes (90 deg)",
    "trans_flip_h": "transpose + flip left-right",
    "trans_flip_v": "transpose + flip top-bottom",
    "trans_flip_hv": "transpose + flip both",
}

SKIP_DIRS = {".git", "node_modules", "__pycache__", "$RECYCLE.BIN",
             "System Volume Information"}


def log(msg):
    line = "[PROBE] " + msg
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


def fourier(k, mode, norm, pre, post):
    x = k
    if pre == "ifftshift":
        x = np.fft.ifftshift(x, axes=(-2, -1))
    elif pre == "fftshift":
        x = np.fft.fftshift(x, axes=(-2, -1))
    if mode == "ifft2":
        x = np.fft.ifft2(x, axes=(-2, -1), norm=norm)
    else:
        x = np.fft.fft2(x, axes=(-2, -1), norm=norm)
    if post == "ifftshift":
        x = np.fft.ifftshift(x, axes=(-2, -1))
    elif post == "fftshift":
        x = np.fft.fftshift(x, axes=(-2, -1))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def corr(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def psnr_norm(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ma = a.max()
    mb = b.max()
    if ma > 0:
        a = a / ma
    if mb > 0:
        b = b / mb
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-20:
        return 99.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def load_h5(path):
    with h5py.File(path, "r") as f:
        ks = f["kspace"][...]
        ref = None
        ref_key = None
        for key in ("reconstruction_esc", "reconstruction_rss"):
            if key in f:
                ref = f[key][...]
                ref_key = key
                break
        attrs = {}
        for k in ("acquisition", "patient_id", "max", "norm"):
            if k in f.attrs:
                attrs[k] = str(f.attrs[k])
    return ks, ref, ref_key, attrs


def find_probe_files(data_dir, n):
    h5s = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    picked = []
    for p in h5s:
        try:
            with h5py.File(p, "r") as f:
                if "kspace" not in f:
                    continue
                sh = f["kspace"].shape
                if len(sh) != 3:
                    continue
                if sh[1] >= MIN_H and sh[2] >= MIN_W:
                    picked.append(p)
        except Exception:
            continue
        if len(picked) >= n:
            break
    return picked


def make_image(k_slice, combo, crop):
    mode, norm, pre, post = combo
    x = k_slice
    if crop == "crop_k":
        x = center_crop(x, CROP_H, CROP_W)
        if x is None:
            return None
    img = fourier(x, mode, norm, pre, post)
    if crop == "crop_img":
        img = center_crop(img, CROP_H, CROP_W)
        if img is None:
            return None
    return img


def combo_str(combo):
    return "fft=(%s,%s,%s,%s)" % (combo[0], combo[1], combo[2], combo[3])


def main():
    t0 = time.time()
    data_dir, src = resolve_data_dir()
    if not data_dir:
        log("ERROR: could not find the fastMRI data folder")
        log("  either set the env var in THIS window first:")
        log("  $env:FMRI_DATA_DIR='C:/path/to/singlecoil_val'")
        log("  or make sure a folder named 'singlecoil_val' with .h5 files")
        log("  exists somewhere under the Desktop")
        return 2
    if not os.path.isdir(OUT_DIR):
        try:
            os.makedirs(OUT_DIR)
        except Exception:
            pass

    log("data_dir_source=%s data_dir_len=%d" % (src, len(data_dir)))
    n_h5 = len(glob.glob(os.path.join(data_dir, "*.h5")))
    log("n_h5_in_dir=%d" % n_h5)
    files = find_probe_files(data_dir, N_FILES)
    if not files:
        log("ERROR: no suitable .h5 files found under the resolved data folder")
        return 2
    log("probe_files=%d" % len(files))
    for p in files:
        log("  file: %s" % os.path.basename(p))

    loaded = []
    for p in files:
        ks, ref, ref_key, attrs = load_h5(p)
        loaded.append((p, ks, ref, ref_key, attrs))
        log("loaded %s kspace=%s ref=%s ref_key=%s acq=%s" % (
            os.path.basename(p), str(ks.shape),
            str(ref.shape) if ref is not None else "None", ref_key,
            attrs.get("acquisition", "")))
    if any(ref is None for _, _, ref, _, _ in loaded):
        log("ERROR: at least one file has no reference reconstruction")
        return 2

    combos = [(m, n, pre, post) for m in FFT_MODES for n in NORMS
              for pre in PRES for post in POSTS]
    crops = ["crop_img", "crop_k"]
    log("fft_grid=%d orient_grid=%d crop_grid=%d" % (len(combos), len(ORIENTS), len(crops)))

    # ---- stage 1: full grid on first file, middle slice ----
    p0, ks0, ref0, refk0, at0 = loaded[0]
    s_mid = ks0.shape[0] // 2
    k1 = ks0[s_mid]
    r1 = ref0[s_mid]
    log("")
    log("stage1: file=%s slice=%d ref_max=%.3e" % (
        os.path.basename(p0), s_mid, float(np.max(np.abs(r1)))))

    results = []
    for combo in combos:
        for crop in crops:
            img = make_image(k1, combo, crop)
            if img is None:
                continue
            for oname, ofn in ORIENTS.items():
                m = np.abs(ofn(img))
                c = corr(m, r1)
                p = psnr_norm(m, r1)
                score = c
                if combo == FASTMRI_STD and crop == "crop_img" and oname == "none":
                    score = c + 1e-6
                results.append((score, c, p, combo, crop, oname))
    results.sort(key=lambda t: t[0], reverse=True)
    log("stage1 evals=%d" % len(results))
    log("stage1 top10 (corr | psnr | fft | crop | orient | meaning):")
    for i, (score, c, p, combo, crop, oname) in enumerate(results[:10]):
        log("  #%d corr=%.4f psnr=%7.2f fft=(%s,%s,%s,%s) crop=%s orient=%s [%s]" % (
            i + 1, c, p, combo[0], combo[1], combo[2], combo[3],
            crop, oname, ORIENT_LEGEND[oname]))

    std_img = make_image(k1, FASTMRI_STD, "crop_img")
    std_m = np.abs(std_img)
    log("  (fastMRI standard ifft2c+crop corr=%.4f psnr=%7.2f)" % (
        corr(std_m, r1), psnr_norm(std_m, r1)))

    # ---- stage 2: top-3 combos across probe slices + slice-correspondence check
    top3 = results[:3]
    probe_rows = []
    for fi, (p, ks, ref, ref_key, attrs) in enumerate(loaded):
        ns = min(ks.shape[0], ref.shape[0])
        slices = sorted(set([0, ns // 2, ns - 1]))
        for si in slices:
            k_s = ks[si]
            for rank, (score, c0, p0v, combo, crop, oname) in enumerate(top3):
                img = make_image(k_s, combo, crop)
                if img is None:
                    continue
                m = np.abs(ORIENTS[oname](img))
                own_c = corr(m, ref[si])
                own_p = psnr_norm(m, ref[si])
                best_i = -1
                best_c = -1.0
                best_p = -99.0
                for ri in range(ref.shape[0]):
                    cc = corr(m, ref[ri])
                    if cc > best_c:
                        best_c = cc
                        best_i = ri
                        best_p = psnr_norm(m, ref[ri])
                probe_rows.append((fi, si, rank, own_c, own_p, best_i, best_c, best_p, combo, crop, oname))

    log("")
    log("stage2 per-slice for the stage1 winner:")
    win = results[0]
    w_score, w_c, w_p, w_combo, w_crop, w_orient = win
    for row in probe_rows:
        if row[8] == w_combo and row[9] == w_crop and row[10] == w_orient:
            log("  f%d slice=%2d own_corr=%.4f own_psnr=%6.2f | best_ref_idx=%2d best_corr=%.4f best_psnr=%6.2f" % (
                row[0], row[1], row[3], row[4], row[5], row[6], row[7]))

    log("")
    log("stage2 winner also vs rss reference (own index):")
    for fi, (p, ks, ref, ref_key, attrs) in enumerate(loaded):
        ns = min(ks.shape[0], ref.shape[0])
        with h5py.File(p, "r") as f:
            rss = f["reconstruction_rss"][...] if "reconstruction_rss" in f else None
        for si in sorted(set([0, ns // 2, ns - 1])):
            img = make_image(ks[si], w_combo, w_crop)
            if img is None:
                continue
            m = np.abs(ORIENTS[w_orient](img))
            if rss is not None:
                log("  f%d slice=%2d esc_corr=%.4f rss_corr=%.4f" % (fi, si, corr(m, ref[si]), corr(m, rss[si])))

    log("")
    log("baseline (fastMRI standard recipe) own-index corr/psnr per probe slice:")
    for fi, (p, ks, ref, ref_key, attrs) in enumerate(loaded):
        ns = min(ks.shape[0], ref.shape[0])
        for si in sorted(set([0, ns // 2, ns - 1])):
            img = make_image(ks[si], FASTMRI_STD, "crop_img")
            m = np.abs(img)
            log("  f%d slice=%2d corr=%.4f psnr=%6.2f" % (fi, si, corr(m, ref[si]), psnr_norm(m, ref[si])))

    # ---- verdict ----
    own_corrs = []
    mapping_mismatch = 0
    for row in probe_rows:
        if row[8] == w_combo and row[9] == w_crop and row[10] == w_orient:
            own_corrs.append(row[3])
            if row[5] != row[1]:
                mapping_mismatch += 1
    min_own = float(min(own_corrs)) if own_corrs else 0.0
    max_own = float(max(own_corrs)) if own_corrs else 0.0
    if min_own >= 0.90:
        verdict = "ALIGNED"
    elif min_own >= 0.60:
        verdict = "PARTIAL"
    else:
        verdict = "MISALIGNED"
    log("")
    log("winner: %s crop=%s orient=%s" % (combo_str(w_combo), w_crop, w_orient))
    log("winner own-corr across probe slices: min=%.4f max=%.4f" % (min_own, max_own))
    log("probe slices where best_ref_idx != own_idx: %d / %d" % (
        mapping_mismatch, len(own_corrs) if own_corrs else 0))
    log("VERDICT: %s" % verdict)
    if verdict != "ALIGNED":
        log("  hint: inspect best_ref_idx columns in the report; a constant offset")
        log("  (e.g. best = n-1-own) means the kspace slice order is reversed.")

    # ---- write outputs ----
    report = {
        "data_dir": data_dir,
        "data_dir_source": src,
        "files": [os.path.basename(p) for p in files],
        "verdict": verdict,
        "winner": {"combo": list(w_combo), "crop": w_crop, "orient": w_orient,
                   "corr_stage1": results[0][1], "psnr_stage1": results[0][2]},
        "stage1_top20": [
            {"rank": i + 1, "corr": r[1], "psnr": r[2], "combo": list(r[3]),
             "crop": r[4], "orient": r[5]}
            for i, r in enumerate(results[:20])],
        "stage2": [
            {"file": row[0], "slice": row[1], "rank": row[2], "own_corr": row[3],
             "own_psnr": row[4], "best_ref_idx": row[5], "best_corr": row[6],
             "best_psnr": row[7], "combo": list(row[8]), "crop": row[9], "orient": row[10]}
            for row in probe_rows],
        "runtime_s": round(time.time() - t0, 1),
    }
    sum_path = os.path.join(OUT_DIR, "fastmri_align_probe_summary.txt")
    rep_path = os.path.join(OUT_DIR, "fastmri_align_probe_report.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(chr(10).join(_LOG_LINES) + chr(10))
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log("")
    log("summary written: %s" % sum_path)
    log("report written: %s" % rep_path)
    log("done in %.1fs" % (time.time() - t0))
    return 0 if verdict == "ALIGNED" else (1 if verdict == "PARTIAL" else 2)


if __name__ == "__main__":
    sys.exit(main())
# -*- coding: ascii -*-
"""
fastmri_320_fix.py
==================
Repair the 320x320 fastMRI knee prepared dataset (chunked + verified).

WHY THIS SCRIPT EXISTS
----------------------
fastmri_320_prepared.pt (written by fastmri_320_prep.py, 2,148,250,639
bytes) is CORRUPT:
    torch.load -> RuntimeError: PytorchStreamReader failed reading zip
    archive: not a ZIP archive
Diagnosis (zipfile inspection):
    - central-directory offsets are broken (negative values and values
      near 2**31 = 2147483648)
    - archive/data/0 (the full gt tensor) claims file_size =
      5,844,992,000 (~5.44 GiB) but the file is only 2.00 GiB
    - reading any entry raises OSError [Errno 22]
Conclusion: torch.save produced a zip whose large gt entry was truncated
around the 2 GiB boundary; the archive is unusable.  The underlying
fastMRI knee .h5 files are intact.

FIX
---
Regenerate the dataset from the original fastMRI knee singlecoil_val .h5
files with EXACTLY the same transform / split / mask recipe as
fastmri_320_prep.py (verified ALIGNED, corr = 1.0000), but store the big
gt tensor as N_CHUNKS separate .pt files (each < 1 GiB, far below the
2 GiB boundary) plus one small meta file.

OUTPUTS (in OUT_DIR, default = this script's folder)
----------------------------------------------------
    fastmri_320_meta.pt              small dict (split, masks, config, chunk spec)
    fastmri_320_gt_chunk_000.pt ...  complex64 (892,320,320) each; last chunk smaller
    fastmri_320_fix_summary.txt      human-readable log -> paste back to the agent
    fastmri_320_fix_report.json      machine-readable report
    fastmri_320_fix_stdout.log       full console log (tee)

OPTIONAL ENV VARS
-----------------
    FMRI_DATA_DIR   folder that directly contains the .h5 files
    OUT_DIR         output folder (default: this script's folder)
    FIX_MAX_FILES   process only this many .h5 files (0 = all, default)

REFERENCE (from the 2026-08-05 fastmri_320_prep run, same recipe)
-----------------------------------------------------------------
    split : train 5668 / val 663 / test 804 slices (158/19/22 files)
    zf    : R4 psnr ~25.1-25.4 dB | R8 psnr ~24.3-24.7 dB
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows OpenMP fix
import sys
import json
import glob
import time
import math
import gc

import numpy as np
import h5py
import torch

_LOG_LINES = []
_LOG_FH = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("OUT_DIR", "").strip() or HERE
try:
    FIX_MAX_FILES = int(os.environ.get("FIX_MAX_FILES", "0"))
except Exception:
    FIX_MAX_FILES = 0

IMG = 320                # final image size
CTR_FRAC = 0.08          # center kspace fully-sampled fraction
RATES = [4, 8]
MASK_SEEDS = [42, 123, 2025]
SPLIT_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
N_CHUNKS = 8
CHUNK_TAG = "fastmri_320_gt_chunk"
META_NAME = "fastmri_320_meta.pt"
SUMMARY_NAME = "fastmri_320_fix_summary.txt"
REPORT_NAME = "fastmri_320_fix_report.json"
LOG_NAME = "fastmri_320_fix_stdout.log"
REF_SPLIT = {"train": 5668, "val": 663, "test": 804}
REF_ZF = {
    "r4_s42": (25.08, 25.42),
    "r8_s42": (24.27, 24.65),
}
SKIP_DIRS = {".git", "node_modules", "__pycache__", "$RECYCLE.BIN",
             "System Volume Information"}


def log(msg):
    line = "[FIX] " + msg
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
    return str(o)


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
    """Returns (kspace, esc, rss, attrs); same fields as the prep script."""
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
    """Variable-density 1D phase-encode mask; identical to the prep script."""
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


def _pearson_corr(a, b):
    a = np.abs(a).ravel().astype(np.float64)
    b = np.abs(b).ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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


def verify_tensor(t, expected_shape, tag):
    """Returns (ok, detail) for one loaded chunk tensor."""
    ok = True
    detail = {}
    if not hasattr(t, "shape"):
        return False, {"reason": "not a tensor"}
    if t.dtype != torch.complex64:
        ok = False
        detail["dtype"] = str(t.dtype)
    if tuple(t.shape) != tuple(expected_shape):
        ok = False
        detail["shape"] = list(t.shape)
    if ok:
        fin = bool(torch.isfinite(t.real).all().item()
                   and torch.isfinite(t.imag).all().item())
        mx = float(t.abs().max().item())
        detail["finite"] = fin
        detail["max_abs"] = round(mx, 6)
        if not fin:
            ok = False
            detail["reason"] = "non-finite values"
        if mx > 1.01:
            ok = False
            detail["reason"] = "max_abs > 1.01 (normalization drift)"
    return ok, detail


def _finish(errors, warnings, t0, start_str, extra):
    global _LOG_FH
    duration = time.time() - t0
    verdict = "PASS" if not errors else "FAIL"
    log("")
    log("---- final ----")
    log("verdict: %s (errors=%d warnings=%d)" % (verdict, len(errors), len(warnings)))
    if errors:
        for e in errors:
            log("error: %s" % e)
    if warnings:
        for w in warnings:
            log("warning: %s" % w)
    log("outputs:")
    log("  meta   : %s" % os.path.join(OUT_DIR, META_NAME))
    for row in extra.get("chunk_rows", []):
        log("  chunk  : %s (%d slices, %.2f MB)" % (row["name"], row["n"], row["size_mb"]))
    log("  summary: %s" % os.path.join(OUT_DIR, SUMMARY_NAME))
    log("  report : %s" % os.path.join(OUT_DIR, REPORT_NAME))
    log("  stdout : %s" % os.path.join(OUT_DIR, LOG_NAME))
    log("done in %.1fs" % duration)

    s_lines = list(_LOG_LINES)
    s_lines.append("")
    s_lines.append("[FIX] ---- split summary (by patient_id, seed 42) ----")
    sp = extra.get("split", {})
    s_lines.append("[FIX]   train: files=%d slices=%d" % (
        len(sp.get("train_files", [])), sp.get("n_slices", {}).get("train", 0)))
    s_lines.append("[FIX]   val  : files=%d slices=%d" % (
        len(sp.get("val_files", [])), sp.get("n_slices", {}).get("val", 0)))
    s_lines.append("[FIX]   test : files=%d slices=%d" % (
        len(sp.get("test_files", [])), sp.get("n_slices", {}).get("test", 0)))
    s_lines.append("")
    summary_path = os.path.join(OUT_DIR, SUMMARY_NAME)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(s_lines) + "\n")

    report = {
        "script": "fastmri_320_fix.py",
        "started": start_str,
        "duration_s": round(duration, 2),
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "h5py": h5py.__version__, "torch": torch.__version__},
        "out_dir": OUT_DIR,
        "n_slices": extra.get("n_slices"),
        "n_files_used": extra.get("n_files_used"),
        "split": sp,
        "chunks": extra.get("chunk_rows", []),
        "verify": extra.get("verify_rows", []),
        "zf_sanity": extra.get("zf_rows", []),
        "outputs": {
            "meta": os.path.join(OUT_DIR, META_NAME),
            "summary": summary_path,
            "report": os.path.join(OUT_DIR, REPORT_NAME),
            "stdout": os.path.join(OUT_DIR, LOG_NAME),
        },
        "errors": errors,
        "warnings": warnings,
        "verdict": verdict,
    }
    report_path = os.path.join(OUT_DIR, REPORT_NAME)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True, default=json_default)
    if _LOG_FH is not None:
        try:
            _LOG_FH.close()
        except Exception:
            pass
    return 0 if not errors else 1


def main():
    global _LOG_FH
    t0 = time.time()
    errors, warnings = [], []
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, LOG_NAME)
    _LOG_FH = open(log_path, "w", encoding="utf-8")

    log("fastmri_320_fix.py (chunked repair of the 320x320 prepared dataset)")
    log("started: " + start_str)
    log("env: python=%s numpy=%s h5py=%s torch=%s" % (
        sys.version.split()[0], np.__version__, h5py.__version__, torch.__version__))
    log("out_dir=%s n_chunks=%d chunk_tag=%s" % (OUT_DIR, N_CHUNKS, CHUNK_TAG))
    log("reference split: train=%d val=%d test=%d" % (
        REF_SPLIT["train"], REF_SPLIT["val"], REF_SPLIT["test"]))

    data_dir, src = resolve_data_dir()
    if not data_dir:
        log("ERROR: fastMRI data dir not found; set env FMRI_DATA_DIR to the folder with the .h5 files")
        return _finish(errors, warnings, t0, start_str, {})
    log("data_dir=%s (source=%s)" % (data_dir, src))

    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    n_files = len(files)
    log("h5 files=%d" % n_files)
    if n_files == 0:
        log("ERROR: no .h5 files found in data dir")
        return _finish(errors, warnings, t0, start_str, {})
    if FIX_MAX_FILES and FIX_MAX_FILES < n_files:
        files = files[:FIX_MAX_FILES]
        log("WARNING: PARTIAL run FIX_MAX_FILES=%d (NOT the final dataset)" % FIX_MAX_FILES)
        warnings.append("partial")

    # ---- pass 1: convert + light alignment check + per-file normalize ----
    gt_list = []
    slice_files = []
    slice_local = []
    file_meta = []
    align_corrs = []
    for fi, path in enumerate(files):
        try:
            kspace, esc, rss, attrs = load_h5(path)
        except Exception as e:
            log("ERROR: load failed %s : %s" % (os.path.basename(path), e))
            errors.append("load:%s" % os.path.basename(path))
            continue
        raw = center_crop(ifft2c(kspace), IMG, IMG)
        if raw is None:
            log("ERROR: crop failed %s shape=%s" % (os.path.basename(path), list(kspace.shape)))
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
        mid = raw.shape[0] // 2
        corr = _pearson_corr(raw[mid], esc[mid])
        align_corrs.append(corr)
        if corr < 0.99:
            log("ERROR: alignment corr=%.4f (<0.99) file=%s slice=%d" % (
                corr, os.path.basename(path), mid))
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
        return _finish(errors, warnings, t0, start_str, {})

    gt = np.concatenate(gt_list, axis=0).astype(np.complex64)
    del gt_list
    gc.collect()
    slice_files = np.asarray(slice_files, dtype=np.int64)
    slice_local = np.asarray(slice_local, dtype=np.int64)
    N = gt.shape[0]
    n_used = len(file_meta)
    log("converted: slices=%d files=%d avg=%.1f slices/file" % (
        N, n_used, N / float(n_used)))
    log("alignment: files_checked=%d corr_min=%.6f corr_max=%.6f" % (
        len(align_corrs), min(align_corrs), max(align_corrs)))
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
    for k, got, ref in (("train", tr_s, REF_SPLIT["train"]),
                        ("val", va_s, REF_SPLIT["val"]),
                        ("test", te_s, REF_SPLIT["test"])):
        if got != ref:
            warnings.append("split_ref_%s:%d_vs_%d" % (k, got, ref))
            log("WARNING: split %s slices=%d differs from reference %d" % (k, got, ref))

    # ---- masks ----
    masks = {}
    for rate in RATES:
        for seed in MASK_SEEDS:
            key = "r%d_s%d" % (rate, seed)
            m1d, meta = build_vd_mask(IMG, rate, CTR_FRAC, seed)
            m2d = np.repeat(m1d.reshape(1, IMG), IMG, axis=0)
            masks[key] = {"mask": torch.from_numpy(m2d.astype(np.bool_)), "meta": meta}
            log("mask %s: effR=%.4f total=%d center=%d outer=%d sym=%s" % (
                key, meta["effR"], meta["total_rows"], meta["center_rows"],
                meta["outer_rows"], meta["symmetric"]))
            if abs(meta["effR"] - rate) > 1e-9:
                warnings.append("effR:%s" % key)
            if not meta["symmetric"]:
                warnings.append("sym:%s" % key)

    # ---- save gt chunks ----
    per = int(math.ceil(N / float(N_CHUNKS)))
    chunk_specs = []
    chunk_rows = []
    for ci in range(N_CHUNKS):
        start = ci * per
        end = min(N, start + per)
        name = "%s_%03d.pt" % (CHUNK_TAG, ci)
        path = os.path.join(OUT_DIR, name)
        n = end - start
        if n <= 0:
            chunk_specs.append({"name": name, "path": path, "start": start, "n": 0})
            continue
        payload = {"gt": torch.from_numpy(gt[start:end])}
        t_s = time.time()
        torch.save(payload, path)
        size_b = os.path.getsize(path)
        log("chunk %s: slices=%d size=%.2f MB (%.1fs)" % (
            name, n, size_b / 1e6, time.time() - t_s))
        chunk_specs.append({"name": name, "path": path, "start": start, "n": n})
        chunk_rows.append({"name": name, "path": path, "start": start,
                           "n": n, "size_mb": round(size_b / 1e6, 2)})
    total_chunk_n = sum(c["n"] for c in chunk_specs)
    if total_chunk_n != N:
        log("ERROR: chunk slice sum %d != N %d" % (total_chunk_n, N))
        errors.append("chunk_sum")
    del gt
    gc.collect()

    # ---- save meta ----
    split_payload = {
        "seed": SPLIT_SEED,
        "split_by": "patient_id",
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "n_files": {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
        "n_slices": {"train": tr_s, "val": va_s, "test": te_s},
    }
    meta = {
        "version": 2,
        "repair_note": "chunked save to avoid the >2GiB zip-entry corruption",
        "gt": {
            "n_slices": int(N),
            "dtype": "complex64",
            "shape": [int(N), IMG, IMG],
            "n_chunks": int(len([c for c in chunk_specs if c["n"] > 0])),
            "chunk_size_rows": per,
            "chunks": chunk_specs,
        },
        "slice_files": torch.from_numpy(slice_files),
        "slice_local": torch.from_numpy(slice_local),
        "file_meta": file_meta,
        "split": split_payload,
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
            "chunk_tag": CHUNK_TAG,
            "prepared_version": 2,
        },
    }
    meta_path = os.path.join(OUT_DIR, META_NAME)
    torch.save(meta, meta_path)
    log("meta saved: %s (%.2f MB)" % (meta_path, os.path.getsize(meta_path) / 1e6))

    # ---- verify: reload every chunk (mmap + plain) ----
    verify_rows = []
    verr = 0
    for spec in chunk_specs:
        if spec["n"] <= 0:
            continue
        row = {"name": spec["name"], "n": spec["n"], "shape": [spec["n"], IMG, IMG]}
        try:
            t1 = torch.load(spec["path"], map_location="cpu", weights_only=False, mmap=True)
            ok1, det1 = verify_tensor(t1["gt"], (spec["n"], IMG, IMG), spec["name"])
            del t1
            gc.collect()
        except Exception as e:
            ok1, det1 = False, {"error": str(e)[:120]}
        try:
            t2 = torch.load(spec["path"], map_location="cpu", weights_only=False)
            ok2, det2 = verify_tensor(t2["gt"], (spec["n"], IMG, IMG), spec["name"])
            del t2
            gc.collect()
        except Exception as e:
            ok2, det2 = False, {"error": str(e)[:120]}
        row["mmap_ok"] = ok1
        row["mmap_detail"] = det1
        row["plain_ok"] = ok2
        row["plain_detail"] = det2
        if not (ok1 and ok2):
            verr += 1
        log("verify %s: mmap=%s plain=%s %s" % (
            spec["name"], "PASS" if ok1 else "FAIL", "PASS" if ok2 else "FAIL",
            "" if (ok1 and ok2) else str(det1 if not ok1 else det2)))
        verify_rows.append(row)
    if verr:
        errors.append("chunk_verify:%d_failed" % verr)

    # ---- verify meta ----
    try:
        m2 = torch.load(meta_path, map_location="cpu", weights_only=False)
        n2 = int(m2["gt"]["n_slices"])
        if n2 != N:
            errors.append("meta_n_mismatch")
        sf = m2["slice_files"].numpy()
        tr2 = int(np.sum(np.isin(sf, np.asarray(m2["split"]["train_files"]))))
        va2 = int(np.sum(np.isin(sf, np.asarray(m2["split"]["val_files"]))))
        te2 = int(np.sum(np.isin(sf, np.asarray(m2["split"]["test_files"]))))
        log("verify meta: n_slices=%d split(train/val/test)=%d/%d/%d" % (n2, tr2, va2, te2))
        if (tr2, va2, te2) != (tr_s, va_s, te_s):
            errors.append("meta_split_mismatch")
        for key in ["r4_s42", "r4_s123", "r4_s2025", "r8_s42", "r8_s123", "r8_s2025"]:
            if key not in m2["masks"]:
                errors.append("meta_mask_missing:%s" % key)
                continue
            mk = m2["masks"][key]
            if tuple(mk["mask"].shape) != (IMG, IMG):
                errors.append("meta_mask_shape:%s" % key)
            rate = float(mk["meta"]["rate"])
            if abs(mk["meta"]["effR"] - rate) > 1e-9:
                errors.append("meta_mask_effR:%s" % key)
        del m2
        gc.collect()
        log("verify meta: OK")
    except Exception as e:
        errors.append("meta_verify:%s" % str(e)[:120])
        log("verify meta: FAIL %s" % str(e)[:120])

    # ---- sanity: zero-filled PSNR on a subset of test slices ----
    zf_rows = []
    test_idx_all = np.where(np.isin(slice_files, te_set))[0]
    n_samp = min(24, len(test_idx_all))
    step = max(1, len(test_idx_all) // 24) if len(test_idx_all) > 24 else 1
    test_samp = test_idx_all[::step][:n_samp].tolist()
    log("zf sanity: test_slices=%d sampled=%d" % (len(test_idx_all), len(test_samp)))
    for key in ["r4_s42", "r8_s42"]:
        m2d = masks[key]["mask"].numpy()
        psnrs = []
        for si in test_samp:
            ci = si // per
            off = si - ci * per
            spec = chunk_specs[ci]
            t = torch.load(spec["path"], map_location="cpu", weights_only=False, mmap=True)
            g = t["gt"].numpy()[off]
            del t
            gc.collect()
            y = m2d * fft2c(g)
            zf = ifft2c(y)
            psnrs.append(compute_psnr(np.abs(g), np.abs(zf)))
        psnrs = np.asarray(psnrs, dtype=np.float64)
        lo, hi = REF_ZF[key]
        if len(psnrs) > 0:
            log("zf sanity %s: psnr=%.2f+-%.2f (n=%d, reference %.2f-%.2f)" % (
                key, psnrs.mean(), psnrs.std(), len(psnrs), lo, hi))
        else:
            log("zf sanity %s: no test slices (skipped)" % key)
        zf_rows.append({"mask": key, "n": int(len(psnrs)),
                        "psnr_mean": round(float(psnrs.mean()), 2) if len(psnrs) else None,
                        "psnr_std": round(float(psnrs.std()), 2) if len(psnrs) else None,
                        "ref_lo": lo, "ref_hi": hi})
        if len(psnrs) and not (lo - 2.0 <= float(psnrs.mean()) <= hi + 2.0):
            warnings.append("zf_out_of_range:%s" % key)
            log("WARNING: zf sanity %s outside reference range" % key)

    log("note: old corrupt file fastmri_320_prepared.pt (2.0 GB) can be deleted after this PASS")

    return _finish(errors, warnings, t0, start_str, {
        "chunk_rows": chunk_rows,
        "verify_rows": verify_rows,
        "zf_rows": zf_rows,
        "split": split_payload,
        "n_slices": int(N),
        "n_files_used": n_used,
    })


if __name__ == "__main__":
    sys.exit(main())
# -*- coding: utf-8 -*-
"""
step5_k_scan.py —— QB-DEQ 级联深度 K 扫描（论文核心卖点图）
================================================================
读取 runs/step5_final/full/checkpoint_best.pt（与 step5_train_final.py
--eval-only 使用完全相同的权重），在全部 test 切片 x 3 个 mask 上，
用一次 forward(k_max=4) 同时得到 K=1,2,3,4 各级输出，计算逐切片幅值
PSNR / SSIM（fastMRI 官方口径：per-image 峰值归一化的 |recon| vs |gt|，
与 C.per_slice_psnr_full 完全一致）。

输出：
  1) K 扫描表（每个 mask：K1..K4 的 PSNR/SSIM mean±std）；
  2) 深度增益 K4-K1（核心卖点，要求单调递增）；
  3) 与 ZF 基线（r4_s42 ~ 25.42 dB）的对比；
  4) 可选 --deep N：把第 4 级 UNet 权重共享迭代到 N 级（fixed-point 诊断），
     报告残差 ||z_{n+1}-z_n|| 与 PSNR 收敛曲线 —— 对应隐式深度（DEQ）故事，
     诚实标注为诊断性结果（不是训练模型在 K>4 的正式成绩）；
  5) 图 step5_final_figs/fig_k_scan.png（英文标签，避免 CJK 字形缺失警告）。

用法:
  python step5_k_scan.py               # K=1..4 全 test x 3 mask（约 5-10 分钟）
  python step5_k_scan.py --deep 32     # 附加 last-stage 权重共享迭代诊断
  python step5_k_scan.py --subset 100  # 只跑前 100 个 test 切片（快速调试）

验收（诚实标注，不伪造指标）:
  PASS   : 各 mask 深度增益单调 且 主 mask K4-K1 >= 2.0 dB 且 K4 > ZF
  REVIEW : 单调 且 K4-K1 >= 1.0 dB 且 K4 > ZF - 1.0
  FAIL   : 其余
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import step5_320_ceiling as C
from step5_train_final import CascadeNet

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(HERE, "runs", "step5_final")
FIG_DIR = os.path.join(HERE, "step5_final_figs")
META_PATH = C.META_PATH
MAIN_MASK = "r4_s42"
TEST_MASKS = list(C.TEST_MASKS)      # ["r4_s42", "r4_s123", "r4_s2025"]
KS = (1, 2, 3, 4)
GAIN_PASS = 2.0    # dB，主 mask 深度增益验收线
GAIN_REVIEW = 1.0  # dB
REL_CONV = 1e-3    # 深度外推收敛判据（平均相对残差）


def log(msg):
    print("[KSCAN] %s" % msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="QB-DEQ cascade depth K-scan (honest, no fake metrics)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--subset", type=int, default=0,
                   help="只用前 N 个 test 切片（0=全部 804）")
    p.add_argument("--deep", type=int, default=0,
                   help="附加深度外推迭代次数（复用第 4 级 UNet，0=关闭）")
    p.add_argument("--deep-n", type=int, default=50,
                   help="深度外推使用的 test 切片数（默认 50）")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def find_checkpoint():
    for cand in (os.path.join(RUN_ROOT, "full", "checkpoint_best.pt"),
                 os.path.join(RUN_ROOT, "eval", "checkpoint_best.pt")):
        if os.path.exists(cand):
            return cand
    return None


def load_model(ck_path, device):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = CascadeNet(in_ch=2, base=64, K=4, eta_init=0.5).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    info = {"epoch": int(ck.get("epoch", 0)),
            "val_psnr": float(ck.get("val_psnr", -1.0)),
            "n_params": int(ck.get("n_params", 0))}
    return model, info


def scan_k(model, store, idx, mask_store, mask_key, device, batch=8):
    """一次 forward(k_max=4) 得到 K=1..4 各级，逐切片 PSNR/SSIM + ZF 基线。"""
    mask = mask_store.get(mask_key, device=device)
    model.eval()
    ssim = C.SSIMComputer()
    ps = {k: [] for k in KS}
    ss = {k: [] for k in KS}
    zf_ps, zf_ss = [], []
    with torch.no_grad():
        for s0 in range(0, len(idx), batch):
            b_idx = [int(i) for i in idx[s0:s0 + batch]]
            g = store.get_batch(b_idx, device=device)
            yb = C.fft2_t(g) * mask
            z0 = C.to_2ch(C.ifft2_t(yb))
            outs = model(z0, yb, mask, k_max=4)
            zf = C.ifft2_t(yb)
            g_np = g.detach().cpu().numpy()
            zf_np = zf.detach().cpu().numpy()
            for k in KS:
                zk = outs[k - 1]
                pv = C.per_slice_psnr_full(zk, g).detach().cpu().numpy()
                ps[k].extend(float(v) for v in pv.tolist())
                xk_np = zk.detach().cpu().numpy()
                for i in range(len(b_idx)):
                    ss[k].append(ssim.compute(np.abs(g_np[i]), np.abs(xk_np[i])))
            zp = C.per_slice_psnr_full(zf, g).detach().cpu().numpy()
            zf_ps.extend(float(v) for v in zp.tolist())
            for i in range(len(b_idx)):
                zf_ss.append(ssim.compute(np.abs(g_np[i]), np.abs(zf_np[i])))
    zf_ps = np.asarray(zf_ps, dtype=np.float64)
    zf_ss = np.asarray(zf_ss, dtype=np.float64)
    out = {"zf": {"psnr": C.stat_arr(zf_ps), "ssim": C.stat_arr(zf_ss),
                  "n": int(len(zf_ps))}}
    for k in KS:
        ps_k = np.asarray(ps[k], dtype=np.float64)
        ss_k = np.asarray(ss[k], dtype=np.float64)
        out["K%d" % k] = {"psnr": C.stat_arr(ps_k),
                          "ssim": C.stat_arr(ss_k), "n": int(len(ps_k))}
    return out


def scan_deep(model, store, idx, mask_store, mask_key, device, n_iter, batch=4):
    """权重共享迭代：n<K 用第 n 级，n>=K 复用最后一级（fixed-point 诊断）。"""
    mask = mask_store.get(mask_key, device=device)
    model.eval()
    psnr_it = [[] for _ in range(n_iter)]
    rel_it = [[] for _ in range(n_iter)]
    with torch.no_grad():
        for s0 in range(0, len(idx), batch):
            b_idx = [int(i) for i in idx[s0:s0 + batch]]
            g = store.get_batch(b_idx, device=device)
            yb = C.fft2_t(g) * mask
            z = C.to_2ch(C.ifft2_t(yb))
            for n in range(n_iter):
                n_stage = n if n < model.K else model.K - 1
                zc = C.to_c(z)
                d = model.dc(zc, yb, mask, n_stage)
                z_new = model.nets[n_stage](C.to_2ch(d)).float()
                zc_new = C.to_c(z_new)
                pv = C.per_slice_psnr_full(zc_new, g).detach().cpu().numpy()
                psnr_it[n].extend(float(v) for v in pv.tolist())
                zc_old = zc.detach().cpu()
                zc_new_cpu = zc_new.detach().cpu()
                rel = ((zc_new_cpu - zc_old).norm(dim=(-1, -2))
                       / (zc_old.norm(dim=(-1, -2)) + 1e-12))
                rel_it[n].extend(float(v) for v in rel.tolist())
                z = z_new
    means_p = [float(np.mean(psnr_it[n])) for n in range(n_iter)]
    means_r = [float(np.mean(rel_it[n])) for n in range(n_iter)]
    conv_it = None
    for n in range(n_iter):
        if means_r[n] < REL_CONV:
            conv_it = n + 1
            break
    return {"n_iter": int(n_iter), "n_slices": int(len(idx)),
            "conv_iter": conv_it, "rel_end": round(means_r[-1], 6),
            "psnr_first": round(means_p[0], 4), "psnr_end": round(means_p[-1], 4),
            "psnr_peak": round(float(max(means_p)), 4),
            "psnr_min": round(float(min(means_p)), 4),
            "monotone_after4": bool(all(
                means_p[n] >= means_p[n - 1] for n in range(4, n_iter))),
            "per_it": [{"it": n + 1, "psnr": round(means_p[n], 4),
                        "rel": round(means_r[n], 6)} for n in range(n_iter)]}


def main():
    t0 = time.time()
    args = parse_args()
    log("step5_k_scan mode=%s seed=%d batch=%d subset=%d deep=%d"
        % ("full" if args.subset == 0 else "subset",
           args.seed, args.batch, args.subset, args.deep))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device=%s torch=%s" % (dev, torch.__version__))

    if not os.path.exists(META_PATH):
        log("META NOT FOUND: %s" % META_PATH)
        return 1
    meta = torch.load(META_PATH, map_location="cpu", weights_only=False)
    store = C.ChunkStore(meta)
    mask_store = C.MaskStore(meta["masks"])
    _, _, test_idx = C.load_split(meta)
    if args.subset and args.subset > 0:
        test_idx = test_idx[:int(args.subset)]
    log("test slices=%d masks=%s" % (len(test_idx), TEST_MASKS))

    ck_path = find_checkpoint()
    if ck_path is None:
        log("checkpoint_best.pt NOT FOUND under %s" % RUN_ROOT)
        log("hint: run 'python step5_train_final.py --eval-only' first (it "
            "writes/uses the same checkpoint), or train to get a checkpoint.")
        return 1
    model, ck_info = load_model(ck_path, dev)
    log("loaded %s (epoch=%d val_psnr=%.2f n_params=%d)"
        % (ck_path, ck_info["epoch"], ck_info["val_psnr"], ck_info["n_params"]))

    scan = {}
    for mk in TEST_MASKS:
        r = scan_k(model, store, test_idx, mask_store, mk, dev,
                   batch=args.batch)
        scan[mk] = r
        e = r["K4"]["psnr"]["mean"]
        e1 = r["K1"]["psnr"]["mean"]
        z = r["zf"]["psnr"]["mean"]
        log("TEST %-10s ZF=%6.2f K1=%6.2f K2=%6.2f K3=%6.2f K4=%6.2f "
            "gain=%+5.2f dB | SSIM K1=%.4f K4=%.4f (n=%d)"
            % (mk, z, e1, r["K2"]["psnr"]["mean"], r["K3"]["psnr"]["mean"],
               e, e - e1, r["K1"]["ssim"]["mean"], r["K4"]["ssim"]["mean"],
               r["K4"]["n"]))

    # ---- 深度增益 / 单调性 / 与 ZF 对比 ----
    gains = {}
    mono_ok = True
    for mk in TEST_MASKS:
        r = scan[mk]
        seq = [r["K%d" % k]["psnr"]["mean"] for k in KS]
        gains[mk] = round(seq[-1] - seq[0], 4)
        if not all(seq[i] < seq[i + 1] for i in range(len(seq) - 1)):
            mono_ok = False
    g_main = gains[MAIN_MASK]
    zf_main = scan[MAIN_MASK]["zf"]["psnr"]["mean"]
    k4_main = scan[MAIN_MASK]["K4"]["psnr"]["mean"]
    zf_ref = float(C.REF_ZF[MAIN_MASK]["psnr"]) if hasattr(C, "REF_ZF") else zf_main
    log("depth gain K4-K1: %s | monotone=%s" % (gains, mono_ok))
    log("main mask K4=%.2f vs ZF=%.2f (ref %.2f) delta=%+.2f dB"
        % (k4_main, zf_main, zf_ref, k4_main - zf_ref))

    if mono_ok and g_main >= GAIN_PASS and k4_main > zf_main:
        verdict = "PASS"
    elif mono_ok and g_main >= GAIN_REVIEW and k4_main > zf_main - 1.0:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"
    log("VERDICT: %s (mono=%s gain_main=%+.2f dB K4=%.2f ZF=%.2f)"
        % (verdict, mono_ok, g_main, k4_main, zf_main))

    # ---- 深度外推诊断（可选） ----
    deep = None
    if args.deep and args.deep >= 5:
        d_idx = test_idx[:int(args.deep_n)]
        log("deep extrapolation: %d slices x %d iterations (weight-shared "
            "last stage, diagnostic)" % (len(d_idx), args.deep))
        deep = scan_deep(model, store, d_idx, mask_store, MAIN_MASK, dev,
                         int(args.deep), batch=4)
        log("deep: conv_iter=%s rel_end=%.2e psnr first=%.2f end=%.2f "
            "peak=%.2f min=%.2f mono_after4=%s"
            % (deep["conv_iter"], deep["rel_end"], deep["psnr_first"],
               deep["psnr_end"], deep["psnr_peak"], deep["psnr_min"],
               deep["monotone_after4"]))

    report = {
        "script": "step5_k_scan.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args), "device": str(dev), "torch": torch.__version__,
        "checkpoint": {"path": ck_path, "epoch": ck_info["epoch"],
                       "val_psnr": ck_info["val_psnr"],
                       "n_params": ck_info["n_params"]},
        "n_test_slices": int(len(test_idx)),
        "masks": scan,
        "depth_gain_db": gains, "monotone": bool(mono_ok),
        "main_mask": MAIN_MASK, "zf_ref_db": round(zf_ref, 4),
        "verdict": {"status": verdict, "monotone": bool(mono_ok),
                    "gain_main_db": round(g_main, 4),
                    "k4_main_db": round(k4_main, 4),
                    "zf_main_db": round(zf_main, 4)},
        "deep": deep,
    }
    rp = os.path.join(HERE, "step5_k_scan_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log("report written: %s" % rp)

    # ---- 摘要 txt ----
    lines = []
    lines.append("STEP5_K_SCAN SUMMARY  (%s)" % report["timestamp"])
    lines.append("=" * 70)
    lines.append("checkpoint: %s (epoch=%d val_psnr=%.2f)"
                 % (ck_path, ck_info["epoch"], ck_info["val_psnr"]))
    lines.append("test slices=%d  masks=%s" % (len(test_idx), TEST_MASKS))
    lines.append("")
    lines.append("mask        | ZF psnr | K1    | K2    | K3    | K4    | gain K4-K1")
    for mk in TEST_MASKS:
        r = scan[mk]
        lines.append("%-11s | %6.2f  | %5.2f | %5.2f | %5.2f | %5.2f | %+5.2f dB"
                     % (mk, r["zf"]["psnr"]["mean"],
                        r["K1"]["psnr"]["mean"], r["K2"]["psnr"]["mean"],
                        r["K3"]["psnr"]["mean"], r["K4"]["psnr"]["mean"],
                        gains[mk]))
    lines.append("")
    lines.append("mask        | ZF ssim | K1     | K2     | K3     | K4")
    for mk in TEST_MASKS:
        r = scan[mk]
        lines.append("%-11s | %6.4f | %6.4f | %6.4f | %6.4f | %6.4f"
                     % (mk, r["zf"]["ssim"]["mean"],
                        r["K1"]["ssim"]["mean"], r["K2"]["ssim"]["mean"],
                        r["K3"]["ssim"]["mean"], r["K4"]["ssim"]["mean"]))
    lines.append("")
    lines.append("monotone depth gain: %s | main mask K4-K1 = %+.2f dB"
                 % (mono_ok, g_main))
    lines.append("main mask K4 = %.2f dB vs ZF = %.2f dB (ref %.2f) | "
                 "delta = %+.2f dB" % (k4_main, zf_main, zf_ref, k4_main - zf_ref))
    if deep:
        lines.append("")
        lines.append("DEEP EXTRAPOLATION (weight-shared last stage, diagnostic):")
        lines.append("  n_iter=%d slices=%d conv_iter=%s rel_end=%.2e"
                     % (deep["n_iter"], deep["n_slices"], deep["conv_iter"],
                        deep["rel_end"]))
        lines.append("  psnr first=%.2f end=%.2f peak=%.2f min=%.2f mono_after4=%s"
                     % (deep["psnr_first"], deep["psnr_end"], deep["psnr_peak"],
                        deep["psnr_min"], deep["monotone_after4"]))
    lines.append("")
    lines.append("VERDICT: %s" % verdict)
    sp = os.path.join(HERE, "step5_k_scan_summary.txt")
    with open(sp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("summary written: %s" % sp)

    # ---- 图（英文标签） ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_panels = 3 if deep else 2
        fig, axs = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.5))
        axs = np.atleast_1d(axs)
        colors = {"r4_s42": "#1f77b4", "r4_s123": "#ff7f0e",
                  "r4_s2025": "#2ca02c"}
        ax = axs[0]
        for mk in TEST_MASKS:
            r = scan[mk]
            ax.plot(list(KS), [r["K%d" % k]["psnr"]["mean"] for k in KS],
                    marker="o", label=mk, color=colors.get(mk))
            ax.axhline(r["zf"]["psnr"]["mean"], ls="--", lw=1,
                       color=colors.get(mk), alpha=0.5)
        ax.set_xlabel("cascade depth K")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("magnitude PSNR vs cascade depth (test)")
        ax.legend()
        ax.grid(alpha=0.3)
        ax = axs[1]
        for mk in TEST_MASKS:
            r = scan[mk]
            ax.plot(list(KS), [r["K%d" % k]["ssim"]["mean"] for k in KS],
                    marker="s", label=mk, color=colors.get(mk))
        ax.set_xlabel("cascade depth K")
        ax.set_ylabel("SSIM")
        ax.set_title("SSIM vs cascade depth (test)")
        ax.legend()
        ax.grid(alpha=0.3)
        if deep:
            ax = axs[2]
            ax.plot(range(1, deep["n_iter"] + 1), [p["psnr"] for p in deep["per_it"]],
                    marker=".", lw=1)
            ax.set_xlabel("iteration n (last stage weight-shared)")
            ax.set_ylabel("PSNR (dB)")
            ax.set_title("deep extrapolation diagnostic (%s)" % MAIN_MASK)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        if not os.path.isdir(FIG_DIR):
            os.makedirs(FIG_DIR, exist_ok=True)
        fp = os.path.join(FIG_DIR, "fig_k_scan.png")
        fig.savefig(fp, dpi=130, bbox_inches="tight")
        log("FIG saved %s" % fp)
    except Exception as e:
        log("WARN figure failed: %s: %s" % (type(e).__name__, str(e)))

    log("done in %.1fs | verdict: %s" % (time.time() - t0, verdict))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
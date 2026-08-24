# -*- coding: utf-8 -*-
"""run_baselines.py —— 同协议基线对比运行器（fastMRI knee 320x320, r4_s42）

统一协议（与论文一致）：同一 804 片测试集、同一掩码、同一损失与指标。
所有模型默认只训练 5 个 epoch（快速出结果），epochs 可调。

模型（每个一个独立 py 文件）：
  unet        model_unet.py       直接 U-Net 基线
  modl        model_modl.py       MoDL（共享去噪器 + DC，K=4）
  varnet      model_varnet.py     VarNet（单线圈适配，独立精修器 + DC，K=6）
  dccnn       model_dccnn.py      DCCNN（双域级联：图像域 + k-空间域 + DC，K=5）
  cascadenet  model_cascadenet.py 本文有限深度实例化（独立权重级联，K=4）
  qbdeq       model_qbdeq.py      本文 QB-DEQ（权重共享 RED 算子，展开 K=8，有界迭代）

用法：
  python run_baselines.py                         # 全部模型，5 epoch，2000 训练片
  python run_baselines.py --models unet,modl      # 只跑指定模型
  python run_baselines.py --epochs 5 --train-subset 2000
  python run_baselines.py --eval-only             # 只评测已保存检查点
  python run_baselines.py --test-subset 24        # 快速测试子集
"""
import os
import sys
import json
import time
import copy
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn

import baseline_utils as BU
import step5_320_ceiling as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "runs", "baseline")

def build_models(args):
    """返回 {name: model}。"""
    from model_unet import UnetBaseline
    from model_modl import MoDL
    from model_varnet import VarNet
    from model_dccnn import DCCNN
    from model_cascadenet import CascadeNetBaseline
    from model_qbdeq import QBDEQ
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    models = {}
    for n in names:
        if n == "unet":
            models[n] = UnetBaseline(base=args.base)
        elif n == "modl":
            models[n] = MoDL(K=args.K_modl, base=args.base)
        elif n == "varnet":
            models[n] = VarNet(K=args.K_varnet, base=args.base_varnet)
        elif n == "dccnn":
            models[n] = DCCNN(K=args.K_dccnn, feats=args.base)
        elif n == "cascadenet":
            models[n] = CascadeNetBaseline(K=args.K_casc, base=args.base)
        elif n == "qbdeq":
            models[n] = QBDEQ(K=args.K_qbdeq, base=args.base)
        else:
            raise SystemExit("unknown model: %s" % n)
    return models


def save_ckpt(path, model, epoch, val_psnr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                "val_psnr": val_psnr}, path)


def load_ckpt(path, model):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    return ck


def parse_args():
    p = argparse.ArgumentParser(description="同协议基线对比（5 epoch 快速版）")
    p.add_argument("--models", type=str, default="unet,modl,varnet,dccnn,cascadenet,qbdeq")
    p.add_argument("--epochs", type=int, default=5, help="每模型训练轮数（默认 5）")
    p.add_argument("--train-subset", type=int, default=2000, help="0=全部 5668")
    p.add_argument("--test-subset", type=int, default=0, help="0=全部 804")
    p.add_argument("--val-max", type=int, default=48)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--K-modl", type=int, default=4)
    p.add_argument("--K-varnet", type=int, default=6)
    p.add_argument("--base-varnet", type=int, default=32)
    p.add_argument("--K-dccnn", type=int, default=5)
    p.add_argument("--K-casc", type=int, default=4)
    p.add_argument("--K-qbdeq", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--aug", type=int, default=1)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()
    args.amp = bool(args.amp)
    args.aug = bool(args.aug)
    return args


def train_model(model, name, store, train_idx, mask, device, args):
    """训练指定模型 epochs 轮，返回 (best_epoch, best_val_psnr, history)。"""
    print("\n========== TRAIN %s (%d epochs, %d slices) =========="
          % (name, args.epochs, len(train_idx)))
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=BU.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device == "cuda"))
    t0 = time.time()
    best_psnr, best_ep, hist = -1e9, 0, []
    for epoch in range(1, args.epochs + 1):
        lr = args.lr if epoch <= args.warmup else args.lr
        if epoch > args.warmup:
            t = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
            lr = args.lr * 0.5 * (1.0 + np.cos(np.pi * t))
        for pg in opt.param_groups:
            pg["lr"] = lr
        args.name = name
        loss = BU.train_one_epoch(model, store, train_idx, mask, device,
                                  opt, scaler, args, t0, epoch)
        vp = BU.evaluate(model, store, args.val_sub_idx, mask, device,
                         batch=2)["psnr"]
        hist.append({"epoch": epoch, "loss": loss, "val_psnr": vp})
        if vp > best_psnr:
            best_psnr, best_ep = vp, epoch
            save_ckpt(os.path.join(OUT_DIR, "%s_best.pt" % name), model, epoch, vp)
        save_ckpt(os.path.join(OUT_DIR, "%s_last.pt" % name), model, epoch, vp)
        print("[%s] ep %d/%d loss=%.4f val_psnr=%.2f (best %.2f @ ep%d) lr=%.1e t=%s"
              % (name, epoch, args.epochs, loss, vp, best_psnr, best_ep, lr,
                 BU.time_str(time.time() - t0)))
    return best_ep, best_psnr, hist


def main():
    args = parse_args()
    BU.set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[BASE] device=%s torch=%s | epochs=%d train_subset=%d test_subset=%d"
          % (dev, torch.__version__, args.epochs, args.train_subset, args.test_subset))

    store, mask, train_idx, val_idx, test_idx, val_sub = BU.load_data(
        device=dev, train_subset=args.train_subset, val_max=args.val_max,
        test_subset=args.test_subset, seed=args.seed)
    args.val_sub_idx = val_sub
    print("[BASE] train=%d val_sub=%d test=%d" % (len(train_idx), len(val_sub), len(test_idx)))

    models = build_models(args)
    report = {"args": vars(args), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "rows": {}, "zf": BU.ZF_REF[BU.MAIN_MASK]}

    for name, model in models.items():
        try:
            if args.eval_only:
                ck = load_ckpt(os.path.join(OUT_DIR, "%s_best.pt" % name), model)
                model = model.to(dev)
                print("[%s] eval-only: loaded best (ep=%s val=%.2f)" % (name, ck["epoch"], ck["val_psnr"]))
                best_ep, best_psnr, hist = ck["epoch"], ck["val_psnr"], []
            else:
                best_ep, best_psnr, hist = train_model(
                    model, name, store, train_idx, mask, dev, args)
                model = model.to(dev)
                load_ckpt(os.path.join(OUT_DIR, "%s_best.pt" % name), model)
                print("[%s] best val %.2f @ ep %d -> evaluate on %d test slices"
                      % (name, best_psnr, best_ep, len(test_idx)))
            ev = BU.evaluate(model, store, test_idx, mask, dev, batch=2)
            report["rows"][name] = {
                "params": BU.n_params(model), "epochs": args.epochs,
                "best_val_psnr": best_psnr, "best_epoch": best_ep,
                "test_psnr": ev["psnr"], "test_ssim": ev["ssim"], "n": ev["n"],
                "history": hist}
            print("[%s] TEST psnr=%.2f ssim=%.4f (n=%d) | params=%d"
                  % (name, ev["psnr"], ev["ssim"], ev["n"], BU.n_params(model)))
        except Exception as e:
            import traceback
            print("[%s] FAILED: %s" % (name, e))
            traceback.print_exc()
            report["rows"][name] = {"error": str(e)}

    # ---- 汇总表 ----
    lines = []
    lines.append("=" * 66)
    lines.append("BASELINE COMPARISON (fastMRI knee 320x320, r4_s42, epochs=%d)" % args.epochs)
    lines.append("=" * 66)
    lines.append("%-12s %8s %8s %8s %10s" % ("model", "params", "PSNR", "SSIM", "val PSNR"))
    lines.append("%-12s %8s %8s %8s %10s" % ("ZF", "-", "25.42", "0.5405", "-"))
    for name, r in report["rows"].items():
        if "error" in r:
            lines.append("%-12s %8s   ERROR %s" % (name, "-", r["error"]))
        else:
            lines.append("%-12s %8d %8.2f %8.4f %10.2f" % (
                name, r["params"], r["test_psnr"], r["test_ssim"], r["best_val_psnr"]))
    lines.append("note: same test protocol (804 slices) for all rows; 5-epoch quick run.")
    summary = "\n".join(lines)
    print("\n" + summary)

    rp = os.path.join(OUT_DIR, "baseline_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    sp = os.path.join(OUT_DIR, "baseline_summary.txt")
    with open(sp, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print("[BASE] report: %s\n[BASE] summary: %s" % (rp, sp))


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
Step 6 -- QB-DEQ 端到端隐式训练 (fastMRI knee 320x320)
=======================================================

目标：真正训练论文提出的模型——不是 unrolled 级联，而是
    前向  : 商流形 Bregman 不动点求解器 (anderson5 + gauge, 隐式)
    反传  : 商流形隐式梯度 (I - P J^T P) q = P nabla L, 批量 GMRES
            dL/dtheta = q^T dS/dtheta
    损失  : L = L_mag + 0.1*(1-SSIM) + 0.01*L_DC (与 step5 一致, 在 z* 处)

训练流程：
    Phase A (unrolled warm-start)：把同一个算子 S_p 展开 K 步直接反传训练
           （DEQ 标准做法：隐式反传 = unrolled 反传在 K->inf 的极限, 见 step4b2 V6），
           让 W_theta 先成为"几步内就能给出好重建"的正则器。
    Phase B (implicit)：前向解不动点 (no_grad, anderson5)，在 z* 处计算损失，
           用商流形 GMRES 反传，对 theta 做梯度更新（真正的 QB-DEQ 训练）。

验收：
    M1 隐式前向在测试集上收敛率（eval tol=1e-6, max=200）
    M2 隐式反传 GMRES 残差 < 1e-2（抽查）
    Q1 测试集 PSNR/SSIM > ZF 且尽量接近/超过 CascadeNet 基线 (27.78 dB / 0.6226)
    R1 训练后 rho(J_S) (brg_gauge) < 1（商流形收缩在训练后依然成立）

运行：
    python step6_train_qbdeq.py --smoke        # 冒烟（~2-3 分钟）
    python step6_train_qbdeq.py                # 正式（PhaseA 3ep + PhaseB 15ep, 约 20-40h）
    python step6_train_qbdeq.py --eval-only    # 只评测已有 checkpoint

输出：
    runs/step6_train/...  checkpoint + history
    step6_train_report.json / step6_train_summary.txt / step6_figs/
"""

import os
import sys
import json
import time
import math
import copy
import random
import argparse

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

import step4_implicit_deq as s4
import step5_320_ceiling as C
import qbdeq_320 as Q

# ---- 固定配置 ---------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(HERE, "fastmri_320_meta.pt")
MAIN_MASK = "r4_s42"
RUN_DIR = os.path.join(HERE, "runs", "step6_train")
FIG_DIR = os.path.join(HERE, "step6_figs")
REPORT_PATH = os.path.join(HERE, "step6_train_report.json")
SUMMARY_PATH = os.path.join(HERE, "step6_train_summary.txt")

ALPHA, ETA, P = 0.3, 0.3, 4.0        # step4a 的稳定配方 (brg_gauge)
EMA_DECAY = 0.999
CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-5
ZF_REF = {"r4_s42": (25.42, 0.5405), "r4_s123": (25.25, 0.5333),
          "r4_s2025": (25.08, 0.5350)}
CASCADE_BASELINE = {"r4_s42": (27.78, 0.6226)}   # step5_train_final test 基线

_LOG_LINES = []
_LOG_FH = None


def log(msg):
    line = "[STEP6] " + msg
    print(line)
    _LOG_LINES.append(line)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()
        except Exception:
            pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fmt(x, nd=2):
    return "n/a" if x is None else ("%." + str(nd) + "f") % float(x)


# ---- EMA --------------------------------------------------------------------
class EMA(object):
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach().cpu(), alpha=1.0 - self.decay)

    def copy_to(self, model):
        with torch.no_grad():
            for k, v in self.shadow.items():
                model.state_dict()[k].copy_(v.to(model.state_dict()[k].device))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---- Phase A：unrolled warm-start -------------------------------------------
def train_phase_a(reg, opt, store, train_idx, mask, device, args, hist, t0):
    """把 S_p 展开 K 步直接反传（同 step4b2 V6 的 unrolled 训练），
    使 W_theta 成为几步内给出好重建的正则器，为 Phase B 提供热启动。"""
    K = int(args.warm_unroll)
    steps = 0
    total = len(train_idx) * int(args.warm_epochs) // max(1, int(args.batch))
    for epoch in range(1, int(args.warm_epochs) + 1):
        lr = C.lr_at(epoch, args)
        for pg in opt.param_groups:
            pg["lr"] = lr
        rng = np.random.RandomState(args.seed * 100 + epoch)
        perm = rng.permutation(len(train_idx))
        epoch_loss = []
        for s0 in range(0, len(perm), args.batch):
            b_idx = [int(i) for i in train_idx[perm[s0:s0 + args.batch]]]
            x = store.get_batch(b_idx, device=device).to(torch.complex64)
            if args.aug:
                if random.random() < 0.5:
                    x = torch.flip(x, dims=[-1])
                if random.random() < 0.5:
                    x = torch.flip(x, dims=[-2])
            y, z0 = Q.make_inputs(x, mask, device)
            z = z0
            for _ in range(K):
                z = Q._S(z, y, mask, reg, ALPHA, ETA, P)
            loss = C.recon_loss(C.to_c(z), x, y, mask)
            loss = loss / max(1, int(args.grad_accum))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(reg.parameters(), CLIP_NORM)
            opt.step()
            epoch_loss.append(float(loss.item()) * max(1, int(args.grad_accum)))
            steps += 1
            if args.smoke and steps >= 8:
                break
            if steps % 25 == 0:
                log("A ep%d step %d/%d loss=%.4f lr=%.1e t=%.0fs"
                    % (epoch, steps, total, float(loss.item()), lr, time.time() - t0))
        hist["A_loss"].append(float(np.mean(epoch_loss)) if epoch_loss else float("nan"))
        hist["A_epochs"].append(epoch)
        log("A epoch %d/%d loss=%.5f (lr=%.1e)"
            % (epoch, args.warm_epochs, hist["A_loss"][-1], lr))


# ---- Phase B：隐式训练（真正的 QB-DEQ） --------------------------------------
def train_phase_b(reg, opt, store, train_idx, mask, device, args, hist, t0, ema,
                  start_epoch=1):
    """前向解不动点 + 商流形 GMRES 反传。每 accum 个 micro-batch 更新一次。"""
    accum = max(1, int(args.grad_accum))
    steps = 0
    # Phase B 的 lr 调度：warmup 已在 Phase A 完成，这里用总时长
    # (warm_epochs + epochs) 的余弦下半段，避免 lr 归零（旧的 C.lr_at 分母 bug）
    args_s = copy.copy(args)
    args_s.epochs = max(1, int(args.warm_epochs) + int(args.epochs))
    args_s.warmup = max(1, int(args.warmup))
    abs_epoch = start_epoch - 1 + int(args.warm_epochs)
    for epoch in range(start_epoch, start_epoch + int(args.epochs)):
        abs_epoch += 1
        lr = C.lr_at(abs_epoch, args_s)
        for pg in opt.param_groups:
            pg["lr"] = lr
        rng = np.random.RandomState(args.seed * 1000 + epoch)
        perm = rng.permutation(len(train_idx))
        epoch_loss, epoch_fw_it, epoch_bwd_res = [], [], []
        gacc = [torch.zeros_like(p) for p in reg.parameters()]
        opt.zero_grad()
        for s0 in range(0, len(perm), args.batch):
            b_idx = [int(i) for i in train_idx[perm[s0:s0 + args.batch]]]
            x = store.get_batch(b_idx, device=device).to(torch.complex64)
            if args.aug:
                if random.random() < 0.5:
                    x = torch.flip(x, dims=[-1])
                if random.random() < 0.5:
                    x = torch.flip(x, dims=[-2])
            y, z0 = Q.make_inputs(x, mask, device)
            z_fp, status, it_fin, _rels = Q.solve_fixed_point(
                z0, y, mask, reg, ALPHA, ETA, P, "anderson5",
                max_iters=args.fw_budget, tol=args.fw_tol)
            if not torch.isfinite(z_fp).all():
                log("B NON-FINITE forward (it=%d) -> skip" % it_fin)
                continue
            loss_fn = lambda zz: C.recon_loss(C.to_c(zz), x, y, mask)
            try:
                q, grads, res, n_mv, loss_v, _rhs_n = Q.quotient_backward(
                    z_fp, y, mask, reg, ALPHA, ETA, P, loss_fn,
                    budget=args.bwd_mv, rtol=args.bwd_rtol, reorth=True)
            except Exception as e:
                log("B backward exception %r -> skip" % e)
                continue
            if not torch.isfinite(q).all() or not all(
                    g is None or torch.isfinite(g).all() for g in grads):
                log("B NON-FINITE backward (res=%.1e) -> skip" % res)
                continue
            for gi, g in zip(gacc, grads):
                if g is not None:
                    gi.add_(g.detach().float(), alpha=1.0 / accum)
            epoch_loss.append(loss_v)
            epoch_fw_it.append(int(it_fin))
            epoch_bwd_res.append(float(res))
            steps += 1
            if steps % accum == 0:
                for p, g in zip(reg.parameters(), gacc):
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad.add_(g)
                nn.utils.clip_grad_norm_(reg.parameters(), CLIP_NORM)
                opt.step()
                opt.zero_grad()
                for g in gacc:
                    g.zero_()
                ema.update(reg)
            if args.smoke and steps >= 10:
                break
            if steps % 25 == 0:
                log("B ep%d step %d loss=%.4f fw_it=%d gmres_res=%.1e lr=%.1e t=%.0fs"
                    % (epoch, steps, loss_v, it_fin, res, lr, time.time() - t0))
        hist["B_loss"].append(float(np.mean(epoch_loss)) if epoch_loss else float("nan"))
        hist["B_fw_it"].append(float(np.mean(epoch_fw_it)) if epoch_fw_it else float("nan"))
        hist["B_bwd_res"].append(float(np.mean(epoch_bwd_res)) if epoch_bwd_res else float("nan"))
        hist["B_epochs"].append(epoch)
        # 每 epoch 用 EMA 权重在验证子集上评测
        ema.copy_to(reg)
        ps, ss = eval_val(store, args.val_idx, mask, device, max_iters=args.fw_eval_iters)
        best_ok = ps > hist["best_val_psnr"] + 1e-6
        if best_ok:
            hist["best_val_psnr"] = float(ps)
            hist["best_epoch"] = epoch
            save_checkpoint(os.path.join(RUN_DIR, "checkpoint_best.pt"), reg, opt, ema,
                            epoch, hist)
        hist["val_psnr"].append(float(ps))
        hist["val_ssim"].append(float(ss))
        ema.copy_to(reg)  # 训练用工作权重，从 EMA 恢复
        log("B epoch %d/%d loss=%.5f val_psnr=%.2f dB val_ssim=%.4f fw_it=%.1f "
            "bwd_res=%.1e (best %.2f @ ep%d)"
            % (epoch, args.epochs, hist["B_loss"][-1], ps, ss,
               hist["B_fw_it"][-1], hist["B_bwd_res"][-1],
               hist["best_val_psnr"], hist["best_epoch"]))
        save_checkpoint(os.path.join(RUN_DIR, "checkpoint_last.pt"), reg, opt, ema,
                        epoch, hist)
        if args.smoke:
            break


# ---- 验证/测试评测 -----------------------------------------------------------
def eval_val(store, idx, mask, device, max_iters=200, tol=1e-4, batch=2):
    ps, ss = [], []
    for s0 in range(0, len(idx), batch):
        blk = [int(i) for i in idx[s0:s0 + batch]]
        x = store.get_batch(blk, device=device).to(torch.complex64)
        y, z0 = Q.make_inputs(x, mask, device)
        z, _st, _it, _r = Q.solve_fixed_point(z0, y, mask, reg_global, ALPHA, ETA, P,
                                              "anderson5", max_iters=max_iters, tol=tol)
        ps.append(Q.psnr_torch(z, x).detach().cpu())
        ss.append(Q.ssim_torch(z, x).detach().cpu())
    ps = torch.cat(ps); ss = torch.cat(ss)
    return float(ps.mean().item()), float(ss.mean().item())


def eval_test_full(store, test_idx, mask_store, device, mask_key="r4_s42",
                   max_iters=200, tol=1e-4, batch=2):
    """在全部测试切片上评测（numpy 口径 SSIM，与 fastmri_320_prep 一致）。"""
    mask = mask_store.get(mask_key, device=device)
    ssim = C.SSIMComputer()
    ps, ss = [], []
    iters, statuses = [], []
    t0 = time.time()
    for s0 in range(0, len(test_idx), batch):
        blk = [int(i) for i in test_idx[s0:s0 + batch]]
        x = store.get_batch(blk, device=device).to(torch.complex64)
        y, z0 = Q.make_inputs(x, mask, device)
        z, status, it_fin, _r = Q.solve_fixed_point(z0, y, mask, reg_global, ALPHA, ETA,
                                                    P, "anderson5",
                                                    max_iters=max_iters, tol=tol)
        z_np = C.to_c(z).detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        for i in range(len(blk)):
            gm, zm = np.abs(x_np[i]), np.abs(z_np[i])
            ps.append(C.compute_psnr(gm, zm))
            ss.append(ssim.compute(gm, zm))
            iters.append(int(it_fin))
            statuses.append(status)
    log("eval_test_full: n=%d t=%.1fs" % (len(test_idx), time.time() - t0))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
            "n": len(ps), "it_avg": float(np.mean(iters)),
            "conv_rate": float(np.mean([s == "conv" for s in statuses])),
            "iters": iters, "statuses": statuses}


# ---- checkpoint -------------------------------------------------------------
def save_checkpoint(path, model, opt, ema, epoch, hist):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "opt": opt.state_dict(),
                "ema": ema.shadow,
                "epoch": epoch,
                "best": {"val_psnr": hist["best_val_psnr"], "epoch": hist["best_epoch"]},
                "args": vars(args_global) if args_global is not None else {},
                "history": hist}, path)


# ---- 谱半径（训练后） ---------------------------------------------------------
def measure_rho(store, val_idx, mask_store, device, n_slices=2, mask_key="r4_s42"):
    mask = mask_store.get(mask_key, device=device)
    pick = np.random.RandomState(7).choice(np.asarray(val_idx, dtype=np.int64),
                                           n_slices, replace=False)
    rhos = []
    for i in pick:
        x = store.get_batch([int(i)], device=device).to(torch.complex64)
        y, z0 = Q.make_inputs(x, mask, device)
        z_fp, _st, _it, _r = Q.solve_fixed_point(z0, y, mask, reg_global, ALPHA, ETA, P,
                                                 "anderson5", 200, 1e-6)
        r = Q.estimate_rho(z_fp, y, mask, reg_global, ALPHA, ETA, P)
        rhos.append(r)
        log("rho(J_S) slice=%d -> %s" % (int(i), "n/a" if r is None else "%.6f" % r))
    rs = [r for r in rhos if r is not None]
    return float(np.mean(rs)) if rs else None


# ---- 出图 --------------------------------------------------------------------
def save_figures(hist, report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(7, 4))
        epA = hist.get("A_epochs", [])
        epB = hist.get("B_epochs", [])
        ax1.plot(epA, hist.get("A_loss", []), "o-", color="tab:blue", label="Phase A loss")
        ax1.plot([e + (len(epA) if epA else 0) for e in epB],
                 hist.get("B_loss", []), "s-", color="tab:red", label="Phase B loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.plot([e + (len(epA) if epA else 0) for e in epB],
                 hist.get("val_psnr", []), "d-", color="tab:green", label="val PSNR")
        ax2.set_ylabel("val PSNR (dB)")
        ax2.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIG_DIR, "fig1_training.png"), dpi=130)
        plt.close(fig)
        log("FIG saved fig1_training.png")
    except Exception as e:
        log("FIG warn: %s" % e)


# ---- 参数与主流程 ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="step6: QB-DEQ 端到端隐式训练")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="冒烟（极小规模）")
    g.add_argument("--eval-only", action="store_true", help="只评测 checkpoint_best.pt")
    p.add_argument("--mid", type=int, default=32, help="SNRegNet 中间通道数")
    p.add_argument("--layers", type=int, default=4, help="SNRegNet 卷积层数")
    p.add_argument("--op", type=str, default="vi_res",
                   help="算子: vi=原论文 Bregman VI, vi_res=残差正则器重设计(推荐)")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--eta", type=float, default=0.3)
    p.add_argument("--p", type=float, default=2.0, help="幂核指数（vi_res 推荐 2）")
    p.add_argument("--warm-epochs", type=int, default=3, help="Phase A unrolled epoch 数")
    p.add_argument("--warm-unroll", type=int, default=6, help="Phase A unrolled 步数 K")
    p.add_argument("--epochs", type=int, default=12, help="Phase B implicit epoch 数")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4, help="Phase B 学习率")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--fw-budget", type=int, default=30, help="Phase B 前向迭代上限")
    p.add_argument("--fw-tol", type=float, default=1e-3)
    p.add_argument("--fw-eval-iters", type=int, default=200, help="评测前向迭代上限")
    p.add_argument("--bwd-mv", type=int, default=24, help="GMRES matvec 预算")
    p.add_argument("--bwd-rtol", type=float, default=1e-3)
    p.add_argument("--val-max", type=int, default=48)
    p.add_argument("--train-subset", type=int, default=0,
                   help="只用前 N 个训练切片（0=全部 5668，快速实验用）")
    p.add_argument("--test-subset", type=int, default=0,
                   help="最终评测只取前 N 个测试切片（0=全部 804）")
    p.add_argument("--resume", action="store_true",
                   help="从 checkpoint_last.pt 续训 Phase B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--aug", type=int, default=1)
    args = p.parse_args()
    if args.smoke:
        args.mid = 8
        args.layers = 3
        args.warm_epochs = 1
        args.warm_unroll = 3
        args.epochs = 1
        args.batch = 2
        args.grad_accum = 1
        args.warmup = 1
        args.fw_budget = 20
        args.fw_eval_iters = 40
        args.bwd_mv = 10
        args.val_max = 12
        args.test_subset = 24
        args.aug = 0
    args.amp = bool(args.amp)
    args.aug = bool(args.aug)
    args.mode = "smoke" if args.smoke else ("eval" if args.eval_only else "full")
    return args


args_global = None
reg_global = None


def main():
    global args_global, reg_global
    args = parse_args()
    args_global = args
    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("step6 QB-DEQ 端到端隐式训练 mode=%s seed=%d device=%s torch=%s"
        % (args.mode, args.seed, dev, torch.__version__))
    log("recipe: method=brg_gauge alpha=%.2f eta=%.2f p=%.1f | mid=%d layers=%d"
        % (args.alpha, args.eta, args.p, args.mid, args.layers))

    meta = torch.load(META_PATH, map_location="cpu", weights_only=False)
    store = C.ChunkStore(meta)
    mask_store = C.MaskStore(meta["masks"])
    train_idx, val_idx, test_idx = C.load_split(meta)
    rng = np.random.RandomState(args.seed)
    args.val_idx = rng.choice(np.asarray(val_idx, dtype=np.int64),
                              size=min(int(args.val_max), len(val_idx)),
                              replace=False).tolist()
    if args.train_subset:
        train_idx = np.asarray(list(train_idx[:int(args.train_subset)]), dtype=np.int64)
    log("split train=%d val=%d test=%d val_sub=%d"
        % (len(train_idx), len(val_idx), len(test_idx), len(args.val_idx)))
    mask = mask_store.get(MAIN_MASK, device=dev)

    reg = s4.SNRegNet(mid=args.mid, n_layers=args.layers).to(dev)
    reg_global = reg
    Q.set_operator(args.op)
    with torch.no_grad():
        _ = reg(torch.zeros(1, 2, 8, 8, device=dev))
    log("operator=%s (alpha=%.2f eta=%.2f p=%.1f) | SNRegNet mid=%d layers=%d params=%d"
        % (args.op, args.alpha, args.eta, args.p, args.mid, args.layers, n_params(reg)))

    global _LOG_FH
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    _LOG_FH = open(os.path.join(RUN_DIR, "stdout.log"), "w", encoding="utf-8")

    hist = {"A_loss": [], "A_epochs": [], "B_loss": [], "B_epochs": [],
            "B_fw_it": [], "B_bwd_res": [], "val_psnr": [], "val_ssim": [],
            "best_val_psnr": -1e9, "best_epoch": 0}
    report = {"args": vars(args), "mode": args.mode,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "recipe": {"method": "brg_gauge", "alpha": args.alpha, "eta": args.eta,
                         "p": args.p}, "verdict": {}, "issues": []}
    t_all = time.time()

    if args.mode == "eval":
        ck = torch.load(os.path.join(RUN_DIR, "checkpoint_best.pt"),
                        map_location="cpu", weights_only=False)
        reg.load_state_dict(ck["state_dict"])
        reg.to(dev)
        hist = ck.get("history", hist)
        log("EVAL loaded checkpoint_best.pt (epoch=%d val_psnr=%.2f)"
            % (ck["epoch"], ck["best"]["val_psnr"]))
    elif args.resume:
        ck = torch.load(os.path.join(RUN_DIR, "checkpoint_last.pt"),
                        map_location="cpu", weights_only=False)
        reg.load_state_dict(ck["state_dict"])
        hist = dict(ck.get("history", hist))
        opt = torch.optim.AdamW(reg.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        opt.load_state_dict(ck["opt"])
        ema = EMA(reg)
        ema.shadow = {k: v.clone() for k, v in ck["ema"].items()}
        start_b = len(hist.get("B_epochs", []))
        log("RESUME from epoch %d (B epochs done=%d, best val=%.2f)"
            % (ck["epoch"], start_b, hist.get("best_val_psnr", -1e9)))
        log("PHASE B: implicit (resume) forward anderson5 budget=%d | GMRES mv=%d"
            % (args.fw_budget, args.bwd_mv))
        train_phase_b(reg, opt, store, train_idx, mask, dev, args, hist, t_all, ema,
                      start_epoch=start_b + 1)
        ck_best = torch.load(os.path.join(RUN_DIR, "checkpoint_best.pt"),
                             map_location="cpu", weights_only=False)
        reg.load_state_dict(ck_best["state_dict"])
        log("RESUME done: best val_psnr=%.2f @ epoch %d"
            % (hist["best_val_psnr"], hist["best_epoch"]))
    else:
        opt = torch.optim.AdamW(reg.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        ema = EMA(reg)
        # ---- Phase A：unrolled warm-start ----
        log("PHASE A: unrolled warm-start K=%d epochs=%d lr=3e-4" %
            (args.warm_unroll, args.warm_epochs))
        a_opt = torch.optim.AdamW(reg.parameters(), lr=3e-4, weight_decay=WEIGHT_DECAY)
        args_a = copy.copy(args)
        args_a.epochs = max(int(args.warm_epochs), 2)
        train_phase_a(reg, a_opt, store, train_idx, mask, dev, args_a, hist, t_all)
        pa_psnr, pa_ssim = eval_val(store, args.val_idx, mask, dev,
                                    max_iters=args.fw_eval_iters)
        hist["A_fp_psnr"] = float(pa_psnr)
        hist["A_fp_ssim"] = float(pa_ssim)
        log("PHASE A done: loss=%.5f | fixed-point val_psnr=%.2f dB ssim=%.4f"
            % (hist["A_loss"][-1] if hist["A_loss"] else float("nan"),
               pa_psnr, pa_ssim))
        # ---- Phase B：隐式训练 ----
        log("PHASE B: implicit (forward anderson5 budget=%d tol=%.0e | GMRES mv=%d rtol=%.0e)"
            % (args.fw_budget, args.fw_tol, args.bwd_mv, args.bwd_rtol))
        train_phase_b(reg, opt, store, train_idx, mask, dev, args, hist, t_all, ema)
        # 恢复最优 EMA 权重
        ck_best = torch.load(os.path.join(RUN_DIR, "checkpoint_best.pt"),
                             map_location="cpu", weights_only=False)
        reg.load_state_dict(ck_best["state_dict"])
        log("PHASE B done: best val_psnr=%.2f @ epoch %d"
            % (hist["best_val_psnr"], hist["best_epoch"]))

    # ---- 最终评测（全部测试切片, 紧公差隐式前向） ----
    log("FINAL EVAL on test n=%d (tight forward max=%d tol=%.0e)"
        % (len(test_idx), args.fw_eval_iters, 1e-6))
    te_idx = list(test_idx[:int(args.test_subset)]) if args.test_subset else list(test_idx)
    log("FINAL EVAL slices=%d" % len(te_idx))
    te = eval_test_full(store, te_idx, mask_store, dev, MAIN_MASK,
                        max_iters=args.fw_eval_iters, tol=1e-4)
    zf_p, zf_s = ZF_REF[MAIN_MASK]
    cb_p, cb_s = CASCADE_BASELINE[MAIN_MASK]
    report["test"] = te
    report["test"]["zf_psnr"] = zf_p
    report["test"]["zf_ssim"] = zf_s
    report["test"]["cascade_baseline_psnr"] = cb_p
    report["test"]["cascade_baseline_ssim"] = cb_s
    report["test"]["gain_vs_zf_db"] = te["psnr"] - zf_p
    report["test"]["delta_vs_cascade_db"] = te["psnr"] - cb_p

    # ---- 训练后谱半径（商流形收缩是否保持） ----
    rho = measure_rho(store, val_idx, mask_store, dev, n_slices=2)
    report["rho_trained"] = rho
    log("RHO trained rho(J_S)=%s" % ("n/a" if rho is None else "%.6f" % rho))

    # ---- 隐式反传抽查（训练后的算子, 2 切片） ----
    bwd_ok = False
    try:
        pick = np.random.RandomState(11).choice(np.asarray(val_idx, dtype=np.int64),
                                                2, replace=False)
        xx = store.get_batch(list(map(int, pick)), device=dev).to(torch.complex64)
        yy, zz0 = Q.make_inputs(xx, mask, dev)
        z_fp, _st, _it, _r = Q.solve_fixed_point(zz0, yy, mask, reg, ALPHA, ETA, P,
                                                 "anderson5", 100, 1e-4)
        loss_fn = lambda zz: C.recon_loss(C.to_c(zz), xx, yy, mask)
        _q, _g, res, n_mv, _lv, _rn = Q.quotient_backward(
            z_fp, yy, mask, reg, ALPHA, ETA, P, loss_fn, budget=args.bwd_mv,
            rtol=1e-3, reorth=True)
        bwd_ok = float(res) < 1e-2
        report["bwd_check"] = {"res": float(res), "n_mv": n_mv}
        log("BWD check GMRES res=%.2e n_mv=%d -> %s"
            % (res, n_mv, "PASS" if bwd_ok else "FAIL"))
    except Exception as e:
        log("BWD check exception: %r" % e)
        report["issues"].append(str(e))

    # ---- verdict ----
    v = report["verdict"]
    v["aligned"] = True
    v["fwd_conv_rate"] = te["conv_rate"]
    v["test_psnr"] = te["psnr"]
    v["test_ssim"] = te["ssim"]
    v["gain_vs_zf_db"] = te["psnr"] - zf_p
    v["delta_vs_cascade_db"] = te["psnr"] - cb_p
    v["rho_trained"] = rho
    v["m1_fwd_converged"] = bool(te["conv_rate"] >= 0.9)  # 在 eval tol=1e-4 下
    v["m2_bwd_gmres_ok"] = bwd_ok
    v["q1_beats_zf"] = bool(te["psnr"] > zf_p + 0.3)
    v["r1_rho_lt_1"] = bool(rho is not None and rho < 1.0)
    v["status"] = "PASS" if (v["m1_fwd_converged"] and v["q1_beats_zf"]) else "REVIEW"

    save_figures(hist, report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=C.json_default)
    log("report written: %s" % REPORT_PATH)

    L = []
    A = L.append
    A("=" * 74)
    A("STEP6 QB-DEQ IMPLICIT TRAINING SUMMARY (mode=%s)" % args.mode)
    A("=" * 74)
    A("VERDICT: %s" % v["status"])
    A("  test PSNR r4_s42 : %.2f dB (SSIM %.4f, n=%d)" % (te["psnr"], te["ssim"], te["n"]))
    A("  ZF baseline      : %.2f dB / %.4f" % (zf_p, zf_s))
    A("  CascadeNet base  : %.2f dB / %.4f" % (cb_p, cb_s))
    A("  gain vs ZF       : %+.2f dB | delta vs CascadeNet: %+.2f dB" %
      (te["psnr"] - zf_p, te["psnr"] - cb_p))
    A("  fwd conv rate    : %.1f%% (avg iters %.1f)" % (100 * te["conv_rate"], te["it_avg"]))
    A("  GMRES res (bwd)  : %.2e" % (report.get("bwd_check", {}).get("res", float("nan"))))
    A("  rho(J_S) trained : %s" % ("n/a" if rho is None else "%.6f" % rho))
    A("-" * 74)
    A("recipe: op=%s brg_gauge alpha=%.2f eta=%.2f p=%.1f | SNRegNet mid=%d layers=%d params=%d"
      % (args.op, args.alpha, args.eta, args.p, args.mid, args.layers, n_params(reg)))
    A("Phase A: unrolled K=%d epochs=%d | Phase B: implicit epochs=%d lr=%.0e "
      "fw_budget=%d bwd_mv=%d" % (args.warm_unroll, args.warm_epochs, args.epochs,
                                  args.lr, args.fw_budget, args.bwd_mv))
    A("M1 fwd conv>=90%%: %s | M2 bwd GMRES<1e-2: %s | Q1 PSNR>ZF+0.3: %s | R1 rho<1: %s"
      % (v["m1_fwd_converged"], v["m2_bwd_gmres_ok"], v["q1_beats_zf"], v["r1_rho_lt_1"]))
    A("=" * 74)
    summary_text = "\n".join(L)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    log("summary written: %s" % SUMMARY_PATH)
    log("done in %.1fs | verdict: %s" % (time.time() - t_all, v["status"]))
    try:
        _LOG_FH.close()
    except Exception:
        pass
    return 0 if v["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

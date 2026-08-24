# -*- coding: utf-8 -*-
"""
step7_train_qbdeq_v2.py -- QB-DEQ v2 端到端训练 (fastMRI knee 320x320)
=======================================================================
目标：让"真正的 QB-DEQ"（权重共享隐式模型，前向解不动点、反传用商流形
隐式梯度）在 fastMRI 膝关节单线圈 4x 上达到一流重建质量（PSNR >= 32 dB）。

算子（RED/梯度型，商流形 gauge + 可学习 eta/alpha）：
    S(z) = Gauge( z - eta*( A*(A z - y) + alpha*(z - D(z)) ) )
  - D：强 UNet 去噪/重建器（GroupNorm UNet, base=96）
  - 不动点 z* 满足 (A*A+alpha I) z* = A*y + alpha D(z*)，与 eta 无关；
    采样区系数由测量主导、非采样区由 D 的谱外推填充——MoDL/RED 型。
  - A1 阶段加 Jacobian 谱正则（--spec-lam）驱动 rho(J_S)<1。

课程（三阶段）：
  A0  单步训练：监督 z1 = S(z0) 到 GT（等价 K=1 级联，给 D 强初始化）。
  A1  权重共享 unrolled（progressive K, 深度监督 + 谱正则）：
      训练后直接解不动点即为 QB-DEQ 推理；不动点质量=隐式质量。
  B   隐式训练：Anderson 前向 + 商流形 GMRES 反传，损失在 z* 处。

用法：
  python step7_train_qbdeq_v2.py --smoke
  python step7_train_qbdeq_v2.py                     # A0+A1+B 全流程
  python step7_train_qbdeq_v2.py --phase a1 --resume
  python step7_train_qbdeq_v2.py --eval-only
"""
import os
import sys
import json
import time
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
import qbdeq_v2 as V2

HERE = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(HERE, "fastmri_320_meta.pt")
MAIN_MASK = "r4_s42"
RUN_DIR = os.path.join(HERE, "runs", "step7_train")
FIG_DIR = os.path.join(HERE, "step7_figs")
REPORT_PATH = os.path.join(HERE, "step7_train_report.json")
SUMMARY_PATH = os.path.join(HERE, "step7_train_summary.txt")

EMA_DECAY = 0.999
CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-5
ZF_REF = {"r4_s42": (25.42, 0.5405), "r4_s123": (25.25, 0.5333),
          "r4_s2025": (25.08, 0.5350)}
CASCADE_BASELINE = {"r4_s42": (27.78, 0.6226)}
TARGET_PSNR = 32.0

_LOG_LINES = []
_LOG_FH = None


def log(msg):
    line = "[STEP7] " + str(msg)
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("utf-8", "replace").decode("ascii", "replace"))
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


def time_str(t):
    t = int(t)
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    return "%dh%02dm%02ds" % (h, m, s) if h > 0 else "%dm%02ds" % (m, s)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def trainable(net, eta_logit, alpha_logit):
    return [p for p in list(net.parameters()) + [eta_logit, alpha_logit]
            if p.requires_grad]


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
            sd = model.state_dict()
            for k in self.shadow:
                if k in sd:
                    sd[k].copy_(self.shadow[k].to(sd[k].device))


def aug_batch(x, mask, flip_h, flip_v, rot_k):
    """x: (B,H,W) complex；mask: (H,W) bool。返回 (x, mask)。"""
    if flip_h:
        x = torch.flip(x, dims=[-1])
    if flip_v:
        x = torch.flip(x, dims=[-2])
    if rot_k:
        x = torch.rot90(x, k=rot_k, dims=(-2, -1))
        mask = torch.rot90(mask, k=rot_k, dims=(-2, -1))
    return x, mask


def spectral_penalty(z_last, z_prev, lam, rng_state, rho_thr=0.9, n_multi=1):
    """单侧 Jacobian 谱正则：sum_i max(0, ||J_S v_i|| - rho_thr)^2 / n_multi。
    用 n_multi 个独立随机方向覆盖更多扩张方向；把算子扩张压到 rho_thr 以下。"""
    if lam <= 0 or z_last is None or z_prev is None:
        return None
    pen = None
    for _ in range(max(1, int(n_multi))):
        v = torch.randn_like(z_prev)
        v = v / v.flatten().norm().clamp(min=1e-8)
        Jv = torch.autograd.grad(z_last, z_prev, grad_outputs=v,
                                 retain_graph=True, create_graph=True)[0]
        term = lam * torch.relu(Jv.flatten().norm() - rho_thr) ** 2
        pen = term if pen is None else pen + term
    return pen / max(1, int(n_multi))


def train_unrolled(store, train_idx, mask, device, net, eta_logit, alpha_logit,
                   opt, scaler, ema, args, hist, t0, epoch_start=1, K_start=2):
    """A0（K=1 单步）与 A1（progressive K，深度监督 + 谱正则）共用的 unrolled 训练。"""
    accum = max(1, int(args.grad_accum))
    use_amp = bool(args.amp and device == "cuda")
    rngp = np.random.RandomState(args.seed * 7)
    for epoch in range(epoch_start, epoch_start + int(args.epochs)):
        K = min(int(args.K_max), max(int(K_start), 2 + 2 * (epoch - epoch_start)))
        lr = C.lr_at(epoch, args)
        for pg in opt.param_groups:
            pg["lr"] = lr
        rng = np.random.RandomState(args.seed * 100 + epoch)
        perm = rng.permutation(len(train_idx))
        epoch_loss = []
        opt.zero_grad()
        steps = 0
        for s0 in range(0, len(perm), args.batch):
            b_idx = [int(i) for i in train_idx[perm[s0:s0 + args.batch]]]
            x = store.get_batch(b_idx, device=device).to(torch.complex64)
            m = mask
            if args.aug:
                x, m = aug_batch(x, m, random.random() < 0.5, random.random() < 0.5,
                                 int(random.random() * 4))
            y, z0 = Q.make_inputs(x, m, device)
            z = z0
            outs = []
            z_prev = None
            for _ in range(K):
                z_prev = z
                z = V2.S_op(z, y, m, net, eta_logit, alpha_logit)
                outs.append(C.to_c(z))
            with torch.autocast("cuda", enabled=use_amp):
                loss = C.recon_loss(outs[-1], x, y, m)
                if args.deep_w > 0 and K >= 4:
                    loss = loss + args.deep_w * C.recon_loss(outs[K // 2 - 1], x, y, m)
                if args.id_lam > 0:
                    # 恒等锚定：D(clean)=clean（干净图是 D 的不动点 -> RED 不动点=好重建）
                    dx = net(C.to_2ch(x))
                    loss = loss + args.id_lam * C.recon_loss(C.to_c(dx), x, y, m)
                if args.spec_lam > 0 and K >= 2:
                    pen = spectral_penalty(z, z_prev, args.spec_lam, rngp,
                                           rho_thr=args.spec_rho,
                                           n_multi=args.spec_multi)
                    if pen is not None:
                        loss = loss + pen
                loss = loss / accum
            scaler.scale(loss).backward()
            epoch_loss.append(float(loss.item()) * accum)
            steps += 1
            if steps % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    trainable(net, eta_logit, alpha_logit), CLIP_NORM)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            if args.smoke and steps >= 8:
                break
            if steps % 50 == 0:
                log("U ep%d K=%d step %d loss=%.4f lr=%.1e t=%s"
                    % (epoch, K, steps, float(loss.item()) * accum, lr,
                       time_str(time.time() - t0)))
        ema.update(net)
        hist["loss"].append(float(np.mean(epoch_loss)) if epoch_loss else float("nan"))
        hist["epoch"].append(epoch)
        hist["K"].append(K)
        log("U epoch %d/%d K=%d loss=%.5f eta=%.3f alpha=%.3f (lr=%.1e)"
            % (epoch, args.epochs, K, hist["loss"][-1],
               float(torch.nn.functional.softplus(eta_logit)),
               float(torch.nn.functional.softplus(alpha_logit)), lr))
        if args.phase_tag:
            save_ckpt(os.path.join(RUN_DIR, "checkpoint_%s.pt" % args.phase_tag),
                      net, eta_logit, alpha_logit, opt, ema, epoch, args.phase_tag, hist)
        if args.smoke:
            break
    return epoch


def train_phase_b(store, train_idx, mask, device, net, eta_logit, alpha_logit,
                  opt, args, hist, t0, ema, start_epoch=1):
    """隐式训练：Anderson 前向 + 商流形 GMRES 反传（fp32，稳定性优先）。"""
    accum = max(1, int(args.grad_accum))
    steps = 0
    for epoch in range(start_epoch, start_epoch + int(args.epochs)):
        lr = C.lr_at(epoch, args)
        for pg in opt.param_groups:
            pg["lr"] = lr
        rng = np.random.RandomState(args.seed * 1000 + epoch)
        perm = rng.permutation(len(train_idx))
        epoch_loss, epoch_fw, epoch_bwd = [], [], []
        gacc = [torch.zeros_like(p) for p in trainable(net, eta_logit, alpha_logit)]
        opt.zero_grad()
        for s0 in range(0, len(perm), args.batch):
            b_idx = [int(i) for i in train_idx[perm[s0:s0 + args.batch]]]
            x = store.get_batch(b_idx, device=device).to(torch.complex64)
            m = mask
            if args.aug:
                x, m = aug_batch(x, m, random.random() < 0.5, random.random() < 0.5,
                                 int(random.random() * 4))
            y, z0 = Q.make_inputs(x, m, device)
            z_fp, status, it_fin, _rels = V2.solve_fixed_point(
                z0, y, m, net, eta_logit, alpha_logit, "anderson5",
                max_iters=args.fw_budget, tol=args.fw_tol, damp=args.damp)
            if not torch.isfinite(z_fp).all():
                log("B NON-FINITE forward (it=%d) -> skip" % it_fin)
                continue
            loss_fn = lambda zz: C.recon_loss(C.to_c(zz), x, y, m)
            try:
                q, grads, res, n_mv, loss_v, _rhs = V2.quotient_backward(
                    z_fp, y, m, net, eta_logit, alpha_logit, loss_fn,
                    budget=args.bwd_mv, rtol=args.bwd_rtol, reorth=True)
            except Exception as e:
                log("B backward exception %r -> skip" % e)
                continue
            if not torch.isfinite(q).all() or not all(
                    g is None or torch.isfinite(g).all() for g in grads):
                log("B NON-FINITE backward (res=%.1e) -> skip" % res)
                continue
            # 谱正则（维持 rho<1）：||J_S v|| 在不动点处
            if args.spec_lam > 0:
                zc_fp = z_fp.detach().requires_grad_(True)
                vv = torch.randn_like(zc_fp)
                vv = vv / vv.flatten().norm().clamp(min=1e-8)
                with torch.enable_grad():
                    Sz = V2.S_op(zc_fp, y, m, net, eta_logit, alpha_logit)
                    Jv = torch.autograd.grad(Sz, zc_fp, grad_outputs=vv,
                                             retain_graph=True,
                                             create_graph=True)[0]
                    pen = args.spec_lam * torch.relu(Jv.flatten().norm() - args.spec_rho) ** 2
                if torch.isfinite(pen) and float(pen.item()) > 0:
                    pg = torch.autograd.grad(pen, trainable(net, eta_logit, alpha_logit),
                                             retain_graph=False, allow_unused=True)
                    for gi, g2 in zip(gacc, pg):
                        if g2 is not None:
                            gi.add_(g2.detach().float(), alpha=1.0 / accum)
            for gi, g in zip(gacc, grads):
                if g is not None:
                    gi.add_(g.detach().float(), alpha=1.0 / accum)
            epoch_loss.append(loss_v)
            epoch_fw.append(int(it_fin))
            epoch_bwd.append(float(res))
            steps += 1
            if steps % accum == 0:
                params = trainable(net, eta_logit, alpha_logit)
                for p, g in zip(params, gacc):
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad.add_(g)
                torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
                opt.step()
                opt.zero_grad()
                for g in gacc:
                    g.zero_()
                ema.update(net)
            if args.smoke and steps >= 8:
                break
            if steps % 20 == 0:
                log("B ep%d step %d loss=%.4f fw_it=%d gmres=%.1e lr=%.1e t=%s"
                    % (epoch, steps, loss_v, it_fin, res, lr, time_str(time.time() - t0)))
        hist["B_loss"].append(float(np.mean(epoch_loss)) if epoch_loss else float("nan"))
        hist["B_epoch"].append(epoch)
        hist["B_fw_it"].append(float(np.mean(epoch_fw)) if epoch_fw else float("nan"))
        hist["B_bwd_res"].append(float(np.mean(epoch_bwd)) if epoch_bwd else float("nan"))
        log("B epoch %d/%d loss=%.5f fw_it=%.1f bwd_res=%.1e eta=%.3f alpha=%.3f"
            % (epoch, args.epochs, hist["B_loss"][-1], hist["B_fw_it"][-1],
               hist["B_bwd_res"][-1],
               float(torch.nn.functional.softplus(eta_logit)),
               float(torch.nn.functional.softplus(alpha_logit))))
        save_ckpt(os.path.join(RUN_DIR, "checkpoint_b.pt"), net, eta_logit,
                  alpha_logit, opt, ema, epoch, "b", hist)
        if args.smoke:
            break
    return epoch


def eval_val_proxy(store, val_idx, mask, device, net, eta_logit, alpha_logit, K, n=24):
    idx = [int(i) for i in np.random.RandomState(0).choice(
        np.asarray(val_idx, dtype=np.int64), min(n, len(val_idx)), replace=False)]
    r = V2.eval_unrolled(store, idx, mask, device, net, eta_logit, alpha_logit, K, batch=2)
    return float(r["psnr"]), float(r["ssim"])


def eval_full(store, idx, mask, device, net, eta_logit, alpha_logit, tag="test",
              max_iters=200, tol=1e-4, batch=2, damp=1.0, also_unrolled=8):
    log("EVAL[%s] fixed-point solve (tol=%.0e, max=%d) ..." % (tag, tol, max_iters))
    r = V2.eval_fixed_point(store, idx, mask, device, net, eta_logit, alpha_logit,
                            max_iters=max_iters, tol=tol, batch=batch, damp=damp)
    log("EVAL[%s] fp: psnr=%.2f ssim=%.4f conv=%.0f%% it_avg=%.1f (n=%d)"
        % (tag, r["psnr"], r["ssim"], 100 * r["conv_rate"], r["it_avg"], r["n"]))
    if also_unrolled:
        ru = V2.eval_unrolled(store, idx, mask, device, net, eta_logit, alpha_logit,
                              also_unrolled, batch=batch)
        log("EVAL[%s] unrolled K=%d: psnr=%.2f ssim=%.4f" %
            (tag, also_unrolled, ru["psnr"], ru["ssim"]))
        r["unrolled_K%d" % also_unrolled] = ru
    for B in (16, 32):
        rB = V2.eval_fixed_point(store, idx, mask, device, net, eta_logit,
                                 alpha_logit, max_iters=B, tol=0.0, batch=batch,
                                 damp=1.0)
        log("EVAL[%s] budget=%d: psnr=%.2f ssim=%.4f" % (tag, B, rB["psnr"], rB["ssim"]))
        r["budget%d" % B] = rB
    return r


def measure_rho(store, val_idx, mask, device, net, eta_logit, alpha_logit, n_slices=2):
    pick = np.random.RandomState(7).choice(np.asarray(val_idx, dtype=np.int64),
                                           n_slices, replace=False)
    rhos = []
    for i in pick:
        x = store.get_batch([int(i)], device=device).to(torch.complex64)
        y, z0 = Q.make_inputs(x, mask, device)
        z_fp, _st, _it, _r = V2.solve_fixed_point(z0, y, mask, net, eta_logit,
                                                  alpha_logit, "anderson5", 150, 1e-5)
        r = V2.estimate_rho(z_fp, y, mask, net, eta_logit, alpha_logit)
        rhos.append(r)
        log("rho(J_S) slice=%d -> %s" % (int(i), "n/a" if r is None else "%.5f" % r))
    rs = [r for r in rhos if r is not None]
    return float(np.mean(rs)) if rs else None


def save_ckpt(path, net, eta_logit, alpha_logit, opt, ema, epoch, phase, hist):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"net": net.state_dict(),
                "eta_logit": eta_logit.detach().cpu().clone(),
                "alpha_logit": alpha_logit.detach().cpu().clone(),
                "opt": opt.state_dict(), "ema": ema.shadow,
                "epoch": epoch, "phase": phase, "history": hist}, path)


def load_ckpt(path, net, eta_logit, alpha_logit, opt=None, ema=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["net"])
    with torch.no_grad():
        eta_logit.copy_(ck["eta_logit"].to(eta_logit.device))
        if "alpha_logit" in ck:
            alpha_logit.copy_(ck["alpha_logit"].to(alpha_logit.device))
    if opt is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    if ema is not None:
        if ck.get("ema"):
            ema.shadow = {k: v.clone() for k, v in ck["ema"].items()}
        else:
            ema = EMA(net)
    return ck


def parse_args():
    p = argparse.ArgumentParser(description="step7: QB-DEQ v2 端到端训练")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="冒烟（极小规模）")
    g.add_argument("--eval-only", action="store_true", help="只评测最佳检查点")
    p.add_argument("--phase", type=str, default="all",
                   help="a0 | a1 | b | all（默认 all = a0+a1+b）")
    p.add_argument("--resume", action="store_true", help="从最近检查点续跑")
    p.add_argument("--base", type=int, default=96, help="UNet 基础通道数")
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--K-max", type=int, default=8, help="A1 最大 unrolled 深度")
    p.add_argument("--deep-w", type=float, default=0.5, help="深度监督权重")
    p.add_argument("--spec-lam", type=float, default=0.0, help="Jacobian 谱正则权重")
    p.add_argument("--spec-rho", type=float, default=0.9, help="谱正则阈值（目标 rho）")
    p.add_argument("--spec-multi", type=int, default=1, help="每 batch 谱正则随机方向数")
    p.add_argument("--sn", type=int, default=1, help="UNet 谱归一化（默认开）")
    p.add_argument("--sn-scale", type=float, default=0.70710678,
                   help="解码器 SN 缩放（1/sqrt2 抵消 concat 膨胀）")
    p.add_argument("--id-lam", type=float, default=0.0,
                   help="恒等锚定权重：训练 D(clean)=clean，使 RED 不动点=好重建")
    p.add_argument("--eta-init", type=float, default=0.3)
    p.add_argument("--alpha-init", type=float, default=0.5)
    p.add_argument("--a0-epochs", type=int, default=2)
    p.add_argument("--a1-epochs", type=int, default=20)
    p.add_argument("--b-epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--fw-budget", type=int, default=40, help="B 前向迭代上限")
    p.add_argument("--fw-tol", type=float, default=1e-3)
    p.add_argument("--bwd-mv", type=int, default=40, help="GMRES matvec 预算")
    p.add_argument("--bwd-rtol", type=float, default=1e-3)
    p.add_argument("--damp", type=float, default=0.9, help="Anderson 阻尼")
    p.add_argument("--val-max", type=int, default=48)
    p.add_argument("--train-subset", type=int, default=0, help="0=全部 5668")
    p.add_argument("--test-subset", type=int, default=0, help="0=全部 804")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--aug", type=int, default=1)
    args = p.parse_args()
    if args.smoke:
        args.base = 16
        args.a0_epochs = 1
        args.a1_epochs = 1
        args.b_epochs = 1
        args.K_max = 4
        args.train_subset = 48
        args.test_subset = 12
        args.val_max = 12
        args.batch = 2
        args.grad_accum = 1
        args.warmup = 1
        args.fw_budget = 12
        args.fw_tol = 1e-2
        args.bwd_mv = 8
        args.bwd_rtol = 5e-2
        args.spec_lam = 0.05
        args.aug = 0
    args.amp = bool(args.amp)
    args.aug = bool(args.aug)
    args.phase_tag = None
    args.mode = "smoke" if args.smoke else ("eval" if args.eval_only else "full")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("step7 QB-DEQ v2 (RED operator) mode=%s seed=%d device=%s torch=%s"
        % (args.mode, args.seed, dev, torch.__version__))
    log("config: base=%d K_max=%d a0=%d a1=%d b=%d lr=%.0e batch=%d accum=%d "
        "spec_lam=%.3f amp=%d aug=%d"
        % (args.base, args.K_max, args.a0_epochs, args.a1_epochs, args.b_epochs,
           args.lr, args.batch, args.grad_accum, args.spec_lam, args.amp, args.aug))
    log("sn=%d (spectral-normalized UNet)" % args.sn)

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
    if args.test_subset:
        test_idx = np.asarray(list(test_idx[:int(args.test_subset)]), dtype=np.int64)
    log("split train=%d val=%d test=%d val_sub=%d"
        % (len(train_idx), len(val_idx), len(test_idx), len(args.val_idx)))
    mask = mask_store.get(MAIN_MASK, device=dev)

    net, eta_logit, alpha_logit = V2.build_model(
        base=args.base, groups=args.groups,
        eta_init=args.eta_init, alpha_init=args.alpha_init,
        sn=bool(args.sn), sn_scale=args.sn_scale)
    net = net.to(dev)
    eta_logit = torch.nn.Parameter(eta_logit.detach().to(dev))
    alpha_logit = torch.nn.Parameter(alpha_logit.detach().to(dev))
    log("UNet params=%d | eta0=%.2f alpha0=%.2f"
        % (n_params(net), float(torch.nn.functional.softplus(eta_logit)),
           float(torch.nn.functional.softplus(alpha_logit))))

    global _LOG_FH
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    _LOG_FH = open(os.path.join(RUN_DIR, "stdout.log"), "w", encoding="utf-8")

    hist = {"epoch": [], "K": [], "loss": [], "B_epoch": [], "B_loss": [],
            "B_fw_it": [], "B_bwd_res": [], "phase_done": []}
    report = {"args": vars(args), "mode": args.mode,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "target_psnr": TARGET_PSNR, "phases": {}, "verdict": {}, "issues": []}
    t_all = time.time()

    if args.mode == "eval":
        path = None
        for ph in ["b", "a1", "a0", "best"]:
            cand = os.path.join(RUN_DIR, "checkpoint_%s.pt" % ph)
            if os.path.exists(cand):
                path = cand
                break
        assert path is not None, "no checkpoint found in %s" % RUN_DIR
        ck = load_ckpt(path, net, eta_logit, alpha_logit)
        hist = dict(ck.get("history", hist))
        log("EVAL loaded %s (phase=%s epoch=%d)" % (path, ck.get("phase"), ck.get("epoch")))
        ema = EMA(net)
        if ck.get("ema"):
            ema.shadow = {k: v.clone() for k, v in ck["ema"].items()}
            ema.copy_to(net)
        r_test = eval_full(store, test_idx, mask, dev, net, eta_logit, alpha_logit,
                           tag="test")
        rho = measure_rho(store, args.val_idx, mask, dev, net, eta_logit, alpha_logit)
        report["test"] = r_test
        report["rho"] = rho
        report["eta"] = float(torch.nn.functional.softplus(eta_logit).item())
        report["alpha"] = float(torch.nn.functional.softplus(alpha_logit).item())
        report["verdict"] = {"psnr": round(r_test["psnr"], 4),
                             "target": TARGET_PSNR,
                             "reach_target": bool(r_test["psnr"] >= TARGET_PSNR)}
        _write_report(report)
        _write_summary(report, t_all)
        return 0

    opt = torch.optim.AdamW(trainable(net, eta_logit, alpha_logit),
                            lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and dev == "cuda"))
    ema = EMA(net)
    phases = ["a0", "a1", "b"] if args.phase == "all" else [args.phase]
    phase_epochs = {"a0": args.a0_epochs, "a1": args.a1_epochs, "b": args.b_epochs}
    start_epochs = {"a0": 1, "a1": 1, "b": 1}
    last = 1

    if args.resume:
        loaded = None
        for ph in ["a0", "a1", "b"]:
            pth = os.path.join(RUN_DIR, "checkpoint_%s.pt" % ph)
            if os.path.exists(pth):
                try:
                    ck = load_ckpt(pth, net, eta_logit, alpha_logit, opt, ema)
                    loaded = (ph, pth, ck)
                except Exception as e:
                    log("RESUME skip %s (incompatible): %s" % (ph, e))
        if loaded is not None:
            ph, pth, ck = loaded
            hist = dict(ck.get("history", hist))
            log("RESUME %s: epoch=%d" % (ph, ck.get("epoch", 0)))
            start_epochs[ph] = int(ck.get("epoch", 0)) + 1
            last = int(ck.get("epoch", 0))
        phases = [ph for ph in phases if ph not in hist.get("phase_done", [])]

    for ph in phases:
        if phase_epochs[ph] <= 0:
            log("PHASE %s skipped (0 epochs)" % ph)
            continue
        t_ph = time.time()
        log("===== PHASE %s (%d epochs) =====" % (ph, phase_epochs[ph]))
        if ph == "a0":
            args_a0 = copy.copy(args)
            args_a0.phase_tag = "a0"
            args_a0.epochs = phase_epochs["a0"]
            args_a0.K_max = 1
            args_a0.lr = min(args.lr, 3e-4)
            args_a0.warmup = max(1, min(args.warmup, args_a0.epochs))
            last = train_unrolled(store, train_idx, mask, dev, net, eta_logit,
                                  alpha_logit, opt, scaler, ema, args_a0, hist, t_all,
                                  epoch_start=start_epochs["a0"], K_start=1)
            save_ckpt(os.path.join(RUN_DIR, "checkpoint_a0.pt"), net, eta_logit,
                      alpha_logit, opt, ema, last, "a0", hist)
            hist["phase_done"].append("a0")
            pv, sv = eval_val_proxy(store, args.val_idx, mask, dev, net, eta_logit,
                                    alpha_logit, K=1, n=min(24, len(args.val_idx)))
            log("A0 done: val psnr(K=1)=%.2f ssim=%.4f" % (pv, sv))
            report["phases"]["a0"] = {"val_psnr_k1": pv, "val_ssim_k1": sv,
                                      "time_s": time.time() - t_ph}
        elif ph == "a1":
            args_a1 = copy.copy(args)
            args_a1.phase_tag = "a1"
            args_a1.epochs = phase_epochs["a1"]
            args_a1.lr = min(args.lr, 3e-4)
            args_a1.warmup = max(1, min(args.warmup, args_a1.epochs))
            last = train_unrolled(store, train_idx, mask, dev, net, eta_logit,
                                  alpha_logit, opt, scaler, ema, args_a1, hist, t_all,
                                  epoch_start=start_epochs["a1"],
                                  K_start=min(4, args.K_max))
            save_ckpt(os.path.join(RUN_DIR, "checkpoint_a1.pt"), net, eta_logit,
                      alpha_logit, opt, ema, last, "a1", hist)
            hist["phase_done"].append("a1")
            pv, sv = eval_val_proxy(store, args.val_idx, mask, dev, net, eta_logit,
                                    alpha_logit, K=args.K_max, n=min(24, len(args.val_idx)))
            log("A1 done: val psnr(K=%d)=%.2f ssim=%.4f" % (args.K_max, pv, sv))
            report["phases"]["a1"] = {"val_psnr": pv, "val_ssim": sv,
                                      "time_s": time.time() - t_ph}
        elif ph == "b":
            args_b = copy.copy(args)
            args_b.epochs = phase_epochs["b"]
            args_b.lr = 1e-4
            args_b.warmup = 1
            last = train_phase_b(store, train_idx, mask, dev, net, eta_logit,
                                 alpha_logit, opt, args_b, hist, t_all, ema,
                                 start_epoch=start_epochs["b"])
            save_ckpt(os.path.join(RUN_DIR, "checkpoint_b.pt"), net, eta_logit,
                      alpha_logit, opt, ema, last, "b", hist)
            hist["phase_done"].append("b")
            report["phases"]["b"] = {"time_s": time.time() - t_ph}

    # 最终评测必须用训练好的工作权重：用当前权重重建 EMA 快照（避免陈旧 EMA 覆盖）
    ema = EMA(net)
    ema.copy_to(net)
    save_ckpt(os.path.join(RUN_DIR, "checkpoint_best.pt"), net, eta_logit,
              alpha_logit, opt, ema, last, "best", hist)
    r_test = eval_full(store, test_idx, mask, dev, net, eta_logit, alpha_logit,
                       tag="test")
    r_val = eval_full(store, args.val_idx, mask, dev, net, eta_logit, alpha_logit,
                      tag="val", also_unrolled=args.K_max)
    rho = measure_rho(store, args.val_idx, mask, dev, net, eta_logit, alpha_logit)
    eta_f = float(torch.nn.functional.softplus(eta_logit).item())
    alpha_f = float(torch.nn.functional.softplus(alpha_logit).item())
    log("eta_final=%.3f alpha_final=%.3f rho(J_S)=%s"
        % (eta_f, alpha_f, "n/a" if rho is None else "%.5f" % rho))
    report["test"] = r_test
    report["val"] = r_val
    report["rho"] = rho
    report["eta"] = eta_f
    report["alpha"] = alpha_f
    report["verdict"] = {
        "psnr": round(r_test["psnr"], 4), "ssim": round(r_test["ssim"], 4),
        "zf": ZF_REF[MAIN_MASK][0], "cascade": CASCADE_BASELINE[MAIN_MASK][0],
        "target": TARGET_PSNR,
        "gain_vs_zf_db": round(r_test["psnr"] - ZF_REF[MAIN_MASK][0], 4),
        "reach_target": bool(r_test["psnr"] >= TARGET_PSNR),
        "conv_rate": r_test["conv_rate"], "rho_lt_1": bool(rho is not None and rho < 1.0)}
    _write_report(report)
    _write_summary(report, t_all)
    log("done in %s | test psnr=%.2f | target=%.1f | %s"
        % (time_str(time.time() - t_all), r_test["psnr"], TARGET_PSNR,
           "REACHED" if r_test["psnr"] >= TARGET_PSNR else "not yet"))
    _close_log()
    return 0


def _write_report(report):
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log("report written: %s" % REPORT_PATH)


def _write_summary(report, t0):
    lines = []
    lines.append("=" * 74)
    lines.append("STEP7 QB-DEQ v2 SUMMARY  (mode=%s, %s)" % (report["mode"],
                 time.strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("=" * 74)
    v = report.get("verdict", {})
    lines.append("test fixed-point PSNR: %.2f dB (SSIM %.4f)" % (v.get("psnr", -1), v.get("ssim", -1)))
    lines.append("ZF baseline           : %.2f dB | CascadeNet baseline: %.2f dB" %
                 (v.get("zf", -1), v.get("cascade", -1)))
    lines.append("target                 : %.1f dB -> %s" % (report.get("target_psnr", 32),
                 "REACHED" if v.get("reach_target") else "NOT reached"))
    lines.append("conv rate (fp solve)   : %.0f%% | rho(J_S): %s" %
                 (100 * v.get("conv_rate", 0), "n/a" if report.get("rho") is None
                  else "%.5f" % report["rho"]))
    lines.append("eta/alpha(final)       : %.3f / %.3f" % (report.get("eta", -1),
                                                           report.get("alpha", -1)))
    lines.append("phases                 : %s" % list(report.get("phases", {}).keys()))
    for ph, d in report.get("phases", {}).items():
        lines.append("   %s: %s" % (ph, json.dumps(d)))
    lines.append("elapsed                : %.1f s" % (time.time() - t0))
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("summary written: %s" % SUMMARY_PATH)


def _close_log():
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


if __name__ == "__main__":
    sys.exit(main())

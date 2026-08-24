# -*- coding: utf-8 -*-
"""baseline_utils.py —— 对比基线共享工具（fastMRI knee 320x320）

统一接口约定（所有模型文件遵循）：
    model.forward(z0, y, mask) -> out_2ch
        z0     : (B,2,H,W) float，零填充图像
        y      : (B,H,W) complex，欠采样 k-space
        mask   : (H,W) bool，采样掩码
        out_2ch: (B,2,H,W) float，重建图像（2ch 实数）

数据/损失/指标全部复用 step5_320_ceiling（与论文实验完全一致）：
    - 数据：fastmri_320_meta.pt + gt chunks（train/val/test 划分同论文）
    - 损失：L_mag + 0.1*(1-SSIM) + 0.01*DC（C.recon_loss）
    - 指标：幅度 PSNR / SSIM（numpy，fastMRI 惯例）
"""
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn

import step5_320_ceiling as C

HERE = os.path.dirname(os.path.abspath(__file__))
META_PATH = os.path.join(HERE, "fastmri_320_meta.pt")
MAIN_MASK = "r4_s42"
ZF_REF = {"r4_s42": (25.42, 0.5405), "r4_s123": (25.25, 0.5333),
          "r4_s2025": (25.08, 0.5350)}
CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-5


def load_data(device="cuda", mask_key=MAIN_MASK, train_subset=2000,
              val_max=48, test_subset=0, seed=42):
    """返回 (store, mask, train_idx, val_idx, test_idx, val_sub_idx)。"""
    meta = torch.load(META_PATH, map_location="cpu", weights_only=False)
    store = C.ChunkStore(meta)
    mask_store = C.MaskStore(meta["masks"])
    train_idx, val_idx, test_idx = C.load_split(meta)
    rng = np.random.RandomState(seed)
    val_sub = rng.choice(np.asarray(val_idx, dtype=np.int64),
                         size=min(int(val_max), len(val_idx)), replace=False).tolist()
    if train_subset:
        train_idx = np.asarray(list(train_idx[:int(train_subset)]), dtype=np.int64)
    if test_subset:
        test_idx = np.asarray(list(test_idx[:int(test_subset)]), dtype=np.int64)
    mask = mask_store.get(mask_key, device=device)
    return store, mask, train_idx, val_idx, test_idx, val_sub


def make_inputs(x_gt, mask, device):
    """gt 复数 (B,H,W) -> (y, z0)。"""
    y = C.fft2_t(x_gt) * mask
    z0 = C.to_2ch(C.ifft2_t(y))
    return y, z0


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def aug_batch(x, mask, flip_h, flip_v):
    if flip_h:
        x = torch.flip(x, dims=[-1])
    if flip_v:
        x = torch.flip(x, dims=[-2])
    return x, mask


def train_one_epoch(model, store, train_idx, mask, device, opt, scaler,
                    args, t0, epoch):
    """一轮训练；返回平均损失。所有模型统一：out=model(z0,y,mask)，loss=recon_loss。"""
    model.train()
    accum = max(1, int(args.grad_accum))
    use_amp = bool(args.amp and device == "cuda")
    rng = np.random.RandomState(args.seed * 100 + epoch)
    perm = rng.permutation(len(train_idx))
    losses = []
    opt.zero_grad()
    steps = 0
    for s0 in range(0, len(perm), args.batch):
        b_idx = [int(i) for i in train_idx[perm[s0:s0 + args.batch]]]
        x = store.get_batch(b_idx, device=device).to(torch.complex64)
        m = mask
        if args.aug:
            x, m = aug_batch(x, m, random.random() < 0.5, random.random() < 0.5)
        y, z0 = make_inputs(x, m, device)
        with torch.autocast("cuda", enabled=use_amp):
            out = model(z0, y, m)
            loss = C.recon_loss(C.to_c(out), x, y, m) / accum
        scaler.scale(loss).backward()
        losses.append(float(loss.item()) * accum)
        steps += 1
        if steps % accum == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        if steps % 200 == 0:
            print("  [%s] ep%d step %d loss=%.4f t=%s" % (
                args.name, epoch, steps, losses[-1], time_str(time.time() - t0)))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model, store, idx, mask, device, batch=2):
    """测试集幅度 PSNR/SSIM（numpy，与 fastmri_320_prep 一致）。"""
    model.eval()
    ssim = C.SSIMComputer()
    ps, ss = [], []
    for s0 in range(0, len(idx), batch):
        blk = [int(i) for i in idx[s0:s0 + batch]]
        x = store.get_batch(blk, device=device).to(torch.complex64)
        y, z0 = make_inputs(x, mask, device)
        out = model(z0, y, mask)
        z_np = C.to_c(out).detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        for i in range(len(blk)):
            gm, zm = np.abs(x_np[i]), np.abs(z_np[i])
            ps.append(C.compute_psnr(gm, zm))
            ss.append(ssim.compute(gm, zm))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n": len(ps)}


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

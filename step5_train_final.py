# -*- coding: utf-8 -*-
"""
step5_train_final.py —— QB-DEQ 论文最终正式训练脚本（fastMRI knee 320x320 复数重建）

方法（学习型级联 CascadeNet）:
    z_{n+1} = UNet_n( z_n - softplus(eta_n) * A^H (A z_n - y) ),  n = 0..K-1
    - K 个独立 U-Net（GroupNorm）级联，端到端展开训练（unrolled）；
    - 第 1 级退化为直接 U-Net（与 step5_320_ceiling 同拓扑），保证 K=1 即可复现
      已验证的直接重建结果；多级级联提供逐步细化（cascade refinement）；
    - eta_n 为可学习步长（softplus 保证为正），DC 项强制与观测 k 空间一致；
    - 隐式深度（DEQ）的理论价值由 step4a/4b/4b2 机制实验验证；本脚本只负责
      最终重建质量的正式训练。

用法:
    python step5_train_final.py --smoke       # 冒烟（约 5 分钟，2 epoch）
    python step5_train_final.py               # 正式训练（RTX 3070 预计 8-20 小时）
    python step5_train_final.py --resume      # 断点续训（runs/step5_final/<mode>/checkpoint_last.pt）
    python step5_train_final.py --eval-only   # 仅评测 checkpoint_best.pt

验收线（fastMRI 幅值 PSNR / SSIM 口径，test 804 切片，r4_s42）:
    PSNR >= 30.5 dB 且 SSIM >= 0.75   -> PASS（可写论文）
    PSNR >= 28.5 dB                   -> REVIEW（可写但需如实说明）
    其余                               -> FAIL（如实报告，绝不改数据）
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import step5_320_ceiling as C

HERE = os.path.dirname(os.path.abspath(__file__))
META_PATH = C.META_PATH
MASK_KEYS = list(C.MASK_KEYS)
MAIN_MASK = "r4_s42"
TEST_MASKS = list(C.TEST_MASKS)
TRAIN_MASKS_DEFAULT = ["r4_s42", "r4_s123", "r4_s2025"]
PASS_PSNR = float(C.PASS_PSNR)
PASS_SSIM = float(C.PASS_SSIM)
MID_PSNR = float(C.MID_PSNR)
MIN_GAIN_DB = float(C.MIN_GAIN_DB)
DEEP_W = 0.5
EMA_DECAY = 0.999
CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-4
RUN_ROOT = os.path.join(HERE, "runs", "step5_final")
FIG_ROOT = os.path.join(HERE, "step5_final_figs")

_LOG_LINES = []
_LOG_FH = None


def log(msg):
    line = "[FINAL] " + str(msg)
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
    C.set_seed(seed)
    torch.backends.cudnn.benchmark = True


def time_str(t):
    t = int(t)
    h = t // 3600
    m = (t % 3600) // 60
    s = t % 60
    if h > 0:
        return "%dh%02dm%02ds" % (h, m, s)
    if m > 0:
        return "%dm%02ds" % (m, s)
    return "%ds" % s
# ============================================================================
# 第 2 段 模型：GroupNorm U-Net（与 UNet3 同拓扑）+ 学习型级联 CascadeNet
# ============================================================================
class GNBlock(nn.Module):
    """Conv-ReLU-GN 双卷积块。GroupNorm 对小 batch（batch=2）比 BatchNorm 更稳定。"""

    def __init__(self, cin, cout, groups=8):
        super(GNBlock, self).__init__()
        g = max(1, min(int(groups), int(cout)))
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetGN(nn.Module):
    """与 step5_320_ceiling.UNet3 完全相同的拓扑，仅 BatchNorm -> GroupNorm。"""

    def __init__(self, in_ch=2, base=64, groups=8):
        super(UNetGN, self).__init__()
        self.enc1 = GNBlock(in_ch, base, groups)
        self.enc2 = GNBlock(base, base * 2, groups)
        self.enc3 = GNBlock(base * 2, base * 4, groups)
        self.pool = nn.MaxPool2d(2)
        self.bot = GNBlock(base * 4, base * 4, groups)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = GNBlock(base * 8, base * 4, groups)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = GNBlock(base * 6, base * 2, groups)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = GNBlock(base * 3, base, groups)
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


class CascadeNet(nn.Module):
    """K 级学习型级联（unrolled）：
        z_{n+1} = UNet_n( z_n - softplus(eta_n) * A^H (A z_n - y) )
    第 1 级即直接 U-Net（可复现 step5_320_ceiling 结果），后续级逐步细化。
    forward(z0, y, mask, k_max, burnin) 返回各级复数输出列表 outs[0..k_max-1]；
    burnin=True 时前 k_max-1 级输出 detach（解冻窗口只训练最后一级以稳定训练）。
    """

    def __init__(self, in_ch=2, base=64, K=4, groups=8, eta_init=0.5):
        super(CascadeNet, self).__init__()
        self.K = int(K)
        self.nets = nn.ModuleList(
            [UNetGN(in_ch=in_ch, base=base, groups=groups) for _ in range(self.K)]
        )
        self.eta = nn.ParameterList(
            [nn.Parameter(torch.full((1,), float(eta_init))) for _ in range(self.K)]
        )

    def dc(self, z_c, y, mask, n):
        eta = F.softplus(self.eta[n]).view(-1)
        return z_c - eta * C.ifft2_t((C.fft2_t(z_c) - y) * mask)

    def forward(self, z0, y, mask, k_max=None, burnin=False):
        k_max = self.K if k_max is None else int(k_max)
        z = z0
        outs = []
        for n in range(k_max):
            zc = C.to_c(z)
            d = self.dc(zc, y, mask, n)
            z = self.nets[n](C.to_2ch(d)).float()
            if burnin and n < k_max - 1:
                z = z.detach()
            outs.append(C.to_c(z))
        return outs
# ============================================================================
# 第 3 段 EMA / 级联深度监督损失 / lr 调度 / 单步训练 / 验证评测 / 级联表
# ============================================================================
class EMA(object):
    """指数滑动平均权重（decay=0.999）。每个优化器步后更新一次，
    保证 EMA 与工作权重几乎同步（滞后 ~0.1%），评测与 best 检查点
    均使用 EMA 权重（约等于当前工作权重）。"""

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = float(decay)
        self.step = 0
        self.reset_from(model)

    def reset_from(self, model):
        """以当前模型权重重建 shadow（用于 Phase B 广播后与旧格式续训）。"""
        with torch.no_grad():
            self.shadow = {}
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.detach().clone().cpu()
            for name, b in model.named_buffers():
                self.shadow[name] = b.detach().clone().cpu()
        self.step = 0

    def update(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(p.detach().cpu(), alpha=1.0 - self.decay)
            for name, b in model.named_buffers():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(b.detach().cpu(), alpha=1.0 - self.decay)
        self.step += 1

    def copy_to(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow:
                    p.copy_(self.shadow[name].to(p.device))
            for name, b in model.named_buffers():
                if name in self.shadow:
                    b.copy_(self.shadow[name].to(b.device))

    def state_dict(self):
        return {"decay": self.decay, "step": self.step,
                "shadow": {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, sd):
        self.decay = float(sd["decay"])
        self.step = int(sd.get("step", 0))
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}


def cascade_loss(outs_c, x_gt_c, y, mask, deep_w=DEEP_W):
    """级联深度监督损失（不切断梯度）：
        total = 末级损失 + deep_w * 前级平均损失。
    返回 (total, parts)，parts 为各级损失的标量张量列表。
    """
    parts = []
    for z in outs_c:
        lm, ls, ld = C.recon_loss_parts(z, x_gt_c, y, mask)
        parts.append(lm + ls + ld)
    total = parts[-1]
    if len(parts) > 1:
        total = total + deep_w * (sum(parts[:-1]) / float(len(parts) - 1))
    return total, parts


def set_lr(opt, epoch, args):
    lr = C.lr_at(epoch, args)
    for pg in opt.param_groups:
        pg["lr"] = lr
    return lr


def train_step(model, x, mask, device, phase, burnin, k_max, deep_w, accum, use_amp):
    """单个 micro-batch 前向 + 损失（已按累积数缩放，供外层 backward）。"""
    yb = C.fft2_t(x) * mask
    z0 = C.to_2ch(C.ifft2_t(yb))
    with torch.autocast(device_type="cuda", enabled=use_amp):
        outs = model(z0, yb, mask, k_max=k_max, burnin=burnin)
        loss, parts = cascade_loss(outs, x, yb, mask, deep_w=deep_w)
        loss = loss / float(accum)
    return loss, parts


def eval_val(model, store, val_sub, mask_store, mask_key, device, batch=8, max_n=64, k_max=None):
    """在验证子集上用当前权重评测末级幅值 PSNR / SSIM（fastMRI 口径）。"""
    rng = np.random.RandomState(123)
    pick = rng.choice(np.asarray(val_sub, dtype=np.int64),
                      size=min(int(max_n), len(val_sub)), replace=False).tolist()
    if not pick:
        return 0.0, 0.0
    mask = mask_store.get(mask_key, device=device)
    ssim = C.SSIMComputer()
    model.eval()
    ps_all, ss_all = [], []
    with torch.no_grad():
        for s0 in range(0, len(pick), batch):
            b_idx = pick[s0:s0 + batch]
            g = store.get_batch(b_idx, device=device)
            yb = C.fft2_t(g) * mask
            z0 = C.to_2ch(C.ifft2_t(yb))
            outs = model(z0, yb, mask, k_max=(model.K if k_max is None else int(k_max)))
            zk = outs[-1]
            ps = C.per_slice_psnr_full(zk, g).detach().cpu().numpy()
            ps_all.extend(float(v) for v in ps.tolist())
            gm = g.detach().cpu().numpy()
            xhm = zk.detach().cpu().numpy()
            for i in range(len(b_idx)):
                ss_all.append(ssim.compute(np.abs(gm[i]), np.abs(xhm[i])))
    return float(np.mean(ps_all)), float(np.mean(ss_all))


def cascade_table(model, store, val_sub, mask_store, mask_key, device, batch=4, ks=(1, 2, 3, 4)):
    """在验证子集上评测各级输出 PSNR，量化级联细化收益。"""
    rng = np.random.RandomState(42)
    pick = rng.choice(np.asarray(val_sub, dtype=np.int64),
                      size=min(32, len(val_sub)), replace=False).tolist()
    if not pick:
        return {int(k): 0.0 for k in ks}
    mask = mask_store.get(mask_key, device=device)
    model.eval()
    per = {int(k): [] for k in ks}
    with torch.no_grad():
        for s0 in range(0, len(pick), batch):
            b_idx = pick[s0:s0 + batch]
            g = store.get_batch(b_idx, device=device)
            yb = C.fft2_t(g) * mask
            z0 = C.to_2ch(C.ifft2_t(yb))
            outs = model(z0, yb, mask, k_max=max(int(k) for k in ks))
            for k in ks:
                ps = C.per_slice_psnr_full(outs[int(k) - 1], g).detach().cpu().numpy()
                per[int(k)].extend(float(v) for v in ps.tolist())
    res = {}
    for k in ks:
        res[int(k)] = round(float(np.mean(per[int(k)])), 4)
    log("CASCADE " + " ".join("K%d=%.2f" % (k, res[k]) for k in sorted(res)))
    return res
# ============================================================================
# 第 4 段 检查点 / 断点续训 / 主训练循环
# ============================================================================
def save_checkpoint(path, model, ema, opt, scaler, epoch, best, history,
                    cascade_full, warm_phase, args):
    """保存断点（工作权重 + EMA + 优化器 + scaler + 状态）。"""
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "ema_step": int(getattr(ema, "step", 0)),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "best": best,
        "history": history,
        "cascade_full": cascade_full,
        "warm_phase": bool(warm_phase),
        "args": vars(args),
    }, path)


def load_resume(path, model, ema, opt, scaler, device):
    """从 checkpoint_last.pt 恢复。EMA shadow 保持在 CPU，优化器状态搬回参数设备。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    ema.load_state_dict(ck["ema"])
    opt.load_state_dict(ck["opt"])
    with torch.no_grad():
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state.get(p)
                if not st:
                    continue
                for k, v in st.items():
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to(p.device)
    if "scaler" in ck:
        try:
            scaler.load_state_dict(ck["scaler"])
        except Exception as e:
            log("WARN scaler state not restored: %s" % str(e))
    if int(ck.get("ema_step", -1)) < 0:
        # 旧格式检查点：EMA 只在 epoch 末更新一次，影子权重陈旧
        # （实测 epoch 23 的 EMA K4=19.8 dB，而工作权重 K4=27.8 dB）。
        # 强制把 EMA 重置为当前工作权重，避免续训时评测继续滞后。
        ema.reset_from(model)
        log("resume: old checkpoint format (no ema_step) -> EMA reset to working weights")
    else:
        log("resume: ema_step=%d" % int(ck["ema_step"]))
    return ck


def _set_frozen(model, warm_phase):
    """阶段 A（warm_phase=True）：只训练第 1 级（含其 eta），其余级冻结。"""
    for n in range(model.K):
        freeze = bool(warm_phase) and n > 0
        model.nets[n].requires_grad_(not freeze)
        model.eta[n].requires_grad_(not freeze)


def _unfreeze_broadcast(model):
    """阶段切换：把训练好的第 1 级权重复制到所有级，并解冻全部参数。"""
    with torch.no_grad():
        w0 = copy.deepcopy(model.nets[0].state_dict())
        for n in range(1, model.K):
            model.nets[n].load_state_dict(copy.deepcopy(w0))
    for n in range(model.K):
        model.nets[n].requires_grad_(True)
        model.eta[n].requires_grad_(True)


def train_final(args):
    """CascadeNet 端到端训练（阶段 A 暖启动 -> 阶段 B 级联展开 + burnin）。

    返回 dict：
        status: OK / TRAIN_DIVERGED / OOM / INTERRUPTED
        best: {"val_psnr", "epoch"} / history / model / n_params / ...
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device=%s torch=%s" % (str(device), torch.__version__))
    set_seed(args.seed)

    meta = torch.load(META_PATH, map_location="cpu", weights_only=False)
    store = C.ChunkStore(meta)
    mask_store = C.MaskStore(meta["masks"])
    train_idx, val_idx, test_idx = C.load_split(meta)
    rng = np.random.RandomState(args.seed)
    val_sub = rng.choice(np.asarray(val_idx, dtype=np.int64),
                         size=min(int(args.val_max), len(val_idx)),
                         replace=False).tolist()
    log("split train=%d val=%d test=%d val_sub=%d"
        % (len(train_idx), len(val_idx), len(test_idx), len(val_sub)))

    model = CascadeNet(in_ch=2, base=args.base, K=args.K,
                       eta_init=args.eta_init).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log("model CascadeNet base=%d K=%d params=%d"
        % (args.base, args.K, n_params))

    # 不再从 step5_320_ceiling（BatchNorm）热启动：BN->GN 仅按形状搬运会把
    # BN running stats 错误载入 GN 仿射参数（实测 K1 只有 ~11 dB）。
    # 直接从随机初始化训练第 1 级（旧检查点的工作权重证明 20+ epoch 内
    # K1 可训练到 25+ dB，见 Phase A 的 K1 自检）。
    log("stage1 random init (BN checkpoint warm-start disabled: BN->GN mismatch)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=WEIGHT_DECAY)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model)

    best = {"val_psnr": -1.0, "epoch": 0}
    history = {"loss": [], "val_psnr": [], "val_ssim": [], "lr": []}
    cascade_full = None
    start_epoch = 1
    was_warm = True
    ckpt_dir = os.path.join(RUN_ROOT, args.mode)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_last = os.path.join(ckpt_dir, "checkpoint_last.pt")
    ckpt_best = os.path.join(ckpt_dir, "checkpoint_best.pt")

    if args.resume and os.path.exists(ckpt_last):
        ck = load_resume(ckpt_last, model, ema, opt, scaler, device)
        start_epoch = int(ck.get("epoch", 0)) + 1
        best = ck.get("best") or best
        history = ck.get("history") or history
        cascade_full = ck.get("cascade_full")
        was_warm = bool(ck.get("warm_phase", True))
        log("resumed from epoch %d (warm_phase=%s)" % (start_epoch - 1, was_warm))
    _set_frozen(model, was_warm)

    t0 = time.perf_counter()
    no_impr = 0
    last_epoch = 0
    accum = max(1, int(args.grad_accum))
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            warm_phase = epoch <= int(args.warm_cascades)
            if not warm_phase and was_warm:
                _unfreeze_broadcast(model)
                ema = EMA(model)
                log("PHASE B start: broadcast stage1 -> all stages, burnin=%d epochs"
                    % int(args.unfreeze))
            was_warm = warm_phase

            lr = set_lr(opt, epoch, args)
            burnin = (not warm_phase) and (
                epoch - int(args.warm_cascades)) <= int(args.unfreeze)
            k_max = 1 if warm_phase else int(model.K)
            model.train()
            gen = torch.Generator().manual_seed(args.seed * 10000 + epoch)
            perm = torch.randperm(len(train_idx), generator=gen).cpu().numpy()
            losses = []
            n_step = 0
            opt.zero_grad()
            for s in range(0, len(perm), args.batch):
                b_idx = [int(i) for i in train_idx[perm[s:s + args.batch]]]
                x = store.get_batch(b_idx, device=device)
                if args.aug:
                    if random.random() < 0.5:
                        x = torch.flip(x, dims=[-1])
                    if random.random() < 0.5:
                        x = torch.flip(x, dims=[-2])
                mk = TRAIN_MASKS_DEFAULT[n_step % len(TRAIN_MASKS_DEFAULT)]
                mask = mask_store.get(mk, device=device)
                loss, _parts = train_step(
                    model, x, mask, device, "A" if warm_phase else "B",
                    burnin, k_max, DEEP_W, accum, use_amp)
                if not torch.isfinite(loss):
                    log("TRAIN epoch %d NON-FINITE loss %.4e -> abort"
                        % (epoch, float(loss)))
                    return {"status": "TRAIN_DIVERGED", "epoch": epoch,
                            "last_epoch": epoch, "history": history,
                            "model": model, "n_params": n_params,
                            "best": best}
                scaler.scale(loss).backward()
                n_step += 1
                if n_step % accum == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    ema.update(model)
                    losses.append(float(loss.item()) * float(accum))
            if n_step % accum != 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                ema.update(model)
                losses.append(float(loss.item()) * float(accum))
            mean_loss = float(np.mean(losses)) if losses else 0.0
            last_epoch = epoch
            # ---- 验证（用 EMA 权重评测，之后还原工作权重）----
            # EMA 已在每个优化器步后更新，滞后 ~0.1%，评测即当前工作权重
            work_state = copy.deepcopy(model.state_dict())
            ema.copy_to(model)
            val_psnr, val_ssim = eval_val(
                model, store, val_sub, mask_store, MAIN_MASK, device,
                batch=args.batch, max_n=args.val_max,
                k_max=(1 if warm_phase else None))
            if warm_phase:
                casc = {"1": round(float(val_psnr), 4)}
                log("CASCADE warm-phase K1=%.2f (later stages frozen)"
                    % casc["1"])
                if not args.smoke and val_psnr < 20.0:
                    log("WARN Phase A epoch %d: K1 val PSNR %.2f dB < 20 dB "
                        "(stage1 未学起来，请检查 lr/batch)" % (epoch, val_psnr))
            else:
                ks = (1, 2, 3, 4) if model.K >= 4 else tuple(
                    range(1, model.K + 1))
                casc = cascade_table(model, store, val_sub, mask_store,
                                     MAIN_MASK, device, batch=args.batch,
                                     ks=ks)
            if val_psnr > best["val_psnr"]:
                best["val_psnr"] = float(val_psnr)
                best["epoch"] = int(epoch)
                no_impr = 0
                ema.copy_to(model)
                torch.save({"state_dict": model.state_dict(),
                            "epoch": int(epoch),
                            "val_psnr": float(val_psnr),
                            "n_params": int(n_params)}, ckpt_best)
                log("BEST val PSNR %.2f dB @ epoch %d (SSIM %.4f)"
                    % (val_psnr, epoch, val_ssim))
            else:
                no_impr += 1
            model.load_state_dict(work_state)
            del work_state
            cascade_full = casc

            history["loss"].append(round(mean_loss, 6))
            history["val_psnr"].append(round(float(val_psnr), 4))
            history["val_ssim"].append(round(float(val_ssim), 4))
            history["lr"].append(round(float(lr), 8))
            save_checkpoint(ckpt_last, model, ema, opt, scaler, epoch,
                            best, history, cascade_full, warm_phase, args)
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                log("TRAIN epoch %d/%d loss=%.5f lr=%.2e val_psnr=%.2f dB "
                    "val_ssim=%.4f (best %.2f @ %d)"
                    % (epoch, args.epochs, mean_loss, lr, val_psnr, val_ssim,
                       best["val_psnr"], best["epoch"]))
            if epoch >= args.min_epochs and no_impr >= args.patience:
                log("early stop at epoch %d (no improvement for %d epochs)"
                    % (epoch, args.patience))
                break
    except torch.cuda.OutOfMemoryError:
        log("OOM: GPU 显存不足。请减小 --batch 或增大 --grad-accum "
            "(batch*grad_accum 有效批量不变)，然后 --resume 继续")
        return {"status": "OOM", "last_epoch": last_epoch, "history": history,
                "model": model, "n_params": n_params, "best": best,
                "cascade_full": cascade_full}
    except KeyboardInterrupt:
        log("interrupted by user; checkpoint_last.pt 已保存，可用 --resume 续训")
        return {"status": "INTERRUPTED", "last_epoch": last_epoch,
                "history": history, "model": model, "n_params": n_params,
                "best": best, "cascade_full": cascade_full}

    train_s = time.perf_counter() - t0
    log("TRAIN done in %.1fs (%s) best val PSNR %.2f dB @ epoch %d"
        % (train_s, time_str(train_s), best["val_psnr"], best["epoch"]))
    return {"status": "OK", "train_sec": train_s, "best": best,
            "last_epoch": last_epoch, "history": history, "model": model,
            "n_params": n_params, "cascade_full": cascade_full,
            "val_sub": val_sub}
# ============================================================================
# 第 5 段 测试评测 / 图 / 报告 / 入口
# ============================================================================
def eval_test(model, store, test_idx, mask_store, mask_key, device, batch=16):
    """全量测试切片评测（幅值 PSNR + numpy SSIM，fastMRI 口径）。

    返回 dict：n / psnr_full / ssim / x_hat（前 8 张，CPU complex）。
    """
    mask = mask_store.get(mask_key, device=device)
    model.eval()
    ssim = C.SSIMComputer()
    ps_all, ss_all = [], []
    xh_keep = []
    with torch.no_grad():
        for s0 in range(0, len(test_idx), batch):
            b_idx = [int(i) for i in test_idx[s0:s0 + batch]]
            g = store.get_batch(b_idx, device=device)
            yb = C.fft2_t(g) * mask
            z0 = C.to_2ch(C.ifft2_t(yb))
            outs = model(z0, yb, mask, k_max=model.K)
            zk = outs[-1]
            ps = C.per_slice_psnr_full(zk, g).detach().cpu().numpy()
            ps_all.extend(float(v) for v in ps.tolist())
            xh = zk.detach().cpu()
            gm = g.detach().cpu().numpy()
            xhm = xh.numpy()
            for i in range(len(b_idx)):
                ss_all.append(ssim.compute(np.abs(gm[i]), np.abs(xhm[i])))
            if len(xh_keep) < 8:
                need = 8 - len(xh_keep)
                xh_keep.append(xh[:need])
    ps_all = np.asarray(ps_all, dtype=np.float64)
    ss_all = np.asarray(ss_all, dtype=np.float64)
    x_hat = torch.cat(xh_keep, dim=0) if xh_keep else torch.empty(0)
    return {"n": int(len(ps_all)), "psnr_full": C.stat_arr(ps_all),
            "ssim": C.stat_arr(ss_all), "x_hat": x_hat}


def save_figs(report, res, store, test_idx, mask_store, device, fig_dir):
    """生成 4 张图：训练曲线 / ZF 对齐 / 重建示例 / 级联细化。图内用英文标签。"""
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
        hist = (report.get("train") or {}).get("history") or {}
        if hist.get("loss"):
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            ax[0].plot(hist["loss"])
            ax[0].set_xlabel("epoch")
            ax[0].set_ylabel("train loss")
            ax[0].set_title("training loss")
            ax[1].plot(hist["val_psnr"])
            ax[1].set_xlabel("epoch")
            ax[1].set_ylabel("val PSNR (dB)")
            ax[1].set_title("val PSNR (magnitude, per-image peak)")
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
        ax.bar(x - w / 2, refs, w, label="reference (official recipe)")
        ax.bar(x + w / 2, ours, w, label="recomputed (this script)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("zero-filled PSNR (dB)")
        ax.set_title("ZF alignment (test slices)")
        ax.legend()
        ax.grid(alpha=0.3)
        save(fig, "fig2_zf_alignment.png")
    except Exception as e:
        log("WARN fig2 failed: %s: %s" % (type(e).__name__, str(e)))

    try:
        n_show = min(4, int(len(test_idx)))
        idx4 = [int(i) for i in test_idx[:n_show]]
        g = store.get_batch(idx4)
        mask = mask_store.get(MAIN_MASK, device=device)
        yb = C.fft2_t(g.to(device)) * mask
        zf = C.ifft2_t(yb).detach().cpu()
        xh = res.get("x_hat")
        if xh is not None and xh.shape[0] >= n_show:
            fig, axes = plt.subplots(n_show, 3, figsize=(10.5, 3.6 * n_show))
            if n_show == 1:
                axes = axes[None, :]
            for i in range(n_show):
                gm = np.abs(g[i].numpy())
                zm = np.abs(zf[i].numpy())
                xm = np.abs(xh[i].numpy())
                vmax = float(np.percentile(gm, 99.5))
                axes[i, 0].imshow(gm, cmap="gray", vmax=vmax)
                axes[i, 0].set_title("GT")
                axes[i, 1].imshow(zm, cmap="gray", vmax=vmax)
                axes[i, 1].set_title("ZF")
                axes[i, 2].imshow(xm, cmap="gray", vmax=vmax)
                axes[i, 2].set_title("Ours")
                for j in range(3):
                    axes[i, j].axis("off")
            fig.tight_layout()
            save(fig, "fig3_recon.png")
    except Exception as e:
        log("WARN fig3 failed: %s: %s" % (type(e).__name__, str(e)))

    try:
        cc = report.get("cascade_full") or {}
        if cc:
            ks = sorted(int(x) for x in cc)
            vals = [float(cc[str(k)]) for k in ks]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar([str(k) for k in ks], vals)
            for k, v in zip(ks, vals):
                ax.text(k - 1, v + 0.05, "%.2f" % v, ha="center", fontsize=9)
            ax.set_xlabel("cascade depth K")
            ax.set_ylabel("val PSNR (dB)")
            ax.set_title("cascade refinement (best weights, val subset)")
            ax.grid(alpha=0.3, axis="y")
            save(fig, "fig4_cascade.png")
    except Exception as e:
        log("WARN fig4 failed: %s: %s" % (type(e).__name__, str(e)))
    return made


def parse_args():
    p = argparse.ArgumentParser(
        description="step5_train_final: fastMRI knee 320 CascadeNet 正式训练")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true", help="冒烟运行（2 epoch）")
    g.add_argument("--eval-only", action="store_true",
                   help="跳过训练，仅评测 checkpoint_best.pt")
    p.add_argument("--resume", action="store_true",
                   help="从 checkpoint_last.pt 断点续训")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--warm-cascades", type=int, default=10,
                   help="阶段 A 只训练第 1 级的 epoch 数")
    p.add_argument("--unfreeze", type=int, default=5,
                   help="阶段 B 开始时 burnin（逐级 detach）的 epoch 数")
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--min-epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=2, help="micro-batch")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="梯度累积步数（有效批量 = batch*grad_accum）")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--K", type=int, default=4, help="级联深度")
    p.add_argument("--eta-init", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-max", type=int, default=96)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--aug", type=int, default=1, help="随机翻转增强")
    args = p.parse_args()
    if args.smoke:
        args.epochs = 2
        args.warmup = 1
        args.warm_cascades = 1
        args.unfreeze = 1
        args.base = 48
        args.batch = 4
        args.grad_accum = 2
        args.val_max = 32
        args.patience = 1
        args.min_epochs = 1
    args.amp = bool(args.amp)
    args.aug = bool(args.aug)
    args.mode = "smoke" if args.smoke else ("eval" if args.eval_only else "full")
    return args
# ============================================================================
# 第 5 段（续） 报告 / 摘要 / main / 入口
# ============================================================================

def _close_log():
    global _LOG_FH
    if _LOG_FH is not None:
        try:
            _LOG_FH.close()
        except Exception:
            pass
        _LOG_FH = None


def _fmt(x, nd=2):
    if x is None:
        return "n/a"
    return ("%." + str(nd) + "f") % float(x)


def _save_report(report):
    p = os.path.join(HERE, "step5_train_final_%s_report.json" % report["mode"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=C.json_default)
    log("report written: %s" % p)
    return p


def _write_summary(report, t0):
    L = []
    A = L.append
    v = report["verdict"]
    A("=" * 74)
    A("STEP5_TRAIN_FINAL SUMMARY  (mode=%s, %s)"
      % (report["mode"], report["timestamp"]))
    A("=" * 74)
    A("VERDICT: %s" % v["status"])
    A("  aligned (ZF)     : %s" % v["aligned"])
    A("  test PSNR r4_s42 : %s dB (SSIM %s)"
      % (_fmt(v.get("test_psnr")), _fmt(v.get("test_ssim"), 4)))
    A("  gain vs ZF       : %s dB" % _fmt(v.get("gain_vs_zf_db")))
    A("  headroom K-K1    : %s dB (val subset)" % _fmt(v.get("headroom_db")))
    A("-" * 74)
    A("1. DATA (fastMRI knee singlecoil_val, 320x320 complex)")
    st = report["data"]["stats"]
    A("   split train/val/test slices: %d/%d/%d"
      % (report["data"]["n_train"], report["data"]["n_val"],
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
             row["d_psnr"], row["d_ssim"],
             "OK" if row["aligned"] else "MISMATCH"))
    A("-" * 74)
    ar = report["args"]
    A("3. TRAINING (CascadeNet base=%d K=%d, lr=%.1e, warm-cascades=%d, unfreeze=%d)"
      % (ar["base"], ar["K"], ar["lr"], ar["warm_cascades"], ar["unfreeze"]))
    tr = report.get("train") or {}
    if tr.get("status") == "OK":
        A("   params: %d | best val PSNR: %.2f dB @ epoch %d | last epoch: %d | train time: %s"
          % (tr["n_params"], tr["best"]["val_psnr"], tr["best"]["epoch"],
             tr["last_epoch"], time_str(float(tr["train_sec"]))))
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
    A("5. CASCADE REFINEMENT (val subset, best weights)")
    cc = report.get("cascade_full") or {}
    if cc:
        for k in sorted(int(x) for x in cc):
            A("   K=%d : %.2f dB" % (k, cc[str(k)]))
    else:
        A("   n/a")
    A("-" * 74)
    A("6. VERDICT RATIONALE")
    A("   PASS   : aligned and train OK and PSNR>=%.1f and SSIM>=%.2f and gain>=%+.1f dB"
      % (PASS_PSNR, PASS_SSIM, MIN_GAIN_DB))
    A("   REVIEW : aligned and train OK and PSNR %.1f-%.1f"
      % (MID_PSNR, PASS_PSNR))
    A("   FAIL   : otherwise (or ZF mismatch / training incomplete)")
    A("   Note: --smoke is expected to stay REVIEW/FAIL; the full run decides.")
    A("=" * 74)
    A("elapsed %.1f s | report: %s"
      % (time.perf_counter() - t0,
         os.path.join(HERE, "step5_train_final_%s_report.json" % report["mode"])))
    txt = "\n".join(L)
    p = os.path.join(HERE, "step5_train_final_%s_summary.txt" % report["mode"])
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    return p


def main():
    args = parse_args()
    t0 = time.perf_counter()
    mode = args.mode
    ckpt_dir = os.path.join(RUN_ROOT, mode)
    fig_dir = os.path.join(FIG_ROOT, mode)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    global _LOG_FH
    _LOG_FH = open(os.path.join(HERE, "step5_train_final_%s_stdout.log" % mode),
                   "w", encoding="utf-8", errors="replace")
    log("step5_train_final mode=%s seed=%d amp=%s lr=%.1e K=%d base=%d"
        % (mode, args.seed, args.amp, args.lr, args.K, args.base))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device=%s torch=%s" % (dev, torch.__version__))
    if not os.path.exists(META_PATH):
        log("META NOT FOUND: %s" % META_PATH)
        _close_log()
        return 1

    meta = torch.load(META_PATH, map_location="cpu", weights_only=False)
    store = C.ChunkStore(meta)
    mask_store = C.MaskStore(meta["masks"])
    train_idx, val_idx, test_idx = C.load_split(meta)
    log("split train=%d val=%d test=%d"
        % (len(train_idx), len(val_idx), len(test_idx)))
    ds = C.data_stats(store, test_idx, max_n=512, seed=0)
    log("data stats n=%d fg_p50=%.3f max_p50=%.3f snr_p50=%.2f"
        % (ds["n"], ds["foreground_fraction"]["p50"], ds["per_slice_max"]["p50"],
           ds["gt_fg_snr_db"]["p50"]))
    data_common = {"n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                   "n_test": int(len(test_idx)), "stats": ds}
    zf = C.run_zf_alignment(store, test_idx, mask_store)

    def fail_report(reason):
        return {"script": "step5_train_final.py",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode,
                "args": C.sanitize(vars(args)), "data": data_common, "zf": zf,
                "train": None, "test": None, "cascade_full": None,
                "verdict": {"status": "FAIL", "aligned": bool(zf["ok"]),
                            "test_psnr": None, "test_ssim": None,
                            "gain_vs_zf_db": None, "headroom_db": None,
                            "train_status": reason}}

    if not zf["ok"]:
        log("ZF alignment FAILED -> abort (data pipeline mismatch)")
        rpt = fail_report("zf-mismatch")
        _save_report(rpt)
        _write_summary(rpt, t0)
        _close_log()
        return 1

    res = None
    model = None
    if mode == "eval":
        ck_path = None
        for cand in (os.path.join(RUN_ROOT, "full", "checkpoint_best.pt"),
                     os.path.join(RUN_ROOT, "eval", "checkpoint_best.pt")):
            if os.path.exists(cand):
                ck_path = cand
                break
        if ck_path is None:
            log("eval-only: checkpoint_best.pt not found under %s"
                % os.path.join(RUN_ROOT, "full"))
            rpt = fail_report("no-checkpoint")
            _save_report(rpt)
            _write_summary(rpt, t0)
            _close_log()
            return 1
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = CascadeNet(in_ch=2, base=args.base, K=args.K,
                           eta_init=args.eta_init).to(dev)
        model.load_state_dict(ck["state_dict"], strict=True)
        model.eval()
        val_psnr = float(ck.get("val_psnr", -1.0))
        epoch = int(ck.get("epoch", 0))
        res = {"status": "OK", "n_params": int(ck.get("n_params", 0)),
               "best": {"val_psnr": val_psnr, "epoch": epoch},
               "last_epoch": epoch, "train_sec": 0.0, "history": {},
               "cascade_full": None}
        log("eval-only: loaded %s (epoch=%d val_psnr=%.2f)"
            % (ck_path, epoch, val_psnr))
    else:
        res = train_final(args)
        model = res.get("model")
        if model is None:
            log("train_final returned no model -> FAIL")
            rpt = fail_report(res.get("status", "no-model"))
            _save_report(rpt)
            _write_summary(rpt, t0)
            _close_log()
            return 1
        if res["status"] == "OK":
            ck_best = os.path.join(ckpt_dir, "checkpoint_best.pt")
            if os.path.exists(ck_best):
                try:
                    ck = torch.load(ck_best, map_location="cpu",
                                    weights_only=False)
                    m2 = CascadeNet(in_ch=2, base=args.base, K=args.K,
                                    eta_init=args.eta_init).to(dev)
                    m2.load_state_dict(ck["state_dict"], strict=True)
                    m2.eval()
                    model = m2
                    log("test eval uses best EMA checkpoint (epoch=%d val_psnr=%.2f)"
                        % (int(ck.get("epoch", 0)),
                           float(ck.get("val_psnr", -1.0))))
                except Exception as e:
                    log("WARN best checkpoint load failed: %s: %s (use working weights)"
                        % (type(e).__name__, str(e)))
                    model = res["model"]
        else:
            log("training status=%s -> evaluate working weights (report, verdict FAIL)"
                % res["status"])
            model = res["model"]

    evals = {}
    for mk in TEST_MASKS:
        evals[mk] = eval_test(model, store, test_idx, mask_store, mk, dev,
                              batch=8)
        e = evals[mk]
        log("TEST %-10s psnr=%.2f+-%.2f ssim=%.4f+-%.4f (n=%d)"
            % (mk, e["psnr_full"]["mean"], e["psnr_full"]["std"],
               e["ssim"]["mean"], e["ssim"]["std"], e["n"]))

    te = evals[MAIN_MASK]
    zf_main = next(r for r in zf["rows"] if r["mask"] == MAIN_MASK)
    psnr = float(te["psnr_full"]["mean"])
    ssim = float(te["ssim"]["mean"])
    gain = psnr - float(zf_main["psnr"]["mean"])
    cascade_full = res.get("cascade_full")
    if cascade_full:
        cascade_full = {str(k): float(v) for k, v in cascade_full.items()}
    hr = None
    if cascade_full and "1" in cascade_full and str(model.K) in cascade_full:
        hr = float(cascade_full[str(model.K)]) - float(cascade_full["1"])
    ok_train = bool(res.get("status") == "OK")
    if (zf["ok"] and ok_train and psnr >= PASS_PSNR and ssim >= PASS_SSIM
            and gain >= MIN_GAIN_DB):
        verdict = "PASS"
    elif zf["ok"] and ok_train and psnr >= MID_PSNR:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"
    log("VERDICT: %s (PSNR %.2f SSIM %.4f gain %+.2f dB headroom %s)"
        % (verdict, psnr, ssim, gain,
           "n/a" if hr is None else "%+.2f" % hr))

    hist = res.get("history") or {}
    if hist.get("loss"):
        import csv
        with open(os.path.join(ckpt_dir, "history.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss", "val_psnr"])
            for i in range(len(hist["loss"])):
                w.writerow([i + 1, hist["loss"][i], hist["val_psnr"][i]])
    cfg = C.sanitize(vars(args))
    cfg["main_mask"] = MAIN_MASK
    cfg["test_masks"] = TEST_MASKS
    cfg["data_files"] = "fastmri_320_meta.pt + fastmri_320_gt_chunk_*.pt"
    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    report = {
        "script": "step5_train_final.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "args": C.sanitize(vars(args)),
        "device": str(dev),
        "torch": torch.__version__,
        "data": data_common,
        "zf": zf,
        "train": {k: v for k, v in res.items() if k != "model"},
        "test": {mk: {"psnr_full": evals[mk]["psnr_full"],
                      "ssim": evals[mk]["ssim"],
                      "n": evals[mk]["n"]} for mk in TEST_MASKS},
        "cascade_full": cascade_full,
        "verdict": {"status": verdict, "aligned": bool(zf["ok"]),
                    "train_status": res.get("status"),
                    "test_psnr": round(psnr, 4), "test_ssim": round(ssim, 4),
                    "gain_vs_zf_db": round(gain, 4),
                    "headroom_db": None if hr is None else round(hr, 4)},
    }
    try:
        made = save_figs(report, evals[MAIN_MASK], store, test_idx,
                         mask_store, dev, fig_dir)
        report["figures"] = made
    except Exception as e:
        log("WARN figures: %s: %s" % (type(e).__name__, str(e)))
        report["figures"] = []
    _save_report(report)
    _write_summary(report, t0)
    log("done in %.1fs | verdict: %s" % (time.perf_counter() - t0, verdict))
    _close_log()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

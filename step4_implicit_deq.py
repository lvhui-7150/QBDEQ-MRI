# -*- coding: utf-8 -*-
"""
Step 4 (重写版) -- QB-DEQ 隐式不动点机制实验 (Bregman vs Euclidean, fastMRI 128)
================================================================================

为什么重写 (前一轮诊断结论, 见 _probe_route_summary.txt / _probe_nan_summary.txt
/ _probe_contrast_summary.txt / _probe_align2_summary.txt)：

    旧 step4 把 28.7 万参数的 U-Net 放进不动点算子 S(z)=Gauge(D(DC(z))) 里，
    探针实验证明该算子谱半径 > 1，迭代 200 次仍不收敛或直接 NaN。
    这不是普通代码 bug，而正是论文研究的核心现象：Euclidean 型不动点算子不稳定；
    收敛性必须依靠 Bregman 结构 + gauge 投影 + 谱归一化算子。

本脚本 = step4a（机制实验，确定性、快速，对应论文实验2 / 路线图 Phase 3）：

    算子   S(z) = Gauge( z - eta * ( A*(A z - y) + alpha * reg_term ) )
      brg_gauge : reg_term = grad h( W_theta(z) ),  h(z)=sum_i|z_i|^p/p, p=4
      euc_gauge : reg_term = W_theta(z)（无 Bregman 结构），有 gauge
      euc_nog   : reg_term = W_theta(z)，无 gauge
    W_theta    = SNRegNet：真复卷积 + 谱归一化 (Lip(W)<=1, 论文 eq:learned_F)
    求解方案  : plain 迭代 / Anderson(m=5, KKT 逐样本)，max_iters=200, tol=1e-6
    强度      : gentle (alpha=0.3, eta=0.3) / strong (alpha=1.0, eta=1.0)
    p 幂核扫描: p in {2,3,4,6}（p=2 退化为 Euclidean 核）
    谱半径估计: 在收敛不动点上用 power iteration 估 rho(J_S)

    注意：本步 W_theta 是随机初始化、未训练的（与旧 theorem_experiments.py
    experiment6 的机制验证一致）。因此重建质量约等于零填充是正常预期；
    质量提升留到 step4b（隐式训练）。

验收线（决定是否进入 step4b）:
    M1 Bregman 收敛: brg_gauge 在 >=90% 的设置下 class=good (DC<0.2 且 PSNR>=ZF-1dB)
    M2 Euclidean 不稳定: strong 设置下 euc 至少 1 组发散/NaN

运行:
    python step4_implicit_deq.py

输出（我会读取的文本）:
    step4_summary.txt  中文表格 + 结论 + verdict
    step4_report.json  机器可读完整结果
    step4_figs/        收敛曲线 + 重建对比图（供你查看/写论文）
"""

import os
import sys
import json
import time
import math

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
import torch.nn.functional as F

import step3_unrolled_fix as base   # 数据 + FFT + 指标（device 已修好）

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 固定配置 ---------------------------------------------------------------
SEED = 42
N_SLICES = 8
MAX_ITERS = 200
TOL = 1e-6
ANDERSON_M = 5
ANDERSON_BETA = 1.0
KKT_LAM = 1e-4
DC_GOOD = 0.2          # 数据一致性残差合格线
PSNR_FLOOR = -20.0     # 低于此 PSNR 视为发散
P_DEFAULT = 4.0
P_SWEEP = [2.0, 3.0, 4.0, 6.0]
SETTINGS = [
    {"name": "gentle", "alpha": 0.3, "eta": 0.3},
    {"name": "strong", "alpha": 1.0, "eta": 1.0},
]
METHODS = ["euc_nog", "euc_gauge", "brg_gauge"]
SCHEMES = ["plain", "anderson5"]
RATE_MASKS = [(4, "r4_s42"), (8, "r8_s42")]

# 需要保留不动点张量的配置（供谱半径估计与出图用）
KEEP_Z_KEYS = {
    (4, "gentle", "brg_gauge", "anderson5"),
    (4, "gentle", "euc_gauge", "anderson5"),
    (4, "gentle", "euc_nog", "anderson5"),
    (4, "strong", "brg_gauge", "plain"),
    (4, "strong", "euc_gauge", "plain"),
    (4, "strong", "euc_nog", "plain"),
}

HERE = os.path.dirname(os.path.abspath(__file__))
PREPARED = os.path.join(HERE, "fastmri_128_prepared.pt")
FIG_DIR = os.path.join(HERE, "step4_figs")
REPORT_PATH = os.path.join(HERE, "step4_report.json")
SUMMARY_PATH = os.path.join(HERE, "step4_summary.txt")
os.makedirs(FIG_DIR, exist_ok=True)

report = {"config": {}, "rows": [], "p_sweep": [], "rho": [], "verdict": {}, "issues": []}


def line(tag, msg):
    print(f"[{tag}] {msg}")


def add_issue(level, msg):
    report["issues"].append({"level": level, "msg": msg})
    print(f"[{level}] {msg}")


# ---- Bregman 可分离幂核（逐像素, 2ch 实数表示, 论文 eq: separable kernel） ----
def norm_val(z2):
    """逐像素模长 (B,H,W)。"""
    return (z2[:, 0].square() + z2[:, 1].square() + 1e-8).sqrt()


def grad_kernel(z2, p):
    """grad h(z)_i = |z_i|^(p-2) z_i, 逐像素可分离。p=2 时退化为恒等。"""
    return norm_val(z2).pow(p - 2.0).unsqueeze(1) * z2


# ---- gauge 投影（论文: 把最大模像素相位旋转为 0） -----------------------------
def gauge_fix_batch(z2):
    """(B,2,H,W) -> (B,2,H,W)，每样本独立做全局相位旋转。"""
    B = z2.shape[0]
    mag = (z2[:, 0].square() + z2[:, 1].square()).reshape(B, -1)
    idx = mag.argmax(dim=1)
    zf = z2.reshape(B, 2, -1)
    ar = torch.arange(B, device=z2.device)
    ph = torch.atan2(zf[ar, 1, idx], zf[ar, 0, idx])
    c = torch.cos(-ph).view(B, 1, 1, 1)
    s = torch.sin(-ph).view(B, 1, 1, 1)
    return torch.cat([z2[:, 0:1] * c - z2[:, 1:2] * s,
                      z2[:, 0:1] * s + z2[:, 1:2] * c], dim=1)


# ---- 相位对齐：把整个问题旋转到"零填充最大像素相位=0"的规范 ------------------
def phase_align(z0, y, gt):
    """返回 (z0', y', gt')，三者共享同一全局相位规范，使比较/质量指标有意义。"""
    zc = base.to_c(z0)
    idx = zc.abs().reshape(zc.shape[0], -1).argmax(dim=1)
    zf = zc.reshape(zc.shape[0], -1)
    ar = torch.arange(zc.shape[0], device=zc.device)
    ph = torch.angle(zf[ar, idx])
    rot = torch.exp(-1j * ph)
    y_rot = y * rot.view(-1, 1, 1)
    gt_rot = gt * rot.view(-1, 1, 1)
    z0_rot = base.to_2ch(base.adjoint(y_rot))
    return z0_rot, y_rot, gt_rot


# ---- 不动点算子 S（论文 Alg1 的单元） ----------------------------------------
def S_op(z2, y, mask, reg, alpha, eta, method, p):
    """一步迭代：S(z) = Gauge( z - eta*( A*(Az-y) + alpha*reg_term ) )。

    brg_gauge: reg_term = grad h(W(z))；euc_*: reg_term = W(z)。
    """
    r = reg(z2)
    if method.startswith("brg"):
        reg_term = alpha * grad_kernel(r, p)
    else:
        reg_term = alpha * r
    dc_grad = base.to_2ch(base.adjoint((base.fwd_kspace(base.to_c(z2)) - y) * mask))
    out = z2 - eta * (dc_grad + reg_term)
    if method.endswith("gauge"):
        out = gauge_fix_batch(out)
    return out


# ---- Anderson 加速（KKT 逐样本, 论文 Alg1 L12-17） ---------------------------
def anderson_batched(x_hist, s_hist, B, device):
    """从最近 n 对 (x_i, S(x_i)) 求混合系数 alpha，返回 beta*sum(a_i*S(x_i))
    + (1-beta)*sum(a_i*x_i)。失败时回退到最后一次 S 输出。"""
    n = len(x_hist)
    fs = [(si - xi).reshape(B, -1) for xi, si in zip(x_hist, s_hist)]
    F = torch.stack(fs, dim=1)                                      # (B, n, N)
    G = torch.bmm(F, F.transpose(1, 2)) + KKT_LAM * torch.eye(n, device=device).unsqueeze(0)
    ones = torch.ones(B, n, 1, device=device)
    M = torch.cat([torch.cat([G, ones], dim=2),
                   torch.cat([ones.transpose(1, 2), torch.zeros(B, 1, 1, device=device)], dim=2)], dim=1)
    rhs = torch.zeros(B, n + 1, 1, device=device)
    rhs[:, -1, 0] = 1.0
    try:
        a = torch.linalg.solve(M, rhs)[:, :n, 0]                    # (B, n)
        if not torch.isfinite(a).all():
            return s_hist[-1]
    except Exception:
        return s_hist[-1]
    X = torch.stack([xi.reshape(B, -1) for xi in x_hist], dim=1)    # (B, n, N)
    SX = torch.stack([si.reshape(B, -1) for si in s_hist], dim=1)
    ax = torch.bmm(X.transpose(1, 2), a.unsqueeze(2)).squeeze(2)
    as_ = torch.bmm(SX.transpose(1, 2), a.unsqueeze(2)).squeeze(2)
    return (ANDERSON_BETA * as_ + (1.0 - ANDERSON_BETA) * ax).reshape_as(s_hist[-1])


# ---- 不动点求解 --------------------------------------------------------------
def solve_fixed_point(z0, y, mask, reg, alpha, eta, method, p, scheme,
                      max_iters=MAX_ITERS, tol=TOL):
    """plain 或 anderson5 迭代。返回 (z, status, it_fin, rels)。"""
    B = z0.shape[0]
    device = z0.device
    z = z0.clone()
    x_hist, s_hist = [], []
    rels = []
    status, it_fin = "maxiter", max_iters
    for it in range(1, max_iters + 1):
        z_prev = z
        with torch.no_grad():
            s = S_op(z_prev, y, mask, reg, alpha, eta, method, p)
        if scheme == "plain":
            z_new = s
        else:
            x_hist.append(z_prev.detach())
            s_hist.append(s.detach())
            if len(x_hist) > ANDERSON_M:
                x_hist.pop(0)
                s_hist.pop(0)
            if len(x_hist) >= 2:
                z_new = anderson_batched(x_hist, s_hist, B, device)
            else:
                z_new = s
            if not torch.isfinite(z_new).all():
                z_new = s
        rel = float((z_new - z_prev).abs().flatten(1).norm(dim=1).max().item()
                    / (z_prev.abs().flatten(1).norm(dim=1).max().item() + 1e-8))
        rels.append(rel)
        z = z_new
        if not math.isfinite(rel):
            status, it_fin = "nan", it
            break
        if rel < tol:
            status, it_fin = "conv", it
            break
    return z, status, it_fin, rels


# ---- 谱半径估计（power iteration 估 rho(J_S), 论文 Thm. quotient contraction）--
def estimate_rho(z_fp, y, mask, reg, alpha, eta, method, p, power_iters=6):
    """在收敛不动点 z_fp(单样本) 上用 |Jv|/|v| 的 power iteration 估计谱半径。"""
    if z_fp.shape[0] != 1:
        z_fp = z_fp[:1]
    if not torch.isfinite(z_fp).all():
        return None
    try:
        z = z_fp.clone().detach().requires_grad_(True)
        v = torch.randn_like(z)
        v = v / (v.flatten().norm() + 1e-12)
        ratios = []
        for _ in range(power_iters):
            Sz = S_op(z, y[:1], mask, reg, alpha, eta, method, p)
            Jtv = torch.autograd.grad(Sz, z, grad_outputs=v, retain_graph=True)[0]
            nrm = float(Jtv.flatten().norm().item())
            if not math.isfinite(nrm) or nrm > 1e6:
                return None
            ratios.append(nrm)
            v = (Jtv / (nrm + 1e-12)).detach()
        return float(max(ratios))
    except Exception:
        return None


# ---- 谱归一化真复卷积网络（论文 W_theta, Lip<=1） -----------------------------
class ComplexSpectralNorm2d(nn.Module):
    """真复卷积：块矩阵 [Wr -Wi; Wi Wr]，谱归一化到 Lip<=1。"""

    def __init__(self, cin, cout, k=3, n_power=3):
        super().__init__()
        self.pad, self.n_power = k // 2, n_power
        self.wr = nn.Parameter(torch.empty(cout, cin, k, k))
        self.wi = nn.Parameter(torch.empty(cout, cin, k, k))
        self.b = nn.Parameter(torch.zeros(2 * cout))
        nn.init.kaiming_uniform_(self.wr, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.wi, a=math.sqrt(5))
        self.register_buffer("u", None)
        self.register_buffer("v", None)

    def _normed(self):
        w = torch.cat([torch.cat([self.wr, -self.wi], 1),
                       torch.cat([self.wi, self.wr], 1)], 0)
        wm = w.reshape(w.shape[0], -1)
        if self.u is None or self.u.numel() != wm.shape[0]:
            self.u = F.normalize(torch.randn(wm.shape[0], device=wm.device), dim=0)
            self.v = F.normalize(torch.randn(wm.shape[1], device=wm.device), dim=0)
        u, v = self.u, self.v
        for _ in range(self.n_power):
            v = F.normalize(wm.t() @ u, dim=0, eps=1e-12)
            u = F.normalize(wm @ v, dim=0, eps=1e-12)
        sig = (u @ (wm @ v)).abs().clamp(min=1e-8)
        self.u = u.detach()
        self.v = v.detach()
        return w / sig

    def forward(self, x):
        return F.conv2d(x, self._normed(), self.b, padding=self.pad)


class ComplexModReLU(nn.Module):
    """相位保持的模激活。"""

    def __init__(self, c, eps=1e-8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        r, im = x.chunk(2, 1)
        mag = (r.square() + im.square() + self.eps).sqrt()
        s = F.relu(mag + self.bias.view(1, -1, 1, 1)) / mag
        return torch.cat([s * r, s * im], 1)


class SNRegNet(nn.Module):
    """小规模谱归一化复 CNN：ComplexSN2d -> ModReLU x2 -> ComplexSN2d。"""

    def __init__(self, mid=16, n_layers=3, k=3, scale=1.0):
        super().__init__()
        layers, ch = [], 1
        for _ in range(max(0, n_layers - 1)):
            layers.append(ComplexSpectralNorm2d(ch, mid, k))
            layers.append(ComplexModReLU(mid))
            ch = mid
        layers.append(ComplexSpectralNorm2d(ch, 1, k))
        self.net = nn.Sequential(*layers)
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        return self.scale * self.net(x)


# ---- 结果分类 ----------------------------------------------------------------
def classify(status, rel_end, dc, psnr, zf_psnr):
    """good: 收敛且 DC<0.2 且 PSNR>=ZF-1dB；diverged: NaN/爆炸；其余 conv_bad/slow。"""
    if status == "nan" or (not math.isfinite(psnr)) or (not math.isfinite(dc)) \
            or psnr < PSNR_FLOOR or dc > 1e4:
        return "diverged"
    if status == "conv" or (math.isfinite(rel_end) and rel_end < 1e-3):
        if dc < DC_GOOD and psnr >= zf_psnr - 1.0:
            return "good"
        return "conv_bad"
    return "slow"


# ---- 出图（可选，失败不影响主流程） -------------------------------------------
def save_figures(rows, keep_z, cache_r4):
    try:
        picks = [("brg_gauge", "plain"), ("brg_gauge", "anderson5"),
                 ("euc_gauge", "plain"), ("euc_gauge", "anderson5")]
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for method, scheme in picks:
            r = next((x for x in rows if x["rate"] == 4 and x["setting"] == "strong"
                      and x["method"] == method and x["scheme"] == scheme), None)
            if r is None or not r["rels"]:
                continue
            rels = np.clip(np.asarray(r["rels"], dtype=np.float64), 1e-12, None)
            ax.plot(np.arange(1, len(rels) + 1), rels, label=f"{method}/{scheme}")
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("relative change (max over 8 slices)")
        ax.set_title("R4 strong: fixed-point convergence")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig1_conv_R4_strong.png"), dpi=130)
        plt.close(fig)
        line("FIG", "saved fig1_conv_R4_strong.png")
    except Exception as e:
        line("FIG", f"fig1 skipped: {e}")

    try:
        z0a = cache_r4["z0a"]
        gt0 = cache_r4["gta"][0].abs().detach().cpu()

        def mag2(z2):
            return torch.sqrt(z2[0].square() + z2[1].square())

        titles = ["GT", "ZF", "brg_gauge", "euc_gauge", "euc_nog"]
        imgs = [gt0, mag2(z0a[0].detach().cpu())]
        for method in ["brg_gauge", "euc_gauge", "euc_nog"]:
            z = keep_z.get((4, "gentle", method, "anderson5"))
            imgs.append(mag2(z[0].detach().cpu()) if z is not None else torch.zeros_like(gt0))
        fig, axes = plt.subplots(2, len(imgs), figsize=(3.2 * len(imgs), 6.4))
        vmax = float(gt0.max())
        for j, (t, im) in enumerate(zip(titles, imgs)):
            axes[0, j].imshow(im.numpy(), cmap="gray", vmin=0, vmax=vmax)
            axes[0, j].set_title(t, fontsize=9)
            axes[0, j].axis("off")
            err = (im - gt0).abs()
            axes[1, j].imshow(err.numpy(), cmap="hot")
            axes[1, j].set_title(f"|err| max={float(err.max()):.3f}", fontsize=8)
            axes[1, j].axis("off")
        fig.suptitle("R4 gentle / anderson5, slice 0", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig2_recon_R4_gentle.png"), dpi=130)
        plt.close(fig)
        line("FIG", "saved fig2_recon_R4_gentle.png")
    except Exception as e:
        line("FIG", f"fig2 skipped: {e}")


# ---- 主流程 ------------------------------------------------------------------
def main():
    t_all = time.time()
    base.set_seed(SEED)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    line("MAIN", f"step4a 机制实验 seed={SEED} device={device} torch={torch.__version__} "
                 f"max_iters={MAX_ITERS} tol={TOL}")

    data = torch.load(PREPARED, map_location="cpu", weights_only=False)
    gt = data["gt_complex"]
    split = data["split"]
    test_idx = [int(i) for i in split["test"][:N_SLICES]]
    line("DATA", f"gt={tuple(gt.shape)} 测试切片(前{N_SLICES}个)={test_idx}")

    reg = SNRegNet(mid=16, n_layers=3, scale=1.0).to(device).eval()
    n_reg = sum(p.numel() for p in reg.parameters())
    line("MODEL", f"SNRegNet(真复卷积+谱归一化) params={n_reg} 随机初始化(未训练, 机制实验)")

    report["config"] = {
        "seed": SEED, "n_slices": N_SLICES, "test_idx": test_idx,
        "max_iters": MAX_ITERS, "tol": TOL, "anderson_m": ANDERSON_M,
        "anderson_beta": ANDERSON_BETA, "kkt_lam": KKT_LAM,
        "settings": SETTINGS, "methods": METHODS, "schemes": SCHEMES,
        "p_default": P_DEFAULT, "p_sweep": P_SWEEP, "rate_masks": RATE_MASKS,
        "reg": {"type": "SNRegNet", "mid": 16, "n_layers": 3, "scale": 1.0,
                "params": n_reg, "trained": False},
        "dc_good": DC_GOOD, "psnr_floor": PSNR_FLOOR,
    }

    rows = []
    keep_z = {}
    zf_by_rate = {}
    cache_r4 = {}

    for rate, mkey in RATE_MASKS:
        mask = data["masks"][mkey].to(device)
        gt8 = gt[test_idx].to(device)
        y = base.sense(gt8, mask)
        z0 = base.to_2ch(base.adjoint(y))
        z0a, ya, gta = phase_align(z0, y, gt8)
        with torch.no_grad():
            zf_psnr = float(base.per_slice_psnr(base.to_c(z0a), gta).mean().item())
        zf_by_rate[rate] = zf_psnr
        if rate == 4:
            cache_r4 = {"mask": mask, "y": y, "ya": ya, "z0a": z0a, "gta": gta}
        line("MAIN", f"=== R={rate} 零填充(相位对齐) PSNR={zf_psnr:.2f} dB ===")

        for setting in SETTINGS:
            alpha, eta = setting["alpha"], setting["eta"]
            for method in METHODS:
                for scheme in SCHEMES:
                    t0 = time.time()
                    z, status, it_fin, rels = solve_fixed_point(
                        z0a, ya, mask, reg, alpha, eta, method, P_DEFAULT, scheme)
                    with torch.no_grad():
                        dc = float(base.per_slice_dc_residual(base.to_c(z), ya, mask).max().item())
                        psnr = float(base.per_slice_psnr(base.to_c(z), gta).mean().item())
                    cls = classify(status, rels[-1] if rels else float("nan"), dc, psnr, zf_psnr)
                    row = {
                        "rate": rate, "setting": setting["name"], "alpha": alpha, "eta": eta,
                        "method": method, "scheme": scheme, "status": status,
                        "iters": it_fin, "rel_end": rels[-1] if rels else None,
                        "dc": dc, "psnr": psnr, "class": cls, "rels": rels,
                    }
                    rows.append(row)
                    key = (rate, setting["name"], method, scheme)
                    if key in KEEP_Z_KEYS:
                        keep_z[key] = z
                    line("SOLVE", f"R{rate}/{setting['name']}/{method:9s}/{scheme:8s}: "
                                  f"{status:7s} it={it_fin:3d} rel={row['rel_end']:.1e} "
                                  f"dc={dc:.4f} psnr={psnr:6.2f} [{time.time()-t0:.1f}s] -> {cls}")

        if rate == 8:
            line("MAIN", "=== p 幂核扫描 R8/gentle/brg_gauge/plain ===")
            for p in P_SWEEP:
                t0 = time.time()
                z, status, it_fin, rels = solve_fixed_point(
                    z0a, ya, mask, reg, 0.3, 0.3, "brg_gauge", p, "plain")
                with torch.no_grad():
                    dc = float(base.per_slice_dc_residual(base.to_c(z), ya, mask).max().item())
                    psnr = float(base.per_slice_psnr(base.to_c(z), gta).mean().item())
                cls = classify(status, rels[-1] if rels else float("nan"), dc, psnr, zf_psnr)
                report["p_sweep"].append({
                    "p": p, "status": status, "iters": it_fin,
                    "rel_end": rels[-1] if rels else None,
                    "dc": dc, "psnr": psnr, "class": cls})
                line("PSWEEP", f"p={p}: {status:7s} it={it_fin:3d} rel={rels[-1]:.1e} "
                               f"dc={dc:.4f} psnr={psnr:6.2f} [{time.time()-t0:.1f}s] -> {cls}")

    # 谱半径估计 (R4/strong/plain, 第1切片)
    line("MAIN", "=== 谱半径估计 rho(J_S) R4/strong/plain (power iteration, 第1切片) ===")
    for method in ["brg_gauge", "euc_gauge", "euc_nog"]:
        z = keep_z.get((4, "strong", method, "plain"))
        if z is None:
            report["rho"].append({"method": method, "rho": None, "note": "未保留不动点"})
            line("RHO", f"{method}: n/a (未保留不动点)")
            continue
        if not torch.isfinite(z).all():
            report["rho"].append({"method": method, "rho": None, "note": "发散(无不动点)"})
            line("RHO", f"{method}: 发散(无不动点)")
            continue
        rho = estimate_rho(z, cache_r4["ya"], cache_r4["mask"], reg,
                           1.0, 1.0, method, P_DEFAULT)
        report["rho"].append({"method": method, "rho": rho,
                              "note": "ok" if rho is not None else "failed"})
        line("RHO", f"{method}: rho(J_S)~={rho if rho is not None else 'n/a'}")

    # ---- verdict -------------------------------------------------------------
    brg = [r for r in rows if r["method"] == "brg_gauge"]
    euc = [r for r in rows if r["method"].startswith("euc")]
    brg_good = [r for r in brg if r["class"] == "good"]
    brg_div = [r for r in brg if r["class"] == "diverged"]
    euc_div_strong = [r for r in euc if r["class"] == "diverged" and r["setting"] == "strong"]
    euc_div_all = [r for r in euc if r["class"] == "diverged"]
    brg_mean_dc = float(np.mean([r["dc"] for r in brg if math.isfinite(r["dc"])]))
    euc_mean_dc = float(np.mean([r["dc"] for r in euc if math.isfinite(r["dc"])]))

    m1 = (len(brg_good) / max(1, len(brg))) >= 0.9 and brg_mean_dc <= 0.05
    m2 = len(euc_div_strong) >= 1

    and_saved, and_cmp = 0, 0
    for rate in (4, 8):
        for setting in ("gentle", "strong"):
            plain_ = [r for r in brg if r["rate"] == rate and r["setting"] == setting
                      and r["scheme"] == "plain"]
            and_ = [r for r in brg if r["rate"] == rate and r["setting"] == setting
                    and r["scheme"] == "anderson5"]
            if plain_ and and_:
                and_cmp += 1
                if and_[0]["iters"] <= plain_[0]["iters"]:
                    and_saved += 1

    best_brg_by_rate = {rate: max([r["psnr"] for r in brg if r["rate"] == rate
                                   and math.isfinite(r["psnr"])] or [float("-inf")])
                        for rate in (4, 8)}
    qbar = all(best_brg_by_rate[r] >= zf_by_rate[r] - 0.3 for r in zf_by_rate)
    overall = "PASS" if (m1 and m2) else "REVIEW"

    report["verdict"] = {
        "m1_bregman_conv": bool(m1), "m2_euclidean_unstable": bool(m2),
        "quality_bar_untrained": bool(qbar),
        "brg_good_rate": len(brg_good) / max(1, len(brg)),
        "brg_mean_dc": brg_mean_dc, "brg_diverged": len(brg_div),
        "euc_mean_dc": euc_mean_dc, "euc_diverged_strong": len(euc_div_strong),
        "euc_diverged_all": len(euc_div_all),
        "anderson_saved_ratio": and_saved / max(1, and_cmp),
        "best_brg_psnr_by_rate": best_brg_by_rate,
        "overall": overall,
    }

    # 先写 JSON（防御：summary 出错也不丢机器可读结果）
    report["rows"] = rows
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    line("MAIN", f"report written: {REPORT_PATH}")

    # 中文 summary
    L = []
    L.append("=" * 80)
    L.append("Step 4a 机制实验汇总 (fastMRI 128, 前8个测试切片, seed=42)")
    L.append("=" * 80)
    L.append("验证内容: 论文实验2 的核心主张 -- Bregman 结构不动点算子收敛, Euclidean 型算子不稳定")
    L.append(f"算子   : S(z) = Gauge( z - eta*( A*(Az-y) + alpha*reg_term ) ), max_iters={MAX_ITERS}, tol={TOL}")
    L.append("  brg_gauge: reg_term = grad h(W(z)), h(z)=sum|z|^p/p, p=4 (可分离幂核, 论文选择)")
    L.append("  euc_gauge: reg_term = W(z) (无 Bregman 结构) + gauge ;  euc_nog: 无 gauge")
    L.append("W      : SNRegNet (真复卷积 + 谱归一化, Lip<=1), 随机初始化未训练 (机制验证)")
    L.append("求解   : plain / Anderson(m=5, KKT 逐样本) ; 强度 gentle(a=0.3,e=0.3)/strong(a=1,e=1)")
    L.append("指标   : dc = ||M(Fx-y)||/||y|| (数据一致性残差, <0.2 合格) ; psnr 相对相位对齐零填充")
    L.append("")
    for rate in (4, 8):
        L.append(f"--- R={rate}  ZF(对齐)={zf_by_rate[rate]:.2f} dB ---")
        hdr = f"{'setting':8s} {'method':10s} {'scheme':9s} {'status':8s} {'iters':>5s} {'rel_end':>9s} {'dc':>9s} {'psnr':>7s}  class"
        L.append(hdr)
        L.append("-" * len(hdr))
        for r in rows:
            if r["rate"] != rate:
                continue
            rel = "n/a" if r["rel_end"] is None else f"{r['rel_end']:.1e}"
            L.append(f"{r['setting']:8s} {r['method']:10s} {r['scheme']:9s} {r['status']:8s} "
                     f"{r['iters']:5d} {rel:>9s} {r['dc']:9.4f} {r['psnr']:7.2f}  {r['class']}")
        L.append("")
    L.append("--- p 幂核扫描 (R8/gentle/brg_gauge/plain, p=2 即退化为 Euclidean) ---")
    for pr in report["p_sweep"]:
        rel = "n/a" if pr["rel_end"] is None else f"{pr['rel_end']:.1e}"
        L.append(f"  p={pr['p']}: {pr['status']:8s} it={pr['iters']:3d} rel_end={rel} "
                 f"dc={pr['dc']:.4f} psnr={pr['psnr']:6.2f} class={pr['class']}")
    L.append("")
    L.append("--- 谱半径估计 rho(J_S) (R4/strong/plain, power iteration, rho<1 收缩) ---")
    for rr in report["rho"]:
        L.append(f"  {rr['method']:10s}: rho={rr['rho'] if rr['rho'] is not None else 'n/a'}  ({rr['note']})")
    L.append("")
    L.append("--- 结论 ---")
    L.append(f"M1 Bregman 收敛性: brg_gauge good 比例 = {len(brg_good)}/{len(brg)} "
             f"({len(brg_good)/max(1,len(brg)):.0%}), 平均 DC={brg_mean_dc:.4f} -> {'PASS' if m1 else 'FAIL'}")
    L.append(f"M2 Euclidean 不稳定性: strong 设置下发散/NaN 组数 = {len(euc_div_strong)} "
             f"(euc 总发散组 {len(euc_div_all)}) -> {'PASS' if m2 else 'FAIL'}")
    L.append(f"I1 Anderson 加速: brg_gauge 中 anderson<=plain 迭代数的比例 = {and_saved}/{and_cmp} (信息性)")
    L.append(f"I2 质量基准(未训练): 各 R 最佳 brg PSNR = "
             f"{ {r: round(best_brg_by_rate[r], 2) for r in best_brg_by_rate} } ; "
             f"零填充 = { {r: round(zf_by_rate[r], 2) for r in zf_by_rate} } "
             f"(质量提升需 step4b 隐式训练)")
    L.append(f"总体: {overall}  (M1&M2 同时通过 => 机制成立, 可进入 step4b)")
    L.append("")
    summary_text = "\n".join(L)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    line("MAIN", f"summary written: {SUMMARY_PATH}")

    save_figures(rows, keep_z, cache_r4)

    line("MAIN", f"done in {time.time()-t_all:.1f}s | verdict: {overall} "
                 f"(M1={m1}, M2={m2}) errors={len(report['issues'])}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
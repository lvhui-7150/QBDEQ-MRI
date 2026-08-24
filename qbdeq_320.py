# -*- coding: utf-8 -*-
"""
qbdeq_320.py -- QB-DEQ 训练引擎 (fastMRI 320x320)
==================================================

把 step4a 的机制算子与 step4b2 的商流形隐式反传移植到 320x320 数据上，
作为 step6（端到端隐式训练）的共享引擎。所有函数与 step4/step4b2 同构：

  算子     S(z) = Gauge( z - eta*( A*(A z - y) + alpha*grad h(W_theta(z)) ) )
            h(z) = sum_i |z_i|^p / p   (逐像素可分离幂核)
  W_theta  = SNRegNet：真复卷积 + 谱归一化 (Lip(W)<=1)
  求解     Anderson(m=5, KKT 逐样本) / plain, 可选 tol/max_iters
  反传     商流形隐式梯度:  (I - P J^T P) q = P nabla_z L,  GMRES 批量求解
            dL/dtheta = q^T dS/dtheta (autograd.grad, retain_graph)

与 step4b2 的区别：GMRES 用纯 PyTorch 批量实现（float32, GPU 原生），
可放进训练循环；梯度精度由 --bwd-rtol / --bwd-max-mv 控制。

本文件只提供函数，不执行实验（由 _probe_qbdeq_bwd.py / step6_train_qbdeq.py 调用）。
"""

import math
import time

import numpy as np
import torch

import step4_implicit_deq as s4          # SNRegNet / grad_kernel / anderson_batched
import step5_320_ceiling as C            # fft / to_2ch / to_c / loss / ssim

# ---- 幂核 + gauge（与 step4a 完全一致，仅数据尺寸不同） -------------------------
def norm_val(z2):
    """逐像素模长 (B,H,W)。"""
    return (z2[:, 0].square() + z2[:, 1].square() + 1e-8).sqrt()


def grad_kernel(z2, p):
    """grad h(z)_i = |z_i|^(p-2) z_i, 逐像素可分离。p=2 退化为恒等。"""
    return norm_val(z2).pow(p - 2.0).unsqueeze(1) * z2


def gauge_fix_batch(z2):
    """(B,2,H,W) -> (B,2,H,W)，每样本独立做全局相位旋转（最大模像素相位->0）。"""
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


# ---- 320 数据一致性步（与 step5 的 FFT 约定一致） ------------------------------
def dc_grad(z2, y, mask):
    """A*(A z - y)，2ch 实数表示。mask: (H,W) bool，广播到 (B,2,H,W)。"""
    zc = C.to_c(z2)
    return C.to_2ch(C.ifft2_t((C.fft2_t(zc) - y) * mask))


def S_op(z2, y, mask, reg, alpha, eta, p):
    """一步不动点算子 S(z) = Gauge( z - eta*( A*(Az-y) + alpha*grad h(W(z)) ) )。"""
    r = reg(z2)
    reg_term = alpha * grad_kernel(r, p)
    out = z2 - eta * (dc_grad(z2, y, mask) + reg_term)
    return gauge_fix_batch(out)


def make_inputs(x_gt, mask, device):
    """gt 复数 (B,H,W) -> (y, z0)。y=FFT(x)*mask, z0=零填充图像(2ch)。"""
    y = C.fft2_t(x_gt) * mask
    z0 = C.to_2ch(C.ifft2_t(y))
    return y, z0


# ---- 重设计算子：残差正则器 W(z)=z+Net(z)（商流形原则内） --------------------
# 原算子 S(z)=Gauge(z-eta(A*(Az-y)+alpha*grad h(W(z)))) 中 W 的输出很小，
# 被幂核二次压制，固定点几乎不动（实测 ~24dB≈ZF）。改为残差捷径：
#   W(z) = z + Net(z)   (Net 谱归一化, Lip<=1)
#   p=2 时 grad h(W) = W，正则项 alpha*(z+Net(z)) 以强线性方式进入固定点方程：
#       (A*A + alpha*I) z = A*y - alpha*Net(z)
#   即"正则化最小二乘 + 学习到的补全"，是 VarNet 类结构的隐式（无限深度）版本，
#   训练梯度不再被 |W|^2 衰减。gauge/商流形隐式反传机制完全保留。

def S_op_res(z2, y, mask, net, alpha, eta, p):
    """一步不动点算子：S(z)=Gauge( z - eta*( A*(Az-y) + alpha*grad h(z+Net(z)) ) )。"""
    u = z2 + net(z2)                     # 残差正则器输出 W(z)
    reg_term = alpha * grad_kernel(u, p)  # p=2 -> alpha*u（线性，无阻尼）
    out = z2 - eta * (dc_grad(z2, y, mask) + reg_term)
    return gauge_fix_batch(out)


_S = S_op          # 当前算子（step6 通过 set_operator 切换）


def set_operator(name):
    """选择不动点算子：'vi'=原论文算子, 'vi_res'=残差正则器重设计。"""
    global _S
    if name == "vi_res":
        _S = S_op_res
    else:
        _S = S_op
    return _S


# ---- 不动点求解（Anderson m=5，逐样本 KKT；同 step4a） -------------------------
def solve_fixed_point(z0, y, mask, reg, alpha, eta, p, scheme="anderson5",
                      max_iters=200, tol=1e-6, anderson_m=5):
    """返回 (z, status, it_fin, rels)。全 no_grad，供训练前向/评测使用。"""
    B = z0.shape[0]
    device = z0.device
    z = z0.clone()
    x_hist, s_hist = [], []
    rels = []
    status, it_fin = "maxiter", max_iters
    for it in range(1, max_iters + 1):
        z_prev = z
        with torch.no_grad():
            s = _S(z_prev, y, mask, reg, alpha, eta, p)
        if scheme == "plain":
            z_new = s
        else:
            x_hist.append(z_prev.detach())
            s_hist.append(s.detach())
            if len(x_hist) > anderson_m:
                x_hist.pop(0)
                s_hist.pop(0)
            if len(x_hist) >= 2:
                z_new = s4.anderson_batched(x_hist, s_hist, B, device)
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


# ---- 商流形反传：投影算子 ----------------------------------------------
def fiber_dir(z):
    """gauge fiber 方向 iz = (-Im z, Re z)，返回 (B, N)。"""
    return torch.cat([-z[:, 1:2], z[:, 0:1]], 1).reshape(z.shape[0], -1)


def proj_flat(vf, iz):
    """P_sigma(v) = v - <v, iz>/||iz||^2 * iz，逐样本，输入 (B,N)。"""
    num = (vf * iz).sum(dim=1, keepdim=True)
    den = (iz * iz).sum(dim=1, keepdim=True) + 1e-12
    return vf - num / den * iz


# ---- 批量 GMRES（纯 PyTorch, 无 restart, 逐样本正交化） ------------------------
def batched_gmres(matvec, b, x0, m=60, rtol=1e-3, atol=0.0, reorth=False, verbose=False):
    """解 A x = b, A 为逐样本独立算子。返回 (x, res_max, n_mv)。

    matvec: (B,C,H,W) -> (B,C,H,W)，逐样本线性。
    b     : (B,C,H,W)；x0    : (B,C,H,W)
    残差   res = ||Ax - b|| / (||b|| + atol)，按 batch 取最大。
    """
    B = b.shape[0]
    N = b.numel() // B
    dev = b.device
    bn = b.reshape(B, N)
    x = x0.reshape(B, N).clone()
    n_mv = 0

    r = bn - matvec(x.reshape_as(b)).reshape(B, N)
    n_mv += 1
    beta0 = r.norm(dim=1)
    if float(beta0.max()) <= 1e-30:
        return x.reshape_as(b), 0.0, n_mv

    Vs = []          # Arnoldi 基 (B,N)，始终单位范数
    Hs = []          # Hessenberg 列, 每列 (B, k+2)
    for k in range(m):
        beta = r.norm(dim=1)                     # ||r_k|| = h[k+1,k]（每步更新！）
        if float(beta.max()) < 1e-30:
            break
        v = r / beta.clamp(min=1e-30).unsqueeze(1)
        Vs.append(v)
        w = matvec(v.reshape_as(b)).reshape(B, N)
        n_mv += 1
        h = torch.zeros(B, k + 2, device=dev, dtype=b.dtype)
        for j in range(k + 1):
            h[:, j] = (w * Vs[j]).sum(dim=1)
            w = w - h[:, j].unsqueeze(1) * Vs[j]
        if reorth:
            for j in range(k + 1):
                hh = (w * Vs[j]).sum(dim=1)
                h[:, j] = h[:, j] + hh
                w = w - hh.unsqueeze(1) * Vs[j]
        h[:, k + 1] = w.norm(dim=1)
        Hs.append(h)
        r = w

    k_fin = len(Hs)
    if k_fin == 0:
        return x.reshape_as(b), float("nan"), n_mv
    # 组装 H (B, k_fin+1, k_fin) 与 g = beta0*e1，逐样本最小二乘
    Hk = torch.zeros(B, k_fin + 1, k_fin, device=dev, dtype=b.dtype)
    for j in range(k_fin):
        Hk[:, : j + 2, j] = Hs[j]
    g = torch.zeros(B, k_fin + 1, 1, device=dev, dtype=b.dtype)
    g[:, 0, 0] = beta0
    try:
        sol = torch.linalg.lstsq(Hk, g)[0]                # (B, k_fin, 1)
        if not torch.isfinite(sol).all():
            return x.reshape_as(b), float("nan"), n_mv
    except Exception as _exc:
        if verbose:
            print("[GMRES] lstsq exception:", repr(_exc))
        return x.reshape_as(b), float("nan"), n_mv
    y = sol.squeeze(2)                                     # (B, k_fin)
    for j in range(k_fin):
        x = x + y[:, j:j + 1] * Vs[j]
    # 理论残差 ||g - H y||（无需额外 matvec）
    Hym = (Hk @ y.unsqueeze(2)).squeeze(2)                 # (B, k_fin+1)
    res = (Hym - g.squeeze(2)).norm(dim=1) / (beta0 + atol)
    res_max = float(res.max().item())
    if verbose:
        print("[GMRES] k_fin=%d n_mv=%d res_max=%.3e" % (k_fin, n_mv, res_max))
    return x.reshape_as(b), res_max, n_mv


# ---- 商流形隐式反传 ----------------------------------------------
def quotient_backward(z_fp, y, mask, reg, alpha, eta, p, loss_fn, budget=60,
                      rtol=1e-3, reorth=True, verbose=False):
    """给定不动点 z_fp 与损失函数 loss_fn(z)->scalar，返回
    (q, grads, res, n_mv, loss_val, rhs_norm)。

    求解 (I - P J^T P) q = P nabla_z L，然后 dL/dtheta = q^T dS/dtheta。
    grads 与 reg.parameters() 对齐；不存在的参数给 None。

    内存策略：每个 matvec 重新构建一次 S_op 计算图并在调用后释放
    （step4b2 的 jt_vjp 同款），避免 retain_graph 长期挂图导致的显存泄漏。
    """
    B = z_fp.shape[0]
    z = z_fp.detach().requires_grad_(True)
    loss = loss_fn(z)
    loss_val = float(loss.item())
    rhs = torch.autograd.grad(loss, z)[0].detach()
    iz = fiber_dir(z).detach()          # 投影算子固定在切空间（step4b2 同款）
    z = z.detach().requires_grad_(True)

    def proj(vf):
        return proj_flat(vf.reshape(B, -1).detach(), iz).reshape_as(vf)

    def matvec(v):
        pv = proj(v)
        with torch.enable_grad():
            Sz = _S(z, y, mask, reg, alpha, eta, p)
            Jtpv = torch.autograd.grad(Sz, z, grad_outputs=pv,
                                       retain_graph=False, create_graph=False)[0]
        return proj(pv - Jtpv)

    b_hat = proj(rhs)
    q, res, n_mv = batched_gmres(matvec, b_hat, torch.zeros_like(b_hat),
                                 m=budget, rtol=rtol, reorth=reorth,
                                 verbose=verbose)
    params = list(reg.parameters())
    with torch.enable_grad():
        Sz = _S(z, y, mask, reg, alpha, eta, p)
        grads = torch.autograd.grad(Sz, params, grad_outputs=q,
                                    retain_graph=False, allow_unused=True)
    return q, grads, res, n_mv, loss_val, float(rhs.norm().item())


# ---- 质量指标（幅值 PSNR/SSIM, fastMRI 口径） -------------------------------
def psnr_torch(z2, x_gt):
    """逐样本幅值 PSNR (dB)。"""
    return C.per_slice_psnr_full(C.to_c(z2), x_gt)


def ssim_torch(z2, x_gt):
    """逐样本 SSIM（复用 C.torch_ssim，逐样本调用）。"""
    out = []
    for i in range(z2.shape[0]):
        out.append(C.torch_ssim(C.to_c(z2[i:i + 1]), x_gt[i:i + 1]).item())
    return torch.tensor(out, dtype=torch.float32)


def eval_slices(store, idx, mask_key, mask_store, reg, alpha, eta, p,
                max_iters=200, tol=1e-6, batch=2, device="cuda"):
    """在给定切片列表上做隐式前向求解并返回逐样本指标。"""
    mask = mask_store.get(mask_key, device=device)
    ps, ss, iters, statuses = [], [], [], []
    for s0 in range(0, len(idx), batch):
        blk = [int(i) for i in idx[s0:s0 + batch]]
        x = store.get_batch(blk, device=device).to(torch.complex64)
        y, z0 = make_inputs(x, mask, device)
        z, status, it_fin, _rels = solve_fixed_point(
            z0, y, mask, reg, alpha, eta, p, "anderson5",
            max_iters=max_iters, tol=tol)
        ps.append(psnr_torch(z, x).detach().cpu())
        ss.append(ssim_torch(z, x).detach().cpu())
        iters.append(it_fin)
        statuses += [status] * len(blk)
    return (torch.cat(ps).numpy(), torch.cat(ss).numpy(),
            np.asarray(iters), statuses)


# ---- 谱半径估计（power iteration, 与 step4a 相同，320 版） -------------------
def estimate_rho(z_fp, y, mask, reg, alpha, eta, p, power_iters=6):
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
            Sz = _S(z, y[:1], mask, reg, alpha, eta, p)
            Jtv = torch.autograd.grad(Sz, z, grad_outputs=v, retain_graph=True)[0]
            nrm = float(Jtv.flatten().norm().item())
            if not math.isfinite(nrm) or nrm > 1e6:
                return None
            ratios.append(nrm)
            v = (Jtv / (nrm + 1e-12)).detach()
        return float(max(ratios))
    except Exception:
        return None

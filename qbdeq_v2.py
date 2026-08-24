# -*- coding: utf-8 -*-
"""
qbdeq_v2.py -- QB-DEQ v2 引擎：强去噪算子 + 商流形隐式训练 (fastMRI 320x320)
================================================================================
设计动机（针对 step6 的负结果）：
  step6 用 SNRegNet(38K 参数) 作为正则器，算子 S(z)=Gauge(z-eta*(A*(Az-y)+
  alpha*grad h(W(z))))，固定点被"吸"在零填充附近（|W(z*)| 很小，耦合弱）。

v2 改为 MoDL/DEQ-MRI 型算子（权重共享的"数据一致性 + 强去噪器"）：
      S(z) = Gauge(  UNet( z - eta * A*(A z - y) )  )
  - UNet：GroupNorm U-Net（base=96/128，~11-19M 参数），输出为重建图像（非小残差）
  - eta：可学习 softplus 步长（共享）
  - Gauge：每步最大模像素相位归零（商流形；相移自由度被折叠）
  - 前向：Anderson(m=5) 不动点求解（可加阻尼）
  - 反传：商流形隐式梯度 (I - P J^T P) q = P nabla L，批量 GMRES
            dL/dtheta = q^T dS/dtheta

训练（step7_train_qbdeq_v2.py）：
  A0  一步去噪预训练（z1 = S(z0)，监督到 GT）—— 给 UNet 一个强初始化
  A1  权重共享 unrolled 训练（同一 S 展开 K 步，逐步加深 + 深度监督）
  B   真正的隐式训练（前向解不动点 + 商流形 GMRES 反传）
评测：不动点（Anderson 紧容差）在测试集上的幅值 PSNR/SSIM（与 fastmri_320_prep 口径一致）。
"""
import math

import numpy as np
import torch

import step4_implicit_deq as s4          # anderson_batched
import step5_320_ceiling as C            # fft / to_2ch / to_c / loss / ssim / store
import qbdeq_320 as Q                    # gauge_fix_batch / batched_gmres / fiber_dir / proj_flat


def apply_sn(module, n_power_iterations=1, dec_scale=0.70710678, dec=False):
    """对模块内所有 Conv2d 应用谱归一化（Lip<=1 per conv）。
    dec=True（解码路径，含 concat 的 GNBlock）时卷积输入先乘以 dec_scale，
    把 concat 引入的 sqrt(2) 膨胀抵消，使整网 Lip 大致 <=1。"""
    import torch.nn as nn

    def scaled(conv, scale):
        conv = nn.utils.spectral_norm(conv, n_power_iterations=n_power_iterations)
        if abs(scale - 1.0) > 1e-6:
            conv.register_forward_pre_hook(
                lambda m, x: x * scale if isinstance(x, torch.Tensor)
                else tuple(xi * scale for xi in x))
        return conv

    dec_names = {"dec3", "dec2", "dec1"}
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            scale = dec_scale if dec else 1.0
            setattr(module, name, scaled(child, scale))
        else:
            apply_sn(child, n_power_iterations, dec_scale,
                     dec=(dec or name in dec_names))


def build_model(base=96, groups=8, eta_init=0.3, alpha_init=0.5, net=None,
                sn=False, sn_scale=0.70710678):
    """返回 (net, eta_logit, alpha_logit)。net 为 2ch UNet 去噪器；
    eta/alpha 为可学习 softplus 标量（RED 算子步长与先验权重）。
    sn=True 时对 UNet 卷积做谱归一化（解码路径乘 sn_scale 抵消 concat 膨胀）
    并固定输出缩放=1，使 D 非扩张，从而 RED 算子 J_S 满足 rho<1。"""
    if net is None:
        from step5_train_final import UNetGN
        net = UNetGN(in_ch=2, base=base, groups=groups)
    if sn:
        apply_sn(net, dec_scale=sn_scale)
        with torch.no_grad():
            net.out_scale.fill_(1.0)
        net.out_scale.requires_grad_(False)
    eta_logit = torch.nn.Parameter(torch.tensor([float(eta_init)], dtype=torch.float32))
    alpha_logit = torch.nn.Parameter(torch.tensor([float(alpha_init)], dtype=torch.float32))
    return net, eta_logit, alpha_logit


def eta_of(eta_logit):
    return torch.nn.functional.softplus(eta_logit)


def alpha_of(alpha_logit):
    return torch.nn.functional.softplus(alpha_logit)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# 算子：S(z) = Gauge( UNet( z - eta * A*(A z - y) ) )
# ----------------------------------------------------------------------------
def S_op(z2, y, mask, net, eta_logit, alpha_logit=None):
    """一步不动点算子（RED/梯度型）：
       S(z) = Gauge( z - eta*( A*(Az-y) + alpha*(z - D(z)) ) )
    z2: (B,2,H,W) float；y: (B,H,W) complex；mask: (H,W) bool。"""
    eta = eta_of(eta_logit)
    alpha = alpha_of(alpha_logit) if alpha_logit is not None else torch.tensor(1.0,
                                                                               device=z2.device)
    zc = C.to_c(z2)
    dc_res = C.ifft2_t((C.fft2_t(zc) - y) * mask)          # A*(Az-y), complex
    dz = net(z2)                                            # D(z) 去噪/重建
    out = z2 - eta * (C.to_2ch(dc_res) + alpha * (z2 - dz))
    return Q.gauge_fix_batch(out)


# ----------------------------------------------------------------------------
# Anderson 不动点求解（可加阻尼；damp<1 提升鲁棒性）
# ----------------------------------------------------------------------------
def solve_fixed_point(z0, y, mask, net, eta_logit, alpha_logit=None,
                      scheme="anderson5", max_iters=200, tol=1e-6,
                      anderson_m=5, damp=1.0, verbose=False):
    """返回 (z, status, it_fin, rels)。全 no_grad。"""
    B = z0.shape[0]
    device = z0.device
    z = z0.clone()
    x_hist, s_hist = [], []
    rels = []
    status, it_fin = "maxiter", max_iters
    for it in range(1, max_iters + 1):
        z_prev = z
        with torch.no_grad():
            s = S_op(z_prev, y, mask, net, eta_logit, alpha_logit)
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
        if damp < 1.0:
            z_new = damp * z_new + (1.0 - damp) * s
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
    if verbose:
        print("[FP] status=%s it=%d rel=%.2e" % (status, it_fin, rels[-1] if rels else -1))
    return z, status, it_fin, rels


# ----------------------------------------------------------------------------
# 商流形隐式反传（算子为 S_op 去噪形式）
# ----------------------------------------------------------------------------
def quotient_backward(z_fp, y, mask, net, eta_logit, alpha_logit, loss_fn,
                      budget=60, rtol=1e-3, reorth=True, verbose=False):
    """解 (I - P J^T P) q = P nabla L，然后 dL/dtheta = q^T dS/dtheta。

    返回 (q, grads, res, n_mv, loss_val, rhs_norm)。
    grads 与 [*net.parameters(), eta_logit, alpha_logit] 对齐。
    """
    B = z_fp.shape[0]
    z = z_fp.detach().requires_grad_(True)
    loss = loss_fn(z)
    loss_val = float(loss.item())
    rhs = torch.autograd.grad(loss, z)[0].detach()
    iz = Q.fiber_dir(z).detach()          # 投影固定在切空间
    z = z.detach().requires_grad_(True)

    def proj(vf):
        return Q.proj_flat(vf.reshape(B, -1).detach(), iz).reshape_as(vf)

    def matvec(v):
        pv = proj(v)
        with torch.enable_grad():
            Sz = S_op(z, y, mask, net, eta_logit, alpha_logit)
            Jtpv = torch.autograd.grad(Sz, z, grad_outputs=pv,
                                       retain_graph=False, create_graph=False)[0]
        return proj(pv - Jtpv)

    b_hat = proj(rhs)
    q, res, n_mv = Q.batched_gmres(matvec, b_hat, torch.zeros_like(b_hat),
                                   m=budget, rtol=rtol, reorth=reorth,
                                   verbose=verbose)
    params = [pp for pp in list(net.parameters()) + [eta_logit, alpha_logit]
              if pp.requires_grad]
    with torch.enable_grad():
        Sz = S_op(z, y, mask, net, eta_logit, alpha_logit)
        grads = torch.autograd.grad(Sz, params, grad_outputs=q,
                                    retain_graph=False, allow_unused=True)
    return q, grads, res, n_mv, loss_val, float(rhs.norm().item())


# ----------------------------------------------------------------------------
# 评测：测试集上的不动点幅值 PSNR / SSIM（numpy 口径，与 fastmri_320_prep 一致）
# ----------------------------------------------------------------------------
def eval_fixed_point(store, idx, mask, device, net, eta_logit, alpha_logit,
                     max_iters=200, tol=1e-5, batch=2, scheme="anderson5",
                     damp=1.0):
    ssim = C.SSIMComputer()
    ps, ss, iters, statuses = [], [], [], []
    for s0 in range(0, len(idx), batch):
        blk = [int(i) for i in idx[s0:s0 + batch]]
        x = store.get_batch(blk, device=device).to(torch.complex64)
        y, z0 = Q.make_inputs(x, mask, device)
        z, status, it_fin, _r = solve_fixed_point(z0, y, mask, net, eta_logit, alpha_logit,
                                                  scheme=scheme, max_iters=max_iters,
                                                  tol=tol, damp=damp)
        z_np = C.to_c(z).detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        for i in range(len(blk)):
            gm, zm = np.abs(x_np[i]), np.abs(z_np[i])
            ps.append(C.compute_psnr(gm, zm))
            ss.append(ssim.compute(gm, zm))
            iters.append(int(it_fin))
            statuses.append(status)
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
            "n": len(ps), "it_avg": float(np.mean(iters)),
            "conv_rate": float(np.mean([s == "conv" for s in statuses])),
            "iters": iters, "statuses": statuses}


def eval_unrolled(store, idx, mask, device, net, eta_logit, alpha_logit,
                      K, batch=2):
    """有限展开 K 步的幅值 PSNR/SSIM（对照：隐式 = unrolled 在 K->inf 的极限）。"""
    ssim = C.SSIMComputer()
    ps, ss = [], []
    for s0 in range(0, len(idx), batch):
        blk = [int(i) for i in idx[s0:s0 + batch]]
        x = store.get_batch(blk, device=device).to(torch.complex64)
        y, z0 = Q.make_inputs(x, mask, device)
        z = z0
        with torch.no_grad():
            for _ in range(K):
                z = S_op(z, y, mask, net, eta_logit, alpha_logit)
        z_np = C.to_c(z).detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        for i in range(len(blk)):
            gm, zm = np.abs(x_np[i]), np.abs(z_np[i])
            ps.append(C.compute_psnr(gm, zm))
            ss.append(ssim.compute(gm, zm))
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n": len(ps)}


# ----------------------------------------------------------------------------
# 训练后谱半径（power iteration，商流形意义；rho<1 说明不动点收缩）
# ----------------------------------------------------------------------------
def estimate_rho(z_fp, y, mask, net, eta_logit, alpha_logit, power_iters=6):
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
            Sz = S_op(z, y[:1], mask, net, eta_logit, alpha_logit)
            Jtv = torch.autograd.grad(Sz, z, grad_outputs=v, retain_graph=True)[0]
            nrm = float(Jtv.flatten().norm().item())
            if not math.isfinite(nrm) or nrm > 1e6:
                return None
            ratios.append(nrm)
            v = (Jtv / (nrm + 1e-12)).detach()
        return float(max(ratios))
    except Exception:
        return None

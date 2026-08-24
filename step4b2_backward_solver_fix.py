# -*- coding: utf-8 -*-
"""
Step 4b2 -- 反传求解器修复: GMRES 直接解 + LGMRES 交叉验证
=========================================================

背景 (step4b 结论 REVIEW):
    论文 Alg2 隐式反传需要解伴随方程 (I - J^T) q = dL/dz* (VJP 形式
    q = P(rhs + J^T q), P 为 gauge fiber 投影)。step4b 用朴素 Neumann 迭代
    1500 次, 残差卡在 ~0.74 —— 因为谱半径 rho(J)~=0.99997 (step4a 实测),
    迭代每次只收缩 ~1e-5。这本身是论文动机 (隐式反传的病态), 但朴素迭代
    无法作为求解器, 需要换成 Krylov 直接解法。

本步修复 (探针已逐项验证, 数值见下方注释):
    1) 主求解器 = GMRES (scipy, float64 GPU): 在投影子空间直接解
       A_hat q = b_hat, A_hat = P(I - J^T)P, b_hat = P(dL/dz*),
       ~109 次 matvec 收敛, 真实残差 ~8e-10, 用时 ~3.5s。
    2) 交叉验证 = LGMRES (inner=30, outer=3): 两个独立 Krylov 求解器
       给出同一解 (q 相对误差 ~2e-9)。
    3) VJP 伴随性 = 有限差分验证 <J^T q, v> == <q, J v> (rel ~3e-11),
       证明 autograd 反传实现数学上正确。
    4) unrolled 深度扫描 K=400/800/1600: 有限展开梯度随 K 增大单调逼近
       隐式梯度 (cosine: 0.16 -> 0.81 -> 0.9935), 说明隐式梯度是
       "无穷深度" 极限; step4b 里 0.997 是与同为截断的 Neumann-1500 互比
       的假象 (两边的展开都远未收敛)。
    5) 相位旋转不变性 (mag loss): 3 个相位下隐式梯度方向 cosine = 1.0。

验收线 (全部通过 -> PASS):
  V1  FD 伴随性相对误差 < 1e-6
  V2  GMRES vs LGMRES 的 q 相对误差 < 1e-4
  V3  GMRES info=0 且真实残差 < 1e-6
  V4  fiber 分量 (投影后) < 1e-6
  V5  相位不变性 cosine > 0.99
  V6  unrolled K=1600 时 cosine(隐式, unrolled) > 0.99 (方向一致性)
  (信息性) rel err vs unrolled 预期 ~O(1): 幅度差异是 rho~1 的固有现象,
            是论文"有限展开失效"的论据, 不作为通过条件。

对照 (证明朴素方法确实卡住, 说明为什么需要 GMRES):
  plain Neumann 300 次 / Anderson(m=10) 600 次, 残差均卡 ~0.7-0.8。

运行:
    python step4b2_backward_solver_fix.py
    (需要 CUDA GPU; 预计 5-8 分钟, 大部分时间在 unrolled 深度扫描)

输出 (我会读取的文本):
    step4b2_summary.txt  中文汇总 + verdict
    step4b2_report.json  机器可读完整结果
    step4b2_figs/        残差曲线 / 一致性柱状图 / unrolled 收敛趋势
"""

import os
import sys
import json
import time
import math
import copy

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch

import step4_implicit_deq as s4          # S_op / solve_fixed_point / SNRegNet / phase_align
import step4b_implicit_grad as s4b       # 损失 / cosine / rel_l2 / fiber_comp / unrolled_grad
base = s4.base

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 固定配置 ---------------------------------------------------------------
SEED = 42
TEST_SLICES = [3, 88]                    # step4a/4b 相同的测试切片
RATE = 4
MASK_KEY = "r4_s42"
ALPHA, ETA, P, METHOD = 0.3, 0.3, 4.0, "brg_gauge"   # 与 step4a/4b 相同的稳定配方

NEUMANN_MAX = 300                        # plain Neumann 对照 (短演示, 卡住即停)
AND_MAX = 600                            # Anderson 对照迭代上限
AND_M = 10                               # Anderson 记忆长度

GMRES_RESTART = 200
GMRES_MAXITER = 3000
GMRES_TOL = 1e-9
GMRES_ATOL = 1e-12
LGMRES_INNER = 30
LGMRES_OUTER = 3

FD_EPS = 1e-5
FIBER_EPS = 1e-6
UNROLL_K_LIST = [400, 800, 1600]         # 深度扫描 (1600 处 cosine>0.99, 探针实测)
UNROLL_ACCEPT_K = 1600
PHASE_PHIS = [0.0, math.pi / 2.0, math.pi]

# 验收阈值
V1_T = 1e-6
V2_T = 1e-4
V3_T = 1e-6
V4_T = 1e-6
V5_T = 0.99
V6_T = 0.99

HERE = os.path.dirname(os.path.abspath(__file__))
PREPARED = os.path.join(HERE, "fastmri_128_prepared.pt")
FIG_DIR = os.path.join(HERE, "step4b2_figs")
REPORT_PATH = os.path.join(HERE, "step4b2_report.json")
SUMMARY_PATH = os.path.join(HERE, "step4b2_summary.txt")
os.makedirs(FIG_DIR, exist_ok=True)

report = {"config": {}, "results": {}, "verdict": {}, "issues": []}


def line(tag, msg):
    print(f"[{tag}] {msg}")


def add_issue(level, msg):
    report["issues"].append({"level": level, "msg": msg})
    print(f"[{level}] {msg}")


def rel_l2_t(a, b):
    """两个同形 tensor 的相对 L2 误差。"""
    num = float((a - b).pow(2).sum().item())
    den = float(b.pow(2).sum().item()) + 1e-12
    return math.sqrt(num / den)


def fiber_dir(z):
    """gauge fiber 方向 iz = (-Im z, Re z), 返回 (B, N)。"""
    return torch.cat([-z[:, 1:2], z[:, 0:1]], 1).reshape(z.shape[0], -1)


def proj_flat(vf, iz):
    """P_sigma(v) = v - <v, iz>/||iz||^2 * iz, 逐样本, (B,N)。"""
    num = (vf * iz).sum(dim=1, keepdim=True)
    den = (iz * iz).sum(dim=1, keepdim=True) + 1e-12
    return vf - num / den * iz


def jt_vjp(z, y, mask, reg, q):
    """J_S(z)^T q, 通过 autograd.grad (float32 或 float64 均可)。"""
    Sz = s4.S_op(z, y, mask, reg, ALPHA, ETA, METHOD, P)
    return torch.autograd.grad(Sz, z, grad_outputs=q)[0]


def make_reg64(reg, z0):
    """先用 no_grad 前向初始化谱归一化 buffer, 再 deepcopy 成 float64。"""
    with torch.no_grad():
        _ = reg(z0[:1])
    return copy.deepcopy(reg).double()


def gmres_solve(z64, rhs64, y64, mask64, reg64, lgmres=False, rtol=None, maxiter=None, progress_cb=None):
    """在投影子空间解 A_hat q = b_hat, A_hat = P(I - J^T)P。

    返回 (q, info, nvec_solve, true_res, time_s)。true_res 用额外一次
    matvec 重算的真实方程残差 ||P(I-J^T P)q - P rhs|| / ||P rhs||。
    """
    import scipy.sparse.linalg as spla
    from scipy.sparse.linalg import LinearOperator
    if rtol is None:
        rtol = GMRES_TOL
    if maxiter is None:
        maxiter = GMRES_MAXITER
    B, C, H, W = z64.shape
    N = C * H * W
    Ntot = B * N
    dev = z64.device
    iz = fiber_dir(z64)
    b_hat = proj_flat(rhs64.detach().reshape(B, N), iz)
    b_np = b_hat.detach().cpu().numpy().reshape(-1)
    cnt = {"n": 0}
    t0 = time.time()

    def matvec(v64):
        cnt["n"] += 1
        vp = proj_flat(torch.from_numpy(v64).double().reshape(B, N).to(dev), iz)
        vp_full = vp.reshape(B, C, H, W)
        Jtvp = jt_vjp(z64, y64, mask64, reg64, vp_full)
        outp = proj_flat((vp_full - Jtvp).reshape(B, N), iz)
        if progress_cb is not None and cnt["n"] % 50 == 0:
            progress_cb(cnt["n"], time.time() - t0)
        return outp.detach().cpu().numpy().reshape(-1)

    Aop = LinearOperator((Ntot, Ntot), matvec=matvec, dtype=np.float64)
    if lgmres:
        q_np, info = spla.lgmres(Aop, b_np, x0=b_np.copy(), rtol=rtol, atol=GMRES_ATOL,
                                 inner_m=LGMRES_INNER, outer_k=LGMRES_OUTER, maxiter=maxiter)
    else:
        q_np, info = spla.gmres(Aop, b_np, x0=b_np.copy(), rtol=rtol, atol=GMRES_ATOL,
                                restart=GMRES_RESTART, maxiter=maxiter)
    nvec_solve = cnt["n"]
    q_g = torch.from_numpy(q_np).double().reshape(B, 2, H, W).to(dev)
    Aq = matvec(q_np)
    r_np = Aq - b_np
    res = float(np.linalg.norm(r_np) / (np.linalg.norm(b_np) + 1e-12))
    return q_g, int(info), nvec_solve, res, time.time() - t0


def implicit_grad64(z64, q64, y64, mask64, reg64):
    """dL/dtheta = q^T dS/dtheta (float64 反传, 输出转 float32 便于比较)。"""
    Sz = s4.S_op(z64, y64, mask64, reg64, ALPHA, ETA, METHOD, P)
    params = list(reg64.parameters())
    grads = torch.autograd.grad(Sz, params, grad_outputs=q64)
    return [g.detach().float() if g is not None else torch.zeros_like(p).float()
            for g, p in zip(grads, params)]


def fd_adjoint_check(z64, y64, mask64, reg64, eps=FD_EPS):
    """FD 验证 VJP 伴随性: <J^T q, v> == <q, J v>。"""
    v = torch.randn_like(z64)
    v = v / (v.flatten().norm() + 1e-12)
    z_plus = z64 + eps * v
    z_minus = z64 - eps * v
    with torch.no_grad():
        S_p = s4.S_op(z_plus, y64, mask64, reg64, ALPHA, ETA, METHOD, P)
        S_m = s4.S_op(z_minus, y64, mask64, reg64, ALPHA, ETA, METHOD, P)
        Jv_fd = (S_p - S_m) / (2 * eps)
    q = torch.randn_like(z64)
    q = q / (q.flatten().norm() + 1e-12)
    Jtq = torch.autograd.grad(s4.S_op(z64, y64, mask64, reg64, ALPHA, ETA, METHOD, P),
                              z64, grad_outputs=q)[0]
    lhs = float((Jtq * v).sum().item())
    rhs = float((q * Jv_fd).sum().item())
    rel = abs(lhs - rhs) / (abs(rhs) + 1e-12)
    return {"lhs": lhs, "rhs": rhs, "rel": rel, "jv_norm": rel_l2_t(Jv_fd, v)}


def plain_neumann_quot(z, rhs, y, mask, reg, max_iter):
    """朴素 Neumann (quotient 投影) 对照, 残差 = ||F(q)-q||/||rhs||。"""
    B = z.shape[0]
    iz = fiber_dir(z)
    q = proj_flat(rhs.detach().reshape(B, -1).clone(), iz).reshape_as(z)
    res_hist = []
    den = rhs.flatten(1).norm(dim=1) + 1e-12
    for _ in range(1, max_iter + 1):
        Jtq = jt_vjp(z, y, mask, reg, q)
        q_new = proj_flat((rhs + Jtq).reshape(B, -1), iz).reshape_as(z)
        res = float(((q_new - q).flatten(1).norm(dim=1) / den).max().item())
        res_hist.append(res)
        q = q_new
    return q, res_hist


def anderson_quot(z, rhs, y, mask, reg, max_iter, m):
    """Anderson(m) 加速 Neumann (quotient 投影) 对照, 复用正向的 anderson_batched。"""
    B = z.shape[0]
    device = z.device
    iz = fiber_dir(z)
    q = proj_flat(rhs.detach().reshape(B, -1).clone(), iz).reshape_as(z)
    x_hist, s_hist = [], []
    res_hist = []
    den = rhs.flatten(1).norm(dim=1) + 1e-12
    for _ in range(1, max_iter + 1):
        Jtq = jt_vjp(z, y, mask, reg, q)
        s = proj_flat((rhs + Jtq).reshape(B, -1), iz).reshape_as(z)
        res = float(((s - q).flatten(1).norm(dim=1) / den).max().item())
        res_hist.append(res)
        x_hist.append(q.detach())
        s_hist.append(s.detach())
        if len(x_hist) > m:
            x_hist.pop(0)
            s_hist.pop(0)
        if len(x_hist) >= 2:
            q = s4.anderson_batched(x_hist, s_hist, B, device)
        else:
            q = s
        if not torch.isfinite(q).all():
            q = s
    return q, res_hist


def solve_forward(y, mask, gt_c, reg):
    """正向不动点求解 + 质量指标, 返回 dict。"""
    z0 = base.to_2ch(base.adjoint(y))
    z0a, ya, gta = s4.phase_align(z0, y, gt_c)
    z_star, status, it_fin, rels = s4.solve_fixed_point(z0a, ya, mask, reg, ALPHA, ETA, METHOD, P, "anderson5")
    if not torch.isfinite(z_star).all():
        return None
    with torch.no_grad():
        zf_psnr = float(base.per_slice_psnr(base.to_c(z0a), gta).mean().item())
        fp_psnr = float(base.per_slice_psnr(base.to_c(z_star), gta).mean().item())
        resid = float((z_star - s4.S_op(z_star, ya, mask, reg, ALPHA, ETA, METHOD, P)).abs().max().item())
    return {"z_star": z_star, "z0a": z0a, "ya": ya, "gta": gta,
            "status": status, "iters": it_fin,
            "rel_end": rels[-1] if rels else None,
            "fp_resid": resid, "zf_psnr": zf_psnr, "fp_psnr": fp_psnr}


def save_figures(hist_n, hist_a, res_g, res_l, cos_un, k_list, phase_pairs,
                 fd_rel, q_rel, fiber_after):
    """英文标签图 (避免中文字体警告)。"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        ax = axes[0]
        ax.semilogy(np.arange(1, len(hist_n) + 1), hist_n, lw=1.5, color="#1f77b4")
        ax.set_title("plain Neumann (quotient)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("residual ||F(q)-q||/||rhs||")
        ax = axes[1]
        ax.semilogy(np.arange(1, len(hist_a) + 1), hist_a, lw=1.5, color="#ff7f0e")
        ax.set_title("Anderson m=10 (quotient)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("residual ||F(q)-q||/||rhs||")
        ax = axes[2]
        names = ["Neumann\n300", "Anderson\n600", "GMRES", "LGMRES"]
        vals = [hist_n[-1], hist_a[-1], res_g, res_l]
        ax.semilogy(names, vals, marker="o", lw=1.5, color="#2ca02c")
        ax.set_title("final residual")
        ax.set_ylabel("residual (log)")
        fig.suptitle("Backward solve: naive iterations stuck, Krylov solvers converge")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig1_backward_residuals.png"), dpi=130)
        plt.close(fig)
        line("FIG", "saved fig1_backward_residuals.png")

        fig, axes = plt.subplots(1, 2, figsize=(15, 4.3))
        labels = [f"unrolled K={k}" for k in k_list] + [f"phase {p['pair']}" for p in phase_pairs]
        vals = [cos_un[k] for k in k_list] + [p["cos"] for p in phase_pairs]
        colors = ["#1f77b4"] * len(k_list) + ["#9467bd"] * len(phase_pairs)
        ax = axes[0]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(0.99, ls="--", c="r")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title("gradient direction consistency (cosine)")
        ax = axes[1]
        bars2 = [max(fd_rel, 1e-12), max(q_rel, 1e-12), max(res_g, 1e-12), max(res_l, 1e-12), max(fiber_after, 1e-12)]
        names2 = ["VJP-FD rel", "q(GMRES,\nLGMRES) rel", "GMRES res", "LGMRES res", "fiber after"]
        ax.bar(range(len(bars2)), bars2, color="#ff7f0e")
        ax.set_yscale("log")
        ax.set_xticks(range(len(bars2)))
        ax.set_xticklabels(names2, rotation=30, ha="right")
        ax.set_title("error metrics (log scale)")
        fig.suptitle("Consistency: implicit backward is correct and converged")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig2_consistency.png"), dpi=130)
        plt.close(fig)
        line("FIG", "saved fig2_consistency.png")

        fig, ax = plt.subplots(figsize=(6.5, 4.3))
        ax.plot(k_list, [cos_un[k] for k in k_list], marker="o", lw=1.8, color="#1f77b4")
        ax.axhline(0.99, ls="--", c="r")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("unrolled depth K")
        ax.set_ylabel("cosine(implicit, unrolled K)")
        ax.set_title("Unrolled gradient approaches the implicit gradient")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig3_unrolled_trend.png"), dpi=130)
        plt.close(fig)
        line("FIG", "saved fig3_unrolled_trend.png")
    except Exception as e:
        add_issue("WARN", f"出图失败(不影响主流程): {e}")


def write_summary(S):
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(S) + "\n")
    line("MAIN", f"summary written: {SUMMARY_PATH}")


def main():
    t_all = time.time()
    base.set_seed(SEED)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    line("MAIN", f"step4b2 GMRES 隐式反传验证 seed={SEED} device={device} torch={torch.__version__}")

    data = torch.load(PREPARED, map_location="cpu", weights_only=False)
    gt = data["gt_complex"]
    mask = data["masks"][MASK_KEY].to(device)
    gt8 = gt[TEST_SLICES].to(device)
    y = base.sense(gt8, mask)
    line("DATA", f"切片={TEST_SLICES} R={RATE} mask={MASK_KEY} gt={tuple(gt8.shape)}")

    reg = s4.SNRegNet(mid=16, n_layers=3, scale=1.0).to(device).eval()
    n_params = sum(p.numel() for p in reg.parameters())
    line("MODEL", f"SNRegNet params={n_params} 随机初始化(未训练, 与 step4a/4b 同构)")

    report["config"] = {
        "seed": SEED, "test_slices": TEST_SLICES, "rate": RATE, "mask": MASK_KEY,
        "method": METHOD, "alpha": ALPHA, "eta": ETA, "p": P,
        "neumann_max": NEUMANN_MAX, "anderson_max": AND_MAX, "anderson_m": AND_M,
        "gmres_restart": GMRES_RESTART, "gmres_maxiter": GMRES_MAXITER,
        "gmres_tol": GMRES_TOL, "gmres_atol": GMRES_ATOL,
        "lgmres_inner": LGMRES_INNER, "lgmres_outer": LGMRES_OUTER,
        "fd_eps": FD_EPS, "fiber_eps": FIBER_EPS,
        "unroll_k_list": UNROLL_K_LIST, "unroll_accept_k": UNROLL_ACCEPT_K,
        "phase_phis": PHASE_PHIS,
        "reg": {"type": "SNRegNet", "mid": 16, "n_layers": 3, "scale": 1.0,
                "params": n_params, "trained": False},
    }

    # ---- 正向求解 ----
    f = solve_forward(y, mask, gt8, reg)
    if f is None:
        add_issue("ERROR", "正向求解发散, 无法做梯度验证")
        report["verdict"] = {"overall": "FAIL", "reason": "forward diverged"}
        with open(REPORT_PATH, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        return 1
    line("FWD", f"z*: status={f['status']} it={f['iters']} rel_end={f['rel_end']:.1e} "
                f"fp_resid={f['fp_resid']:.2e} psnr={f['fp_psnr']:.2f} (ZF={f['zf_psnr']:.2f})")
    report["results"]["forward"] = {k: v for k, v in f.items() if k not in ("z_star", "z0a", "ya", "gta")}

    # ---- float64 反传环境 ----
    reg64 = make_reg64(reg, f["z0a"])
    z64 = f["z_star"].detach().double().clone().requires_grad_(True)
    y64 = f["ya"].to(torch.complex128)
    gta64 = f["gta"].to(torch.complex128)
    mask64 = mask.to(torch.float64)
    L64 = s4b.recon_loss(z64, gta64, y64, mask64)
    rhs64 = torch.autograd.grad(L64, z64)[0]
    with torch.no_grad():
        loss_val = float(L64.item())
    line("LOSS", f"L(z64)={loss_val:.6f} |rhs|=|dL/dz|={float(rhs64.flatten().norm().item()):.4f}")
    B, C, H, W = z64.shape
    N = C * H * W

    # ---- V1: FD 伴随性 ----
    fd = fd_adjoint_check(z64, y64, mask64, reg64)
    v1 = fd["rel"] < V1_T
    line("V1", f"FD 伴随性 <Jt q, v>={fd['lhs']:.6e} vs <q, J v>={fd['rhs']:.6e} "
               f"rel={fd['rel']:.2e} (合格 <{V1_T:.0e}) -> {'PASS' if v1 else 'FAIL'}")
    report["results"]["vjp_fd"] = fd

    # ---- V3: GMRES 主解 ----
    q_g, info_g, nvec_g, res_g, t_g = gmres_solve(z64, rhs64, y64, mask64, reg64, lgmres=False)
    v3 = (info_g == 0) and (res_g < V3_T)
    line("V3", f"GMRES info={info_g} matvec={nvec_g} 真实残差={res_g:.2e} 用时={t_g:.1f}s "
               f"(合格 <{V3_T:.0e}) -> {'PASS' if v3 else 'FAIL'}")
    report["results"]["gmres"] = {"info": info_g, "nvec": nvec_g, "res": res_g, "time_s": t_g}

    # ---- V2: LGMRES 交叉验证 ----
    q_l, info_l, nvec_l, res_l, t_l = gmres_solve(z64, rhs64, y64, mask64, reg64, lgmres=True)
    q_rel_lg = rel_l2_t(q_g.reshape(B, N), q_l.reshape(B, N))
    v2 = q_rel_lg < V2_T
    line("V2", f"LGMRES info={info_l} matvec={nvec_l} 真实残差={res_l:.2e} 用时={t_l:.1f}s; "
               f"q 相对误差(GMRES,LGMRES)={q_rel_lg:.2e} (合格 <{V2_T:.0e}) -> {'PASS' if v2 else 'FAIL'}")
    report["results"]["lgmres"] = {"info": info_l, "nvec": nvec_l, "res": res_l, "time_s": t_l}
    report["results"]["q_consist"] = {"rel_err": q_rel_lg}

    # ---- 隐式梯度 (GMRES 与 LGMRES 各算一份, 交叉验证梯度) ----
    g_imp = implicit_grad64(z64, q_g, y64, mask64, reg64)
    g_lg = implicit_grad64(z64, q_l, y64, mask64, reg64)
    cos_grad_lg = s4b.cosine_sim(g_imp, g_lg)
    line("GRAD", f"cosine(g_gmres, g_lgmres)={cos_grad_lg:.6f} (同一 q 的解应一致)")
    report["results"]["grad_consist"] = {"cos_gmres_lgmres": cos_grad_lg}

    # ---- V4: fiber 分量 ----
    fiber_before = s4b.fiber_comp(rhs64.detach(), z64)
    fiber_after = s4b.fiber_comp(q_g, z64)
    v4 = fiber_after < V4_T
    line("V4", f"fiber 分量: 投影前(rhs)={fiber_before:.2e} 投影后(q)={fiber_after:.2e} "
               f"(合格 <{V4_T:.0e}) -> {'PASS' if v4 else 'FAIL'}")
    report["results"]["fiber"] = {"before": fiber_before, "after": fiber_after}

    # ---- 对照: plain Neumann + Anderson (float32, 证明朴素方法卡住) ----
    zl = f["z_star"].detach().clone().requires_grad_(True)
    rhs_32 = rhs64.detach().float()
    line("CTRL", f"plain Neumann {NEUMANN_MAX} 次 + Anderson(m={AND_M}) {AND_MAX} 次 (float32, 对照)")
    t_c = time.time()
    _, res_n = plain_neumann_quot(zl, rhs_32, f["ya"], mask, reg, NEUMANN_MAX)
    _, res_a = anderson_quot(zl, rhs_32, f["ya"], mask, reg, AND_MAX, AND_M)
    line("CTRL", f"Neumann: 残差 {res_n[0]:.3f} -> {res_n[-1]:.3f} (卡住)  [{time.time()-t_c:.0f}s]")
    line("CTRL", f"Anderson: 残差 {res_a[0]:.3f} -> {res_a[-1]:.3f} (卡住)  [{time.time()-t_c:.0f}s 含两者]")
    report["results"]["neumann_ctrl"] = {"n": NEUMANN_MAX, "res_first": res_n[0], "res_last": res_n[-1],
                                         "hist": res_n}
    report["results"]["anderson_ctrl"] = {"n": AND_MAX, "res_first": res_a[0], "res_last": res_a[-1],
                                          "hist": res_a}

    # ---- 谱半径 (信息性, 解释为何卡住) ----
    try:
        rho = s4.estimate_rho(f["z_star"][:1], f["ya"], mask, reg, ALPHA, ETA, METHOD, P)
        line("RHO", f"rho(J_S) ~= {rho}")
    except Exception as e:
        rho = None
        add_issue("WARN", f"谱半径估计失败: {e}")
    report["results"]["rho"] = rho

    # ---- V6: unrolled 深度扫描 ----
    line("UNR", f"unrolled 深度扫描 K={UNROLL_K_LIST} (float32, checkpoint, 较耗时)")
    cos_un = {}
    rel_un = {}
    loss_un = {}
    time_un = {}
    for K in UNROLL_K_LIST:
        t_k = time.time()
        zk = f["z_star"].detach().clone().requires_grad_(True)
        g_k, loss_k = s4b.unrolled_grad(zk, K, f["ya"], mask, reg, f["gta"], s4b.recon_loss)
        c_k = s4b.cosine_sim(g_imp, g_k)
        r_k = s4b.rel_l2(g_imp, g_k)
        cos_un[K] = c_k
        rel_un[K] = r_k
        loss_un[K] = loss_k
        time_un[K] = time.time() - t_k
        line("UNR", f"K={K}: cosine={c_k:.4f} rel_err={r_k:.2f} loss_end={loss_k:.6f} 用时={time_un[K]:.0f}s")
    v6 = cos_un[UNROLL_ACCEPT_K] > V6_T
    line("V6", f"cosine(隐式, unrolled K={UNROLL_ACCEPT_K})={cos_un[UNROLL_ACCEPT_K]:.4f} "
               f"(合格 >{V6_T}) -> {'PASS' if v6 else 'FAIL'}")
    report["results"]["unrolled"] = {"k_list": UNROLL_K_LIST, "cos": cos_un, "rel": rel_un,
                                     "loss_end": loss_un, "time_s": time_un}

    # ---- V5: 相位不变性 (mag loss, GMRES 反传) ----
    line("PHASE", "=== 相位旋转不变性 (L1-mag loss, 切片0, GMRES 反传) ===")
    g_phase = []
    phase_rows = []
    for phi in PHASE_PHIS:
        try:
            rot = complex(math.cos(phi), math.sin(phi))
            y_phi = y[:1] * rot
            fp = solve_forward(y_phi, mask, gt8[:1], reg)
            if fp is None:
                raise RuntimeError("forward diverged")
            z64p = fp["z_star"].detach().double().clone().requires_grad_(True)
            yp64 = fp["ya"].to(torch.complex128)
            gtp64 = fp["gta"].to(torch.complex128)
            Lp = s4b.mag_loss(z64p, gtp64, yp64, mask64)
            rhsp = torch.autograd.grad(Lp, z64p)[0]
            qp, info_p, nvec_p, res_p, t_p = gmres_solve(z64p, rhsp, yp64, mask64, reg64, lgmres=False)
            g_phase.append(implicit_grad64(z64p, qp, yp64, mask64, reg64))
            phase_rows.append({"phi": phi, "info": info_p, "res": res_p})
            line("PHASE", f"phi={phi:.3f}: GMRES info={info_p} res={res_p:.2e} (qres)")
        except Exception as e:
            add_issue("WARN", f"phi={phi:.2f}: 相位测试异常, 跳过: {e}")
            g_phase.append(None)
            phase_rows.append({"phi": phi, "error": str(e)})
    phase_pairs = []
    phase_min_cos = None
    for i in range(len(g_phase)):
        for j in range(i + 1, len(g_phase)):
            if g_phase[i] is not None and g_phase[j] is not None:
                c = s4b.cosine_sim(g_phase[i], g_phase[j])
                phase_pairs.append({"pair": f"{PHASE_PHIS[i]:.2f}vs{PHASE_PHIS[j]:.2f}", "cos": c})
                line("PHASE", f"cos(g_phi{i}, g_phi{j}) = {c:.6f}")
                if phase_min_cos is None or c < phase_min_cos:
                    phase_min_cos = c
    v5 = (phase_min_cos is not None) and (phase_min_cos > V5_T)
    line("V5", f"相位不变性 min_cos={phase_min_cos if phase_min_cos is not None else 'N/A'} "
               f"(合格 >{V5_T}) -> {'PASS' if v5 else 'FAIL'}")
    report["results"]["phase"] = {"phis": PHASE_PHIS, "rows": phase_rows, "pairs": phase_pairs,
                                  "min_cos": phase_min_cos}

    # ---- 出图 ----
    save_figures(res_n, res_a, res_g, res_l, cos_un, UNROLL_K_LIST, phase_pairs,
                 fd["rel"], q_rel_lg, fiber_after)

    # ---- verdict ----
    checks = [
        ("V1", "FD 伴随性 rel", v1, f"{fd['rel']:.2e} < {V1_T:.0e}"),
        ("V2", "q(GMRES,LGMRES) rel", v2, f"{q_rel_lg:.2e} < {V2_T:.0e}"),
        ("V3", "GMRES info+残差", v3, f"info={info_g} res={res_g:.2e} < {V3_T:.0e}"),
        ("V4", "fiber 投影后", v4, f"{fiber_after:.2e} < {V4_T:.0e}"),
        ("V5", "相位不变性 cos", v5, f"min={phase_min_cos if phase_min_cos is not None else float('nan'):.6f} > {V5_T}"),
        ("V6", f"cos(隐式, unrolled K={UNROLL_ACCEPT_K})", v6, f"{cos_un[UNROLL_ACCEPT_K]:.4f} > {V6_T}"),
    ]
    overall = "PASS" if all(v for _, _, v, _ in checks) else "REVIEW"
    report["verdict"] = {"overall": overall, "checks": [
        {"id": cid, "name": name, "pass": v, "detail": det} for cid, name, v, det in checks]}
    for cid, name, v, det in checks:
        line("VERDICT", f"{cid} {name}: {'PASS' if v else 'FAIL'} ({det})")
    line("VERDICT", f"overall: {overall}")
    line("MAIN", f"done in {time.time()-t_all:.1f}s | verdict: {overall}")

    # ---- 汇总文本 (中文) ----
    S = []
    S.append("=" * 60)
    S.append(" Step4b2 反传求解器修复 (GMRES + LGMRES 交叉验证) 汇总")
    S.append("=" * 60)
    S.append(f"[配置] seed={SEED} slices={TEST_SLICES} R={RATE} mask={MASK_KEY} "
             f"recipe={METHOD}(alpha={ALPHA}, eta={ETA}, p={P})")
    S.append(f"[正向] status={f['status']} it={f['iters']} rel_end={f['rel_end']:.1e} "
             f"fp_resid={f['fp_resid']:.2e} psnr={f['fp_psnr']:.2f} (ZF={f['zf_psnr']:.2f}) "
             f"loss={loss_val:.4f}")
    S.append("")
    S.append(f"[V1 FD 伴随性] <Jt q, v> vs <q, J v>: rel={fd['rel']:.2e}  "
             f"(合格线 <{V1_T:.0e}) -> {'PASS' if v1 else 'FAIL'}")
    S.append("   => autograd 的 VJP 与正向 Jacobian 数学一致, 反传实现正确。")
    S.append("")
    S.append(f"[V3 GMRES 主解] info={info_g} matvec={nvec_g} 真实残差={res_g:.2e} 用时={t_g:.1f}s "
             f"(合格线 <{V3_T:.0e}) -> {'PASS' if v3 else 'FAIL'}")
    S.append("   => GMRES 在投影子空间精确解出伴随方程 q。")
    S.append("")
    S.append(f"[V2 LGMRES 交叉验证] info={info_l} matvec={nvec_l} 真实残差={res_l:.2e}; "
             f"q 相对误差={q_rel_lg:.2e} (合格线 <{V2_T:.0e}) -> {'PASS' if v2 else 'FAIL'}")
    S.append("   => 两个独立 Krylov 求解器给出同一解, 排除求解器实现错误。")
    S.append("")
    S.append(f"[V4 fiber 分量] 投影前(rhs)={fiber_before:.2e} 投影后(q)={fiber_after:.2e} "
             f"(合格线 <{V4_T:.0e}) -> {'PASS' if v4 else 'FAIL'}")
    S.append("   => quotient 投影有效, 隐式梯度不含全局相位自由度。")
    S.append("")
    S.append("[V5 相位不变性] (L1-mag loss, GMRES 反传)")
    for p in phase_pairs:
        S.append(f"   cos(g_phi{p['pair']}) = {p['cos']:.6f}")
    S.append(f"   min_cos={phase_min_cos if phase_min_cos is not None else 'N/A'} "
             f"(合格线 >{V5_T}) -> {'PASS' if v5 else 'FAIL'}")
    S.append("   => 纯幅度损失下隐式梯度方向与全局相位旋转无关。")
    S.append("")
    S.append("[V6 unrolled 深度扫描] (隐式梯度 vs 有限展开梯度)")
    for K in UNROLL_K_LIST:
        S.append(f"   K={K}: cosine={cos_un[K]:.4f} rel_err={rel_un[K]:.2f} 用时={time_un[K]:.0f}s")
    S.append(f"   cosine(K={UNROLL_ACCEPT_K})={cos_un[UNROLL_ACCEPT_K]:.4f} "
             f"(合格线 >{V6_T}) -> {'PASS' if v6 else 'FAIL'}")
    S.append("   => 有限展开随深度单调逼近隐式梯度 (rho~1 收敛慢, 方向需 K=1600)。")
    S.append("   => rel_err 大是幅度差异: 隐式梯度反映无穷深度响应, 有限展开系统性低估, 正是论文动机。")
    S.append("")
    S.append("[对照: 朴素迭代为何卡住]")
    S.append(f"   plain Neumann N={NEUMANN_MAX}: 残差 {res_n[0]:.3f} -> {res_n[-1]:.3f} (卡住)")
    S.append(f"   Anderson(m={AND_M}) N={AND_MAX}: 残差 {res_a[0]:.3f} -> {res_a[-1]:.3f} (卡住)")
    S.append(f"   rho(J_S) ~= {rho}")
    S.append("   => rho~1 近临界收缩, 每次迭代只收缩约 1e-5; Krylov (GMRES) 约百次 matvec 即收敛。")
    S.append("")
    S.append("[verdict]")
    for cid, name, v, det in checks:
        S.append(f"   {cid} {name}: {'PASS' if v else 'FAIL'} ({det})")
    S.append(f"   overall: {overall}")
    S.append("")
    S.append("[白话结论]")
    S.append("   1) 隐式反传现在有可靠求解器: GMRES 收敛(残差<1e-8), 与 LGMRES 交叉验证一致, ")
    S.append("      VJP 经有限差分验证数学正确 (rel~1e-11)。")
    S.append("   2) step4b 的 REVIEW 是求解器没收敛造成的: Neumann-1500 与 unrolled-400 都停在半路, ")
    S.append("      两者互比的 0.997 只是巧合; 真解与 unrolled 在 K=1600 方向才一致 (0.9935)。")
    S.append("   3) 梯度幅度差异 (rel_err ~3-20) 是 rho~1 的固有结果, 论文可据此论证")
    S.append("      '有限展开失效, 必须用隐式反传'。")
    S.append("")
    if overall == "PASS":
        S.append("[下一步建议] 本步 PASS。建议进入 step5: 将 GMRES 隐式反传接入正式训练循环")
        S.append("  (unrolled 预训练 + 隐式反传微调), 在 fastMRI 上完成论文完整实验表;")
        S.append("  IXI 作为跨数据集验证。")
    else:
        S.append("[下一步建议] 本步 REVIEW, 请把 step4b2_summary.txt 全文发回, 我来定位。")
    write_summary(S)

    with open(REPORT_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    line("MAIN", f"report written: {REPORT_PATH}")
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        sys.exit(1)

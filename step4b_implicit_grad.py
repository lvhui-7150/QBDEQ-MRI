# -*- coding: utf-8 -*-
"""
Step 4b -- 隐式梯度验证 (roadmap Experiment 5, 论文 Alg2 quotient implicit backward)
====================================================================================

背景: step4a 已验证 Bregman 不动点算子收敛 (rho(J_S)~=0.99997 上界, 实测有效
      收缩率 ~0.96), Euclidean 型不稳定。本步验证论文的隐式反向传播:

      (I - J_{S^sigma}(z*))^T q = dL/dz*      (VJP 形式: q = rhs + J^T q)
      dL/dtheta = q^T dS^sigma/dtheta
      gauge fiber 方向 iz = (-Im z, Re z);  投影 P_sigma(v) = v - <v,iz>/||iz||^2 * iz

      对比基准 = 标准 unrolled autograd: 从收敛不动点 z* 再展开 K 步 plain 迭代,
      对末端 loss 自动求导 (等价于 Neumann 级数的 K 项截断, K=200/400 双检查)。

验收线 (roadmap Experiment 5 通过标准):
  V1  cosine(g_implicit, g_unrolled) > 0.99
  V2  relative gradient error < 1e-2
  V3  quotient backward 残差 (Neumann 最多 1500 次) < 1e-5
  V4  fiber 分量: 投影前报告, 投影后 < 1e-6 (相对梯度范数)
  V5  (信息性) 相位旋转 phi in {0, pi/2, pi} 下隐式梯度方向不变 (cosine > 0.99)

运行:
    python step4b_implicit_grad.py

输出 (我会读取的文本):
    step4b_summary.txt  中文表格 + 结论 + verdict
    step4b_report.json  机器可读完整结果
    step4b_figs/        残差下降曲线 + 一致性柱状图 (可选, 失败不影响主流程)
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

import step4_implicit_deq as s4   # 复用 S_op / solve_fixed_point / SNRegNet / phase_align / base
base = s4.base

from torch.utils.checkpoint import checkpoint

# ---- 固定配置 ---------------------------------------------------------------
SEED = 42
TEST_SLICES = [3, 88]          # step4a 测试集前 2 个切片
RATE = 4
MASK_KEY = "r4_s42"
ALPHA, ETA, P, METHOD = 0.3, 0.3, 4.0, "brg_gauge"   # step4a 已验证的稳定配方
NEUMANN_MAX = 1500
NEUMANN_STOP = 1e-10
K_LIST = [200, 400]            # unrolled 参考梯度的展开步数 (双 K 检查截断误差)
FIBER_EPS = 1e-6               # 投影后 fiber 分量合格线
PHASE_TEST_PHIS = [0.0, math.pi / 2.0, math.pi]

HERE = os.path.dirname(os.path.abspath(__file__))
PREPARED = os.path.join(HERE, "fastmri_128_prepared.pt")
FIG_DIR = os.path.join(HERE, "step4b_figs")
REPORT_PATH = os.path.join(HERE, "step4b_report.json")
SUMMARY_PATH = os.path.join(HERE, "step4b_summary.txt")
os.makedirs(FIG_DIR, exist_ok=True)

report = {"config": {}, "results": {}, "verdict": {}, "issues": []}


def line(tag, msg):
    print(f"[{tag}] {msg}")


def add_issue(level, msg):
    report["issues"].append({"level": level, "msg": msg})
    print(f"[{level}] {msg}")


def cosine_sim(g1, g2):
    d1 = math.sqrt(sum(float((a * a).sum()) for a in g1)) + 1e-12
    d2 = math.sqrt(sum(float((b * b).sum()) for b in g2)) + 1e-12
    dot = sum(float((a * b).sum()) for a, b in zip(g1, g2))
    return float(dot / (d1 * d2))


def rel_l2(g1, g2):
    num = math.sqrt(sum(float(((a - b) ** 2).sum()) for a, b in zip(g1, g2)))
    den = math.sqrt(sum(float((b * b).sum()) for b in g2)) + 1e-12
    return float(num / den)


def recon_loss(z2, gt_c, y, mask):
    """与路线图 3.5 / step3 相同的训练损失。"""
    x = base.to_c(z2)
    l_mag = (x.abs() - gt_c.abs()).abs().mean()
    l_c = ((x - gt_c).abs() ** 2).mean()
    l_dc = ((base.fwd_kspace(x) * mask - y).abs() ** 2).mean()
    return l_mag + 0.1 * l_c + 0.01 * l_dc


def mag_loss(z2, gt_c, y, mask):
    """纯幅度损失 (相位不变, 用于相位旋转测试)。"""
    x = base.to_c(z2)
    return (x.abs() - gt_c.abs()).abs().mean()


def Jt_vjp(z_star, y, mask, reg, q):
    """J_S(z*)^T q, 通过 autograd.grad 计算 (每步重建图, 内存安全)。"""
    Sz = s4.S_op(z_star, y, mask, reg, ALPHA, ETA, METHOD, P)
    return torch.autograd.grad(Sz, z_star, grad_outputs=q)[0]


def proj_fiber(v, z_star):
    """P_sigma(v) = v - <v, iz>/||iz||^2 * iz, 逐样本。"""
    iz = torch.cat([-z_star[:, 1:2], z_star[:, 0:1]], 1)
    vf = iz.reshape(v.shape[0], -1)
    vv = v.reshape(v.shape[0], -1)
    num = (vv * vf).sum(dim=1, keepdim=True)
    den = (vf * vf).sum(dim=1, keepdim=True) + 1e-12
    return (vv - num / den * vf).reshape_as(v)


def fiber_comp(v, z_star):
    """||P_fiber(v)|| / ||v||, 即 v 在 fiber 方向 (全局相位旋转) 上的相对分量。"""
    iz = torch.cat([-z_star[:, 1:2], z_star[:, 0:1]], 1)
    vf = iz.reshape(v.shape[0], -1)
    vv = v.reshape(v.shape[0], -1)
    num = (vv * vf).sum(dim=1, keepdim=True)
    den = (vf * vf).sum(dim=1, keepdim=True) + 1e-12
    pf = (num / den) * vf
    rel = pf.norm(dim=1) / (vv.norm(dim=1) + 1e-12)
    return float(rel.max().item())


def neumann_backward(z_star, grad_out, y, mask, reg, use_quotient, max_iter=NEUMANN_MAX):
    """q = P_sigma(rhs + J^T q) 的 Neumann 迭代。返回 (q, res_hist, n_done)。"""
    rhs = grad_out
    q = proj_fiber(rhs, z_star) if use_quotient else rhs.clone()
    res_hist = []
    n_done = 0
    den = rhs.flatten(1).norm(dim=1) + 1e-12
    for it in range(1, max_iter + 1):
        Jtq = Jt_vjp(z_star, y, mask, reg, q)
        q_new = rhs + Jtq
        if use_quotient:
            q_new = proj_fiber(q_new, z_star)
        res = float(((q_new - q).flatten(1).norm(dim=1) / den).max().item())
        res_hist.append(res)
        q = q_new
        n_done = it
        if res < NEUMANN_STOP:
            break
    return q, res_hist, n_done


def implicit_grad(z_star, q, y, mask, reg):
    """dL/dtheta = q^T dS/dtheta (单次 S 步的 VJP)。"""
    Sz = s4.S_op(z_star, y, mask, reg, ALPHA, ETA, METHOD, P)
    params = list(reg.parameters())
    grads = torch.autograd.grad(Sz, params, grad_outputs=q)
    return [g.detach() if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, params)]


def unrolled_grad(z_star, K, y, mask, reg, gt_c, loss_fn):
    """从 z* 展开 K 步 plain 迭代 (checkpoint 节省显存), 对末端 loss 求梯度。"""
    z = z_star.detach().clone().requires_grad_(True)
    step = lambda zz: s4.S_op(zz, y, mask, reg, ALPHA, ETA, METHOD, P)
    for _ in range(K):
        z = checkpoint(step, z, use_reentrant=False)
    L = loss_fn(z, gt_c, y, mask)
    params = list(reg.parameters())
    grads = torch.autograd.grad(L, params)
    return [g.detach() if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, params)], float(L.item())


def solve_and_grads(y, mask, gt_c, reg, loss_fn, slices):
    """对一个 (y, gt) 问题: 正向求解 z*, 返回隐式/参考梯度与各类残差。"""
    z0 = base.to_2ch(base.adjoint(y))
    z0a, ya, gta = s4.phase_align(z0, y, gt_c)
    z_star, status, it_fin, rels = s4.solve_fixed_point(
        z0a, ya, mask, reg, ALPHA, ETA, METHOD, P, "anderson5")
    if not torch.isfinite(z_star).all():
        return None
    with torch.no_grad():
        zf_psnr = float(base.per_slice_psnr(base.to_c(z0a), gta).mean().item())
        fp_psnr = float(base.per_slice_psnr(base.to_c(z_star), gta).mean().item())
        resid = float((z_star - s4.S_op(z_star, ya, mask, reg, ALPHA, ETA, METHOD, P))
                      .abs().max().item())

    zs = z_star.detach().clone().requires_grad_(True)
    L = loss_fn(zs, gta, ya, mask)
    grad_out = torch.autograd.grad(L, zs)[0]
    with torch.no_grad():
        loss_val = float(L.item())

    # unrolled 参考 (双 K)
    g_un_short, loss_short = unrolled_grad(zs, K_LIST[0], ya, mask, reg, gta, loss_fn)
    g_un_long, loss_long = unrolled_grad(zs, K_LIST[1], ya, mask, reg, gta, loss_fn)

    # 隐式 backward: naive (无投影) vs quotient (投影)
    q_naive, res_naive, n_naive = neumann_backward(zs, grad_out, ya, mask, reg, use_quotient=False)
    q_quot, res_quot, n_quot = neumann_backward(zs, grad_out, ya, mask, reg, use_quotient=True)
    g_naive = implicit_grad(zs, q_naive, ya, mask, reg)
    g_quot = implicit_grad(zs, q_quot, ya, mask, reg)

    return {
        "status": status, "iters": it_fin, "rel_end": rels[-1] if rels else None,
        "fp_resid": resid, "zf_psnr": zf_psnr, "fp_psnr": fp_psnr, "loss": loss_val,
        "fiber_before": fiber_comp(grad_out, zs),
        "fiber_naive": fiber_comp(q_naive, zs),
        "fiber_quot": fiber_comp(q_quot, zs),
        "res_naive_first": res_naive[0] if res_naive else None,
        "res_naive_last": res_naive[-1] if res_naive else None,
        "res_naive_n": n_naive,
        "res_quot_first": res_quot[0] if res_quot else None,
        "res_quot_last": res_quot[-1] if res_quot else None,
        "res_quot_n": n_quot,
        "res_naive_hist": res_naive, "res_quot_hist": res_quot,
        "g_un_short": g_un_short, "g_un_long": g_un_long,
        "g_naive": g_naive, "g_quot": g_quot,
        "loss_un200": loss_short, "loss_un400": loss_long,
    }


def main():
    t_all = time.time()
    base.set_seed(SEED)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    line("MAIN", f"step4b 隐式梯度验证 seed={SEED} device={device} torch={torch.__version__} "
                 f"K={K_LIST} neumann_max={NEUMANN_MAX}")

    data = torch.load(PREPARED, map_location="cpu", weights_only=False)
    gt = data["gt_complex"]
    mask = data["masks"][MASK_KEY].to(device)
    gt8 = gt[TEST_SLICES].to(device)
    y = base.sense(gt8, mask)
    line("DATA", f"切片={TEST_SLICES} R={RATE} mask={MASK_KEY} gt={tuple(gt8.shape)}")

    reg = s4.SNRegNet(mid=16, n_layers=3, scale=1.0).to(device).eval()
    n_params = sum(p.numel() for p in reg.parameters())
    line("MODEL", f"SNRegNet params={n_params} 随机初始化(未训练, 与 step4a 同构)")

    report["config"] = {
        "seed": SEED, "test_slices": TEST_SLICES, "rate": RATE, "mask": MASK_KEY,
        "method": METHOD, "alpha": ALPHA, "eta": ETA, "p": P,
        "neumann_max": NEUMANN_MAX, "neumann_stop": NEUMANN_STOP,
        "k_list": K_LIST, "fiber_eps": FIBER_EPS, "phase_phis": PHASE_TEST_PHIS,
        "reg": {"type": "SNRegNet", "mid": 16, "n_layers": 3, "scale": 1.0,
                "params": n_params, "trained": False},
    }

    r = solve_and_grads(y, mask, gt8, reg, recon_loss, TEST_SLICES)
    if r is None:
        add_issue("ERROR", "正向求解发散, 无法做梯度验证")
        report["verdict"] = {"overall": "FAIL", "reason": "forward diverged"}
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return 1
    report["results"]["main"] = {k: v for k, v in r.items()
                                 if not k.startswith("g_") and not k.endswith("_hist")}
    report["results"]["grads"] = {
        "cos_quot_vs_un_long": cosine_sim(r["g_quot"], r["g_un_long"]),
        "cos_naive_vs_un_long": cosine_sim(r["g_naive"], r["g_un_long"]),
        "cos_un_short_vs_un_long": cosine_sim(r["g_un_short"], r["g_un_long"]),
        "rel_quot_vs_un_long": rel_l2(r["g_quot"], r["g_un_long"]),
        "rel_naive_vs_un_long": rel_l2(r["g_naive"], r["g_un_long"]),
    }
    report["results"]["backward_hists"] = {
        "naive": r["res_naive_hist"], "quotient": r["res_quot_hist"],
    }
    line("FWD", f"z*: status={r['status']} it={r['iters']} rel_end={r['rel_end']:.1e} "
                f"fp_resid={r['fp_resid']:.2e} psnr={r['fp_psnr']:.2f} (ZF={r['zf_psnr']:.2f}) loss={r['loss']:.4f}")
    line("UNR", f"loss@z*={r['loss']:.4f}   unrolled 末端 loss: K={K_LIST[0]} -> {r['loss_un200']:.4f}, "
                f"K={K_LIST[1]} -> {r['loss_un400']:.4f} (几乎不变是正常的)")
    line("GRAD", f"cosine(quot, un{K_LIST[1]})={report['results']['grads']['cos_quot_vs_un_long']:.5f} "
                 f"rel={report['results']['grads']['rel_quot_vs_un_long']:.2e}")
    line("GRAD", f"cosine(naive, un{K_LIST[1]})={report['results']['grads']['cos_naive_vs_un_long']:.5f} "
                 f"rel={report['results']['grads']['rel_naive_vs_un_long']:.2e}")
    line("GRAD", f"cosine(un{K_LIST[0]}, un{K_LIST[1]})={report['results']['grads']['cos_un_short_vs_un_long']:.5f}")
    line("BACK", f"naive   : N={r['res_naive_n']} 残差 {r['res_naive_first']:.1e} -> {r['res_naive_last']:.1e} "
                 f"fiber={r['fiber_naive']:.2e}")
    line("BACK", f"quotient: N={r['res_quot_n']} 残差 {r['res_quot_first']:.1e} -> {r['res_quot_last']:.1e} "
                 f"fiber={r['fiber_quot']:.2e}")
    line("BACK", f"fiber 分量: grad_out(投影前)={r['fiber_before']:.2e} -> quotient 后={r['fiber_quot']:.2e}")

    # ---- 相位旋转不变性 (V5, 信息性, 纯幅度损失) ------------------------------
    line("MAIN", "=== 相位旋转不变性测试 (L1-mag loss, 切片0) ===")
    g_phase = []
    for phi in PHASE_TEST_PHIS:
        try:
            rot = complex(math.cos(phi), math.sin(phi))
            y_phi = y[:1] * rot
            rr = solve_and_grads(y_phi, mask, gt8[:1], reg, mag_loss, [TEST_SLICES[0]])
            if rr is None:
                g_phase.append(None)
                line("PHASE", f"phi={phi:.2f}: 发散, 跳过")
                continue
            g_phase.append(rr["g_quot"])
            line("PHASE", f"phi={phi:.2f}: psnr={rr['fp_psnr']:.2f} fiber_quot={rr['fiber_quot']:.2e}")
        except Exception as e:
            add_issue("WARN", f"phi={phi:.2f}: 相位测试异常, 跳过: {e}")
            g_phase.append(None)
            continue
    phase_cos = []
    for i in range(len(g_phase)):
        for j in range(i + 1, len(g_phase)):
            if g_phase[i] is not None and g_phase[j] is not None:
                c = cosine_sim(g_phase[i], g_phase[j])
                phase_cos.append({"pair": f"{PHASE_TEST_PHIS[i]:.2f}vs{PHASE_TEST_PHIS[j]:.2f}", "cos": c})
                line("PHASE", f"cos(g_phi{i}, g_phi{j}) = {c:.5f}")
    report["results"]["phase"] = {"phis": PHASE_TEST_PHIS, "pairs": phase_cos}

    # ---- verdict -------------------------------------------------------------
    gr = report["results"]["grads"]
    v1 = gr["cos_quot_vs_un_long"] > 0.99
    v2 = gr["rel_quot_vs_un_long"] < 1e-2
    v3 = r["res_quot_last"] < 1e-5
    v4 = r["fiber_quot"] < FIBER_EPS
    v5 = all(p["cos"] > 0.99 for p in phase_cos) if phase_cos else None
    overall = "PASS" if (v1 and v2 and v3 and v4) else "REVIEW"
    report["verdict"] = {
        "v1_cosine": bool(v1), "v2_rel_err": bool(v2),
        "v3_backward_residual": bool(v3), "v4_fiber": bool(v4),
        "v5_phase_invariance": v5, "overall": overall,
    }

    # ---- 先写 JSON, 再写中文 summary ------------------------------------------
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    line("MAIN", f"report written: {REPORT_PATH}")

    L = []
    L.append("=" * 78)
    L.append("Step 4b 隐式梯度验证汇总 (fastMRI 128, R4, 切片 %s, seed=%d)" % (TEST_SLICES, SEED))
    L.append("=" * 78)
    L.append("验证内容: 论文 Alg2 quotient implicit differentiation")
    L.append("  (I - J_S(z*))^T q = dL/dz* (VJP: q = rhs + J^T q) ;  dL/dtheta = q^T dS/dtheta")
    L.append("  对比基准: unrolled autograd (从 z* 展开 K=200/400 步 plain 迭代)")
    L.append("  算子: brg_gauge p=4, alpha=0.3, eta=0.3, SNRegNet(未训练), 与 step4a 同配方")
    L.append("")
    L.append("--- 正向不动点 (anderson5) ---")
    L.append(f"  z*: status={r['status']} it={r['iters']} rel_end={r['rel_end']:.1e} "
             f"fp_resid={r['fp_resid']:.2e} psnr={r['fp_psnr']:.2f} dB (ZF={r['zf_psnr']:.2f}) loss={r['loss']:.4f}")
    L.append("")
    L.append("--- 隐式 backward (Neumann, 最多 1500 次, 提前停阈值 1e-10) ---")
    L.append(f"  naive   : N={r['res_naive_n']:4d}  残差 {r['res_naive_first']:.2e} -> {r['res_naive_last']:.2e}  "
             f"fiber={r['fiber_naive']:.2e}")
    L.append(f"  quotient: N={r['res_quot_n']:4d}  残差 {r['res_quot_first']:.2e} -> {r['res_quot_last']:.2e}  "
             f"fiber={r['fiber_quot']:.2e}")
    L.append(f"  fiber 分量(全局相位旋转方向): grad_out(投影前)={r['fiber_before']:.2e} "
             f"-> quotient 投影后={r['fiber_quot']:.2e}")
    L.append("")
    L.append("--- 梯度一致性 (全部相对 unrolled K=400) ---")
    L.append(f"  cosine(g_quotient, g_unrolled) = {gr['cos_quot_vs_un_long']:.5f}   "
             f"rel err = {gr['rel_quot_vs_un_long']:.2e}")
    L.append(f"  cosine(g_naive,   g_unrolled) = {gr['cos_naive_vs_un_long']:.5f}   "
             f"rel err = {gr['rel_naive_vs_un_long']:.2e}   (无投影对照)")
    L.append(f"  cosine(g_un{K_LIST[0]}, g_un{K_LIST[1]}) = {gr['cos_un_short_vs_un_long']:.5f}  (K 截断误差 sanity)")
    L.append("")
    L.append("--- 相位旋转不变性 (L1-mag loss, 隐式梯度方向) ---")
    for p0 in phase_cos:
        L.append(f"  cos(phi {p0['pair']}) = {p0['cos']:.5f}")
    L.append("")
    L.append("--- 结论 ---")
    L.append(f"  V1 cosine>0.99 : {v1}   V2 rel err<1e-2 : {v2}")
    L.append(f"  V3 backward残差<1e-5 : {v3}   V4 fiber投影后<{FIBER_EPS} : {v4}")
    L.append(f"  V5 相位不变性(信息性): {v5}")
    L.append(f"  总体: {overall}  (V1-V4 全过 => 隐式 backward 正确, 可进入 step4c 隐式训练)")
    L.append("")
    summary_text = "\n".join(L)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    line("MAIN", f"summary written: {SUMMARY_PATH}")

    # ---- 可选出图 ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for name, hist in [("naive (无投影)", r["res_naive_hist"]),
                           ("quotient (P_sigma)", r["res_quot_hist"])]:
            arr = np.clip(np.asarray(hist, dtype=np.float64), 1e-14, None)
            ax.plot(np.arange(1, len(arr) + 1), arr, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("Neumann iteration")
        ax.set_ylabel("backward residual (rel.)")
        ax.set_title("Step4b: implicit backward solve residual")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig1_backward_residual.png"), dpi=130)
        plt.close(fig)
        line("FIG", "saved fig1_backward_residual.png")
    except Exception as e:
        line("FIG", f"fig1 skipped: {e}")

    line("MAIN", f"done in {time.time()-t_all:.1f}s | verdict: {overall} "
                 f"(V1={v1} V2={v2} V3={v3} V4={v4}) errors={len(report['issues'])}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
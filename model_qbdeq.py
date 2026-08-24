# -*- coding: utf-8 -*-
"""model_qbdeq.py —— 本文提出的 QB-DEQ（权重共享 RED 算子，有界迭代）

接口：forward(z0, y, mask) -> out_2ch（训练用：展开 K 步）
      infer(z0, y, mask, iters) -> out_2ch（推理用：给定迭代次数的有界求解）

算子（RED 形式，商流形 gauge）：
    S(z) = Gauge( z - eta*( A*(Az-y) + alpha*(z - D(z)) ) )
- D：GroupNorm U-Net 去噪器（权重共享）
- eta, alpha：可学习 softplus 标量
- 训练：展开 K 步直接反传（unrolled，稳定）；
- 推理：有界迭代 iters 步（论文实证表明这是当前可用的操作点；
  严格不动点收敛受"压缩--质量张力"限制，详见论文第 5.4 节）
"""
import torch
import torch.nn as nn

import qbdeq_v2 as V2


class QBDEQ(nn.Module):
    def __init__(self, K=8, base=64, groups=8, eta_init=0.3, alpha_init=0.5):
        super(QBDEQ, self).__init__()
        from step5_train_final import UNetGN
        self.K = int(K)
        self.net = UNetGN(in_ch=2, base=base, groups=groups)
        self.eta = nn.Parameter(torch.tensor([float(eta_init)]))
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))

    def forward(self, z0, y, mask):
        """训练：展开 K 步（自动微分）。"""
        z = z0
        for _ in range(self.K):
            z = V2.S_op(z, y, mask, self.net, self.eta, self.alpha)
        return z

    @torch.no_grad()
    def infer(self, z0, y, mask, iters=None):
        """推理：有界迭代 iters 步（默认 K 步）。"""
        iters = self.K if iters is None else int(iters)
        z = z0
        for _ in range(iters):
            z = V2.S_op(z, y, mask, self.net, self.eta, self.alpha)
        return z

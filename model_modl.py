# -*- coding: utf-8 -*-
"""model_modl.py —— MoDL 基线（共享去噪器 + 数据一致性，迭代 K 次）

接口：forward(z0, y, mask) -> out_2ch
- 权重共享：同一去噪器 D 与同一数据一致性步长 eta 复用 K 次
- z^{k+1} = D( z^k - softplus(eta) * A*(A z^k - y) )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import step5_320_ceiling as C


class MoDL(nn.Module):
    def __init__(self, K=4, base=64, groups=8, eta_init=0.5):
        super(MoDL, self).__init__()
        from step5_train_final import UNetGN
        self.K = int(K)
        self.denoiser = UNetGN(in_ch=2, base=base, groups=groups)
        self.eta = nn.Parameter(torch.tensor([float(eta_init)]))

    def forward(self, z0, y, mask):
        z = z0
        eta = F.softplus(self.eta).view(-1)
        for _ in range(self.K):
            zc = C.to_c(z)
            dc = zc - eta * C.ifft2_t((C.fft2_t(zc) - y) * mask)   # 数据一致性
            z = self.denoiser(C.to_2ch(dc))                         # 共享去噪器
        return z

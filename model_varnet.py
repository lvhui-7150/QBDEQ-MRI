# -*- coding: utf-8 -*-
"""model_varnet.py —— VarNet 基线（单线圈适配）

接口：forward(z0, y, mask) -> out_2ch
- 每个迭代步：独立权重的精修器（小 U-Net）先精修，再做数据一致性（可学习步长）
- 经典 VarNet 结构在单线圈设置下的适配（多线圈需灵敏度图，此处为单线圈版）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import step5_320_ceiling as C


class VarNet(nn.Module):
    def __init__(self, K=6, base=32, groups=8, eta_init=0.5):
        super(VarNet, self).__init__()
        from step5_train_final import UNetGN
        self.K = int(K)
        self.refiners = nn.ModuleList(
            [UNetGN(in_ch=2, base=base, groups=groups) for _ in range(self.K)])
        self.etas = nn.Parameter(torch.full((self.K,), float(eta_init)))

    def forward(self, z0, y, mask):
        z = z0
        for k in range(self.K):
            z = self.refiners[k](z)                       # 精修（独立权重）
            zc = C.to_c(z)
            eta = F.softplus(self.etas[k]).view(-1)
            z = C.to_2ch(zc - eta * C.ifft2_t((C.fft2_t(zc) - y) * mask))  # DC
        return z

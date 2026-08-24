# -*- coding: utf-8 -*-
"""model_dccnn.py —— DCCNN 基线（双域级联：图像域 + k-空间域 + 数据一致性）

接口：forward(z0, y, mask) -> out_2ch
- 每个级联：图像域残差 CNN 精修 + k-空间域残差 CNN 精修 + k-空间数据一致性
- 经典 DC-CNN（Schlemper et al.）的双域思想在复值设置下的实现
"""
import torch
import torch.nn as nn

import step5_320_ceiling as C


class _Block(nn.Module):
    """3 层卷积残差块（图像域或 k-空间域共用）。"""

    def __init__(self, feats=64):
        super(_Block, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, feats, 3, padding=1), nn.InstanceNorm2d(feats), nn.ReLU(inplace=True),
            nn.Conv2d(feats, feats, 3, padding=1), nn.InstanceNorm2d(feats), nn.ReLU(inplace=True),
            nn.Conv2d(feats, 2, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class DCCNN(nn.Module):
    def __init__(self, K=5, feats=64):
        super(DCCNN, self).__init__()
        self.K = int(K)
        self.cascades = nn.ModuleList([
            nn.ModuleDict({"img": _Block(feats), "ksp": _Block(feats)})
            for _ in range(self.K)])

    def forward(self, z0, y, mask):
        z = z0
        mf = mask.float()
        for c in self.cascades:
            z = c["img"](z)                                 # 图像域精修
            k = C.to_2ch(C.fft2_t(C.to_c(z)))               # 图像 -> k-空间
            k = c["ksp"](k)                                 # k-空间域精修
            kc = C.to_c(k) * (1 - mf) + y * mf              # 数据一致性
            z = C.to_2ch(C.ifft2_t(kc))                     # k-空间 -> 图像
        return z

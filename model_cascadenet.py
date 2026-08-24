# -*- coding: utf-8 -*-
"""model_cascadenet.py —— 本文的有限深度实例化（CascadeNet，K 级独立权重）

接口：forward(z0, y, mask) -> out_2ch
- 每级 = 数据一致性步（可学习 softplus 步长）+ 独立权重的 GroupNorm U-Net
- 与论文实验（step5_train_final）完全同构，此处作为统一基线接口的封装
"""
import torch
import torch.nn as nn
import step5_320_ceiling as C


class CascadeNetBaseline(nn.Module):
    def __init__(self, K=4, base=64, groups=8, eta_init=0.5):
        super(CascadeNetBaseline, self).__init__()
        from step5_train_final import CascadeNet
        self.casc = CascadeNet(in_ch=2, base=base, K=int(K),
                               groups=groups, eta_init=eta_init)
        self.K = int(K)

    def forward(self, z0, y, mask):
        outs = self.casc(z0, y, mask, k_max=self.K)   # 复数 (B,H,W) 列表
        return C.to_2ch(outs[-1])                      # -> (B,2,H,W)

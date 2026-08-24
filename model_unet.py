# -*- coding: utf-8 -*-
"""model_unet.py —— U-Net 基线（直接重建）

接口：forward(z0, y, mask) -> out_2ch
- z0: (B,2,H,W) 零填充图像；y: (B,H,W) complex k-space；mask: (H,W) bool
- 标准 U-Net：输入零填充图像，直接输出重建（与官方 fastMRI U-Net 基线同构）
"""
import torch
import torch.nn as nn


class UnetBaseline(nn.Module):
    def __init__(self, base=64, groups=8):
        super(UnetBaseline, self).__init__()
        from step5_train_final import UNetGN
        self.net = UNetGN(in_ch=2, base=base, groups=groups)

    def forward(self, z0, y, mask):
        return self.net(z0)

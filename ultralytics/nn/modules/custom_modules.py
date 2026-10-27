# custom_modules.py
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

__all__ = ["GSConv", "MambaoutLayer"]

# ========================== 辅助函数 ==========================
def autopad(k, p=None):
    """自动计算padding以保持尺寸不变"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# ========================== GSConv ==========================
class GSConv(nn.Module):
    """
    GSConv: 混合标准卷积与深度可分离卷积，配合通道重排。
    c1: 输入通道，c2: 输出通道，k: 卷积核大小，s: 步长，g: 分组数，act: 是否使用激活函数
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        # 标准卷积分支（负责下采样）
        self.cv1 = Conv(c1, c_, k, s, autopad(k, p), g, act=act)
        # 深度可分离卷积分支（不改变尺寸）
        self.cv2 = Conv(c_, c_, 5, 1, 2, c_, act=act)  # groups=c_ 实现 depthwise

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        out = torch.cat([x1, x2], dim=1)
        # 通道重排：增强两组特征之间的信息交换
        b, c, h, w = out.shape
        out = out.view(b, 2, c // 2, h, w).transpose(1, 2).contiguous().view(b, c, h, w)
        return out

# ========================== MambaBlock ==========================
class MambaBlock(nn.Module):
    """
    简化的视觉Mamba块，用于建模长距离依赖。
    dim: 输入通道数，d_state: 状态维度，d_conv: 1D卷积核大小，expand: 隐藏层膨胀系数
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=1):
        super().__init__()
        hidden = int(dim * expand)
        self.in_proj = nn.Linear(dim, hidden * 2)
        self.conv1d = nn.Conv1d(hidden, hidden, d_conv, groups=hidden, padding=d_conv-1)
        self.x_proj = nn.Linear(hidden, d_state * 2)
        self.dt_proj = nn.Linear(d_state, hidden)
        self.out_proj = nn.Linear(hidden, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # (B, N, C)
        res = x
        x_proj = self.in_proj(x)
        x, z = x_proj.chunk(2, dim=-1)
        x = x.transpose(1, 2)                     # (B, hidden, N)
        x = self.conv1d(x)[..., :x.shape[-1]]     # 保持长度
        x = x.transpose(1, 2)                     # (B, N, hidden)
        x_act = self.act(x)
        x_ssm = self.x_proj(x_act)
        dt, B_ssm = x_ssm.chunk(2, dim=-1)
        dt = self.dt_proj(dt)
        x = x_act * dt
        x = self.out_proj(x) * z
        x = x + res
        return x.transpose(1, 2).reshape(B, C, H, W)

# ========================== MambaoutLayer ==========================
class MambaoutLayer(nn.Module):
    """
    全局依赖建模层，由多个MambaBlock堆叠而成。
    c1: 输入通道，c2: 输出通道，depth: MambaBlock数量
    """
    def __init__(self, c1, c2, depth=1):
        super().__init__()
        self.proj = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1)
        self.blocks = nn.ModuleList([MambaBlock(c2) for _ in range(depth)])

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return x
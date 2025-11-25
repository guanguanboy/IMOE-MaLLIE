import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeedForwardMoE(nn.Module):
    """Illumination-aware Mixture of Experts module.

    Args:
        dim: 输入/输出通道数
        expand: 隐藏层扩展倍数
        bias: 卷积是否使用偏置
        num_experts: 专家数量
        top_k: 若 >0，启用稀疏路由 (Top-K experts per position)
        temperature: softmax 温度系数
    """
    def __init__(self, dim, expand=2.0, bias=True,
                 num_experts: int = 3, top_k: int = 0, temperature: float = 1.0):
        super().__init__()
        assert num_experts >= 2, "num_experts must be >= 2"
        hidden_features = int(dim * expand)

        # 输入通道映射到隐藏通道
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        # 定义专家（这里用深度可分离卷积）
        kernel_cfg = [3, 5, 3]  # 前几个专家不同卷积核大小
        experts = []
        for i in range(num_experts):
            k = kernel_cfg[i] if i < len(kernel_cfg) else kernel_cfg[-1]
            pad = (k - 1) // 2
            experts.append(
                nn.Conv2d(hidden_features, hidden_features, kernel_size=k, padding=pad,
                          groups=hidden_features, bias=bias)
            )
        self.experts = nn.ModuleList(experts)
        self.num_experts = num_experts

        # illumination -> 路由logits
        self.illum_to_logits = nn.Conv2d(1, num_experts, kernel_size=3, padding=1, bias=bias)

        # 混合后的深度卷积
        self.dwconv4 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1,
                                 padding=1, groups=hidden_features, bias=bias)

        # 输出映射回原通道
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.top_k = int(top_k)
        self.temperature = float(temperature)

    def _apply_topk_mask(self, logits: torch.Tensor, k: int):
        """将非Top-K专家的logits置为 -inf"""
        if k <= 0 or k >= logits.size(1):
            return logits
        B, N, H, W = logits.shape
        flat = logits.view(B, N, -1)  # (B, N, HW)
        topk_vals, topk_idx = torch.topk(flat, k=k, dim=1)  # (B, k, HW)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        batch_idx = torch.arange(B, device=logits.device)[:, None, None]
        pos_idx = torch.arange(flat.size(2), device=logits.device)[None, None, :]
        for i in range(k):
            idx = topk_idx[:, i, :]
            mask[batch_idx.squeeze(1), idx, pos_idx.squeeze(0)] = True
        flat_masked = torch.full_like(flat, float('-inf'))
        flat_masked[mask] = flat[mask]
        return flat_masked.view(B, N, H, W)

    def forward(self, x_in: torch.Tensor, illum: torch.Tensor):
        """
        x_in: (B, C, H, W)   输入特征
        illum: (B, 1, H, W) 或 (B, C, H, W) 光照特征
        """
        # 映射到隐藏通道
        x = self.project_in(x_in)

        # 每个专家独立处理
        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(x))
        expert_stack = torch.stack(expert_outs, dim=1)  # (B, N, hidden, H, W)

        # 处理illum到单通道
        if illum.dim() == 2:
            raise ValueError("illum shape (B, HW) not supported directly, reshape before input")
        elif illum.dim() == 3:  # (B,H,W)
            illum = illum.unsqueeze(1)
        elif illum.dim() == 4:  # (B,C,H,W)
            if illum.shape[1] != 1:
                illum = illum.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"illum must have 3/4 dims, got {illum.dim()}")

        # 光照图 -> 路由权重
        logits = self.illum_to_logits(illum) / (self.temperature + 1e-8)
        if self.top_k > 0 and self.top_k < self.num_experts:
            logits = self._apply_topk_mask(logits, self.top_k)
        alphas = F.softmax(logits, dim=1)  # (B, N, H, W)

        # 融合专家输出
        fused = (expert_stack * alphas.unsqueeze(2)).sum(dim=1)  # (B, hidden, H, W)

        # 残差融合
        x = x + fused * x

        # 输出映射
        x = F.gelu(self.dwconv4(x))
        x = self.project_out(x)
        return x


if __name__ == "__main__":
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)
    illum_multi = torch.randn(B, 8, H, W)  # 多通道光照特征
    illum_single = torch.randn(B, 1, H, W) # 单通道光照图

    model = FeedForwardMoE(dim=C, num_experts=3, top_k=2)
    out_multi = model(x, illum_multi)
    out_single = model(x, illum_single)

    print(f"Input shape: {x.shape}")
    print(f"Output shape (multi-channel illum): {out_multi.shape}")
    print(f"Output shape (single-channel illum): {out_single.shape}")

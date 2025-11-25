import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
import sys
import os
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
#from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import warnings

class DepthwiseExpert(nn.Module):
    """A simple depthwise conv expert: depthwise conv + pointwise conv (optional) + BN + activation."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, use_pointwise: bool = True):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding,
                            dilation=dilation, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.use_pw = use_pointwise
        if use_pointwise:
            self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.act(x)
        if self.use_pw:
            x = self.pw(x)
            x = self.bn2(x)
            x = self.act(x)
        return x


class IlluminationAwareMoE(nn.Module):
    """
    Illumination-aware MoE module.

    Args:
        in_channels: number of channels in input feature.
        experts: list of nn.Module experts. If None, will create default depthwise experts with kernels [3,5,7,...].
        gating_hidden: hidden dimension for gating MLP.
        mode: "global" or "spatial" gating.
        top_k: if >0 uses top-k sparse routing (keeps top_k experts per sample), otherwise dense softmax routing.
        temperature: softmax temperature for gating (useful for controlling sharpness).
    Inputs:
        x: feature tensor, shape (B, C, H, W)
        illum: illumination prior, shape (B, 1, H, W) or (B, H, W) or (B, 1, 1, 1)
    Output:
        aggregated feature, same shape as x
    """
    def __init__(self,
                 in_channels: int,
                 experts: Optional[List[nn.Module]] = None,
                 gating_hidden: int = 128,
                 mode: str = "global",
                 top_k: int = 0,
                 temperature: float = 1.0):
        super().__init__()
        assert mode in ("global", "spatial")
        self.in_channels = in_channels
        self.mode = mode
        self.top_k = int(top_k)
        self.temperature = temperature

        # default experts if not provided
        if experts is None:
            kernels = [3, 5, 7]  # you can add more or include dilations
            experts = [DepthwiseExpert(in_channels, k) for k in kernels]
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        # gating network
        if mode == "global":
            # input: global pooled [I_prior; F] -> MLP -> logits per expert
            # combine illumination prior (global pooled) and feature global pooled
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_channels + 1, gating_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gating_hidden, self.num_experts)
            )
        else:  # spatial
            # spatial gating: produce per-expert maps (B, N, H, W) logits
            # we fuse illum and feature via conv and predict N channels
            self.gate_conv = nn.Sequential(
                nn.Conv2d(in_channels + 1, gating_hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(gating_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(gating_hidden, self.num_experts, kernel_size=1)
            )

    def _topk_mask(self, logits: torch.Tensor, k: int):
        """
        logits: (B, N) or (B, N, H, W)
        returns mask same shape with 1 for selected experts, 0 otherwise
        """
        if logits.dim() == 2:  # (B, N)
            topk = torch.topk(logits, k=k, dim=1)[1]  # indices (B, k)
            mask = torch.zeros_like(logits, dtype=logits.dtype, device=logits.device)
            batch_idx = torch.arange(logits.size(0), device=logits.device).unsqueeze(1).expand(-1, k)
            mask[batch_idx, topk] = 1.0
            return mask
        elif logits.dim() == 4:  # (B, N, H, W)
            B, N, H, W = logits.shape
            flat = logits.view(B, N, -1)  # (B, N, HW)
            topk = torch.topk(flat, k=k, dim=1)[1]  # (B, k, HW)
            mask_flat = torch.zeros_like(flat)
            batch_idx = torch.arange(B, device=logits.device)[:, None, None]
            # advanced indexing to set 1s
            for i in range(k):
                idx = topk[:, i, :]  # (B, HW)
                # create coords to scatter
                batch_coords = batch_idx.expand(-1, 1, idx.size(1)).squeeze(1)  # (B, HW)
                mask_flat[batch_coords, idx] = 1.0
            mask = mask_flat.view(B, N, H, W)
            return mask
        else:
            raise ValueError("Unsupported logits dim for topk_mask")

    def forward(self, x: torch.Tensor, illum: torch.Tensor):
        """
        x: (B, C, H, W)
        illum: (B,1,H,W) or (B,H,W) or broadcastable
        """
        B, C, H, W = x.shape
        # normalize/reshape illum to (B,1,H,W)
        if illum.dim() == 2:  # (B, H, W)
            illum = illum.unsqueeze(1)
        elif illum.dim() == 3:  # (B, 1, H, W) expected, but allow (B, H, W)
            if illum.shape[1] != 1:
                illum = illum.unsqueeze(1)
        elif illum.dim() == 4:
            if illum.shape[1] != 1:
                # allow multi-channel illum but compress to 1-channel by mean
                illum = illum.mean(dim=1, keepdim=True)
        else:
            raise ValueError("illum must be dim 2/3/4")

        # Optionally downsample illum to match x spatial dims if needed (here assume same)
        # Experts outputs
        expert_outs = []
        for e in self.experts:
            expert_outs.append(e(x))  # each (B, C, H, W)
        # stack to (B, N, C, H, W)
        expert_stack = torch.stack(expert_outs, dim=1)

        if self.mode == "global":
            # Build gating input: global pooled feature + illum (pooled to scalar)
            feat_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)              # (B, C)
            illum_pool = F.adaptive_avg_pool2d(illum, 1).view(B, 1)        # (B, 1)
            gate_in = torch.cat([feat_pool, illum_pool], dim=1)           # (B, C+1)
            logits = self.gate_mlp(gate_in)                               # (B, N)
            logits = logits / (self.temperature + 1e-8)

            if self.top_k > 0 and self.top_k < self.num_experts:
                # sparse routing: mask out not top-k (use logits as scores)
                mask = self._topk_mask(logits, k=self.top_k)              # (B, N)
                # set non-topk logits to -inf before softmax to zero them after
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(mask == 0, neg_inf)
            alphas = F.softmax(logits, dim=1)                             # (B, N)

            # expand alphas to (B, N, 1, 1, 1) to weight expert_stack
            alphas = alphas.view(B, self.num_experts, 1, 1, 1)
            out = (expert_stack * alphas).sum(dim=1)                      # (B, C, H, W)
            return out

        else:  # spatial gating
            # fuse x and illum along channel dim
            fuse = torch.cat([x, illum], dim=1)                           # (B, C+1, H, W)
            logits_map = self.gate_conv(fuse)                             # (B, N, H, W)
            logits_map = logits_map / (self.temperature + 1e-8)

            if self.top_k > 0 and self.top_k < self.num_experts:
                mask = self._topk_mask(logits_map, k=self.top_k)         # (B, N, H, W)
                neg_inf = torch.finfo(logits_map.dtype).min
                logits_map = logits_map.masked_fill(mask == 0, neg_inf)
            alphas_map = F.softmax(logits_map, dim=1)                     # (B, N, H, W)

            # expand to (B, N, C, H, W)
            alphas_map = alphas_map.unsqueeze(2)                          # (B, N, 1, H, W)
            out = (expert_stack * alphas_map).sum(dim=1)                  # (B, C, H, W)
            return out

class IlluminationAwareMoENew(nn.Module):
    def __init__(self,
                 in_channels: int,
                 illum_channels: int,
                 experts: Optional[List[nn.Module]] = None,
                 gating_hidden: int = 128,
                 mode: str = "global",
                 top_k: int = 0,
                 temperature: float = 1.0):
        super().__init__()
        assert mode in ("global", "spatial")
        self.in_channels = in_channels
        self.illum_channels = illum_channels
        self.mode = mode
        self.top_k = int(top_k)
        self.temperature = temperature

        # default experts if not provided
        if experts is None:
            kernels = [3, 5, 7]
            experts = [DepthwiseExpert(in_channels, k) for k in kernels]
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

        # gating network
        if mode == "global":
            # 输入是 global pooled feature + illum pooled
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_channels + illum_channels, gating_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gating_hidden, self.num_experts)
            )
        else:
            # 空间 gating: 拼接特征和 illum
            self.gate_conv = nn.Sequential(
                nn.Conv2d(in_channels + illum_channels, gating_hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(gating_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(gating_hidden, self.num_experts, kernel_size=1)
            )

    def _topk_mask(self, logits: torch.Tensor, k: int):
        if logits.dim() == 2:
            topk = torch.topk(logits, k=k, dim=1)[1]
            mask = torch.zeros_like(logits, dtype=logits.dtype, device=logits.device)
            batch_idx = torch.arange(logits.size(0), device=logits.device).unsqueeze(1).expand(-1, k)
            mask[batch_idx, topk] = 1.0
            return mask
        elif logits.dim() == 4:
            B, N, H, W = logits.shape
            flat = logits.view(B, N, -1)
            topk = torch.topk(flat, k=k, dim=1)[1]
            mask_flat = torch.zeros_like(flat)
            batch_idx = torch.arange(B, device=logits.device)[:, None, None]
            for i in range(k):
                idx = topk[:, i, :]
                batch_coords = batch_idx.expand(-1, 1, idx.size(1)).squeeze(1)
                mask_flat[batch_coords, idx] = 1.0
            mask = mask_flat.view(B, N, H, W)
            return mask
        else:
            raise ValueError("Unsupported logits dim for topk_mask")

    def forward(self, x: torch.Tensor, illum: torch.Tensor):
        B, C, H, W = x.shape
        assert illum.shape[0] == B and illum.shape[2:] == (H, W), \
            f"illum shape {illum.shape} must match spatial size {(H, W)}"

        expert_outs = [e(x) for e in self.experts]
        expert_stack = torch.stack(expert_outs, dim=1)  # (B, N, C, H, W)

        if self.mode == "global":
            feat_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)                     # (B, C)
            illum_pool = F.adaptive_avg_pool2d(illum, 1).view(B, self.illum_channels)  # (B, illum_channels)
            gate_in = torch.cat([feat_pool, illum_pool], dim=1)                    # (B, C+illum_channels)
            logits = self.gate_mlp(gate_in) / (self.temperature + 1e-8)

            if self.top_k > 0 and self.top_k < self.num_experts:
                mask = self._topk_mask(logits, k=self.top_k)
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(mask == 0, neg_inf)
            alphas = F.softmax(logits, dim=1).view(B, self.num_experts, 1, 1, 1)
            out = (expert_stack * alphas).sum(dim=1)
            return out

        else:
            fuse = torch.cat([x, illum], dim=1)                                    # (B, C+illum_channels, H, W)
            logits_map = self.gate_conv(fuse) / (self.temperature + 1e-8)

            if self.top_k > 0 and self.top_k < self.num_experts:
                mask = self._topk_mask(logits_map, k=self.top_k)
                neg_inf = torch.finfo(logits_map.dtype).min
                logits_map = logits_map.masked_fill(mask == 0, neg_inf)
            alphas_map = F.softmax(logits_map, dim=1).unsqueeze(2)                 # (B, N, 1, H, W)
            out = (expert_stack * alphas_map).sum(dim=1)
            return out


class FeedForward(nn.Module): ## Implicit Retinex-Aware
    def __init__(self, dim, expand, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*expand)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, 2, kernel_size=3, padding=1, bias=bias)
        self.dwconv4 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.act = nn.Sigmoid()

    def forward(self, x_in,illum):
        x = self.project_in(x_in)
        attn1 = self.dwconv(x) 
        attn2 = self.dwconv2(attn1)
        illum1,illum2 = self.dwconv3(illum).chunk(2, dim=1)
        attn = attn1*self.act(illum1)+attn2*self.act(illum2)
        x = x + attn*x
        x = F.gelu(self.dwconv4(x))
        x = self.project_out(x)
        return x
    
# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 64, 64, 64
    x = torch.randn(B, C, H, W)
    # e.g., illum prior: per-pixel single-channel map predicted by a Retinex branch
    illum = torch.rand(B, C, H, W)

    # Default: 3 depthwise experts, global softmax gating
    module = IlluminationAwareMoENew(in_channels=C, illum_channels=C, mode="global", top_k=0)
    out = module(x, illum)
    print("out shape (global):", out.shape)

    # Spatial gating + top-1 sparsity (only one expert active per pixel)
    module_sp = IlluminationAwareMoENew(in_channels=C, illum_channels=C, mode="spatial", top_k=1)
    out_sp = module_sp(x, illum)
    print("out shape (spatial top-1):", out_sp.shape)

    # Custom experts
    custom_experts = [
        DepthwiseExpert(C, kernel_size=3),
        DepthwiseExpert(C, kernel_size=5, dilation=1),
        DepthwiseExpert(C, kernel_size=3, dilation=2),
    ]
    module_custom = IlluminationAwareMoENew(in_channels=C, illum_channels=C, experts=custom_experts, mode="global", top_k=2)
    out_custom = module_custom(x, illum)
    print("out shape (custom):", out_custom.shape)


    illum_for_ffn = torch.rand(B, C, H, W)

    ffn = FeedForward(64, 1, False)

    out_ffn = ffn(x, illum_for_ffn)
    print("out shape (spatial top-1):", out_ffn.shape)


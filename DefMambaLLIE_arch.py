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
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import warnings

try:
    from Dwconv.dwconv_layer import DepthwiseFunction
except:
    DepthwiseFunction = None

from .basicsr.archs.utils import SelectiveScan,\
    flops_selective_scan_ref,print_jit_input_names, Mlp,\
    x_selective_scan, DeformableLayer, DeformableLayerReverse

class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DSSM(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        forward_type="v2",
        # ======================
        stage = 0,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = d_expand
        self.d_inner = d_inner
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.stage = stage

        self.K = 3
        self.K2 = self.K

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            stride = 1
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                stride=stride,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.DS = DeformableLayer(index=stage, embed_dim=d_inner, debug=False)
        self.DR = DeformableLayerReverse()

        # other kwargs =======================================
        self.kwargs = kwargs
        if simple_init:
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.K2 * d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        self.debug = False
        self.outnorm = None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A) 
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D) 
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, nrows=-1, channel_first=False):
        nrows = 1
        if self.debug: debug_rec = []
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = x_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, self.outnorm,
            nrows=nrows, delta_softplus=True, force_fp32=self.training, stage=self.stage, DS=self.DS, DR=self.DR,
            **self.kwargs,
        )
        x, debug_rec = x[0], x[1]
        if self.debug:
            return x, debug_rec
        return  x

    def forward(self, x: torch.Tensor,illum: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        b, h, w, d = x.shape
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out



class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            drop_rate: float = 0.1,
            d_state: int = 16,
            expand: float = 2.,
            img_size: int = 224,
            patch_size: int = 4,
            embed_dim: int = 64,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)

        self.ss2d = DSSM(d_model=hidden_dim, d_state=d_state,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.ffn = FeedForwardMoE(hidden_dim, expand, bias=True)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim, norm_layer=None)

        self.conv2d = nn.Conv2d(int(hidden_dim*expand), int(hidden_dim*expand), kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False)

    def forward(self, inputs):
        input, illum = inputs
        input_size = (input.shape[2], input.shape[3])
        input = self.patch_embed(input) 
        input = self.pos_drop(input)
        illum = F.gelu(self.conv2d(illum))
        B, L, C = input.shape
        input = input.view(B, *input_size, C).contiguous()  # [B,H,W,C]
        x = input + self.drop_path(self.ss2d(self.ln_1(input),illum))
        x = x.view(B, -1, C).contiguous()
        x = self.patch_unembed(x, input_size) + self.ffn(self.patch_unembed(self.ln_2(x), input_size),illum)
        return (x,illum)

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

@ARCH_REGISTRY.register()
class DefMambaLLIE(nn.Module):
    def __init__(self, nf=32,
                img_size=128,
                patch_size=1,
                embed_dim=32,
                depths=(1,2,2,2,2,2),  
                d_state = 32,
                mlp_ratio=2.,
                norm_layer=nn.LayerNorm,
                num_layer=3):
        super(DefMambaLLIE, self).__init__()

        self.nf = nf
        self.depths = depths

        self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=False)
        self.conv_first_1_fea = nn.Conv2d(5,int(nf*mlp_ratio),3,1,1)
        self.VSSB_1 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size,patch_size=patch_size,embed_dim=embed_dim) for i in range(self.depths[0])])
        
        self.conv_first_2 = nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)
        self.conv_first_2_fea = nn.Conv2d(5,int(nf*2*mlp_ratio),3,1,1)
        self.VSSB_2 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf*2,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size//2,patch_size=patch_size,embed_dim=embed_dim*2) for i in range(self.depths[1])])
        
        self.conv_first_3 = nn.Conv2d(nf*2, nf * 4, 4, 2, 1, bias=False)
        self.conv_first_3_fea = nn.Conv2d(5,int(nf*4*mlp_ratio),3,1,1)
        self.VSSB_3 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf*4,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size//4,patch_size=patch_size,embed_dim=embed_dim*4) for i in range(self.depths[2])])

        self.conv_first_4 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False)
        self.conv_first_4_fea = nn.Conv2d(5,int(nf*4*mlp_ratio),3,1,1)
        self.VSSB_4 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf*4,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size//4,patch_size=patch_size,embed_dim=embed_dim*4) for i in range(self.depths[3])])

        self.upconv1 = nn.ConvTranspose2d(nf*4, nf*4 // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0)
        self.conv_first_5 = nn.Conv2d(nf*4, nf*4 // 2, 3, 1, 1, bias=False)
        self.conv_first_5_fea = nn.Conv2d(5,int(nf*2*mlp_ratio),3,1,1)
        self.VSSB_5 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf*2,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size//2,patch_size=patch_size,embed_dim=embed_dim*2) for i in range(self.depths[4])])
        
        self.upconv2 = nn.ConvTranspose2d(nf*2, nf*2 // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0)
        self.conv_first_6 = nn.Conv2d(nf*2, nf*2 // 2, 3, 1, 1, bias=False)
        self.conv_first_6_fea = nn.Conv2d(5,int(nf*mlp_ratio),3,1,1)
        self.VSSB_6 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size,patch_size=patch_size,embed_dim=embed_dim) for i in range(self.depths[5])])

        self.out_embed = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, x_in):

        x_max = torch.max(x_in, dim=1, keepdim=True)[0]
        x_mean = torch.mean(x_in, dim=1, keepdim=True)
        x_in_cat = torch.cat((x_in,x_max,x_mean), dim=1)

        x_2 = F.avg_pool2d(x_in_cat, kernel_size=2, stride=2)
        x_4 = F.avg_pool2d(x_in_cat, kernel_size=4, stride=4)
        
        x_conv_1 = self.conv_first_1(x_in)
        illum_conv_1 = self.conv_first_1_fea(x_in_cat)
        vssb_fea_1 = self.VSSB_1((x_conv_1,illum_conv_1))[0]
        
        x_conv_2 = self.conv_first_2(vssb_fea_1)
        illum_conv_2 = self.conv_first_2_fea(x_2)
        vssb_fea_2 = self.VSSB_2((x_conv_2,illum_conv_2))[0]

        x_conv_3 = self.conv_first_3(vssb_fea_2)
        illum_conv_3 = self.conv_first_3_fea(x_4)
        vssb_fea_3 = self.VSSB_3((x_conv_3,illum_conv_3))[0]

        x_conv_4 = self.conv_first_4(vssb_fea_3)
        illum_conv_4 = self.conv_first_4_fea(x_4)
        vssb_fea_4 = self.VSSB_4((x_conv_4,illum_conv_4))[0]

        up_feat_1 = self.upconv1(vssb_fea_4)
        x_cat_1 = torch.cat([up_feat_1, vssb_fea_2], dim=1)
        vssb_fea_5 = self.conv_first_5(x_cat_1)
        illum_conv_5 = self.conv_first_5_fea(x_2)
        vssb_fea_5 = self.VSSB_5((vssb_fea_5,illum_conv_5))[0]

        up_feat_2 = self.upconv2(vssb_fea_5)
        x_cat_2 = torch.cat([up_feat_2, vssb_fea_1], dim=1)
        vssb_fea_6 = self.conv_first_6(x_cat_2)
        illum_conv_6 = self.conv_first_6_fea(x_in_cat)
        vssb_fea_6 = self.VSSB_6((vssb_fea_6,illum_conv_6))[0]

        out = self.out_embed(vssb_fea_6) + x_in

        return out


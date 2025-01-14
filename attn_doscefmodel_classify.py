from nwtk.mlp import Mlp
from nwtk.drop import DropPath
from nwtk.downsample import DownSample1d

import torch
from torch import nn
import torch.nn.functional as F

#input shape [B,C,N] .torch.float32, default for 1d sequence tensor.
#mask shape [B,1,N], fill with 0,1. torch.float32

def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

def attn_mask_pool1d(x, kernel_size, stride):
    B, C, N = x.shape
    pad_right = (N // stride - 1) * stride + kernel_size - N
    x = F.pad(x, (0, pad_right), "constant", 0)
    x = F.max_pool1d(x, kernel_size, stride = stride)
    return x

#attention apply on layer
class SpatialAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=True,
                 use_fused_attn=False
        ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        #self.softmax = nn.Softmax(dim=-1)
        #self.softmax = softmax_one

    def forward(self, x, attn_mask):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            scores = (q @ k.transpose(-2, -1))
            #if attn_mask is not None:
            if attn_mask.any():
                #attn_mask = attn_mask.reshape(B_, 1, self.num_heads, C // self.num_heads).transpose(1, 2).expand(B_, self.num_heads, C // self.num_heads, C // self.num_heads)
                attn_mask = attn_mask.unsqueeze(-1).expand(B_, self.num_heads, N, N)
                scores = scores.masked_fill_(attn_mask == 0, -1e9)
            scores = softmax_one(scores, dim =-1)
            x = scores @ v

        x = x.permute(0, 3, 1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x
        
class LayerScale1d(nn.Module):
    def __init__(self, in_features, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(in_features))

    def forward(self, x):
        gamma = self.gamma.view(1, 1, -1)
        return x.mul_(gamma) if self.inplace else x * gamma

class SpatialAttentionBlock(nn.Module):
    def __init__(self, 
                 dim,
                 mlp_ratio=4.,
                 num_heads=8,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 drop_path=0.,
                 layer_scale_init_value=None, #1e-5
                 use_attn=True,
        ):
        super().__init__()
        #self.attn = 
        self.attn = SpatialAttention(dim = dim, num_heads = num_heads) if use_attn else None
        self.ls1 = LayerScale1d(dim,
                                layer_scale_init_value) if layer_scale_init_value is not None else nn.Identity()
        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer,
            )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls2 = LayerScale1d(dim,
                                layer_scale_init_value) if layer_scale_init_value is not None else nn.Identity()

    def forward(self, x):
        if isinstance(x, tuple):
            x, attn_mask = x
        else:
            attn_mask = False

        if self.attn is not None:
            x = x.transpose(1,2)
            x = self.attn(x, attn_mask)
            x = x + self.drop_path1(self.ls1(x))
            x = x.transpose(2,1)
        x = x.transpose(1,2)
        x = x + self.drop_path2(self.ls2(self.mlp(x)))
        x = x.transpose(2,1)
        return x, attn_mask
    
class AttentionStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            depth=4,
            num_heads=8,
            block_use_attn=True,
            num_vit=1,
            mlp_ratio=4.,
            drop_path=0.,
            layer_scale_init_value=1e-5,
            downsample = True,
            down_kernel_size=3,
            down_stride=2,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.down_kernel_size = down_kernel_size
        self.down_stride = down_stride
        if downsample:
            self.ds = DownSample1d(in_chs, out_chs, kernel_size=down_kernel_size, stride = down_stride)
        else:
            self.ds = nn.Identity()

        self.downsample = downsample
        
        blocks = []
        for block_idx in range(depth):
            remain_idx = depth - num_vit - 1
            b = SpatialAttentionBlock(
                dim = out_chs,
                mlp_ratio=mlp_ratio,
                num_heads=num_heads,
                use_attn=block_use_attn and block_idx >= remain_idx,
                drop_path=drop_path,
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            blocks += [b]
            
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        if isinstance(x, tuple):
            x, attn_mask = x
            if self.downsample:
                attn_mask = attn_mask_pool1d(attn_mask , kernel_size = self.down_kernel_size, stride = self.down_stride)
            #attn_mask = attn_mask == 0
        else:
            attn_mask = False
        x = self.ds(x)
        x = self.blocks((x,attn_mask))
        return x

class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm1d,
            act_layer=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.pad_right = kernel_size - stride
        self.bn = norm_layer(out_channels)
        self.act = act_layer()
        self.pad_right = kernel_size - stride
        
    def forward(self, x):
        
        x = F.pad(x, (0, self.pad_right), "constant", 0.)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
        
class Stem(nn.Sequential):
    def __init__(self, in_chs, out_chs, act_layer=nn.GELU, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.conv1 = ConvNormAct(
            in_chs, out_chs // 2, kernel_size=3, stride=2,
            norm_layer=norm_layer, act_layer=act_layer
        )
        self.conv2 = ConvNormAct(
            out_chs // 2, out_chs, kernel_size=3, stride=2,
            norm_layer=norm_layer, act_layer=act_layer
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_features,
        num_classes,
        drop_rate= 0.,
    ):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(in_features, num_classes, bias=True)
        self.drop = nn.Dropout(drop_rate)
        
    def forward(self, x):
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        return self.flatten(x)
        
class AttentionDos(nn.Module):
    def __init__(
        self,
        in_chs,
        #out_chs,
        depths=(3, 3),
        embed_dims=(96, 192),
        num_heads=(6, 6),
        num_classes=2,
        head_norm_first=True,
        norm_layer=nn.BatchNorm1d
        ):
        super().__init__()
        num_stages = len(depths)
        num_features = embed_dims[-1]

        self.stem = Stem(in_chs, embed_dims[0])
        stages = []
        in_chs = embed_dims[0]
        for stage_idx in range(num_stages):
            out_chs = embed_dims[stage_idx]
            stage = AttentionStage(  
                in_chs,
                out_chs,
                #depth=depths[stage_idx],
                num_heads=num_heads[stage_idx],
                block_use_attn=True,
                num_vit=1,
                mlp_ratio=4.,
                drop_path=0.2,
                layer_scale_init_value=1e-5,
                downsample = stage_idx > 0,
                down_kernel_size=3,
                down_stride=2,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
            )
            in_chs = out_chs
            stages.append(stage)
        self.stages = nn.Sequential(*stages)
        
        self.norm_pre = norm_layer(num_features) if head_norm_first else nn.Identity()
        self.head = ClassifierHead(
                    num_features,
                    num_classes,
                    #pool_type=global_pool,
                    drop_rate=0.,
                    )
    def forward_features(self, x, attn_mask):
        x = self.stem(x)
        attn_mask = attn_mask_pool1d(attn_mask , kernel_size = 3, stride = 4)
        x, _ = self.stages((x, attn_mask))
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)
    
    def forward(self, x, attn_mask):
        x = self.forward_features(x, attn_mask)
        x = self.forward_head(x)
        return x
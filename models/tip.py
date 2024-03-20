import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_, lecun_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model

class Mlp(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
            
    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLPMixerLayer(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 drop_path):

        super().__init__()

        patch_mix_dims = int(patch_expansion * embed_dims)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Linear(patch_mix_dims, num_patches),
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Linear(channel_mix_dims, embed_dims),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.drop_path(self.patch_mixer(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.channel_mixer(self.norm2(x)))
        return x

class TIP_Attention(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads,
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        ## define kernel
        self.ks = 3
        self.kernel = nn.Parameter(torch.randn(self.ks*self.ks, dim) * 0.02)
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.mlpmixer = MLPMixerLayer(num_patches=self.ks*self.ks,
                                      embed_dims=dim,
                                      patch_expansion=2,
                                      channel_expansion=2,
                                      drop_path=drop_path)

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)

    def cross_attention(self, q, k, v):
        B, num_heads, N, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) #[B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        w = self.kernel.expand(B, -1, -1)
        w_m = self.ln(w).reshape(
            B, self.ks*self.ks, self.num_heads, C//self.num_heads).transpose(1, 2).contiguous()
        kv = self.kv(x)
        k, v = kv.view(
            B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        w = w + self.cross_attention(w_m, k, v) # [B, N, C]
        w = self.mlpmixer(w)
        w = w.transpose(1, 2).reshape(B*C, 1, self.ks, self.ks)
        x = self.proj_q(x).transpose(1, 2).reshape(1, B*C, H, W)
        ##
        x = F.conv2d(x, w, stride=1, padding=(self.ks-1)//2, groups=B*C)
        x = x.reshape(B, C, H, W).flatten(2).transpose(1, 2)
        return x

class Convolution(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads,
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim)
        self.proj = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(self.dwconv(x))
        x = x.flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads,
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 kernel_size=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TIP_Attention(
            dim, 
            num_heads=num_heads,
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop,
            drop_path=drop_path)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
  
class ConvStem(nn.Module):
    def __init__(self,  
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        stem_dim = embed_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim, eps=1e-5),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      stride=1, padding=1, groups=stem_dim, bias=False),
            nn.BatchNorm2d(stem_dim, eps=1e-5),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      stride=1, padding=1, groups=stem_dim, bias=False),
            nn.BatchNorm2d(stem_dim, eps=1e-5),
            nn.GELU())
        
        self.patch_embed = nn.Conv2d(stem_dim, embed_dim, kernel_size=3,
                                     stride=2, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(self.stem(x))
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)
    
class Patch_Embedding(nn.Module):
    def __init__(self, dim, out_dim, kernel_size=3):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=dim,
                              out_channels=out_dim,
                              kernel_size=kernel_size,
                              stride=2,
                              padding=(kernel_size-1)//2)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, (H, W)

class ECCV(nn.Module):

    def __init__(self, 
                 img_size=224, 
                 in_chans=3,
                 num_classes=1000,
                 patch_sizes=[4, 2, 2, 2],
                 embed_dims=[32, 64, 128, 256], 
                 depths=[2, 2, 6, 6],
                 num_heads=[2, 2, 4, 8],
                 mlp_ratios=[8, 8, 4, 4],
                 kernel_sizes=[3, 3, 3, 3],
                 neck_dim=1280, 
                 qkv_bias=True,
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=None,
                 act_layer=None,
                 weight_init='',
                 pretrained=None,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.mlp_ratios = mlp_ratios
        self.patch_sizes = patch_sizes
        self.depths = depths
        self.num_heads = num_heads
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # stage 1
        self.patch_embed1 = ConvStem(patch_size=patch_sizes[0], 
                                     in_chans=in_chans, 
                                     embed_dim=embed_dims[0])
        # stage 2
        self.patch_embed2 = Patch_Embedding(dim=embed_dims[0],
                                          out_dim=embed_dims[1],
                                          kernel_size=3)
        # stage 3
        self.patch_embed3 = Patch_Embedding(dim=embed_dims[1],
                                          out_dim=embed_dims[2],
                                          kernel_size=3)
        # stage 4
        self.patch_embed4 = Patch_Embedding(dim=embed_dims[2],
                                          out_dim=embed_dims[3],
                                          kernel_size=3)

        total_depth = sum(self.depths) # all layers
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        cur = 0
        self.stage1 = nn.ModuleList([
            Block(dim=embed_dims[0],
                  num_heads=num_heads[0],
                  mlp_ratio=mlp_ratios[0],
                  qkv_bias=qkv_bias,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_layer=norm_layer,
                  act_layer=act_layer,
                  kernel_size=kernel_sizes[0])
            for i in range(self.depths[0])
        ])

        cur += self.depths[0]
        self.stage2 = nn.ModuleList([
            Block(dim=embed_dims[1],
                  num_heads=num_heads[1],
                  mlp_ratio=mlp_ratios[1],
                  qkv_bias=qkv_bias,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_layer=norm_layer,
                  act_layer=act_layer,
                  kernel_size=kernel_sizes[1])
            for i in range(self.depths[1])
        ])

        cur += self.depths[1]
        self.stage3 = nn.ModuleList([
            Block(dim=embed_dims[2],
                  num_heads=num_heads[2],
                  mlp_ratio=mlp_ratios[2],
                  qkv_bias=qkv_bias,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_layer=norm_layer,
                  act_layer=act_layer,
                  kernel_size=kernel_sizes[2])
            for i in range(self.depths[2])
        ])
        
        cur += self.depths[2]
        self.stage4 = nn.ModuleList([
            Block(dim=embed_dims[3],
                  num_heads=num_heads[3],
                  mlp_ratio=mlp_ratios[3],
                  qkv_bias=qkv_bias,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i],
                  norm_layer=norm_layer,
                  act_layer=act_layer,
                  kernel_size=kernel_sizes[3])
            for i in range(self.depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.neck = nn.Sequential(
            nn.Linear(embed_dims[3], neck_dim),
            nn.LayerNorm(neck_dim),
            nn.GELU())

        self.head = nn.Linear(neck_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            #trunc_normal_(self.global_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'global_token', '[g]relative_position_bias_table'}

    def forward(self, x):
        B = x.shape[0]
        x, (H, W) = self.patch_embed1(x)
        
        # stage 1
        for block in self.stage1:
            x = block(x, H, W)
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # stage 2
        x, (H, W) = self.patch_embed2(x)
        for block in self.stage2:
            x = block(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # stage 3
        x, (H, W) = self.patch_embed3(x)
        for block in self.stage3:
            x = block(x, H, W)
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # stage 4
        x, (H, W) = self.patch_embed4(x)
        for block in self.stage4:
            x = block(x, H, W)
        x = self.norm4(x)

        # neck part
        x = self.neck(x)
        x = self.head(x.mean(dim=1))
        return x

def _init_vit_weights(
        module: nn.Module, 
        name: str = '', 
        head_bias: float = 0., 
        jax_impl: bool = False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': 0.90, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs}

default_cfgs = {
    'bvit_t': _cfg(crop_pct=0.9),
    'bvit_s': _cfg(crop_pct=0.9),
    'bvit_b': _cfg(crop_pct=0.9)}

@register_model
def tip_b0(pretrained=False, **kwargs):
    model_kwargs = dict(embed_dims=[24, 48, 96, 192], depths=[2, 2, 6, 2],
                        num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], **kwargs)
    model = ECCV(**model_kwargs)
    model.default_cfg = default_cfgs['bvit_t']
    return model

@register_model
def tip_b1(pretrained=False, **kwargs):
    model_kwargs = dict(embed_dims=[32, 64, 128, 256], depths=[2, 2, 6, 6],
                        num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], **kwargs)
    model = ECCV(**model_kwargs)
    model.default_cfg = default_cfgs['bvit_s']
    return model

# -----------------------------------------------------------------------
# Agent Attention: On the Integration of Softmax and Linear Attention
# Modified by Dongchen Han
# -----------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = [
    'agent_pvt_tiny', 'agent_pvt_small', 'agent_pvt_medium', 'agent_pvt_large'
]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

'''
def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, agent_num=49, **kwargs)：这是AgentAttention类的构造函数。它接受一些参数，例如dim（表示输入特征的维度）、num_patches（表示输入图像被划分为的图块数）、num_heads（注意力头的数量）等等。还有一些其他的可选参数，例如qkv_bias（用于控制是否添加偏置项）等等。

assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."：这是一个断言语句，用于确保dim能够被num_heads整除。

self.dim = dim：将dim赋值给模块内部的变量self.dim。

self.num_patches = num_patches：将num_patches赋值给模块内部的变量self.num_patches。

window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))：计算window_size，即窗口的大小。这里假设num_patches是一个平方数，然后将其开方取整。

self.window_size = window_size：将window_size赋值给模块内部的变量self.window_size。

self.num_heads = num_heads：将num_heads赋值给模块内部的变量self.num_heads。

head_dim = dim // num_heads：计算每个注意力头的维度。

self.scale = head_dim ** -0.5：计算一个缩放因子，用于调整注意力头的输出。

self.q = nn.Linear(dim, dim, bias=qkv_bias)：定义一个线性变换操作，将输入特征映射到查询向量q。

self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)：定义一个线性变换操作，将输入特征映射到键向量k和值向量v。

self.attn_drop = nn.Dropout(attn_drop)：定义一个丢弃层，用于在注意力计算之前对注意力权重进行丢弃操作。

self.proj = nn.Linear(dim, dim)：定义一个线性变换操作，用于将注意力输出映射为最后的特征表示。

self.proj_drop = nn.Dropout(proj_drop)：定义一个丢弃层，用于在最终特征表示之前对特征进行丢弃操作。

self.sr_ratio = sr_ratio：将sr_ratio赋值给模块内部的变量self.sr_ratio。

self.agent_num = agent_num：将agent_num赋值给模块内部的变量self.agent_num。

self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)：定义一个深度可分离卷积层。

self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))：定义一个可训练的模型参数，表示注意力偏置。

self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))：定义一个可训练的模型参数，表示注意力偏置。

self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))：定义一个可训练的模型参数，表示注意力偏置。

self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))：定义一个可训练的模型参数，表示注意力偏置。

self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))：定义一个可训练的模型参数，表示注意力偏置。

self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))：定义一个可训练的模型参数，表示注意力偏置。

trunc_normal_(self.an_bias, std=.02)：对self.an_bias进行截断正态分布初始化。

trunc_normal_(self.na_bias, std=.02)：对self.na_bias进行截断正态分布初始化。

trunc_normal_(self.ah_bias, std=.02)：对self.ah_bias进行截断正态分布初始化。

trunc_normal_(self.aw_bias, std=.02)：对self.aw_bias进行截断正态分布初始化。

trunc_normal_(self.ha_bias, std=.02)：对self.ha_bias进行截断正态分布初始化。

trunc_normal_(self.wa_bias, std=.02)：对self.wa_bias进行截断正态分布初始化。

pool_size = int(agent_num ** 0.5)：计算池化层的输出大小。

self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))：定义一个自适应平均池化层。

self.softmax = nn.Softmax(dim=-1)：定义一个softmax层。
'''
class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))#它对输入进行池化操作，使得输出张量在空间维度上的高度和宽度均为 pool_size
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        '''
输入 x 维度：(b,n,c)
执行查询变换 q=Linear(x)，维度变为：(b,n,c)
执行键值对变换 kv=Linear(x)，维度变为：(b,n,2,c)
将 kv 的维度进行置换，变为：(2,b,n,c)
提取出 k 和 v，维度变为：(b,n,num_heads,head_dim)
对查询q进行形状变换，维度变为：(b,num_heads,n,head_dim)
对键和值进行形状变换，维度变为：(b, n,num_heads,head_dim)
将代理 tokens 进行形状变换，维度变为：(b,agent_num,num_heads,head_dim)
对位置偏置项进行插值，维度变为：(b,num_heads,agent_num, window_size[0]/sr_ratio,window_size[1]/sr_ratio)
计算代理的注意力权重，维度变为：(b,num_heads,agent_num,n)
对代理的注意力权重进行 dropout 操作
计算加权和后的代理值，维度变为：(b,num_heads,agent_num,head_dim)
对代理值进行插值，维度变为：(b,num_heads, H/sr_ratio,W/sr_ratio,c)
将代理值进行形状变换，维度变为：(b,n,c)
执行最终输出的线性变换，维度变为：(b,n,c)
对输出进行 dropout 操作。
最终，返回的张量维度为：(b,n,c)
        '''
        b, n, c = x.shape#(1,196,64)
        num_heads = self.num_heads#第一阶段num_heads为1
        head_dim = c // num_heads#第一阶段64
        q = self.q(x)#(1,196,64)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]#(2,1,196,64)

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2))# (b,p×p,c)(1,9,64)
        agent_tokens =agent_tokens.reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)#(1,1,196,64)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)#(1,1,agent_num=9,64)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')#(1,9,14,14)
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)#(1,1,9,196)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)##(1,1,9,196)
        position_bias = position_bias1 + position_bias2##(1,1,9,196)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)#(1,1,9,196)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v#(1,1,9,196)x(1,1,196,64)=#(1,1,9,64)

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')#(1,9,14,14)
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)#(1,1,196,9)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)#(1,1,196,9)
        agent_bias = agent_bias1 + agent_bias2#(1,1,196,9)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)#(1,1,196,9)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v#(1,1,196,9)x(1,1,9,64)=#(1,1,196,64)

        x = x.transpose(1, 2).reshape(b, n, c)#(1,196,64)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)#(1,64,14,14)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)##(1,196,64)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 agent_num=49, attn_type='A'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        assert attn_type in ['A', 'B']
        if attn_type == 'A':
            self.attn = AgentAttention(
                dim, num_patches,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                agent_num=agent_num)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], agent_sr_ratios='1111', num_stages=4,
                 agent_num=[9, 16, 49, 49], attn_type='AAAA', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        attn_type = 'AAAA' if attn_type is None else attn_type
        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i - 1) * patch_size),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_patches=num_patches, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i] if attn_type[i] == 'B' else int(agent_sr_ratios[i]),
                agent_num=int(agent_num[i]), attn_type=attn_type[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)

            pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


def agent_pvt_tiny(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


def agent_pvt_small(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


def agent_pvt_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


def agent_pvt_large(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model
if __name__ == '__main__':

    model = PyramidVisionTransformer()


    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3,224,224), as_strings=True, print_per_layer_stat=False,
                                              verbose=False)
    print(flops, params)

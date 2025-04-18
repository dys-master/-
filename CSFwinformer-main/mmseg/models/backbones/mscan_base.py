# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
# 新版from mmengine.model import BaseModule
from mmcv.runner import BaseModule #老版
# 新版from mmengine.model.weight_init import (constant_init, normal_init,
#                                         trunc_normal_init)
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmseg.models.builder import BACKBONES
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
class Mlp(BaseModule):
    """Multi Layer Perceptron (MLP) Module.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features.
            Defaults: None.
        out_features (int): The dimension of output features.
            Defaults: None.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=True,
            groups=hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward function."""

        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    """Stem Block at the beginning of Semantic Branch.

    Args:
        in_channels (int): The dimension of input channels.
        out_channels (int): The dimension of output channels.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAAttention(BaseModule):
    """Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        channels (int): The dimension of channels.
        kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
    """

    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """Forward function."""

        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        # Channel Mixing
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x


class MSCASpatialAttention(BaseModule):
    """Spatial Attention Module in Multi-Scale Convolutional Attention Module
    (MSCA).

    Args:
        in_channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        """Forward function."""

        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class MSCABlock(BaseModule):
    """Basic Multi-Scale Convolutional Attention Block. It leverage the large-
    kernel attention (LKA) mechanism to build both channel and spatial
    attention. In each branch, it uses two depth-wise strip convolutions to
    approximate standard depth-wise convolutions with large kernels. The kernel
    size for each branch is set to 7, 11, and 21, respectively.

    Args:
        channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        mlp_ratio (float): The ratio of multiple input dimension to
            calculate hidden feature in MLP layer. Defaults: 4.0.
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
        drop_path (float): The ratio of drop paths.
            Defaults: 0.0.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = MSCASpatialAttention(channels, attention_kernel_sizes,
                                         attention_kernel_paddings, act_cfg)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""

        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        patch_size (int): The patch size.
            Defaults: 7.
        stride (int): Stride of the convolutional layer.
            Default: 4.
        in_channels (int): The number of input channels.
            Defaults: 3.
        embed_dims (int): The dimensions of embedding.
            Defaults: 768.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_channels=3,
                 embed_dim=768,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W




@BACKBONES.register_module()
class MSCAN_Unet_Base(BaseModule):

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 320, 512],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 depths=[3, 3, 12, 3],
                 num_stages=4,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 n_class=2,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        
        self.n_class = n_class
        self.in_channel = 3
        self.depths = depths
        self.num_stages = num_stages
        config=[64, 128, 320, 512]
        self.Up5 = up_conv(ch_in=config[3], ch_out=config[2])
        self.Up_conv5 = conv_block(ch_in=640, ch_out=config[2])

        self.Up4 = up_conv(ch_in=config[2], ch_out=config[1])
        self.Up_conv4 = conv_block(ch_in=256, ch_out=config[1])

        self.Up3 = up_conv(ch_in=config[1], ch_out=config[0])
        self.Up_conv3 = conv_block(ch_in=128, ch_out=config[0])

        self.Up4x = LRDU(config[0],4)
        self.convout = nn.Conv2d(config[0], n_class, kernel_size=1, stride=1, padding=0)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg)

            block = nn.ModuleList([
                MSCABlock(
                    channels=embed_dims[i],
                    attention_kernel_sizes=attention_kernel_sizes,
                    attention_kernel_paddings=attention_kernel_paddings,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def init_weights(self):
        """Initialize modules of MSCAN."""

        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x):
        """Forward function."""

        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        x128 = outs[0]
        x64 =outs[1]
        x32= outs[2]
        x16 = outs[3]
        d32 = self.Up5(x16) #torch.Size([2, 160, 32, 32])
        d32 = torch.cat([x32,d32],dim=1) #torch.Size([2, 320, 32, 32])
        d32 = self.Up_conv5(d32) #torch.Size([2, 160, 32, 32])
        d64 = self.Up4(d32) #torch.Size([2, 64, 64, 64])
        d64 = torch.cat([x64,d64],dim=1) #torch.Size([2, 128, 64, 64])
        d64 = self.Up_conv4(d64)
        d128 = self.Up3(d64)
        d128 = torch.cat([x128,d128],dim=1)
        d128 = self.Up_conv3(d128)
        d512 = self.Up4x(d128)
        out = self.convout(d512)
        return out
    
class LRDU(nn.Module):
    """
    large receptive detailed upsample
    """
    def __init__(self,in_c,factor):
        super(LRDU,self).__init__()

        self.up_factor = factor
        self.factor1 = factor*factor//2
        self.factor2 = factor*factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1*in_c, (1,7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1*in_c, self.factor2*in_c, (7,1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor),
            nn.Conv2d(in_c, in_c, 3 ,groups= in_c//4,padding=1), 
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()

        self.up = nn.Sequential(
            LRDU(ch_in,2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x



# def MSCAN_tiny(pretrained=False, **kwargs):
#     model = MSCAN(embed_dims=[32, 64, 160, 256],
#         mlp_ratios=[8, 8, 4, 4],
#         drop_rate=0.0,
#         drop_path_rate=0.1,
#         depths=[3, 3, 5, 2],
#         attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
#         attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]], **kwargs)
#     if pretrained:
#         url = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        
#         # Get the state_dict of the model and the loaded state_dict
#         model_state_dict = model.state_dict()
#         loaded_state_dict = checkpoint['state_dict']
        
#         # Match state_dict keys
#         for key in model_state_dict.keys():
#             if key in loaded_state_dict.keys():
#                 model_state_dict[key] = loaded_state_dict[key]
#             else:
#                 print(f"**NOT LOADED** {key}")
        
#         # Load the modified state_dict into the model
#         model.load_state_dict(model_state_dict)
        
#         # Print loaded state_dict keys
#         print("Loaded state_dict keys:")
#         for key in model_state_dict.keys():
#             print(key)
         #tiny [32, 64, 160, 256]
#     return model
# def MSCAN_Base(pretrained=False, **kwargs):
#     model = MSCAN(embed_dims=[64, 128, 320, 512],
#         mlp_ratios=[8, 8, 4, 4],
#         drop_rate=0.0,
#         drop_path_rate=0.1,
#         depths=[3, 3, 12, 3],
#         attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
#         attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]], **kwargs)
#     if pretrained:
#         url ='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_b_20230227-3ab7d230.pth'
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
#         model.load_state_dict(checkpoint['state_dict'], strict=False)
#         print("Pretrained model loaded successfully")
#     return model
# @BACKBONES.register_module()
# class MSCAN_Unet_Base(nn.Module):
#     def __init__(self, n_class=2):
#         super(MSCAN_Unet_Base, self).__init__()
#         self.n_class = n_class
#         self.in_channel = 3
#         config=[64, 128, 320, 512]
#         self.backbone = MSCAN_Base(True)

#         self.Up5 = up_conv(ch_in=config[3], ch_out=config[2])
#         self.Up_conv5 = conv_block(ch_in=640, ch_out=config[2])

#         self.Up4 = up_conv(ch_in=config[2], ch_out=config[1])
#         self.Up_conv4 = conv_block(ch_in=256, ch_out=config[1])

#         self.Up3 = up_conv(ch_in=config[1], ch_out=config[0])
#         self.Up_conv3 = conv_block(ch_in=128, ch_out=config[0])

#         self.Up4x = LRDU(config[0],4)      
#         self.convout = nn.Conv2d(config[0], n_class, kernel_size=1, stride=1, padding=0)                
        
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)
    
#     def forward(self, x):
#         x128,x64,x32,x16 = self.backbone(x)
#         '''
#          x128 torch.Size([2, 32, 128, 128])
#          x64  torch.Size([2, 64, 64, 64])
#          x32  torch.Size([2, 160, 32, 32])
#          x16  torch.Size([2, 256, 16, 16])
#         '''
#         d32 = self.Up5(x16) #torch.Size([2, 160, 32, 32])
#         d32 = torch.cat([x32,d32],dim=1) #torch.Size([2, 320, 32, 32])
#         d32 = self.Up_conv5(d32) #torch.Size([2, 160, 32, 32])
#         d64 = self.Up4(d32) #torch.Size([2, 64, 64, 64])
#         d64 = torch.cat([x64,d64],dim=1) #torch.Size([2, 128, 64, 64])
#         d64 = self.Up_conv4(d64)
#         d128 = self.Up3(d64)
#         d128 = torch.cat([x128,d128],dim=1)
#         d128 = self.Up_conv3(d128)
#         d512 = self.Up4x(d128)
#         out = self.convout(d512)
#         return out
# def debug_pam():
#     pam_module = MSCAN_Unet_Base()
#     # B,C,H,W
#     x = torch.rand((2, 3, 512, 512))
#     output_dict = pam_module(x)

#     # 访问字典中的张量并打印其形状
#     print("Output logits shape:", output_dict.shape)
#     # print("Output logits shape:", output_dict[0].shape)#([2, 64, 128, 128])
#     # print("Output logits shape:", output_dict[1].shape)#([2, 128, 64, 64])
#     # print("Output logits shape:", output_dict[2].shape)#([2, 256, 32, 32])
#     # print("Output logits shape:", output_dict[3].shape)#([2, 512, 16, 16])
# if __name__ == '__main__':
#     debug_pam()
if __name__ == '__main__':
    model = MSCAN_Unet_Base()
    print(model.backbone.init_cfg)
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print(macs, params)
    #Base 39.54 GMac 31.69 M
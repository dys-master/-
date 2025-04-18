""" zjd of the U-Net model """

from tkinter import W
from tkinter.tix import MAIN
from turtle import mainloop
from unicodedata import name
from pip import main
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import complexPyTorch.complexLayers as CPL
import complexPyTorch.complexFunctions as CPF
from mmseg.models.builder import BACKBONES
from mmseg.models.utils import InvertedResidual, make_divisible

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BAM(nn.Module):
    def __init__(self):
        super(BAM, self).__init__()
        self.scale = 0.0001

    def forward(self, in_tensor):
        # PMF和MMF的地方--傅里叶attention
        n = in_tensor.shape[2] * in_tensor.shape[3] - 1
        t_real = in_tensor.real
        t_imag = in_tensor.imag
        d_r = (t_real-t_real.mean(dim=[2,3], keepdim=True)).pow(2)
        d_i = (t_imag-t_imag.mean(dim=[2,3], keepdim=True)).pow(2)
        v_r = d_r.sum(dim=[2, 3], keepdim=True) / n
        v_i = d_i.sum(dim=[2, 3], keepdim=True) / n
        att_r = d_r/ (4 * (v_r + self.scale)) + 0.5
        att_i = d_i/ (4 * (v_i + self.scale)) + 0.5
        return att_r * in_tensor.real + 1j*att_i*in_tensor.imag


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Linear(dim, dim * 4, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B,C,H,W
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        x = self.expand(x)
        x = x.view(B, H, W, -1)
        # print(x.shape)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, H * 2, W * 2, C // 4)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H,W
        """
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x).view(B, H // 2, W // 2, -1)  # [B, H/2*W/2, 2*C]
        x = x.permute(0, 3, 1, 2)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768, group_num=4):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim // group_num)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shortcut = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shortcut = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class F_Block(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def __init__(self, dim, H, W):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 1, 1)
        self.f_conv = CPL.ComplexConv2d(dim, dim, 3, 1, 1)
        self.f_bn = CPL.ComplexBatchNorm2d(dim)
        self.f_relu = CPF.complex_relu


    def forward(self, x):
        bias = x
        dtype = x.dtype
        # B,C,H,W=x.shape
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        # shortcut
        x2=self.conv(x)
        # 此处应该为[B,C,H,W//2+1]
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        # GFNet---GML全局卷积核
        # 此处weight为[C,H,W//2+1]
        # weight = torch.view_as_complex(self.complex_weight)
        #GF_x = x * weight
        GF_x = self.f_conv(x)
        complex_GF = self.f_relu(self.f_bn(GF_x))
        # shape为[B,C,H,W//2+1]
        # complex_GF = self.complex_bam(GF_x)
        x = torch.fft.irfft2(complex_GF, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.reshape(B, C, N).permute(0, 2, 1)
        x = x.type(dtype)
        x2 = x2.reshape(B, C, N).permute(0, 2, 1)
        output = F.gelu(x + x2)
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        # hidden_features = out_features // 4
        self.fc1 = Conv1X1(in_features, hidden_features)
        self.gn1 = nn.GroupNorm(hidden_features // 4, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gn2 = nn.GroupNorm(hidden_features // 4, hidden_features)
        self.act = act_layer()
        self.fc2 = Conv1X1(hidden_features, out_features)
        self.gn3 = nn.GroupNorm(out_features // 4, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.gn1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.gn2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.gn3(x)
        x = self.drop(x)
        x = x.reshape(B, -1, N).permute(0, 2, 1)
        return x


class F_BlockLayerScale(nn.Module):

    def __init__(self, dim, dim_out, mlp_ratio=0.25, drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = F_Block(dim, H=h, W=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim_out, act_layer=act_layer,
                       drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim_out)), requires_grad=True)
        self.dim = dim
        self.dim_out = dim_out

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.dim == self.dim_out:
            x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
        else:
            x = self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, resolution_H=448, resolution_W=448):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            F_BlockLayerScale(in_channels, mid_channels, h=resolution_H, w=resolution_W),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            F_BlockLayerScale(mid_channels, out_channels, h=resolution_H, w=resolution_W),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, H, W):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            PatchMerging(in_channels, out_channels),
            DoubleConv(out_channels, out_channels, resolution_H=H, resolution_W=W)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, H, W, bilinear=True, ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, resolution_H=H, resolution_W=W)
        else:
            # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.up = PatchExpand(in_channels)
            self.conv = DoubleConv(in_channels, out_channels, resolution_H=H, resolution_W=W)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.models.builder import BACKBONES
from mmseg.models.utils import InvertedResidual, make_divisible

# 下面这句话 单独跑该文件时去掉，否则保留
#@BACKBONES.register_module()
class zjd_F_Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(zjd_F_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # 修改处--为了能减轻负担
        self.H, self.W = 448, 448
        # self.inc = DoubleConv(n_channels, 64,resolution_H=self.H,resolution_W=self.W)
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.down1 = Down(64, 128, self.H // 2, self.W // 2)
        self.down2 = Down(128, 256, self.H // 4, self.W // 4)
        self.down3 = Down(256, 512, self.H // 8, self.W // 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, self.W // 16, self.H // 16)
        self.up1 = Up(1024, 512 // factor, self.H // 8, self.W // 8, bilinear)
        self.up2 = Up(512, 256 // factor, self.H // 4, self.W // 4, bilinear)
        self.up3 = Up(256, 128 // factor, self.H // 2, self.W // 2, bilinear)
        self.up4 = Up(128, 64, self.H, self.W, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
if __name__ == '__main__':
    model = zjd_F_Unet()
    # model=F_Block(256,32,32)
    # inp=torch.randn(1,3,128,128)
    # inp=torch.randn(1,256*448,64)
    # out = model(inp)
    # print(out.shape)
    # model=nn.Conv2d(32,32,3,1,1)
    # (256* 256,32)
    # model=nn.Conv2d(256,256,3,1,1)
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print(macs, params)

#589.68 GMac 93.89 M
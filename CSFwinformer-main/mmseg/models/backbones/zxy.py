""" Parts of the U-Net model """

from tkinter import W
from tkinter.tix import MAIN
from turtle import mainloop
from unicodedata import name
from pip import main
from torch import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
import complexPyTorch.complexLayers as CPL
import complexPyTorch.complexFunctions as CPF
from mmseg.models.builder import BACKBONES
from mmseg.models.utils import InvertedResidual, make_divisible
import numpy as np

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d



class Down(nn.Module):
    def __init__(self, in_channels, out_channels, H, W):
        super().__init__()
        self.down = PatchMerging(in_channels, out_channels)
        self.conv = WrappedAFEBlock(out_channels, out_channels, resolution_H=H, resolution_W=W)

    def forward(self, x, image):
        x = self.down(x)
        x = self.conv(x, image)
        return x

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

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):  # dim=out-dim=64
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

        # print(f'this shape x is {x.shape}\n') # B,H,W,C

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        # print(f'this shape x0 is {x0.shape}\n')

        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        # print(f'this shape x1 is {x1.shape}\n')

        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C] 0::2 从0开始每隔2个选一个，1::2 从1开始每隔2个选一个
        # print(f'this shape x2 is {x2.shape}\n')

        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        # print(f'this shape x3 is {x3.shape}\n')

        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C] 在x3的维度拼接
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


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class CrossAttention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(CrossAttention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out

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

class SCModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        return attn1, attn2


# Selective feature Fusion Module
class SFFModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv3 = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, high_feature, low_feature, x):
        out = torch.cat([high_feature, low_feature], dim=1)
        avg_attn = torch.mean(out, dim=1, keepdim=True)

        max_attn, _ = torch.max(out, dim=1, keepdim=True)

        agg = torch.cat([avg_attn, max_attn], dim=1)

        sig = self.conv_squeeze(agg)

        sig = sig.sigmoid()

        out = high_feature * sig[:, 0, :, :].unsqueeze(1) + low_feature * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv3(out)
        result = x * out

        return result
class AFEBlock(nn.Module):
    """
    AFEBlock integrates Adaptive Frequency and Spatial feature Interaction Module (AFSIM)
    with Spatial Conv Module (SCM) and Selective Feature Fusion Module (SFFModule) to enhance
    feature representation by combining high and low frequency features.
    """
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(AFEBlock, self).__init__()
        self.AFSIM = AFSIModule(dim, num_heads, bias, in_dim)
        self.SCM = SCModule(dim)
        self.fusion = SFFModule(dim)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()

    def forward(self, image, x):
        # print(f"image: {image.shape}")
        # print(f"x: {x.shape}")
        _, _, H, W = x.size()
        image = F.interpolate(image, (H, W), mode='bilinear')
        shortcut = x.clone()

        x = self.proj_1(x)
        x = self.activation(x)
        s_high, s_low = self.SCM(x)

        high_feature, low_feature = self.AFSIM(s_high, s_low, image, x)
        out = self.fusion(high_feature, low_feature, x)

        result = self.proj_2(out)

        result = shortcut + result
        return result
class AFSIModule(nn.Module):

    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(AFSIModule, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        rdim = self.get_reduction_dim(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, rdim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(rdim, 2, 1, bias=False),
        )
        # Define learnable parameters for gating
        self.alpha_h = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha_w = torch.nn.Parameter(torch.tensor(0.5))

        self.CA_low = CrossAttention(dim // 2, num_head=num_heads, bias=bias)
        self.CA_high = CrossAttention(dim // 2, num_head=num_heads, bias=bias)

        self.conv2_1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2_2 = nn.Conv2d(dim, dim // 2, 1)

    def forward(self, s_high, s_low, image, x):

        f_high, f_low = self.fft(image)

        f_high = self.conv2_1(f_high)
        f_low = self.conv2_2(f_low)

        high_feature = self.CA_low(f_high, s_high)
        low_feature = self.CA_high(f_low, s_low)

        return high_feature, low_feature

    def get_reduction_dim(self, dim):
        if dim < 8:  # 最小维度保护
            return max(2, dim)
        log_dim = math.log2(dim)
        reduction = max(2, int(dim // log_dim))
        return reduction

    def shift(self, x):
        """shift FFT feature map to center"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)

        threshold = self.rate_conv(threshold).sigmoid()

        # 这个阈值用于确定频谱中心的大小，即决定多大范围的频率被认为是低频。
        # 加入了两个可学习参数帮助确定h和w
        blended_threshold_h = self.alpha_h * threshold[:, 0, :, :] + (1 - self.alpha_h) * threshold[:, 1, :, :]
        blended_threshold_w = self.alpha_w * threshold[:, 0, :, :] + (1 - self.alpha_w) * threshold[:, 1, :, :]

        # Calculate the dimensions of the mask based on the blended thresholds
        for i in range(mask.shape[0]):
            h_ = (h // 2 * blended_threshold_h[i]).round().int()  # Convert to int after rounding
            w_ = (w // 2 * blended_threshold_w[i]).round().int()  # Convert to int after rounding

            # Apply the mask based on blended h and w
            mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1

        # 对于mask的每个元素，根据阈值在频谱的中心位置创建一个正方形窗口，窗口内的值设为1，表示这部分是低频区域。
        fft = torch.fft.fft2(x, norm='forward', dim=(-2, -1))
        fft = self.shift(fft)
        # 对x执行FFT变换，得到频谱，并通过shift方法将低频分量移动到中心。
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2, -1))
        high = torch.abs(high)

        fft_low = fft * mask
        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2, -1))
        low = torch.abs(low)

        return high, low

class WrappedAFEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, resolution_H=448, resolution_W=448):
        super().__init__()
        # 新增一个投影层：把 x 从 in_channels -> out_channels
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.block = AFEBlock(
            dim=out_channels,
            num_heads=4,
            bias=False,
            in_dim=3
        )

    def forward(self, x, image):
        x = self.proj(x)  # 加这一行解决通道不一致
        return self.block(image, x)


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        # ConvTranspose
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_copy, x, interpolate=True):
        out = self.up(x)
        if interpolate:
            # Iterative filling, in order to obtain better results
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True
                                )
        else:
            # Different filling volume
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2))
        # Splicing
        out = torch.cat([x_copy, out], dim=1)
        out_conv = self.up_conv(out)
        return out_conv


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = WrappedAFEBlock(in_channels, out_channels, resolution_H=H, resolution_W=W)
        else:
            # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.up = PatchExpand(in_channels)
            self.conv = WrappedAFEBlock(in_channels, out_channels, resolution_H=H, resolution_W=W)
    def forward(self, x2, x1, image):
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
        return self.conv(x,image)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out
class Bottleneck1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, block1, layers, num_classes, bilinear=False):
        self.H = self.W = 448  # 注意这里改分辨率
        self.inplanes = 128
        self.bilinear = bilinear
        super(ResNet, self).__init__()
        #  ResNet # for feature extraction

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

        )

        factor = 2 if bilinear else 1
        self.down1 = Down(64, 128, self.H // 2, self.W // 2)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)

        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block1, 512, layers[3], stride=2)

        self.down4 = Down(512, 1024 // factor, self.W // 16, self.H // 16)
        #  TwofoldGCN # for channel and spatial feature
        # self.gcn_out =TwofoldGCN(512, 512, 512)
        self.double_conv1 = WrappedAFEBlock(128, 128, resolution_H=self.H // 2, resolution_W=self.W // 2)

        self.up0 = decoder(1024, 512)
        self.up1 = Up(512, 256 // factor, self.H // 4, self.W // 4, bilinear)

        self.up2 = Up(256, 128 // factor, self.H // 2, self.W // 2, bilinear)
        self.up3 = Up(128, 64, self.H, self.W, bilinear)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initalize_weights()

    def _initalize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        image = x  # 原图作为 image 输入
        x0 = self.conv0(x)
        print(f"第一步之后: {x0.shape}")
        x1 = self.down1(x0, image)  # 下采样1，带 image
        print(f"donw1之后: {x1.shape}")
        x2 = self.double_conv1(x1, image)  # DoubleConv 替换为 AFEBlock，带 image
        print(f"double_conv1: {x2.shape}")
        x2 = self.layer1(x2)
        print(f"layer1: {x2.shape}")
        x3 = self.layer2(x2)
        x3 = self.layer3(x3)
        x3 = self.layer4(x3)

        x4 = self.down4(x3, image)  # 下采样2，带 image
        print(f"down4: {x4.shape}")
        print(f"down4: {x3.shape}")
        x = self.up0(x3, x4)  # 上采样0，带 image
        print(f"up0: {x.shape}")
        x = self.up1(x2, x,image)  # 上采样1，带 image
        print(f"up1: {x.shape}")
        x = self.up2(x1, x,image)  # 上采样2，带 image
        print(f"up2: {x.shape}")
        x = self.up3(x0, x,image)  # 上采样3，带 image
        print(f"up3: {x.shape}")
        final = self.final_conv(x)
        return final


from mmseg.models.builder import BACKBONES
from mmseg.models.utils import InvertedResidual, make_divisible
import torch.utils.model_zoo as model_zoo

#@BACKBONES.register_module()
# def zxy_af(num_classes=150):
#     model = ResNet(Bottleneck, Bottleneck1, [3, 4, 6, 3], num_classes)
#
#     # 加载预训练权重
#     pretrained_dict = model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data')
#     model_dict = model.state_dict()
#
#     # 过滤掉 shape 不匹配的权重
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
#
#     # 更新模型参数
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#
#     return model
def zxy_af(num_classes=150):
    model = ResNet(Bottleneck, Bottleneck1, [3, 4, 6, 3], num_classes)

    # 加载预训练权重
    pretrained_dict = model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data')
    model_dict = model.state_dict()

    matched_keys = []
    skipped_keys = []

    # 对比每一层，分类为匹配或跳过
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
                matched_keys.append(k)
            else:
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)

    # 更新并加载
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print(f"\n✅ Matched layers ({len(matched_keys)}):")
    for k in matched_keys:
        print(f"  - {k}")

    print(f"\n⚠️ Skipped layers due to shape mismatch ({len(skipped_keys)}):")
    for k in skipped_keys:
        print(f"  - {k}")

    return model



if __name__ == '__main__':
    model = zxy_af()
    # model=F_Block(256,32,32)
    # inp=torch.randn(1,3,128,128)
    # inp=torch.randn(1,256*448,64)
    # out = model(inp)
    # print(out.shape)
    # model=nn.Conv2d(32,32,3,1,1)
    # (256* 256,32)
    # model=nn.Conv2d(256,256,3,1,1)
    from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False,
    #                                          verbose=False)
    macs, params = get_model_complexity_info(model, (3, 448, 448), as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print(macs, params)
# def debug_pam():
#     pam_module = WrappedAFEBlock(128,64)
#     # B,C,H,W
#     x = torch.rand((1, 64, 256, 256))
#     image = torch.rand((1, 3, 512, 512))
#     out = pam_module(x,image)
#     print('==out.shape:', out.shape)
# if __name__ == '__main__':
#     debug_pam()
# 第一步之后: torch.Size([1, 64, 448, 448])
# donw1之后: torch.Size([1, 128, 224, 224])
# double_conv1: torch.Size([1, 128, 224, 224])
# layer1: torch.Size([1, 256, 112, 112])
# down4: torch.Size([1, 1024, 28, 28])
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


def complex_gelu(input):
    return F.gelu(input.real).type(torch.complex64) + 1j * F.gelu(input.imag).type(torch.complex64)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, H, W):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            PatchMerging(in_channels, out_channels),
            DoubleConv(out_channels, out_channels, resolution_H=H, resolution_W=W)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AFilter(nn.Module):
    """ 实现语义自适应 卷积定理 等价于空间域全局大核卷积"""

    def __init__(self, in_channel, h, w):
        super().__init__()
        self.in_channels = in_channel
        #  对频域的1x1卷积 再改一下 TODO直接精简成复数卷积

        # 使用库中的 复数卷积
        self.conv1x1a = CPL.ComplexConv2d(self.in_channels, self.in_channels, 1, 1)
        self.f_gelu = complex_gelu
        self.conv1x1b = CPL.ComplexConv2d(self.in_channels, self.in_channels, 1, 1)
        self.f_bn = CPL.ComplexBatchNorm2d(in_channel)  # 复数域的BN 归一化

    def forward(self, x):
        x_after = self.conv1x1b(self.f_gelu(self.conv1x1a(x)))  # 语义自适应 复数掩码
        x = x * x_after  # 逐元素乘

        x = self.f_bn(x)  # 归一化

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BAM(nn.Module):
    """
    中间平均池化、max池化 的结构，提取全局
    """

    def __init__(self, in_channels, W, H, freq_sel_method='top16'):
        super(BAM, self).__init__()
        self.in_channels = in_channels

        # local channel
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.lc = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        # self.lcln = nn.LayerNorm([self.in_channels, 1])
        # transformation weights
        self.tw = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0,
                            groups=self.in_channels)
        self.twln = nn.LayerNorm([self.in_channels, 1, 1])
        self.sigmoid = nn.Sigmoid()
        self.register_parameter('wdct', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))
        self.register_parameter('wmax', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))

    def forward(self, x):
        N, C, H, W = x.shape  # global
        # self and local
        # w fusion 权重融合的过程
        x_s = (self.wmax * self.maxpool(x).squeeze(-1)) + self.wdct * (self.gap(x).squeeze(-1))
        # x_l = self.lc(x_s.permute(0, 2, 1)).transpose(-1, -2)
        # attention weights
        x_s = x_s.unsqueeze(-1)
        # att_c = self.lcln(x_s + x_l).unsqueeze(-1)
        att_c = self.sigmoid(self.twln(self.tw(x_s)))
        return att_c


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


class ComplexConv2d(nn.Module):
    """ 这是 复数分离卷积 操作
    output 是特征融合后的复数"""

    def __init__(self, channels, H, W, dtype=torch.complex64):
        super(ComplexConv2d, self).__init__()
        self.dtype = dtype
        # gcc_dk 是指定方向上做卷积，实现分离卷积；并融合中间分支的全局信息
        # 卷积的实部虚部化分开
        self.conv_r1 = gcc_dk(channels, 'W', W, H // 2 + 1)  # s输出的size位[1, channels , 1, 113]
        self.conv_i1 = gcc_dk(channels, 'H', W, H // 2 + 1)
        self.conv_i2 = gcc_dk(channels, 'W', W, H // 2 + 1)
        self.conv_r2 = gcc_dk(channels, 'H', W, H // 2 + 1)  # s输出的size位[1, channels , 224, 1]
        # self.register_parameter('W', nn.Parameter(torch.Tensor(channels).float()))
        self.weights_h = nn.Parameter(torch.randn(channels, 1, H // 2 + 1))
        self.weights_w = nn.Parameter(torch.randn(channels, W, 1))
        self.H = H // 2 + 1
        self.W = W

        # self.complex_bam = BAM(channels)

    def forward(self, input):
        a = self.conv_r1(input.real)  # + self.conv_r2(input.real)
        B, C, _, _ = a.shape
        b = self.conv_r2(input.real).expand(B, C, self.W, self.H)
        a = a.expand(B, C, self.W, self.H)

        a = self.weights_w * a + self.weights_h * b
        b = self.weights_w * self.conv_r1(input.imag).expand(B, C, self.W, self.H) + self.weights_h * self.conv_r2(
            input.imag).expand(B, C, self.W, self.H)
        c = self.weights_h * self.conv_i1(input.real).expand(B, C, self.W, self.H) + self.weights_w * self.conv_i2(
            input.real).expand(B, C, self.W, self.H)
        d = self.weights_h * self.conv_i1(input.imag).expand(B, C, self.W, self.H) + self.weights_w * self.conv_i2(
            input.imag).expand(B, C, self.W, self.H)

        real = (a - c)
        imag = (b + d)
        return real.type(self.dtype) + 1j * imag.type(self.dtype)


class gcc_dk(nn.Module):
    """水平or垂直卷积后，按元素相乘中间分支提取的全局信息，并输出；每次只中一个方向"""

    def __init__(self, channel, direction, W, H):
        super(gcc_dk, self).__init__()
        self.direction = direction
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.att = BAM(channel, W, H) # 中间提特征的分支 c*1*1
        if direction == 'H':
            self.kernel_generate_conv = nn.Sequential(
                # 垂直的3*1卷积
                nn.Conv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0), bias=False, groups=channel),
                nn.BatchNorm2d(channel),
                nn.Hardswish(),
                nn.Conv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0), bias=False, groups=channel),
            )
        elif direction == 'W':
            self.kernel_generate_conv = nn.Sequential(
                # 水平的1*3卷积

                nn.Conv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1), bias=False, groups=channel),
                nn.BatchNorm2d(channel),
                nn.Hardswish(),
                nn.Conv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1), bias=False, groups=channel),
            )

    def forward(self, x):

        # x输入为[b,c,H,W//2+1]
        # glob_info为[1,c,1,1]
        # glob_info = self.att(x) #中间分支提出的 全局信息
        # 每次水平、垂直只中一个

        if self.direction == 'H':
            # H_info[1, c, h, 1]
            H_info = torch.mean(x, dim=3, keepdim=True)  # 垂直取mean，保留H高
            H_info = self.kernel_generate_conv(H_info)  # 垂直3*1卷积操作
            #  kernel_input[1, c, h, 1]
            res = H_info
            # kernel_input = H_info * glob_info.expand_as(H_info) #H_info [1,c,h,1]和global_info扩展成H_info尺寸一致后，按元素相乘，
            # 使用元素相乘融合中间分支的全局信息
        elif self.direction == 'W':
            W_info = torch.mean(x, dim=2, keepdim=True)  # 水平取mean，保留H高
            W_info = self.kernel_generate_conv(W_info)  # 水平3*1卷积操作
            res = W_info
            # kernel_input = W_info * glob_info.expand_as(W_info)#H_info [1,c,h,1]和global_info扩展成H_info尺寸一致后，按元素相乘，
            # 使用元素相乘融合中间分支的全局信息

        # kernel_weight = self.kernel_generate_conv(kernel_input)

        return res


def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class F_Block(nn.Module):
    """
    输入 空间域x=[b,c,h,w],对x进行完整复数分离卷积与正则化处理，并输出 空间域x=[b,c,h,w]
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """

    def __init__(self, dim, H, W):
        super().__init__()
        # self.f_conv = ComplexConv2d(dim, H, W) #复数分离卷积
        self.f_bn = CPL.ComplexBatchNorm2d(dim)  # 复数域的BN 归一化
        self.f_relu = complex_gelu  # 复数域的gelu
        self.conv = nn.Conv2d(dim, dim, 1)

        self.adaptive = AFilter(dim, h=H, w=W)
        # self.weights_s = nn.Parameter(torch.randn(dim, H ,W//2+1))
        # self.weights_t = nn.Parameter(torch.randn(dim, H ,W//2+1))

    def forward(self, x):
        bias = x
        dtype = x.dtype
        # B,C,H,W=x.shape
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        # shortcut
        x2 = self.conv(x)
        # 此处应该为[B,C,H,W//2+1]
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")  # 傅里叶变换
        # GFNet---GML全局卷积核
        # 此处weight为[C,H,W//2+1]
        # weight = torch.view_as_complex(self.complex_weight)
        # GF_x = x * weight
        # GF_x = self.f_conv(x)
        AF_x = self.adaptive(x)

        # GF_x =self.weights_s* GF_x + self.weights_t*AF_x # 加权融合
        complex_GF = self.f_relu(AF_x)  # 对两方向 融合后的复数特征，作复值bn、复数relu 正则操作

        # complex_GF = self.f_relu(self.f_bn(AF_x)) #对两方向 融合后的复数特征，作复值bn、复数relu 正则操作
        # shape为[B,C,H,W//2+1]
        # complex_GF = self.complex_bam(GF_x)
        # x = complex_GF
        x = torch.fft.irfft2(complex_GF, s=(H, W), dim=(2, 3), norm="ortho")  # 正则后的复值特征 逆傅里叶转回空间域
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
        self.gamma = nn.Parameter(init_values * torch.ones((dim_out)), requires_grad=True)  # 可学习参数
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

    def __init__(self, in_channels, out_channels, mid_channels=None, resolution_H=448, resolution_W=448):  # 448改512
        # def __init__(self, in_channels, out_channels, mid_channels=None, resolution_H=512, resolution_W=512):

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


class Sobel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Sobel, self).__init__()
        kernel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        kernel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        kernel_x = torch.FloatTensor(kernel_x).expand(out_channel, in_channel, 3, 3)
        kernel_x = kernel_x.type(torch.cuda.FloatTensor)
        kernel_y = torch.cuda.FloatTensor(kernel_y).expand(out_channel, in_channel, 3, 3)
        kernel_y = kernel_y.type(torch.cuda.FloatTensor)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).clone()
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).clone()
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, c, h, w = x.size()
        sobel_x = F.conv2d(x, self.weight_x, stride=1, padding=1)
        sobel_x = torch.abs(sobel_x)
        sobel_y = F.conv2d(x, self.weight_y, stride=1, padding=1)
        sobel_y = torch.abs(sobel_y)
        if c == 1:
            sobel_x = sobel_x.view(b, h, -1)
            sobel_y = sobel_y.view(b, h, -1).permute(0, 2, 1)
        else:
            sobel_x = sobel_x.view(b, c, -1)
            sobel_y = sobel_y.view(b, c, -1).permute(0, 2, 1)
        sobel_A = torch.bmm(sobel_x, sobel_y)
        sobel_A = self.softmax(sobel_A)
        return sobel_A


class GCNSpatial(nn.Module):
    def __init__(self, channels):
        super(GCNSpatial, self).__init__()
        self.sobel = Sobel(channels, channels)
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def normalize(self, A):
        b, c, im = A.size()
        out = np.array([])
        for i in range(b):
            A1 = A[i].to(device="cpu")
            I = torch.eye(c, im)
            A1 = A1 + I
            # degree matrix
            d = A1.sum(1)
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            new_A = D.mm(A1).mm(D).detach().numpy()
            out = np.append(out, new_A)
        out = out.reshape(b, c, im)
        normalize_A = torch.from_numpy(out)
        normalize_A = normalize_A.type(torch.cuda.FloatTensor)
        return normalize_A

    def forward(self, x):
        b, c, h, w = x.size()
        A = self.sobel(x)
        A = self.normalize(A)
        x = x.view(b, c, -1)
        x = F.relu(self.fc1(A.bmm(x)))
        x = F.relu(self.fc2(A.bmm(x)))
        x = self.fc3(A.bmm(x))
        out = x.view(b, c, h, w)
        return out


class GCNChannel(nn.Module):
    def __init__(self, channels):
        super(GCNChannel, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.sobel = Sobel(1, 1)
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    # def sobel_channel(self,x):
    #     b,c,h,w = x.size()
    #     sobel = Sobel(h*w,h*w)
    #     return sobel

    def pre(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = x.view(b, 1, h * w, c)
        return x

    def normalize(self, A):
        b, c, im = A.size()
        out = np.array([])
        for i in range(b):
            # A = A = I
            A1 = A[i].to(device="cpu")
            I = torch.eye(c, im)
            A1 = A1 + I
            # degree matrix
            d = A1.sum(1)
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            new_A = D.mm(A1).mm(D).detach().numpy()
            out = np.append(out, new_A)
        out = out.reshape(b, c, im)
        normalize_A = torch.from_numpy(out)
        normalize_A = normalize_A.type(torch.cuda.FloatTensor)
        return normalize_A

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.input(x)
        b, c, h1, w1 = x.size()
        x = self.pre(x)
        # A = self.sobel_channel(x)
        A = self.sobel(x)
        A = self.normalize(A)
        x = x.view(b, -1, c)
        x = F.relu(self.fc1(A.bmm(x).permute(0, 2, 1))).permute(0, 2, 1)
        x = F.relu(self.fc2(A.bmm(x).permute(0, 2, 1))).permute(0, 2, 1)
        x = self.fc3(A.bmm(x).permute(0, 2, 1))
        out = x.view(b, c, h1, w1)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return out


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


class TwofoldGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(TwofoldGCN, self).__init__()
        # Depthwise convolution # for spatial feature extraction
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            BatchNorm2d(out_channels)
        )
        # GCN Spatial #
        self.spatial_in = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.gcn_s = GCNSpatial(out_channels // 2)
        self.conv_s = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channels // 2)
        )

        # Pointwise convolution # for channel feature extraction
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            BatchNorm2d(out_channels)
        )
        # GCN Channel #
        self.channel_in = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1),
            BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.gcn_c = GCNChannel(out_channels // 2)
        self.conv_c = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_channels // 2)
        )
        # output
        self.combine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels)
        )
        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channels),
            nn.ReLU(out_channels),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, bias=True)
            )

    def forward(self, x):
        # GCN_Spatial
        x_spatial_in = self.depth_conv(x)
        x_spatial_in = self.spatial_in(x_spatial_in)
        x_spatial = self.gcn_s(x_spatial_in)
        x_spatial = x_spatial_in + x_spatial
        # GCN_Channel
        x_channel_in = self.channel_conv(x)
        x_channel_in = self.channel_in(x_channel_in)
        x_channel = self.gcn_c(x_channel_in)
        x_channel = x_channel_in + x_channel
        # out
        out = torch.cat((x_spatial, x_channel), 1) + x
        out = self.combine(out)
        out = self.output(out)
        return out


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

    def forward(self, x2, x1):
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
            # nn.Conv2d(64,64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.GELU(),
        )

        factor = 2 if bilinear else 1
        self.down1 = Down(64, 128, self.H // 2, self.W // 2)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.ddown = PatchMerging(64, 64) 都没用这个ddown
        # self.bn1 = BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=False)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        # self.layer2 = self._make_layer(block, 128, layers[1])
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 2, 4))
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block1, 512, layers[3], stride=2)

        self.down4 = Down(512, 1024 // factor, self.W // 16, self.H // 16)
        #  TwofoldGCN # for channel and spatial feature
        # self.gcn_out =TwofoldGCN(512, 512, 512)
        self.double_conv1 = DoubleConv(128, 128, resolution_H=self.H // 2, resolution_W=self.W // 2)
        # self.double_conv2 = DoubleConv(512, 512, resolution_H=self.H // 8, resolution_W=self.W // 8)
        self.up0 = decoder(1024, 512)
        self.up1 = Up(512, 256 // factor, self.H // 4, self.W // 4, bilinear)
        # self.up2 = decoder(256, 128)
        self.up2 = Up(256, 128 // factor, self.H // 2, self.W // 2, bilinear)
        self.up3 = Up(128, 64, self.H, self.W, bilinear)
        # self.up3 = Up(128, 64, self.H, self.W, bilinear)
        # Full connection
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
        x0 = self.conv0(x)
        print(f"第一步之后: {x0.shape}")
        x1 = self.down1(x0)
        print(f"第一步之后: {x1.shape}")
        x2 = self.double_conv1(x1)
        print(f"第一步之后: {x2.shape}")
        x2 = self.layer1(x2)
        print(f"第一步之后: {x2.shape}")
        x3 = self.layer2(x2)
        x3 = self.layer3(x3)
        x3 = self.layer4(x3)

        x4 = self.down4(x3)
        print(f"down4: {x4.shape}")
        print(f"down4: {x3.shape}")
        x = self.up0(x3, x4)
        print(f"up0: {x.shape}")
        x = self.up1(x2, x)
        print(f"up1: {x.shape}")
        x = self.up2(x1, x)
        print(f"up2: {x.shape}")
        x = self.up3(x0, x)
        print(f"up3: {x.shape}")
        final = self.final_conv(x)



        return final


from mmseg.models.builder import BACKBONES
from mmseg.models.utils import InvertedResidual, make_divisible


#@BACKBONES.register_module()
def zxy_af(num_classes=150):
    model = ResNet(Bottleneck, Bottleneck1, [3, 4, 6, 3], num_classes)
    return model

#
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
#     pam_module = DoubleConv(128,64)
#     # B,C,H,W
#     x = torch.rand((1, 128, 64, 64))
#     image=  torch.rand((1, 3, 512, 512))
#     out = pam_module(x)
#     print('==out.shape:', out.shape)
# if __name__ == '__main__':
#     debug_pam()
# up0: torch.Size([1, 512, 56, 56])
# up1: torch.Size([1, 256, 112, 112])
# up2: torch.Size([1, 128, 224, 224])
# up3: torch.Size([1, 64, 448, 448])
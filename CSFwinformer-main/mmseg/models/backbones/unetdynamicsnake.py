from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d, ModulatedDeformConv2dPack

import torch.fft
from collections import OrderedDict
import torch
import torch.nn as nn
import math
import torch.fft
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from mmseg.models.utils import InvertedResidual, make_divisible
from mmseg.models.backbones.conv_custom import AdaptiveDilatedConv
from mmseg.models.backbones.dynamic_snake_conv import DySnakeConv


class FADC(nn.Module):
    def __init__(self,in_channel,filters):
        super(FADC, self).__init__()
        c1, c2, c3 = filters
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            AdaptiveDilatedConv(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        output_x = self.basicblock(x)
        X = identity + output_x
        X = self.relu(X)
        return X
class Bottleneck(nn.Module): # Convblock
    def __init__(self, in_channel, filters, s):
        super(Bottleneck, self).__init__()
        c1, c2, c3 = filters
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=1, stride=s, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.short_cut = nn.Conv2d(in_channel, c3, kernel_size=1, stride=s, padding=0, bias=False)
        self.batch1 = nn.BatchNorm2d(c3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output_x = self.bottleneck(x)
        short_cut_x = self.batch1(self.short_cut(x))
        result = output_x + short_cut_x
        X = self.relu(result)
        return X


class BasicBlock(nn.Module):
    def __init__(self,in_channel,filters):
        super(BasicBlock, self).__init__()
        c1, c2, c3 = filters
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        output_x = self.basicblock(x)
        X = identity + output_x
        X = self.relu(X)
        return X
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class BasicBlock_dynamicsnake_conv(nn.Module):
    def __init__(self,in_channel,filters):
        super(BasicBlock_dynamicsnake_conv, self).__init__()
        c1, c2, c3 = filters
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            DySnakeConv(inc=c1, ouc=c2, k=3),
            Conv(c2 * 3, c2, k=3, s=1, g=c2, act=nn.ReLU()),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        output_x = self.basicblock(x)
        X = identity + output_x
        X = self.relu(X)
        return X

class DoubleConv_dynamicsnake_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_dynamicsnake_conv, self).__init__(
            DySnakeConv(inc=in_channels, ouc=out_channels, k=3),
            Conv(out_channels * 3, out_channels, k=3, s=1, g=out_channels, act=nn.ReLU()),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DySnakeConv(inc=mid_channels, ouc=out_channels, k=3),
            Conv(out_channels * 3, out_channels, k=3, s=1, g=out_channels, act=nn.ReLU()),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Down__dynamicsnake_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down__dynamicsnake_conv, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_dynamicsnake_conv(in_channels, out_channels)
        )

    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
#csa注意力,添加在x3和x2
class AttentionBlock(nn.Module):
    def __init__(self, channel_l, channel_g, init_channel=64):
        super(AttentionBlock, self).__init__()
        self.W_x1 = nn.Conv2d(channel_l, channel_l, kernel_size=1)
        self.W_x2 = nn.Conv2d(channel_l, channel_g, kernel_size=int(channel_g/channel_l),
                              stride=int(channel_g/channel_l), padding=(channel_g//channel_l//2)-1)
        self.W_g1 = nn.Conv2d(init_channel, channel_l, kernel_size=int(channel_l/init_channel),
                              stride=int(channel_l/init_channel), padding=(channel_l//init_channel//2)-1)
        self.W_g2 = nn.Conv2d(channel_g, channel_g, kernel_size=1)
        self.relu = nn.ReLU()
        self.psi1 = nn.Conv2d(channel_l, out_channels=1, kernel_size=1)
        self.psi2 = nn.Conv2d(channel_g, out_channels=1, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x_l, x_g, first_layer_f):
        # First Attention Operation
        first_layer_afterconv = self.W_g1(first_layer_f)
        xl_afterconv = self.W_x1(x_l)
        att_map_first = self.sig(self.psi1(self.relu(first_layer_afterconv + xl_afterconv)))
        xl_after_first_att = x_l * att_map_first

        # Second Attention Operation
        xg_afterconv = self.W_g2(x_g)
        xl_after_first_att_and_conv = self.W_x2(xl_after_first_att)
        att_map_second = self.sig(self.psi2(self.relu(xg_afterconv + xl_after_first_att_and_conv)))
        att_map_second_upsample = F.interpolate(att_map_second, size=x_l.size()[2:], mode='bilinear', align_corners=True)
        out = xl_after_first_att * att_map_second_upsample
        return out
    

@BACKBONES.register_module()
class UNet_dynamicsnake(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_dynamicsnake, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        # self.down1 = Down(base_c, base_c * 2)
        # self.down2 = Down(base_c * 2, base_c * 4)
        self.stage2 = nn.Sequential(#代替down1
                    Bottleneck(64, [64, 64, 128],s=2),
                    BasicBlock_dynamicsnake_conv(128, filters=[64, 64, 128]),
                    BasicBlock_dynamicsnake_conv(128, filters=[64, 64, 128]),
                )       
        self.stage3 = nn.Sequential(#代替down2
                    Bottleneck(128, [128, 128, 256],s=2),
                    FADC(256, filters=[128, 128, 256]),
                    FADC(256, filters=[128, 128, 256]),
                    FADC(256, filters=[128, 128, 256]),
                )        
        self.down3 = Down__dynamicsnake_conv(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 =Down__dynamicsnake_conv(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        # self.stage4 = nn.Sequential(#代替down3
        #             Bottleneck(256, [256, 256, 512],s=2),
        #             BasicBlock(512, filters=[256, 256, 512]),
        #             BasicBlock(512, filters=[256, 256, 512]),
        #             BasicBlock(512, filters=[256, 256, 512]),
        #             BasicBlock(512, filters=[256, 256, 512]),
        #             BasicBlock(512, filters=[256, 256, 512]),
        # )
        self.attentions1 = AttentionBlock(4 *  base_c,8 *  base_c,  base_c)
        self.attentions2 = AttentionBlock(2 *  base_c,4 *  base_c,  base_c)       

        #down1和down2用动态蛇形卷积的残差块替换,down3和down4用动态蛇形卷积的double卷积替换 180.29 GMac 29.24 M
        #在上面的基础上加入两个csa注意力 195.4 GMac 30.6 M
        #在上面的基础上加入一个csa注意力 188.91 GMac 30.36 M
        #把第三阶段的BasicBlock_dynamicsnake_conv换成FADC 168.1 GMac 28.68 M
        #在上面的基础上加入两个注意力 183.21 GMac 30.05 M
    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x3 =self.attentions1(x3,x4,x1)#torch.Size([2, 256, 128, 128])
        x2 =self.attentions2(x2,x3,x1)#torch.Size([2, 128, 256, 256])
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits
if __name__ == '__main__':
    model = UNet_dynamicsnake()
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print(macs, params)
#替换第一和第二部分下采样的卷积为瓶颈层,后面两部分下采样的卷积为动态蛇形卷积 参数量 :180.29 GMac 35.54 M
# if __name__ == "__main__":
#     # 创建随机输入张量
#     x = torch.rand((2, 64, 120, 120)).cuda()

#     # 初始化 BasicBlock_FADC 实例
#     # bottleneck = BasicBlock_dynamicsnake_conv(128, filters=[64, 64, 128]).cuda()
#     bottleneck = DoubleConv_dynamicsnake_conv(64, 128).cuda()

#     # 将输入张量传递给模型
#     model_output = bottleneck(x)

#     # 打印模型输出
#     print(model_output.shape)
    

# def debug_pam():
#     pam_module = UNet_dynamicsnake(3,2,64)
#     # B,C,H,W
#     x = torch.rand((2, 3, 512, 512))
#     output_dict = pam_module(x)

#     # 访问字典中的张量并打印其形状
    
#     print("Output logits shape:", output_dict.shape)

# if __name__ == '__main__':
#     debug_pam()

# if __name__ == '__main__':
#     model = BasicBlock_FADC(3,64,1,False)
#     from ptflops import get_model_complexity_info

#     macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False,
#                                              verbose=False)
#     print(macs, params)
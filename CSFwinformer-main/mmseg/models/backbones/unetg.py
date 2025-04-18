from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d, ModulatedDeformConv2dPack

import torch.fft

import torch
import torch.nn as nn

import torch.fft
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from mmseg.models.utils import InvertedResidual, make_divisible
from mmseg.models.backbones.conv_custom import AdaptiveDilatedConv

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

class DoubleConv_FADC(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_FADC, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            AdaptiveDilatedConv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
class Down_FADC(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down_FADC, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_FADC(in_channels, out_channels)
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
    

class Up_FADC(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_FADC, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_FADC(in_channels, out_channels, in_channels // 2)
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
        att_map_second_upsample = F.upsample(att_map_second, size=x_l.size()[2:], mode='bilinear')
        out = xl_after_first_att * att_map_second_upsample
        return out
    
class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(ASPP, self).__init__()
        out_channels = 512
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x
@BACKBONES.register_module()
class UNet_dysg(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet_dysg, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.aspp = ASPP(512, [12, 24, 36], norm_layer=nn.BatchNorm2d, norm_kwargs=None)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        # self.up1_fadc = Up_FADC(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        # self.up2_fadc = Up_FADC(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        # self.up3_fadc = Up_FADC(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        # self.up4_fadc = Up_FADC(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        # self.stage2= Down_FADC(base_c, base_c * 2)
        self.stage3 = nn.Sequential(#代替down2
            Bottleneck(128, [128, 128, 256],s=2),
            FADC(256, filters=[128, 128, 256]),
            FADC(256, filters=[128, 128, 256]),
            FADC(256, filters=[128, 128, 256]),
        )
        self.stage4 = nn.Sequential(#代替down3
            Bottleneck(256, [256, 256, 512],s=2),
            FADC(512, filters=[256, 256, 512]),
            FADC(512, filters=[256, 256, 512]),
            FADC(512, filters=[256, 256, 512]),
            FADC(512, filters=[256, 256, 512]),
            FADC(512, filters=[256, 256, 512]),
        )
        # self.stage5= Down_FADC(base_c * 8, base_c * 16 // factor)
        # self.attentions1 = AttentionBlock(4 *  base_c,8 *  base_c,  base_c)
        # self.attentions2 = AttentionBlock(2 *  base_c,4 *  base_c,  base_c)
        


        #原始unet 160.36 GMac 17.26 M
        #加入csa注意力175.47 GMac 18.63 M
        #引入两个瓶颈FADC模块 149.12 GMac 23.7 M
        #加入assp模块 157.99 GMac 32.62 M
       
    def forward(self, x):
        x1 = self.in_conv(x)#torch.Size([2, 64, 512, 512])
        x2 = self.down1(x1) #torch.Size([2, 128, 256, 256])
        x3 = self.stage3(x2)#torch.Size([2, 256, 128, 128])
        x4 = self.stage4(x3)#torch.Size([2, 512, 64, 64])
        x5 = self.down4(x4)#torch.Size([2, 512, 32, 32])
        x5 = self.aspp(x5)
        # x3 =self.attentions1(x3,x4,x1)#torch.Size([2, 256, 128, 128])
        # x2 =self.attentions2(x2,x3,x1)#torch.Size([2, 128, 256, 256])
        x = self.up1(x5, x4)#torch.Size([2, 256, 64, 64])
        x = self.up2(x, x3)#torch.Size([2, 128, 128, 128])
        x = self.up3(x, x2)#torch.Size([2, 64, 256, 256])
        x = self.up4(x, x1)#torch.Size([2, 64, 512, 512])
        logits = self.out_conv(x)

        return logits
if __name__ == '__main__':
    model = UNet_dysg()
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print(macs, params)

# def debug_pam():
#     pam_module = UNet_dysg(3,2,64)
#     # B,C,H,W
#     x = torch.rand((2, 3, 512, 512))
#     output_dict = pam_module(x)

#     # 访问字典中的张量并打印其形状
    
#     print("Output logits shape:", output_dict.shape)

# if __name__ == '__main__':
#     debug_pam()
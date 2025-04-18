import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchsummary import summary
from mmseg.models.builder import BACKBONES
from mmseg.models.utils import InvertedResidual, make_divisible
from mmseg.models.backbones.conv_custom import *
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, modulated_deform_conv2d, ModulatedDeformConv2dPack
from timm.models.layers import DropPath, trunc_normal_
import torch.fft

import torch
import torch.nn as nn
import math
import torch.fft
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from mmseg.models.utils import InvertedResidual, make_divisible
#https://github.com/ZhouyuPOP/FracSeg-Net/blob/master/CSAUnet.py
@BACKBONES.register_module()
class CsaUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, final_sigmoid_flag=False, init_channel_number=64):
        super(CsaUnet, self).__init__()

        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, max_pool_flag=False),
            Encoder(init_channel_number, 2 * init_channel_number),
            Encoder(2 * init_channel_number, 4 * init_channel_number),
            Encoder(4 * init_channel_number, 8 * init_channel_number)
        ])

        self.decoders = nn.ModuleList([
            Decoder((4+8) * init_channel_number, 4 * init_channel_number),
            Decoder((2+4) * init_channel_number, 2 * init_channel_number),
            Decoder((1+2) * init_channel_number, init_channel_number)
        ])

        self.attentions = nn.ModuleList([
            AttentionBlock(4 * init_channel_number, 8 * init_channel_number, init_channel_number),
            AttentionBlock(2 * init_channel_number, 4 * init_channel_number, init_channel_number),
            None
        ])
        # 1×1×1 convolution reduces the number of output channels to the number of class
        self.final_conv = nn.Conv2d(init_channel_number, out_channels, 1)

        if final_sigmoid_flag:
            self.final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
    def forward(self, x):#torch.Size([2, 3, 512, 512])
        encoders_features = []

        # 编码器部分
        # 第一层编码器
        x = self.encoders[0](x)
        encoders_features.insert(0, x)#x1=torch.Size([2, 64, 512, 512])
        
        # 第二层编码器
        x = self.encoders[1](x) 
        encoders_features.insert(0, x)#x2=torch.Size([2, 128, 256, 256])
        
        # 第三层编码器
        x = self.encoders[2](x)
        encoders_features.insert(0, x)#x3=torch.Size([2, 256, 128, 128])
        
        # 第四层编码器
        x = self.encoders[3](x)
        encoders_features.insert(0, x)#x4=torch.Size([2, 512, 64, 64])

        # 提取最后一层的特征和其他特征
        first_layer_feature = encoders_features[-1]#x1=torch.Size([2, 64, 512, 512])
        encoders_feature = encoders_features[1:]
        # encoders_feature[0].shape
        # torch.Size([2, 256, 128, 128])
        # x.shape
        # torch.Size([2, 512, 64, 64])
        # first_layer_feature.shape
        # torch.Size([2, 64, 512, 512])
        # 解码器和注意力机制部分
        # 第一层解码器
        if self.attentions[0]:
            features_after_att = self.attentions[0](encoders_feature[0], x, first_layer_feature)
        else:
            features_after_att = first_layer_feature
        x = self.decoders[0](features_after_att, x)
        
        # 第二层解码器
        if self.attentions[1]:
            features_after_att = self.attentions[1](encoders_feature[1], x, first_layer_feature)
        else:
            features_after_att = first_layer_feature
        x = self.decoders[1](features_after_att, x)
        
        # 第三层解码器
        # 注意：第三层没有注意力机制
        features_after_att = first_layer_feature
        x = self.decoders[2](features_after_att, x)

        # 最后一层卷积
        x = self.final_conv(x)

        # 最终激活函数
        if hasattr(CsaUnet, 'final_activation'):
            x = self.final_activation(x)

        return x

    # def forward(self, x):
    #     encoders_features = []

    #     # 编码器部分
    #     for encoder in self.encoders:
    #         x = encoder(x)
    #         encoders_features.insert(0, x)

    #     # 重要！！从列表中移除最后一个编码器的输出
    #     first_layer_feature = encoders_features[-1]
    #     encoders_feature = encoders_features[1:]

    #     # 解码器和注意力机制部分
    #     for decoder, attention, encoder_feature in zip(self.decoders, self.attentions, encoders_feature):
    #         if attention:
    #             # 如果存在注意力机制，使用它
    #             features_after_att = attention(encoder_feature, x, first_layer_feature)
    #         else:
    #             # 如果不存在注意力机制，在第一层没有注意力机制的情况下，直接使用第一层的特征
    #             features_after_att = first_layer_feature
    #         # 解码器处理
    #         x = decoder(features_after_att, x)

    #     # 最后一层卷积
    #     x = self.final_conv(x)
    #     # 最终激活函数
    #     if hasattr(CsaUnet, 'final_activation'):
    #         x = self.final_activation(x)
    #     return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3,
                 max_pool_flag=True, max_pool_kernel_size=(2, 2)):
        super(Encoder, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_kernel_size, stride=2) if max_pool_flag else None
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=conv_kernel_size)

    def forward(self, x):
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.double_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super(Decoder, self).__init__()
        self.upsample = nn.ConvTranspose2d(2*out_channels, 2*out_channels, kernel_size, scale_factor, padding=1, output_padding=1)
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, encoder_features, x):
        x = self.upsample(x)
        x = torch.cat((encoder_features, x), dim=1)
        x = self.double_conv(x)
        return x


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConv, self).__init__()

        if in_channels < out_channels:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels // 2
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_conv(1, conv1_in_channels, conv1_out_channels, kernel_size)
        # conv2
        self.add_conv(2, conv2_in_channels, conv2_out_channels, kernel_size)


    def add_conv(self, pos, in_channels, out_channels, kernel_size):
        assert pos in [1, 2], 'pos must be either 1 or 2'

        self.add_module(f'conv{pos}', nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
        self.add_module(f'relu{pos}', nn.ReLU(inplace=True))
        self.add_module(f'norm{pos}', nn.BatchNorm2d(out_channels))


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

def debug_pam():
    pam_module = CsaUnet(3, 2, final_sigmoid_flag=True, init_channel_number=64)
    # B,C,H,W
    x = torch.rand((2, 3, 512, 512))
    output_dict = pam_module(x)

    # 访问字典中的张量并打印其形状
    
    print("Output logits shape:", output_dict.shape)

if __name__ == '__main__':
    debug_pam()
# if __name__ == '__main__':
#     model = CsaUnet(1, 5, final_sigmoid_flag=True, init_channel_number=64).cuda()
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#     summary(model, ( 1, 56, 56), batch_size=1)
#     print(model)


#!/usr/bin/python
#
# Copyright 2022 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import timm
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from einops import rearrange
from mmseg.models.backbones.ffc import FFC_BN_ACT, ConcatTupleLayer
from mmseg.models.builder import BACKBONES


'''
in_channels：输入通道数，默认为3（通常用于 RGB 图像）。
out_channels：输出通道数，默认为2（用于二分类）。
init_features：初始特征数量，默认为32。
ratio_in：用于 FFC 的比率，默认为0.5。
ffc：是否使用 FFC（Fast Fourier Convolution），默认为 True。
skip_ffc：是否在跳跃连接中使用 FFC，默认为 False。
cat_merge：是否在合并过程中使用 cat 操作，默认为 True。
编码器部分由四个卷积块和最大池化层组成，每个块都对特征进行下采样。根据 ffc 参数，编码器可以选择使用常规卷积或 FFC
在编码器和解码器之间有一个瓶颈层，用于捕捉高层次的特征

输入 x 经过四层常规卷积编码器（enc1, enc2, enc3, enc4）。
输入 x 经过四层 FFC 编码器（enc1_f, enc2_f, enc3_f, enc4_f），如果 ffc 为 True。
瓶颈阶段：
合并常规编码器和 FFC 编码器的输出，并经过瓶颈层。
解码阶段：
解码器部分通过上采样和卷积块进行特征恢复和拼接。
根据 skip_ffc 和 ffc 参数，选择不同的合并方式。
'''
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
#@BACKBONES.register_module()
class YNet_general(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, init_features=64, ratio_in=0.5,backbone_name='swsl_resnet18',pretrained=None,bias=False):
        super(YNet_general, self).__init__()
        self.ratio_in = ratio_in
        self.backbone_name = backbone_name
        self.pretrained = pretrained

        features = init_features
        #################### 新加一步resnet的第一步
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)




        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        ################ FFC #######################################
        self.encoder1_f = FFC_BN_ACT(64, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
        self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
        self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
        self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4_f = FFC_BN_ACT(features * 4, features * 8, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
        self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.AFEB1 = AFEBlock(512, 4, bias=bias)

        self.bottleneck = YNet_general._block(features * 16, features * 16, name="bottleneck")  # 8, 16


        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2  # 16
        )
        self.decoder4 = YNet_general._block(features * 16, features * 8, name="dec4")  # 8, 12
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = YNet_general._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = YNet_general._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = YNet_general._block(features * 2, features, name="dec1")  # 2,3

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted
    def forward(self, x):
        batch = x.shape[0]
        h, w = x.size()[-2:]
        print(f"Input: {x.shape}")
        x1 = x
        enc1, enc2, enc3, enc4 = self.backbone(x)
        print(f"enc1: {enc1.shape}")
        print(f"enc2: {enc2.shape}")
        print(f"enc3: {enc3.shape}")
        print(f"enc4: {enc4.shape}")
        enc4 = self.AFEB1(x1, enc4)
        enc4_2 = self.pool4(enc4)
        print(f"enc4_2 (after pool): {enc4_2.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        enc1_f = self.encoder1_f(x)
        enc1_l, enc1_g = enc1_f


        enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))
        enc2_l, enc2_g = enc2_f


        enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))
        enc3_l, enc3_g = enc3_f


        enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))
        enc4_l, enc4_g = enc4_f


        enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))


        a = torch.zeros_like(enc4_2)
        b = torch.zeros_like(enc4_f2)

        enc4_2 = enc4_2.view(torch.numel(enc4_2), 1)
        enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)




        bottleneck = torch.cat((enc4_2, enc4_f2), 1)
        bottleneck = bottleneck.view_as(torch.cat((a, b), 1))
        bottleneck = self.bottleneck(bottleneck)


        dec4 = self.upconv4(bottleneck)

        dec4 = torch.cat((dec4, enc4), dim=1)

        dec4 = self.decoder4(dec4)


        dec3 = self.upconv3(dec4)

        dec3 = torch.cat((dec3, enc3), dim=1)

        dec3 = self.decoder3(dec3)


        dec2 = self.upconv2(dec3)

        dec2 = torch.cat((dec2, enc2), dim=1)

        dec2 = self.decoder2(dec2)


        dec1 = self.upconv1(dec2)

        dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)


        out = self.conv(dec1)

        out = self.softmax(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)


        print(f"enc4_2: {enc1.shape}")
        print(f"enc1_f: local={enc1_l.shape}, global={enc1_g.shape}")
        print(f"enc2_f: local={enc2_l.shape}, global={enc2_g.shape}")
        print(f"enc3_f: local={enc3_l.shape}, global={enc3_g.shape}")
        print(f"enc4_f: local={enc4_l.shape}, global={enc4_g.shape}")
        print(f"enc4_f2 (after catLayer): {enc4_f2.shape}")
        print(f"bottleneck input: {bottleneck.shape}")
        print(f"bottleneck output: {bottleneck.shape}")
        print(f"dec4 after upconv4: {dec4.shape}")
        print(f"dec4 after concat: {dec4.shape}")
        print(f"dec4 output: {dec4.shape}")
        print(f"dec3 after upconv3: {dec3.shape}")
        print(f"dec3 after concat: {dec3.shape}")
        print(f"dec3 output: {dec3.shape}")
        print(f"dec2 after upconv2: {dec2.shape}")
        print(f"dec2 after concat: {dec2.shape}")
        print(f"dec2 output: {dec2.shape}")
        print(f"dec1 after upconv1: {dec1.shape}")
        print(f"dec1 after concat: {dec1.shape}")
        print(f"dec1 output: {dec1.shape}")
        print(f"final conv: {out.shape}")
        print(f"softmax output: {out.shape}")

        return out
    # def forward(self, x):
    #     batch = x.shape[0]
    #     h, w = x.size()[-2:]
    #     print(f"Input: {x.shape}")
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #
    #     x = self.maxpool(x)
    #     print(f"第一步之后: {x.shape}")
    #     enc1 = self.encoder1(x)
    #     print(f"enc1: {enc1.shape}")
    #     enc2 = self.encoder2(self.pool1(enc1))
    #     print(f"enc2: {enc2.shape}")
    #     enc3 = self.encoder3(self.pool2(enc2))
    #     print(f"enc3: {enc3.shape}")
    #     enc4 = self.encoder4(self.pool3(enc3))
    #     print(f"enc4: {enc4.shape}")
    #     enc4_2 = self.pool4(enc4)
    #     print(f"enc4_2 (after pool): {enc4_2.shape}")
    #
    #     enc1_f = self.encoder1_f(x)
    #     enc1_l, enc1_g = enc1_f
    #
    #
    #     enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))
    #     enc2_l, enc2_g = enc2_f
    #
    #
    #     enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))
    #     enc3_l, enc3_g = enc3_f
    #
    #
    #     enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))
    #     enc4_l, enc4_g = enc4_f
    #
    #
    #     enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
    #
    #
    #     a = torch.zeros_like(enc4_2)
    #     b = torch.zeros_like(enc4_f2)
    #
    #     enc4_2 = enc4_2.view(torch.numel(enc4_2), 1)
    #     enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)
    #
    #     bottleneck = torch.cat((enc4_2, enc4_f2), 1)
    #     bottleneck = bottleneck.view_as(torch.cat((a, b), 1))
    #
    #
    #     bottleneck = self.bottleneck(bottleneck)
    #
    #
    #     dec4 = self.upconv4(bottleneck)
    #
    #     dec4 = torch.cat((dec4, enc4), dim=1)
    #
    #     dec4 = self.decoder4(dec4)
    #
    #
    #     dec3 = self.upconv3(dec4)
    #
    #     dec3 = torch.cat((dec3, enc3), dim=1)
    #
    #     dec3 = self.decoder3(dec3)
    #
    #
    #     dec2 = self.upconv2(dec3)
    #
    #     dec2 = torch.cat((dec2, enc2), dim=1)
    #
    #     dec2 = self.decoder2(dec2)
    #
    #
    #     dec1 = self.upconv1(dec2)
    #
    #     dec1 = torch.cat((dec1, enc1), dim=1)
    #
    #     dec1 = self.decoder1(dec1)
    #
    #
    #     out = self.conv(dec1)
    #
    #     out = self.softmax(out)
    #     out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
    #     print(f"enc1_f: local={enc1_l.shape}, global={enc1_g.shape}")
    #     print(f"enc2_f: local={enc2_l.shape}, global={enc2_g.shape}")
    #     print(f"enc3_f: local={enc3_l.shape}, global={enc3_g.shape}")
    #     print(f"enc4_f: local={enc4_l.shape}, global={enc4_g.shape}")
    #     print(f"enc4_f2 (after catLayer): {enc4_f2.shape}")
    #     print(f"bottleneck input: {bottleneck.shape}")
    #     print(f"bottleneck output: {bottleneck.shape}")
    #     print(f"dec4 after upconv4: {dec4.shape}")
    #     print(f"dec4 after concat: {dec4.shape}")
    #     print(f"dec4 output: {dec4.shape}")
    #     print(f"dec3 after upconv3: {dec3.shape}")
    #     print(f"dec3 after concat: {dec3.shape}")
    #     print(f"dec3 output: {dec3.shape}")
    #     print(f"dec2 after upconv2: {dec2.shape}")
    #     print(f"dec2 after concat: {dec2.shape}")
    #     print(f"dec2 output: {dec2.shape}")
    #     print(f"dec1 after upconv1: {dec1.shape}")
    #     print(f"dec1 after concat: {dec1.shape}")
    #     print(f"dec1 output: {dec1.shape}")
    #     print(f"final conv: {out.shape}")
    #     print(f"softmax output: {out.shape}")
    #
    #     return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
if __name__ == '__main__':
    model = YNet_general()
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print(macs, params)
# def debug_pam():
#     pam_module = YNet_general( )
#     # B,C,H,W
#     x = torch.rand((1, 3, 512, 512))
#
#     out = pam_module(x)
#     print('==out.shape:', out.shape)
# if __name__ == '__main__':
#     debug_pam()
# enc1: torch.Size([1, 64, 512, 512])
# enc2: torch.Size([1, 128, 256, 256])
# enc3: torch.Size([1, 256, 128, 128])
# enc4: torch.Size([1, 512, 64, 64])
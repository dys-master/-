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
from torchsummary import summary
from my_add.conv_custom import AdaptiveDilatedConv
BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d
class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        group_num = 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, groups=group_num)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = AdaptiveDilatedConv(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False, groups=group_num)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False, groups=group_num)
        self.bn3 = BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            BatchNorm2d(planes)
        )
        self.dilation = dilation
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        residual = x
        if (self.inplanes != self.planes and x.shape[1] == self.inplanes) or self.stride != 1:
            residual = self.downsample(residual)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu_inplace(out)

        return out
# if __name__ == '__main__':
#     model =Bottleneck(3,64)
#     from ptflops import get_model_complexity_info
#
#     macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, print_per_layer_stat=False,
#                                              verbose=False)
#     print(macs, params)


#一层FADC3.75 GMac 56.98 k
#原始11.07 GMac 41.86 k
def debug_pam():
    pam_module = Bottleneck(3,64)
    # B,C,H,W
    x = torch.rand((2, 3, 280, 280))
    out = pam_module(x)
    print('==out.shape:', out.shape)
if __name__ == '__main__':
    debug_pam()
# if __name__ == "__main__":
#
#     bottleneck = Bottleneck(3,64).cuda()
#
#     summary(bottleneck,(3,224,224))
